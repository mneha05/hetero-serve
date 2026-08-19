"""GPT-2 built directly as an OpenVINO graph, so it can run on Arc GPU or NPU.

Why hand-build the graph instead of importing ONNX? Because the KV cache has to
be an *input and an output* of the graph rather than something the runtime hides
internally. The scheduler owns the cache — it pages it, shares it between
sequences, and ships it across the network — so the engine must hand the new K/V
back every step and accept whatever past the block allocator gathered.

Device-specific reality, measured rather than assumed (see the probe in the
README):

  CPU  dynamic shapes fine
  GPU  dynamic shapes fine
  NPU  rejects dynamic shapes *and* rejects >4D tensors, so past KV is passed as
       one 4D tensor per layer, and the NPU path compiles a small set of static
       shape buckets on demand. Padding to a bucket wastes some compute; that is
       the honest cost of running a decoder on a fixed-shape accelerator.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:
    import openvino as ov
    import openvino.opset15 as op
except ImportError as exc:  # pragma: no cover
    raise ImportError("openvino is required for the GPU/NPU engines") from exc

from ..config import ModelConfig
from .weights import GPT2Weights

NEG_INF = np.float32(-3.0e38)
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / ".ov_cache"


def _const(a: np.ndarray):
    return op.constant(np.ascontiguousarray(a))


def _i32(vals) -> "op.Constant":
    return op.constant(np.asarray(vals, dtype=np.int32))


def _layer_norm(x, gamma: np.ndarray, beta: np.ndarray, eps: float):
    n = op.mvn(x, _i32([-1]), True, float(eps), "INSIDE_SQRT")
    return op.add(op.multiply(n, _const(gamma)), _const(beta))


def _linear(x, w: np.ndarray, b: np.ndarray):
    """x @ W + b, matching HF GPT-2's Conv1D convention (W is [in, out])."""
    return op.add(op.matmul(x, _const(w), False, False), _const(b))


class OpenVINOEngine:
    name = "openvino"

    def __init__(
        self,
        weights: GPT2Weights,
        cfg: ModelConfig | None = None,
        device: str = "GPU",
        max_batch: int = 8,
        bucket: int = 128,
        max_ctx: int | None = None,
        cache_dir: Path | None = None,
    ):
        self.w = weights
        self.cfg = cfg or weights.cfg
        self.device = device.upper()
        self.max_batch = max_batch
        self.bucket = bucket
        self.max_ctx = max_ctx or self.cfg.n_ctx
        self.dtype = np.float32

        self.core = ov.Core()
        if self.device not in self.core.available_devices:
            raise RuntimeError(f"device {self.device} not available: {self.core.available_devices}")

        cache = Path(cache_dir or os.environ.get("HETEROSERVE_OV_CACHE", DEFAULT_CACHE))
        cache.mkdir(parents=True, exist_ok=True)
        self.core.set_property({"CACHE_DIR": str(cache)})

        # NPU cannot take dynamic shapes; everything else can.
        self.static_only = self.device == "NPU"
        self._compiled: dict[tuple, object] = {}
        self._requests: dict[tuple, object] = {}

        if not self.static_only:
            self._get_request(None)      # compile once, up front

    def warmup(self) -> list[tuple]:
        """Compile every shape bucket this engine could hit, before serving.

        Only matters for NPU. Compiling a bucket costs ~15-20 s, and discovering
        a new one *mid-run* would inject that stall straight into a latency
        measurement, so we pay it all up front. Compiled blobs are cached on
        disk by CACHE_DIR, so this is a one-time cost per machine.
        """
        if not self.static_only:
            return []
        keys = []
        for p in range(self.bucket, self.max_ctx + self.bucket, self.bucket):
            keys.append(("prefill", self.bucket, p))
            keys.append(("decode", 1, p))
        built = []
        for phase, s, p in keys:
            key = self._key_for(phase, s, p)
            if key not in self._requests:
                self._get_request(key)
                built.append(key)
        return built

    # -- graph --------------------------------------------------------------

    def _build(self, B: int | None, S: int | None, P: int | None) -> "ov.Model":
        cfg = self.cfg
        L, H, D, E = cfg.n_layer, cfg.n_head, cfg.head_dim, cfg.n_embd
        b = -1 if B is None else B
        s = -1 if S is None else S
        p = -1 if P is None else P
        scale = np.float32(1.0 / np.sqrt(D))

        input_ids = op.parameter([b, s], ov.Type.i32, name="input_ids")
        position_ids = op.parameter([b, s], ov.Type.i32, name="position_ids")
        # additive mask over [B, 1, S, P+S]
        attn_mask = op.parameter(
            [b, 1, s, -1 if (P is None or S is None) else p + s], ov.Type.f32, name="attn_mask"
        )

        past_k, past_v = [], []
        for i in range(L):
            past_k.append(op.parameter([b, H, p, D], ov.Type.f32, name=f"past_k_{i}"))
            past_v.append(op.parameter([b, H, p, D], ov.Type.f32, name=f"past_v_{i}"))

        x = op.add(
            op.gather(_const(self.w.wte), input_ids, _i32(0)),
            op.gather(_const(self.w.wpe), position_ids, _i32(0)),
        )  # [B, S, E]

        results = []
        presents = []

        for i, lw in enumerate(self.w.layers):
            h = _layer_norm(x, lw.ln1_g, lw.ln1_b, cfg.layer_norm_eps)
            qkv = _linear(h, lw.attn_w, lw.attn_b)                      # [B, S, 3E]

            def head_split(t):
                # [B, S, E] -> [B, H, S, D]; the zeros mean "keep that dim".
                r = op.reshape(t, _i32([0, 0, H, D]), True)
                return op.transpose(r, _i32([0, 2, 1, 3]))

            q = head_split(op.slice(qkv, _i32([0]), _i32([E]), _i32([1]), _i32([2])))
            k = head_split(op.slice(qkv, _i32([E]), _i32([2 * E]), _i32([1]), _i32([2])))
            v = head_split(op.slice(qkv, _i32([2 * E]), _i32([3 * E]), _i32([1]), _i32([2])))

            presents.append(k)
            presents.append(v)

            k_full = op.concat([past_k[i], k], axis=2)                   # [B, H, P+S, D]
            v_full = op.concat([past_v[i], v], axis=2)

            scores = op.multiply(
                op.matmul(q, k_full, False, True), op.constant(scale)
            )                                                            # [B, H, S, P+S]
            scores = op.add(scores, attn_mask)
            ctx = op.matmul(op.softmax(scores, 3), v_full, False, False) # [B, H, S, D]

            merged = op.reshape(
                op.transpose(ctx, _i32([0, 2, 1, 3])), _i32([0, 0, E]), True
            )                                                            # [B, S, E]
            x = op.add(x, _linear(merged, lw.proj_w, lw.proj_b))

            h2 = _layer_norm(x, lw.ln2_g, lw.ln2_b, cfg.layer_norm_eps)
            ff = _linear(op.gelu(_linear(h2, lw.fc_w, lw.fc_b), "tanh"), lw.fcp_w, lw.fcp_b)
            x = op.add(x, ff)

        x = _layer_norm(x, self.w.lnf_g, self.w.lnf_b, cfg.layer_norm_eps)
        # Only the final position matters for sampling — emitting [B, S, V]
        # would move ~50 MB per prefill for nothing.
        last = op.squeeze(
            op.slice(x, _i32([-1]), _i32([np.iinfo(np.int32).max]), _i32([1]), _i32([1])),
            _i32([1]),
        )                                                                # [B, E]
        logits = op.matmul(last, _const(self.w.lm_head), False, True)    # [B, V]

        results.append(op.result(logits))
        results.extend(op.result(t) for t in presents)

        params = [input_ids, position_ids, attn_mask]
        for i in range(L):
            params.append(past_k[i])
            params.append(past_v[i])

        return ov.Model(results, params, f"gpt2_{self.device}")

    # -- compiled-model cache ----------------------------------------------

    def _get_request(self, key: tuple | None):
        if key in self._requests:
            return self._requests[key]

        if key is None:
            model = self._build(None, None, None)
            hint = {"PERFORMANCE_HINT": "LATENCY"}
        else:
            B, S, P = key
            model = self._build(B, S, P)
            hint = {}

        compiled = self.core.compile_model(model, self.device, hint)
        req = compiled.create_infer_request()
        self._compiled[key] = compiled
        self._requests[key] = req
        return req

    def _key_for(self, phase: str, S: int, P: int) -> tuple | None:
        """Dynamic devices need no bucketing; NPU rounds up to a static bucket.

        Batch is bucketed too, and that matters: keying on the *actual* batch
        size would compile a fresh 15-second graph the first time the scheduler
        happened to run 3 sequences instead of 4. Decode always pads to
        `max_batch`; prefill is always a single sequence.
        """
        if not self.static_only:
            return None
        pad_p = int(np.ceil(max(P, 1) / self.bucket) * self.bucket)
        if phase == "decode":
            return (self.max_batch, 1, pad_p)
        pad_s = int(np.ceil(max(S, 1) / self.bucket) * self.bucket)
        return (1, pad_s, pad_p)

    # -- inference helpers --------------------------------------------------

    def _run(self, inputs: dict, key: tuple | None) -> list[np.ndarray]:
        req = self._get_request(key)
        req.infer(inputs)
        n_out = 1 + 2 * self.cfg.n_layer
        return [req.get_output_tensor(i).data for i in range(n_out)]

    def _empty_past(self, B: int, P: int) -> np.ndarray:
        return np.zeros((B, self.cfg.n_head, P, self.cfg.head_dim), dtype=self.dtype)

    # -- public API ---------------------------------------------------------

    def prefill(
        self,
        token_ids: np.ndarray,
        start_pos: int,
        past_k: np.ndarray | None = None,
        past_v: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.cfg
        toks = np.asarray(token_ids, dtype=np.int32).reshape(1, -1)
        T = toks.shape[1]
        P = 0 if past_k is None else int(past_k.shape[2])

        key = self._key_for("prefill", T, P)
        pad_s, pad_p = (key[1], key[2]) if key else (T, P)
        lead = pad_s - T          # padding slots before the real tokens

        # The real tokens go at the *end* of the padded window so the graph's
        # "last position" is genuinely the last prompt token. Right-padding
        # would put a filler slot there and force a second inference pass.
        ids = np.zeros((1, pad_s), dtype=np.int32)
        ids[0, lead:] = toks[0]
        pos = np.zeros((1, pad_s), dtype=np.int32)
        pos[0, lead:] = np.arange(start_pos, start_pos + T, dtype=np.int32)

        # Past occupies slots [0, P) of a pad_p-wide window; new tokens occupy
        # slots [pad_p + lead, pad_p + pad_s).
        mask = np.full((pad_s, pad_p + pad_s), NEG_INF, dtype=np.float32)
        q_abs = start_pos + np.arange(pad_s) - lead
        for r in range(lead, pad_s):
            if P:
                mask[r, :P] = 0.0
            new_abs = start_pos + np.arange(T)
            mask[r, pad_p + lead :] = np.where(new_abs > q_abs[r], NEG_INF, np.float32(0.0))
        # Filler rows attend to a single slot purely to keep softmax finite;
        # their outputs are sliced away below.
        for r in range(lead):
            mask[r, 0] = 0.0
        mask = mask[None, None]

        inputs = {"input_ids": ids, "position_ids": pos, "attn_mask": mask}
        for i in range(cfg.n_layer):
            pk = self._empty_past(1, pad_p)
            pv = self._empty_past(1, pad_p)
            if P:
                pk[0, :, :P] = past_k[i]
                pv[0, :, :P] = past_v[i]
            inputs[f"past_k_{i}"] = pk
            inputs[f"past_v_{i}"] = pv

        outs = self._run(inputs, key)
        logits = np.array(outs[0][0], dtype=np.float32)

        L, H, D = cfg.n_layer, cfg.n_head, cfg.head_dim
        k_new = np.empty((L, H, T, D), dtype=self.dtype)
        v_new = np.empty((L, H, T, D), dtype=self.dtype)
        for i in range(L):
            k_new[i] = outs[1 + 2 * i][0, :, lead:]
            v_new[i] = outs[2 + 2 * i][0, :, lead:]

        return logits, k_new, v_new

    def decode_batch(
        self,
        token_ids: np.ndarray,
        positions: np.ndarray,
        past_ks: list[np.ndarray],
        past_vs: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.cfg
        L, H, D = cfg.n_layer, cfg.n_head, cfg.head_dim
        B = len(past_ks)
        lens = [int(p.shape[2]) for p in past_ks]
        P_max = max(lens) if lens else 0

        key = self._key_for("decode", 1, P_max)
        pad_b, pad_p = (key[0], key[2]) if key else (B, P_max)

        ids = np.zeros((pad_b, 1), dtype=np.int32)
        ids[:B, 0] = np.asarray(token_ids, dtype=np.int32)
        pos = np.zeros((pad_b, 1), dtype=np.int32)
        pos[:B, 0] = np.asarray(positions, dtype=np.int32)

        # Every sequence has a different real context length, so mask the slack.
        mask = np.full((pad_b, 1, 1, pad_p + 1), NEG_INF, dtype=np.float32)
        for b in range(B):
            mask[b, 0, 0, : lens[b]] = 0.0
            mask[b, 0, 0, pad_p] = 0.0            # the token being decoded
        for b in range(B, pad_b):
            mask[b, 0, 0, pad_p] = 0.0            # keep padded rows finite

        inputs = {"input_ids": ids, "position_ids": pos, "attn_mask": mask}
        for i in range(L):
            pk = self._empty_past(pad_b, pad_p)
            pv = self._empty_past(pad_b, pad_p)
            for b in range(B):
                n = lens[b]
                if n:
                    pk[b, :, :n] = past_ks[b][i]
                    pv[b, :, :n] = past_vs[b][i]
            inputs[f"past_k_{i}"] = pk
            inputs[f"past_v_{i}"] = pv

        outs = self._run(inputs, key)
        logits = np.array(outs[0][:B], dtype=np.float32)

        k_new = np.empty((B, L, H, 1, D), dtype=self.dtype)
        v_new = np.empty((B, L, H, 1, D), dtype=self.dtype)
        for i in range(L):
            k_new[:, i] = outs[1 + 2 * i][:B]
            v_new[:, i] = outs[2 + 2 * i][:B]

        return logits, k_new, v_new
