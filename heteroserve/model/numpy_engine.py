"""Real GPT-2 forward pass in numpy, built around a paged KV cache.

Two entry points mirror the two phases a serving scheduler cares about:

  prefill(...)      compute-bound, one sequence, chunkable
  decode_batch(...) memory-bound, many sequences, one token each

The split inside `decode_batch` is deliberate and mirrors a real paged-attention
runtime: the *linear* layers are batched into single big GEMMs (that's where the
FLOPs are), while *attention* runs per sequence over that sequence's gathered
blocks (that's where paging lives, and where every sequence has a different
context length). Padding all sequences to a common length would waste both
memory and arithmetic, which is exactly what paging exists to avoid.
"""

from __future__ import annotations

import numpy as np

from ..config import ModelConfig
from .weights import GPT2Weights, LayerWeights

SQRT_2_OVER_PI = np.float32(0.7978845608028654)


def layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    return xc / np.sqrt(var + eps) * g + b


def gelu_new(x: np.ndarray) -> np.ndarray:
    """The tanh approximation GPT-2 actually ships with."""
    inner = SQRT_2_OVER_PI * (x + np.float32(0.044715) * x * x * x)
    return np.float32(0.5) * x * (np.float32(1.0) + np.tanh(inner))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


class NumpyEngine:
    """CPU reference engine. Correctness oracle for the OpenVINO backends."""

    name = "numpy"

    def __init__(self, weights: GPT2Weights, cfg: ModelConfig | None = None, device: str = "CPU"):
        self.w = weights
        self.cfg = cfg or weights.cfg
        self.device = device
        self.eps = np.float32(self.cfg.layer_norm_eps)
        self.scale = np.float32(1.0 / np.sqrt(self.cfg.head_dim))
        self.dtype = np.float32

    # -- helpers ------------------------------------------------------------

    def _split_heads(self, x: np.ndarray, heads: int | None = None) -> np.ndarray:
        """[T, heads*D] -> [heads, T, D]. `heads` differs for K/V under GQA."""
        T = x.shape[0]
        h = heads or self.cfg.n_head
        return x.reshape(T, h, self.cfg.head_dim).transpose(1, 0, 2)

    def _expand_kv(self, x: np.ndarray) -> np.ndarray:
        """[kv_heads, T, D] -> [n_head, T, D] by sharing each KV head.

        Materialising the expansion keeps the reference implementation obvious.
        The CUDA kernels never do this -- they index the shared KV head directly,
        which is exactly where GQA's bandwidth saving comes from.
        """
        g = self.cfg.kv_group
        return x if g == 1 else np.repeat(x, g, axis=0)

    def _mlp(self, x: np.ndarray, l: LayerWeights) -> np.ndarray:
        h = gelu_new(x @ l.fc_w + l.fc_b)
        return h @ l.fcp_w + l.fcp_b

    def _embed(self, token_ids: np.ndarray, positions: np.ndarray) -> np.ndarray:
        return (self.w.wte[token_ids] + self.w.wpe[positions]).astype(self.dtype, copy=False)

    def logits_from_hidden(self, x: np.ndarray) -> np.ndarray:
        x = layer_norm(x, self.w.lnf_g, self.w.lnf_b, self.eps)
        return x @ self.w.lm_head.T

    # -- prefill ------------------------------------------------------------

    def prefill(
        self,
        token_ids: np.ndarray,
        start_pos: int,
        past_k: np.ndarray | None = None,
        past_v: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run `token_ids` (positions start_pos..start_pos+T) against an optional past.

        Returns (logits_for_last_token [V], k_new [L,H,T,D], v_new [L,H,T,D]).
        `past_k/past_v` carry the cached prefix — that is how a prefix-cache hit
        turns into skipped compute rather than just skipped memory.
        """
        token_ids = np.asarray(token_ids, dtype=np.int64)
        T = int(token_ids.shape[0])
        if T == 0:
            raise ValueError("prefill needs at least one token")

        P = 0 if past_k is None else int(past_k.shape[2])
        positions = np.arange(start_pos, start_pos + T, dtype=np.int64)
        x = self._embed(token_ids, positions)

        L, H, D = self.cfg.n_layer, self.cfg.n_head, self.cfg.head_dim
        HKV = self.cfg.kv_heads
        k_new = np.empty((L, HKV, T, D), dtype=self.dtype)
        v_new = np.empty((L, HKV, T, D), dtype=self.dtype)

        # causal mask: query i (absolute position P+i) may see key j iff j <= P+i
        q_abs = np.arange(P, P + T)[:, None]
        k_abs = np.arange(P + T)[None, :]
        mask = (k_abs > q_abs)  # [T, P+T]

        for li, l in enumerate(self.w.layers):
            h = layer_norm(x, l.ln1_g, l.ln1_b, self.eps)
            qkv = h @ l.attn_w + l.attn_b
            E, KVD = self.cfg.n_embd, self.cfg.kv_dim
            HKV = self.cfg.kv_heads
            q = self._split_heads(qkv[:, :E])
            k = self._split_heads(qkv[:, E : E + KVD], HKV)
            v = self._split_heads(qkv[:, E + KVD :], HKV)

            k_new[li] = k
            v_new[li] = v

            if P:
                k_full = np.concatenate([past_k[li].astype(self.dtype, copy=False), k], axis=1)
                v_full = np.concatenate([past_v[li].astype(self.dtype, copy=False), v], axis=1)
            else:
                k_full, v_full = k, v
            k_full = self._expand_kv(k_full)
            v_full = self._expand_kv(v_full)

            scores = (q @ k_full.transpose(0, 2, 1)) * self.scale   # [H, T, P+T]
            scores = np.where(mask[None, :, :], np.float32(-1e30), scores)
            ctx = softmax(scores, axis=-1) @ v_full                  # [H, T, D]

            attn_out = ctx.transpose(1, 0, 2).reshape(T, E)
            x = x + (attn_out @ l.proj_w + l.proj_b)
            x = x + self._mlp(layer_norm(x, l.ln2_g, l.ln2_b, self.eps), l)

        logits = self.logits_from_hidden(x[-1:])[0]
        return logits, k_new, v_new

    # -- decode -------------------------------------------------------------

    def decode_batch(
        self,
        token_ids: np.ndarray,
        positions: np.ndarray,
        past_ks: list[np.ndarray],
        past_vs: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One token for each of B sequences with independent context lengths.

        past_ks[b] / past_vs[b] are [L, H, P_b, D]. Returns
        (logits [B, V], k_new [B, L, H, 1, D], v_new [B, L, H, 1, D]).
        """
        token_ids = np.asarray(token_ids, dtype=np.int64)
        positions = np.asarray(positions, dtype=np.int64)
        B = int(token_ids.shape[0])
        L, H, D, E = self.cfg.n_layer, self.cfg.n_head, self.cfg.head_dim, self.cfg.n_embd

        HKV = self.cfg.kv_heads
        x = self._embed(token_ids, positions)          # [B, E]
        k_new = np.empty((B, L, HKV, 1, D), dtype=self.dtype)
        v_new = np.empty((B, L, HKV, 1, D), dtype=self.dtype)

        for li, l in enumerate(self.w.layers):
            h = layer_norm(x, l.ln1_g, l.ln1_b, self.eps)
            qkv = h @ l.attn_w + l.attn_b               # [B, E+2*kv_dim] <- batched GEMM

            KVD = self.cfg.kv_dim
            q = qkv[:, :E].reshape(B, H, D)
            k = qkv[:, E : E + KVD].reshape(B, HKV, D)
            v = qkv[:, E + KVD :].reshape(B, HKV, D)

            k_new[:, li, :, 0, :] = k
            v_new[:, li, :, 0, :] = v

            attn_out = np.empty((B, E), dtype=self.dtype)
            for b in range(B):                          # <- paged attention, per sequence
                pk = past_ks[b][li]
                pv = past_vs[b][li]
                if pk.shape[1]:
                    k_full = np.concatenate(
                        [pk.astype(self.dtype, copy=False), k[b][:, None, :]], axis=1
                    )
                    v_full = np.concatenate(
                        [pv.astype(self.dtype, copy=False), v[b][:, None, :]], axis=1
                    )
                else:
                    k_full = k[b][:, None, :]
                    v_full = v[b][:, None, :]

                k_full = self._expand_kv(k_full)
                v_full = self._expand_kv(v_full)
                # [H,1,D] x [H,D,P+1] -> [H,1,P+1]; no mask needed, all past is visible
                scores = (q[b][:, None, :] @ k_full.transpose(0, 2, 1)) * self.scale
                ctx = softmax(scores, axis=-1) @ v_full   # [H, 1, D]
                attn_out[b] = ctx.reshape(E)

            x = x + (attn_out @ l.proj_w + l.proj_b)
            x = x + self._mlp(layer_norm(x, l.ln2_g, l.ln2_b, self.eps), l)

        logits = self.logits_from_hidden(x)             # [B, V]
        return logits, k_new, v_new
