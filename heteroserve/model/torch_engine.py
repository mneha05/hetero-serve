"""GPT-2 on CUDA via PyTorch, with two decode paths.

`decode_batch`        the portable baseline. Caller gathers each sequence's KV
                      into contiguous tensors, exactly like the numpy and
                      OpenVINO engines. Same interface, so it drops into the
                      existing worker with no other changes.

`decode_batch_paged`  the fast path. Takes *block tables* instead of gathered
                      tensors, writes the new K/V straight into the paged pool,
                      and calls the fused kernel that walks the block table
                      inside the attention loop. No contiguous copy ever exists.

Having both in one class is deliberate: they are the control and the treatment
for the kernel benchmark, and `scripts/bench_kernel.py` checks they agree
numerically before reporting any speedup.
"""

from __future__ import annotations

import numpy as np

from ..config import ModelConfig
from .paged_attn import paged_attention, which_backend
from .weights import GPT2Weights


class TorchEngine:
    name = "torch"

    def __init__(
        self,
        weights: GPT2Weights,
        cfg: ModelConfig | None = None,
        device: str = "cuda:0",
        dtype: str = "float32",
    ):
        import torch

        self.torch = torch
        self.cfg = cfg or weights.cfg
        self.device = torch.device(device)
        # float32 by default: GPT-2's residual stream reaches norms in the
        # thousands, and correctness matters more here than the last 30% of
        # throughput. Pass dtype="float16" once you have verified outputs.
        self.tdtype = getattr(torch, dtype)
        self.eps = float(self.cfg.layer_norm_eps)
        self.scale = 1.0 / np.sqrt(self.cfg.head_dim)
        self.kv_heads = self.cfg.kv_heads
        self.kv_group = self.cfg.kv_group

        def T(a):
            return torch.as_tensor(
                np.ascontiguousarray(a), dtype=self.tdtype, device=self.device
            )

        self.wte = T(weights.wte)
        self.wpe = T(weights.wpe)
        self.lnf_g, self.lnf_b = T(weights.lnf_g), T(weights.lnf_b)
        self.layers = [
            {
                "ln1_g": T(l.ln1_g), "ln1_b": T(l.ln1_b),
                "attn_w": T(l.attn_w), "attn_b": T(l.attn_b),
                "proj_w": T(l.proj_w), "proj_b": T(l.proj_b),
                "ln2_g": T(l.ln2_g), "ln2_b": T(l.ln2_b),
                "fc_w": T(l.fc_w), "fc_b": T(l.fc_b),
                "fcp_w": T(l.fcp_w), "fcp_b": T(l.fcp_b),
            }
            for l in weights.layers
        ]
        self.kernel_backend = which_backend()

    # -- pieces -------------------------------------------------------------

    def _ln(self, x, g, b):
        return self.torch.nn.functional.layer_norm(x, (self.cfg.n_embd,), g, b, self.eps)

    def _mlp(self, x, lw):
        h = self.torch.nn.functional.gelu(x @ lw["fc_w"] + lw["fc_b"], approximate="tanh")
        return h @ lw["fcp_w"] + lw["fcp_b"]

    def _expand_kv(self, x, dim):
        """Share each KV head across its query group. Reference path only --
        the fused kernels index the shared head instead of materialising it."""
        return x if self.kv_group == 1 else x.repeat_interleave(self.kv_group, dim=dim)

    def _logits(self, x):
        return (self._ln(x, self.lnf_g, self.lnf_b) @ self.wte.T).float()

    def _embed(self, ids, pos):
        return self.wte[ids] + self.wpe[pos]

    # -- prefill ------------------------------------------------------------

    def prefill(self, token_ids, start_pos: int, past_k=None, past_v=None):
        torch = self.torch
        cfg = self.cfg
        L, H, D, E = cfg.n_layer, cfg.n_head, cfg.head_dim, cfg.n_embd

        ids = torch.as_tensor(np.asarray(token_ids), dtype=torch.long, device=self.device)
        T = int(ids.shape[0])
        if T == 0:
            raise ValueError("prefill needs at least one token")
        P = 0 if past_k is None else int(past_k.shape[2])

        pos = torch.arange(start_pos, start_pos + T, device=self.device)
        x = self._embed(ids, pos)

        HKV = self.kv_heads
        k_new = torch.empty((L, HKV, T, D), dtype=self.tdtype, device=self.device)
        v_new = torch.empty_like(k_new)

        q_abs = torch.arange(P, P + T, device=self.device).unsqueeze(1)
        k_abs = torch.arange(P + T, device=self.device).unsqueeze(0)
        mask = (k_abs > q_abs)                       # [T, P+T] True = blocked

        pk = self._to_dev(past_k) if past_k is not None else None
        pv = self._to_dev(past_v) if past_v is not None else None

        with torch.inference_mode():
            for li, lw in enumerate(self.layers):
                h = self._ln(x, lw["ln1_g"], lw["ln1_b"])
                qkv = h @ lw["attn_w"] + lw["attn_b"]
                KVD = cfg.kv_dim
                q = qkv[:, :E].view(T, H, D).permute(1, 0, 2)
                k = qkv[:, E:E + KVD].view(T, HKV, D).permute(1, 0, 2)
                v = qkv[:, E + KVD:].view(T, HKV, D).permute(1, 0, 2)

                k_new[li], v_new[li] = k, v

                k_full = torch.cat([pk[li], k], dim=1) if P else k
                v_full = torch.cat([pv[li], v], dim=1) if P else v
                k_full = self._expand_kv(k_full, 0)
                v_full = self._expand_kv(v_full, 0)

                scores = (q @ k_full.transpose(1, 2)) * self.scale
                scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
                ctx = torch.softmax(scores, dim=-1) @ v_full          # [H, T, D]

                x = x + (ctx.permute(1, 0, 2).reshape(T, E) @ lw["proj_w"] + lw["proj_b"])
                x = x + self._mlp(self._ln(x, lw["ln2_g"], lw["ln2_b"]), lw)

            logits = self._logits(x[-1:])[0]

        return self._out(logits), self._out(k_new), self._out(v_new)

    # -- decode: portable gather-based path ---------------------------------

    def decode_batch(self, token_ids, positions, past_ks, past_vs):
        torch = self.torch
        cfg = self.cfg
        L, H, D, E = cfg.n_layer, cfg.n_head, cfg.head_dim, cfg.n_embd
        B = len(past_ks)

        ids = torch.as_tensor(np.asarray(token_ids), dtype=torch.long, device=self.device)
        pos = torch.as_tensor(np.asarray(positions), dtype=torch.long, device=self.device)
        x = self._embed(ids, pos)                                     # [B, E]

        pks = [self._to_dev(p) for p in past_ks]
        pvs = [self._to_dev(p) for p in past_vs]

        HKV = self.kv_heads
        k_new = torch.empty((B, L, HKV, 1, D), dtype=self.tdtype, device=self.device)
        v_new = torch.empty_like(k_new)

        with torch.inference_mode():
            for li, lw in enumerate(self.layers):
                h = self._ln(x, lw["ln1_g"], lw["ln1_b"])
                qkv = h @ lw["attn_w"] + lw["attn_b"]
                KVD = cfg.kv_dim
                q = qkv[:, :E].view(B, H, D)
                k = qkv[:, E:E + KVD].view(B, HKV, D)
                v = qkv[:, E + KVD:].view(B, HKV, D)

                k_new[:, li, :, 0, :] = k
                v_new[:, li, :, 0, :] = v

                attn = torch.empty((B, E), dtype=self.tdtype, device=self.device)
                for b in range(B):
                    kf = torch.cat([pks[b][li], k[b].unsqueeze(1)], dim=1)
                    vf = torch.cat([pvs[b][li], v[b].unsqueeze(1)], dim=1)
                    kf = self._expand_kv(kf, 0)
                    vf = self._expand_kv(vf, 0)
                    s = (q[b].unsqueeze(1) @ kf.transpose(1, 2)) * self.scale
                    attn[b] = (torch.softmax(s, dim=-1) @ vf).reshape(E)

                x = x + (attn @ lw["proj_w"] + lw["proj_b"])
                x = x + self._mlp(self._ln(x, lw["ln2_g"], lw["ln2_b"]), lw)

            logits = self._logits(x)

        return self._out(logits), self._out(k_new), self._out(v_new)

    # -- decode: fused paged path -------------------------------------------

    def decode_batch_paged(self, token_ids, positions, block_tables, context_lens, alloc):
        """Attention reads the paged pool directly; nothing is gathered.

        `alloc` is a TorchBlockAllocator whose pool lives on this device. New K/V
        for the token being decoded is written into the pool first, then
        `context_lens + 1` lets attention see it — same ordering vLLM uses.

        Returns logits only: the KV is already in the cache, so there is nothing
        for the caller to write back.
        """
        torch = self.torch
        cfg = self.cfg
        H, D, E = cfg.n_head, cfg.head_dim, cfg.n_embd
        B = len(context_lens)

        ids = torch.as_tensor(np.asarray(token_ids), dtype=torch.long, device=self.device)
        pos_np = np.asarray(positions)
        pos = torch.as_tensor(pos_np, dtype=torch.long, device=self.device)
        x = self._embed(ids, pos)

        tables = block_tables if torch.is_tensor(block_tables) else \
            alloc.block_table_tensor(block_tables)
        ctx_lens = torch.as_tensor(
            np.asarray(context_lens) + 1, dtype=torch.int32, device=self.device
        )

        # Flat slot for each sequence's new token, via its own block table.
        slots = torch.stack([
            alloc.slots_for(
                tables[b][tables[b] >= 0].tolist(), np.asarray([pos_np[b]])
            )[0]
            for b in range(B)
        ])

        with torch.inference_mode():
            for li, lw in enumerate(self.layers):
                h = self._ln(x, lw["ln1_g"], lw["ln1_b"])
                qkv = h @ lw["attn_w"] + lw["attn_b"]
                KVD = cfg.kv_dim
                q = qkv[:, :E].view(B, H, D)
                k = qkv[:, E:E + KVD].view(B, alloc.model.kv_heads, D)
                v = qkv[:, E + KVD:].view(B, alloc.model.kv_heads, D)

                fk, fv = alloc.flat_kv(li)
                fk[slots] = k.to(alloc.torch_dtype)
                fv[slots] = v.to(alloc.torch_dtype)

                ctx = paged_attention(
                    q.float(), alloc.pool[li, 0], alloc.pool[li, 1],
                    tables, ctx_lens, self.scale,
                )                                                     # [B, H, D]

                attn = ctx.to(self.tdtype).reshape(B, E)
                x = x + (attn @ lw["proj_w"] + lw["proj_b"])
                x = x + self._mlp(self._ln(x, lw["ln2_g"], lw["ln2_b"]), lw)

            logits = self._logits(x)

        return self._out(logits)

    # -- conversion ---------------------------------------------------------

    def _to_dev(self, a):
        torch = self.torch
        if torch.is_tensor(a):
            return a.to(device=self.device, dtype=self.tdtype)
        return torch.as_tensor(
            np.ascontiguousarray(a), dtype=self.tdtype, device=self.device
        )

    def _out(self, t):
        """Hand results back as numpy so the rest of the system is unchanged."""
        return t.detach().to("cpu").float().numpy()
