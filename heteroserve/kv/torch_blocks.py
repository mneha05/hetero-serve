"""A KV pool that lives in GPU memory instead of host RAM.

All the interesting bookkeeping — free list, refcounts, chain hashing, LRU
eviction, prefix matching — is device-independent and inherited unchanged from
`BlockAllocator`. Only the five methods that actually touch storage are
overridden here.

That split is what makes the fused CUDA kernel possible: the pool is a single
contiguous `[n_layer, 2, num_blocks, block_size, n_head, head_dim]` CUDA tensor,
so `pool[layer, 0]` is exactly the `[num_blocks, block_size, H, D]` view the
kernel indexes through the block table. Nothing is ever gathered into a
contiguous per-sequence buffer first.

Migration still crosses the network as host bytes, so `export_blocks` /
`import_blocks` move through CPU — a real deployment would use GPUDirect RDMA
and skip the bounce, which is noted as future work rather than pretended away.
"""

from __future__ import annotations

import numpy as np

from ..config import KVConfig, ModelConfig
from .blocks import BlockAllocator


class TorchBlockAllocator(BlockAllocator):
    """Same allocator, storage on a CUDA (or any torch) device."""

    def __init__(
        self,
        kv: KVConfig,
        model: ModelConfig,
        device: str = "cuda:0",
        dtype: str | None = None,
    ):
        import torch

        # Build all the metadata but skip the numpy pool allocation.
        super().__init__(kv, model, allocate_pool=False)

        self.torch = torch
        self.device = torch.device(device)
        self.torch_dtype = getattr(torch, dtype or kv.dtype)
        self.pool = torch.zeros(self.shape, dtype=self.torch_dtype, device=self.device)

    # -- helpers ------------------------------------------------------------

    def _as_tensor(self, a):
        if isinstance(a, self.torch.Tensor):
            return a.to(device=self.device, dtype=self.torch_dtype)
        return self.torch.as_tensor(
            np.ascontiguousarray(a), dtype=self.torch_dtype, device=self.device
        )

    def flat_kv(self, layer: int):
        """[num_blocks*block_size, H, D] views of one layer's K and V."""
        H, D = self.model.kv_heads, self.model.head_dim
        n = self.num_blocks * self.block_size
        return (
            self.pool[layer, 0].reshape(n, H, D),
            self.pool[layer, 1].reshape(n, H, D),
        )

    def slots_for(self, block_ids, positions):
        """Map absolute token positions to flat slots via the block table."""
        torch = self.torch
        tbl = torch.as_tensor(block_ids, dtype=torch.long, device=self.device)
        pos = torch.as_tensor(positions, dtype=torch.long, device=self.device)
        return tbl[pos // self.block_size] * self.block_size + (pos % self.block_size)

    def block_table_tensor(self, tables: list[list[int]], pad_to: int | None = None):
        """[B, max_blocks] int32 block table, -1 padded — what the kernel walks."""
        torch = self.torch
        width = pad_to or max((len(t) for t in tables), default=1)
        out = torch.full((len(tables), width), -1, dtype=torch.int32, device=self.device)
        for i, t in enumerate(tables):
            if t:
                out[i, : len(t)] = torch.as_tensor(
                    t[:width], dtype=torch.int32, device=self.device
                )
        return out

    # -- storage overrides --------------------------------------------------

    def write_kv(self, block_ids, start_pos, k, v) -> None:
        """Scatter [n_layer, H, T, D] K/V into the paged pool."""
        torch = self.torch
        k = self._as_tensor(k)
        v = self._as_tensor(v)
        n_tokens = int(k.shape[2])
        if n_tokens == 0:
            return
        slots = self.slots_for(
            block_ids, np.arange(start_pos, start_pos + n_tokens)
        )
        # [L, H, T, D] -> [L, T, H, D] to match the pool's token-major pages
        kt = k.permute(0, 2, 1, 3).contiguous()
        vt = v.permute(0, 2, 1, 3).contiguous()
        for l in range(self.model.n_layer):
            fk, fv = self.flat_kv(l)
            fk[slots] = kt[l]
            fv[slots] = vt[l]

    def gather_kv(self, block_ids, length):
        """Contiguous [n_layer, H, length, D] view — the cost the kernel avoids."""
        torch = self.torch
        L, H, D = self.model.n_layer, self.model.kv_heads, self.model.head_dim
        if length == 0:
            z = torch.zeros((L, H, 0, D), dtype=self.torch_dtype, device=self.device)
            return z, z.clone()

        slots = self.slots_for(block_ids, np.arange(length))
        ks = torch.empty((L, length, H, D), dtype=self.torch_dtype, device=self.device)
        vs = torch.empty_like(ks)
        for l in range(L):
            fk, fv = self.flat_kv(l)
            ks[l] = fk[slots]
            vs[l] = fv[slots]
        return ks.permute(0, 2, 1, 3), vs.permute(0, 2, 1, 3)

    def gather_kv_batch(self, tables, length: int, layer: int):
        """Batched single-layer gather -> ([B, length, H, D], same for V).

        This is precisely the work the fused kernel deletes, isolated so
        `scripts/bench_kernel.py` can time it against the kernel directly.
        """
        torch = self.torch
        pos = np.arange(length)
        slots = torch.stack([self.slots_for(t, pos) for t in tables])   # [B, length]
        fk, fv = self.flat_kv(layer)
        return fk[slots], fv[slots]

    def export_blocks(self, block_ids) -> np.ndarray:
        """Serialise for migration. Bounces via host — see module docstring."""
        torch = self.torch
        idx = torch.as_tensor(block_ids, dtype=torch.long, device=self.device)
        sel = self.pool[:, :, idx]                      # [L, 2, n, BS, H, D]
        return sel.permute(2, 0, 1, 3, 4, 5).contiguous().cpu().numpy()

    def write_blocks(self, block_ids, payload) -> None:
        torch = self.torch
        data = torch.as_tensor(
            np.ascontiguousarray(payload), dtype=self.torch_dtype, device=self.device
        )
        for i, blk in enumerate(block_ids):
            self.pool[:, :, blk] = data[i]

    def import_blocks(self, payload) -> list[int]:
        ids = self.reserve_blocks(int(payload.shape[0]))
        self.write_blocks(ids, payload)
        return ids
