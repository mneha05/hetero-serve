"""Paged KV cache: fixed-size blocks, refcounts, prefix-hash sharing, LRU eviction.

This is the piece that makes the whole project interesting. Because the cache is
paged, a prefix is a *set of blocks* rather than a contiguous slab, which means
we can (a) share it between sequences for free, and (b) ship it across a network
link to another accelerator and have the receiver splice it straight in.

Layout of the pool:

    [n_layer, 2, num_blocks, block_size, n_head, head_dim]
                ^-- 0 = K, 1 = V

Blocks are hashed by *chain*: block i's hash covers every token from position 0
through the end of block i, so a hash match is a genuine shared-prefix match and
not just a coincidental block collision.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from ..config import KVConfig, ModelConfig


def hash_block(prev_hash: int | None, token_ids: tuple[int, ...]) -> int:
    """Stable 64-bit chain hash. Must agree across processes, so no builtin hash()."""
    h = hashlib.blake2b(digest_size=8)
    h.update(b"\x00" * 8 if prev_hash is None else prev_hash.to_bytes(8, "little"))
    h.update(np.asarray(token_ids, dtype=np.int32).tobytes())
    return int.from_bytes(h.digest(), "little")


def chain_hashes(token_ids: list[int], block_size: int) -> list[int]:
    """Chain hash for every *full* block of the token sequence."""
    out: list[int] = []
    prev: int | None = None
    n_full = len(token_ids) // block_size
    for i in range(n_full):
        chunk = tuple(token_ids[i * block_size : (i + 1) * block_size])
        prev = hash_block(prev, chunk)
        out.append(prev)
    return out


class OutOfBlocks(RuntimeError):
    pass


@dataclass
class Allocation:
    """Result of reserving blocks for a sequence."""

    block_ids: list[int]
    num_cached_tokens: int          # tokens served from an existing cached prefix
    reused_blocks: list[int] = field(default_factory=list)


class BlockAllocator:
    """Owns one worker's KV memory."""

    def __init__(self, kv: KVConfig, model: ModelConfig, allocate_pool: bool = True):
        self.kv = kv
        self.model = model
        self.block_size = kv.block_size
        self.num_blocks = kv.num_blocks
        self.dtype = kv.np_dtype

        self.shape = (
            model.n_layer,
            2,
            kv.num_blocks,
            kv.block_size,
            model.n_head,
            model.head_dim,
        )
        # The real memory. ~576 KiB per block for GPT-2 @ fp16 / block_size 16.
        self.pool = np.zeros(self.shape, dtype=self.dtype) if allocate_pool else None

        self._free: list[int] = list(range(kv.num_blocks - 1, -1, -1))  # pop() = lowest id
        self._ref = np.zeros(kv.num_blocks, dtype=np.int32)

        # prefix cache
        self._hash_to_block: dict[int, int] = {}
        self._block_to_hash: dict[int, int] = {}
        # hash -> None, ordered oldest-first; only holds blocks with refcount 0
        self._evictable: OrderedDict[int, None] = OrderedDict()

        # counters
        self.stat_hits = 0
        self.stat_misses = 0
        self.stat_evictions = 0

    # -- capacity -----------------------------------------------------------

    @property
    def num_free(self) -> int:
        return len(self._free) + len(self._evictable)

    @property
    def num_used(self) -> int:
        return self.num_blocks - self.num_free

    @property
    def utilisation(self) -> float:
        return self.num_used / self.num_blocks if self.num_blocks else 0.0

    def block_bytes(self) -> int:
        return self.kv.block_bytes(self.model)

    # -- raw block lifecycle ------------------------------------------------

    def _take_free_block(self) -> int:
        if self._free:
            return self._free.pop()
        # Nothing free: evict the least-recently-used cached block.
        if self._evictable:
            victim_hash, _ = self._evictable.popitem(last=False)
            block = self._hash_to_block.pop(victim_hash)
            self._block_to_hash.pop(block, None)
            self.stat_evictions += 1
            return block
        raise OutOfBlocks("KV pool exhausted")

    def incref(self, block_id: int) -> None:
        if self._ref[block_id] == 0:
            h = self._block_to_hash.get(block_id)
            if h is not None:
                self._evictable.pop(h, None)
        self._ref[block_id] += 1

    def decref(self, block_id: int) -> None:
        self._ref[block_id] -= 1
        if self._ref[block_id] <= 0:
            self._ref[block_id] = 0
            h = self._block_to_hash.get(block_id)
            if h is not None and self._hash_to_block.get(h) == block_id:
                # Keep the contents around — it may be a useful prefix later.
                self._evictable[h] = None
                self._evictable.move_to_end(h)
            else:
                self._free.append(block_id)

    def free_sequence(self, block_ids: list[int]) -> None:
        for b in block_ids:
            self.decref(b)

    # -- prefix-aware allocation -------------------------------------------

    def match_prefix(self, token_ids: list[int]) -> tuple[list[int], int]:
        """Longest cached prefix. Returns (block_ids, num_cached_tokens).

        Stops at the first miss: a prefix is only valid if every block before it
        also matched.
        """
        hashes = chain_hashes(token_ids, self.block_size)
        matched: list[int] = []
        for h in hashes:
            block = self._hash_to_block.get(h)
            if block is None:
                break
            matched.append(block)
        return matched, len(matched) * self.block_size

    def allocate(self, token_ids: list[int], max_new_tokens: int = 0) -> Allocation:
        """Reserve blocks to hold `token_ids` (+ room to grow), reusing cached prefix."""
        total = len(token_ids) + max_new_tokens
        need_blocks = (total + self.block_size - 1) // self.block_size

        reused, cached_tokens = self.match_prefix(token_ids)
        # Never reuse more blocks than the sequence actually needs.
        reused = reused[:need_blocks]
        cached_tokens = min(cached_tokens, len(reused) * self.block_size)

        block_ids = list(reused)
        for b in reused:
            self.incref(b)

        try:
            while len(block_ids) < need_blocks:
                b = self._take_free_block()
                self._ref[b] = 1
                block_ids.append(b)
        except OutOfBlocks:
            # roll back so a failed admission leaves no residue
            for b in block_ids:
                self.decref(b)
            raise

        if cached_tokens:
            self.stat_hits += cached_tokens
        self.stat_misses += len(token_ids) - cached_tokens

        return Allocation(
            block_ids=block_ids,
            num_cached_tokens=cached_tokens,
            reused_blocks=reused,
        )

    def grow(self, block_ids: list[int], new_length: int) -> None:
        """Append blocks in-place so the table can hold `new_length` tokens."""
        need = (new_length + self.block_size - 1) // self.block_size
        while len(block_ids) < need:
            b = self._take_free_block()
            self._ref[b] = 1
            block_ids.append(b)

    def register_full_blocks(
        self, token_ids: list[int], block_ids: list[int], first_index: int = 0
    ) -> None:
        """Publish completed blocks into the prefix cache so others can share them.

        `first_index` says which block of the sequence `block_ids[0]` is, so a
        partial migration can register just the delta it received.
        """
        hashes = chain_hashes(token_ids, self.block_size)[first_index:]
        for i, h in enumerate(hashes):
            if i >= len(block_ids):
                break
            block = block_ids[i]
            existing = self._hash_to_block.get(h)
            if existing == block:
                continue
            if existing is not None:
                continue  # someone already published identical content
            self._hash_to_block[h] = block
            self._block_to_hash[block] = h

    # -- data movement ------------------------------------------------------

    def write_kv(
        self,
        block_ids: list[int],
        start_pos: int,
        k: np.ndarray,
        v: np.ndarray,
    ) -> None:
        """Scatter [n_layer, n_head, T, head_dim] K/V into the paged pool."""
        assert self.pool is not None
        n_tokens = k.shape[2]
        if n_tokens == 0:
            return
        pos = np.arange(start_pos, start_pos + n_tokens)
        bidx = np.asarray(block_ids, dtype=np.int64)[pos // self.block_size]
        offs = pos % self.block_size

        # [n_layer, H, T, D] -> [n_layer, T, H, D] to match pool axis order
        kt = np.ascontiguousarray(k.transpose(0, 2, 1, 3)).astype(self.dtype, copy=False)
        vt = np.ascontiguousarray(v.transpose(0, 2, 1, 3)).astype(self.dtype, copy=False)

        self.pool[:, 0][:, bidx, offs] = kt
        self.pool[:, 1][:, bidx, offs] = vt

    def gather_kv(self, block_ids: list[int], length: int) -> tuple[np.ndarray, np.ndarray]:
        """Gather a contiguous [n_layer, n_head, length, head_dim] view of a sequence.

        This gather is the real cost of paging — we pay it on every decode step,
        exactly like a production paged-attention kernel does.
        """
        assert self.pool is not None
        if length == 0:
            z = np.zeros(
                (self.model.n_layer, self.model.n_head, 0, self.model.head_dim),
                dtype=self.dtype,
            )
            return z, z.copy()

        n_needed = (length + self.block_size - 1) // self.block_size
        bidx = np.asarray(block_ids[:n_needed], dtype=np.int64)

        k = self.pool[:, 0][:, bidx]        # [L, nb, bs, H, D]
        v = self.pool[:, 1][:, bidx]
        L, nb, bs, H, D = k.shape
        k = k.reshape(L, nb * bs, H, D)[:, :length]
        v = v.reshape(L, nb * bs, H, D)[:, :length]
        return k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)

    def export_blocks(self, block_ids: list[int]) -> np.ndarray:
        """Serialise blocks for migration. Shape [n_blocks, n_layer, 2, bs, H, D]."""
        assert self.pool is not None
        bidx = np.asarray(block_ids, dtype=np.int64)
        return np.ascontiguousarray(self.pool[:, :, bidx].transpose(2, 0, 1, 3, 4, 5))

    def import_blocks(self, payload: np.ndarray) -> list[int]:
        """Inverse of export_blocks; returns freshly-allocated block ids."""
        ids = self.reserve_blocks(int(payload.shape[0]))
        self.write_blocks(ids, payload)
        return ids

    def reserve_blocks(self, n: int) -> list[int]:
        """Claim `n` blocks. Cheap, but must run under the worker's state lock.

        Split out from the copy so a multi-megabyte migration does not hold the
        lock — and therefore the engine — for the duration of a memcpy.
        """
        out: list[int] = []
        for _ in range(n):
            b = self._take_free_block()
            self._ref[b] = 1
            out.append(b)
        return out

    def write_blocks(self, block_ids: list[int], payload: np.ndarray) -> None:
        """Bulk-copy migrated blocks in. Safe to call *without* the state lock.

        The blocks are reserved (refcount >= 1), so the allocator will not hand
        them to anyone else and no running sequence points at them. Writing
        disjoint regions of an already-allocated pool from another task cannot
        race with the engine reading its own blocks, and numpy drops the GIL for
        a copy this size, so it genuinely overlaps with the device step.
        """
        assert self.pool is not None
        for i, blk in enumerate(block_ids):
            self.pool[:, :, blk] = payload[i]

    def adopt_migrated(
        self,
        token_ids: list[int],
        block_ids: list[int],
        n_tokens: int,
        first_index: int = 0,
    ) -> None:
        """Publish migrated blocks into this worker's prefix cache.

        `token_ids` is the full prefix (so chain hashes line up); `block_ids`
        holds only the blocks that actually came over the wire, starting at
        block `first_index`.
        """
        full = n_tokens // self.block_size
        self.register_full_blocks(
            token_ids[: full * self.block_size], block_ids, first_index=first_index
        )

    # -- introspection ------------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "num_blocks": self.num_blocks,
            "used": int(self.num_used),
            "free": int(self.num_free),
            "utilisation": round(self.utilisation, 4),
            "cached_blocks": len(self._hash_to_block),
            "evictable": len(self._evictable),
            "hits": self.stat_hits,
            "misses": self.stat_misses,
            "evictions": self.stat_evictions,
            "block_bytes": self.block_bytes(),
        }
