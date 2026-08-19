"""Correctness gates for the PyTorch/CUDA path and the paged-attention kernel.

Almost all of this runs on CPU torch, so the engine, the GPU-resident allocator
and the paged-attention *algorithm* are verified without any GPU. Only the fused
CUDA kernel itself needs real hardware, and those tests skip cleanly.

The numpy engine is the oracle throughout: if the torch path disagrees with it,
the torch path is wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from heteroserve.config import KVConfig, ModelConfig
from heteroserve.kv.blocks import BlockAllocator
from heteroserve.kv.torch_blocks import TorchBlockAllocator
from heteroserve.model.numpy_engine import NumpyEngine
from heteroserve.model.paged_attn import (
    paged_attention,
    paged_attention_torch,
    which_backend,
)
from heteroserve.model.torch_engine import TorchEngine
from heteroserve.model.weights import synthetic_gpt2

HAS_CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA device")


@pytest.fixture(scope="module")
def pair():
    cfg = ModelConfig.tiny()
    w = synthetic_gpt2(cfg, seed=11)
    return NumpyEngine(w, cfg), TorchEngine(w, cfg, device="cpu", dtype="float32"), cfg


# ---------------------------------------------------------------------------
# engine parity with the numpy oracle
# ---------------------------------------------------------------------------


def test_torch_prefill_matches_numpy(pair):
    ref, eng, _ = pair
    toks = np.array([4, 19, 200, 7, 88, 3, 41, 900, 12, 6])
    rl, rk, rv = ref.prefill(toks, 0)
    tl, tk, tv = eng.prefill(toks, 0)
    np.testing.assert_allclose(rl, tl, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(rk, tk, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(rv, tv, rtol=1e-4, atol=1e-4)


def test_torch_chunked_prefill_matches_numpy(pair):
    """Prefill with an existing past — the prefix-cache-hit path."""
    ref, eng, _ = pair
    toks = np.array([4, 19, 200, 7, 88, 3, 41, 900, 12, 6])
    _, rk, rv = ref.prefill(toks, 0)
    tail = np.array([55, 66, 77])
    rl, _, _ = ref.prefill(tail, 10, rk, rv)
    tl, _, _ = eng.prefill(tail, 10, rk, rv)
    np.testing.assert_allclose(rl, tl, rtol=1e-4, atol=1e-4)


def test_torch_decode_matches_numpy(pair):
    ref, eng, cfg = pair
    rng = np.random.default_rng(3)
    L, H, D = cfg.n_layer, cfg.n_head, cfg.head_dim
    lens = [5, 13, 9]
    pk = [rng.standard_normal((L, H, n, D)).astype(np.float32) for n in lens]
    pv = [rng.standard_normal((L, H, n, D)).astype(np.float32) for n in lens]
    toks, pos = np.array([7, 400, 21]), np.array(lens)

    rl, rk, rv = ref.decode_batch(toks, pos, pk, pv)
    tl, tk, tv = eng.decode_batch(toks, pos, pk, pv)
    np.testing.assert_allclose(rl, tl, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(rk, tk, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# the GPU-resident allocator behaves like the numpy one
# ---------------------------------------------------------------------------


def test_torch_allocator_roundtrip():
    cfg = ModelConfig.tiny()
    kv = KVConfig(block_size=8, num_blocks=32, dtype="float32")
    alloc = TorchBlockAllocator(kv, cfg, device="cpu")

    tokens = list(range(20))
    a = alloc.allocate(tokens)
    L, H, D = cfg.n_layer, cfg.n_head, cfg.head_dim
    k = np.random.default_rng(0).standard_normal((L, H, 20, D)).astype(np.float32)
    v = np.random.default_rng(1).standard_normal((L, H, 20, D)).astype(np.float32)

    alloc.write_kv(a.block_ids, 0, k, v)
    gk, gv = alloc.gather_kv(a.block_ids, 20)
    np.testing.assert_allclose(gk.cpu().numpy(), k, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gv.cpu().numpy(), v, rtol=1e-6, atol=1e-6)


def test_torch_allocator_matches_numpy_allocator_semantics():
    """Prefix sharing / refcounting is inherited, so it must behave identically."""
    cfg = ModelConfig.tiny()
    kv = KVConfig(block_size=4, num_blocks=64, dtype="float32")
    npa = BlockAllocator(kv, cfg)
    tpa = TorchBlockAllocator(kv, cfg, device="cpu")

    shared = list(range(100, 112))
    for alloc in (npa, tpa):
        a = alloc.allocate(shared)
        alloc.register_full_blocks(shared, a.block_ids)
        b = alloc.allocate(shared + [7, 7, 7, 7])
        assert b.num_cached_tokens == 12
        assert b.block_ids[:3] == a.block_ids[:3]

    assert npa.num_used == tpa.num_used


def test_torch_allocator_migration_roundtrip():
    cfg = ModelConfig.tiny()
    kv = KVConfig(block_size=4, num_blocks=16, dtype="float32")
    src = TorchBlockAllocator(kv, cfg, device="cpu")
    dst = TorchBlockAllocator(kv, cfg, device="cpu")

    tokens = list(range(16))
    a = src.allocate(tokens)
    L, H, D = cfg.n_layer, cfg.n_head, cfg.head_dim
    k = np.random.default_rng(2).standard_normal((L, H, 16, D)).astype(np.float32)
    src.write_kv(a.block_ids, 0, k, k)

    payload = src.export_blocks(a.block_ids)
    new_ids = dst.import_blocks(payload)
    gk, _ = dst.gather_kv(new_ids, 16)
    np.testing.assert_allclose(gk.cpu().numpy(), k, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# paged attention: the algorithm, then the kernel
# ---------------------------------------------------------------------------


def _reference_attention(q, k, v, scale):
    """Plain dense attention over the true (unpaged) context."""
    s = np.einsum("hd,khd->hk", q, k) * scale
    p = np.exp(s - s.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    return np.einsum("hk,khd->hd", p, v)


def _build_paged_case(device, dtype=torch.float32, lens=(9, 31, 16), seed=5):
    """Lay sequences of differing lengths into a paged pool, shuffled blocks."""
    cfg = ModelConfig.tiny()
    kv = KVConfig(block_size=8, num_blocks=64, dtype="float32")
    alloc = TorchBlockAllocator(kv, cfg, device=device)
    rng = np.random.default_rng(seed)

    tables, truth = [], []
    for i, n in enumerate(lens):
        a = alloc.allocate([int(t) for t in rng.integers(1, 4000, size=n)])
        L, H, D = cfg.n_layer, cfg.n_head, cfg.head_dim
        k = rng.standard_normal((L, H, n, D)).astype(np.float32)
        v = rng.standard_normal((L, H, n, D)).astype(np.float32)
        alloc.write_kv(a.block_ids, 0, k, v)
        tables.append(a.block_ids)
        truth.append((k, v))

    q = rng.standard_normal((len(lens), cfg.n_head, cfg.head_dim)).astype(np.float32)
    return cfg, alloc, tables, truth, q


def test_paged_attention_torch_matches_dense_attention():
    """The block-table walk must equal attention over the real context."""
    cfg, alloc, tables, truth, q = _build_paged_case("cpu")
    lens = [t[0].shape[2] for t in truth]
    width = max(len(t) for t in tables)
    bt = alloc.block_table_tensor(tables, pad_to=width)
    ctx_lens = torch.tensor(lens, dtype=torch.int32)
    scale = 1.0 / np.sqrt(cfg.head_dim)

    got = paged_attention_torch(
        torch.as_tensor(q), alloc.pool[0, 0], alloc.pool[0, 1], bt, ctx_lens, scale
    ).numpy()

    for b, (k, v) in enumerate(truth):
        want = _reference_attention(
            q[b], k[0].transpose(1, 0, 2), v[0].transpose(1, 0, 2), scale
        )
        np.testing.assert_allclose(got[b], want, rtol=1e-4, atol=1e-4)


@cuda_only
def test_cuda_kernel_matches_torch_reference():
    """The fused kernel must agree with the torch paged implementation."""
    assert which_backend() == "cuda", "CUDA kernel did not compile"
    cfg, alloc, tables, truth, q = _build_paged_case("cuda:0")
    lens = [t[0].shape[2] for t in truth]
    bt = alloc.block_table_tensor(tables, pad_to=max(len(t) for t in tables))
    ctx_lens = torch.tensor(lens, dtype=torch.int32, device="cuda:0")
    scale = 1.0 / np.sqrt(cfg.head_dim)
    qt = torch.as_tensor(q, device="cuda:0")

    fused = paged_attention(qt, alloc.pool[0, 0], alloc.pool[0, 1], bt, ctx_lens, scale)
    ref = paged_attention_torch(
        qt, alloc.pool[0, 0], alloc.pool[0, 1], bt, ctx_lens, scale
    )
    np.testing.assert_allclose(
        fused.cpu().numpy(), ref.cpu().numpy(), rtol=1e-3, atol=1e-3
    )


@cuda_only
def test_paged_decode_matches_gather_decode():
    """End to end: the fused decode path == the gather decode path."""
    cfg = ModelConfig.tiny()
    w = synthetic_gpt2(cfg, seed=11)
    ref = NumpyEngine(w, cfg)
    eng = TorchEngine(w, cfg, device="cuda:0", dtype="float32")

    kv = KVConfig(block_size=8, num_blocks=64, dtype="float32")
    alloc = TorchBlockAllocator(kv, cfg, device="cuda:0")

    rng = np.random.default_rng(17)
    lens, tables, pasts = [11, 20], [], []
    for n in lens:
        a = alloc.allocate([int(t) for t in rng.integers(1, 4000, size=n + 1)])
        L, H, D = cfg.n_layer, cfg.n_head, cfg.head_dim
        k = rng.standard_normal((L, H, n, D)).astype(np.float32)
        v = rng.standard_normal((L, H, n, D)).astype(np.float32)
        alloc.write_kv(a.block_ids, 0, k, v)
        tables.append(a.block_ids)
        pasts.append((k, v))

    toks = np.array([31, 77])
    pos = np.array(lens)

    want, _, _ = ref.decode_batch(toks, pos, [p[0] for p in pasts], [p[1] for p in pasts])
    got = eng.decode_batch_paged(toks, pos, tables, np.array(lens), alloc)

    np.testing.assert_allclose(want, got, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# v2 kernel: the online-softmax algorithm, then the kernel itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,amp", [(1, 1.0), (37, 1.0), (512, 1.0), (200, 25.0)])
def test_online_softmax_matches_dense_attention(n, amp):
    """The recurrence the v2 kernel runs must equal ordinary softmax attention.

    `amp=25` pushes the scores far enough that a naive exp() would overflow —
    the running-max rescale is exactly what stops that, so this case is the
    point of the algorithm, not an edge case.
    """
    from heteroserve.model.paged_attn_v2 import online_softmax_reference

    rng = np.random.default_rng(n)
    H, D = 4, 64
    q = (rng.standard_normal((H, D)) * amp).astype(np.float32)
    k = (rng.standard_normal((n, H, D)) * amp).astype(np.float32)
    v = rng.standard_normal((n, H, D)).astype(np.float32)
    scale = 1.0 / np.sqrt(D)

    got = online_softmax_reference(q, k, v, scale)

    s = np.einsum("hd,nhd->hn", q, k) * scale
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    want = np.einsum("hn,nhd->hd", p, v)

    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


@cuda_only
def test_v2_kernel_matches_reference():
    from heteroserve.model.paged_attn_v2 import build_error, is_available, paged_attention_v2

    assert is_available(), f"v2 kernel did not compile: {build_error()}"
    cfg, alloc, tables, truth, q = _build_paged_case("cuda:0")
    lens = [t[0].shape[2] for t in truth]
    bt = alloc.block_table_tensor(tables, pad_to=max(len(t) for t in tables))
    ctx_lens = torch.tensor(lens, dtype=torch.int32, device="cuda:0")
    scale = 1.0 / np.sqrt(cfg.head_dim)
    qt = torch.as_tensor(q, device="cuda:0")

    got = paged_attention_v2(qt, alloc.pool[0, 0], alloc.pool[0, 1], bt, ctx_lens, scale)
    ref = paged_attention_torch(qt, alloc.pool[0, 0], alloc.pool[0, 1], bt, ctx_lens, scale)
    np.testing.assert_allclose(
        got.cpu().numpy(), ref.cpu().numpy(), rtol=1e-3, atol=1e-3
    )


@cuda_only
def test_v2_matches_v1():
    """Both kernels compute the same function by different routes."""
    from heteroserve.model.paged_attn_v2 import is_available, paged_attention_v2

    assert is_available()
    assert which_backend() == "cuda"
    cfg, alloc, tables, truth, q = _build_paged_case("cuda:0", seed=9)
    lens = [t[0].shape[2] for t in truth]
    bt = alloc.block_table_tensor(tables, pad_to=max(len(t) for t in tables))
    ctx_lens = torch.tensor(lens, dtype=torch.int32, device="cuda:0")
    scale = 1.0 / np.sqrt(cfg.head_dim)
    qt = torch.as_tensor(q, device="cuda:0")

    a = paged_attention(qt, alloc.pool[0, 0], alloc.pool[0, 1], bt, ctx_lens, scale)
    b = paged_attention_v2(qt, alloc.pool[0, 0], alloc.pool[0, 1], bt, ctx_lens, scale)
    np.testing.assert_allclose(a.cpu().numpy(), b.cpu().numpy(), rtol=1e-3, atol=1e-3)
