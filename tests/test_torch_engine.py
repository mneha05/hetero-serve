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
    from heteroserve.model.paged_attn import build_error

    assert which_backend() == "cuda", f"v1 kernel did not compile: {build_error()}"
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


# ---------------------------------------------------------------------------
# v3: context-split attention, and fuzzing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,splits,amp", [
    (37, 1, 1.0), (37, 4, 1.0), (512, 8, 1.0), (512, 32, 1.0),
    (5, 8, 1.0),        # more splits than tokens -> empty slices
    (200, 7, 25.0),     # scores large enough that a naive exp() overflows
    (1, 4, 1.0),        # single token
])
def test_split_merge_matches_dense_attention(n, splits, amp):
    """Splitting the context and merging the softmax states must be exact.

    This is the associativity claim v3 rests on: independent online-softmax
    runs over slices, recombined by rescaling against a global max, equal one
    pass over the whole context.
    """
    from heteroserve.model.paged_attn_v3 import split_merge_reference

    rng = np.random.default_rng(n * 31 + splits)
    H, D = 4, 64
    q = (rng.standard_normal((H, D)) * amp).astype(np.float32)
    k = (rng.standard_normal((n, H, D)) * amp).astype(np.float32)
    v = rng.standard_normal((n, H, D)).astype(np.float32)
    scale = 1.0 / np.sqrt(D)

    got = split_merge_reference(q, k, v, scale, splits)

    s = np.einsum("hd,nhd->hn", q, k) * scale
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    want = np.einsum("hn,nhd->hd", p, v)

    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_paged_attention_fuzz_cpu():
    """Fuzz the block-table walk across random geometries.

    Randomised sequence counts, lengths, block sizes and (crucially) *shuffled,
    non-contiguous* block tables, since a prefix-shared cache never lays a
    sequence out in consecutive blocks.
    """
    rng = np.random.default_rng(1234)
    for trial in range(40):
        n_seqs = int(rng.integers(1, 6))
        block_size = int(rng.choice([4, 8, 16]))
        lens = [int(rng.integers(1, 200)) for _ in range(n_seqs)]

        cfg = ModelConfig.tiny()
        need = sum(-(-n // block_size) for n in lens) + 8
        kv = KVConfig(block_size=block_size, num_blocks=max(16, need), dtype="float32")
        alloc = TorchBlockAllocator(kv, cfg, device="cpu")

        tables, truth = [], []
        for n in lens:
            a = alloc.allocate([int(t) for t in rng.integers(1, 4000, size=n)])
            L, H, D = cfg.n_layer, cfg.n_head, cfg.head_dim
            k = rng.standard_normal((L, H, n, D)).astype(np.float32)
            v = rng.standard_normal((L, H, n, D)).astype(np.float32)
            alloc.write_kv(a.block_ids, 0, k, v)
            tables.append(a.block_ids)
            truth.append((k, v))

        q = rng.standard_normal((n_seqs, cfg.n_head, cfg.head_dim)).astype(np.float32)
        bt = alloc.block_table_tensor(tables, pad_to=max(len(t) for t in tables))
        scale = 1.0 / np.sqrt(cfg.head_dim)
        got = paged_attention_torch(
            torch.as_tensor(q), alloc.pool[0, 0], alloc.pool[0, 1],
            bt, torch.tensor(lens, dtype=torch.int32), scale,
        ).numpy()

        for b, (k, v) in enumerate(truth):
            want = _reference_attention(
                q[b], k[0].transpose(1, 0, 2), v[0].transpose(1, 0, 2), scale
            )
            np.testing.assert_allclose(
                got[b], want, rtol=1e-4, atol=1e-4,
                err_msg=f"trial {trial}, seq {b}, len {lens[b]}, block {block_size}",
            )


@cuda_only
def test_v3_kernel_matches_reference():
    from heteroserve.model.paged_attn_v3 import build_error, is_available, paged_attention_v3

    assert is_available(), f"v3 kernel did not compile: {build_error()}"
    cfg, alloc, tables, truth, q = _build_paged_case("cuda:0")
    lens = [t[0].shape[2] for t in truth]
    bt = alloc.block_table_tensor(tables, pad_to=max(len(t) for t in tables))
    ctx_lens = torch.tensor(lens, dtype=torch.int32, device="cuda:0")
    scale = 1.0 / np.sqrt(cfg.head_dim)
    qt = torch.as_tensor(q, device="cuda:0")
    ref = paged_attention_torch(
        qt, alloc.pool[0, 0], alloc.pool[0, 1], bt, ctx_lens, scale
    ).cpu().numpy()

    # Every split count must give the same answer, including more splits than
    # tokens (empty slices) and the auto-chosen value.
    for splits in (0, 1, 2, 4, 8, 16, 64):
        got = paged_attention_v3(
            qt, alloc.pool[0, 0], alloc.pool[0, 1], bt, ctx_lens, scale, splits
        )
        np.testing.assert_allclose(
            got.cpu().numpy(), ref, rtol=1e-3, atol=1e-3,
            err_msg=f"num_splits={splits}",
        )


@cuda_only
def test_all_three_kernels_agree_under_fuzz():
    """v1, v2 and v3 must agree with each other and the reference, always."""
    from heteroserve.model.paged_attn_v2 import paged_attention_v2
    from heteroserve.model.paged_attn_v3 import paged_attention_v3

    rng = np.random.default_rng(7)
    for trial in range(15):
        n_seqs = int(rng.integers(1, 9))
        block_size = int(rng.choice([8, 16, 32]))
        lens = [int(rng.integers(1, 700)) for _ in range(n_seqs)]

        cfg = ModelConfig.tiny()
        need = sum(-(-n // block_size) for n in lens) + 8
        kv = KVConfig(block_size=block_size, num_blocks=max(32, need), dtype="float16")
        alloc = TorchBlockAllocator(kv, cfg, device="cuda:0")

        tables = []
        for n in lens:
            a = alloc.allocate([int(t) for t in rng.integers(1, 4000, size=n)])
            L, H, D = cfg.n_layer, cfg.n_head, cfg.head_dim
            k = rng.standard_normal((L, H, n, D)).astype(np.float16)
            alloc.write_kv(a.block_ids, 0, k, k)
            tables.append(a.block_ids)

        q = torch.randn(n_seqs, cfg.n_head, cfg.head_dim, device="cuda:0")
        bt = alloc.block_table_tensor(tables, pad_to=max(len(t) for t in tables))
        ctx = torch.tensor(lens, dtype=torch.int32, device="cuda:0")
        scale = 1.0 / np.sqrt(cfg.head_dim)
        kp, vp = alloc.pool[0, 0], alloc.pool[0, 1]

        ref = paged_attention_torch(q, kp, vp, bt, ctx, scale).float()
        for name, got in (
            ("v1", paged_attention(q, kp, vp, bt, ctx, scale)),
            ("v2", paged_attention_v2(q, kp, vp, bt, ctx, scale)),
            ("v3", paged_attention_v3(q, kp, vp, bt, ctx, scale, 0)),
        ):
            err = (got.float() - ref).abs().max().item()
            assert err < 5e-3, f"{name} trial {trial} lens={lens} bs={block_size} err={err:.2e}"


# ---------------------------------------------------------------------------
# prefill: many query tokens against the paged cache, causally masked
# ---------------------------------------------------------------------------


def _dense_causal(q, k, v, scale, q_start):
    """q [S,H,D] starting at absolute position q_start; k/v [N,Hkv,D]."""
    S, H, D = q.shape
    N, HKV, _ = k.shape
    if H != HKV:
        g = H // HKV
        k = np.repeat(k, g, axis=1)
        v = np.repeat(v, g, axis=1)
    out = np.zeros((S, H, D), np.float32)
    for s in range(S):
        end = q_start + s + 1
        sc = np.einsum("hd,nhd->hn", q[s], k[:end]) * scale
        p = np.exp(sc - sc.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        out[s] = np.einsum("hn,nhd->hd", p, v[:end])
    return out


def _prefill_case(lens, S, n_kv_head, device="cpu", seed=0):
    cfg = ModelConfig.tiny(n_kv_head=n_kv_head)
    H, D, HKV = cfg.n_head, cfg.head_dim, cfg.kv_heads
    kv = KVConfig(block_size=16, num_blocks=96, dtype="float32")
    alloc = TorchBlockAllocator(kv, cfg, device=device)
    rng = np.random.default_rng(seed)

    tables, truth = [], []
    for n in lens:
        a = alloc.allocate([int(t) for t in rng.integers(1, 4000, size=n)])
        k = rng.standard_normal((cfg.n_layer, HKV, n, D)).astype(np.float32)
        v = rng.standard_normal((cfg.n_layer, HKV, n, D)).astype(np.float32)
        alloc.write_kv(a.block_ids, 0, k, v)
        tables.append(a.block_ids)
        truth.append((k[0], v[0]))

    q = rng.standard_normal((len(lens), S, H, D)).astype(np.float32)
    bt = alloc.block_table_tensor(tables, pad_to=max(len(t) for t in tables))
    ctx = torch.tensor(lens, dtype=torch.int32, device=device)
    return cfg, alloc, bt, ctx, q, truth


@pytest.mark.parametrize("lens,S,kvh", [
    ([64], 64, None),          # fresh chunk, nothing cached
    ([64], 16, None),          # chunked: 48 already cached, 16 new
    ([37, 64, 19], 16, None),  # ragged batch
    ([64], 32, 2),             # GQA 2:1
    ([48], 48, 1),             # MQA
    ([80], 1, None),           # degenerates to the decode case
])
def test_prefill_paged_matches_dense_causal(lens, S, kvh):
    """Causal paged prefill must equal dense causal attention over the real context.

    The mask is implicit: query s of a chunk sits at `ctx_len - S + s`, so each
    query row attends to a different number of keys. Getting that off by one in
    either direction still produces plausible-looking output, which is why this
    is asserted against a dense reference rather than eyeballed.
    """
    from heteroserve.model.paged_attn_prefill import paged_attention_prefill_torch

    cfg, alloc, bt, ctx, q, truth = _prefill_case(lens, S, kvh, seed=len(lens) * 7 + S)
    scale = 1.0 / np.sqrt(cfg.head_dim)
    got = paged_attention_prefill_torch(
        torch.as_tensor(q), alloc.pool[0, 0], alloc.pool[0, 1], bt, ctx, scale
    ).numpy()

    for b, (k, v) in enumerate(truth):
        want = _dense_causal(q[b], k.transpose(1, 0, 2), v.transpose(1, 0, 2),
                             scale, lens[b] - S)
        np.testing.assert_allclose(got[b], want, rtol=1e-4, atol=1e-4,
                                   err_msg=f"sequence {b}, len {lens[b]}, S={S}")


@cuda_only
def test_prefill_kernel_matches_reference():
    from heteroserve.model.paged_attn_prefill import (
        build_error, is_available, paged_attention_prefill,
        paged_attention_prefill_torch,
    )

    assert is_available(), f"prefill kernel did not compile: {build_error()}"
    for lens, S, kvh in [([64], 64, None), ([37, 64, 19], 16, None),
                         ([64], 32, 2), ([48], 48, 1)]:
        cfg, alloc, bt, ctx, q, _ = _prefill_case(lens, S, kvh, device="cuda:0", seed=S)
        scale = 1.0 / np.sqrt(cfg.head_dim)
        qt = torch.as_tensor(q, device="cuda:0")
        got = paged_attention_prefill(qt, alloc.pool[0, 0], alloc.pool[0, 1], bt, ctx, scale)
        ref = paged_attention_prefill_torch(qt, alloc.pool[0, 0], alloc.pool[0, 1],
                                            bt, ctx, scale)
        np.testing.assert_allclose(got.cpu().numpy(), ref.cpu().numpy(),
                                   rtol=1e-3, atol=1e-3,
                                   err_msg=f"lens={lens} S={S} kv={kvh}")


@cuda_only
def test_prefill_kernel_with_one_query_equals_the_decode_kernel():
    """S=1 is the decode case, so the two kernels must agree exactly there.

    A useful cross-check: the prefill and decode kernels were written separately
    and share no code, so agreement is real evidence rather than a tautology.
    """
    from heteroserve.model.paged_attn_prefill import paged_attention_prefill
    from heteroserve.model.paged_attn_v2 import paged_attention_v2

    lens = [31, 64, 17]
    cfg, alloc, bt, ctx, q, _ = _prefill_case(lens, 1, None, device="cuda:0", seed=3)
    scale = 1.0 / np.sqrt(cfg.head_dim)
    qt = torch.as_tensor(q, device="cuda:0")                     # [B, 1, H, D]

    pre = paged_attention_prefill(qt, alloc.pool[0, 0], alloc.pool[0, 1], bt, ctx, scale)
    dec = paged_attention_v2(qt[:, 0], alloc.pool[0, 0], alloc.pool[0, 1], bt, ctx, scale)

    np.testing.assert_allclose(pre[:, 0].cpu().numpy(), dec.cpu().numpy(),
                               rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# tensor-core prefill
# ---------------------------------------------------------------------------


def _dense_causal_1h(q, k, v, scale, ctx_len):
    """Single-head dense causal attention, queries being the trailing rows."""
    S, D = q.shape
    out = np.zeros((S, D), np.float64)
    for s in range(S):
        end = ctx_len - S + s + 1
        sc = (q[s] @ k[:end].T) * scale
        p = np.exp(sc - sc.max())
        p /= p.sum()
        out[s] = p @ v[:end]
    return out.astype(np.float32)


@pytest.mark.parametrize("ctx,S,D", [
    (64, 32, 64),     # everything aligned
    (64, 19, 64),     # query count not a multiple of the tile
    (57, 16, 64),     # context not a multiple of the tile
    (91, 23, 64),     # neither
    (48, 48, 64),     # full prefill, S == ctx
    (80, 1, 64),      # single query row
    (100, 40, 128),   # head_dim 128 -> 8 fragments per GEMM
    (512, 16, 64),    # long context, small chunk
])
def test_wmma_tiling_matches_dense_causal(ctx, S, D):
    """The kernel's tile loop, mirrored in numpy, against dense attention.

    This is the half of the WMMA kernel that can be checked without a GPU: the
    tiling, the per-row causal bound, and the online rescale across tiles. Only
    the MMA instruction itself is left to hardware. The misaligned cases matter
    most -- a tile loop that is right when everything divides by 16 and wrong
    when it does not is the classic way this goes bad.
    """
    from heteroserve.model.paged_attn_wmma import tiled_prefill_reference

    rng = np.random.default_rng(ctx * 31 + S)
    q = rng.standard_normal((S, D))
    k = rng.standard_normal((ctx, D))
    v = rng.standard_normal((ctx, D))
    scale = 1.0 / np.sqrt(D)

    got = tiled_prefill_reference(q, k, v, scale, ctx)
    want = _dense_causal_1h(q, k, v, scale, ctx)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_wmma_declines_geometry_it_cannot_serve():
    """`supports()` must refuse a cache the kernel cannot read.

    A 32-token page is not a WMMA fragment and an fp32 cache is not an MMA
    operand. Getting this wrong would mean silently wrong output rather than a
    clean fall back to the scalar kernel.
    """
    from heteroserve.model.paged_attn_wmma import supports

    assert supports(32, 64, torch.float16) is False, "block_size must be 16"
    assert supports(16, 64, torch.float32) is False, "MMA needs fp16 operands"
    assert supports(16, 48, torch.float16) is False, "head_dim must tile by 16"


@cuda_only
def test_wmma_kernel_matches_the_scalar_prefill_kernel():
    """Tensor-core and scalar prefill must compute the same function.

    Tolerance is looser than elsewhere because MMA takes fp16 operands: Q and P
    are rounded before each GEMM. The accumulation stays fp32, so this is
    rounding, not drift.
    """
    from heteroserve.model.paged_attn_prefill import paged_attention_prefill_torch
    from heteroserve.model.paged_attn_wmma import (
        build_error, is_available, paged_attention_prefill_wmma, supports,
    )

    assert is_available(), f"WMMA kernel did not compile: {build_error()}"

    for lens, S, kvh in [([64], 64, None), ([37, 64, 19], 16, None), ([64], 32, 2)]:
        cfg = ModelConfig.tiny(n_kv_head=kvh)
        kv = KVConfig(block_size=16, num_blocks=96, dtype="float16")
        alloc = TorchBlockAllocator(kv, cfg, device="cuda:0")
        rng = np.random.default_rng(S + len(lens))

        tables = []
        for n in lens:
            a = alloc.allocate([int(t) for t in rng.integers(1, 4000, size=n)])
            k = rng.standard_normal(
                (cfg.n_layer, cfg.kv_heads, n, cfg.head_dim)).astype(np.float16)
            alloc.write_kv(a.block_ids, 0, k, k)
            tables.append(a.block_ids)

        assert supports(alloc.block_size, cfg.head_dim, alloc.torch_dtype)

        q = torch.randn(len(lens), S, cfg.n_head, cfg.head_dim, device="cuda:0")
        bt = alloc.block_table_tensor(tables, pad_to=max(len(t) for t in tables))
        ctx = torch.tensor(lens, dtype=torch.int32, device="cuda:0")
        scale = 1.0 / np.sqrt(cfg.head_dim)

        got = paged_attention_prefill_wmma(
            q, alloc.pool[0, 0], alloc.pool[0, 1], bt, ctx, scale)
        ref = paged_attention_prefill_torch(
            q, alloc.pool[0, 0], alloc.pool[0, 1], bt, ctx, scale)

        np.testing.assert_allclose(
            got.cpu().numpy(), ref.cpu().numpy(), rtol=2e-2, atol=2e-2,
            err_msg=f"lens={lens} S={S} kv={kvh}")
