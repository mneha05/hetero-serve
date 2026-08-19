"""Correctness gates for the model runtime and the paged KV cache.

The interesting one is `test_paged_matches_contiguous`: it proves that running a
sequence through the block-paged cache produces bit-comparable logits to running
it in one contiguous shot. Everything the scheduler does — sharing prefixes,
migrating blocks across the network, preempting and resuming — rests on that
equivalence holding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heteroserve.config import KVConfig, ModelConfig
from heteroserve.kv.blocks import BlockAllocator, chain_hashes
from heteroserve.model.numpy_engine import NumpyEngine
from heteroserve.model.weights import synthetic_gpt2

WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "gpt2"
HAS_REAL_WEIGHTS = (WEIGHTS / "model.safetensors").exists()


@pytest.fixture(scope="module")
def tiny():
    cfg = ModelConfig.tiny()
    return NumpyEngine(synthetic_gpt2(cfg, seed=7), cfg)


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------


def test_block_roundtrip():
    cfg = ModelConfig.tiny()
    kv = KVConfig(block_size=8, num_blocks=32)
    alloc = BlockAllocator(kv, cfg)

    tokens = list(range(20))
    a = alloc.allocate(tokens)
    assert a.num_cached_tokens == 0
    assert len(a.block_ids) == 3  # ceil(20/8)

    L, H, D = cfg.n_layer, cfg.n_head, cfg.head_dim
    k = np.random.default_rng(0).standard_normal((L, H, 20, D)).astype(np.float16)
    v = np.random.default_rng(1).standard_normal((L, H, 20, D)).astype(np.float16)
    alloc.write_kv(a.block_ids, 0, k, v)

    gk, gv = alloc.gather_kv(a.block_ids, 20)
    assert np.array_equal(gk, k)
    assert np.array_equal(gv, v)


def test_prefix_sharing_and_refcounts():
    cfg = ModelConfig.tiny()
    alloc = BlockAllocator(KVConfig(block_size=4, num_blocks=64), cfg)

    shared = list(range(100, 112))          # 12 tokens = 3 full blocks
    a = alloc.allocate(shared)
    alloc.register_full_blocks(shared, a.block_ids)

    used_after_first = alloc.num_used

    b = alloc.allocate(shared + [7, 7, 7, 7])
    assert b.num_cached_tokens == 12
    assert b.block_ids[:3] == a.block_ids[:3]
    # Only the one new block should have been consumed.
    assert alloc.num_used == used_after_first + 1

    alloc.free_sequence(a.block_ids)
    alloc.free_sequence(b.block_ids)
    assert alloc.num_free == alloc.num_blocks


def test_eviction_reclaims_unreferenced_blocks():
    cfg = ModelConfig.tiny()
    alloc = BlockAllocator(KVConfig(block_size=4, num_blocks=8), cfg)

    first = list(range(32))                 # exactly fills the pool
    a = alloc.allocate(first)
    alloc.register_full_blocks(first, a.block_ids)
    alloc.free_sequence(a.block_ids)
    assert alloc.num_free == 8

    other = list(range(500, 532))
    b = alloc.allocate(other)               # must evict to succeed
    assert len(b.block_ids) == 8
    assert alloc.stat_evictions > 0


def test_chain_hash_is_prefix_sensitive():
    a = chain_hashes([1, 2, 3, 4, 9, 9, 9, 9], 4)
    b = chain_hashes([1, 2, 3, 4, 8, 8, 8, 8], 4)
    assert a[0] == b[0]      # same first block
    assert a[1] != b[1]      # diverges after
    c = chain_hashes([5, 5, 5, 5, 9, 9, 9, 9], 4)
    assert a[1] != c[1]      # same 2nd block content, different history


def test_migration_export_import_is_lossless():
    cfg = ModelConfig.tiny()
    src = BlockAllocator(KVConfig(block_size=4, num_blocks=16), cfg)
    dst = BlockAllocator(KVConfig(block_size=4, num_blocks=16), cfg)

    tokens = list(range(16))
    a = src.allocate(tokens)
    L, H, D = cfg.n_layer, cfg.n_head, cfg.head_dim
    k = np.random.default_rng(2).standard_normal((L, H, 16, D)).astype(np.float16)
    v = np.random.default_rng(3).standard_normal((L, H, 16, D)).astype(np.float16)
    src.write_kv(a.block_ids, 0, k, v)

    payload = src.export_blocks(a.block_ids)
    new_ids = dst.import_blocks(payload)
    gk, gv = dst.gather_kv(new_ids, 16)

    assert np.array_equal(gk, k)
    assert np.array_equal(gv, v)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def test_paged_matches_contiguous(tiny):
    """Prefill-then-decode through paged blocks == one-shot contiguous prefill."""
    cfg = tiny.cfg
    alloc = BlockAllocator(KVConfig(block_size=8, num_blocks=64, dtype="float32"), cfg)

    tokens = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 222, 333]
    ref_logits, _, _ = tiny.prefill(np.array(tokens), 0)

    # Same sequence, but split across two chunks with the KV living in blocks.
    a = alloc.allocate(tokens)
    split = 8
    _, k1, v1 = tiny.prefill(np.array(tokens[:split]), 0)
    alloc.write_kv(a.block_ids, 0, k1, v1)

    pk, pv = alloc.gather_kv(a.block_ids, split)
    chunked_logits, k2, v2 = tiny.prefill(np.array(tokens[split:]), split, pk, pv)
    alloc.write_kv(a.block_ids, split, k2, v2)

    np.testing.assert_allclose(ref_logits, chunked_logits, rtol=1e-4, atol=1e-4)


def test_decode_batch_matches_single(tiny):
    """Batched decode must equal decoding each sequence on its own."""
    cfg = tiny.cfg
    rng = np.random.default_rng(5)
    L, H, D = cfg.n_layer, cfg.n_head, cfg.head_dim

    lengths = [3, 11, 7]
    pasts_k = [rng.standard_normal((L, H, n, D)).astype(np.float32) for n in lengths]
    pasts_v = [rng.standard_normal((L, H, n, D)).astype(np.float32) for n in lengths]
    toks = np.array([5, 900, 42])
    pos = np.array(lengths)

    batched, _, _ = tiny.decode_batch(toks, pos, pasts_k, pasts_v)

    for i in range(len(lengths)):
        one, _, _ = tiny.decode_batch(
            toks[i : i + 1], pos[i : i + 1], [pasts_k[i]], [pasts_v[i]]
        )
        np.testing.assert_allclose(batched[i], one[0], rtol=1e-4, atol=1e-4)


def test_prefix_cache_hit_changes_nothing(tiny):
    """A cache hit must be a pure speed optimisation, never a semantic one."""
    cfg = tiny.cfg
    alloc = BlockAllocator(KVConfig(block_size=4, num_blocks=64, dtype="float32"), cfg)

    shared = [101, 202, 303, 404, 505, 606, 707, 808]
    full = shared + [1, 2, 3]

    cold, k, v = tiny.prefill(np.array(full), 0)

    a = alloc.allocate(shared)
    _, ks, vs = tiny.prefill(np.array(shared), 0)
    alloc.write_kv(a.block_ids, 0, ks, vs)
    alloc.register_full_blocks(shared, a.block_ids)

    b = alloc.allocate(full)
    assert b.num_cached_tokens == 8
    pk, pv = alloc.gather_kv(b.block_ids, 8)
    warm, _, _ = tiny.prefill(np.array(full[8:]), 8, pk, pv)

    np.testing.assert_allclose(cold, warm, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not HAS_REAL_WEIGHTS, reason="GPT-2 weights not downloaded")
def test_real_gpt2_predicts_unambiguous_continuations():
    """Sanity-check real weights on prompts GPT-2 124M is actually good at.

    (Deliberately not "The capital of France is" — GPT-2 small ranks ' the'
    above ' Paris' there. That is the model being small, not the engine being
    wrong, so it makes a terrible regression gate.)
    """
    from heteroserve.model.tokenizer import GPT2Tokenizer
    from heteroserve.model.weights import load_gpt2

    w, cfg = load_gpt2(WEIGHTS)
    eng = NumpyEngine(w, cfg)
    tok = GPT2Tokenizer.from_dir(WEIGHTS)

    cases = [
        ("Barack Obama was the president of the United", " States"),
        ("1, 2, 3, 4, 5, 6, 7,", " 8"),
    ]
    for prompt, expected in cases:
        logits, _, _ = eng.prefill(np.array(tok.encode(prompt)), 0)
        assert tok.decode([int(np.argmax(logits))]) == expected, prompt


@pytest.mark.skipif(not HAS_REAL_WEIGHTS, reason="GPT-2 weights not downloaded")
def test_real_gpt2_decode_loop_matches_full_prefill():
    """Incremental decode against the KV cache == re-prefilling the whole thing.

    This is the property that makes caching sound: if it ever broke, every
    generated token after the first would silently drift.
    """
    from heteroserve.model.tokenizer import GPT2Tokenizer
    from heteroserve.model.weights import load_gpt2

    w, cfg = load_gpt2(WEIGHTS)
    eng = NumpyEngine(w, cfg)
    tok = GPT2Tokenizer.from_dir(WEIGHTS)

    ids = tok.encode("The Eiffel Tower is located in the city of")
    logits, k, v = eng.prefill(np.array(ids), 0)

    generated = []
    pk, pv = k, v
    for _ in range(8):
        nxt = int(np.argmax(logits))
        generated.append(nxt)
        lg, nk, nv = eng.decode_batch(
            np.array([nxt]), np.array([pk.shape[2]]), [pk], [pv]
        )
        logits = lg[0]
        pk = np.concatenate([pk, nk[0]], axis=2)
        pv = np.concatenate([pv, nv[0]], axis=2)

    # Same tokens, recomputed from scratch in a single prefill.
    ref, _, _ = eng.prefill(np.array(ids + generated[:-1]), 0)
    assert int(np.argmax(ref)) == generated[-1]
    assert len(tok.decode(generated)) > 0


# ---------------------------------------------------------------------------
# grouped-query attention
# ---------------------------------------------------------------------------


def _gqa_to_mha_weights(gqa_cfg, gqa_w):
    """Build MHA weights that must compute exactly what the GQA ones do.

    Duplicating each shared KV head across its query group turns a GQA model
    into an algebraically identical MHA model. If the two disagree, the GQA
    plumbing is wrong somewhere.
    """
    import copy

    from heteroserve.config import ModelConfig

    H, HKV, D, E = gqa_cfg.n_head, gqa_cfg.kv_heads, gqa_cfg.head_dim, gqa_cfg.n_embd
    g = H // HKV
    mha_cfg = ModelConfig(name="mha", n_layer=gqa_cfg.n_layer, n_head=H,
                          n_kv_head=None, n_embd=E, vocab_size=gqa_cfg.vocab_size,
                          n_ctx=gqa_cfg.n_ctx)
    w = copy.deepcopy(gqa_w)
    w.cfg = mha_cfg
    for l in w.layers:
        q = l.attn_w[:, :E]
        k = l.attn_w[:, E:E + HKV * D].reshape(E, HKV, D)
        v = l.attn_w[:, E + HKV * D:].reshape(E, HKV, D)
        l.attn_w = np.concatenate([
            q,
            np.repeat(k, g, axis=1).reshape(E, H * D),
            np.repeat(v, g, axis=1).reshape(E, H * D),
        ], axis=1)
        qb = l.attn_b[:E]
        kb = l.attn_b[E:E + HKV * D].reshape(HKV, D)
        vb = l.attn_b[E + HKV * D:].reshape(HKV, D)
        l.attn_b = np.concatenate([
            qb,
            np.repeat(kb, g, axis=0).reshape(H * D),
            np.repeat(vb, g, axis=0).reshape(H * D),
        ])
    return mha_cfg, w


@pytest.mark.parametrize("n_kv_head", [1, 2])
def test_gqa_equals_mha_with_duplicated_kv_heads(n_kv_head):
    from heteroserve.config import ModelConfig
    from heteroserve.model.weights import synthetic_gpt2

    gqa_cfg = ModelConfig.tiny(n_kv_head=n_kv_head)
    gqa_w = synthetic_gpt2(gqa_cfg, seed=21)
    mha_cfg, mha_w = _gqa_to_mha_weights(gqa_cfg, gqa_w)

    gqa, mha = NumpyEngine(gqa_w, gqa_cfg), NumpyEngine(mha_w, mha_cfg)
    toks = np.array([4, 19, 200, 7, 88, 3, 41, 900, 12, 6])

    lg_g, k_g, v_g = gqa.prefill(toks, 0)
    lg_m, k_m, v_m = mha.prefill(toks, 0)
    np.testing.assert_allclose(lg_g, lg_m, rtol=1e-4, atol=1e-4)

    # the cache itself is g times smaller -- that is the entire point
    assert k_g.shape[1] == n_kv_head
    assert k_m.shape[1] == gqa_cfg.n_head

    d_g, _, _ = gqa.decode_batch(np.array([13]), np.array([10]), [k_g], [v_g])
    d_m, _, _ = mha.decode_batch(np.array([13]), np.array([10]), [k_m], [v_m])
    np.testing.assert_allclose(d_g, d_m, rtol=1e-4, atol=1e-4)


def test_gqa_shrinks_the_kv_cache_proportionally():
    """KV bytes scale exactly with the KV-head count, which is what moves the
    migrate-vs-recompute crossover."""
    from heteroserve.config import KVConfig, ModelConfig

    full = ModelConfig.tiny()
    half = ModelConfig.tiny(n_kv_head=2)
    quarter = ModelConfig.tiny(n_kv_head=1)
    fp16 = np.dtype("float16")

    assert full.kv_bytes_per_token(fp16) == 2 * half.kv_bytes_per_token(fp16)
    assert full.kv_bytes_per_token(fp16) == 4 * quarter.kv_bytes_per_token(fp16)

    kv = KVConfig(block_size=16, num_blocks=64)
    assert kv.pool_bytes(half) * 2 == kv.pool_bytes(full)

    # Llama-3-8B geometry: 32 query heads, 8 KV heads -> 4x less cache to move
    llama = ModelConfig.llama3_8b_shape()
    assert llama.kv_group == 4
    mha_equiv = ModelConfig(name="x", n_layer=32, n_head=32, n_embd=4096)
    assert mha_equiv.kv_bytes_per_token(fp16) == 4 * llama.kv_bytes_per_token(fp16)
