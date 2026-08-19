"""Integration tests for the distributed path: real processes, real sockets.

These are the claims the project actually rests on:

  1. a prefix-cache hit skips prefill compute and collapses TTFT
  2. migrating KV to another accelerator changes nothing about the output
  3. the link budget really constrains the wire (not just the cost model)
  4. the migrate-vs-recompute decision flips at a bandwidth crossover
  5. different policies genuinely place work differently
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heteroserve.config import (
    ClusterConfig,
    KVConfig,
    LinkConfig,
    ModelConfig,
    WorkerConfig,
)
from heteroserve.sched.router import Placement, Request, Router, WorkerHandle
from heteroserve.worker import protocol as P


def make_cluster(policy: str = "cache_aware", num_blocks: int = 128, **kw) -> ClusterConfig:
    kv = KVConfig(block_size=16, num_blocks=num_blocks)
    return ClusterConfig(
        model=ModelConfig.tiny(),
        workers=[
            WorkerConfig("w0", device="CPU", engine="numpy", kv=kv, max_batch=4),
            WorkerConfig("w1", device="CPU", engine="numpy", kv=kv, max_batch=4),
        ],
        policy=policy,
        **kw,
    )


async def _with_router(cluster, fn):
    router = Router(cluster, weights_dir=None, model_name="tiny")
    await router.start()
    try:
        return await fn(router)
    finally:
        await router.stop()


# ---------------------------------------------------------------------------
# 1. prefix cache
# ---------------------------------------------------------------------------


def test_prefix_hit_skips_prefill_compute():
    async def scenario(router: Router):
        shared = list(range(3000, 3000 + 128))
        cold = await router.submit(
            Request(prompt_ids=shared + [1, 2, 3], max_new_tokens=4)
        )
        warm = await router.submit(
            Request(prompt_ids=shared + [4, 5, 6], max_new_tokens=4)
        )
        return cold, warm

    cold, warm = asyncio.run(_with_router(make_cluster(policy="prefix_affinity"), scenario))

    assert cold.cached_prefix_tokens == 0
    assert warm.cached_prefix_tokens == 128
    # The hit must translate into *skipped compute*, not just a bookkeeping win.
    assert warm.prefill_tokens_computed < cold.prefill_tokens_computed / 4
    assert warm.ttft < cold.ttft


# ---------------------------------------------------------------------------
# 2. migration correctness
# ---------------------------------------------------------------------------


def test_migrated_kv_produces_identical_output():
    """Moving a prefix across the network must be semantically invisible."""

    async def scenario(router: Router):
        shared = list(range(4100, 4100 + 96))
        tail = [7, 7, 7]
        prompt = shared + tail

        # Warm w0's cache, then read the greedy continuation it produces.
        baseline = await router.submit_nowait(
            Request(prompt_ids=prompt, max_new_tokens=8, seed=5), worker_id="w0"
        )
        base_rec = await baseline
        base_out = _output_of(router, base_rec.req_id)

        # w1 is cold: verify that, then migrate the prefix over to it.
        req = Request(prompt_ids=prompt, max_new_tokens=8, seed=5)
        placement = Placement("w1", donor_id="w0", skip_blocks=0)
        ok = await router._migrate(req, placement, router.records.setdefault(
            req.req_id, _blank_record(req)))
        assert ok, "migration did not complete"

        fut = await router.submit_nowait(req, worker_id="w1")
        rec = await fut
        return base_rec, base_out, rec, _output_of(router, rec.req_id)

    base_rec, base_out, rec, out = asyncio.run(_with_router(make_cluster(), scenario))

    assert rec.worker_id == "w1"
    assert rec.cached_prefix_tokens >= 80, "migrated blocks were not adopted into the cache"
    assert out == base_out, "migration changed the generated tokens"


# ---------------------------------------------------------------------------
# 3. the link is real
# ---------------------------------------------------------------------------


def test_bandwidth_actually_constrains_the_wire():
    """A 10x slower link must make the same transfer measurably slower."""

    async def scenario(router: Router):
        shared = list(range(5000, 5000 + 256))     # 16 blocks
        await router.submit_nowait(
            Request(prompt_ids=shared + [1], max_new_tokens=2), worker_id="w0"
        )
        await asyncio.sleep(0.6)

        timings = {}
        for label, mbps in (("fast", 2000.0), ("slow", 100.0)):
            await router.set_link(LinkConfig(bandwidth_mbps=mbps, latency_ms=1.0))
            req = Request(prompt_ids=shared + [1], max_new_tokens=2)
            router.records[req.req_id] = _blank_record(req)
            t0 = time.perf_counter()
            ok = await router._migrate(
                req, Placement("w1", donor_id="w0", skip_blocks=0), router.records[req.req_id]
            )
            timings[label] = time.perf_counter() - t0
            assert ok
        return timings, router.records

    timings, _ = asyncio.run(_with_router(make_cluster(num_blocks=256), scenario))

    # tiny model: 16 blocks * 4 layers * 2 * 16 tok * 256 dim * 2B = 2.1 MB
    # 100 Mbps => ~168 ms of pure serialisation; 2 Gbps => ~8 ms.
    assert timings["slow"] > timings["fast"] * 3, timings
    assert timings["slow"] > 0.10, timings


# ---------------------------------------------------------------------------
# 4. the decision flips at a crossover
# ---------------------------------------------------------------------------


def _fake_router(bandwidth_mbps: float, latency_ms: float = 1.0) -> Router:
    cluster = make_cluster()
    cluster.link = LinkConfig(bandwidth_mbps=bandwidth_mbps, latency_ms=latency_ms)
    r = Router(cluster, weights_dir=None, model_name="tiny")
    for wid in ("w0", "w1"):
        h = WorkerHandle(cfg=cluster.worker(wid), ch=None)  # type: ignore[arg-type]
        h.block_size = 16
        h.block_bytes = 576 * 1024          # GPT-2 @ fp16, block_size 16
        h.t_prefill_per_token = 2.0e-3      # 2 ms/token
        h.t_decode_step = 3.0e-2
        r.workers[wid] = h
    return r


def test_migrate_vs_recompute_flips_with_bandwidth():
    prompt = list(range(9000, 9000 + 512))       # 32 blocks
    hashes = [str(h) for h in __import__(
        "heteroserve.kv.blocks", fromlist=["chain_hashes"]).chain_hashes(prompt, 16)]

    # w0 holds the whole prefix; w1 holds nothing. w0 is heavily backlogged, so
    # the router wants to use w1 — the only question is how to get the KV there.
    def build(bw: float) -> Router:
        r = _fake_router(bw)
        r.workers["w0"].hashes.update(hashes)
        r.workers["w0"].pending_prefill_tokens = 4000     # ~8 s of backlog
        return r

    # 32 blocks * 576 KiB = 18.9 MB. Recomputing 512 tokens on w1 costs ~1.02 s.
    fast = build(10_000.0)      # 10 Gbps -> ~15 ms transfer: migrate
    slow = build(50.0)          # 50 Mbps -> ~3.0 s transfer: recompute

    p_fast = fast.place(prompt)
    p_slow = slow.place(prompt)

    assert p_fast.worker_id == "w1" and p_fast.donor_id == "w0", p_fast
    assert p_slow.worker_id == "w1" and p_slow.donor_id is None, p_slow


def test_migration_disabled_never_migrates():
    prompt = list(range(9000, 9000 + 512))
    hashes = [str(h) for h in __import__(
        "heteroserve.kv.blocks", fromlist=["chain_hashes"]).chain_hashes(prompt, 16)]
    r = _fake_router(10_000.0)
    r.cluster.enable_migration = False
    r.workers["w0"].hashes.update(hashes)
    r.workers["w0"].pending_prefill_tokens = 4000
    assert r.place(prompt).donor_id is None


# ---------------------------------------------------------------------------
# 5. policies differ
# ---------------------------------------------------------------------------


def test_policies_place_differently():
    async def scenario(router: Router):
        shared = list(range(6000, 6000 + 64))
        first = await router.submit(Request(prompt_ids=shared + [0], max_new_tokens=3))
        rest = await asyncio.gather(
            *[
                router.submit(Request(prompt_ids=shared + [i], max_new_tokens=3))
                for i in range(1, 7)
            ]
        )
        return first, rest

    rr_first, rr_rest = asyncio.run(
        _with_router(make_cluster(policy="round_robin"), scenario)
    )
    aff_first, aff_rest = asyncio.run(
        _with_router(make_cluster(policy="prefix_affinity"), scenario)
    )

    rr_workers = {r.worker_id for r in rr_rest}
    aff_workers = {r.worker_id for r in aff_rest}

    assert len(rr_workers) == 2, "round robin should spread across both workers"
    assert len(aff_workers) == 1, "affinity should pin everything to the cache owner"
    # And the cache-blind policy pays for it in wasted prefill.
    rr_cached = sum(r.cached_prefix_tokens for r in rr_rest)
    aff_cached = sum(r.cached_prefix_tokens for r in aff_rest)
    assert aff_cached > rr_cached


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _blank_record(req: Request):
    from heteroserve.metrics import RequestRecord

    return RequestRecord(req_id=req.req_id, prompt_tokens=len(req.prompt_ids),
                         t_submit=time.time())


_outputs: dict[str, list[int]] = {}


def _output_of(router: Router, req_id: str) -> list[int]:
    return _outputs.get(req_id, [])


@pytest.fixture(autouse=True)
def _capture_outputs(monkeypatch):
    """Record FINISHED payloads so tests can compare generated token streams."""
    original = Router._finish

    def patched(self, meta, h):
        _outputs[meta["seq_id"]] = list(meta.get("output_ids", []))
        return original(self, meta, h)

    monkeypatch.setattr(Router, "_finish", patched)
    yield
    _outputs.clear()


# ---------------------------------------------------------------------------
# 6. admission failure must terminate, never hang
# ---------------------------------------------------------------------------


def test_oversized_prompt_fails_fast_instead_of_hanging():
    """No worker can hold this prompt. The caller must get an error, not a wait.

    Regression test: the first version recorded the retry target but never
    re-sent the request, so the future was never resolved and `await submit()`
    blocked forever.
    """

    async def scenario(router: Router):
        # 8 blocks x 16 tokens = 128 token capacity; ask for far more.
        req = Request(prompt_ids=list(range(7000, 7000 + 900)), max_new_tokens=4)
        fut = await router.submit_nowait(req)
        with pytest.raises(RuntimeError, match="KV pool exhausted"):
            await asyncio.wait_for(fut, timeout=20)
        return True

    assert asyncio.run(_with_router(make_cluster(num_blocks=8), scenario))


def test_request_reroutes_when_first_worker_is_full():
    """A request refused by a full worker still completes somewhere else."""

    async def scenario(router: Router):
        # Fill w0 so it cannot admit anything more.
        hog = Request(prompt_ids=list(range(8000, 8000 + 240)), max_new_tokens=24)
        hog_fut = await router.submit_nowait(hog, worker_id="w0")

        small = Request(prompt_ids=list(range(9500, 9500 + 200)), max_new_tokens=4)
        small_fut = await router.submit_nowait(small, worker_id="w0")

        rec = await asyncio.wait_for(small_fut, timeout=30)
        await asyncio.wait_for(hog_fut, timeout=30)
        return rec

    # 20 blocks = 320 tokens: enough for one of these but not both.
    rec = asyncio.run(_with_router(make_cluster(num_blocks=20), scenario))
    assert rec.generated_tokens == 4
