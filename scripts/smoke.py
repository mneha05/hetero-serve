"""Minimal end-to-end check: spin up a 2-worker cluster, serve shared-prefix
requests, and confirm the second one hits the prefix cache.

Uses the tiny synthetic model so it runs in a couple of seconds.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heteroserve.config import KVConfig, LinkConfig, ModelConfig, ClusterConfig, WorkerConfig
from heteroserve.sched.router import Request, Router


async def main() -> int:
    kv = KVConfig(block_size=16, num_blocks=128)
    cluster = ClusterConfig(
        model=ModelConfig.tiny(),
        workers=[
            WorkerConfig("w0-cpu", device="CPU", engine="numpy", kv=kv, max_batch=4),
            WorkerConfig("w1-cpu", device="CPU", engine="numpy", kv=kv, max_batch=4),
        ],
        link=LinkConfig(bandwidth_mbps=200, latency_ms=2.0),
        policy="cache_aware",
    )

    router = Router(cluster, weights_dir=None, model_name="tiny", verbose=True)
    await router.start()
    print("-- cluster up --")

    shared = list(range(2000, 2000 + 96))       # 6 blocks of shared context
    r1 = await router.submit(Request(prompt_ids=shared + [11, 12, 13], max_new_tokens=6))
    print(f"r1 worker={r1.worker_id} cached={r1.cached_prefix_tokens} "
          f"ttft={r1.ttft:.3f}s reason={r1.placement_reason}")

    r2 = await router.submit(Request(prompt_ids=shared + [21, 22, 23], max_new_tokens=6))
    print(f"r2 worker={r2.worker_id} cached={r2.cached_prefix_tokens} "
          f"ttft={r2.ttft:.3f}s reason={r2.placement_reason}")

    # Force the other worker to want the prefix: pin nothing, just fire a burst.
    burst = await asyncio.gather(
        *[
            router.submit(Request(prompt_ids=shared + [30 + i], max_new_tokens=5))
            for i in range(6)
        ]
    )
    for r in burst:
        print(f"  burst {r.req_id[:6]} w={r.worker_id} cached={r.cached_prefix_tokens} "
              f"mig={r.migrated} e2e={r.e2e:.3f}s :: {r.placement_reason}")

    await router.poll_states()
    await asyncio.sleep(0.3)
    snap = router.snapshot()
    for w in snap["workers"]:
        st = w.get("state") or {}
        print(f"  {w['worker_id']:9s} {w['engine']:12s} kv_util={w['kv_util']:.2f} "
              f"cached_blocks={w['cached_blocks']:3d} gen={st.get('tokens_generated')} "
              f"saved={st.get('tokens_saved_by_cache')} mig_in={st.get('migrations_in')}")
    print(f"migrations={snap['migrations']} bytes={snap['migration_bytes']}")

    await router.stop()

    ok = r2.cached_prefix_tokens >= 80 and all(r.generated_tokens > 0 for r in burst)
    print("SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
