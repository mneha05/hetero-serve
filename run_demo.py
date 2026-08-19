"""One-command demo: see the whole system do its thing in about a minute.

    python run_demo.py

Walks through the four behaviours the project exists to show:

  1. a cold request  — full prefill, nothing cached
  2. a warm request  — same prefix, prefill skipped, TTFT collapses
  3. a slow link     — the scheduler declines to migrate and recomputes instead
  4. a fast link     — the same decision flips, and KV crosses the wire

Everything printed is measured, including the generated text.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from heteroserve.config import (
    ClusterConfig,
    KVConfig,
    LinkConfig,
    ModelConfig,
    WorkerConfig,
)
from heteroserve.metrics import RequestRecord
from heteroserve.sched.router import Placement, Request, Router

REPO = Path(__file__).resolve().parent
WEIGHTS = REPO / "weights" / "gpt2"

SYSTEM = (
    "You are a careful assistant. Answer using only the provided context, cite the "
    "section you used, and say plainly when the context does not contain the answer. "
    "Never invent numbers, names, or dates. Keep answers short and concrete. "
)
SHARED = SYSTEM * 3        # a chunky shared prefix, like a real system prompt


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "-" * 74)


def detect(explicit: str) -> list[str]:
    if explicit:
        return explicit.split(",")
    try:
        import openvino as ov

        avail = ov.Core().available_devices
        return [d for d in ("GPU", "NPU", "CPU") if d in avail] or ["CPU"]
    except Exception:
        return ["CPU", "CPU"]


async def main_async(args) -> int:
    if args.model == "gpt2" and not (WEIGHTS / "model.safetensors").exists():
        print("downloading GPT-2 weights (~550 MB, one time) ...")
        from heteroserve.model.fetch import ensure_weights

        ensure_weights(WEIGHTS)

    devices = detect(args.devices)
    if len(devices) < 2:
        devices = devices * 2          # need two workers to show migration

    use_real = args.model == "gpt2" and WEIGHTS.exists()
    tok = None
    if use_real:
        from heteroserve.model.tokenizer import GPT2Tokenizer

        tok = GPT2Tokenizer.from_dir(WEIGHTS)

    kv = KVConfig(block_size=16, num_blocks=args.num_blocks)
    cluster = ClusterConfig(
        model=ModelConfig() if use_real else ModelConfig.tiny(),
        workers=[
            WorkerConfig(f"w{i}-{d.lower()}", device=d, kv=kv, max_batch=8,
                         max_prefill_tokens=256)
            for i, d in enumerate(devices)
        ],
        policy="cache_aware",
        link=LinkConfig(bandwidth_mbps=10_000, latency_ms=1.0),
    )

    rule(f"booting cluster on {' + '.join(devices)}")
    print("(first run compiles OpenVINO graphs; they are cached on disk afterwards)")
    t0 = time.time()
    router = Router(cluster, weights_dir=WEIGHTS if use_real else None,
                    model_name=args.model, max_ctx=512)
    await router.start()
    print(f"up in {time.time()-t0:.1f}s")
    for h in router.workers.values():
        print(f"  {h.worker_id:10s} {h.engine:16s} "
              f"KV pool {h.num_blocks} blocks x {h.block_bytes/1024:.0f} KiB "
              f"= {h.num_blocks*h.block_bytes/1e6:.0f} MB")

    await router.calibrate(prompt_len=128, gen=4)
    rule("measured device speeds (the cost model uses these, not guesses)")
    for h in router.workers.values():
        print(f"  {h.worker_id:10s} prefill {h.t_prefill_per_token*1e3:6.2f} ms/token   "
              f"decode {h.t_decode_step*1e3:6.1f} ms/step")

    def encode(text: str) -> list[int]:
        return tok.encode(text) if tok else [
            (abs(hash(text[: i + 1])) % 40000) + 1 for i in range(min(len(text), 400))
        ]

    def show(label: str, rec, extra: str = "") -> None:
        text = tok.decode(rec.output_ids) if tok else "(synthetic model)"
        print(f"  {label:22s} worker={rec.worker_id:10s} "
              f"cached={rec.cached_prefix_tokens:4d}/{rec.prompt_tokens:<4d} "
              f"computed={rec.prefill_tokens_computed:4d}  "
              f"TTFT={rec.ttft*1000:7.1f}ms  E2E={rec.e2e*1000:7.1f}ms {extra}")
        print(f"  {'':22s} -> {text.strip()[:90]!r}")
        print(f"  {'':22s}    {rec.placement_reason}")

    # ---- 1 & 2: the prefix cache -----------------------------------------
    rule("1) cold request, then the same prefix again")
    p1 = encode(SHARED + " Question: where is the Eiffel Tower located? Answer: the city of")
    cold = await router.submit(Request(prompt_ids=p1, max_new_tokens=12))
    show("cold (nothing cached)", cold)

    p2 = encode(SHARED + " Question: what is the largest planet? Answer: the planet")
    warm = await router.submit(Request(prompt_ids=p2, max_new_tokens=12))
    show("warm (shared prefix)", warm)

    if warm.ttft > 0 and cold.ttft > 0:
        print(f"\n  prefill skipped: {warm.cached_prefix_tokens} tokens   "
              f"TTFT {cold.ttft*1000:.0f}ms -> {warm.ttft*1000:.0f}ms "
              f"({cold.ttft/max(warm.ttft,1e-6):.1f}x faster)")

    # ---- 3 & 4: the link decides -----------------------------------------
    rule("2) does the scheduler move the KV cache? the link budget decides")
    owner = warm.worker_id
    other = next(w for w in router.workers if w != owner)
    hashes = router._block_hashes(p2)
    hits = router._hits(hashes)
    blocks = hits[owner] - hits[other]
    mb = blocks * router.workers[other].block_bytes / 1e6
    print(f"  {owner} holds {hits[owner]} prefix blocks; {other} holds {hits[other]}.")
    print(f"  Moving the difference would put {mb:.1f} MB on the wire.\n")

    for bw in (args.slow, args.fast):
        await router.set_link(LinkConfig(bandwidth_mbps=bw, latency_ms=2.0))
        # Load the cache owner so the scheduler genuinely wants the other device.
        router.workers[owner].pending_prefill_tokens = 6000
        plan: Placement = router.place(p2)
        router.workers[owner].pending_prefill_tokens = 0
        verdict = (f"MIGRATE {mb:.1f} MB from {plan.donor_id}"
                   if plan.donor_id else "RECOMPUTE locally (transfer too expensive)")
        est_transfer = cluster.link.transfer_seconds(blocks * router.workers[other].block_bytes)
        print(f"  {bw:>6.0f} Mbps  ->  target={plan.worker_id:10s} {verdict}")
        print(f"  {'':6s}       (transfer would cost ~{est_transfer*1000:.0f} ms; "
              f"recomputing {hits[owner]*16} tokens costs "
              f"~{hits[owner]*16*router.workers[other].t_prefill_per_token*1000:.0f} ms)")

    # ---- 5: prove a migration is semantically invisible -------------------
    rule("3) migrate the KV for real, and check the output is identical")
    await router.set_link(LinkConfig(bandwidth_mbps=args.fast, latency_ms=2.0))
    req = Request(prompt_ids=p2, max_new_tokens=12, seed=3)
    rec_stub = router.records.setdefault(
        req.req_id,
        RequestRecord(req_id=req.req_id, prompt_tokens=len(p2), t_submit=time.time()),
    )
    ok = await router._migrate(req, Placement(other, donor_id=owner, skip_blocks=hits[other]),
                               rec_stub)
    print(f"  migration {'succeeded' if ok else 'FAILED'}: "
          f"{rec_stub.migration_bytes/1e6:.1f} MB in {rec_stub.migration_s*1000:.0f} ms "
          f"({rec_stub.migration_bytes/1e6/max(rec_stub.migration_s,1e-9):.0f} MB/s effective)")
    moved = await (await router.submit_nowait(req, worker_id=other))
    show("after migration", moved)

    baseline = tok.decode(warm.output_ids) if tok else ""
    after = tok.decode(moved.output_ids) if tok else ""
    same = warm.output_ids == moved.output_ids
    print(f"\n  same tokens as before the move: {same}")
    if not same and tok:
        print(f"    before: {baseline.strip()[:70]!r}\n    after : {after.strip()[:70]!r}")

    rule("done")
    print("Next:")
    print("  live dashboard :  python -m heteroserve.dashboard.server")
    print("  full benchmark :  python -m heteroserve.bench.sweep --repeats 3")
    print("  charts         :  python -m heteroserve.bench.plot")

    await router.stop()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="", help="comma list (default: auto-detect)")
    ap.add_argument("--model", default="gpt2", choices=["gpt2", "tiny"])
    ap.add_argument("--num-blocks", dest="num_blocks", type=int, default=384)
    ap.add_argument("--slow", type=float, default=50.0, help="slow link, Mbps")
    ap.add_argument("--fast", type=float, default=10000.0, help="fast link, Mbps")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
