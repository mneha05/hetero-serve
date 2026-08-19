"""Benchmark harness: sweep the link budget and the routing policy, measure the
throughput/latency tradeoff, write CSV + JSON.

The experiment this is built to answer:

    Given a shared prefix already resident on one accelerator, and a second
    accelerator that is idle, when is it faster to *move the KV cache* than to
    *recompute the prefix*? And what does getting that decision wrong cost you
    at p99?

Every point in the sweep is a real run against real processes over real sockets.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from heteroserve.config import (
    ClusterConfig,
    KVConfig,
    LinkConfig,
    ModelConfig,
    WorkerConfig,
)
from heteroserve.metrics import RequestRecord, RunSummary, summarise
from heteroserve.sched.router import Request, Router
from heteroserve.bench.workload import WorkloadSpec, describe, generate

REPO = Path(__file__).resolve().parents[2]
WEIGHTS = REPO / "weights" / "gpt2"


def detect_devices() -> list[str]:
    # NVIDIA first when present: it is the interesting hardware, and the fused
    # paged-attention kernel only exists on that path.
    try:
        import torch

        if torch.cuda.is_available():
            return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    try:
        import openvino as ov

        avail = ov.Core().available_devices
        return [d for d in ("GPU", "NPU", "CPU") if d in avail] or ["CPU"]
    except Exception:
        return ["CPU"]


async def run_workload(router: Router, reqs) -> tuple[list[RequestRecord], float]:
    """Fire requests at their Poisson arrival times and wait for all of them."""
    t0 = time.perf_counter()
    futs = []
    for i, r in enumerate(reqs):
        delay = r.arrival_offset - (time.perf_counter() - t0)
        if delay > 0:
            await asyncio.sleep(delay)
        futs.append(
            await router.submit_nowait(
                Request(
                    prompt_ids=r.prompt_ids,
                    max_new_tokens=r.max_new_tokens,
                    temperature=0.0,
                    seed=i,
                )
            )
        )

    done = await asyncio.gather(*futs, return_exceptions=True)
    wall = time.perf_counter() - t0
    return [d for d in done if isinstance(d, RequestRecord)], wall


def build_cluster(args, devices: list[str]) -> ClusterConfig:
    kv = KVConfig(block_size=args.block_size, num_blocks=args.num_blocks)
    model = ModelConfig() if args.model == "gpt2" else ModelConfig.tiny()
    workers = [
        WorkerConfig(
            worker_id=f"w{i}-{d.lower()}",
            device=d,
            engine=args.engine,
            kv=kv,
            max_batch=args.max_batch,
            max_prefill_tokens=args.max_prefill,
        )
        for i, d in enumerate(devices)
    ]
    return ClusterConfig(model=model, workers=workers, policy="cache_aware")


async def main_async(args) -> int:
    devices = args.devices.split(",") if args.devices else detect_devices()
    cluster = build_cluster(args, devices)

    tokenizer = None
    if args.model == "gpt2" and WEIGHTS.exists():
        from heteroserve.model.tokenizer import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_dir(WEIGHTS)

    spec = WorkloadSpec(
        n_requests=args.requests,
        n_prefixes=args.prefixes,
        prefix_tokens=args.prefix_tokens,
        suffix_tokens=args.suffix_tokens,
        max_new_tokens=args.gen,
        arrival_rate=args.rate,
        zipf_s=args.zipf,
        seed=args.seed,
    )
    reqs = generate(spec, tokenizer)
    meta = describe(spec, reqs)

    print("=" * 96)
    print(f"hetero-serve sweep   model={args.model}   devices={devices}")
    print(f"workload: {meta}")
    print("=" * 96)

    router = Router(
        cluster,
        weights_dir=WEIGHTS if (args.model == "gpt2" and WEIGHTS.exists()) else None,
        model_name=args.model,
        verbose=args.verbose,
        max_ctx=args.max_ctx,
    )
    await router.start()

    print("calibrating device speeds ...")
    await router.calibrate(prompt_len=args.prefix_tokens // 2, gen=4)
    devinfo = {
        h.worker_id: {
            "device": h.device,
            "engine": h.engine,
            "prefill_ms_per_token": round(h.t_prefill_per_token * 1e3, 3),
            "decode_ms_per_step": round(h.t_decode_step * 1e3, 2),
        }
        for h in router.workers.values()
    }
    for wid, d in devinfo.items():
        print(f"  {wid:12s} {d['engine']:16s} prefill {d['prefill_ms_per_token']:7.3f} ms/tok"
              f"   decode {d['decode_ms_per_step']:7.2f} ms/step")

    policies = args.policies.split(",")
    bandwidths = [float(b) for b in args.bandwidths.split(",")]

    header = (
        f"\n{'policy':16s} {'BW(Mbps)':>9s} {'tok/s':>8s} {'TTFT p50':>9s} {'TTFT p99':>9s} "
        f"{'E2E p50':>8s} {'E2E p95':>8s} {'hit%':>6s} {'migr':>5s} {'MB moved':>9s} {'+/-':>6s}"
    )
    print(header)
    print("-" * len(header))

    summaries: list[RunSummary] = []
    all_records: list[tuple[str, float, RequestRecord]] = []

    # One throwaway pass before measuring anything. The very first run on a cold
    # machine pays OpenVINO graph compilation and page-faults the weights in;
    # letting that land inside a measured run is how the first version of this
    # harness reported a 28% throughput swing for a policy whose code had not
    # changed.
    if args.warmup:
        router.cluster.policy = "cache_aware"
        await router.set_link(LinkConfig(bandwidth_mbps=bandwidths[-1],
                                         latency_ms=args.latency))
        await router.reset()
        await run_workload(router, reqs[: max(8, len(reqs) // 3)])
        await router.reset()
        print("warmup pass done\n")

    # Only migrating policies react to the link at all — the others never put a
    # KV block on the wire, so sweeping bandwidth for them would print the same
    # row four times. Run those once, at the first bandwidth.
    MIGRATING = {"cache_aware"}

    for policy in policies:
        for bw in (bandwidths if policy in MIGRATING else bandwidths[:1]):
            router.cluster.policy = policy
            await router.set_link(LinkConfig(bandwidth_mbps=bw, latency_ms=args.latency,
                                             jitter_ms=args.jitter))
            await router.reset()
            await asyncio.sleep(0.2)

            # Repeat each configuration and pool the records: with 48 requests
            # a "p99" is just the single worst sample. Pooling repeats gives the
            # tail enough data to mean something.
            pooled: list[RequestRecord] = []
            walls, tputs = 0.0, []
            for _ in range(args.repeats):
                await router.reset()
                await asyncio.sleep(0.15)
                recs, wall = await run_workload(router, reqs)
                pooled.extend(recs)
                walls += wall
                gen = sum(r.generated_tokens for r in recs)
                tputs.append(gen / wall if wall else 0.0)

            s = summarise(pooled, walls, label=f"{policy}@{bw:g}", policy=policy,
                          bandwidth_mbps=bw, latency_ms=args.latency,
                          repeat_throughputs=tputs)
            summaries.append(s)
            all_records.extend((policy, bw, r) for r in pooled)

            print(
                f"{policy:16s} {bw:9.0f} {s.throughput_tok_s:8.1f} {s.ttft_p50:9.3f} "
                f"{s.ttft_p99:9.3f} {s.e2e_p50:8.3f} {s.e2e_p95:8.3f} "
                f"{100*s.cache_hit_rate:6.1f} {s.migrations:5d} "
                f"{s.migration_bytes/1e6:9.1f} {s.throughput_std:6.1f}"
            )

    await router.poll_states()
    await asyncio.sleep(0.3)
    snap = router.snapshot()
    await router.stop()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    summary_csv = out / f"sweep-{stamp}.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summaries[0].as_dict().keys()))
        w.writeheader()
        for s in summaries:
            w.writerow(s.as_dict())

    req_csv = out / f"requests-{stamp}.csv"
    with open(req_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["policy", "bandwidth_mbps",
                                           *RequestRecord("x").as_dict().keys()])
        w.writeheader()
        for policy, bw, r in all_records:
            w.writerow({"policy": policy, "bandwidth_mbps": bw, **r.as_dict()})

    blob = {
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "devices": devinfo,
        "workload": meta,
        "spec": asdict(spec),
        "cluster": {
            "block_size": args.block_size,
            "num_blocks": args.num_blocks,
            "kv_bytes_per_token": cluster.model.kv_bytes_per_token("float16"),
            "block_bytes": KVConfig(args.block_size, args.num_blocks).block_bytes(cluster.model),
            "max_batch": args.max_batch,
        },
        "results": [{**s.as_dict(), "per_worker": s.per_worker} for s in summaries],
        "final_snapshot": snap,
    }
    (out / f"sweep-{stamp}.json").write_text(json.dumps(blob, indent=2), encoding="utf-8")

    print(f"\nwrote {summary_csv}")
    print(f"wrote {req_csv}")
    print(f"wrote {out / f'sweep-{stamp}.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="hetero-serve benchmark sweep")
    ap.add_argument("--devices", default="", help="comma list, e.g. GPU,NPU,CPU (default: auto)")
    ap.add_argument("--engine", default="auto", choices=["auto", "numpy", "openvino"])
    ap.add_argument("--model", default="gpt2", choices=["gpt2", "tiny"])
    ap.add_argument("--policies", default="round_robin,least_loaded,prefix_affinity,cache_aware")
    ap.add_argument("--bandwidths", default="50,200,1000,10000", help="Mbps, comma list")
    ap.add_argument("--latency", type=float, default=2.0, help="one-way link latency, ms")
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--requests", type=int, default=48)
    ap.add_argument("--prefixes", type=int, default=6)
    ap.add_argument("--prefix-tokens", dest="prefix_tokens", type=int, default=256)
    ap.add_argument("--suffix-tokens", dest="suffix_tokens", type=int, default=24)
    ap.add_argument("--gen", type=int, default=24, help="max new tokens per request")
    ap.add_argument("--rate", type=float, default=8.0, help="arrivals/sec")
    ap.add_argument("--zipf", type=float, default=1.1)
    ap.add_argument("--block-size", dest="block_size", type=int, default=16)
    ap.add_argument("--num-blocks", dest="num_blocks", type=int, default=384)
    ap.add_argument("--max-batch", dest="max_batch", type=int, default=8)
    ap.add_argument("--max-prefill", dest="max_prefill", type=int, default=256)
    ap.add_argument("--max-ctx", dest="max_ctx", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=3,
                    help="runs per configuration; records are pooled for percentiles")
    ap.add_argument("--no-warmup", dest="warmup", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "results"))
    ap.add_argument("--verbose", action="store_true")
    return ap


def main() -> int:
    return asyncio.run(main_async(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
