"""How fast is paged attention, and how close is that to the hardware limit?

Decode attention is memory-bandwidth bound: it must read every cached K and V
exactly once and does almost no arithmetic per byte. So the only honest score is
**achieved GB/s against the card's peak**, not a speedup over an arbitrary
baseline. This reports both.

Three paths, same data:

  gather + dense   host materialises each sequence's context, then attends.
                   What every non-CUDA engine in this repo does.
  v1 kernel        naive fused kernel: scores in shared memory, scalar loads,
                   shared-memory tree reduction.
  v2 kernel        online softmax (no score vector at all), one warp per
                   (sequence, head), coalesced per-lane slices, warp shuffles.

Correctness is checked against the torch reference before any timing is printed,
and a path that disagrees is reported as FAILED rather than timed.

    python scripts/bench_kernel.py --batch 16 --context 512
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heteroserve.config import KVConfig, ModelConfig
from heteroserve.model import paged_attn_v2 as v2mod
from heteroserve.model.paged_attn import (
    build_error,
    paged_attention,
    paged_attention_torch,
    which_backend,
)


def _time(fn, iters: int, sync) -> float:
    fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--block-size", dest="block_size", type=int, default=16)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--model", default="gpt2", choices=["gpt2", "tiny"])
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("torch is not installed - nothing to benchmark")
        return 1

    from heteroserve.kv.torch_blocks import TorchBlockAllocator

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    on_cuda = device.startswith("cuda")

    print("=" * 74)
    if on_cuda:
        p = torch.cuda.get_device_properties(0)
        peak = v2mod.peak_bandwidth_gbs()
        print(f"device : {p.name}  ({p.total_memory/1e9:.1f} GB, "
              f"{p.multi_processor_count} SMs, sm_{p.major}{p.minor})")
        print(f"peak BW: {peak:.0f} GB/s" if peak else "peak BW: unknown")
    else:
        print(f"device : {device}  -- NO GPU, so neither CUDA kernel can run.")
        print("         Timings below compare host paths only and are NOT")
        print("         kernel results. Run this on a CUDA box for real numbers.")
    print(f"v1 kernel: {which_backend()}"
          + ("" if which_backend() == "cuda" else f"  ({build_error()})"))
    print(f"v2 kernel: {'ready' if v2mod.is_available() else 'unavailable'}"
          + ("" if v2mod.is_available() else f"  ({v2mod.build_error()})"))
    print("=" * 74)

    cfg = ModelConfig() if args.model == "gpt2" else ModelConfig.tiny()
    blocks_needed = args.batch * (args.context // args.block_size + 4)
    kv = KVConfig(block_size=args.block_size,
                  num_blocks=max(64, blocks_needed), dtype=args.dtype)
    alloc = TorchBlockAllocator(kv, cfg, device=device)

    rng = np.random.default_rng(0)
    tables = []
    for _ in range(args.batch):
        a = alloc.allocate([int(t) for t in rng.integers(1, 40000, size=args.context)])
        k = rng.standard_normal(
            (cfg.n_layer, cfg.n_head, args.context, cfg.head_dim)
        ).astype(np.float32)
        alloc.write_kv(a.block_ids, 0, k, k)
        tables.append(a.block_ids)

    bt = alloc.block_table_tensor(tables, pad_to=max(len(t) for t in tables))
    ctx_lens = torch.full((args.batch,), args.context, dtype=torch.int32, device=device)
    q = torch.randn(args.batch, cfg.n_head, cfg.head_dim, device=device)
    scale = 1.0 / np.sqrt(cfg.head_dim)
    k_pool, v_pool = alloc.pool[0, 0], alloc.pool[0, 1]
    sync = torch.cuda.synchronize if on_cuda else (lambda: None)

    ref = paged_attention_torch(q, k_pool, v_pool, bt, ctx_lens, scale).float()

    def gather_path():
        ks, vs = alloc.gather_kv_batch(tables, args.context, layer=0)
        s = torch.einsum("bhd,bkhd->bhk", q, ks.float()) * scale
        return torch.einsum("bhk,bkhd->bhd", torch.softmax(s, -1), vs.float())

    candidates = [("gather + dense attention", gather_path)]
    if which_backend() == "cuda":
        candidates.append(
            ("v1 kernel (naive fused)",
             lambda: paged_attention(q, k_pool, v_pool, bt, ctx_lens, scale))
        )
    if v2mod.is_available():
        candidates.append(
            ("v2 kernel (online softmax)",
             lambda: v2mod.paged_attention_v2(q, k_pool, v_pool, bt, ctx_lens, scale))
        )
    if not on_cuda:
        candidates.append(
            ("torch paged (not a kernel)",
             lambda: paged_attention_torch(q, k_pool, v_pool, bt, ctx_lens, scale))
        )

    itemsize = 2 if args.dtype == "float16" else 4
    moved = v2mod.bytes_moved(args.batch, args.context, cfg.n_head, cfg.head_dim, itemsize)
    peak = v2mod.peak_bandwidth_gbs() if on_cuda else None

    print(f"\nmodel={cfg.name}  batch={args.batch}  context={args.context}  "
          f"dtype={args.dtype}  one layer")
    print(f"minimum traffic per call: {moved/1e6:.1f} MB (K and V, read once each)\n")

    hdr = f"{'path':30s} {'per call':>10s} {'speedup':>8s} {'GB/s':>8s}"
    if peak:
        hdr += f" {'% peak':>7s}"
    hdr += "   correctness"
    print(hdr)
    print("-" * len(hdr))

    base = None
    for name, fn in candidates:
        got = fn().float()
        err = (got - ref).abs().max().item()
        ok = err < 2e-2
        if not ok:
            print(f"{name:30s} {'--':>10s} {'--':>8s} {'--':>8s}"
                  + (f" {'--':>7s}" if peak else "")
                  + f"   FAILED (max diff {err:.2e})")
            continue

        t = _time(fn, args.iters, sync)
        base = base if base is not None else t
        gbs = moved / t / 1e9
        row = (f"{name:30s} {t*1e6:8.1f}us {base/t:7.2f}x {gbs:7.1f}")
        if peak:
            row += f" {100*gbs/peak:6.1f}%"
        row += f"   ok ({err:.1e})"
        print(row)

    print()
    if on_cuda and v2mod.is_available():
        print("Attention decode is bandwidth bound, so '% peak' is the number that")
        print("matters -- a kernel at 80% of peak has almost nothing left to win.")
    else:
        print("Run this on a CUDA device for the numbers that actually count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
