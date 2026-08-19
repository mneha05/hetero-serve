"""Does the fused paged-attention kernel actually pay for itself?

Compares the two decode paths on identical data:

  gather   host gathers each sequence's blocks into contiguous tensors, then
           runs dense attention  (what every other engine here does)
  fused    the CUDA kernel walks the block table inside the attention loop

Correctness is checked before any timing is reported — a speedup from a kernel
that computes the wrong thing is worth nothing, so this refuses to print one.

Run on a CUDA box:
    python scripts/bench_kernel.py --batch 16 --context 512

On CPU it still runs, comparing gather against the *torch* paged implementation.
That measures the algorithm, not the kernel, and it says so.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heteroserve.config import KVConfig, ModelConfig
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
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--model", default="gpt2", choices=["gpt2", "tiny"])
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("torch is not installed — nothing to benchmark")
        return 1

    from heteroserve.kv.torch_blocks import TorchBlockAllocator

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    backend = which_backend()

    print(f"device        : {device}")
    print(f"kernel backend: {backend}")
    if backend != "cuda":
        print(f"  (fused CUDA kernel not active: {build_error()})")
        print("  reporting the torch paged path instead — this measures the")
        print("  algorithm, NOT the kernel. Numbers here are not a kernel result.")
    print()

    cfg = ModelConfig() if args.model == "gpt2" else ModelConfig.tiny()
    kv = KVConfig(block_size=args.block_size,
                  num_blocks=max(64, args.batch * (args.context // args.block_size + 4)),
                  dtype="float16")
    alloc = TorchBlockAllocator(kv, cfg, device=device)

    rng = np.random.default_rng(0)
    tables = []
    for _ in range(args.batch):
        a = alloc.allocate([int(t) for t in rng.integers(1, 40000, size=args.context)])
        k = rng.standard_normal(
            (cfg.n_layer, cfg.n_head, args.context, cfg.head_dim)
        ).astype(np.float16)
        alloc.write_kv(a.block_ids, 0, k, k)
        tables.append(a.block_ids)

    width = max(len(t) for t in tables)
    bt = alloc.block_table_tensor(tables, pad_to=width)
    ctx_lens = torch.full((args.batch,), args.context, dtype=torch.int32, device=device)
    q = torch.randn(args.batch, cfg.n_head, cfg.head_dim, device=device)
    scale = 1.0 / np.sqrt(cfg.head_dim)

    k_pool, v_pool = alloc.pool[0, 0], alloc.pool[0, 1]
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)

    # -- correctness first --------------------------------------------------
    fused = paged_attention(q, k_pool, v_pool, bt, ctx_lens, scale)
    ref = paged_attention_torch(q, k_pool, v_pool, bt, ctx_lens, scale)
    diff = (fused.float() - ref.float()).abs().max().item()
    print(f"max abs diff vs torch reference: {diff:.2e}")
    if diff > 2e-2:
        print("FAILED correctness check — refusing to report timings")
        return 1
    print("correctness OK\n")

    # -- the two paths ------------------------------------------------------
    def gather_path():
        """What the rest of the system does: materialise, then attend."""
        ks, vs = alloc.gather_kv_batch(tables, args.context, layer=0)
        s = torch.einsum("bhd,bkhd->bhk", q, ks.float()) * scale
        return torch.einsum("bhk,bkhd->bhd", torch.softmax(s, -1), vs.float())

    def fused_path():
        return paged_attention(q, k_pool, v_pool, bt, ctx_lens, scale)

    t_gather = _time(gather_path, args.iters, sync)
    t_fused = _time(fused_path, args.iters, sync)

    kv_mb = (args.batch * args.context * cfg.n_head * cfg.head_dim * 2 * 2) / 1e6
    label = "fused CUDA kernel" if backend == "cuda" else "torch paged (NOT the kernel)"

    print(f"model={cfg.name}  batch={args.batch}  context={args.context}  "
          f"one layer, {kv_mb:.1f} MB of KV")
    print(f"{'path':32s} {'per call':>11s} {'speedup':>9s}")
    print("-" * 55)
    print(f"{'gather + dense attention':32s} {t_gather*1e3:9.3f}ms {'1.00x':>9s}")
    print(f"{label:32s} {t_fused*1e3:9.3f}ms "
          f"{t_gather/max(t_fused,1e-12):8.2f}x")
    print()
    print(f"extrapolated over {cfg.n_layer} layers: "
          f"{t_gather*cfg.n_layer*1e3:.1f}ms -> {t_fused*cfg.n_layer*1e3:.1f}ms per decode step")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
