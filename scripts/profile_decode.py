"""Where does a decode step actually spend its time?

The sweep reported ~200 ms per decode step on all three accelerators, which is
suspicious: they are not equally fast at anything else. This splits one decode
step into its three parts to find out what is really dominating.

  gather   pull each sequence's KV out of the block pool into contiguous tensors
  engine   the actual forward pass on the accelerator
  write    scatter the new K/V back into the pool

`gather` is host-side numpy. A production paged-attention kernel reads the block
table directly and never materialises the contiguous copy, so any time spent
here is overhead this implementation pays and a fused kernel would not.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heteroserve.config import KVConfig, ModelConfig
from heteroserve.kv.blocks import BlockAllocator
from heteroserve.model.weights import load_gpt2, synthetic_gpt2

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "weights" / "gpt2"


def build_engine(device: str, weights, cfg, batch: int):
    if device == "numpy":
        from heteroserve.model.numpy_engine import NumpyEngine

        return NumpyEngine(weights, cfg)
    from heteroserve.model.ov_engine import OpenVINOEngine

    e = OpenVINOEngine(weights, cfg, device=device, max_batch=batch,
                       bucket=256, max_ctx=512)
    e.warmup()
    return e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="GPU,NPU,CPU,numpy")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--context", type=int, default=288)
    ap.add_argument("--iters", type=int, default=8)
    args = ap.parse_args()

    if WEIGHTS.exists():
        weights, cfg = load_gpt2(WEIGHTS)
        model = "gpt2-124M"
    else:
        cfg = ModelConfig.tiny()
        weights = synthetic_gpt2(cfg)
        model = "tiny (no weights downloaded)"

    kv = KVConfig(block_size=16, num_blocks=512)
    alloc = BlockAllocator(kv, cfg)

    # Lay out `batch` sequences of `context` tokens in the paged pool.
    rng = np.random.default_rng(0)
    tables = []
    for b in range(args.batch):
        toks = rng.integers(1, 40000, size=args.context).tolist()
        # +1 block of headroom: the decode step writes at position `context`.
        a = alloc.allocate([int(t) for t in toks], max_new_tokens=16)
        k = rng.standard_normal(
            (cfg.n_layer, cfg.n_head, args.context, cfg.head_dim)
        ).astype(np.float16)
        alloc.write_kv(a.block_ids, 0, k, k)
        tables.append(a.block_ids)

    toks = np.arange(args.batch) + 5
    pos = np.array([args.context] * args.batch)

    kv_mb = (
        args.batch * cfg.kv_bytes_per_token(np.dtype("float16")) * args.context / 1e6
    )
    print(f"model={model}  batch={args.batch}  context={args.context}")
    print(f"KV touched per decode step: {kv_mb:.1f} MB\n")
    print(f"{'device':10s} {'gather':>10s} {'engine':>10s} {'write':>9s} "
          f"{'total':>9s} {'gather%':>8s}")
    print("-" * 62)

    for dev in args.devices.split(","):
        try:
            eng = build_engine(dev, weights, cfg, args.batch)
        except Exception as exc:
            print(f"{dev:10s} unavailable: {str(exc).splitlines()[-1][:38]}")
            continue

        # warm up
        pk = [alloc.gather_kv(t, args.context) for t in tables]
        eng.decode_batch(toks, pos, [p[0] for p in pk], [p[1] for p in pk])

        tg = te = tw = 0.0
        for _ in range(args.iters):
            t0 = time.perf_counter()
            gathered = [alloc.gather_kv(t, args.context) for t in tables]
            t1 = time.perf_counter()
            _, k_new, v_new = eng.decode_batch(
                toks, pos, [g[0] for g in gathered], [g[1] for g in gathered]
            )
            t2 = time.perf_counter()
            for i, tbl in enumerate(tables):
                alloc.write_kv(tbl, args.context, k_new[i], v_new[i])
            t3 = time.perf_counter()
            tg += t1 - t0
            te += t2 - t1
            tw += t3 - t2

        n = args.iters
        total = (tg + te + tw) / n
        print(f"{dev:10s} {tg/n*1e3:9.1f}ms {te/n*1e3:9.1f}ms {tw/n*1e3:8.1f}ms "
              f"{total*1e3:8.1f}ms {100*tg/(tg+te+tw):7.1f}%")

    print("\ngather is host-side numpy paging overhead, not accelerator time:")
    print("a fused paged-attention kernel would read the block table directly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
