"""What can each accelerator on this machine actually run?

Answers three questions the engine design depends on, by compiling real graphs
rather than trusting documentation:

  1. which devices exist
  2. do they accept dynamic shapes (if not, the engine must bucket)
  3. how fast is each one at prefill and decode, on the real model

Run it before believing any benchmark numbers from a different machine.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "weights" / "gpt2"


def main() -> int:
    try:
        import openvino as ov
        import openvino.opset15 as op
    except ImportError:
        print("openvino not installed — only the numpy reference engine is available")
        return 1

    core = ov.Core()
    devices = core.available_devices
    print(f"OpenVINO {ov.__version__}")
    print(f"devices: {devices}\n")

    print(f"{'device':8s} {'name':42s} {'dynamic shapes':>15s} {'>4D tensors':>12s}")
    print("-" * 82)

    support = {}
    for dev in devices:
        try:
            name = core.get_property(dev, "FULL_DEVICE_NAME")
        except Exception:
            name = "?"

        # dynamic rank/dims
        x = op.parameter([-1, -1, 64], ov.Type.f32, name="x")
        W = op.constant(np.random.randn(64, 64).astype(np.float32))
        m = ov.Model([op.result(op.matmul(x, W, False, False))], [x], "dyn")
        try:
            core.compile_model(m, dev)
            dyn = "yes"
        except Exception:
            dyn = "NO"

        # rank > 4
        y = op.parameter([2, 2, 2, 2, 2, 8], ov.Type.f32, name="y")
        m6 = ov.Model(
            [op.result(op.reduce_sum(y, op.constant(np.array([5], np.int32))))], [y], "r6"
        )
        try:
            core.compile_model(m6, dev)
            hi = "yes"
        except Exception:
            hi = "NO"

        support[dev] = (dyn, hi)
        print(f"{dev:8s} {name[:42]:42s} {dyn:>15s} {hi:>12s}")

    print("\nConsequences for the engine:")
    for dev, (dyn, hi) in support.items():
        notes = []
        if dyn == "NO":
            notes.append("must compile static shape buckets")
        if hi == "NO":
            notes.append("past KV must be one 4D tensor per layer, not a single 6D one")
        print(f"  {dev:5s} {'; '.join(notes) if notes else 'no constraints'}")

    if not (WEIGHTS / "model.safetensors").exists():
        print("\n(no GPT-2 weights yet — run `python -m heteroserve.model.fetch` "
              "for the speed table)")
        return 0

    from heteroserve.model.numpy_engine import NumpyEngine
    from heteroserve.model.ov_engine import OpenVINOEngine
    from heteroserve.model.weights import load_gpt2

    print("\nmeasuring GPT-2 124M (128-token prefill, batch-4 decode) ...")
    w, cfg = load_gpt2(WEIGHTS)
    ids = np.arange(100, 228)
    ref = NumpyEngine(w, cfg)
    t = time.perf_counter()
    rl, rk, rv = ref.prefill(ids, 0)
    np_pre = time.perf_counter() - t
    pk, pv = [rk] * 4, [rv] * 4
    toks, pos = np.array([1, 2, 3, 4]), np.array([128] * 4)
    t = time.perf_counter()
    rd, _, _ = ref.decode_batch(toks, pos, pk, pv)
    np_dec = time.perf_counter() - t

    print(f"\n{'engine':18s} {'prefill 128':>12s} {'decode b4':>11s} {'vs numpy':>9s} "
          f"{'argmax ==':>10s}")
    print("-" * 66)
    print(f"{'numpy (reference)':18s} {np_pre*1e3:10.1f}ms {np_dec*1e3:9.1f}ms "
          f"{'1.0x':>9s} {'--':>10s}")

    for dev in devices:
        try:
            e = OpenVINOEngine(w, cfg, device=dev, max_batch=4, bucket=128, max_ctx=256)
            e.warmup()
            e.prefill(ids, 0)
            e.decode_batch(toks, pos, pk, pv)
            t = time.perf_counter()
            ol, _, _ = e.prefill(ids, 0)
            pre = time.perf_counter() - t
            t = time.perf_counter()
            od, _, _ = e.decode_batch(toks, pos, pk, pv)
            dec = time.perf_counter() - t
            agree = int(np.argmax(rl)) == int(np.argmax(ol))
            print(f"{'openvino:' + dev:18s} {pre*1e3:10.1f}ms {dec*1e3:9.1f}ms "
                  f"{np_pre/pre:8.1f}x {str(agree):>10s}")
        except Exception as exc:
            print(f"{'openvino:' + dev:18s} unavailable: {str(exc).splitlines()[-1][:40]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
