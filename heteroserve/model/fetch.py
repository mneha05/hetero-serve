"""Fetch GPT-2 weights + BPE tokenizer straight from HuggingFace.

Deliberately dependency-free: safetensors is a trivial container (8-byte
little-endian header length, then a JSON header, then a raw tensor blob), so
numpy is all we need to read it. No torch, no transformers, no safetensors pkg.
"""

from __future__ import annotations

import json
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = "openai-community/gpt2"
BASE = f"https://huggingface.co/{REPO}/resolve/main"

FILES = {
    "model.safetensors": f"{BASE}/model.safetensors",
    "vocab.json": f"{BASE}/vocab.json",
    "merges.txt": f"{BASE}/merges.txt",
    "config.json": f"{BASE}/config.json",
}


def default_cache_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "weights" / "gpt2"


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "hetero-serve/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100.0 * done / total
                    print(
                        f"\r  {dest.name}: {done/1e6:7.1f}/{total/1e6:7.1f} MB ({pct:5.1f}%)",
                        end="",
                        flush=True,
                    )
    print()
    tmp.replace(dest)


def ensure_weights(cache_dir: Path | None = None) -> Path:
    """Download anything missing. Returns the directory holding the files."""
    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    for name, url in FILES.items():
        dest = cache_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"[fetch] {name} <- {url}")
        _download(url, dest)

    return cache_dir


# ---------------------------------------------------------------------------
# safetensors reader
# ---------------------------------------------------------------------------

_ST_DTYPES = {
    "F64": "<f8",
    "F32": "<f4",
    "F16": "<f2",
    "BF16": "bf16",  # handled specially
    "I64": "<i8",
    "I32": "<i4",
    "I16": "<i2",
    "I8": "<i1",
    "U8": "<u1",
    "BOOL": "|b1",
}


def load_safetensors(path: Path) -> dict:
    """Read a .safetensors file into {name: np.ndarray} using only numpy."""
    import numpy as np

    path = Path(path)
    with open(path, "rb") as fh:
        header_len = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(header_len).decode("utf-8"))
        data_start = 8 + header_len
        blob = np.memmap(path, dtype=np.uint8, mode="r", offset=data_start)

    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype_tag = meta["dtype"]
        begin, end = meta["data_offsets"]
        raw = blob[begin:end]

        if dtype_tag == "BF16":
            # bfloat16 == top 16 bits of a float32; widen by left-shifting.
            u16 = raw.view(np.uint16).astype(np.uint32) << 16
            arr = u16.view(np.float32)
        else:
            np_dtype = _ST_DTYPES.get(dtype_tag)
            if np_dtype is None:
                raise ValueError(f"unsupported safetensors dtype {dtype_tag}")
            arr = raw.view(np.dtype(np_dtype))

        out[name] = np.ascontiguousarray(arr.reshape(meta["shape"]))

    return out


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    d = ensure_weights(target)
    total = sum(f.stat().st_size for f in d.glob("*") if f.is_file())
    print(f"[fetch] ready: {d}  ({total/1e6:.1f} MB)")

    free = shutil.disk_usage(d).free
    print(f"[fetch] free disk: {free/1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
