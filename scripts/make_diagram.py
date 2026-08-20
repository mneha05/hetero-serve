"""Render the architecture diagram used by the README, profile and portfolio site.

PNG rather than SVG on purpose: this image is embedded from *absolute* URLs on
three different surfaces, and GitHub serves raw SVG as text/plain, so an SVG that
renders fine inside this repo's own README silently breaks everywhere else.
Drawn at 2x and displayed at 1x so it stays crisp on a retina screen.

    python scripts/make_diagram.py        # -> docs/architecture.png
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "docs" / "architecture.png"

S = 2                       # supersample factor
W, H = 860 * S, 574 * S

BG = (13, 17, 23)
PANEL = (22, 27, 34)
PANEL2 = (28, 33, 40)
EDGE = (48, 54, 61)
INK = (230, 237, 243)
DIM = (139, 148, 158)
FAINT = (110, 118, 129)
BLUE = (88, 166, 255)
GREEN = (63, 185, 80)
VIOLET = (163, 113, 247)
AMBER = (210, 153, 34)


def font(size: int, bold: bool = False, mono: bool = True):
    import matplotlib

    base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    if mono:
        name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    else:
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(base / name), size * S)


F_H1 = font(17, bold=True, mono=False)
F_H2 = font(12, bold=True)
F_B = font(10)
F_S = font(8.5)
F_TAG = font(8, bold=True)


def box(d, x, y, w, h, fill=PANEL, outline=EDGE, r=8, width=1):
    d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + h) * S], r * S,
                        fill=fill, outline=outline, width=width * S)


def text(d, x, y, s, f=F_B, fill=INK, center_w=None):
    if center_w:
        tw = d.textlength(s, font=f)
        d.text((x * S + (center_w * S - tw) / 2, y * S), s, font=f, fill=fill)
    else:
        d.text((x * S, y * S), s, font=f, fill=fill)


def arrow(d, x1, y1, x2, y2, colour=EDGE, head=5, w=1, dashed=False):
    if dashed:
        n = max(2, int(abs(y2 - y1) + abs(x2 - x1)) // 7)
        for i in range(n):
            if i % 2:
                continue
            a, b = i / n, min(1.0, (i + 1) / n)
            d.line([(x1 + (x2 - x1) * a) * S, (y1 + (y2 - y1) * a) * S,
                    (x1 + (x2 - x1) * b) * S, (y1 + (y2 - y1) * b) * S],
                   fill=colour, width=w * S)
    else:
        d.line([x1 * S, y1 * S, x2 * S, y2 * S], fill=colour, width=w * S)
    if y2 != y1:                                  # vertical arrowhead
        s = 1 if y2 > y1 else -1
        d.polygon([(x2 * S, y2 * S),
                   ((x2 - head) * S, (y2 - s * head) * S),
                   ((x2 + head) * S, (y2 - s * head) * S)], fill=colour)


def chip(d, x, y, label, colour):
    w = d.textlength(label, font=F_TAG) / S + 12
    d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + 15) * S], 3 * S,
                        fill=PANEL2, outline=colour, width=1 * S)
    d.text((x * S + 6 * S, y * S + 3 * S), label, font=F_TAG, fill=colour)
    return w + 6


def main() -> int:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    text(d, 30, 22, "hetero-serve", F_H1)
    text(d, 30, 48, "a KV cache is just bytes — the question is whether to move them",
         F_B, DIM)

    # ---------------- control plane ----------------
    box(d, 30, 82, 800, 96, fill=PANEL, outline=VIOLET)
    text(d, 46, 94, "ROUTER  ·  CONTROL PLANE", F_H2, VIOLET)
    rows = [
        ("global prefix directory", "block-chain hash → which workers hold it"),
        ("migrate-vs-recompute cost model", "priced in seconds, not guesses"),
        ("device speeds", "measured at startup, per accelerator"),
    ]
    for i, (k, v) in enumerate(rows):
        y = 118 + i * 17
        text(d, 46, y, "• " + k, F_B, INK)
        text(d, 300, y, v, F_S, DIM)

    # ---------------- workers ----------------
    workers = [
        ("cuda:0", "NVIDIA · v3 fused kernel", BLUE),
        ("GPU", "Intel Arc 140V · OpenVINO", GREEN),
        ("NPU", "Intel AI Boost · static shapes", AMBER),
    ]
    wx, ww, gap = 30, 256, 16
    wy, wh = 232, 132
    for i, (name, sub, colour) in enumerate(workers):
        x = wx + i * (ww + gap)
        arrow(d, x + ww / 2, 178, x + ww / 2, wy - 6, EDGE, w=1)
        box(d, x, wy, ww, wh, outline=colour)
        text(d, x + 14, wy + 12, name, F_H2, colour)
        text(d, x + 14, wy + 30, sub, F_S, DIM)
        text(d, x + 14, wy + 50, "paged KV  ·  continuous batching", F_S, FAINT)

        # a little block grid: shared prefix green, in-use blue, rest free
        for b in range(40):
            cx = x + 14 + (b % 20) * 11
            cy = wy + 68 + (b // 20) * 11
            if i == 0:
                c = GREEN if b < 12 else (BLUE if b < 20 else (33, 38, 45))
            elif i == 1:
                c = GREEN if b < 8 else (BLUE if b < 13 else (33, 38, 45))
            else:
                c = BLUE if b < 6 else (33, 38, 45)
            d.rounded_rectangle([cx * S, cy * S, (cx + 8) * S, (cy + 8) * S],
                                1 * S, fill=c)
        text(d, x + 14, wy + 96, "16-token pages · refcount · LRU", F_S, FAINT)

    text(d, 30, 190, "requests", F_S, DIM)

    # ---------------- data plane ----------------
    dy = wy + wh + 26
    box(d, 30, dy, 800, 62, fill=PANEL2, outline=EDGE)
    text(d, 46, dy + 11, "DATA PLANE  ·  KV BLOCKS BETWEEN WORKERS", F_H2, INK)
    x = 46
    x += chip(d, x, dy + 33, "shaped TCP — bandwidth is a knob", BLUE)
    x += chip(d, x, dy + 33, "NCCL / gloo — device to device", GREEN)
    chip(d, x, dy + 33, "token bucket · latency · contention", VIOLET)

    for i in range(2):
        x0 = wx + i * (ww + gap) + ww
        arrow(d, x0 + 2, wy + wh / 2, x0 + gap - 2, wy + wh / 2, EDGE, head=0,
              w=1, dashed=True)

    # ---------------- kernels ----------------
    ky = dy + 78
    text(d, 30, ky, "CUDA KERNELS", F_H2, INK)
    ks = [
        ("v1", "naive fused", "14.4%"),
        ("v2", "online softmax", "31.0%"),
        ("v3", "context split", "55.4%"),
        ("prefill", "S queries, causal", "—"),
        ("wmma", "tensor cores", "—"),
    ]
    bx, bw = 30, 156
    for i, (name, what, peak) in enumerate(ks):
        x = bx + i * (bw + 8)
        hot = name == "v3"
        box(d, x, ky + 20, bw, 44, fill=PANEL, outline=VIOLET if hot else EDGE)
        text(d, x + 10, ky + 28, name, F_H2, VIOLET if hot else INK)
        text(d, x + 10, ky + 45, what, F_S, DIM)
        if peak != "—":
            tw = d.textlength(peak, font=F_S)
            d.text(((x + bw - 10) * S - tw, (ky + 45) * S), peak, font=F_S,
                   fill=GREEN if hot else FAINT)

    text(d, 30, H / S - 24, "% of a Tesla T4's 320 GB/s peak memory bandwidth, measured",
         F_S, FAINT)

    img = img.resize((W // S, H // S), Image.LANCZOS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, optimize=True)
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB, {img.size[0]}x{img.size[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
