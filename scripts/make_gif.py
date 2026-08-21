"""Render the README / portfolio hero as an animated GIF.

`docs/hero.svg` is sharper and theme-aware, but a profile README and a portfolio
site both embed from *absolute* URLs, and GitHub serves raw SVG as text/plain --
so the animated SVG that renders fine inside this repo silently breaks
everywhere else. This draws the same story with Pillow: no rasteriser, no
external tooling, deterministic output.

It shows the payoff, not just the mechanism. The block grids carry what the
scheduler is doing; the metric strip underneath carries what it *bought* --
TTFT collapsing on a cache hit, then KV crossing the wire.

    python scripts/make_gif.py            # -> docs/hero.gif
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "docs" / "hero.gif"

W, H = 880, 372
COLS, ROWS = 20, 5
CELL, GAP = 14, 3
GRID_W = COLS * (CELL + GAP) - GAP
PANEL_W = GRID_W + 30
PANEL_H = ROWS * (CELL + GAP) - GAP + 58
LX = 24
RX = W - PANEL_W - 24
PY = 92

FRAMES, MS = 64, 100

BG = (13, 17, 23)
PANEL = (22, 27, 34)
PANEL2 = (28, 33, 40)
EDGE = (48, 54, 61)
FREE = (33, 38, 45)
BLUE = (88, 166, 255)
BLUE_HOT = (150, 200, 255)
GREEN = (63, 185, 80)
VIOLET = (163, 113, 247)
INK = (240, 246, 252)
DIM = (139, 148, 158)
FAINT = (110, 118, 129)


def _f(size, bold=False, mono=True):
    import matplotlib

    base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    if mono:
        n = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    else:
        n = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(base / n), size)


F_TITLE = _f(18, bold=True, mono=False)
F_SUB = _f(10)
F_CAP = _f(13)
F_PANEL = _f(12, bold=True)
F_TINY = _f(9)
F_STAT = _f(19, bold=True)
F_STATL = _f(8)

# (block index, frame it appears, colour, frame it turns green or None)
LEFT = (
    [(i, 5 + i * 0.46, BLUE, 18) for i in range(24)]                   # cold prefill
    + [(i, 14 + (i - 24) * 0.4, BLUE, 34) for i in range(24, 32)]      # req 1 tail
    + [(i, 23 + (i - 32) * 0.45, BLUE, None) for i in range(32, 40)]   # req 2 tail
)
RIGHT = (
    [(i, 47 + i * 0.20, VIOLET, 55) for i in range(24)]                # arrived
    + [(i, 55 + (i - 24) * 0.35, BLUE, None) for i in range(24, 30)]
)

CAPTIONS = [
    (0, 18, "cold request  —  24 blocks of prefill, computed from scratch"),
    (18, 34, "same system prompt  —  the prefix is cached, prefill skipped"),
    (34, 47, "the other accelerator wants it  —  9.4 MB of KV crosses the link"),
    (47, FRAMES, "same tokens, different device  —  migration is invisible"),
]

STATS = [
    ("REQUESTS SERVED", [(0, "0"), (18, "1"), (34, "2"), (58, "3")]),
    ("TTFT", [(0, "—"), (18, "43 ms"), (34, "6 ms"), (58, "8 ms")]),
    ("PREFILL SKIPPED", [(0, "0 tok"), (34, "384 tok"), (58, "528 tok")]),
    ("KV MIGRATED", [(0, "0 MB"), (47, "9.4 MB")]),
]


def colour_of(plan, idx, f):
    for i, start, col, green_at in plan:
        if i != idx:
            continue
        if f < start:
            return FREE
        if green_at is not None and f >= green_at:
            return GREEN
        # brief highlight on the frame a block is actually written
        return BLUE_HOT if (col is BLUE and f < start + 1.6) else col
    return FREE


def panel(d, x, name, sub, plan, f, accent):
    d.rounded_rectangle([x, PY, x + PANEL_W, PY + PANEL_H], 8, fill=PANEL, outline=EDGE)
    d.rounded_rectangle([x, PY + 8, x + 3, PY + PANEL_H - 8], 2, fill=accent)
    d.text((x + 16, PY + 12), name, font=F_PANEL, fill=INK)
    d.text((x + 16, PY + 30), sub, font=F_TINY, fill=DIM)
    gy = PY + 48
    for i in range(COLS * ROWS):
        cx = x + 16 + (i % COLS) * (CELL + GAP)
        cy = gy + (i // COLS) * (CELL + GAP)
        d.rounded_rectangle([cx, cy, cx + CELL, cy + CELL], 2,
                            fill=colour_of(plan, i, f))


def stat_value(pairs, f):
    v = pairs[0][1]
    for at, text in pairs:
        if f >= at:
            v = text
    return v


def frame(f):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    d.text((24, 20), "hetero-serve", font=F_TITLE, fill=INK)
    d.text((24, 46),
           "PAGED KV CACHE  ·  PREFIX SHARING  ·  MIGRATE-OR-RECOMPUTE UNDER A LINK BUDGET",
           font=F_SUB, fill=FAINT)

    for lo, hi, text in CAPTIONS:
        if lo <= f < hi:
            d.text(((W - d.textlength(text, font=F_CAP)) / 2, 68), text,
                   font=F_CAP, fill=INK)
            break

    panel(d, LX, "cuda:0", "TESLA T4  ·  100 BLOCKS  ·  v3 FUSED KERNEL", LEFT, f, BLUE)
    panel(d, RX, "cuda:1", "TESLA T4  ·  100 BLOCKS  ·  v3 FUSED KERNEL", RIGHT, f, VIOLET)

    wy = PY + PANEL_H // 2
    x0, x1 = LX + PANEL_W + 6, RX - 6
    for xx in range(x0, x1, 8):
        d.line([xx, wy, xx + 3, wy], fill=EDGE, width=1)
    if 36 <= f < 47:
        t = (f - 36) / 11
        t = t * t * (3 - 2 * t)                       # ease in / out
        px = x0 + (x1 - x0 - 18) * t
        d.rounded_rectangle([px, wy - 6, px + 18, wy + 6], 3, fill=VIOLET)
        lbl = "9.4 MB"
        d.text((x0 + (x1 - x0 - d.textlength(lbl, font=F_TINY)) / 2, wy - 26),
               lbl, font=F_TINY, fill=VIOLET)

    sy = PY + PANEL_H + 22
    sw = (W - 48 - 3 * 12) // 4
    for i, (label, pairs) in enumerate(STATS):
        sx = 24 + i * (sw + 12)
        d.rounded_rectangle([sx, sy, sx + sw, sy + 58], 7, fill=PANEL2, outline=EDGE)
        d.text((sx + 14, sy + 11), label, font=F_STATL, fill=FAINT)
        changed = any(at <= f < at + 2 for at, _ in pairs if at > 0)
        d.text((sx + 14, sy + 26), stat_value(pairs, f), font=F_STAT,
               fill=GREEN if changed else INK)

    ly = H - 20
    for dx, col, lab in ((0, BLUE, "IN USE"), (96, GREEN, "CACHED · SHAREABLE"),
                         (252, VIOLET, "ARRIVED OVER THE WIRE")):
        d.rounded_rectangle([24 + dx, ly, 32 + dx, ly + 8], 1, fill=col)
        d.text((38 + dx, ly - 1), lab, font=F_TINY, fill=FAINT)
    tail = "55.4% OF A T4'S PEAK BANDWIDTH  ·  5 CUDA KERNELS  ·  101 TESTS"
    d.text((W - 24 - d.textlength(tail, font=F_TINY), ly - 1), tail,
           font=F_TINY, fill=FAINT)
    return img


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames = [frame(f) for f in range(FRAMES)]

    # One shared palette, built from a strip containing every state any frame
    # reaches, so no frame needs a local palette of its own.
    probe = Image.new("RGB", (W, H * 4))
    for i, f in enumerate((10, 30, 42, 60)):
        probe.paste(frames[min(f, FRAMES - 1)], (0, i * H))
    master = probe.quantize(colors=48, method=Image.MEDIANCUT)
    pal = [im.quantize(palette=master, dither=Image.Dither.NONE) for im in frames]

    pal[0].save(OUT, save_all=True, append_images=pal[1:], duration=MS, loop=0,
                optimize=True, disposal=1)
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB, {FRAMES} frames, "
          f"{FRAMES * MS / 1000:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
