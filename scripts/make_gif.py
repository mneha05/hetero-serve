"""Render the README hero as a real animated GIF.

`docs/hero.svg` is sharper and theme-aware, but a *profile* README and a portfolio
site both need something that renders as an image from an absolute URL, and
GitHub serves raw SVG as text/plain. So this draws the same beats directly with
Pillow -- no SVG rasteriser, no external tooling, deterministic output.

    python scripts/make_gif.py            # -> docs/hero.gif

The palette is GitHub's own dark ground, so the card sits naturally in a README
whichever theme the reader is using.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "docs" / "hero.gif"

W, H = 820, 292
COLS, ROWS = 20, 5
CELL, GAP = 13, 3
GRID_W = COLS * (CELL + GAP) - GAP
PANEL_W = GRID_W + 28
PANEL_H = ROWS * (CELL + GAP) - GAP + 56
LX, RX, PY = 22, W - (GRID_W + 28) - 22, 86

FRAMES = 52
MS = 110

BG = (13, 17, 23)
PANEL = (22, 27, 34)
EDGE = (48, 54, 61)
FREE = (33, 38, 45)
BLUE = (88, 166, 255)
GREEN = (63, 185, 80)
VIOLET = (163, 113, 247)
INK = (230, 237, 243)
DIM = (139, 148, 158)
ACCENT = (139, 92, 246)


def _font(size: int, bold: bool = False):
    """Pillow's built-in bitmap font is unusable at this size; borrow the DejaVu
    faces that ship with matplotlib rather than depending on system fonts."""
    try:
        import matplotlib

        base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
        return ImageFont.truetype(str(base / name), size)
    except Exception:
        return ImageFont.load_default()


F_TITLE = _font(15, bold=True)
F_SUB = _font(10)
F_CAP = _font(12)
F_LBL = _font(11, bold=True)
F_TINY = _font(9)

# Which blocks light up when, per side. (first_frame, colour_after, green_at)
LEFT_PLAN = (
    [(i, 4 + i * 0.5, BLUE, 17) for i in range(24)]          # shared prefix
    + [(i, 12 + (i - 24) * 0.4, BLUE, None) for i in range(24, 34)]   # req 1 tail
    + [(i, 22 + (i - 34) * 0.4, BLUE, None) for i in range(34, 44)]   # req 2 tail
)
RIGHT_PLAN = (
    [(i, 40 + i * 0.16, VIOLET, 47) for i in range(24)]      # arrived over the wire
    + [(i, 47 + (i - 24) * 0.3, BLUE, None) for i in range(24, 30)]
)

CAPTIONS = [
    (0, 17, "cold request  ·  prefill computes 24 blocks"),
    (18, 31, "same prefix  ·  cache hit, prefill skipped"),
    (32, 45, "9.4 MB of KV crosses the interconnect"),
    (46, FRAMES, "same tokens, different accelerator"),
]


def block_colour(plan, idx: int, f: int):
    for i, start, colour, green_at in plan:
        if i != idx:
            continue
        if f < start:
            return FREE
        if green_at is not None and f >= green_at:
            return GREEN
        return colour
    return FREE


def panel(d: ImageDraw.ImageDraw, x: int, name: str, sub: str, plan, f: int):
    d.rounded_rectangle([x, PY, x + PANEL_W, PY + PANEL_H], 6, fill=PANEL, outline=EDGE)
    d.text((x + 14, PY + 11), name, font=F_LBL, fill=INK)
    d.text((x + 14, PY + 27), sub, font=F_TINY, fill=DIM)
    gy = PY + 44
    for i in range(COLS * ROWS):
        cx = x + 14 + (i % COLS) * (CELL + GAP)
        cy = gy + (i // COLS) * (CELL + GAP)
        d.rounded_rectangle([cx, cy, cx + CELL, cy + CELL], 2,
                            fill=block_colour(plan, i, f))


def frame(f: int) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    d.text((22, 16), "hetero-serve", font=F_TITLE, fill=INK)
    d.text((22, 35), "PAGED KV CACHE  ·  PREFIX SHARING  ·  MIGRATION UNDER A LINK BUDGET",
           font=F_SUB, fill=DIM)

    for lo, hi, text in CAPTIONS:
        if lo <= f < hi:
            tw = d.textlength(text, font=F_CAP)
            d.text(((W - tw) / 2, 56), text, font=F_CAP, fill=ACCENT)
            break

    panel(d, LX, "cuda:0", "TESLA T4  ·  100 BLOCKS", LEFT_PLAN, f)
    panel(d, RX, "cuda:1", "TESLA T4  ·  100 BLOCKS", RIGHT_PLAN, f)

    # interconnect
    wy = PY + PANEL_H // 2
    x0, x1 = LX + PANEL_W + 4, RX - 4
    for xx in range(x0, x1, 7):
        d.line([xx, wy, xx + 3, wy], fill=EDGE, width=1)
    if 32 <= f < 40:                                  # the payload in flight
        t = (f - 32) / 8
        px = x0 + (x1 - x0 - 16) * t
        d.rounded_rectangle([px, wy - 5, px + 16, wy + 5], 2, fill=VIOLET)

    # legend + headline figure
    ly = H - 22
    for dx, colour, label in ((0, BLUE, "IN USE"), (92, GREEN, "CACHED"),
                              (188, VIOLET, "OVER THE WIRE")):
        d.rounded_rectangle([22 + dx, ly, 30 + dx, ly + 8], 1, fill=colour)
        d.text((36 + dx, ly - 1), label, font=F_TINY, fill=DIM)
    tail = "55.4% OF T4 PEAK BANDWIDTH  ·  101 TESTS"
    d.text((W - 22 - d.textlength(tail, font=F_TINY), ly - 1), tail, font=F_TINY, fill=DIM)
    return img


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames = [frame(f) for f in range(FRAMES)]

    # Build the palette from a strip containing every state any frame reaches,
    # so no frame needs its own local palette.
    probe = Image.new("RGB", (W, H * 4))
    for i, f in enumerate((6, 20, 36, 50)):
        probe.paste(frames[min(f, FRAMES - 1)], (0, i * H))
    master = probe.quantize(colors=128, method=Image.MEDIANCUT)
    pal = [im.quantize(palette=master, dither=Image.Dither.NONE) for im in frames]
    pal[0].save(
        OUT, save_all=True, append_images=pal[1:], duration=MS, loop=0,
        optimize=False, disposal=2,
    )
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB, {FRAMES} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
