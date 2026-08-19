"""Generate the animated README hero: two workers, a shared prefix, a migration.

An animated SVG rather than a GIF — a tenth the size, sharp at any zoom, diffable
in git, and it follows the reader's GitHub theme instead of baking in one
background. GitHub renders CSS animation inside an <img>-referenced SVG.

    python scripts/make_hero.py        # -> docs/hero.svg
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "hero.svg"

W, H = 860, 260
COLS, ROWS = 20, 5                     # 100 blocks per worker
CELL, GAP = 12, 3
GRID_W = COLS * (CELL + GAP) - GAP
PANEL_W, PANEL_H = GRID_W + 32, ROWS * (CELL + GAP) - GAP + 62
LEFT_X, RIGHT_X, PANEL_Y = 20, W - PANEL_W - 20, 58
CYCLE = 9.0                            # seconds


def cells(x0: int, y0: int, prefix: str, pattern) -> str:
    """pattern(i) -> (css_class, delay_seconds) or None for an untouched block."""
    out = []
    for i in range(COLS * ROWS):
        cx = x0 + (i % COLS) * (CELL + GAP)
        cy = y0 + (i // COLS) * (CELL + GAP)
        spec = pattern(i)
        cls, delay = spec if spec else ("", 0)
        style = f' style="animation-delay:{delay:.2f}s"' if cls else ""
        out.append(
            f'<rect class="b {cls}" x="{cx}" y="{cy}" width="{CELL}" '
            f'height="{CELL}" rx="1.5"{style}/>'
        )
    return "\n".join(out)


def left_pattern(i):
    if i < 24:                       # the shared prefix: fills, then goes green
        return ("fill-shared", 0.25 + i * 0.030)
    if i < 34:                       # this request's own tail
        return ("fill-own", 1.30 + (i - 24) * 0.045)
    if i < 44:                       # second request reuses the prefix, adds its own
        return ("fill-own2", 3.60 + (i - 34) * 0.045)
    return None


def right_pattern(i):
    if i < 24:                       # blocks that arrive over the wire
        return ("fill-recv", 6.35 + i * 0.022)
    if i < 30:
        return ("fill-own3", 7.30 + (i - 24) * 0.05)
    return None


SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}"
     role="img" aria-label="Two accelerators sharing a KV cache prefix: blocks fill on the first
     worker, a second request reuses them, then the cached prefix migrates across the interconnect
     to the second worker.">
<style>
  :root {{ color-scheme: light dark; }}
  .bg     {{ fill:#F4F5F8; }}
  .panel  {{ fill:#FFFFFF; stroke:#DADEE6; }}
  .b      {{ fill:#E2E6ED; }}
  .ink    {{ fill:#11131A; }}
  .ink2   {{ fill:#474D5B; }}
  .ink3   {{ fill:#79808F; }}
  .wire   {{ stroke:#C2C8D4; }}
  @media (prefers-color-scheme: dark) {{
    .bg    {{ fill:#131519; }}
    .panel {{ fill:#1A1D23; stroke:#2A2F38; }}
    .b     {{ fill:#0E1014; }}
    .ink   {{ fill:#F1F3F7; }}
    .ink2  {{ fill:#B2B8C5; }}
    .ink3  {{ fill:#7C8393; }}
    .wire  {{ stroke:#39404C; }}
  }}
  text {{ font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace; }}
  .h  {{ font-size:13px; font-weight:600; }}
  .s  {{ font-size:10px; letter-spacing:.09em; }}
  .k  {{ font-size:11px; }}

  .b {{ animation-duration:{CYCLE}s; animation-iteration-count:infinite;
        animation-timing-function:linear; animation-fill-mode:both; }}
  /* cold prefill -> published to the prefix cache (green) */
  .fill-shared {{ animation-name:shared; }}
  @keyframes shared {{
    0%,2%    {{ fill:#E2E6ED; }}
    4%,13%   {{ fill:#2A6FD6; }}
    16%,96%  {{ fill:#0C7A4A; }}
    100%     {{ fill:#E2E6ED; }}
  }}
  .fill-own  {{ animation-name:own; }}
  .fill-own2 {{ animation-name:own; }}
  .fill-own3 {{ animation-name:own; }}
  @keyframes own {{
    0%,2%   {{ fill:#E2E6ED; }}
    5%,26%  {{ fill:#2A6FD6; }}
    30%,96% {{ fill:#E2E6ED; }}
    100%    {{ fill:#E2E6ED; }}
  }}
  /* blocks that arrived over the interconnect */
  .fill-recv {{ animation-name:recv; }}
  @keyframes recv {{
    0%,70%   {{ fill:#E2E6ED; }}
    73%,78%  {{ fill:#D9541F; }}
    82%,96%  {{ fill:#0C7A4A; }}
    100%     {{ fill:#E2E6ED; }}
  }}
  @media (prefers-color-scheme: dark) {{
    @keyframes shared {{
      0%,2%{{fill:#0E1014}} 4%,13%{{fill:#4E93E8}} 16%,96%{{fill:#3FAE7A}} 100%{{fill:#0E1014}} }}
    @keyframes own {{
      0%,2%{{fill:#0E1014}} 5%,26%{{fill:#4E93E8}} 30%,96%{{fill:#0E1014}} 100%{{fill:#0E1014}} }}
    @keyframes recv {{
      0%,70%{{fill:#0E1014}} 73%,78%{{fill:#E8703C}} 82%,96%{{fill:#3FAE7A}} 100%{{fill:#0E1014}} }}
  }}

  .pkt {{ fill:#D9541F; animation:pkt {CYCLE}s linear infinite; opacity:0; }}
  @keyframes pkt {{
    0%,68%   {{ opacity:0; transform:translateX(0); }}
    70%      {{ opacity:1; transform:translateX(0); }}
    79%      {{ opacity:1; transform:translateX({RIGHT_X - LEFT_X - PANEL_W + 8}px); }}
    81%,100% {{ opacity:0; transform:translateX({RIGHT_X - LEFT_X - PANEL_W + 8}px); }}
  }}
  .cap {{ animation:cap {CYCLE}s linear infinite; }}
  @keyframes cap {{
    0%,14%   {{ opacity:1 }}  18%,33% {{ opacity:0 }}
    100%     {{ opacity:0 }}
  }}
  .cap2 {{ animation:cap2 {CYCLE}s linear infinite; opacity:0 }}
  @keyframes cap2 {{
    0%,36%   {{ opacity:0 }}  40%,62% {{ opacity:1 }}  66%,100% {{ opacity:0 }}
  }}
  .cap3 {{ animation:cap3 {CYCLE}s linear infinite; opacity:0 }}
  @keyframes cap3 {{
    0%,68%   {{ opacity:0 }}  72%,92% {{ opacity:1 }}  96%,100% {{ opacity:0 }}
  }}
  @media (prefers-reduced-motion: reduce) {{
    .b,.pkt,.cap,.cap2,.cap3 {{ animation:none; }}
    .fill-shared,.fill-recv {{ fill:#0C7A4A; }}
    .fill-own,.fill-own2,.fill-own3 {{ fill:#2A6FD6; }}
    .cap2,.cap3,.pkt {{ opacity:0 }}
  }}
</style>

<rect class="bg" width="{W}" height="{H}"/>

<text class="ink h" x="20" y="26">hetero-serve</text>
<text class="ink3 s" x="20" y="42">PAGED KV CACHE · PREFIX SHARING · MIGRATION UNDER A LINK BUDGET</text>

<!-- captions, one per beat -->
<text class="ink2 k cap"  x="{W//2}" y="42" text-anchor="middle">cold request — prefill fills 24 blocks</text>
<text class="ink2 k cap2" x="{W//2}" y="42" text-anchor="middle">same prefix — cache hit, prefill skipped</text>
<text class="ink2 k cap3" x="{W//2}" y="42" text-anchor="middle">9.4 MB of KV crosses the interconnect</text>

<!-- worker A -->
<rect class="panel" x="{LEFT_X}" y="{PANEL_Y}" width="{PANEL_W}" height="{PANEL_H}" rx="3" stroke-width="1"/>
<text class="ink k" x="{LEFT_X + 16}" y="{PANEL_Y + 22}">cuda:0</text>
<text class="ink3 s" x="{LEFT_X + 16}" y="{PANEL_Y + 37}">TESLA T4 · 100 BLOCKS</text>
{cells(LEFT_X + 16, PANEL_Y + 48, "a", left_pattern)}

<!-- worker B -->
<rect class="panel" x="{RIGHT_X}" y="{PANEL_Y}" width="{PANEL_W}" height="{PANEL_H}" rx="3" stroke-width="1"/>
<text class="ink k" x="{RIGHT_X + 16}" y="{PANEL_Y + 22}">cuda:1</text>
<text class="ink3 s" x="{RIGHT_X + 16}" y="{PANEL_Y + 37}">TESLA T4 · 100 BLOCKS</text>
{cells(RIGHT_X + 16, PANEL_Y + 48, "b", right_pattern)}

<!-- interconnect -->
<line class="wire" x1="{LEFT_X + PANEL_W}" y1="{PANEL_Y + PANEL_H // 2}"
      x2="{RIGHT_X}" y2="{PANEL_Y + PANEL_H // 2}" stroke-width="1" stroke-dasharray="3 3"/>
<rect class="pkt" x="{LEFT_X + PANEL_W + 2}" y="{PANEL_Y + PANEL_H // 2 - 4}"
      width="14" height="8" rx="1"/>

<!-- legend -->
<g transform="translate(20,{H - 16})">
  <rect x="0" y="-9" width="9" height="9" rx="1" fill="#2A6FD6"/>
  <text class="ink3 s" x="14" y="-1">IN USE</text>
  <rect x="76" y="-9" width="9" height="9" rx="1" fill="#0C7A4A"/>
  <text class="ink3 s" x="90" y="-1">CACHED — SHAREABLE</text>
  <rect x="238" y="-9" width="9" height="9" rx="1" fill="#D9541F"/>
  <text class="ink3 s" x="252" y="-1">ARRIVED OVER THE WIRE</text>
</g>
<text class="ink3 s" x="{W - 20}" y="{H - 16}" text-anchor="end">55.4% OF T4 PEAK BANDWIDTH · 44 TESTS</text>
</svg>
"""

def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(SVG, encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
