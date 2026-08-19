"""Charts for a sweep result.

Renders each figure twice — once for a light surface, once for a dark one — so a
README can serve whichever the reader's theme asks for. Colors come from a
validated categorical palette (see the project README for the validation run);
the ordering is the colorblind-safety mechanism, so slots are used in order and
never cycled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

REPO = Path(__file__).resolve().parents[2]

# --- theme -----------------------------------------------------------------

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
        # ordinal blue ramp, light: no lighter than step 250
        "ordinal": ["#86b6ef", "#2a78d6", "#0d366b"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
        # ordinal blue ramp, dark: no darker than step 600
        "ordinal": ["#9ec5f4", "#3987e5", "#184f95"],
    },
}

POLICY_LABEL = {
    "round_robin": "round robin",
    "least_loaded": "least loaded",
    "prefix_affinity": "prefix affinity",
    "cache_aware": "cache aware",
}


def style(ax, t, *, xlabel="", ylabel="", title="", subtitle=""):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    ax.grid(axis="y", color=t["grid"], linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=t["ink2"], fontsize=9.5, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=t["ink2"], fontsize=9.5, labelpad=8)
    if title:
        ax.set_title(title, color=t["ink"], fontsize=13, fontweight="600",
                     loc="left", pad=18 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=t["muted"],
                fontsize=9.5, va="bottom")


def legend(ax, t, **kw):
    lg = ax.legend(frameon=False, fontsize=9, labelcolor=t["ink2"], **kw)
    return lg


# --- figures ---------------------------------------------------------------


def _policy_rows(data) -> list[tuple[str, dict]]:
    """One row per policy, plus cache-aware at both ends of the link sweep.

    Showing cache-aware only at its best bandwidth would be cherry-picking: the
    whole point is that the same policy behaves differently when it can afford
    to migrate, so both regimes belong on the chart.
    """
    out: list[tuple[str, dict]] = []
    seen = set()
    for r in data["results"]:
        if r["policy"] == "cache_aware":
            continue
        if r["policy"] not in seen:
            seen.add(r["policy"])
            out.append((POLICY_LABEL.get(r["policy"], r["policy"]), r))

    ca = sorted([r for r in data["results"] if r["policy"] == "cache_aware"],
                key=lambda r: r["bandwidth_mbps"])
    if ca:
        lo, hi = ca[0], ca[-1]
        out.append((f"cache aware\n{lo['bandwidth_mbps']:g} Mbps"
                    f"\n({lo['migrations']} migrations)", lo))
        if hi is not lo:
            out.append((f"cache aware\n{hi['bandwidth_mbps']/1000:g} Gbps"
                        f"\n({hi['migrations']} migrations)", hi))
    return out


def _grouped(ax, t, labels, series, xs, fmt):
    n = len(series)
    width = 0.80 / n
    for i, (name, vals) in enumerate(series):
        offs = [x - 0.40 + width * (i + 0.5) for x in xs]
        bars = ax.bar(offs, vals, width * 0.90, label=name,
                      color=t["ordinal"][i], zorder=3, linewidth=0)
        # Direct labels also satisfy the relief rule for low-contrast fills.
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt(v),
                    ha="center", va="bottom", fontsize=8, color=t["ink2"])
    ax.set_ylim(0, max(max(v) for _, v in series) * 1.28)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)


def fig_ttft(data, mode: str, out: Path) -> Path:
    """Honest version: which policy wins depends on which latency you mean."""
    t = THEMES[mode]
    rows = _policy_rows(data)
    if not rows:
        return Path()
    labels = [lab for lab, _ in rows]
    rs = [r for _, r in rows]
    xs = list(range(len(rs)))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.6, 7.4), sharex=True, facecolor=t["surface"],
        gridspec_kw={"hspace": 0.20},
    )

    _grouped(ax1, t, labels,
             [("p50", [r["ttft_p50"] for r in rs]),
              ("p95", [r["ttft_p95"] for r in rs]),
              ("p99", [r["ttft_p99"] for r in rs])],
             xs, lambda v: f"{v*1000:.0f}")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*1000:.0f}"))
    style(ax1, t, ylabel="time to first token (ms)",
          title="Which policy wins depends on which latency you mean",
          subtitle="same workload, same hardware, only the routing policy changes  ·  lower is better")
    legend(ax1, t, ncol=3, loc="upper left")

    _grouped(ax2, t, labels,
             [("p50", [r["e2e_p50"] for r in rs]),
              ("p95", [r["e2e_p95"] for r in rs])],
             xs, lambda v: f"{v:.2f}")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    style(ax2, t, ylabel="end-to-end latency (s)")
    legend(ax2, t, ncol=2, loc="upper left")

    fig.tight_layout()
    path = out / f"ttft-by-policy-{mode}.png"
    fig.savefig(path, dpi=200, facecolor=t["surface"])
    plt.close(fig)
    return path


def fig_crossover(data, mode: str, out: Path) -> Path:
    """The headline: where moving KV stops being cheaper than recomputing it."""
    t = THEMES[mode]
    rows = sorted([r for r in data["results"] if r["policy"] == "cache_aware"],
                  key=lambda r: r["bandwidth_mbps"])
    if len(rows) < 2:
        return Path()

    # Both panels share ONE categorical x. Putting a log-scaled line plot above a
    # categorical bar plot with sharex=True silently maps the bars' 0..n indices
    # onto the log axis, which piles them at the left and collides the labels.
    bw = [r["bandwidth_mbps"] for r in rows]
    xs = list(range(len(rows)))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.4, 6.6), sharex=True, facecolor=t["surface"],
        gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.22},
    )

    for i, (key, name) in enumerate((("ttft_p50", "TTFT p50"), ("ttft_p99", "TTFT p99"))):
        ax1.plot(xs, [r[key] for r in rows], marker="o", markersize=7, linewidth=2,
                 color=t["series"][i], label=name, zorder=3,
                 markeredgecolor=t["surface"], markeredgewidth=2)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*1000:.0f}"))
    top = max(r["ttft_p99"] for r in rows)
    ax1.set_ylim(0, top * 1.34)          # headroom so the legend clears the peak
    style(ax1, t, ylabel="time to first token (ms)",
          title="Migrate or recompute? The link budget decides",
          subtitle="cache-aware policy, swept across interconnect bandwidth")
    legend(ax1, t, ncol=2, loc="upper left")

    migr = [r["migrations"] for r in rows]
    ax2.bar(xs, migr, 0.5, color=t["series"][0], zorder=3, linewidth=0)
    for x, v in zip(xs, migr):
        ax2.text(x, v, f"{v}", ha="center", va="bottom", fontsize=8.5, color=t["ink2"])
    ax2.set_ylim(0, max(max(migr), 1) * 1.2)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{int(b):,}" for b in bw])
    ax2.set_xlim(-0.5, len(xs) - 0.5)
    style(ax2, t, xlabel="interconnect bandwidth (Mbps)",
          ylabel="KV migrations chosen")
    fig.tight_layout()
    path = out / f"bandwidth-crossover-{mode}.png"
    fig.savefig(path, dpi=200, facecolor=t["surface"])
    plt.close(fig)
    return path


def fig_devices(data, mode: str, out: Path) -> Path:
    t = THEMES[mode]
    rows, seen = [], set()
    for r in data["results"]:
        if r["policy"] not in seen and r.get("per_worker"):
            seen.add(r["policy"])
            rows.append(r)
    if not rows:
        return Path()

    workers = sorted({w for r in rows for w in r["per_worker"]})
    labels = [POLICY_LABEL.get(r["policy"], r["policy"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8.8, 5.0), facecolor=t["surface"])
    bottoms = [0.0] * len(rows)
    for i, w in enumerate(workers):
        vals = [r["per_worker"].get(w, {}).get("generated", 0) for r in rows]
        ax.bar(labels, vals, 0.55, bottom=bottoms, label=w,
               color=t["series"][i % len(t["series"])], zorder=3,
               linewidth=2, edgecolor=t["surface"])   # 2px surface gap between segments
        # Label inside each segment: identity never rests on colour alone, and
        # it covers the relief rule for the low-contrast fill on light surfaces.
        for x, (v, b) in enumerate(zip(vals, bottoms)):
            if v > max(bottoms + vals) * 0.06:
                ax.text(x, b + v / 2, f"{int(v)}", ha="center", va="center",
                        fontsize=9, color="#ffffff", fontweight="600", zorder=4)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    # Every policy serves the identical workload, so the totals are equal by
    # construction — labelling them would just add noise. The split is the point.
    ax.set_ylim(0, max(bottoms) * 1.06)
    style(ax, t, ylabel="tokens generated",
          title="Where the work actually ran",
          subtitle="tokens produced per accelerator, by routing policy  ·  same total work each time")
    legend(ax, t, ncol=len(workers), loc="upper center",
           bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout()
    path = out / f"device-split-{mode}.png"
    fig.savefig(path, dpi=200, facecolor=t["surface"])
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="?", help="sweep-*.json (default: newest in results/)")
    ap.add_argument("--out", default=str(REPO / "results" / "figures"))
    args = ap.parse_args()

    if args.json:
        src = Path(args.json)
    else:
        cands = sorted((REPO / "results").glob("sweep-*.json"))
        if not cands:
            print("no sweep json found; run heteroserve.bench.sweep first")
            return 1
        src = cands[-1]

    data = json.loads(src.read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    made = []
    for mode in ("light", "dark"):
        for fn in (fig_ttft, fig_crossover, fig_devices):
            p = fn(data, mode, out)
            if p and p.name:
                made.append(p)

    print(f"source: {src.name}")
    for p in made:
        print(f"  wrote {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
