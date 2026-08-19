"""Inline web/ into the single self-contained file a Claude Artifact needs.

The site ships as index.html + sim.mjs so the scheduler logic can be unit tested
under Node. An artifact is one file with no same-origin fetches, so the module
gets inlined and the document wrapper stripped.

    python scripts/build_artifact.py [out.html]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"


def build() -> str:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    sim = (WEB / "sim.mjs").read_text(encoding="utf-8")

    # strip the module's export keywords: once inlined it is one script scope
    sim_inline = re.sub(r"^export\s+", "", sim, flags=re.M)

    # Drop the import and paste the module above the app code. Found with a
    # regex but substituted with a plain str.replace: the inlined source is full
    # of backslashes (regex literals), and re.sub would read those as escape
    # sequences in the replacement text.
    m = re.search(r'import \{[^}]*\} from "\./sim\.mjs";', html)
    if not m:
        raise SystemExit("web/index.html no longer imports ./sim.mjs")
    html = html.replace(
        m.group(0),
        "/* --- inlined from web/sim.mjs --- */\n" + sim_inline + "\n/* --- app --- */",
        1,
    )
    html = html.replace('<script type="module">', "<script>", 1)

    # the artifact host supplies <!doctype>/<html>/<head>/<body>
    head = re.search(r"<head>(.*?)</head>", html, re.S).group(1)
    body = re.search(r"<body>(.*?)</body>", html, re.S).group(1)
    keep = "\n".join(
        l for l in head.splitlines()
        if l.strip().startswith(("<title", "<link"))
    )
    return keep.strip() + "\n" + body.strip() + "\n"


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else WEB.parent / "build" / "artifact.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = build()
    for bad in ("<!doctype", "<html", "<head>", "<body>", 'from "./sim.mjs"'):
        assert bad not in text.lower(), f"artifact build still contains {bad}"
    assert "class Sim" in text and "requestAnimationFrame" in text, "inline failed"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, self-contained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
