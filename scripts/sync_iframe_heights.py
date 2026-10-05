"""Keep embedded figures at their intended height.

mystmd turns every ``<iframe ... height="NNNpx">`` in the Markdown into its own
responsive iframe block and drops the ``height`` attribute, so without CSS every
figure would be squeezed into the default box from ``custom.css`` (about 480 px)
and tall figures would be cut off.

This script reads the height written in each page's iframe tag, raises it to the
figure's own Plotly layout height when that is larger (or to 480 px when the
figure has no fixed height), and writes one CSS rule per figure file into
``custom.css`` between the markers
``BEGIN generated iframe heights`` and ``END generated iframe heights``.
Run it after adding or resizing a figure (CI also runs it before the build):

    python3 scripts/sync_iframe_heights.py

Only the standard library is used.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = ROOT / "custom.css"
BEGIN = "/* BEGIN generated iframe heights (scripts/sync_iframe_heights.py) */"
END = "/* END generated iframe heights */"
SKIP = ("_build", "_archive", "node_modules", ".git")
STATIC = ROOT / "_static"
DEFAULT_MIN = 480  # the box custom.css gave every iframe before this script existed

TAG = re.compile(r"<iframe\b[^>]*>", re.IGNORECASE)
SRC = re.compile(r'src="[^"]*?_static/([^"?#]+\.html)', re.IGNORECASE)
HEIGHT = re.compile(r'\bheight="(\d+)(?:px)?"', re.IGNORECASE)


def collect() -> dict[str, int]:
    heights: dict[str, int] = {}
    for md in sorted(ROOT.rglob("*.md")):
        if any(part in SKIP for part in md.relative_to(ROOT).parts):
            continue
        for tag in TAG.findall(md.read_text(encoding="utf-8")):
            src, height = SRC.search(tag), HEIGHT.search(tag)
            if not src or not height:
                continue
            name, h = src.group(1), int(height.group(1))
            if name in heights and heights[name] != h:
                print(f"note: {name} has heights {heights[name]} and {h}; using the larger")
            heights[name] = max(h, heights.get(name, 0))
    return heights


def figure_height(name: str) -> int | None:
    """Height fixed in the figure's Plotly layout, or None (autosized / not Plotly)."""
    path = STATIC / name
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    i = text.find("Plotly.newPlot(")
    if i < 0:
        return None
    dec = json.JSONDecoder()
    try:
        k = text.index(",", i) + 1                       # skip the element id
        while text[k].isspace():
            k += 1
        _, k = dec.raw_decode(text, k)                   # data
        k = text.index(",", k) + 1
        while text[k].isspace():
            k += 1
        layout, _ = dec.raw_decode(text, k)              # layout
    except (ValueError, IndexError):
        return None
    h = layout.get("height") if isinstance(layout, dict) else None
    return int(h) if isinstance(h, (int, float)) else None


def css_block(heights: dict[str, int]) -> str:
    rules = [BEGIN]
    for name, h in sorted(heights.items()):
        sel = f'[src*="/{name}"]'
        rules.append(f"div:has(> iframe{sel}),\n.relative.inline-block:has(> iframe{sel}),\niframe{sel} "
                     f"{{ height: {h}px !important; min-height: {h}px !important; }}")
    rules.append(END)
    return "\n".join(rules) + "\n"


def main() -> None:
    heights = collect()
    for name, h in heights.items():
        fig_h = figure_height(name)
        need = fig_h + 10 if fig_h else DEFAULT_MIN
        if need > h:
            heights[name] = need
    text = CSS.read_text(encoding="utf-8")
    block = css_block(heights)
    if BEGIN in text and END in text:
        start = text.index(BEGIN)
        end = text.index(END) + len(END)
        text = text[:start] + block.rstrip("\n") + text[end:]
    else:
        text = text.rstrip("\n") + "\n\n" + block
    CSS.write_text(text, encoding="utf-8")
    print(f"wrote height rules for {len(heights)} figures to {CSS.name}")


if __name__ == "__main__":
    main()
