"""Post-process the two-project site (English /en, Russian /ru) after `myst build --html`.

Run from the book root after the build (CI does this automatically):

    python3 scripts/postbuild_site.py

1. Interactive figures are embedded with ``<iframe src="../_static/<name>.html">``.
   A page is reached either as ``/en/<slug>`` (client-side navigation, resolves
   to ``/_static``) or as ``/en/<slug>/`` (direct load on GitHub Pages, resolves
   to ``/en/_static``). MyST copies ``_static`` to the site root only, so this
   script also copies English figures to ``en/_static`` and Russian figures
   (``*_ru.html``) to ``ru/_static``.
2. Writes ``index.html`` at the site root that redirects to the English book.
3. Writes redirect pages at the old single-project addresses (``/<slug>/``)
   so links shared before the language split keep working, and redirects for
   pages that were renamed (``MOVED``).

Only the standard library is used, so it runs on a plain CI runner.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "_build" / "html"

REDIRECT = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="0; url={target}">
<link rel="canonical" href="{target}">
<title>Redirecting…</title></head>
<body><p>Moved to <a href="{target}">{target}</a>.</p></body></html>
"""


# Renamed pages: old slug -> new slug (English book).
MOVED = {
    "bayesiancomputationmcmc": "hamiltonianmontecarlonuts",
    "advanced-variationalinference": "variationalinference",
}


def copy_static() -> None:
    src = HTML / "_static"
    for lang, keep in (("en", lambda p: not p.name.endswith("_ru.html")), ("ru", lambda p: p.name.endswith("_ru.html"))):
        dest = HTML / lang / "_static"
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for item in src.iterdir():
            if item.is_dir():
                shutil.copytree(item, dest / item.name, dirs_exist_ok=True)
            elif item.suffix != ".html" or keep(item):
                shutil.copy2(item, dest / item.name)
                n += 1
        print(f"copied {n} static files to {lang}/_static")


def write_redirects() -> None:
    (HTML / "index.html").write_text(REDIRECT.format(target="en/"), encoding="utf-8")
    made = 0
    for page in sorted((HTML / "en").glob("*.json")):
        slug = page.stem
        if slug == "index":
            continue
        old = HTML / slug
        if old.exists():
            continue
        old.mkdir()
        (old / "index.html").write_text(REDIRECT.format(target=f"../en/{slug}/"), encoding="utf-8")
        made += 1
    for old, new in MOVED.items():
        for base, target in ((HTML / old, f"../en/{new}/"), (HTML / "en" / old, f"../{new}/")):
            if base.exists():
                continue
            base.mkdir(parents=True)
            (base / "index.html").write_text(REDIRECT.format(target=target), encoding="utf-8")
            made += 1
    print(f"wrote root redirect and {made} redirects from old addresses")


if __name__ == "__main__":
    if not (HTML / "en").is_dir() or not (HTML / "ru").is_dir():
        raise SystemExit("Expected _build/html/en and _build/html/ru; run `myst build --html` first.")
    copy_static()
    write_redirects()
