# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Source for the book/course **"Bayesian Analysis of Empirical Data"** (14 lessons, Python/PyMC/ArviZ), built
with **MyST Markdown (mystmd, the Jupyter Book 2.x engine)** and deployed to GitHub Pages
(`iknyazeva.github.io/bayes-cogsci-book`). Content is prose + math + generated interactive Plotly figures;
there is no application code and no test suite.

## Environment and commands

All Python (scripts, notebooks, ad-hoc checks) runs in the **`pymc_env` conda environment**:

```bash
conda run -n pymc_env python scripts/<script>.py     # always run scripts this way
```

Build the site locally (mystmd is a Node tool and is not installed globally here):

```bash
npx -p mystmd myst build --html   # writes _build/html (what CI publishes)
npx -p mystmd myst start          # live preview server
```

Regenerate content artefacts (each script is idempotent and overwrites its outputs):

```bash
conda run -n pymc_env python scripts/create_lesson1_demos.py          # -> _static/*.html
conda run -n pymc_env python scripts/create_lesson2_demos.py
conda run -n pymc_env python scripts/create_lesson4_demos.py
conda run -n pymc_env python scripts/generate_dist_icons.py           # -> _static/dist_icons/
conda run -n pymc_env python scripts/prepare_real_data_cases.py       # downloads ROS-Examples -> data/*.csv
conda run -n pymc_env python scripts/build_lesson3_notebooks.py       # -> notebooks/colab/03_*.ipynb (en + ru)
conda run -n pymc_env python scripts/build_real_data_cases_notebook.py # -> notebooks/colab/03_04_*_ru.ipynb
```

Interactive Streamlit companion (separate from the book):

```bash
conda run -n pymc_env streamlit run streamlit_app/app.py
```

CI (`.github/workflows/deploy.yml`) installs `mystmd` via npm and runs `myst build --html` on every push to
`main`; it never executes notebooks or Python, so **anything generated must be committed**.

## Project rules (from AGENTS.md — these bind)

- Project notes, plans, decision records, specs and status updates go to
  `~/Work/Obsidian/MainVault/Projects/BayesBook/` (wiki notes under `_wiki/`, updating `_wiki/_index.md` and
  appending `_wiki/log.md` on material changes). **Do not create planning or staging notes in this repo.**
- This repo holds only source content, code, data, notebooks, configuration and build artefacts.
- Student Colab notebooks (`notebooks/colab/*.ipynb`) must load data from
  `https://raw.githubusercontent.com/iknyazeva/bayes-cogsci-book/main/data/<file>`, falling back to a local
  `data/` directory when not on Colab (see the `load()` helper in the generated notebooks).

## Architecture

**Flat content root.** Every lesson is a top-level `N.Name.md` file; the leading number is the session it
belongs to, not a directory. Multiple files share a number (e.g. `4.*` = Lesson 4 plus its appendices
4A–4E). `Advanced.*.md` is the optional block, `case_studies/` holds five applied cases (a `.md` chapter plus
a companion `.ipynb` and per-case `data/` and `figures/` subfolders), and `_archive/` holds superseded drafts
that are excluded from the build (see `_archive/README.md` for what replaced what).

**Two table-of-contents files must stay in sync.** `myst.yml` (`project.toc`, entries with `.md` extension)
is what actually builds; `_toc.yml` + `_config.yml` are the legacy Jupyter Book 1.x/Sphinx configuration kept
alongside it (entries without extension, `root: 0.Course_overview`). When adding, renaming or retitling a
page, update **both**, and mirror any exclusion added to `myst.yml:exclude` into `_config.yml`'s
`sphinx.config.exclude_patterns`.

**Figures are pre-generated, not executed at build time.** `execute_notebooks: "off"`. Interactive Plotly
figures are written by `scripts/create_lesson*_demos.py` as standalone HTML into `_static/`, then embedded in
prose with a raw iframe using a path relative to the rendered page:

```html
<iframe src="../_static/lesson4_rope.html" width="100%" height="420px" style="border:none; border-radius:8px;"></iframe>
```

`_static` is registered via `project.static_files` in `myst.yml`. Naming is `fig<session>_<slug>.html` or
`lesson<N>_<slug>.html`. To change a figure, edit the generating script and re-run it — never hand-edit the
HTML in `_static/`.

**Some Colab notebooks are generated from a single bilingual source.** `scripts/build_lesson3_notebooks.py`
and `scripts/build_real_data_cases_notebook.py` define each cell once as `{"en": ..., "ru": ...}` and emit
structurally identical English and Russian workbooks. For those notebooks, edit the builder script and re-run
it; hand-edits to the `.ipynb` will be lost. The remaining `notebooks/colab/*.ipynb` are authored directly.

**Datasets.** `data/` holds small teaching extracts; `scripts/prepare_real_data_cases.py` documents and
reproduces the ones derived from the Gelman/Hill/Vehtari ROS-Examples repository (it aggregates a 3 MB Stata
file down to the counts the conjugate cases need). Keep committed extracts small — Colab notebooks fetch them
over raw.githubusercontent.

## Authoring conventions

- MyST directives in use: `:::{dropdown}`, `:::{tip}`, `:::{admonition}`, `:::{note}`, `:::{warning}`,
  `:::{important}`, `::::{grid}` / `:::{grid-item-card}`, and fenced ```` ```{mermaid} ````, ```` ```{seealso} ````,
  ```` ```{bibliography} ````. Diagrams are Mermaid (`sphinxcontrib-mermaid` for the legacy path).
- Citations are `{cite}`key`` against `references.bib`; add the entry there before citing.
- Cross-references between chapters are plain relative Markdown links **with the `.md` extension**
  (`[Lesson 2](2.ProbabilityBayesianUpdating.md)`).
- Lessons that have a workbook open with a Colab badge linking to
  `https://colab.research.google.com/github/iknyazeva/bayes-cogsci-book/blob/main/notebooks/colab/<file>.ipynb`.
- Notebooks set `RANDOM_SEED = 42` with `np.random.default_rng`, detect Colab via
  `"google.colab" in sys.modules`, and set `pio.renderers.default = "colab"` there.
- Site styling lives in `custom.css` (referenced from `myst.yml:site.options.style`).
