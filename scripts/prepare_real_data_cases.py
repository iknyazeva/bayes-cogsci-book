"""Prepare the teaching extracts used by the real-data conjugate case gallery.

Sources: the Regression and Other Stories example repository
(https://github.com/avehtari/ROS-Examples), by Gelman, Hill and Vehtari.

Run from the book root with the pymc_env environment:

    conda run -n pymc_env python scripts/prepare_real_data_cases.py

Writes into data/:
    pew_attendance_by_state.csv   aggregated religious-attendance counts per US state
    risky_behavior.csv            individual-level counts from an HIV-prevention study
    earnings_1990.csv             earnings, sex and education from a 1990 survey

The Pew source is a 3 MB Stata file of 31 201 individual records; only the
aggregated state-by-category counts are stored here, which is all the
Beta-Binomial and Dirichlet-Multinomial cases need.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ROS = "https://raw.githubusercontent.com/avehtari/ROS-Examples/master"

# Ordered religious-attendance categories, most to least frequent.
ATTEND_ORDER = [
    ("more than once a week", "more_than_weekly"),
    ("once a week", "weekly"),
    ("once or twice a month", "monthly"),
    ("a few times a year", "few_times_year"),
    ("seldom", "seldom"),
    ("never", "never"),
]


def fetch(path: str) -> bytes:
    r = requests.get(f"{ROS}/{path}", timeout=120)
    r.raise_for_status()
    return r.content


def build_pew() -> None:
    raw = fetch("Pew/data/pew_research_center_june_elect_wknd_data.dta")
    p = pd.read_stata(io.BytesIO(raw), convert_categoricals=True)
    d = p[["state", "attend"]].dropna()
    d = d[d.attend.isin([c for c, _ in ATTEND_ORDER])]

    wide = (
        d.groupby(["state", "attend"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[c for c, _ in ATTEND_ORDER], fill_value=0)
        .rename(columns=dict(ATTEND_ORDER))
    )
    wide.insert(0, "n_respondents", wide.sum(axis=1))
    # Dichotomised version used by the Beta-Binomial case.
    wide.insert(1, "weekly_or_more", wide["more_than_weekly"] + wide["weekly"])
    wide = wide.reset_index().sort_values("state")
    wide.to_csv(DATA / "pew_attendance_by_state.csv", index=False)
    print(f"pew_attendance_by_state.csv: {len(wide)} states, {wide.n_respondents.sum()} respondents")


def build_risky() -> None:
    r = pd.read_csv(io.BytesIO(fetch("RiskyBehavior/data/risky.csv")))
    r = r.rename(columns={"bupacts": "acts_before", "fupacts": "acts_after"})
    r["acts_after"] = r["acts_after"].round().astype(int)
    r.to_csv(DATA / "risky_behavior.csv", index=False)
    print(
        f"risky_behavior.csv: {len(r)} couples, "
        f"mean before {r.acts_before.mean():.1f}, mean after {r.acts_after.mean():.1f}"
    )


def build_earnings() -> None:
    e = pd.read_csv(io.BytesIO(fetch("Earnings/data/earnings.csv")))
    keep = ["earn", "male", "education", "ethnicity", "age", "height", "weight"]
    e = e[keep].copy()
    e.to_csv(DATA / "earnings_1990.csv", index=False)
    print(f"earnings_1990.csv: {len(e)} respondents, {(e.earn == 0).mean():.1%} with zero earnings")


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    build_pew()
    build_risky()
    build_earnings()
