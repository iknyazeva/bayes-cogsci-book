"""Build the Lesson 1 student workbooks (English and Russian) from one source.

Run from the book root with the pymc_env environment:
    /opt/homebrew/Caskroom/miniforge/base/envs/pymc_env/bin/python scripts/build_lesson1_notebooks.py

Outputs:
    notebooks/colab/01_research_questions_data_measurement.ipynb
    notebooks/colab/01_research_questions_data_measurement_ru.ipynb
"""

from __future__ import annotations
from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks" / "colab"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GITHUB = "https://colab.research.google.com/github/iknyazeva/bayes-cogsci-book/blob/main/notebooks/colab/"
FILES = {
    "en": "01_research_questions_data_measurement.ipynb",
    "ru": "01_research_questions_data_measurement_ru.ipynb",
}

CELLS: list[tuple[str, dict[str, str]]] = []


def md(en: str, ru: str | None = None) -> None:
    CELLS.append(("markdown", {"en": en, "ru": ru if ru is not None else en}))


def code(en: str, ru: str | None = None) -> None:
    CELLS.append(("code", {"en": en, "ru": ru if ru is not None else en}))


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
md(
    """# Lesson 1: Research Questions, Data & Measurement
### Case Study: What Does the Human Development Index Measure?
*Course: Bayesian Analysis of Empirical Data (2026)*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](""" + GITHUB + FILES['en'] + """)
*Russian version: [""" + FILES['ru'] + "](" + GITHUB + FILES['ru'] + """)*

In this laboratory, we explore the critical distinction between abstract scientific constructs and concrete empirical indicators using the Human Development Index (HDI) across U.S. states. We investigate variance dominance, unravel the Kentucky vs. South Carolina measurement puzzle, and inspect rank-rank calibration.
""",
    """# Занятие 1: Исследовательские вопросы, данные и шкалы измерения
### Практический кейс: Что на самом деле измеряет Индекс развития человеческого потенциала?
*Курс: Байесовский анализ эмпирических данных (2026)*

[![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)](""" + GITHUB + FILES['ru'] + """)
*English version: [""" + FILES['en'] + "](" + GITHUB + FILES['en'] + """)*

В этой лабораторной работе мы разберем принципиальное различие между абстрактными научными конструктами и наблюдаемыми эмпирическими индикаторами на примере Индекса развития человеческого потенциала (ИРЧП) по штатам США. Мы изучим доминирование дисперсии, разгадаем парадокс Кентукки против Южной Каролины и построим калибровку рангов.
""",
)

# ---------------------------------------------------------------------------
# Setup Cell
# ---------------------------------------------------------------------------
code(
    """import sys, os, urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# Auto-download dataset if not found locally
DATA_PATH = "data/hdi_state.csv"
if not os.path.exists(DATA_PATH):
    os.makedirs("data", exist_ok=True)
    url = "https://raw.githubusercontent.com/iknyazeva/bayes-cogsci-book/main/data/hdi_state.csv"
    urllib.request.urlretrieve(url, DATA_PATH)

df = pd.read_csv(DATA_PATH)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
print(f"Dataset loaded: {len(df)} rows, columns: {list(df.columns)}")
"""
)

# ---------------------------------------------------------------------------
# Step 1: Composite Measurement & Indicator Dominance
# ---------------------------------------------------------------------------
md(
    """## 1. Inspecting Composite Measurement & Indicator Correlation

### The Idea
The Human Development Index combines health, education, and economic indicators. Nominally, each dimension enters with equal weight. In practice, however, personal income varies across states far more than longevity or literacy, meaning the economic indicator numerically dominates the composite score. We compute the state income rank and HDI rank across the 50 U.S. states to measure how closely the composite index mirrors pure income.
""",
    """## 1. Составное измерение и доминирование индикаторов

### Идея
Индекс развития человеческого потенциала объединяет индикаторы здоровья, образования и дохода. Номинально каждое измерение входит в формулу с равным весом (1/3). На практике личный доход между штатами варьирует значительно сильнее, чем продолжительность жизни или грамотность, поэтому доход начинает численно доминировать над остальными измерениями. Мы рассчитываем ранги штатов по доходу и ИРЧП, чтобы проверить, насколько композитный индекс дублирует сырой доход.
""",
)

code(
    """# Filter 50 states with observed personal income in 2000
valid = df.dropna(subset=["income_2000"]).copy()
valid["rank_income"] = valid["income_2000"].rank(ascending=False)
valid["rank_hdi"] = valid["hdi"].rank(ascending=False)
valid["rank_gain"] = valid["rank_income"] - valid["rank_hdi"]

r_p, p_p = pearsonr(valid["income_2000"], valid["hdi"])
r_s, p_s = spearmanr(valid["rank_income"], valid["rank_hdi"])

print(f"Pearson Correlation (Raw Scale):   r   = {r_p:.3f} (p = {p_p:.2e})")
print(f"Spearman Correlation (Rank Scale): r_s = {r_s:.3f} (p = {p_s:.2e})")
print("\\nTop 5 States by HDI:")
print(valid.sort_values("rank_hdi")[["state", "income_2000", "rank_income", "hdi", "rank_hdi", "rank_gain"]].head())
"""
)

md(
    """### What to Watch For
The raw correlation between income and HDI is high ($r = 0.71, r_s = 0.86$). This shows that economic prosperity is a strong driver of the index. However, it does not explain everything: some states achieve substantially better human outcomes than their income rank alone would suggest.
""",
    """### На что обратить внимание
Корреляция между доходом и ИРЧП высока ($r = 0{,}71, r_s = 0{,}86$). Это подтверждает, что экономическое благосостояние сильно коррелирует с индексом. Тем не менее, доход объясняет далеко не всё: некоторые штаты достигают значительно более высоких социальных результатов, чем предсказывает уровень их доходов.
""",
)

# ---------------------------------------------------------------------------
# Step 2: The Kentucky vs. South Carolina Puzzle
# ---------------------------------------------------------------------------
md(
    """## 2. Measurement Discrepancies: The Kentucky vs. South Carolina Puzzle

### The Idea
Gelman, Hill, and Vehtari (*Regression and Other Stories*, Ch. 2) highlighted a classic measurement discrepancy between South Carolina and Kentucky. In 2000, both states had virtually identical average personal incomes (differing by only $11 per capita), yet their HDI scores diverged dramatically.
""",
    """## 2. Расхождения в измерениях: парадокс Кентукки и Южной Каролины

### Идея
Гельман, Хилл и Вехтари (*Regression and Other Stories*, гл. 2) обратили внимание на классическое расхождение измерений между Южной Каролиной и Кентукки. В 2000 году подушевой доход в этих двух штатах был практически одинаков (разница составляла всего $11 в год), однако их показатели ИРЧП резко разошлись.
""",
)

code(
    """sc_ky = valid[valid["state_abb"].isin(["SC", "KY"])][["state", "income_2000", "rank_income", "hdi", "rank_hdi"]]
print(sc_ky.to_string(index=False))

diff_income = abs(sc_ky.iloc[0]["income_2000"] - sc_ky.iloc[1]["income_2000"])
diff_hdi = abs(sc_ky.iloc[0]["hdi"] - sc_ky.iloc[1]["hdi"])
national_range = valid["hdi"].max() - valid["hdi"].min()

print(f"\\nIncome difference: ${diff_income:.2f} (Rank #{sc_ky.iloc[0]['rank_income']:.0f} vs #{sc_ky.iloc[1]['rank_income']:.0f})")
print(f"HDI difference:    {diff_hdi:.3f} (Rank #{sc_ky.iloc[0]['rank_hdi']:.0f} vs #{sc_ky.iloc[1]['rank_hdi']:.0f})")
print(f"Share of national HDI range: {diff_hdi / national_range * 100:.1f}%")
"""
)

md(
    """### What to Watch For
With only an $11 income gap, Kentucky and South Carolina differ by 13 positions in the national HDI ranking, spanning nearly a third ($31.3\\%$) of the entire national spread! This proves empirically that money alone is not an interchangeable proxy for the broader construct of human capability and health.
""",
    """### На что обратить внимание
При разнице в доходах всего в $11 Кентукки и Южная Каролина различаются на 13 позиций в национальном рейтинге ИРЧП, покрывая почти треть ($31{,}3\\%$) всего национального диапазона! Это эмпирически доказывает, что один лишь доход не может служить полным заменителем конструкта «человеческое развитие».
""",
)

# ---------------------------------------------------------------------------
# Step 3: Rank-Rank Calibration
# ---------------------------------------------------------------------------
md(
    """## 3. Rank-Rank Calibration & Social Efficiency

### The Idea
A rank-rank calibration plot plots income rank against composite HDI rank. The 45-degree diagonal line $y = x$ represents perfect parity: a state whose development matches its income rank falls exactly on the line. States sitting far above the diagonal achieve superior human development relative to their economic wealth (social efficiency), while states below the diagonal underperform.
""",
    """## 3. Калибровка рангов и социальная эффективность

### Идея
График калибровки ранг-ранг сопоставляет ранг штата по доходу с его рангом по ИРЧП. Диагональ $y = x$ обозначает идеальный паритет: штат, развитие которого точно соответствует его доходам, лежит строго на диагонали. Штаты выше диагонали достигают более высокого уровня развития, чем предсказывает их богатство (социальная эффективность), а штаты ниже диагонали отстают.
""",
)

code(
    """fig, ax = plt.subplots(figsize=(8.5, 4.5))
scatter = ax.scatter(valid["rank_income"], valid["rank_hdi"], c=valid["rank_gain"], cmap="viridis", s=55, edgecolor="black", alpha=0.85)
ax.plot([1, 50], [1, 50], color="#e07a5f", linestyle="--", lw=2, label="Equal Rank (y = x)")

# Annotate key diagnostic states
key_states = ["SC", "KY", "UT", "CT", "MS"]
for _, row in valid[valid["state_abb"].isin(key_states)].iterrows():
    ax.annotate(row["state_abb"], (row["rank_income"], row["rank_hdi"]),
                textcoords="offset points", xytext=(5, 5), fontweight="bold", fontsize=9)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Rank Gain (Income Rank - HDI Rank)")
ax.set_xlabel("Income Rank (1 = Highest Income)")
ax.set_ylabel("HDI Rank (1 = Highest HDI)")
ax.set_title(f"Rank-Rank Calibration Across 50 U.S. States (Spearman r_s = {r_s:.2f})")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
"""
)

md(
    """### What to Watch For
Utah (UT) is a striking outlier: while ranking #38 in personal income, it achieves rank #24 in HDI ($+14$ ranks gained), demonstrating high social conversion efficiency. Conversely, high-income states without strong public health or educational infrastructure lag behind their economic potential.
""",
    """### На что обратить внимание
Юта (UT) представляет собой яркий пример: занимая лишь 38-е место по личным доходам, штат занимает 24-е место по ИРЧП ($+14$ позиций выигрыша), демонстрируя высокую эффективность конвертации ресурсов в качество жизни. Напротив, штаты с высоким доходом, но отстающей социальной инфраструктурой, не реализуют свой экономический потенциал.
""",
)


# ---------------------------------------------------------------------------
# Build the notebooks
# ---------------------------------------------------------------------------
def build():
    for lang in ["en", "ru"]:
        nb = new_notebook()
        for cell_type, variants in CELLS:
            text = variants[lang]
            if cell_type == "markdown":
                nb.cells.append(new_markdown_cell(text))
            elif cell_type == "code":
                nb.cells.append(new_code_cell(text))
        
        target = OUT_DIR / FILES[lang]
        with open(target, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"Wrote {target}")


if __name__ == "__main__":
    build()
