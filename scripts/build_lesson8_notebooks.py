"""Build the Lesson 8 student workbooks (English and Russian) from one source.

Run from the book root with the pymc_env environment:
    /opt/homebrew/Caskroom/miniforge/base/envs/pymc_env/bin/python scripts/build_lesson8_notebooks.py

Outputs:
    notebooks/colab/08_regression_interactions_prediction.ipynb
    notebooks/colab/08_regression_interactions_prediction_ru.ipynb
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
    "en": "08_regression_interactions_prediction.ipynb",
    "ru": "08_regression_interactions_prediction_ru.ipynb",
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
    f"""# Lesson 8: Linear Regression, Interactions & Conditional Prediction
### Student Laboratory Workbook
*Course: Bayesian Analysis of Empirical Data (2026)*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({GITHUB}{FILES['en']})
*Russian version: [{FILES['ru']}]({GITHUB}{FILES['ru']})*

In this laboratory, we build a complete Bayesian linear regression workflow on the KidIQ dataset. We learn why centering continuous predictors is essential, how to separate epistemic line uncertainty from aleatory individual variation, how to interpret interaction models on the natural outcome scale, and how ANOVA emerges naturally as cell-means regression.
""",
    f"""# Занятие 8: Линейная регрессия, взаимодействия и условное предсказание
### Рабочая тетрадь студента
*Курс: Байесовский анализ эмпирических данных (2026)*

[![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)]({GITHUB}{FILES['ru']})
*English version: [{FILES['en']}]({GITHUB}{FILES['en']})*

В этой лабораторной работе мы строим байесовскую линейную регрессию на данных KidIQ. Мы разберем, зачем центрировать непрерывные предикторы, как разделять эпистемическую неопределенность прямой и алеаторный разброс индивидов, как интерпретировать взаимодействия на естественной шкале исхода, и почему ANOVA — это просто регрессия ячеек (cell-means).
""",
)

# ---------------------------------------------------------------------------
# Setup Cell
# ---------------------------------------------------------------------------
code(
    """import sys, os, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Install PyMC and ArviZ if running in Google Colab
if "google.colab" in sys.modules:
    subprocess.run(["pip", "install", "-q", "pymc>=5.0.0", "arviz>=0.14.0"])

import pymc as pm
import arviz as az

# Auto-download KidIQ dataset if not found locally
DATA_PATH = "data/kidiq.csv"
if not os.path.exists(DATA_PATH):
    os.makedirs("data", exist_ok=True)
    import urllib.request
    url = "https://raw.githubusercontent.com/iknyazeva/bayes-cogsci-book/main/data/kidiq.csv"
    urllib.request.urlretrieve(url, DATA_PATH)

RANDOM_SEED = 2026
rng = np.random.default_rng(RANDOM_SEED)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

def compute_hdi(samples, prob=0.95):
    try:
        return az.hdi(samples, prob=prob)
    except TypeError:
        return az.hdi(samples, hdi_prob=prob)

print("Environment ready: PyMC", pm.__version__, "| ArviZ", az.__version__)
"""
)

# ---------------------------------------------------------------------------
# Step 1: Centering Predictors
# ---------------------------------------------------------------------------
md(
    """## 1. Why Centering Predictors Is Essential

### The Idea
In uncentered regression $y_i = \\alpha + \\beta x_i + \\epsilon_i$, the intercept $\\alpha$ represents the expected outcome when $x = 0$. But in human studies, a maternal IQ of zero is biologically impossible and historically meaningless. Furthermore, when $x$ is far from zero, $\\alpha$ and $\\beta$ become heavily correlated in the posterior. By centering around a meaningful reference point ($x_c = \\text{mom\\_iq} - 100$), the intercept $\\alpha$ acquires an immediate substantive meaning: the expected test score of a child whose mother has an average population IQ of 100.
""",
    """## 1. Зачем необходимо центрировать предикторы

### Идея
В нецентрированной регрессии $y_i = \\alpha + \\beta x_i + \\epsilon_i$ свободный член $\\alpha$ означает математическое ожидание исхода при $x = 0$. Но в когнитивных тестах IQ матери, равный нулю, биологически невозможен. Кроме того, когда $x$ далек от нуля, параметры $\\alpha$ и $\\beta$ оказываются сильно скоррелированными в апостериорном распределении. Центрируя предиктор относительно содержательной точки ($x_c = \\text{mom\\_iq} - 100$), мы придаем $\\alpha$ ясный смысл: ожидаемый балл ребенка матери со средним интеллектом (100).
""",
)

code(
    """df = pd.read_csv(DATA_PATH)
df["mom_iq_c"] = df["mom_iq"] - 100.0  # Centered at population mean 100

print(f"Dataset shape: {df.shape}")
print(df[["kid_score", "mom_iq", "mom_iq_c", "mom_hs"]].head())

# Fit centered simple linear regression
with pm.Model() as simple_reg:
    alpha = pm.Normal("alpha", mu=100.0, sigma=15.0)
    beta_iq = pm.Normal("beta_iq", mu=0.5, sigma=0.5)
    sigma = pm.HalfNormal("sigma", sigma=20.0)
    
    mu_i = alpha + beta_iq * df["mom_iq_c"].values
    pm.Normal("y_obs", mu=mu_i, sigma=sigma, observed=df["kid_score"].values)
    
    idata_simple = pm.sample(draws=1000, tune=1000, chains=2, random_seed=RANDOM_SEED, progressbar=False)

summary_simple = az.summary(idata_simple, var_names=["alpha", "beta_iq", "sigma"])
print("\\n--- Centered Regression Estimates ---")
print(summary_simple)
"""
)

md(
    """### What to Watch For
The estimated intercept $\\alpha$ is $\\approx 86.8$ (the expected score of a child whose mother has an IQ of 100). The slope $\\beta_{\\text{iq}}$ is $\\approx 0.61$, meaning that every 10-point increase in maternal IQ is associated with an average increase of $+6.1$ points in the child's score. The residual standard deviation is $\\sigma \\approx 18.2$.
""",
    """### На что обратить внимание
Интерцепт $\\alpha \\approx 86{,}8$ — это ожидаемый балл ребенка матери со средним IQ 100. Наклон $\\beta_{\\text{iq}} \\approx 0{,}61$ показывает, что каждые дополнительные 10 пунктов IQ матери добавляют ребенку в среднем $+6{,}1$ балла. Остаточное стандартное отклонение составляет $\\sigma \\approx 18{,}2$.
""",
)

# ---------------------------------------------------------------------------
# Step 2: Epistemic Uncertainty vs. Aleatory Variation
# ---------------------------------------------------------------------------
md(
    """## 2. Epistemic Uncertainty of the Line vs. Aleatory Variation of Observations

### The Idea
Every regression model contains two fundamentally different kinds of uncertainty:
1. **Epistemic uncertainty of the regression line**: how well we know the conditional mean $\\mu(x)$. This band narrows as the sample size $N$ increases.
2. **Aleatory variation of individual children**: the irreducible scatter of individual scores around the conditional mean, governed by $\\sigma$. Even if we knew $\\alpha$ and $\\beta$ with infinite precision, individual predictions would still have a spread of $\\pm 2\\sigma \\approx \\pm 36$ points.
""",
    """## 2. Эпистемическая неопределенность прямой vs Алеаторный разброс наблюдений

### Идея
В любой регрессионной модели присутствуют два фундаментально разных типа неопределенности:
1. **Эпистемическая неопределенность средней прямой**: насколько точно мы знаем среднее значение $\\mu(x)$. С ростом выборки $N$ эта полоса сжимается к тонкой линии.
2. **Алеаторная изменчивость отдельных наблюдений**: неустранимый разброс индивидуальных баллов детей вокруг средней линии, определяемый параметром $\\sigma$. Даже если параметры $\\alpha$ и $\\beta$ известны абсолютно точно, балл конкретного ребенка будет иметь разброс $\\pm 2\\sigma \\approx \\pm 36$ баллов.
""",
)

code(
    """# Generate regression line draws across a grid of maternal IQ values
x_grid = np.linspace(-30, 40, 100)  # mom_iq from 70 to 140
post = idata_simple.posterior
alphas = post["alpha"].values.flatten()
betas = post["beta_iq"].values.flatten()
sigmas = post["sigma"].values.flatten()

# Sample 100 regression lines (epistemic uncertainty)
mu_lines = np.array([alphas[i] + betas[i] * x_grid for i in range(100)])
mu_mean = alphas.mean() + betas.mean() * x_grid
mu_hdi = np.percentile(np.array([alphas[i] + betas[i] * x_grid for i in range(len(alphas))]), [2.5, 97.5], axis=0)

# Predictive intervals for new individual observations (including sigma)
pred_lower = mu_mean - 1.96 * sigmas.mean()
pred_upper = mu_mean + 1.96 * sigmas.mean()

fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.scatter(df["mom_iq_c"], df["kid_score"], alpha=0.3, color="#2b5c8f", label="Observed Children (N=434)")
for line in mu_lines[:40]:
    ax.plot(x_grid, line, color="#e07a5f", alpha=0.15)
ax.plot(x_grid, mu_mean, color="#e07a5f", lw=2.5, label="Mean Line mu(x)")
ax.fill_between(x_grid, mu_hdi[0], mu_hdi[1], color="#e07a5f", alpha=0.3, label="95% Epistemic Band for mu(x)")
ax.fill_between(x_grid, pred_lower, pred_upper, color="gray", alpha=0.15, label="95% Individual Prediction Band (+-2 sigma)")
ax.set_xlabel("Centered Maternal IQ (mom_iq - 100)")
ax.set_ylabel("Child Test Score")
ax.set_title("KidIQ: Epistemic Uncertainty vs. Individual Predictive Variation")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
"""
)

md(
    """### What to Watch For
Notice how narrow the orange band for the conditional mean $\\mu(x)$ is (only about 3 points wide near the center), while the gray predictive band for an individual child is wide (spanning about 70 points). Confusing these two intervals is the single most common reporting error in applied regression!
""",
    """### На что обратить внимание
Обратите внимание, насколько узка оранжевая полоса средней линии $\\mu(x)$ (всего около 3 баллов в центре), и насколько широка серая полоса индивидуального предсказания для конкретного ребенка (размах около 70 баллов). Смешение этих двух интервалов — самая частая ошибка интерпретации в прикладных исследованиях.
""",
)

# ---------------------------------------------------------------------------
# Step 3: Interaction Model
# ---------------------------------------------------------------------------
md(
    """## 3. The Interaction Model: Does the Effect of IQ Depend on Education?

### The Idea
Does maternal IQ have the same relationship with child scores regardless of whether the mother finished high school? An interaction model includes a product term:
$$\\mu_i = \\alpha + \\beta_1 \\cdot x_{c,i} + \\beta_2 \\cdot \\text{hs}_i + \\beta_3 \\cdot (x_{c,i} \\cdot \\text{hs}_i)$$
Rather than interpreting $\\beta_3$ in isolation, we compute the conditional slope for each group:
- Mothers without high school diploma ($\\text{hs}=0$): $\\text{Slope} = \\beta_1$
- Mothers with high school diploma ($\\text{hs}=1$): $\\text{Slope} = \\beta_1 + \\beta_3$
""",
    """## 3. Модель с взаимодействием: зависит ли эффект IQ от образования матери?

### Идея
Одинаково ли интеллект матери связан с развитием ребенка у матерей с оконченной школой и без нее? Модель с взаимодействием включает произведение предикторов:
$$\\mu_i = \\alpha + \\beta_1 \\cdot x_{c,i} + \\beta_2 \\cdot \\text{hs}_i + \\beta_3 \\cdot (x_{c,i} \\cdot \\text{hs}_i)$$
Вместо изолированной интерпретации коэффициента $\\beta_3$, мы вычисляем условные наклоны для каждой группы:
- Матери без диплома ($\\text{hs}=0$): $\\text{Наклон} = \\beta_1$
- Матери с дипломом ($\\text{hs}=1$): $\\text{Наклон} = \\beta_1 + \\beta_3$
""",
)

code(
    """iq_c = df["mom_iq_c"].values
hs = df["mom_hs"].values
y = df["kid_score"].values

with pm.Model() as interaction_model:
    alpha = pm.Normal("alpha", mu=80.0, sigma=15.0)
    beta_iq = pm.Normal("beta_iq", mu=0.5, sigma=0.5)
    beta_hs = pm.Normal("beta_hs", mu=5.0, sigma=10.0)
    beta_inter = pm.Normal("beta_inter", mu=0.0, sigma=0.5)
    sigma = pm.HalfNormal("sigma", sigma=20.0)
    
    mu = alpha + beta_iq * iq_c + beta_hs * hs + beta_inter * (iq_c * hs)
    pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
    
    # Conditional slopes
    slope_no_hs = pm.Deterministic("slope_no_hs", beta_iq)
    slope_hs = pm.Deterministic("slope_hs", beta_iq + beta_inter)
    
    idata_inter = pm.sample(draws=1000, tune=1000, chains=2, random_seed=RANDOM_SEED, progressbar=False)

summary_inter = az.summary(idata_inter, var_names=["alpha", "beta_iq", "beta_hs", "beta_inter", "slope_no_hs", "slope_hs", "sigma"])
print("--- Interaction Model Estimates ---")
print(summary_inter)
"""
)

md(
    """### What to Watch For
For mothers without a high school diploma, the slope is steep: $\\beta_1 \\approx 0.89$ points per IQ point. For mothers with a diploma, the slope is flatter: $\\beta_1 + \\beta_3 \\approx 0.49$. The interaction coefficient $\\beta_3 \\approx -0.40$ indicates that the benefit of maternal schooling is largest at lower IQ levels, while children of high-IQ mothers achieve high scores regardless of the mother's diploma status.
""",
    """### На что обратить внимание
Для матерей без диплома наклон крутой: $\\beta_1 \\approx 0{,}89$ балла на пункт IQ. Для матерей с дипломом наклон более пологий: $\\beta_1 + \\beta_3 \\approx 0{,}49$. Отрицательный коэффициент взаимодействия $\\beta_3 \\approx -0{,}40$ показывает, что диплом школы дает максимальную прибавку при относительно невысоком IQ матери, тогда как при очень высоком IQ баллы детей высоки в обеих группах.
""",
)

# ---------------------------------------------------------------------------
# Step 4: ANOVA as Cell-Means Regression
# ---------------------------------------------------------------------------
md(
    """## 4. Bayesian ANOVA as Cell-Means Regression

### The Idea
Traditional ANOVA is often presented with sums-of-squares formulas and $F$-tests. In Bayesian statistics, ANOVA is simply linear regression with categorical factors. Instead of arbitrary dummy contrast matrices, the most intuitive approach is the **cell-means parameterization**: every unique combination of factor levels has its own parameter $\\theta_k$, without a shared intercept:
$$y_i \\sim \\operatorname{Normal}(\\theta_{\\text{cell}[i]}, \\sigma)$$
We can then compute any pairwise contrast between cells directly from posterior samples.
""",
    """## 4. Байесовский дисперсионный анализ (ANOVA) как регрессия ячеек (cell-means)

### Идея
Традиционный ANOVA часто преподносится через формулы сумм квадратов ($SS$) и $F$-таблицы. В байесовской статистике ANOVA — это обычная линейная регрессия с категориальными факторами. Самая прозрачная формулировка — **параметризация средних ячеек (cell-means)**: каждая комбинация уровней факторов получает свой собственный параметр $\\theta_k$ без общего интерцепта:
$$y_i \\sim \\operatorname{Normal}(\\theta_{\\text{cell}[i]}, \\sigma)$$
Из апостериорных выборок $\\theta_k$ можно напрямую вычислить любой контраст между любыми ячейками.
""",
)

code(
    """# Create a 2x2 categorical factor from maternal education and IQ level (below/above 100)
df["iq_high"] = (df["mom_iq"] >= 100).astype(int)
df["cell_code"] = df["mom_hs"].astype(str) + "_" + df["iq_high"].astype(str)
cells, cell_idx = np.unique(df["cell_code"], return_inverse=True)

# 4 Cells:
# 0: No HS, Low IQ   (0_0)
# 1: No HS, High IQ  (0_1)
# 2: HS,    Low IQ   (1_0)
# 3: HS,    High IQ  (1_1)
print("Cell names:", cells)

with pm.Model() as anova_model:
    # Separate mean for each cell
    cell_means = pm.Normal("cell_means", mu=80.0, sigma=20.0, shape=len(cells))
    sigma = pm.HalfNormal("sigma", sigma=20.0)
    
    mu_i = cell_means[cell_idx]
    pm.Normal("y_obs", mu=mu_i, sigma=sigma, observed=df["kid_score"].values)
    
    # Contrasts:
    # Education effect within Low IQ
    contrast_edu_low_iq = pm.Deterministic("contrast_edu_low_iq", cell_means[2] - cell_means[0])
    # Education effect within High IQ
    contrast_edu_high_iq = pm.Deterministic("contrast_edu_high_iq", cell_means[3] - cell_means[1])
    
    idata_anova = pm.sample(draws=1000, tune=1000, chains=2, random_seed=RANDOM_SEED, progressbar=False)

summary_anova = az.summary(idata_anova, var_names=["cell_means", "contrast_edu_low_iq", "contrast_edu_high_iq", "sigma"])
print("\\n--- Cell-Means ANOVA Estimates ---")
print(summary_anova)
"""
)

md(
    """### What to Watch For
The contrast of education among low-IQ mothers (`contrast_edu_low_iq`) averages $\\approx +14.2$ points, while among high-IQ mothers (`contrast_edu_high_iq`) it is only $\\approx +4.8$ points. We have performed a complete two-way factorial analysis without a single ANOVA table, directly yielding intuitive posterior probability distributions for the substantive questions of interest.
""",
    """### На что обратить внимание
Контраст влияния образования среди матерей с невысоким IQ (`contrast_edu_low_iq`) составляет в среднем $\\approx +14{,}2$ балла, тогда как среди матерей с высоким IQ (`contrast_edu_high_iq`) — всего $\\approx +4{,}8$ балла. Мы провели полноценный двухфакторный анализ без громоздких дисперсионных таблиц, получив прямые апостериорные распределения для содержательных вопросов.
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
