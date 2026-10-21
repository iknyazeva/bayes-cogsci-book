"""Build the Lesson 2 student workbooks (English and Russian) from one source.

Run from the book root with the pymc_env environment:
    /opt/homebrew/Caskroom/miniforge/base/envs/pymc_env/bin/python scripts/build_lesson2_notebooks.py

Outputs:
    notebooks/colab/02_probability_bayesian_updating.ipynb
    notebooks/colab/02_probability_bayesian_updating_ru.ipynb
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
    "en": "02_probability_bayesian_updating.ipynb",
    "ru": "02_probability_bayesian_updating_ru.ipynb",
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
    """# Lesson 2: Probability and Bayesian Updating
### Conditional Probability, Base Rates, Likelihood & Sequential Updating
*Course: Bayesian Analysis of Empirical Data (2026)*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](""" + GITHUB + FILES['en'] + """)
*Russian version: [""" + FILES['ru'] + "](" + GITHUB + FILES['ru'] + """)*

In this laboratory, we explore the core machinery of probabilistic conditioning: Simpson's paradox in the 1973 UC Berkeley admissions data, the false positive paradox and natural frequencies, the mathematical distinction between likelihood and probability density, and order-invariance in sequential updating.
""",
    """# Занятие 2: Вероятность и байесовское обновление
### Условная вероятность, базовые частоты, правдоподобие и последовательное обновление
*Курс: Байесовский анализ эмпирических данных (2026)*

[![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)](""" + GITHUB + FILES['ru'] + """)
*English version: [""" + FILES['en'] + "](" + GITHUB + FILES['en'] + """)*

В этой лабораторной работе мы разберем фундамент вероятностного обуславливания: парадокс Симпсона на данных приема в Беркли 1973 года, парадокс ложноположительных результатов и метод естественных частот, математическое различие между правдоподобием и плотностью вероятности, а также инвариантность последовательного байесовского обновления к порядку данных.
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
from scipy import stats, integrate

# Auto-download dataset if not found locally
DATA_PATH = "data/ucbadmit.csv"
if not os.path.exists(DATA_PATH):
    os.makedirs("data", exist_ok=True)
    url = "https://raw.githubusercontent.com/iknyazeva/bayes-cogsci-book/main/data/ucbadmit.csv"
    urllib.request.urlretrieve(url, DATA_PATH)

df_admit = pd.read_csv(DATA_PATH)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
print("Dataset ready: UC Berkeley admissions (1973)")
print(df_admit.head())
"""
)

# ---------------------------------------------------------------------------
# Step 1: Conditioning and Simpson's Paradox
# ---------------------------------------------------------------------------
md(
    """## 1. Conditioning and Simpson's Paradox (Berkeley Admissions)

### The Idea
Conditioning on information alters our probability distribution. In the famous 1973 UC Berkeley admissions study (Bickel et al., 1975), the aggregate admission rate for men was $44.3\\%$ compared to only $30.4\\%$ for women. However, when we condition on the specific department applied to, women were admitted at equal or higher rates than men in nearly every department! The paradox arises because women systematically applied in greater numbers to departments with lower overall admission rates.
""",
    """## 1. Обуславливание и парадокс Симпсона (прием в Беркли)

### Идея
Обуславливание на новую информацию меняет распределение вероятностей. В исследовании приема в Калифорнийский университет в Беркли (Bickel et al., 1975) совокупная доля принятых мужчин составила $44{,}3\\%$, а женщин — лишь $30{,}4\\%$. Однако при обуславливании на факультет выясняется, что почти на каждом отдельном факультете доля принятых женщин была такой же или выше! Парадокс возникает из-за того, что женщины чаще подавали документы на высококонкурентные факультеты с низким общим процентом приема.
""",
)

code(
    """# Aggregate admission rates
tot_m = df_admit[df_admit["gender"] == "male"]["applications"].sum()
adm_m = df_admit[df_admit["gender"] == "male"]["admit"].sum()
tot_f = df_admit[df_admit["gender"] == "female"]["applications"].sum()
adm_f = df_admit[df_admit["gender"] == "female"]["admit"].sum()

print(f"Overall Men Admission Rate:   {adm_m / tot_m:.1%} ({adm_m}/{tot_m})")
print(f"Overall Women Admission Rate: {adm_f / tot_f:.1%} ({adm_f}/{tot_f})")

# Breakdown by department
dept_rates = df_admit.pivot(index="dept", columns="gender", values=["applications", "admit"])
dept_summary = pd.DataFrame({
    "Male_Admit_Rate": dept_rates["admit"]["male"] / dept_rates["applications"]["male"],
    "Female_Admit_Rate": dept_rates["admit"]["female"] / dept_rates["applications"]["female"],
    "Total_Applicants": dept_rates["applications"]["male"] + dept_rates["applications"]["female"]
})
dept_summary["Advantage"] = np.where(dept_summary["Female_Admit_Rate"] >= dept_summary["Male_Admit_Rate"], "Women", "Men")
print("\\nDepartment-by-Department Rates:")
print(dept_summary)
"""
)

md(
    """### What to Watch For
In 4 out of 6 major departments (A, B, D, F), the female admission rate was higher than the male rate. Aggregating across departments confounded the applicant's gender with department selectivity. Conditioning on the relevant confounding variable completely reversed the apparent conclusion!
""",
    """### На что обратить внимание
На 4 из 6 факультетов (A, B, D, F) процент приема женщин оказался выше, чем мужчин. Совокупное объединение данных смешало пол абитуриента с избирательностью факультета. Обуславливание на вмешивающуюся переменную полностью изменило вывод!
""",
)

# ---------------------------------------------------------------------------
# Step 2: The False Positive Paradox & Natural Frequencies
# ---------------------------------------------------------------------------
md(
    """## 2. The False Positive Paradox and Natural Frequencies

### The Idea
When testing for a rare condition (low prior prevalence), even an accurate test produces surprisingly many false alarms. Suppose a cognitive deficit has a base rate of $0.1\\%$ (1 in 1,000 people). A diagnostic test has $95\\%$ sensitivity (identifies true positives) and $95\\%$ specificity ($5\\%$ false positive rate). If a person tests positive, what is the probability they actually have the condition?
Using natural frequencies: in a cohort of 1,000 people, 1 person has the condition and tests positive ($0.95 \\times 1 \\approx 1$). Of the 999 healthy people, $5\\%$ test positive anyway ($0.05 \\times 999 \\approx 50$). Out of $51$ total positive tests, only $1$ is a true positive: $P(\\text{Condition} \\mid +) = 1 / 51 \\approx 2\\%$.
""",
    """## 2. Парадокс ложноположительных результатов и метод естественных частот

### Идея
При скрининге редких состояний (низкая априорная вероятность) даже качественный тест дает неожиданно много ложных тревог. Предположим, когнитивный дефицит встречается у $0{,}1\\%$ популяции (1 человек из 1000). Тест обладает чувствительностью $95\\%$ и специфичностью $95\\%$ (ложноположительные $5\\%$). Какова вероятность того, что человек с положительным тестом действительно болен?
На естественных частотах это видно сразу: из 1000 человек болен 1, и тест выявит его ($0{,}95 \\times 1 \\approx 1$). Из 999 здоровых людей $5\\%$ ошибочно получат положительный тест ($0{,}05 \\times 999 \\approx 50$). Из 51 положительного результата верен лишь 1: $P(\\text{Болен} \\mid +) = 1 / 51 \\approx 2\\%$.
""",
)

code(
    """def posterior_disease(prevalence, sensitivity=0.95, fpr=0.05):
    # Bayes rule: P(D | +) = P(+ | D) P(D) / [P(+ | D) P(D) + P(+ | healthy) P(healthy)]
    numerator = sensitivity * prevalence
    denominator = numerator + fpr * (1.0 - prevalence)
    return numerator / denominator

prev_grid = np.linspace(0.0005, 0.20, 200)
post_grid = [posterior_disease(p) for p in prev_grid]

fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(prev_grid * 100, np.array(post_grid) * 100, color="#2b5c8f", lw=2.5)
ax.axvline(0.1, color="#e07a5f", linestyle="--", label="Prevalence = 0.1% -> Post = 1.9%")
ax.plot(0.1, posterior_disease(0.001) * 100, "o", color="#e07a5f", markersize=7)
ax.set_xlabel("Prior Prevalence in Population (%)")
ax.set_ylabel("Posterior Probability P(Condition | +) (%)")
ax.set_title("How Prior Prevalence Dominates Single Test Evidence")
ax.legend()
plt.tight_layout()
plt.show()

print(f"Prior Prevalence: 0.1% -> Posterior Probability: {posterior_disease(0.001):.1%}")
print(f"Prior Prevalence: 1.0% -> Posterior Probability: {posterior_disease(0.010):.1%}")
print(f"Prior Prevalence: 5.0% -> Posterior Probability: {posterior_disease(0.050):.1%}")
"""
)

md(
    """### What to Watch For
When the prior base rate is $0.1\\%$, the posterior confidence is only $1.9\\%$. Only when the base rate reaches $5\\%$ does a single positive test push posterior confidence above $50\\%$. A single data point cannot overcome an overwhelmingly skeptical prior!
""",
    """### На что обратить внимание
При базовой частоте $0{,}1\\%$ апостериорная уверенность составляет лишь $1{,}9\\%$. Только при распространенности от $5\\%$ положительный тест поднимает вероятность болезни выше $50\\%$. Единичное наблюдение не способно перевесить сильный скептический априор!
""",
)

# ---------------------------------------------------------------------------
# Step 3: Likelihood vs. Probability Density
# ---------------------------------------------------------------------------
md(
    """## 3. Likelihood Function vs. Probability Density: The Mathematical Duality

### The Idea
The probability density $p(y \\mid \\theta)$ integrates to $1$ over data space $y$ (holding parameter $\\theta$ fixed). In contrast, the **likelihood function** $\\mathcal{L}(\\theta \\mid y)$ is evaluated over parameter space $\\theta$ (holding observed data $y$ fixed). Crucially, the likelihood function is **not** a probability density: for a binomial observation $k$ successes out of $N$ trials, its integral over $\\theta \\in [0, 1]$ is:
$$\\int_0^1 \\theta^k (1 - \\theta)^{N - k} d\\theta = \\frac{k! (N - k)!}{(N + 1)!} = \\frac{1}{(N + 1) \\binom{N}{k}} \\neq 1$$
""",
    """## 3. Функция правдоподобия vs Плотность вероятности: математическая дуальность

### Идея
Плотность вероятности $p(y \\mid \\theta)$ интегрируется в 1 по пространству данных $y$ (при фиксированном параметре $\\theta$). Напротив, **функция правдоподобия** $\\mathcal{L}(\\theta \\mid y)$ рассматривается как функция от параметров $\\theta$ (при фиксированных данных $y$). Принципиально то, что функция правдоподобия **не является** плотностью вероятности: для биномиального наблюдения $k$ успехов из $N$ испытаний ее интеграл по $\\theta \\in [0, 1]$ равен:
$$\\int_0^1 \\theta^k (1 - \\theta)^{N - k} d\\theta = \\frac{1}{(N + 1) \\binom{N}{k}} \\neq 1$$
""",
)

code(
    """# Observation: 7 successes out of 10 trials
k, n = 7, 10
theta_grid = np.linspace(0.0, 1.0, 1000)

# Likelihood function: L(theta | k=7, n=10)
likelihood_curve = stats.binom.pmf(k, n, theta_grid)
area_under_likelihood = integrate.trapezoid(likelihood_curve, theta_grid)

# Theoretical area: 1 / (N + 1) = 1 / 11
theoretical_area = 1.0 / (n + 1)

# Normalized posterior under a uniform prior Beta(1, 1) -> Beta(8, 4)
posterior_density = stats.beta.pdf(theta_grid, 1 + k, 1 + n - k)
area_under_posterior = integrate.trapezoid(posterior_density, theta_grid)

print(f"Area under Likelihood Curve: {area_under_likelihood:.4f} (Exact theory: 1/{n+1} = {theoretical_area:.4f})")
print(f"Area under Posterior Curve:  {area_under_posterior:.4f}")

fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(theta_grid, likelihood_curve, color="#e07a5f", lw=2.5, label=f"Likelihood Curve (Area = {area_under_likelihood:.3f} != 1)")
ax.plot(theta_grid, posterior_density, color="#2b5c8f", lw=2.5, label=f"Posterior Density Beta(8, 4) (Area = {area_under_posterior:.3f} == 1)")
ax.axvline(k / n, color="gray", linestyle="--", alpha=0.7, label=f"MLE theta = {k/n:.2f}")
ax.set_xlabel("Parameter theta")
ax.set_ylabel("Density / Value")
ax.set_title("Likelihood Function Is NOT a Probability Density (Area = 0.091 vs. 1.000)")
ax.legend()
plt.tight_layout()
plt.show()
"""
)

md(
    """### What to Watch For
The area under the likelihood curve is $0.091 \\approx 1/11$. To convert a likelihood into a valid probability distribution over hypotheses, Bayes' theorem multiplies it by a prior distribution and divides by the marginal likelihood (normalizing constant).
""",
    """### На что обратить внимание
Площадь под кривой правдоподобия равна $0{,}091 \\approx 1/11$. Чтобы превратить функцию правдоподобия в полноценное вероятностное распределение параметров, теорема Байеса умножает ее на априорное распределение и делит на нормировочную константу (маргинальное правдоподобие).
""",
)

# ---------------------------------------------------------------------------
# Step 4: Sequential Updating & Order Invariance
# ---------------------------------------------------------------------------
md(
    """## 4. Sequential Updating and Order Invariance

### The Idea
Bayesian updating is commutative: whether we observe data points one-by-one sequentially or all at once in a single batch, the final posterior distribution is identical:
$$p(\\theta \\mid y_1, y_2) \\propto p(y_2 \\mid \\theta) \\cdot \\left[ p(y_1 \\mid \\theta) \\cdot p(\\theta) \\right]$$
Today's posterior becomes tomorrow's prior.
""",
    """## 4. Последовательное обновление и инвариантность к порядку

### Идея
Байесовское обновление коммутативно: обновляем ли мы постериор по одному наблюдению за раз или сразу всей пачкой, итоговое апостериорное распределение оказывается в точности одинаковым:
$$p(\\theta \\mid y_1, y_2) \\propto p(y_2 \\mid \\theta) \\cdot \\left[ p(y_1 \\mid \\theta) \\cdot p(\\theta) \\right]$$
Сегодняшнее апостериорное распределение становится завтрашним априорным.
""",
)

code(
    """# Sequential observations: 1 = success, 0 = failure
data_seq = [1, 0, 1, 1, 0, 1]

# Start with weakly informative Beta(2, 2)
a_prior, b_prior = 2.0, 2.0
a_seq, b_seq = a_prior, b_prior

theta_vals = np.linspace(0, 1, 300)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(theta_vals, stats.beta.pdf(theta_vals, a_prior, b_prior), color="gray", linestyle="--", label="Initial Prior Beta(2, 2)")

for step, y_obs in enumerate(data_seq, start=1):
    a_seq += y_obs
    b_seq += (1 - y_obs)
    ax.plot(theta_vals, stats.beta.pdf(theta_vals, a_seq, b_seq), alpha=0.6, lw=1.5, label=f"Step {step} (y={y_obs})")

# Batch update all at once
a_batch = a_prior + sum(data_seq)
b_batch = b_prior + len(data_seq) - sum(data_seq)
ax.plot(theta_vals, stats.beta.pdf(theta_vals, a_batch, b_batch), color="#2b5c8f", lw=3.0, linestyle=":", label="Batch Update Beta(6, 4)")

ax.set_xlabel("Parameter theta")
ax.set_ylabel("Density")
ax.set_title("Sequential Updating: Today's Posterior Is Tomorrow's Prior")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

print(f"Sequential final parameters: Beta({a_seq:.0f}, {b_seq:.0f})")
print(f"Batch final parameters:      Beta({a_batch:.0f}, {b_batch:.0f})")
"""
)

md(
    """### What to Watch For
The sequential posterior after 6 steps matches the batch posterior exactly: $\\text{Beta}(6, 4)$. Shuffling the order of observations (e.g., $[0, 0, 1, 1, 1, 1]$) would result in the identical final curve, reflecting the exchangeability assumption of independent trials.
""",
    """### На что обратить внимание
Последовательный постериор после 6 шагов абсолютно совпадает с пакетным постериором: $\\text{Beta}(6, 4)$. Если бы мы перемешали порядок наблюдений (например, $[0, 0, 1, 1, 1, 1]$), итоговая кривая осталась бы неизменной в силу предположения о перемешиваемости (exchangeability) независимых испытаний.
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
