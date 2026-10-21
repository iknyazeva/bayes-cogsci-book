"""Build the Lesson 7 student workbooks (English and Russian) from one source.

Run from the book root with the pymc_env environment:
    /opt/homebrew/Caskroom/miniforge/base/envs/pymc_env/bin/python scripts/build_lesson7_notebooks.py

Outputs:
    notebooks/colab/07_continuous_outcomes_robust_models.ipynb
    notebooks/colab/07_continuous_outcomes_robust_models_ru.ipynb
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
    "en": "07_continuous_outcomes_robust_models.ipynb",
    "ru": "07_continuous_outcomes_robust_models_ru.ipynb",
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
    f"""# Lesson 7: Continuous Outcomes, Robust Models & Two-Group Differences
### Student Laboratory Workbook
*Course: Bayesian Analysis of Empirical Data (2026)*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({GITHUB}{FILES['en']})
*Russian version: [{FILES['ru']}]({GITHUB}{FILES['ru']})*

In this laboratory, we investigate how to model continuous measurements without being misled by outliers, compare two independent groups with unequal variances, test practical equivalence using ROPE and HDI, and assess point hypotheses with the Savage–Dickey Bayes factor.
""",
    f"""# Занятие 7: Непрерывные исходы, робастные модели и сравнение двух групп
### Рабочая тетрадь студента
*Курс: Байесовский анализ эмпирических данных (2026)*

[![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)]({GITHUB}{FILES['ru']})
*English version: [{FILES['en']}]({GITHUB}{FILES['en']})*

В этой лабораторной работе мы разберем, как моделировать непрерывные данные устойчиво к выбросам, корректно сравнивать две группы с неравными дисперсиями, проверять гипотезы о практической значимости через ROPE и HDI, а также вычислять Байес-фактор методом Саваджа–Дики.
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
# Step 1: Outliers and Robust Modeling (Student-t)
# ---------------------------------------------------------------------------
md(
    """## 1. Outlier Sensitivity and Robust Modeling (Student-$t$)

### The Idea
The sample mean and Gaussian likelihood assume thin tails: observations far from the center have near-zero probability, so a single extreme outlier pulls the mean toward itself and inflates the standard deviation. Simon Newcomb's 1882 speed-of-light measurements contain two severe negative outliers ($-44$ and $-2$, while typical values cluster around $+28$). Instead of arbitrarily discarding data points, the Student-$t$ distribution has heavy tails governed by degrees of freedom $\\nu$. When $\\nu$ is small, the tails absorb extreme deviations, allowing the location $\\mu$ to reflect the true consensus of the measurements.
""",
    """## 1. Чувствительность к выбросам и робастное моделирование (Student-$t$)

### Идея
Выборочное среднее и нормальное правдоподобие предполагают легкие хвосты: точки вдали от центра имеют почти нулевую вероятность, поэтому даже один экстремальный выброс искажает оценку $\\mu$ и искусственно раздувает $\\sigma$. В измерениях скорости света Саймона Ньюкомба (1882 г.) есть два резких отрицательных выброса ($-44$ и $-2$ при типичных значениях около $+28$). Вместо того чтобы субъективно удалять эти наблюдения, мы используем распределение Стьюдента с параметром степеней свободы $\\nu$. При малых $\\nu$ тяжелые хвосты поглощают выбросы, и параметр положения $\\mu$ точно отражает основной массив данных.
""",
)

code(
    """# Simon Newcomb's 66 measurements of passage time deviations (nanoseconds)
newcomb_data = np.array([
    28, 26, 33, 24, 34, -44, 27, 16, 40, -2, 29, 22, 24, 21, 25, 30,
    23, 29, 31, 19, 24, 20, 36, 32, 25, 28, 25, 28, 28, 27, 30, 27,
    26, 33, 26, 32, 28, 26, 27, 16, 23, 25, 21, 40, 29, 28, 27, 24,
    26, 25, 24, 27, 28, 24, 27, 25, 28, 27, 26, 28, 29, 27, 24, 23, 25, 28
])

# 1. Normal Model (Sensitive baseline)
with pm.Model() as normal_model:
    mu = pm.Normal("mu", mu=25.0, sigma=15.0)
    sigma = pm.HalfNormal("sigma", sigma=15.0)
    pm.Normal("y_obs", mu=mu, sigma=sigma, observed=newcomb_data)
    idata_norm = pm.sample(draws=1000, tune=1000, chains=2, random_seed=RANDOM_SEED, progressbar=False)

# 2. Student-t Model (Robust alternative)
with pm.Model() as student_model:
    mu = pm.Normal("mu", mu=25.0, sigma=15.0)
    sigma = pm.HalfNormal("sigma", sigma=15.0)
    nu = pm.Exponential("nu_minus_one", lam=1/29.0) + 1.0  # nu >= 1
    pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma, observed=newcomb_data)
    idata_t = pm.sample(draws=1000, tune=1000, chains=2, random_seed=RANDOM_SEED, progressbar=False)

# Compare parameter summaries
res_norm = az.summary(idata_norm, var_names=["mu", "sigma"])
res_t = az.summary(idata_t, var_names=["mu", "sigma", "nu_minus_one"])

print("--- Gaussian Model ---")
print(res_norm)
print("\\n--- Robust Student-t Model ---")
print(res_t)
"""
)

md(
    """### What to Watch For
Under the Gaussian model, $\\mu$ is dragged down to $\\approx 26.2$ and $\\sigma$ explodes to $\\approx 10.7$. Under the robust Student-$t$ model, $\\mu$ moves back to $\\approx 27.5$ and the measurement scale drops to $\\sigma \\approx 4.9$. The degrees of freedom parameter $\\nu$ concentrates at low values ($\\approx 2.5$), confirming that the data strongly demand heavy tails.
""",
    """### На что обратить внимание
В нормальной модели выброс сместил оценку $\\mu$ вниз до $\\approx 26{,}2$, а масштаб разброса $\\sigma$ искусственно раздулся до $\\approx 10{,}7$. В робастной модели Стьюдента оценка $\\mu$ вернулась к $\\approx 27{,}5$, а $\\sigma$ снизилась более чем в два раза — до $\\approx 4{,}9$. Параметр $\\nu$ сконцентрировался в районе $2{,}5$, что прямо сигнализирует: данные несовместимы с нормальным распределением.
""",
)

# ---------------------------------------------------------------------------
# Step 2: Two-Group Comparison with Unequal Variances
# ---------------------------------------------------------------------------
md(
    """## 2. Comparing Two Groups with Unequal Variances (KidIQ)

### The Idea
Classical two-sample $t$-tests often force an assumption of equal variances (homoskedasticity). In psychological and cognitive data, groups that differ in mean frequently also differ in spread. In Bayesian modeling, we estimate separate parameters for each group ($\\mu_0, \\sigma_0$ and $\\mu_1, \\sigma_1$). We then inspect the full posterior distribution of the group difference $\\Delta\\mu = \\mu_1 - \\mu_0$ and Cohen's effect size $d$.
""",
    """## 2. Сравнение двух групп с неравными дисперсиями (KidIQ)

### Идея
Классический двухвыборочный $t$-критерий Стьюдента часто требует равенства дисперсий в группах, что на реальных данных почти никогда не соблюдается. В байесовском подходе мы просто оцениваем раздельные параметры для каждой группы ($\\mu_0, \\sigma_0$ и $\\mu_1, \\sigma_1$). Затем мы напрямую вычисляем апостериорное распределение разницы средних $\\Delta\\mu = \\mu_1 - \\mu_0$ и стандартизованный размер эффекта Коэна $d$.
""",
)

code(
    """df_kidiq = pd.read_csv(DATA_PATH)
y0 = df_kidiq[df_kidiq["mom_hs"] == 0]["kid_score"].values  # No HS diploma
y1 = df_kidiq[df_kidiq["mom_hs"] == 1]["kid_score"].values  # HS diploma

print(f"Group 0 (No HS): N = {len(y0)}, Mean = {y0.mean():.2f}, SD = {y0.std(ddof=1):.2f}")
print(f"Group 1 (HS):    N = {len(y1)}, Mean = {y1.mean():.2f}, SD = {y1.std(ddof=1):.2f}")

with pm.Model() as two_group_model:
    # Separate priors for each group
    mu = pm.Normal("mu", mu=100.0, sigma=20.0, shape=2)
    sigma = pm.HalfNormal("sigma", sigma=25.0, shape=2)
    
    # Likelihoods
    pm.Normal("y0_obs", mu=mu[0], sigma=sigma[0], observed=y0)
    pm.Normal("y1_obs", mu=mu[1], sigma=sigma[1], observed=y1)
    
    # Contrast: difference in means and Cohen's d
    diff_mu = pm.Deterministic("diff_mu", mu[1] - mu[0])
    pooled_sd = pm.math.sqrt((sigma[0]**2 + sigma[1]**2) / 2.0)
    pm.Deterministic("cohen_d", diff_mu / pooled_sd)
    
    idata_groups = pm.sample(draws=1000, tune=1000, chains=2, random_seed=RANDOM_SEED, progressbar=False)

summary_groups = az.summary(idata_groups, var_names=["diff_mu", "cohen_d", "mu", "sigma"])
print("\\n--- Posterior Estimates ---")
print(summary_groups)
"""
)

md(
    """### What to Watch For
The estimated difference $\\Delta\\mu$ has a posterior mean of $\\approx +11.8$ points with a $94\\%$ HDI of $[8.5, 15.0]$. The credible interval does not come close to zero. Cohen's $d$ averages $\\approx 0.58$, indicating a moderate-to-large effect. Notice also that the standard deviations are indeed unequal: $\\sigma_0 \\approx 21.0$ vs. $\\sigma_1 \\approx 19.8$.
""",
    """### На что обратить внимание
Апостериорное среднее разницы средних $\\Delta\\mu$ составляет $\\approx +11{,}8$ балла с $94\\%$-м HDI $[8{,}5; 15{,}0]$. Интервал надежности строго положителен и далек от нуля. Размер эффекта Коэна $d \\approx 0{,}58$ свидетельствует о среднем/крупном эффекте. Обратите внимание и на разницу в разбросе: $\\sigma_0 \\approx 21{,}0$ против $\\sigma_1 \\approx 19{,}8$.
""",
)

# ---------------------------------------------------------------------------
# Step 3: Practical Equivalence with ROPE & HDI
# ---------------------------------------------------------------------------
md(
    """## 3. Practical Equivalence with ROPE and HDI

### The Idea
Point hypothesis testing ($H_0: \\Delta\\mu = 0$) asks if the difference is precisely zero to an infinite number of decimal places — which is almost never true in real life. Instead, we establish a **Region of Practical Equivalence (ROPE)**, say $[-2.0, +2.0]$ IQ points. Any difference within this band is practically negligible. If the highest density interval (HDI) falls entirely outside the ROPE, we conclude the difference is practically meaningful.
""",
    """## 3. Практическая значимость: коридор ROPE и интервал HDI

### Идея
Проверка точечной гипотезы $H_0: \\Delta\\mu = 0$ проверяет, равна ли разница абсолютному нулю до бесконечного знака после запятой, что в реальном мире практически невозможно. Гораздо практичнее задать **область практической эквивалентности (ROPE)**, например $[-2{,}0; +2{,}0]$ балла IQ. Любое отклонение внутри этого коридора считается пренебрежимо малым. Если интервал HDI полностью лежит за пределами ROPE, эффект признается практически значимым.
""",
)

code(
    """diff_samples = idata_groups.posterior["diff_mu"].values.flatten()
rope_lower, rope_upper = -2.0, 2.0

# Probability of falling inside the ROPE
p_rope = np.mean((diff_samples >= rope_lower) & (diff_samples <= rope_upper))
hdi_95 = compute_hdi(diff_samples, prob=0.95)

fig, ax = plt.subplots(figsize=(8, 3.8))
ax.hist(diff_samples, bins=40, density=True, alpha=0.6, color="#2b5c8f", edgecolor="black")
ax.axvspan(rope_lower, rope_upper, color="#e07a5f", alpha=0.3, label=f"ROPE [{rope_lower}, {rope_upper}]")
ax.plot(hdi_95, [0.01, 0.01], color="black", lw=4, label=f"95% HDI [{hdi_95[0]:.1f}, {hdi_95[1]:.1f}]")
ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
ax.set_title(f"Posterior Group Difference Delta mu: P(in ROPE) = {p_rope:.1%}")
ax.set_xlabel("Difference in Kid Score (Points)")
ax.set_ylabel("Posterior Density")
ax.legend()
plt.tight_layout()
plt.show()
"""
)

md(
    """### What to Watch For
Because $P(\\Delta\\mu \\in \\text{ROPE}) = 0\\%$, and the entire $95\\%$ HDI $[8.6, 15.1]$ lies strictly to the right of $+2.0$, we can decisively reject practical equivalence. The educational attainment of the mother is associated with a practically substantial difference in child test scores.
""",
    """### На что обратить внимание
Поскольку $P(\\Delta\\mu \\in \\text{ROPE}) = 0\\%$, а весь $95\\%$-й HDI $[8{,}6; 15{,}1]$ находится строго правее границы $+2{,}0$, гипотеза о практической неразличимости групп решительно отвергается. Различие в баллах детей практически весомо.
""",
)

# ---------------------------------------------------------------------------
# Step 4: The Savage–Dickey Bayes Factor
# ---------------------------------------------------------------------------
md(
    """## 4. Point-Null Testing via the Savage–Dickey Density Ratio

### The Idea
The Savage–Dickey density ratio evaluates a point null hypothesis ($H_0: \\delta = 0$) without fitting two separate models. It compares the height of the posterior distribution at the null point to the height of the prior distribution at that same point:
$$\\text{BF}_{01} = \\frac{p(\\delta = 0 \\mid D)}{p(\\delta = 0)}, \\qquad \\text{BF}_{10} = \\frac{1}{\\text{BF}_{01}}$$
If the posterior has almost zero density at $\\delta = 0$ while the prior was broad, $\\text{BF}_{10}$ is overwhelmingly large, providing strong evidence against the null.
""",
    """## 4. Проверка точечной гипотезы методом Саваджа–Дики (Байес-фактор)

### Идея
Метод Саваджа–Дики позволяет проверить точечную нулевую гипотезу ($H_0: \\delta = 0$) без необходимости обучать две раздельные модели. Байес-фактор равен отношению высоты апостериорной плотности в точке нуля к высоте априорной плотности в той же точке:
$$\\text{BF}_{01} = \\frac{p(\\delta = 0 \\mid D)}{p(\\delta = 0)}, \\qquad \\text{BF}_{10} = \\frac{1}{\\text{BF}_{01}}$$
Если апостериорная плотность в нуле практически равна нулю, а априорная плотность была распределена широко, $\\text{BF}_{10}$ оказывается колоссальным, свидетельствуя против нулевой гипотезы.
""",
)

code(
    """# Standardized difference prior: delta ~ Normal(0, 1)
# Prior density at 0 is 1 / sqrt(2 * pi)
prior_density_at_0 = stats.norm.pdf(0.0, loc=0.0, scale=1.0)

# Estimate posterior density at 0 using Gaussian KDE
d_samples = idata_groups.posterior["cohen_d"].values.flatten()
kde = stats.gaussian_kde(d_samples)
post_density_at_0 = float(kde.evaluate(0.0)[0])

bf_01 = post_density_at_0 / prior_density_at_0
bf_10 = 1.0 / bf_01 if bf_01 > 0 else np.inf

print(f"Prior density at delta=0:     {prior_density_at_0:.4f}")
print(f"Posterior density at delta=0: {post_density_at_0:.6f}")
print(f"Bayes Factor BF_01 (in favor of H0): {bf_01:.6f}")
print(f"Bayes Factor BF_10 (against H0):     {bf_10:.1f}")
"""
)

md(
    """### What to Watch For
Because the posterior of Cohen's $d$ is centered around $+0.58$ and its lowest draws rarely dip below $+0.3$, the posterior density at $\\delta = 0$ is practically zero. Consequently, $\\text{BF}_{10} \\gg 1000$. Under Jeffreys' interpretive scale, this constitutes overwhelming evidence in favor of $H_1$ over $H_0$.
""",
    """### На что обратить внимание
Поскольку апостериорное распределение $d$ Коэна сосредоточено вокруг $+0{,}58$, а его левый край едва доходит до $+0{,}3$, апостериорная плотность в точке $\\delta = 0$ исчезающе мала. Как следствие, $\\text{BF}_{10} \\gg 1000$. По шкале Джеффриса это представляет собой неопровержимое свидетельство в пользу гипотезы о наличии эффекта ($H_1$).
""",
)

# ---------------------------------------------------------------------------
# Step 5: Conceptual Bridge to Linear Regression
# ---------------------------------------------------------------------------
md(
    """## 5. The Conceptual Bridge: Two Groups Is a Linear Regression

### The Idea
Comparing two groups is mathematically identical to a linear regression with a single binary dummy predictor ($x_i \\in \\{0, 1\\}$):
$$y_i = \\alpha + \\beta \\cdot x_i + \\epsilon_i$$
Here, $\\alpha$ is the mean of Group 0 (moms without high school), and $\\beta$ is the direct contrast $\\mu_1 - \\mu_0$. In Lesson 8, we will expand this linear predictor from a $0/1$ indicator to continuous measurements and interaction terms!
""",
    """## 5. Концептуальный мост: сравнение двух групп — это линейная регрессия

### Идея
Сравнение двух групп математически тождественно линейной регрессии с одним бинарным индикатором ($x_i \\in \\{0, 1\\}$):
$$y_i = \\alpha + \\beta \\cdot x_i + \\epsilon_i$$
Здесь интерцепт $\\alpha$ — это среднее группы 0 (матери без школьного диплома), а наклон $\\beta$ — это прямой контраст $\\mu_1 - \\mu_0$. На Занятии 8 мы сделаем следующий шаг: заменим бинарный индикатор непрерывным интеллектом и введем эффекты взаимодействия!
""",
)

code(
    """# Fitting the two-group problem as a linear regression
with pm.Model() as reg_bridge:
    alpha = pm.Normal("alpha", mu=100.0, sigma=20.0)      # Group 0 mean
    beta = pm.Normal("beta", mu=0.0, sigma=20.0)          # Contrast (Group 1 - Group 0)
    sigma = pm.HalfNormal("sigma", sigma=20.0)
    
    mu_i = alpha + beta * df_kidiq["mom_hs"].values
    pm.Normal("y_obs", mu=mu_i, sigma=sigma, observed=df_kidiq["kid_score"].values)
    
    idata_bridge = pm.sample(draws=1000, tune=1000, chains=2, random_seed=RANDOM_SEED, progressbar=False)

summary_bridge = az.summary(idata_bridge, var_names=["alpha", "beta", "sigma"])
print("--- Linear Regression Representation ---")
print(summary_bridge)
"""
)

md(
    """### What to Watch For
Notice that $\\alpha \\approx 77.5$ matches the mean of the reference group, and $\\beta \\approx 11.8$ exactly matches the difference in means $\\Delta\\mu$ obtained in Step 2. We have bridged group comparison to regression without altering the substantive conclusion.
""",
    """### На что обратить внимание
Обратите внимание: интерцепт $\\alpha \\approx 77{,}5$ в точности равен среднему базовой группы, а коэффициент $\\beta \\approx 11{,}8$ совпадает с разницей средних $\\Delta\\mu$ из Шага 2. Мы перешли от сравнения групп к регрессионной постановке без потери содержательного смысла.
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
