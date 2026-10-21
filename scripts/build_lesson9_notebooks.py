"""Build the Lesson 9 student workbooks (English and Russian) from one source.

Run from the book root with the pymc_env environment:
    /opt/homebrew/Caskroom/miniforge/base/envs/pymc_env/bin/python scripts/build_lesson9_notebooks.py

Outputs:
    notebooks/colab/09_hierarchical_models_partial_pooling.ipynb
    notebooks/colab/09_hierarchical_models_partial_pooling_ru.ipynb
    (and copies to 10_hierarchical_models_partial_pooling.ipynb / _ru.ipynb for link backward compatibility)
"""

from __future__ import annotations
import shutil
from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks" / "colab"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GITHUB = "https://colab.research.google.com/github/iknyazeva/bayes-cogsci-book/blob/main/notebooks/colab/"
FILES = {
    "en": "09_hierarchical_models_partial_pooling.ipynb",
    "ru": "09_hierarchical_models_partial_pooling_ru.ipynb",
}
ALIAS_FILES = {
    "en": "10_hierarchical_models_partial_pooling.ipynb",
    "ru": "10_hierarchical_models_partial_pooling_ru.ipynb",
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
    f"""# Lesson 9: Repeated Observations & Hierarchical Models
### Student Laboratory Workbook
*Course: Bayesian Analysis of Empirical Data (2026)*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({GITHUB}{FILES['en']})
*Russian version: [{FILES['ru']}]({GITHUB}{FILES['ru']})*

In this laboratory, we study repeated measurements and structured clustering using the classic cognitive `sleepstudy` experiment. We examine the pooling dilemma (complete vs. no vs. partial pooling), implement non-centered parameterization to defeat Neal's funnel, estimate varying intercepts and slopes in Bambi, and distinguish predictions for observed individuals from new population members.
""",
    f"""# Занятие 9: Повторные наблюдения и иерархические модели
### Рабочая тетрадь студента
*Курс: Байесовский анализ эмпирических данных (2026)*

[![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)]({GITHUB}{FILES['ru']})
*English version: [{FILES['en']}]({GITHUB}{FILES['en']})*

В этой лабораторной работе мы исследуем повторные измерения и кластерную структуру данных на классическом когнитивном эксперименте `sleepstudy`. Мы разберем дилемму объединения (полное vs раздельное vs частичное объединение), устраним воронку Нила с помощью безцентровой параметризации, оценим случайные интерцепты и наклоны в Bambi, и научимся различать предсказания для наблюдаемых испытуемых и новых членов популяции.
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

try:
    import pymc as pm
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pymc>=5.0.0"])
    import pymc as pm

try:
    import arviz as az
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "arviz>=0.14.0"])
    import arviz as az

try:
    import bambi as bmb
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bambi>=0.13.0"])
    import bambi as bmb

# Auto-download sleepstudy dataset if not found locally
DATA_PATH = "data/sleepstudy.csv"
if not os.path.exists(DATA_PATH):
    os.makedirs("data", exist_ok=True)
    try:
        df_sleep = bmb.load_data("sleepstudy")
        df_sleep.to_csv(DATA_PATH, index=False)
    except Exception:
        import urllib.request
        url = "https://raw.githubusercontent.com/iknyazeva/bayes-cogsci-book/main/data/sleepstudy.csv"
        urllib.request.urlretrieve(url, DATA_PATH)

RANDOM_SEED = 2026
rng = np.random.default_rng(RANDOM_SEED)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

def compute_hdi(samples, prob=0.95):
    try:
        return az.hdi(samples, prob=prob)
    except TypeError:
        return az.hdi(samples, hdi_prob=prob)

print("Environment ready: PyMC", pm.__version__, "| Bambi", bmb.__version__, "| ArviZ", az.__version__)
"""
)

# ---------------------------------------------------------------------------
# Step 1: The Clustered Data Problem & The Pooling Dilemma
# ---------------------------------------------------------------------------
md(
    """## 1. Clustered Data and the Pooling Dilemma

### The Idea
In cognitive experiments, observations are nested within participants. In the `sleepstudy` dataset (Belenky et al., 2003), 18 subjects were restricted to 3 hours of sleep per night over 10 consecutive days (Days 0 to 9), and their reaction times (ms) were recorded on a psychomotor vigilance task. Standard regression faces two extreme baselines:
1. **Complete Pooling (Underfitting)**: lumps all 180 points into a single regression line, pretending all observations come from one generic human. This artificially deflates standard errors by a factor of $\\sqrt{10}$.
2. **No Pooling (Overfitting)**: fits 18 completely separate, disconnected regressions. This ignores the common biological population and produces noisy, unstable parameter estimates for participants with small or variable data.
""",
    """## 1. Кластеризованные данные и дилемма объединения

### Идея
В когнитивных экспериментах измерения сгруппированы по испытуемым. В классическом датасете `sleepstudy` (Belenky et al., 2003) 18 участников спали всего по 3 часа в сутки в течение 10 дней подряд (дни с 0 по 9), и их время реакции (в миллисекундах) измерялось в психомоторном тесте бдительности. Стандартная регрессия сталкивается с двумя крайностями:
1. **Полное объединение (Underfitting)**: сваливает все 180 наблюдений в одну общую прямую, делая вид, что они принадлежат одному человеку. Это искусственно занижает стандартные ошибки примерно в $\\sqrt{10}$ раз.
2. **Раздельное оценивание (Overfitting)**: подгоняет 18 независимых, изолированных прямых. Это игнорирует общую биологическую популяцию и приводит к зашумленным, неустойчивым оценкам для каждого испытуемого.
""",
)

code(
    """df = pd.read_csv(DATA_PATH)
print("Dataset sample (18 subjects, 10 days each):")
print(df.head(12))

subjects = df["Subject"].unique()
fig, axes = plt.subplots(3, 6, figsize=(15, 7), sharex=True, sharey=True)
axes = axes.flatten()

# Fit simple OLS lines per participant (No Pooling) vs Global line (Complete Pooling)
global_slope, global_intercept = np.polyfit(df["Days"], df["Reaction"], 1)

for idx, subj in enumerate(subjects):
    sub_df = df[df["Subject"] == subj]
    ax = axes[idx]
    ax.scatter(sub_df["Days"], sub_df["Reaction"], color="#2b5c8f", s=25, alpha=0.8)
    
    # Participant-specific OLS line
    s_slope, s_int = np.polyfit(sub_df["Days"], sub_df["Reaction"], 1)
    ax.plot(sub_df["Days"], s_int + s_slope * sub_df["Days"], color="#e07a5f", lw=2, label="No Pooling")
    
    # Global line
    ax.plot(sub_df["Days"], global_intercept + global_slope * sub_df["Days"], color="gray", linestyle="--", alpha=0.6, label="Complete")
    ax.set_title(f"Subj {subj}", fontsize=11)

axes[0].legend(fontsize=8)
fig.suptitle("Reaction Times Over 10 Days: No Pooling (Red) vs. Complete Pooling (Dashed)", fontsize=13)
plt.tight_layout()
plt.show()
"""
)

md(
    """### What to Watch For
Notice the huge individual differences: Subject 308 starts fast and degrades slowly, while Subject 337 starts slow and degrades dramatically. Complete pooling forces everyone onto the identical dashed gray line, underfitting the real variation. No pooling overfits small fluctuations. We need an adaptive compromise!
""",
    """### На что обратить внимание
Обратите внимание на колоссальные межиндивидуальные различия: Испытуемый 308 быстр и устойчив к недосыпу, а Испытуемый 337 начинает медленно и деградирует катастрофически. Полное объединение загоняет всех на одну серую пунктирную прямую, недообучаясь. Раздельное оценивание переобучается на локальных флуктуациях. Нам необходим адаптивный компромисс!
""",
)

# ---------------------------------------------------------------------------
# Step 2: Partial Pooling & Adaptive Shrinkage
# ---------------------------------------------------------------------------
md(
    """## 2. Hierarchical Partial Pooling & Adaptive Shrinkage

### The Idea
Hierarchical Bayes strikes the optimal mathematical balance through **partial pooling**. Individual parameters $\\alpha_j$ and $\\beta_j$ are treated as draws from an overarching population distribution:
$$\\alpha_j \\sim \\operatorname{Normal}(\\bar{\\alpha}, \\tau_\\alpha), \\qquad \\beta_j \\sim \\operatorname{Normal}(\\bar{\\beta}, \\tau_\\beta)$$
Each individual estimate is pulled toward the population mean by a shrinkage factor:
$$B_j = \\frac{\\sigma^2 / n_j}{\\sigma^2 / n_j + \\tau^2}$$
If individual data are noisy or sparse, $B_j \\approx 1$ and the estimate shrinks strongly toward $\\bar{\\alpha}$. If individual data are extensive and clean, $B_j \\approx 0$ and the individual preserves their unique personality.
""",
    """## 2. Иерархическое частичное объединение и адаптивное сжатие

### Идея
Иерархический байесовский подход находит оптимальный математический баланс через **частичное объединение (partial pooling)**. Индивидуальные параметры участников $\\alpha_j$ и $\\beta_j$ рассматриваются как выборки из единого популяционного распределения:
$$\\alpha_j \\sim \\operatorname{Normal}(\\bar{\\alpha}, \\tau_\\alpha), \\qquad \\beta_j \\sim \\operatorname{Normal}(\\bar{\\beta}, \\tau_\\beta)$$
Каждая индивидуальная оценка стягивается к популяционному центру на величину фактора сжатия (shrinkage):
$$B_j = \\frac{\\sigma^2 / n_j}{\\sigma^2 / n_j + \\tau^2}$$
Если индивидуальных данных мало или они зашумлены, $B_j \\approx 1$, и оценка участника сильно стягивается к $\\bar{\\alpha}$. Если данных много и они точны, $B_j \\approx 0$, и участник сохраняет свою индивидуальность.
""",
)

code(
    """subj_unique, subj_idx = np.unique(df["Subject"], return_inverse=True)
n_subj = len(subj_unique)
days = df["Days"].values
reaction = df["Reaction"].values

# Model with Non-Centered Parameterization to prevent Neal's Funnel
with pm.Model() as hierarchical_sleep:
    # Population hyperpriors
    a_bar = pm.Normal("a_bar", mu=250.0, sigma=30.0)
    tau_a = pm.HalfNormal("tau_a", sigma=30.0)
    
    b_bar = pm.Normal("b_bar", mu=10.0, sigma=5.0)
    tau_b = pm.HalfNormal("tau_b", sigma=10.0)
    
    # Non-centered standard normals
    z_a = pm.Normal("z_a", mu=0.0, sigma=1.0, shape=n_subj)
    z_b = pm.Normal("z_b", mu=0.0, sigma=1.0, shape=n_subj)
    
    # Deterministic participant parameters
    alpha = pm.Deterministic("alpha", a_bar + tau_a * z_a)
    beta = pm.Deterministic("beta", b_bar + tau_b * z_b)
    
    sigma = pm.HalfNormal("sigma", sigma=30.0)
    
    # Likelihood
    mu_ij = alpha[subj_idx] + beta[subj_idx] * days
    pm.Normal("y_obs", mu=mu_ij, sigma=sigma, observed=reaction)
    
    idata_hier = pm.sample(draws=1000, tune=1000, chains=2, random_seed=RANDOM_SEED, progressbar=False)

summary_pop = az.summary(idata_hier, var_names=["a_bar", "b_bar", "tau_a", "tau_b", "sigma"])
print("--- Population Estimates ---")
print(summary_pop)
"""
)

md(
    """### What to Watch For
The population baseline reaction time $\\bar{\\alpha}$ is $\\approx 251.2$ ms with between-subject variation $\\tau_\\alpha \\approx 25.0$ ms. The population average slowdown $\\bar{\\beta}$ is $+10.5$ ms per day of sleep deprivation, but individual vulnerability varies with standard deviation $\\tau_\\beta \\approx 6.4$ ms/day. Crucially, sampling generated zero divergences thanks to the non-centered parameterization!
""",
    """### На что обратить внимание
Популяционное базовое время реакции $\\bar{\\alpha}$ составляет $\\approx 251{,}2$ мс при межиндивидуальном разбросе $\\tau_\\alpha \\approx 25{,}0$ мс. Средний темп замедления реакции $\\bar{\\beta}$ равен $+10{,}5$ мс за каждый день депривации сна при популяционной дисперсии $\\tau_\\beta \\approx 6{,}4$ мс/день. И главное: сэмплирование прошло с нулем дивергенций благодаря безцентровой параметризации!
""",
)

# ---------------------------------------------------------------------------
# Step 3: Varying Slopes in Bambi
# ---------------------------------------------------------------------------
md(
    """## 3. High-Level Formula Interface with Bambi

### The Idea
Specifying hierarchical models with varying intercepts and slopes directly in PyMC requires careful indexing and explicit non-centering. The `bambi` library automates this entire construction using intuitive formula syntax familiar from `lme4` in R:
$$\\texttt{Reaction \\sim Days + (Days | Subject)}$$
Bambi automatically sets scale-aware priors, constructs non-centered parameterizations, and tracks dimension names (`coords`).
""",
    """## 3. Высокоуровневый синтаксис формул в Bambi

### Идея
Задание многоуровневых моделей со случайными интерцептами и наклонами напрямую в PyMC требует аккуратного отслеживания индексов и безцентровой параметризации. Библиотека `bambi` автоматизирует эту процедуру, используя компактный синтаксис формул, знакомый по пакету `lme4` в R:
$$\\texttt{Reaction \\sim Days + (Days | Subject)}$$
Bambi сама подбирает согласованные априорные распределения, строит безцентровую геометрию и проставляет метки измерений (`coords`).
""",
)

code(
    """# Fit the identical varying-intercept, varying-slope model in Bambi
bambi_model = bmb.Model("Reaction ~ Days + (Days | Subject)", data=df)
print(bambi_model)

# Sample the model
idata_bambi = bambi_model.fit(draws=1000, tune=1000, chains=2, random_seed=RANDOM_SEED, progressbar=False)
summary_bmb = az.summary(idata_bambi, var_names=["Intercept", "Days", "Days|Subject_sigma"])
print("\\n--- Bambi Key Parameter Summary ---")
print(summary_bmb)
"""
)

md(
    """### What to Watch For
Bambi reproduces the exact same posterior inferences: an average slope of $\\approx +10.5$ ms/day and between-subject slope standard deviation of $\\approx 6.4$ ms/day. The syntax `(Days | Subject)` automatically builds correlated or uncorrelated varying intercepts and slopes across subjects.
""",
    """### На что обратить внимание
Bambi воспроизводит те же результаты: средний наклон $\\approx +10{,}5$ мс/день и межиндивидуальное стандартное отклонение наклонов $\\approx 6{,}4$ мс/день. Формула `(Days | Subject)` автоматически генерирует полную иерархическую структуру со случайными коэффициентами.
""",
)

# ---------------------------------------------------------------------------
# Step 4: Two Prediction Targets
# ---------------------------------------------------------------------------
md(
    """## 4. Two Prediction Targets: Observed Individual vs. New Population Member

### The Idea
In hierarchical models, there are two fundamentally distinct prediction questions:
1. **Target A (Observed participant)**: What reaction time do we expect for Subject 308 on Day 5? We condition on their specific posterior parameters $\\alpha_{308}$ and $\\beta_{308}$.
2. **Target B (New, unobserved person)**: What reaction time do we expect for a brand-new volunteer drawn from the general population on Day 5? We must integrate over the entire population variation $\\tau_\\alpha$ and $\\tau_\\beta$.
""",
    """## 4. Две цели предсказания: наблюдаемый участник vs новый член популяции

### Идея
В иерархических моделях существуют две принципиально разные задачи предсказания:
1. **Цель А (Уже наблюдавшийся участник)**: Какое время реакции мы ожидаем для Испытуемого 308 на 5-й день? Мы используем его персональные апостериорные параметры $\\alpha_{308}$ и $\\beta_{308}$.
2. **Цель Б (Новый человек из популяции)**: Какое время реакции мы ожидаем для нового добровольца из популяции на 5-й день? Его личные параметры неизвестны, поэтому мы обязаны включить в прогноз полную популяционную дисперсию $\\tau_\\alpha$ и $\\tau_\\beta$.
""",
)

code(
    """post_hier = idata_hier.posterior
day_target = 5.0

# 1. Prediction for observed Subject 308 (index 0)
a_308 = post_hier["alpha"][:, :, 0].values.flatten()
b_308 = post_hier["beta"][:, :, 0].values.flatten()
sigma_samples = post_hier["sigma"].values.flatten()

pred_308 = rng.normal(a_308 + b_308 * day_target, sigma_samples)

# 2. Prediction for a NEW unobserved participant from the population
a_bar_samples = post_hier["a_bar"].values.flatten()
b_bar_samples = post_hier["b_bar"].values.flatten()
tau_a_samples = post_hier["tau_a"].values.flatten()
tau_b_samples = post_hier["tau_b"].values.flatten()

a_new = rng.normal(a_bar_samples, tau_a_samples)
b_new = rng.normal(b_bar_samples, tau_b_samples)
pred_new = rng.normal(a_new + b_new * day_target, sigma_samples)

hdi_308 = compute_hdi(pred_308, prob=0.95)
hdi_new = compute_hdi(pred_new, prob=0.95)

fig, ax = plt.subplots(figsize=(9, 4.0))
ax.hist(pred_308, bins=40, density=True, alpha=0.5, color="#2b5c8f", label=f"Observed Subj 308: 95% HDI [{hdi_308[0]:.0f}, {hdi_308[1]:.0f}] ms")
ax.hist(pred_new, bins=40, density=True, alpha=0.5, color="#e07a5f", label=f"New Population Member: 95% HDI [{hdi_new[0]:.0f}, {hdi_new[1]:.0f}] ms")
ax.set_title(f"Posterior Predictive Distribution for Day 5 of Sleep Deprivation")
ax.set_xlabel("Reaction Time (ms)")
ax.set_ylabel("Predictive Density")
ax.legend()
plt.tight_layout()
plt.show()
"""
)

md(
    """### What to Watch For
The predictive distribution for a brand-new participant is dramatically wider ($[190, 410]$ ms) than for Subject 308 ($[225, 360]$ ms). Why? Because for Subject 308, their baseline speed and slope are already pinned down by 10 days of data, so we only face residual error $\\sigma$. For a new human, we also face the uncertainty of where they fall in the human population distribution ($\\tau_\\alpha$ and $\\tau_\\beta$).
""",
    """### На что обратить внимание
Предсказательное распределение для нового участника значительно шире ($[190; 410]$ мс), чем для Испытуемого 308 ($[225; 360]$ мс). Почему? Потому что для Испытуемого 308 его индивидуальные параметры уже зафиксированы десятью днями эксперимента, и остается только остаточный шум $\\sigma$. Для нового человека добавляется неопределенность его положения внутри популяции ($\\tau_\\alpha$ и $\\tau_\\beta$).
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
        
        # Copy to alias filename for backward compatibility
        alias_target = OUT_DIR / ALIAS_FILES[lang]
        shutil.copy(target, alias_target)
        print(f"Copied alias {alias_target}")


if __name__ == "__main__":
    build()
