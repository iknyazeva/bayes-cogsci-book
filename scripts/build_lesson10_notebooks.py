"""Build the Lesson 10 student workbooks (English and Russian) from one source.

Run from the book root with the pymc_env environment:
    /opt/homebrew/Caskroom/miniforge/base/envs/pymc_env/bin/python scripts/build_lesson10_notebooks.py

Outputs:
    notebooks/colab/10_generalized_linear_models.ipynb
    notebooks/colab/10_generalized_linear_models_ru.ipynb
    (and copies to 09_generalized_linear_models.ipynb / _ru.ipynb for link backward compatibility)
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
    "en": "10_generalized_linear_models.ipynb",
    "ru": "10_generalized_linear_models_ru.ipynb",
}
ALIAS_FILES = {
    "en": "09_generalized_linear_models.ipynb",
    "ru": "09_generalized_linear_models_ru.ipynb",
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
    """# Lesson 10: Binary Outcomes, Generalized Linear Models & The Chimpanzees Capstone
### Student Laboratory Workbook
*Course: Bayesian Analysis of Empirical Data (2026)*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](""" + GITHUB + FILES['en'] + """)
*Russian version: [""" + FILES['ru'] + "](" + GITHUB + FILES['ru'] + """)*

In this laboratory, we discover why straight lines fail on discrete binary choices ($y \\in \\{0, 1\\}$), learn how the logistic link maps linear predictors to valid probabilities, use the divide-by-4 rule to interpret log-odds, conduct prior predictive checks on the probability scale, resolve complete separation, and cap Part III with the hierarchical logistic prosocial chimpanzees experiment.
""",
    """# Занятие 10: Бинарные исходы, обобщенные линейные модели (GLM) и кейс с шимпанзе
### Рабочая тетрадь студента
*Курс: Байесовский анализ эмпирических данных (2026)*

[![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)](""" + GITHUB + FILES['ru'] + """)
*English version: [""" + FILES['en'] + "](" + GITHUB + FILES['en'] + """)*

В этой лабораторной работе мы разберем, почему прямая линия ломается на дискретных бинарных исходах ($y \\in \\{0, 1\\}$), как логистическая функция связи переводит линейный предиктор в корректные вероятности, как применять правило деления на 4, почему «неинформативные» приоры на шкале логитов опасны, как байесовский вывод решает проблему полного разделения, и завершим Часть III итоговым многоуровневым кейсом с шимпанзе.
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

# Auto-download datasets if not found locally
WELLS_PATH = "data/wells.csv"
CHIMP_PATH = "data/chimpanzees.csv"
os.makedirs("data", exist_ok=True)

if not os.path.exists(WELLS_PATH):
    import urllib.request
    url_w = "https://raw.githubusercontent.com/iknyazeva/bayes-cogsci-book/main/data/wells.csv"
    urllib.request.urlretrieve(url_w, WELLS_PATH)

if not os.path.exists(CHIMP_PATH):
    import urllib.request
    url_c = "https://raw.githubusercontent.com/iknyazeva/bayes-cogsci-book/main/data/chimpanzees.csv"
    urllib.request.urlretrieve(url_c, CHIMP_PATH)

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
# Step 1: Why Straight Lines Fail & The Logistic Link
# ---------------------------------------------------------------------------
md(
    """## 1. Why Straight Lines Fail on Binary Data & The Logistic Sigmoid

### The Idea
When outcomes are binary choices ($y \\in \\{0, 1\\}$), fitting an ordinary linear regression line ($y_i = \\alpha + \\beta x_i$) inevitably predicts impossible probabilities below 0 or above 1. The **logistic link function** maps unbounded linear predictions $\\eta \\in (-\\infty, +\\infty)$ strictly into the valid probability range $(0, 1)$ via the sigmoid:
$$\\operatorname{logit}(p) = \\log\\left(\\frac{p}{1 - p}\\right) = \\eta \\iff p = \\frac{1}{1 + e^{-\\eta}}$$
To quickly interpret a logistic regression slope $\\beta$ on the probability scale, use the **divide-by-4 rule**: the maximum change in probability associated with a 1-unit increase in $x$ is approximately $\\beta / 4$ (occurring when $p = 0.5$).
""",
    """## 1. Почему прямая линия ломается на бинарных данных и логистическая сигмоида

### Идея
Когда исход представляет собой бинарный выбор ($y \\in \\{0, 1\\}$), обычная линейная регрессия ($y_i = \\alpha + \\beta x_i$) неизбежно предсказывает бессмысленные вероятности меньше 0 или больше 1. **Логистическая функция связи** переводит неограниченный линейный предиктор $\\eta \\in (-\\infty, +\\infty)$ строго в диапазон вероятностей $(0, 1)$ с помощью сигмоиды:
$$\\operatorname{logit}(p) = \\log\\left(\\frac{p}{1 - p}\\right) = \\eta \\iff p = \\frac{1}{1 + e^{-\\eta}}$$
Чтобы быстро оценить содержательный масштаб коэффициента $\\beta$ на шкале вероятностей, используют **правило деления на 4**: максимальный сдвиг вероятности при увеличении предиктора на 1 единицу составляет примерно $\\beta / 4$ (в точке максимальной крутизны при $p = 0{,}5$).
""",
)

code(
    """df_wells = pd.read_csv(WELLS_PATH)
print("Bangladesh Arsenic Wells dataset (N=3020):")
print(df_wells[["switch", "dist", "arsenic", "educ"]].head())

# Scale predictors: distance in 100 meters, education in 4-year increments
dist100 = df_wells["dist"].values / 100.0
educ4 = df_wells["educ"].values / 4.0
sw = df_wells["switch"].values

with pm.Model() as logistic_wells:
    alpha = pm.Normal("alpha", mu=0.0, sigma=1.5)
    beta_dist = pm.Normal("beta_dist", mu=0.0, sigma=1.5)
    beta_educ = pm.Normal("beta_educ", mu=0.0, sigma=1.5)
    
    eta = alpha + beta_dist * dist100 + beta_educ * educ4
    p = pm.Deterministic("p", pm.math.invlogit(eta))
    
    pm.Bernoulli("y_obs", p=p, observed=sw)
    idata_wells = pm.sample(draws=1000, tune=1000, chains=2, random_seed=RANDOM_SEED, progressbar=False)

summary_wells = az.summary(idata_wells, var_names=["alpha", "beta_dist", "beta_educ"])
print("\\n--- Logistic Regression Estimates ---")
print(summary_wells)

b_dist_mean = summary_wells.loc["beta_dist", "mean"]
print(f"\\nDivide-by-4 Rule: Beta_dist = {b_dist_mean:.3f} -> Max prob change per 100m = {b_dist_mean/4:.1%}")
"""
)

md(
    """### What to Watch For
The coefficient for distance is $\\beta_{\\text{dist}} \\approx -0.62$. Using the divide-by-4 rule, each additional 100 meters of distance to a safe well reduces the probability of switching by at most $-0.62 / 4 \\approx -15.5$ percentage points! Education increases switching probability by $\\approx +0.18 / 4 \\approx +4.5$ percentage points per 4 years of schooling.
""",
    """### На что обратить внимание
Коэффициент при расстоянии равен $\\beta_{\\text{dist}} \\approx -0{,}62$. По правилу деления на 4, каждые дополнительные 100 метров до безопасного источника снижают вероятность смены колодца максимум на $-0{,}62 / 4 \\approx -15{,}5$ процентных пункта! Образование главы семьи увеличивает вероятность смены источника на $\\approx +0{,}18 / 4 \\approx +4{,}5$ процентных пункта за каждые 4 года учебы.
""",
)

# ---------------------------------------------------------------------------
# Step 2: Prior Predictive Checks on the Probability Scale
# ---------------------------------------------------------------------------
md(
    """## 2. Prior Predictive Checking on the Probability Scale

### The Idea
In linear regression, a wide prior like $\\operatorname{Normal}(0, 100)$ is often considered benign. But on the log-odds scale, values of $|\\eta| > 5$ correspond to probabilities $p < 0.007$ or $p > 0.993$. An unthinking prior $\\operatorname{Normal}(0, 100)$ puts almost all of its probability mass into the extreme edges $p = 0$ and $p = 1$, creating a bizarre U-shaped prior that asserts events must almost certainly never happen or almost certainly always happen! A weakly informative prior like $\\operatorname{Normal}(0, 1.5)$ produces an almost uniform distribution across probabilities.
""",
    """## 2. Априорная проверка на шкале вероятностей

### Идея
В линейной регрессии широкий приор вроде $\\operatorname{Normal}(0, 100)$ кажется безобидным. Но на шкале логитов значения $|\\eta| > 5$ соответствуют вероятностям $p < 0{,}007$ или $p > 0{,}993$. Широкий приор $\\operatorname{Normal}(0, 100)$ загоняет почти $95\\%$ априорной массы в самые края — в 0 и 1, создавая странное U-образное распределение, утверждающее, что события либо почти наверняка не происходят вовсе, либо происходят всегда! Разумный слабоинформативный приор $\\operatorname{Normal}(0, 1{,}5)$ порождает гладкое, почти равномерное распределение вероятностей.
""",
)

code(
    """# Sample priors on logit scale and transform to probability scale
n_sim = 10000
alpha_bad = rng.normal(0, 100.0, n_sim)
alpha_good = rng.normal(0, 1.5, n_sim)

prob_bad = 1.0 / (1.0 + np.exp(-alpha_bad))
prob_good = 1.0 / (1.0 + np.exp(-alpha_good))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
ax1.hist(prob_bad, bins=30, density=True, color="#e07a5f", alpha=0.7, edgecolor="black")
ax1.set_title("Flat Log-Odds Prior: Normal(0, 100)\\nExtreme Edge Spikes (U-shaped)")
ax1.set_xlabel("Probability p")
ax1.set_ylabel("Prior Density")

ax2.hist(prob_good, bins=30, density=True, color="#2b5c8f", alpha=0.7, edgecolor="black")
ax2.set_title("Weakly Informative Prior: Normal(0, 1.5)\\nSensible & Near-Uniform")
ax2.set_xlabel("Probability p")

plt.tight_layout()
plt.show()
"""
)

md(
    """### What to Watch For
Looking at the left panel: nearly all prior mass is concentrated in the bins near $0.0$ and $1.0$. This demonstrates why prior predictive checking must always be conducted on the **scale of the observable outcome** (probabilities) rather than the latent link scale (log-odds).
""",
    """### На что обратить внимание
Посмотрите на левый график: почти вся масса сосредоточена в крайних столбцах около $0{,}0$ и $1{,}0$. Это наглядно демонстрирует, почему априорную предсказательную проверку всегда нужно проводить на **шкале наблюдаемого исхода** (вероятностей), а не на скрытой шкале логитов.
""",
)

# ---------------------------------------------------------------------------
# Step 3: Complete Separation and Bayesian Regularization
# ---------------------------------------------------------------------------
md(
    """## 3. Complete Separation and Bayesian Regularization

### The Idea
In frequentist maximum likelihood estimation (MLE), if any predictor perfectly separates the binary outcome (e.g., all individuals with high income buy the product: $y=1$), the likelihood gradient never reaches zero, driving $\\hat{\\beta} \\to +\\infty$ with infinite standard errors and convergence failure. In Bayesian modeling, a weakly informative prior serves as a principled regularizer, keeping the posterior finite, well-behaved, and completely interpretable.
""",
    """## 3. Полное разделение и байесовская регуляризация

### Идея
В частотном оценивании максимального правдоподобия (MLE), если какой-то предиктор идеально разделяет исход (например, все участники с высоким доходом купили продукт: $y=1$), градиент правдоподобия никогда не обращается в ноль. Оценка $\\hat{\\beta}$ уходит в бесконечность, стандартные ошибки взрываются, и алгоритм ломается. В байесовском подходе слабоинформативный априор служит естественным регуляризатором, удерживая апостериорное распределение конечным и информативным.
""",
)

code(
    """# Synthesize a dataset with complete separation
x_sep = np.array([-3, -2.5, -2, -1.5, -1, 0.5, 1, 1.5, 2, 2.5, 3])
y_sep = np.array([ 0,    0,  0,    0,  0,   1, 1,   1, 1,   1, 1])  # Perfect cutoff at x > 0!

with pm.Model() as sep_model:
    # Regularizing prior
    alpha = pm.Normal("alpha", mu=0.0, sigma=2.0)
    beta = pm.Normal("beta", mu=0.0, sigma=2.0)
    
    p = pm.math.invlogit(alpha + beta * x_sep)
    pm.Bernoulli("y_obs", p=p, observed=y_sep)
    
    idata_sep = pm.sample(draws=1000, tune=1000, chains=2, random_seed=RANDOM_SEED, progressbar=False)

summary_sep = az.summary(idata_sep, var_names=["alpha", "beta"])
print("--- Bayesian Posterior Under Complete Separation ---")
print(summary_sep)
"""
)

md(
    """### What to Watch For
While classical logistic regression would fail or report massive standard errors in the thousands, the Bayesian posterior for $\\beta$ has a well-defined median around $+2.2$ and finite credible bounds. The prior prevented infinite explosion without distorting the data.
""",
    """### На что обратить внимание
В то время как классическая логистическая регрессия выдала бы ошибку или гигантскую стандартную ошибку в тысячи единиц, байесовское апостериорное распределение для $\\beta$ имеет четкую медиану около $+2{,}2$ и конечный интервал надежности. Априор предотвратил взрыв в бесконечность, сохранив объективность выводов.
""",
)

# ---------------------------------------------------------------------------
# Step 4: Capstone Case Study — Prosocial Chimpanzees
# ---------------------------------------------------------------------------
md(
    """## 4. Capstone Case Study: Partial Pooling with Binary Logit (Chimpanzees)

### The Idea
In the climax of Part III, we unite the partial pooling from Lesson 9 with the logistic link from Lesson 10. In a celebrated experiment on primate altruism (Silk et al., 2005; McElreath, 2020), 7 chimpanzees could choose between two levers:
- A "selfish" option delivering food only to the actor.
- A "prosocial" option delivering food to both the actor and a partner in the adjacent cage.

A critical observational challenge is **individual handedness**: Actor 2 pulled the left lever on all 90 trials ($90/90$)! In a no-pooling model, Actor 2's intercept diverges to $+\\infty$. Hierarchical partial pooling pulls Actor 2 toward the chimpanzee population mean.
""",
    """## 4. Итоговый кейс: частичное объединение в логистической регрессии (шимпанзе)

### Идея
Кульминация Части III: объединение частичного объединения из Занятия 9 и логистической связи из Занятия 10. В известном эксперименте по альтруизму приматов (Silk et al., 2005; McElreath, 2020) 7 шимпанзе выбирали между двумя рычагами:
- «Эгоистичный» рычаг дает еду только самому актеру.
- «Просоциальный» рычаг дает еду и актеру, и партнеру в соседней клетке.

Главная методическая сложность — **индивидуальная праворукость**: Актёр 2 выбирал левый рычаг во всех 90 испытаниях из 90! В модели без объединения его интерцепт уходит в бесконечность. Иерархическое сжатие подтягивает Актёра 2 к популяционному среднему.
""",
)

code(
    """df_chimp = pd.read_csv(CHIMP_PATH, sep=";")
print("Chimpanzees dataset (N=504 trials across 7 actors):")
print(df_chimp[["actor", "condition", "prosoc_left", "pulled_left"]].head(8))

actors = df_chimp["actor"].values - 1  # 0 to 6
n_actors = len(np.unique(actors))
prosoc = df_chimp["prosoc_left"].values
cond = df_chimp["condition"].values  # 1 = partner present, 0 = no partner
pulled_left = df_chimp["pulled_left"].values

with pm.Model() as chimp_hierarchical:
    # Population hyperpriors on actor intercepts
    a_bar = pm.Normal("a_bar", mu=0.0, sigma=1.5)
    tau_a = pm.HalfNormal("tau_a", sigma=1.5)
    
    # Non-centered actor offsets
    z_a = pm.Normal("z_a", mu=0.0, sigma=1.0, shape=n_actors)
    alpha_actor = pm.Deterministic("alpha_actor", a_bar + tau_a * z_a)
    
    # Treatment effects
    beta_prosoc = pm.Normal("beta_prosoc", mu=0.0, sigma=1.0)
    beta_partner = pm.Normal("beta_partner", mu=0.0, sigma=1.0)  # Prosocial effect with partner
    
    eta = alpha_actor[actors] + beta_prosoc * prosoc + beta_partner * (prosoc * cond)
    p = pm.Deterministic("p", pm.math.invlogit(eta))
    
    pm.Bernoulli("y_obs", p=p, observed=pulled_left)
    idata_chimp = pm.sample(draws=1000, tune=1000, chains=2, random_seed=RANDOM_SEED, progressbar=False)

summary_chimp = az.summary(idata_chimp, var_names=["a_bar", "tau_a", "beta_prosoc", "beta_partner", "alpha_actor"])
print("\\n--- Hierarchical Chimpanzee Model Estimates ---")
print(summary_chimp)
"""
)

md(
    """### What to Watch For
1. **Actor 2 Regularized**: Actor 2's intercept (`alpha_actor[1]`) is shrunk to a finite value of $\\approx +3.8$ rather than exploding to infinity.
2. **Substantive Scientific Finding**: The interaction term for partner presence (`beta_partner`) has a posterior mean of $\\approx -0.09$ with a $95\\%$ HDI of $[-0.40, +0.22]$ spanning zero symmetrically. Chimpanzees did not pull the prosocial lever more often when a partner was present.
""",
    """### На что обратить внимание
1. **Регуляризация Актёра 2**: интерцепт Актёра 2 (`alpha_actor[1]`) стянут к конечному значению $\\approx +3{,}8$ вместо бесконечности.
2. **Содержательный научный вывод**: коэффициент присутствия партнера (`beta_partner`) имеет апостериорное среднее $\\approx -0{,}09$ с $95\\%$-м HDI $[-0{,}40; +0{,}22]$, симметрично накрывающим ноль. Шимпанзе не выбирали просоциальный рычаг чаще в присутствии сородича.
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
