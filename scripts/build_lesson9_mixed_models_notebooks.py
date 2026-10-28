"""Build the Lesson 9 Going Further workbook: classical mixed models vs. hierarchical Bayes.

Full transfer of the legacy course notebook `10_MixedModel(Linear).ipynb`
(BayesianModelsMaterials) after the 2026-10-28 revision: the classical ladder
(pooled OLS -> subject dummies -> random intercept -> random intercept+slope),
residual diagnostics, the estimated random-effect covariance, a corrected
pointwise CI band for the population line, and the Bayesian Bambi + PSIS-LOO
comparison. Optional homework: dietox pigs.

Run from the book root with the pymc_env environment:
    /opt/homebrew/Caskroom/miniforge/base/envs/pymc_env/bin/python scripts/build_lesson9_mixed_models_notebooks.py

Outputs:
    notebooks/colab/09_mixed_models_linear.ipynb
    notebooks/colab/09_mixed_models_linear_ru.ipynb
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
    "en": "09_mixed_models_linear.ipynb",
    "ru": "09_mixed_models_linear_ru.ipynb",
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
    f"""# Lesson 9 (Going Further): Classical Mixed Models vs. Hierarchical Bayes
### Student Laboratory Workbook
*Course: Bayesian Analysis of Empirical Data (2026)*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({GITHUB}{FILES['en']})
*Russian version: [{FILES['ru']}]({GITHUB}{FILES['ru']})*

This companion to the *Mixed Models vs. Hierarchical Bayes* page rebuilds the **entire classical ladder** on the `sleepstudy` data: a pooled OLS regression, subject dummy fixed effects, a random-intercept `MixedLM`, and a random-intercept-plus-slope `MixedLM` with an **estimated covariance** between the random effects. We then repeat the two Bayesian hierarchical models in Bambi and compare them with PSIS-LOO. Along the way we fix three classic traps of the classical workflow: what Durbin–Watson actually measures, how to build a correct confidence band for the population line, and why BLUP scatter-plots are not the model-estimated correlation.
""",
    f"""# Занятие 9 (дополнительно): классические смешанные модели против байесовской иерархии
### Рабочая тетрадь студента
*Курс: Байесовский анализ эмпирических данных (2026)*

[![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)]({GITHUB}{FILES['ru']})
*English version: [{FILES['en']}]({GITHUB}{FILES['en']})*

Это приложение к странице «Классические смешанные модели и байесовская иерархия». Мы пройдём **всю классическую лестницу** на данных `sleepstudy`: объединённая OLS-регрессия, фиксированные дамми-эффекты испытуемых, `MixedLM` со случайным интерсептом и `MixedLM` со случайным интерсептом и наклоном, включая **оцениваемую ковариацию** случайных эффектов. Затем повторим две байесовские иерархические модели в Bambi и сравним их по PSIS-LOO. По пути разберём три классические ловушки: что на самом деле измеряет статистика Дарбина–Уотсона, как строить корректный доверительный коридор для популяционной прямой и почему диаграмма рассеяния BLUP — это не оценка корреляции из модели.
""",
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
code(
    """import sys, os, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.gofplots import qqplot

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

RANDOM_SEED = 2026
rng = np.random.default_rng(RANDOM_SEED)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# Data: sleepstudy (Belenky et al., 2003), canonical copy in the book repository.
DATA_URL = "https://raw.githubusercontent.com/iknyazeva/bayes-cogsci-book/main/data/sleepstudy.csv"
DATA_PATH = "data/sleepstudy.csv"
os.makedirs("data", exist_ok=True)
if not os.path.exists(DATA_PATH):
    try:
        df_sleep = bmb.load_data("sleepstudy")
        df_sleep.to_csv(DATA_PATH, index=False)
    except Exception:
        import urllib.request
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)

print("Environment ready: statsmodels", sm.__version__, "| Bambi", bmb.__version__, "| ArviZ", az.__version__)
"""
)

# ---------------------------------------------------------------------------
# Section 1: clustered data
# ---------------------------------------------------------------------------
md(
    """## 1. Clustered Data and the Research Question

### The Idea
Linear models assume independent observations. Repeated measurements break this assumption: reaction times of one person share that person's brain, habits and baseline speed. Any situation with clusters — repeated measurements over time, several schools, several clinics — produces **within-cluster correlation**, and treating such data as $N$ independent points makes the inference wrong. Our research question: how does reaction time change with days of sleep deprivation, both *on average in the population* and *for each individual subject*?

### Example
The `sleepstudy` experiment (Belenky et al., 2003): 18 subjects slept normally the night before day 0, then only 3 hours per night for 9 consecutive days; each day their mean reaction time (ms) was measured. That gives a balanced panel: 18 subjects $\\times$ 10 days $= 180$ observations.
""",
    """## 1. Кластеризованные данные и исследовательский вопрос

### Идея
Линейная модель предполагает независимость наблюдений. Повторные измерения нарушают это условие: времена реакции одного человека объединены общим мозгом, привычками и базовой скоростью. Любая ситуация с кластерами — повторные измерения во времени, несколько школ, несколько клиник — порождает **внутрикластерную корреляцию**, и обращение с такими данными как с $N$ независимыми точками делает выводы некорректными. Наш исследовательский вопрос: как время реакции меняется с днями депривации сна — и *в среднем по популяции*, и *у каждого отдельного испытуемого*?

### Пример
Эксперимент `sleepstudy` (Belenky et al., 2003): 18 участников спали нормально ночь перед днём 0, затем только по 3 часа в сутки 9 дней подряд; каждый день измерялось среднее время реакции (мс). Получается сбалансированная панель: 18 участников $\\times$ 10 дней $= 180$ наблюдений.
""",
)

code(
    """data = pd.read_csv(DATA_PATH)
data["Subject"] = data["Subject"].astype("category")
print(data.head())
print("\\nObservations per subject (balanced panel):")
print(data["Subject"].value_counts().sort_index().to_string())

# Per-subject trajectories: the whole story is visible in this panel
subjects = list(data["Subject"].unique())
fig, axes = plt.subplots(2, 9, figsize=(16, 7), sharey=True, sharex=True)
for ax, subj in zip(axes.ravel(), subjects):
    sub = data[data["Subject"] == subj]
    ax.scatter(sub["Days"], sub["Reaction"], color="C0", ec="black", alpha=0.7, s=22)
    m, b = np.polyfit(sub["Days"], sub["Reaction"], 1)
    ax.plot(sub["Days"], b + m * sub["Days"], color="C3", lw=1.5)
    ax.set_title(str(subj), fontsize=9)
fig.suptitle("sleepstudy: reaction time vs days of sleep deprivation (OLS line per subject)", fontsize=13)
fig.supxlabel("Days")
fig.supylabel("Reaction (ms)")
plt.tight_layout()
plt.show()
"""
)

md(
    """### What to Watch For
Individual differences are huge. Subject 308 starts fast (about 250 ms) and barely degrades; subject 337 starts slower and degrades dramatically (up to 600+ ms). Every subject has their own intercept *and* their own slope. Two naive strategies bracket the truth: one common line for everybody (complete pooling) pretends subjects do not exist; eighteen separate lines (no pooling) pretend the population does not exist. Mixed models — and their Bayesian generalization, hierarchical models — are the principled middle ground.

Below, a look at the mean trajectory also shows the population-level effect we ultimately want to quantify:
""",
    """### На что обратить внимание
Межиндивидуальные различия колоссальны. Участник 308 стартует быстро (около 250 мс) и почти не деградирует; участник 337 стартует медленнее и деградирует резко (до 600+ мс). У каждого своя и интерсепт, и скорость замедления. Две наивные стратегии ограничивают истину с двух сторон: одна общая прямая для всех (полное объединение) делает вид, что испытуемых не существует; восемнадцать отдельных прямых (без объединения) делают вид, что популяции не существует. Смешанные модели — и их байесовское обобщение, иерархические модели — являются принципиальной золотой серединой.

Ниже средняя траектория показывает популяционный эффект, который мы в итоге хотим оценить количественно:
""",
)

code(
    """print("Mean reaction by day:")
print(data.groupby("Days")[["Reaction"]].mean().round(1).to_string())

fig, ax = plt.subplots(figsize=(9, 4))
ax.scatter(data["Days"], data["Reaction"], color="C0", ec="black", alpha=0.35, s=20, label="all observations")
mean_by_day = data.groupby("Days")["Reaction"].mean()
ax.plot(mean_by_day.index, mean_by_day.values, "o-", color="C3", lw=2, label="mean by day")
ax.set_xlabel("Days of sleep deprivation")
ax.set_ylabel("Reaction (ms)")
ax.legend()
plt.tight_layout()
plt.show()
"""
)

# ---------------------------------------------------------------------------
# Section 2: the classical naive ladder
# ---------------------------------------------------------------------------
md(
    """## 2. The Naive Ladder in Classical Territory

### The Idea
Before any mixed model, build the two baselines with the OLS tool you already know.
- **Model 1 (complete pooling)**: one line for all 180 points — $\\text{Reaction} = \\beta_0 + \\beta_1 \\cdot \\text{Days} + \\varepsilon$. The slope answers the population question, but the residuals are correlated within subjects.
- **Model 2 (subject dummies)**: add `C(Subject)` — a separate intercept for each subject. This is "no pooling" on intercepts: each person gets their own free parameter, the population does not constrain them at all.

### Example — Model 1
""",
    """## 2. Наивная лестница в классической территории

### Идея
До всяких смешанных моделей построим два базовых уровня уже знакомым OLS-инструментом.
- **Модель 1 (полное объединение)**: одна прямая на все 180 точек — $\\text{Reaction} = \\beta_0 + \\beta_1 \\cdot \\text{Days} + \\varepsilon$. Наклон отвечает на популяционный вопрос, но остатки коррелированы внутри испытуемых.
- **Модель 2 (дамми испытуемых)**: добавим `C(Subject)` — отдельный интерсепт каждому испытуемому. Это «без объединения» по интерсептам: каждый человек получает собственный свободный параметр, популяция его никак не ограничивает.

### Пример — модель 1
""",
)

code(
    """lm_basic = smf.ols(formula="Reaction ~ Days", data=data).fit()
print(lm_basic.summary())

print("\\nDurbin-Watson on the residual order:", round(durbin_watson(lm_basic.resid), 3))
print("Shapiro-Wilk on residuals: W =", round(stats.shapiro(lm_basic.resid).statistic, 3),
      "p =", stats.shapiro(lm_basic.resid).pvalue)
"""
)

md(
    """### What to Watch For
The summary reports Durbin–Watson $\\approx 0.7$. Two clarifications students often get wrong:
1. **Durbin–Watson measures autocorrelation of residuals in row order, not multicollinearity of predictors.** DW $\\approx 2$ means no lag-1 autocorrelation; our $0.7$ means strong *positive* autocorrelation — consecutive rows belong to the same subject, so their errors are similar.
2. Shapiro–Wilk rejects exact normality, but that is a marginal-distribution symptom; the *structural* problem (clustered errors) is what invalidates standard errors, and no transformation of residuals fixes it.

The prediction plot makes the failure visible — the 95% bands for the mean and for new observations look reassuring, but they are computed under the false independence assumption (effectively $n = 180$ independent points, when we really have $n = 18$ subjects):
""",
    """### На что обратить внимание
Summary выдаёт Дарбина–Уотсона $\\approx 0{,}7$. Два уточнения, которые часто путают:
1. **Дарбин–Уотсон измеряет автокорреляцию остатков в порядке строк, а не мультиколлинеарность предикторов.** DW $\\approx 2$ — нет автокорреляции лага 1; наше $0{,}7$ — сильная *положительная* автокорреляция: соседние строки принадлежат одному испытуемому, поэтому их ошибки похожи.
2. Шапиро–Уилк отвергает точную нормальность, но это симптом маргинального распределения; *структурная* проблема (кластеризованные ошибки) — именно то, что обесценивает стандартные ошибки, и никакое преобразование остатков её не исправляет.

График предсказаний делает провал видимым — 95%-е коридоры для среднего и для новых наблюдений выглядят успокаивающе, но они вычислены при ложном предположении независимости (фактически $n = 180$ независимых точек, хотя на самом деле у нас $n = 18$ испытуемых):
""",
)

code(
    """def residual_diagnostics(residuals, title):
    \"\"\"Histogram, QQ-plot and residuals-in-row-order panel.\"\"\"
    fig, axs = plt.subplots(1, 3, figsize=(11, 3))
    axs[0].hist(residuals, bins=25, color="C0", alpha=0.7, edgecolor="white")
    axs[0].set_title("residual histogram")
    qqplot(residuals, line="s", ax=axs[1], markersize=3)
    axs[1].set_title("QQ-plot vs Normal")
    axs[2].scatter(range(len(residuals)), residuals, s=8, alpha=0.6, color="C0")
    axs[2].set_title("residuals in row order")
    axs[2].set_xlabel("row (subject-major order)")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    print("Shapiro-Wilk:", stats.shapiro(residuals))

residual_diagnostics(np.asarray(lm_basic.resid), "Model 1 (pooled OLS): residuals")

# Predictions with mean-CI and observation-CI bands
pred = lm_basic.get_prediction().summary_frame()
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.scatter(data["Days"], data["Reaction"], color="C0", ec="black", alpha=0.5, s=20, label="data")
order = np.argsort(data["Days"].values)
ax.plot(data["Days"].values[order], pred["mean"].values[order], "r-", label="OLS fit")
ax.plot(data["Days"].values[order], pred["mean_ci_lower"].values[order], "r--", lw=1, label="95% CI for the mean")
ax.plot(data["Days"].values[order], pred["mean_ci_upper"].values[order], "r--", lw=1)
ax.plot(data["Days"].values[order], pred["obs_ci_lower"].values[order], "g--", lw=1, label="95% interval for a new obs")
ax.plot(data["Days"].values[order], pred["obs_ci_upper"].values[order], "g--", lw=1)
ax.set_xlabel("Days"); ax.set_ylabel("Reaction (ms)"); ax.legend()
plt.tight_layout(); plt.show()
"""
)

md(
    """### Example — Model 2 (subject dummies)

Now each subject receives their own intercept. Universality is lost — predictions require knowing *which* subject — but within-subject correlation of intercepts disappears, because it is absorbed by the 18 dummy coefficients:
""",
    """### Пример — модель 2 (дамми испытуемых)

Теперь каждый испытуемый получает собственный интерсепт. Универсальность потеряна — для предсказания нужно знать, *какой именно* испытуемый, — но внутрииспытуемая корреляция интерсептов исчезает: она поглощена 18 дамми-коэффициентами:
""",
)

code(
    """lm_sub = smf.ols(formula="Reaction ~ Days + C(Subject)", data=data).fit()
print(lm_sub.summary().tables[0])
print(lm_sub.summary().tables[1].as_text()[:1200], " ...")

residual_diagnostics(np.asarray(lm_sub.resid), "Model 2 (subject dummies): residuals")

# Per-subject fits of the dummy model
data["fitted_sub"] = lm_sub.fittedvalues
fig, ax = plt.subplots(figsize=(11, 5))
ax.scatter(data["Days"], data["Reaction"], color="C0", ec="black", alpha=0.4, s=18)
sns.lineplot(x="Days", y="fitted_sub", hue="Subject", data=data, ax=ax, palette="tab20", lw=1.2, legend=False)
ax.set_title("Model 2: parallel per-subject lines (common slope, free intercepts)")
plt.tight_layout(); plt.show()
"""
)

md(
    """### What to Watch For
Residual structure improves markedly (DW, QQ-plot, the row-order scatter). But Model 2 has three deep problems:
1. **18 extra parameters with no shrinkage** — noisy subjects get extreme intercepts exactly as extreme as their noise demands.
2. **The slope is still forced to be common** — yet the panel showed individually different slopes; subject 309 degrades almost three times faster than subject 335.
3. **No prediction for a new person is possible** — a dummy for a subject we have never seen does not exist.

These three problems are exactly what the random-effects machinery answers: give each subject parameters that are *draws from a population distribution* instead of free unrelated numbers.
""",
    """### На что обратить внимание
Структура остатков заметно улучшается (DW, QQ-график, рассеяние по порядку строк). Но у модели 2 три глубокие проблемы:
1. **18 дополнительных параметров без сжатия** — зашумлённые испытуемые получают экстремальные интерсепты ровно настолько экстремальные, насколько требует их шум.
2. **Наклон по-прежнему общий** — хотя панель показала индивидуально разные наклоны; участник 309 деградирует почти втрое быстрее участника 335.
3. **Предсказание для нового человека невозможно** — дамми для никогда прежде не наблюдавшегося испытуемого не существует.

Ровно на эти три проблемы отвечает аппарат случайных эффектов: дать каждому испытуемому параметры, *извлечённые из популяционного распределения*, а не свободные несвязанные числа.
""",
)

# ---------------------------------------------------------------------------
# Section 3: mixed models
# ---------------------------------------------------------------------------
md(
    """## 3. Linear Mixed Models: Random Intercepts, then Random Slopes

### The Idea
A **mixed model** contains *fixed effects* — population-level coefficients shared by everyone (here: the average intercept $\\beta_0$ and the average slope $\\beta_1$) — and *random effects* — subject-level deviations modelled as draws from a distribution. The random-intercept model:

$$\\text{Reaction}_{ij} = \\beta_0 + \\beta_1 \\, \\text{Days}_{ij} + b_i + \\varepsilon_{ij}, \\qquad b_i \\sim \\mathcal{N}(0, \\sigma_b^2), \\;\\; \\varepsilon_{ij} \\sim \\mathcal{N}(0, \\sigma^2)$$

where $i$ indexes subjects and $j$ days. In R (`lme4`) this is `lmer(Reaction ~ Days + (1 | Subject))`; in Python `statsmodels` uses `mixedlm`. The variance components $\\sigma_b^2$ and $\\sigma^2$ are estimated together with the fixed effects (by default via REML — restricted maximum likelihood).

### Example — random intercept
""",
    """## 3. Линейные смешанные модели: случайные интерсепты, затем случайные наклоны

### Идея
**Смешанная модель** содержит *фиксированные эффекты* — коэффициенты популяционного уровня, общие для всех (здесь: средний интерсепт $\\beta_0$ и средний наклон $\\beta_1$) — и *случайные эффекты* — отклонения уровня испытуемых, моделируемые как извлечения из распределения. Модель со случайным интерсептом:

$$\\text{Reaction}_{ij} = \\beta_0 + \\beta_1 \\, \\text{Days}_{ij} + b_i + \\varepsilon_{ij}, \\qquad b_i \\sim \\mathcal{N}(0, \\sigma_b^2), \\;\\; \\varepsilon_{ij} \\sim \\mathcal{N}(0, \\sigma^2)$$

где $i$ — номер испытуемого, $j$ — день. В R (`lme4`) это `lmer(Reaction ~ Days + (1 | Subject))`; в Python `statsmodels` использует `mixedlm`. Компоненты дисперсии $\\sigma_b^2$ и $\\sigma^2$ оцениваются вместе с фиксированными эффектами (по умолчанию методом REML — ограниченного максимального правдоподобия).

### Пример — случайный интерсепт
""",
)

code(
    """lm_mixed_ri = smf.mixedlm("Reaction ~ Days", data, groups=data["Subject"]).fit(reml=True)
print(lm_mixed_ri.summary())

# Variance components from the fitted object (never hardcode summary numbers!)
var_subject = lm_mixed_ri.cov_re.iloc[0, 0]
var_residual = lm_mixed_ri.scale
print(f"sigma_b (between-subject SD of intercepts) = sqrt({var_subject:.1f}) = {np.sqrt(var_subject):.2f} ms")
print(f"sigma   (within-subject residual SD)        = sqrt({var_residual:.1f}) = {np.sqrt(var_residual):.2f} ms")
print("Intraclass correlation ICC = sigma_b^2 / (sigma_b^2 + sigma^2) =",
      round(var_subject / (var_subject + var_residual), 3))
"""
)

md(
    """### What to Watch For
- $\\hat\\sigma_b \\approx 37$ ms — the population SD of personal baselines; ICC $\\approx 0.68$ means about two thirds of the residual variance lives *between* subjects, which is exactly the correlation the pooled OLS pretended not to exist.
- The subject deviations printed below are **BLUPs** (conditional modes): point predictions of $b_i$ given the estimated variance components. They come *without* usable standard errors, and they inherit the uncertainty of $\\hat\\sigma_b$ silently — keep this limitation in mind for the Bayesian comparison later.
""",
    """### На что обратить внимание
- $\\hat\\sigma_b \\approx 37$ мс — популяционное SD личных базовых уровней; ICC $\\approx 0{,}68$ означает, что около двух третей остаточной дисперсии живёт *между* испытуемыми — ровно та корреляция, которую объединённый OLS делал вид, что не замечает.
- Отклонения испытуемых, напечатанные ниже, — **BLUP** (условные моды): точечные предсказания $b_i$ при оценённых компонентах дисперсии. Они идут *без* пригодных стандартных ошибок и молча наследуют неопределённость $\\hat\\sigma_b$ — держите это ограничение в уме для байесовского сравнения позже.
""",
)

code(
    """re_ri = pd.DataFrame(lm_mixed_ri.random_effects).T
re_ri.columns = ["b_i (BLUP)"]
print(re_ri.T.round(1).to_string())

fig, ax = plt.subplots(figsize=(11, 3.5))
re_ri["b_i (BLUP)"].plot(kind="bar", ax=ax, color="C0", edgecolor="black")
ax.axhline(0, color="C3", lw=1)
ax.set_ylabel("BLUP $\\\\hat b_i$ (ms)")
ax.set_title("Random-intercept model: predicted subject deviations")
plt.tight_layout(); plt.show()

# Fitted per-subject lines
data["fitted_ri"] = lm_mixed_ri.fittedvalues
fig, ax = plt.subplots(figsize=(11, 5))
ax.scatter(data["Days"], data["Reaction"], color="C0", ec="black", alpha=0.4, s=18)
sns.lineplot(x="Days", y="fitted_ri", hue="Subject", data=data, ax=ax, palette="tab20", lw=1.2, legend=False)
ax.set_title("Random-intercept model: common slope, subject offsets")
plt.tight_layout(); plt.show()

residual_diagnostics(np.asarray(lm_mixed_ri.resid), "Random-intercept MixedLM: residuals")

# Residual structure by day and by subject
data["resid_ri"] = lm_mixed_ri.resid
fig, axs = plt.subplots(1, 2, figsize=(12, 3.5))
sns.boxplot(x="Days", y="resid_ri", data=data, ax=axs[0], color="C0")
axs[0].set_title("residuals by day")
sns.boxplot(x="Subject", y="resid_ri", data=data, ax=axs[1], color="C0")
axs[1].tick_params(axis="x", labelrotation=90)
axs[1].set_title("residuals by subject")
plt.tight_layout(); plt.show()
"""
)

md(
    """### Example — random intercept **and** random slope

The subject-level panels still show systematic patterns: some subjects' residuals trend upward within the boxplots — their true slopes differ. Add a random slope:

$$\\text{Reaction}_{ij} = \\beta_0 + \\beta_1 \\, \\text{Days}_{ij} + b_i + c_i \\, \\text{Days}_{ij} + \\varepsilon_{ij}$$

$$(b_i, c_i) \\sim \\mathcal{N}\\!\\left(\\mathbf{0}, \\begin{bmatrix} \\sigma_b^2 & \\rho\\,\\sigma_b\\sigma_c \\\\ \\rho\\,\\sigma_b\\sigma_c & \\sigma_c^2 \\end{bmatrix}\\right)$$

In `lme4`: `lmer(Reaction ~ Days + (1 + Days | Subject))`. **Important:** `statsmodels` with `re_formula="~ Days"` estimates the full $2\\times 2$ covariance, including the correlation $\\rho$ between random intercepts and slopes (it appears in the summary as `Group x Days Cov`). A common misconception says statsmodels cannot model this covariance structure — it can and does; what it cannot give you is a standard error or a test for $\\rho$.
""",
    """### Пример — случайный интерсепт **и** случайный наклон

В панелях по испытуемым всё ещё видны систематические узоры: у некоторых остатки внутри боксплотов идут с трендом — их истинные наклоны различаются. Добавим случайный наклон:

$$\\text{Reaction}_{ij} = \\beta_0 + \\beta_1 \\, \\text{Days}_{ij} + b_i + c_i \\, \\text{Days}_{ij} + \\varepsilon_{ij}$$

$$(b_i, c_i) \\sim \\mathcal{N}\\!\\left(\\mathbf{0}, \\begin{bmatrix} \\sigma_b^2 & \\rho\\,\\sigma_b\\sigma_c \\\\ \\rho\\,\\sigma_b\\sigma_c & \\sigma_c^2 \\end{bmatrix}\\right)$$

В `lme4`: `lmer(Reaction ~ Days + (1 + Days | Subject))`. **Важно:** `statsmodels` с `re_formula="~ Days"` оценивает полную ковариацию $2\\times 2$, включая корреляцию $\\rho$ между случайными интерсептами и наклонами (в summary она видна как `Group x Days Cov`). Распространённое заблуждение утверждает, что statsmodels не умеет моделировать эту ковариационную структуру — умеет; чего он не даёт, так это стандартной ошибки или теста для $\\rho$.
""",
)

code(
    """lm_mixed_ris = smf.mixedlm("Reaction ~ Days", data, groups=data["Subject"], re_formula="~ Days").fit(reml=True)
print(lm_mixed_ris.summary())

cov_re = lm_mixed_ris.cov_re
print("\\nEstimated random-effects covariance matrix (BLUP scale):")
print(cov_re.round(2))

corr_model = cov_re.iloc[0, 1] / np.sqrt(cov_re.iloc[0, 0] * cov_re.iloc[1, 1])
print(f"\\nModel-estimated corr(intercept, slope) rho = {corr_model:.3f}")
print("statsmodels reports no standard error for rho — remember this for the Bayesian part.")

re_ris = pd.DataFrame(lm_mixed_ris.random_effects).T
re_ris.columns = ["b_i (intercept)", "c_i (slope)"]
print("\\nFirst BLUPs:"); print(re_ris.head().round(1).to_string())

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.scatter(re_ris["b_i (intercept)"], re_ris["c_i (slope)"], color="C0", ec="black", alpha=0.8)
for subj, row in re_ris.iterrows():
    ax.annotate(str(subj), (row["b_i (intercept)"], row["c_i (slope)"]), fontsize=7, alpha=0.7)
ax.set_xlabel("BLUP intercept deviation $\\\\hat b_i$ (ms)")
ax.set_ylabel("BLUP slope deviation $\\\\hat c_i$ (ms/day)")
ax.set_title(f"BLUPs vs model estimate: sample corr = {np.corrcoef(re_ris.iloc[:,0], re_ris.iloc[:,1])[0,1]:.2f}, model rho = {corr_model:.2f}")
plt.tight_layout(); plt.show()

residual_diagnostics(np.asarray(lm_mixed_ris.resid), "Random-intercept+slope MixedLM: residuals")

data["resid_ris"] = lm_mixed_ris.resid
fig, axs = plt.subplots(1, 2, figsize=(12, 3.5))
sns.boxplot(x="Days", y="resid_ris", data=data, ax=axs[0], color="C0")
axs[0].set_title("residuals by day (improved)")
sns.boxplot(x="Subject", y="resid_ris", data=data, ax=axs[1], color="C0")
axs[1].tick_params(axis="x", labelrotation=90)
axs[1].set_title("residuals by subject (improved)")
plt.tight_layout(); plt.show()
"""
)

md(
    """### What to Watch For
- Residual boxplots by day and by subject are now much flatter; the RIS model has absorbed both personal baselines and personal vulnerability to sleep loss.
- The model-estimated correlation $\\hat\\rho \\approx 0.07$ is **not** the correlation of the BLUP scatter (that sample correlation is inflated: BLUPs are shrunk predictions, not parameter estimates, and their sample correlation conflates shrinkage with the true $\\rho$). Report $\\hat\\rho$ from `cov_re`, and remember it arrives without uncertainty quantification.
- The variance-component inference problem in one sentence: `lme4` famously refuses to print p-values for fixed effects in small samples, and neither `lme4` nor `statsmodels` gives you a confidence interval for $\\sigma_b$, $\\sigma_c$ or $\\rho$. The parametric bootstrap is the classical escape; the Bayesian posterior is the more direct one — both are coming next (bootstrap on the homework path, posterior right below).
""",
    """### На что обратить внимание
- Боксплоты остатков по дням и по испытуемым теперь заметно ровнее; модель со случайным наклоном поглотила и личные базовые уровни, и личную уязвимость к недосыпу.
- Оценённая моделью корреляция $\\hat\\rho \\approx 0{,}07$ — **не** корреляция облака BLUP (выборочная корреляция завышена: BLUP — сжатые предсказания, а не оценки параметров, и их выборочная корреляция смешивает сжатие с истинным $\\rho$). Сообщайте $\\hat\\rho$ из `cov_re` и помните, что она приходит без квантификации неопределённости.
- Проблема вывода для компонент дисперсии в одном предложении: `lme4` известен тем, что отказывается печатать p-значения для фиксированных эффектов в малых выборках, и ни `lme4`, ни `statsmodels` не дают доверительный интервал для $\\sigma_b$, $\\sigma_c$ или $\\rho$. Классический выход — параметрический бутстреп; более прямой — байесовское апостериорное распределение (бутстреп остаётся на домашнем пути, апостериорное — сразу ниже).
""",
)

# ---------------------------------------------------------------------------
# Section 4: correct CI band
# ---------------------------------------------------------------------------
md(
    """## 4. A Correct Confidence Band for the Population Line

### The Idea
A tempting shortcut (seen in many course notebooks, including the legacy version of this one) plots the "lower band" by combining the lower CI of the intercept with the lower CI of the slope, and the "upper band" from the two uppers. That is **not** a confidence band: it joins opposite corners of the parameter rectangle, ignores the covariance between $\\hat\\beta_0$ and $\\hat\\beta_1$, and overstates uncertainty at every $x$. The correct pointwise band uses the full covariance matrix of the fixed effects:

$$\\widehat{\\mathrm{se}}\\big(\\hat\\mu(x)\\big) = \\sqrt{\\mathbf{x}^{\\top} \\, \\widehat{\\operatorname{Cov}}(\\hat\\beta)\\, \\mathbf{x}}, \\qquad \\hat\\mu(x) \\pm t_{0.975,\\, \\mathrm{df}} \\cdot \\widehat{\\mathrm{se}}$$

### Example
""",
    """## 4. Корректный доверительный коридор для популяционной прямой

### Идея
Заманчивый ярлык (встречающийся во многих учебных тетрадях, включая прежнюю версию этой) строит «нижнюю границу» из нижних доверительных границ интерсепта и наклона, а «верхнюю» — из двух верхних. Это **не** доверительный коридор: он соединяет противоположные углы прямоугольника параметров, игнорирует ковариацию между $\\hat\\beta_0$ и $\\hat\\beta_1$ и завышает неопределённость при каждом $x$. Корректный поточечный коридор использует полную ковариационную матрицу фиксированных эффектов:

$$\\widehat{\\mathrm{se}}\\big(\\hat\\mu(x)\\big) = \\sqrt{\\mathbf{x}^{\\top} \\, \\widehat{\\operatorname{Cov}}(\\hat\\beta)\\, \\mathbf{x}}, \\qquad \\hat\\mu(x) \\pm t_{0.975,\\, \\mathrm{df}} \\cdot \\widehat{\\mathrm{se}}$$

### Пример
""",
)

code(
    """def fixed_effect_ci_band(res, xs):
    \"\"\"Pointwise t-CI for the fixed-effects mean of a MixedLM at points xs.\"\"\"
    X = np.column_stack([np.ones_like(xs), xs])          # design for [Intercept, Days]
    beta = res.fe_params.values[:2]
    V = res.cov_params().loc[["Intercept", "Days"], ["Intercept", "Days"]].values
    mean = X @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", X, V, X))
    # statsmodels keeps a residual-df-style normal approximation for REML fits;
    # we use the normal quantile, which is what MixedLM itself reports.
    crit = stats.norm.ppf(0.975)
    return mean, mean - crit * se, mean + crit * se

xs = np.linspace(0, 9, 100)
mean, lo, hi = fixed_effect_ci_band(lm_mixed_ris, xs)

# The incorrect "rectangle corner" band for comparison
ci = lm_mixed_ris.conf_int().loc[["Intercept", "Days"]]
corner_lo = ci.loc["Intercept", 0] + ci.loc["Days", 0] * xs
corner_hi = ci.loc["Intercept", 1] + ci.loc["Days", 1] * xs

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.scatter(data["Days"], data["Reaction"], color="C0", ec="black", alpha=0.35, s=18, label="data")
ax.plot(xs, mean, "r-", label="fixed-effects mean")
ax.fill_between(xs, lo, hi, color="C3", alpha=0.35, label="correct pointwise 95% CI")
ax.plot(xs, corner_lo, ":", color="gray", lw=2, label="corner-plugging band (WRONG)")
ax.plot(xs, corner_hi, ":", color="gray", lw=2)
ax.set_xlabel("Days"); ax.set_ylabel("Reaction (ms)"); ax.legend()
ax.set_title("Population line: correct pointwise CI vs the corner-plugging shortcut")
plt.tight_layout(); plt.show()
"""
)

md(
    """### What to Watch For
The dotted "corner" band is visibly wider than the correct band — and conceptually it answers no well-defined question. Two honest limitations remain even in the correct band: it conditions on the estimated variance components (their uncertainty is ignored), and it is a band for the **population mean** $\\beta_0 + \\beta_1 x$, not for a new subject's trajectory. The Bayesian version later today will hand you the full posterior of both automatically.
""",
    """### На что обратить внимание
Пунктирный «угловой» коридор заметно шире корректного — и концептуально он не отвечает ни на один корректно поставленный вопрос. Даже в корректном коридоре остаются два честных ограничения: он условен на оценённые компоненты дисперсии (их неопределённость игнорируется), и это коридор для **популяционного среднего** $\\beta_0 + \\beta_1 x$, а не для траектории нового испытуемого. Байесовская версия ниже выдаст полную апостериорную неопределённость для обоих случаев автоматически.
""",
)

# ---------------------------------------------------------------------------
# Section 5: Bambi + PSIS-LOO
# ---------------------------------------------------------------------------
md(
    """## 5. The Bayesian Counterpart in Bambi, Compared with PSIS-LOO

### The Idea
The same two models — random intercept, then random intercept **and** slope — written in the `lme4`-style formula language of [Bambi](https://bambinos.github.io/bambi/). Differences from the classical fit:
- variance components and $\\rho$ get **full posterior distributions** instead of point estimates without standard errors;
- Bambi places weakly informative default priors on all parameters; internally it builds the non-centered parameterization you met in the main Lesson 9 workbook (no funnel); its `(x | g)` terms fit *independent* group effects — see the subtlety below;
- model choice follows the course convention: **PSIS-LOO** out-of-sample predictive accuracy via `az.compare` and `az.loo` (WAIC was its close relative in older ArviZ versions; ArviZ 1.x maintains PSIS-LOO).

### Example
""",
    """## 5. Байесовский двойник в Bambi и сравнение по PSIS-LOO

### Идея
Те же две модели — случайный интерсепт, затем случайный интерсепт **и** наклон — записанные на языке формул в стиле `lme4` с помощью [Bambi](https://bambinos.github.io/bambi/). Отличия от классической подгонки:
- компоненты дисперсии и $\\rho$ получают **полные апостериорные распределения** вместо точечных оценок без стандартных ошибок;
- Bambi ставит слабоинформативные априорные распределения на все параметры; внутри строится безцентровая параметризация из основной тетради занятия 9 (без воронки); термины `(x | g)` строят *независимые* групповые эффекты — см. тонкость ниже;
- выбор модели следует конвенции курса: вневыборочная предсказательная точность **PSIS-LOO** через `az.compare` и `az.loo` (WAIC — её близкая родственница в старых версиях ArviZ; ArviZ 1.x поддерживает PSIS-LOO).

### Пример
""",
)

code(
    """model1 = bmb.Model("Reaction ~ 1 + Days + (1 | Subject)", data, categorical="Subject")
print(model1)
idata1 = model1.fit(draws=1000, tune=2000, chains=4, random_seed=RANDOM_SEED,
                    idata_kwargs={"log_likelihood": True})
print(az.summary(idata1, var_names=["Intercept", "Days", "1|Subject_sigma", "sigma"],
                 round_to=2))
print("\\nPSIS-LOO (ELPD scale):", az.loo(idata1).elpd.round(1))
print("(The legacy notebook printed WAIC; ArviZ 1.x maintains PSIS-LOO as its criterion.)")
"""
)

code(
    """model2 = bmb.Model("Reaction ~ 1 + Days + (Days | Subject)", data, categorical="Subject")
print(model2)
idata2 = model2.fit(draws=1000, tune=2000, chains=4, random_seed=RANDOM_SEED,
                    idata_kwargs={"log_likelihood": True})
print(az.summary(idata2, var_names=["Intercept", "Days", "1|Subject_sigma",
                                    "Days|Subject_sigma", "sigma"],
                 round_to=2))
print("\\nPSIS-LOO (ELPD scale):", az.loo(idata2).elpd.round(1))
"""
)

code(
    """# PSIS-LOO model comparison (course convention)
cmp_df = az.compare({"RI: (1|Subject)": idata1, "RIS: (Days|Subject)": idata2})
print(cmp_df.round(1))
az.plot_compare(cmp_df)
plt.show()
"""
)

md(
    """### What to Watch For
1. **Trust checks before interpretation.** Check `r_hat` ($\\le 1.01$) and ESS in the summaries above before believing anything. If a notebook you read skips this step, distrust its numbers.
2. The RIS model `(Days | Subject)` wins the LOO comparison: individual slopes carry real predictive information — exactly what the subject panels suggested.
2b. ArviZ printed warnings like `Estimated shape parameter ... greater than 0.70`: in hierarchical models each observation carries the influence of its cluster's group parameters, so PSIS importance sampling often flags $\\hat k > 0.7$ even for a healthy model. Treat these flags as a signal to double-check (or to use leave-one-*cluster*-out cross-validation), not as an automatic verdict of failure.
3. Now look at what the classical fit could not give you: a full posterior for $\\sigma_{\\text{intercept}}$ and $\\sigma_{\\text{slope}}$. The forest plots below show per-subject intercepts and slopes *with* uncertainty — compare them with the lonely BLUP points of Section 3.

*(The legacy notebook compared the two models with WAIC; here we print PSIS-LOO — the criterion maintained by ArviZ 1.x and adopted throughout this course.)*

One structural subtlety: **Bambi's `(Days | Subject)` fits independent varying intercepts and slopes — it does not estimate their correlation** (unlike `lme4`'s `(1 + Days | Subject)`, and unlike the `MixedLM` of Section 3 with its `Group x Days Cov`). The cell below quantifies how much that simplification matters here.
""",
    """### На что обратить внимание
1. **Проверки доверия прежде интерпретаций.** Перед тем как верить числам, проверьте `r_hat` ($\\le 1{,}01$) и ESS в сводках выше. Если тетрадь пропускает этот шаг — не доверяйте её числам.
2. Модель со случайным наклоном `(Days | Subject)` выигрывает по LOO: индивидуальные наклоны несут реальную предсказательную информацию — ровно то, что предполагали панели испытуемых.
2b. ArviZ напечатал предупреждения вида `Estimated shape parameter ... greater than 0.70`: в иерархических моделях каждое наблюдение несёт влияние групповых параметров своего кластера, поэтому PSIS-выборка часто помечает $\\hat k > 0.7$ даже у здоровой модели. Относитесь к этим флагам как к повод перепроверить (или перейти к кросс-валидации с исключением целого кластера), а не как к автоматическому приговору.
3. Теперь посмотрите, чего не мог дать классический фит: полное апостериорное распределение для $\\sigma_{\\text{интерсепт}}$ и $\\sigma_{\\text{наклон}}$. Графики-«лес» ниже показывают интерсепты и наклоны по испытуемым *вместе с* неопределённостью — сравните с одинокими точками BLUP из раздела 3.

*(Прежняя учебная тетрадь сравнивала модели по WAIC; здесь мы печатаем PSIS-LOO — поддерживаемый в ArviZ 1.x критерий, принятый в этом курсе.)*

Один структурный тонкий момент: **Bambi в формуле `(Days | Subject)` строит независимые случайные интерсепты и наклоны — их корреляция не оценивается** (в отличие от `lme4` с `(1 + Days | Subject)` и от `MixedLM` из раздела 3 с его `Group x Days Cov`). Ячейка ниже показывает, насколько это упрощение здесь существенно.
""",
)

code(
    """# Posterior of the variance components — the classical fit gave single numbers only
post_ds = idata2.posterior.to_dataset()
fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, vname, classical in zip(
        axs, ["1|Subject_sigma", "Days|Subject_sigma"],
        [np.sqrt(cov_re.iloc[0, 0]), np.sqrt(cov_re.iloc[1, 1])]):
    draws = post_ds[vname].values.flatten()
    ax.hist(draws, bins=40, color="C0", alpha=0.75, edgecolor="white", density=True)
    lo, hi = az.hdi(draws)
    ax.axvline(lo, color="C3", ls="--", label=f"94% HDI [{lo:.1f}, {hi:.1f}]")
    ax.axvline(hi, color="C3", ls="--")
    ax.axvline(classical, color="black", lw=2, label=f"classical point {classical:.1f}")
    ax.set_title(vname)
    ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

print(f"Classical point estimates: sigma_intercept = {np.sqrt(cov_re.iloc[0,0]):.1f} ms, "
      f"sigma_slope = {np.sqrt(cov_re.iloc[1,1]):.1f} ms/day, rho = {corr_model:.3f}")

# Bambi fits independent group effects: quantify the realized correlation per draw.
# (With 18 subjects this realized statistic is a noisy proxy for the population rho,
#  but its spread shows the orders of magnitude compatible with the data.)
b_draws = post_ds["1|Subject"].values.reshape(-1, len(subjects))
c_draws = post_ds["Days|Subject"].values.reshape(-1, len(subjects))
realized_corr = np.array([np.corrcoef(b_draws[k], c_draws[k])[0, 1] for k in range(b_draws.shape[0])])
print(f"Per-draw realized corr(intercept, slope): mean = {realized_corr.mean():.3f}, "
      f"SD = {realized_corr.std():.3f}")

# Bivariate posterior cloud of subject effects (one dot = one posterior draw of one subject)
_, ax = plt.subplots(figsize=(5.5, 4.5))
post_ds.plot.scatter(x="1|Subject", y="Days|Subject",
                     hue="Subject__factor_dim", add_colorbar=False,
                     add_legend=False, cmap="tab20", s=4, alpha=0.15, ax=ax)
ax.set_xlabel("posterior intercept deviation (ms)")
ax.set_ylabel("posterior slope deviation (ms/day)")
ax.set_title("Joint posterior of subject effects (RIS model)")
plt.show()

# Forest plots with uncertainty — compare with BLUP bars in Section 3
def forest_plot(ax, draws_2d, labels, xlabel):
    # Per-subject HDI line + posterior mean dot (draws_2d: draws x subjects)
    order = np.argsort(draws_2d.mean(axis=0))
    for pos, idx in enumerate(order):
        lo, hi = az.hdi(draws_2d[:, idx])
        mean = draws_2d[:, idx].mean()
        ax.hlines(pos, lo, hi, color="C0", lw=2, alpha=0.8)
        ax.plot(mean, pos, "o", color="C3", ms=4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([str(labels[i]) for i in order], fontsize=7)
    ax.set_xlabel(xlabel)

fig, axs = plt.subplots(1, 2, figsize=(11, 6))
forest_plot(axs[0], b_draws, subjects, "posterior intercept deviation (ms)")
axs[0].set_title("Posterior subject intercepts (94% HDI)")
forest_plot(axs[1], c_draws, subjects, "posterior slope deviation (ms/day)")
axs[1].set_title("Posterior subject slopes (94% HDI)")
plt.tight_layout(); plt.show()
"""
)

md(
    """### What to Watch For
The posterior scatter is a *cloud per subject*, not a point: each subject's $(b_i, c_i)$ carries uncertainty — this is the practical meaning of "hierarchical Bayes propagates uncertainty". The posterior of $\\sigma_{\\text{slope}}$ replaces the single classical number and silently adds the uncertainty the classical fit could not report. The realized per-draw correlation (around $0$) is consistent with the small classical $\\hat\\rho \\approx 0.07$: for this dataset the independence simplification costs little. If your research question targets $\\rho$ itself, build the correlated model by hand in PyMC (an LKJ prior on the Cholesky factor of the $2\\times2$ covariance) — Bambi's formula interface will not do it for you.

**Where classical and Bayesian worlds agree:** the fixed effects here ($\\beta_0 \\approx 251$, $\\beta_1 \\approx 10.5$ ms/day) match the MixedLM estimates closely, and both match `lme4`. The paradigms differ in what they say *about* the numbers: a point with an approximate SE versus a full joint posterior.
""",
    """### На что обратить внимание
Апостериорное рассеяние — это *облако на каждого испытуемого*, а не точка: $(b_i, c_i)$ каждого участника несёт неопределённость — в этом практический смысл слов «иерархический Байес распространяет неопределённость». Апостериорное распределение $\\sigma_{\\text{наклон}}$ заменяет одиночное классическое число и молча добавляет ту неопределённость, которую классический фит сообщить не мог. Реализованная корреляция на дров (около $0$) согласуется с малым классическим $\\hat\\rho \\approx 0{,}07$: для этих данных упрощение независимости почти ничего не стоит. Если же исследовательский вопрос направлен именно на $\\rho$, стройте коррелированную модель вручную в PyMC (LKJ-приор на фактор Холецкого ковариации $2\\times2$) — формульный интерфейс Bambi этого не сделает.

**Где классический и байесовский миры согласны:** фиксированные эффекты здесь ($\\beta_0 \\approx 251$, $\\beta_1 \\approx 10{,}5$ мс/день) близки к оценкам MixedLM, и обе стороны совпадают с `lme4`. Парадигмы различаются тем, что они говорят *о* числах: точка с приближённой SE против полного совместного апостериорного распределения.
""",
)

# ---------------------------------------------------------------------------
# Section 6: optional homework
# ---------------------------------------------------------------------------
md(
    """## 6. Optional Homework: Growth of Slaughter Pigs (`dietox`)

The legacy course notebook ended with this dataset, and it remains a perfect exercise. The `dietox` data (R package `geepack`) contains the weight of slaughter pigs measured weekly for 12 weeks. The treatments are 3 levels of vitamin E (`Evit`: 0, 100, 200 mg/kg feed) combined with 3 levels of copper (`Cu`: 0, 35, 175 mg/kg feed); cumulated feed intake (`Feed`) is also recorded; pigs are littermates (`Litter`).

**Tasks** (the legacy homework said: "test different models, look beyond time"):
1. Load the data from the book repository (cell below) and explore it: how many pigs, how many weeks, balanced or not?
2. Fit a classical random-intercept model `Weight ~ Time + (1 | Pig)`, then a random-intercept-and-slope model `Weight ~ Time + (Time | Pig)`. Compare them (LOO on the Bambi versions, or AIC on the classical fits — and note which comparison you can defend).
3. Add `Evit`, `Cu` and their interaction as fixed effects. Do the supplements change the *growth rate* or only the *baseline*? Formulate the model that answers each question.
4. Add `(1 | Litter)` — crossed random effects (pigs nested in litters, observations nested in pigs). What happens to the pig-level variance?
5. Report everything on the observable scale (kg per week) with uncertainty, following the course protocol.
""",
    """## 6. Необязательное домашнее задание: рост свиней (`dietox`)

Прежняя учебная тетрадь завершалась этим датасетом, и он остаётся идеальным упражнением. Данные `dietox` (R-пакет `geepack`) содержат вес убойных свиней, измеряемый еженедельно на протяжении 12 недель. Обработки — 3 уровня витамина E (`Evit`: 0, 100, 200 мг/кг корма) в комбинации с 3 уровнями меди (`Cu`: 0, 35, 175 мг/кг корма); накопленное потребление корма (`Feed`) также записано; свиньи — одноклассники по помёту (`Litter`).

**Задания** (в прежнем ДЗ говорилось: «протестируйте разные модели, посмотрите и остальные переменные»):
1. Загрузите данные из репозитория книги (ячейка ниже) и изучите их: сколько свиней, сколько недель, сбалансировано или нет?
2. Подгоните классическую модель со случайным интерсептом `Weight ~ Time + (1 | Pig)`, затем со случайным интерсептом и наклоном `Weight ~ Time + (Time | Pig)`. Сравните их (LOO на Bambi-версиях или AIC на классических фитах — и отметьте, какое сравнение вы можете защитить).
3. Добавьте `Evit`, `Cu` и их взаимодействие как фиксированные эффекты. Меняют ли добавки *скорость роста* или только *базовый уровень*? Сформулируйте модель, отвечающую на каждый из вопросов.
4. Добавьте `(1 | Litter)` — пересекающиеся случайные эффекты (свиньи вложены в помёты, наблюдения — в свиней). Что происходит с дисперсией на уровне свиней?
5. Отчитайтесь обо всём на наблюдаемой шкале (кг в неделю) с неопределённостью, следуя протоколу курса.
""",
)

code(
    """# Homework data starter (no solutions below — this one is yours to build)
PIGS_URL = "https://raw.githubusercontent.com/iknyazeva/bayes-cogsci-book/main/data/dietox.csv"
PIGS_PATH = "data/dietox.csv"
if not os.path.exists(PIGS_PATH):
    import urllib.request
    urllib.request.urlretrieve(PIGS_URL, PIGS_PATH)

pigs = pd.read_csv(PIGS_PATH)
pigs = pigs.drop(columns=["rownames"])
pigs["Pig"] = pigs["Pig"].astype("category")
pigs["Litter"] = pigs["Litter"].astype("category")
print(pigs.head())
print("pigs:", pigs["Pig"].nunique(), "| rows:", len(pigs), "| weeks:", sorted(pigs["Time"].unique()))
"""
)

md(
    """## Summary

- Repeated measures create within-cluster correlation; ignoring it (pooled OLS) produces overconfident standard errors, visible in Durbin–Watson $\\approx 0.7$.
- Subject dummies fix intercepts but overfit, cannot generalize, and force a common slope.
- Mixed models estimate population-level fixed effects *and* variance components — including, in `statsmodels`, the intercept–slope covariance. Their blind spot is uncertainty for variance components and correlations (BLUPs without standard errors, no CI for $\\rho$).
- Bambi delivers the same models as full posteriors, with non-centered internals and weakly informative priors; PSIS-LOO decides between random-intercept and random-slope structures on predictive grounds. Note the formula `(x | g)` fits independent group effects; the intercept–slope correlation is only available in the hand-written PyMC/`lme4` worlds.
- Reference literature: Douglas Bates et al., *lme4*; Belenky et al. (2003), *Journal of Sleep Research*; McElreath, *Statistical Rethinking*, Chapter 13–14.

**Back to the book:** the main Lesson 9 workbook builds the same hierarchy by hand in PyMC; the page *Mixed Models vs. Hierarchical Bayes* condenses this ladder; Lesson 10's workbook adds binary outcomes.
""",
    """## Итоги

- Повторные измерения создают внутрикластерную корреляцию; её игнорирование (объединённый OLS) даёт избыточно самоуверенные стандартные ошибки, что видно по Дарбину–Уотсону $\\approx 0{,}7$.
- Дамми испытуемых чинят интерсепты, но переобучаются, не обобщаются и навязывают общий наклон.
- Смешанные модели оценивают фиксированные эффекты популяции *и* компоненты дисперсии — включая, в `statsmodels`, ковариацию интерсепт–наклон. Их слепое пятно — неопределённость компонент дисперсии и корреляций (BLUP без стандартных ошибок, нет CI для $\\rho$).
- Bambi выдаёт те же модели полными апостериорными распределениями, с безцентровой внутренностью и слабоинформативными приорами; PSIS-LOO решает выбор между структурами со случайным интерсептом и наклоном по предсказательным основаниям. Помните: формула `(x | g)` строит независимые групповые эффекты; корреляция интерсепт–наклон доступна только в мирах PyMC-вручную и `lme4`.
- Литература: Douglas Bates et al., *lme4*; Belenky et al. (2003), *Journal of Sleep Research*; McElreath, *Statistical Rethinking*, главы 13–14.

**Назад к книге:** основная тетрадь занятия 9 строит ту же иерархию вручную в PyMC; страница «Классические смешанные модели и байесовская иерархия» сжимает эту лестницу; тетрадь занятия 10 добавляет бинарные исходы.
""",
)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build() -> None:
    for lang in ["en", "ru"]:
        nb = new_notebook()
        for cell_type, variants in CELLS:
            text = variants[lang]
            if cell_type == "markdown":
                nb.cells.append(new_markdown_cell(text))
            else:
                nb.cells.append(new_code_cell(text))
        target = OUT_DIR / FILES[lang]
        with open(target, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"Wrote {target}")


if __name__ == "__main__":
    build()
