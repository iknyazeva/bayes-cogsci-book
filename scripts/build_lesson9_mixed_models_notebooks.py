"""Build the Lesson 9 Going Further workbook: a faithful transfer of the legacy
course notebook `BayesianModelsMaterials/notebooks/10_MixedModel(Linear).ipynb`.

The Russian workbook reproduces the original notebook cell-by-cell (original
texts, original plots, homework). Only documented corrections are applied:
- Durbin-Watson wording (autocorrelation, not collinearity);
- sigma_b computed from cov_re instead of a hardcoded np.sqrt(1378);
- fixed swapped Obs/Obs band labels in the OLS prediction plot;
- the "CI band" built from joint CI corners is replaced by a correct pointwise
  band using the covariance of the fixed effects;
- statsmodels DOES estimate the random-effect covariance (re_formula="~ Days");
  the model correlation rho is read from cov_re, not from the BLUP scatter;
- az.waic / plot_compare(insample_dev=...) replaced by PSIS-LOO API (ArviZ 1.x);
- seeds for the Bambi fits; pandas-3-compatible groupby; data loads from the
  book repository.
Non-content leftovers removed (np.log demo cells, bare displays, duplicated plot).

Run from the book root with the pymc_env environment:
    /opt/homebrew/Caskroom/miniforge/base/envs/pymc_env/bin/python scripts/build_lesson9_mixed_models_notebooks.py

Outputs:
    notebooks/colab/09_mixed_models_linear.ipynb        (English)
    notebooks/colab/09_mixed_models_linear_ru.ipynb     (Russian, primary transfer)
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


def md(ru: str, en: str | None = None) -> None:
    CELLS.append(("markdown", {"ru": ru, "en": en if en is not None else ru}))


def code(ru: str, en: str | None = None) -> None:
    CELLS.append(("code", {"ru": ru, "en": en if en is not None else ru}))


# ---------------------------------------------------------------------------
# Header (workbook wrapper around the original notebook)
# ---------------------------------------------------------------------------
md(
    f"""# Смешанные модели (mixed models). Они же multilevel models. Они же hierarchical models

[![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)]({GITHUB}{FILES['ru']})
*English version: [{FILES['en']}]({GITHUB}{FILES['en']})*

Эта рабочая тетрадь — приложение к занятию 9 «Повторные наблюдения и иерархические модели» и странице «Классические смешанные модели и байесовская иерархия». В ней мы проходим весь путь по данным о депривации сна `sleepstudy`: от обычной линейной регрессии к моделям со случайными эффектами и далее к байесовским иерархическим моделям в Bambi.
""",
    f"""# Mixed models. Also known as multilevel models, also known as hierarchical models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({GITHUB}{FILES['en']})
*Russian version: [{FILES['ru']}]({GITHUB}{FILES['ru']})*

This workbook accompanies Lesson 9 (Repeated Observations & Hierarchical Models) and the "Classical Mixed Models vs. Hierarchical Bayes" page. Using the `sleepstudy` sleep-deprivation data, we go all the way from ordinary linear regression to random-effects models and then to Bayesian hierarchical models in Bambi.
""",
)

# ---------------------------------------------------------------------------
# Cell 0: imports (original + install guards + repo data loading)
# ---------------------------------------------------------------------------
code(
    """import sys, subprocess

try:
    import pingouin as pg
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pingouin"])
    import pingouin as pg

import arviz as az
import seaborn as sns
import bambi as bmb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.gofplots import qqplot

RANDOM_SEED = 2026

az.style.use("arviz-darkgrid")

# Данные: sleepstudy (Belenky et al., 2003) — каноническая копия в репозитории курса
import os, urllib.request
DATA_PATH = "data/sleepstudy.csv"
os.makedirs("data", exist_ok=True)
if not os.path.exists(DATA_PATH):
    url = "https://raw.githubusercontent.com/iknyazeva/bayes-cogsci-book/main/data/sleepstudy.csv"
    urllib.request.urlretrieve(url, DATA_PATH)
"""
)

# ---------------------------------------------------------------------------
# Cell 3: intro (original text, typos fixed)
# ---------------------------------------------------------------------------
md(
    """Когда говорили про условия, которые должны выполняться для линейной модели, то отмечали отдельно требование независимости данных друг от друга. Это действительно важное требование. Если оно нарушается, то оценки модели будут некорректными и вы не сможете делать выводов.

Но очень часто наблюдения бывают объединены в группы по какому-то признаку:
- одни и те же измерения в разные периоды времени;
- измерения проводятся в нескольких организациях (например, исследования на группах в нескольких школах);
- повторные измерения для одного и того же человека.

То есть любая ситуация, где есть основание предполагать наличие кластера и внутрикластерной корреляции.

Начнём с примера, который есть во многих пакетах.

## Недосып и время реакции

Эксперимент: в ночь перед началом исследования всем испытуемым давали поспать нормальное время, а последующие 9 дней — только по 3 часа. Каждый день измеряли время реакции в серии тестов.

Исследовательский вопрос: как время реакции меняется с продолжительностью бессонницы.
""",
    """When we discussed the conditions that a linear model has to satisfy, we singled out the requirement that observations be independent of each other. This is a genuinely important requirement: when it is violated, the model's estimates are incorrect and no valid conclusions can be drawn.

Very often, however, observations are grouped by some attribute:
- the same measurements taken at different time points;
- measurements collected in several organizations (for example, studies run in several schools);
- repeated measurements for the same person.

That is, any situation where there is reason to expect clusters and within-cluster correlation.

Let us start with an example available in many packages.

## Sleep deprivation and reaction time

Experiment: on the night before the study every subject got a full night of sleep; on each of the following 9 nights only 3 hours. Every day, reaction time was measured in a series of tests.

Research question: how does reaction time change with the duration of sleep deprivation.
""",
)

code(
    """data = pd.read_csv(DATA_PATH)
display(data.head())
data.info()
"""
)

code(
    """data['Subject'] = data.Subject.astype("category")
"""
)

code(
    """data.groupby('Days')[['Reaction']].mean()
"""
)

code(
    """#посмотрим сколько людей и по сколько на каждого приходится значений
plt.figure(figsize=(10,3))
data.Subject.value_counts().plot(kind='bar');
"""
)

code(
    """plt.scatter(data.Days, data.Reaction, color="C0", ec="black", alpha=0.7);
"""
)

code(
    """plt.figure(figsize=(15,8))
ax = sns.scatterplot(x = 'Days', y = 'Reaction', hue = 'Subject', data=data)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
"""
)

code(
    """def plot_data(data):
    fig, axes = plt.subplots(2, 9, figsize=(16, 7.5), sharey=True, sharex=True, dpi=100, constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.075, top=0.925, wspace=0.03)

    axes_flat = axes.ravel()

    for i, subject in enumerate(data["Subject"].unique()):
        ax = axes_flat[i]
        idx = data.index[data["Subject"] == subject].tolist()
        days = data.loc[idx, "Days"].values
        reaction = data.loc[idx, "Reaction"].values

        # Plot observed data points
        ax.scatter(days, reaction, color="C0", ec="black", alpha=0.7)

        # Add a title
        ax.set_title(f"Subject: {subject}", fontsize=14)

    ax.xaxis.set_ticks([0, 2, 4, 6, 8])
    fig.text(0.5, 0.02, "Days", fontsize=14)
    fig.text(0.03, 0.5, "Reaction time (ms)", rotation=90, fontsize=14, va="center")

    return axes
"""
)

code(
    """plot_data(data);
"""
)

md(
    """Наша основная цель — измерить связь между днями депривации сна и временем реакции. Нас интересует как общее воздействие на всех испытуемых, так и воздействие, связанное с каждым отдельным испытуемым. Для таких задач есть специальные модели: иерархические регрессионные модели. В рамках этих моделей можно учесть и общие эффекты, и специфические для каждого испытуемого. Такие эффекты в статистической литературе также известны как фиксированные и случайные эффекты. Но пока попробуем построить обычную линейную модель с помощью уже знакомых инструментов.
""",
    """Our main goal is to measure the relationship between days of sleep deprivation and reaction time. We care both about the overall effect shared by all subjects and about the effect specific to each individual subject. For such tasks there are special models: hierarchical regression models. Within these models one can account for both the common effects and the subject-specific ones. In the statistical literature such effects are also known as fixed and random effects. But first let us fit an ordinary linear model with tools you already know.
""",
)

md(
    """### Обычная линейная модель
- Модель 1 — сначала оценим простую модель времени реакции от дня недосыпа;
- Модель 2 — затем оценим более сложную модель, где добавим субъекта в модель.

#### Модель 1
""",
    """### Ordinary linear model
- Model 1 — first we fit a simple model of reaction time on days of deprivation;
- Model 2 — then a more complex model that adds the subject to the model.

#### Model 1
""",
)

code(
    """lm_basic = smf.ols(formula="Reaction ~ Days", data=data).fit()
print(lm_basic.summary())

print("\\nDurbin-Watson:", round(durbin_watson(lm_basic.resid), 3))
"""
)

md(
    """Если смотреть на табличку, то единственный тревожный звонок — это Durbin-Watson. Важное уточнение: статистика Дарбина–Уотсона проверяет **автокорреляцию остатков** (по порядку строк), а не коллинеарность предикторов. Значение около 2 означает отсутствие автокорреляции; у нас ≈ 0.7, то есть сильная положительная автокорреляция: соседние строки относятся к одному и тому же испытуемому, и их ошибки похожи. Правило большого пальца: значения примерно от 1 до 3 обычно не вызывают тревоги, заметные отклонения от 2 — повод разбираться. Давайте посмотрим на остатки. Если сделать тест Шапиро — гипотеза о нормальности будет отвергнута, но основная проблема здесь в структуре остатков, а не в форме распределения.
""",
    """Looking at the table, the only alarm bell is Durbin-Watson. An important clarification: the Durbin-Watson statistic tests the **autocorrelation of residuals** in row order, not collinearity of predictors. A value near 2 means no autocorrelation; ours is ≈ 0.7 — strong positive autocorrelation: consecutive rows belong to the same subject, so their errors are similar. A rough rule of thumb: values between about 1 and 3 are usually unremarkable, clear departures from 2 call for investigation. Let us look at the residuals. A Shapiro test would reject normality, but the main problem here is the structure of the residuals, not the shape of their distribution.
""",
)

code(
    """fig, axs = plt.subplots(1, 3, figsize = (10, 3))
axs[0].hist(lm_basic.resid)
qqplot(lm_basic.resid, line="s", ax=axs[1], markersize=3)
axs[2].scatter(range(len(lm_basic.resid)), lm_basic.resid, s=8)
plt.show()
pg.normality(lm_basic.resid)
"""
)

code(
    """pred_ols = lm_basic.get_prediction()
iv_l = pred_ols.summary_frame()["obs_ci_lower"]
iv_u = pred_ols.summary_frame()["obs_ci_upper"]

mean_iv_l = pred_ols.summary_frame()["mean_ci_lower"]
mean_iv_u = pred_ols.summary_frame()["mean_ci_upper"]

order = np.argsort(data.Days.values)
fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(data.Days, data.Reaction, color="C0", ec="black", alpha=0.7, label="data")
ax.plot(data.Days.values[order], lm_basic.fittedvalues.values[order], "r--.", label="OLS")
ax.plot(data.Days.values[order], mean_iv_l.values[order], "r--", label='Mean 95% CI: lower')
ax.plot(data.Days.values[order], mean_iv_u.values[order], "r--", label='Mean 95% CI: upper')
ax.plot(data.Days.values[order], iv_l.values[order], "b--", label='Obs 95%: lower')
ax.plot(data.Days.values[order], iv_u.values[order], "g--", label='Obs 95%: upper')
ax.legend(loc="best");
"""
)

md(
    """#### Модель 2

В этой модели добавим для каждого субъекта свою «поправку» для свободного члена в уравнении регрессии (дамми-переменные для субъектов). Универсальность модели теряется: предсказание можно сделать только с учётом конкретного субъекта. Но модель становится лучше, и проблема коррелированности остатков практически решается.
""",
    """#### Model 2

In this model we give every subject their own "correction" to the intercept of the regression equation (dummy variables for subjects). The model loses universality: predictions are only possible for a specific, known subject. But the model improves, and the problem of correlated residuals is essentially resolved.
""",
)

code(
    """lm_sub = smf.ols(formula="Reaction ~ Days + C(Subject)", data=data).fit()
print(lm_sub.summary())
"""
)

code(
    """fig, axs = plt.subplots(1, 3, figsize = (10, 3))
axs[0].hist(lm_sub.resid)
qqplot(lm_sub.resid, line="s", ax=axs[1], markersize=3)
axs[2].scatter(range(len(lm_sub.resid)), lm_sub.resid, s=8)
plt.show()
pg.normality(lm_sub.resid)
"""
)

code(
    """data.groupby('Subject')[['Reaction']].mean()
"""
)

code(
    """plt.scatter(data.Days, lm_sub.fittedvalues)
"""
)

code(
    """#посмотрим на прогноз модели по каждому испытуемому
pred_ols = lm_sub.get_prediction()
data['fitted_sub'] = lm_sub.fittedvalues

fig, ax = plt.subplots(figsize=(15, 8))
ax.scatter(data.Days, data.Reaction, color="C0", ec="black", alpha=0.7)
sns.lineplot(x = 'Days', y = 'fitted_sub', hue = 'Subject', data=data)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
"""
)

md(
    """### Смешанные модели

Смешанная модель — это статистическая модель, которая включает как фиксированные, так и случайные эффекты.

Фиксированные эффекты описывают коэффициенты, общие для всей совокупности наблюдений: например, усреднённое воздействие фактора (дозировка лекарства) или уровень группы (пол, условие эксперимента). Случайные эффекты описывают отклонения, которые меняются от группы к группе (например, индивидуальный сдвиг времени реакции каждого испытуемого) и предполагаются извлечёнными из некоторого распределения. В смешанной модели фиксированные и случайные эффекты оцениваются совместно. Разделение на фиксированные и случайные эффекты достаточно условно и зависит от конкретного эксперимента: один и тот же фактор может входить в модель по-разному в зависимости от вопроса исследования.

#### Модель со случайным смещением

Идея: каждый испытуемый обладает своим собственным уровнем времени реакции, при этом то, насколько может варьироваться это смещение, ограничивается дисперсией случайного эффекта. Модель формулируется так:

$$Reaction_{ij} = \\beta_0 + \\beta_1 Days_{ij} + b_i + \\varepsilon_{ij},$$

где $i$ — номер субъекта, $j$ — индекс дня, и

$$b_i \\sim N(0, \\sigma_b^2), \\qquad \\varepsilon_{ij} \\sim N(0, \\sigma^2).$$

В R модель записывается так:
```R
mod1 <- lmer(Reaction ~ Days + (1 | Subject), data = data)
```
""",
    """### Mixed models

A mixed model is a statistical model that includes both fixed and random effects.

Fixed effects describe coefficients shared by the whole set of observations: for example, an averaged effect of a factor (drug dosage) or the level of a group (sex, experimental condition). Random effects describe deviations that vary from group to group (for example, the individual shift of each subject's reaction time) and are assumed to be drawn from some distribution. In a mixed model, fixed and random effects are estimated jointly. The division into fixed and random effects is fairly conventional and depends on the specific experiment: the same factor can enter the model differently depending on the research question.

#### Random-intercept model

The idea: every subject has their own baseline level of reaction time, while how much these shifts may vary is limited by the variance of the random effect. The model is formulated as

$$Reaction_{ij} = \\beta_0 + \\beta_1 Days_{ij} + b_i + \\varepsilon_{ij},$$

where $i$ indexes subjects, $j$ indexes days, and

$$b_i \\sim N(0, \\sigma_b^2), \\qquad \\varepsilon_{ij} \\sim N(0, \\sigma^2).$$

In R the model is written as
```R
mod1 <- lmer(Reaction ~ Days + (1 | Subject), data = data)
```
""",
)

code(
    """lm_mixed = smf.mixedlm("Reaction ~ Days", data, groups=data["Subject"]).fit()
print(lm_mixed.summary())
"""
)

code(
    """lm_mixed.conf_int()
"""
)

code(
    """# Стандартное отклонение случайного эффекта берём из подогнанной модели, а не из summary вручную
var_subject = lm_mixed.cov_re.iloc[0, 0]
print(f"Стандартное отклонение для распределения коэффициентов по субъектам: sqrt({var_subject:.1f}) = {np.sqrt(var_subject):.2f} мс")
print(f"Остаточное стандартное отклонение: sqrt({lm_mixed.scale:.1f}) = {np.sqrt(lm_mixed.scale):.2f} мс")
print(f"Внутриклассовая корреляция ICC = {var_subject / (var_subject + lm_mixed.scale):.2f}")
"""
)

code(
    """print("Можно посмотреть, как по субъектам сдвигается время реакции")
pd.DataFrame(lm_mixed.random_effects).T.plot(kind='bar');
"""
)

code(
    """#посмотрим на прогноз модели по каждому испытуемому
data['fitted_sub'] = lm_mixed.fittedvalues

fig, ax = plt.subplots(figsize=(15, 8))
ax.scatter(data.Days, data.Reaction, color="C0", ec="black", alpha=0.7)
sns.lineplot(x = 'Days', y = 'fitted_sub', hue = 'Subject', data=data)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
"""
)

code(
    """fig, axs = plt.subplots(1, 3, figsize = (10, 3))
axs[0].hist(lm_mixed.resid)
qqplot(lm_mixed.resid, line="s", ax=axs[1], markersize=3)
axs[2].scatter(range(len(lm_mixed.resid)), lm_mixed.resid, s=8)
plt.show()
pg.normality(lm_mixed.resid)
"""
)

code(
    """print("Посмотрим, как устроены остатки модели по дням и субъектам: по дням остаётся систематический тренд")
data['resids'] = lm_mixed.resid
fig, axs = plt.subplots(1, 2, figsize = (10, 3))
sns.boxplot(x = 'Days', y = 'resids', data=data, ax=axs[0])
sns.boxplot(x = 'Subject', y = 'resids', data=data, ax=axs[1])
axs[1].tick_params(axis="x", labelrotation=90)
plt.show()
"""
)

md(
    """#### Доверительный коридор для популяционной прямой

Полезно построить доверительный коридор для среднего отклика $\\mu(x) = \\beta_0 + \\beta_1 x$. Важная тонкость: нельзя подставлять одновременно оба нижних (или оба верхних) конца доверительных интервалов для $\\beta_0$ и $\\beta_1$ — так получается «прямоугольник» в пространстве параметров, который игнорирует ковариацию оценок и завышает неопределённость. Корректный поточечный коридор использует ковариационную матрицу фиксированных эффектов:

$$\\widehat{\\mathrm{se}}\\big(\\hat\\mu(x)\\big) = \\sqrt{\\mathbf{x}^{\\top} \\, \\widehat{\\operatorname{Cov}}(\\hat\\beta)\\, \\mathbf{x}}, \\qquad \\hat\\mu(x) \\pm z_{0.975}\\,\\widehat{\\mathrm{se}}.$$

Ниже серым пунктиром для сравнения показан и некорректный «угловой» коридор.
""",
    """#### A confidence band for the population line

It is useful to construct a confidence band for the mean response $\\mu(x) = \\beta_0 + \\beta_1 x$. An important subtlety: one must not plug in both lower (or both upper) confidence limits of $\\beta_0$ and $\\beta_1$ at once — that traces the corners of a parameter rectangle, ignores the covariance of the estimates and overstates uncertainty. The correct pointwise band uses the covariance matrix of the fixed effects:

$$\\widehat{\\mathrm{se}}\\big(\\hat\\mu(x)\\big) = \\sqrt{\\mathbf{x}^{\\top} \\, \\widehat{\\operatorname{Cov}}(\\hat\\beta)\\, \\mathbf{x}}, \\qquad \\hat\\mu(x) \\pm z_{0.975}\\,\\widehat{\\mathrm{se}}.$$

Below, the incorrect "corner" band is shown in gray dotted lines for comparison.
""",
)

code(
    """def fixed_effect_ci_band(res, xs):
    # Поточечный доверительный коридор для фиксированных эффектов MixedLM
    X = np.column_stack([np.ones_like(xs), xs])
    beta = res.fe_params.values[:2]
    V = res.cov_params().loc[["Intercept", "Days"], ["Intercept", "Days"]].values
    mean = X @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", X, V, X))
    crit = stats.norm.ppf(0.975)
    return mean, mean - crit * se, mean + crit * se

xs = np.linspace(0, 9, 100)
mean, lo, hi = fixed_effect_ci_band(lm_mixed, xs)

# Некорректный «угловой» коридор для сравнения
conf_ints = lm_mixed.conf_int()
corner_lo = conf_ints.loc['Intercept', 0] + conf_ints.loc['Days', 0] * xs
corner_hi = conf_ints.loc['Intercept', 1] + conf_ints.loc['Days', 1] * xs

fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(data.Days, data.Reaction, color="C0", ec="black", alpha=0.35, s=18, label="data")
ax.plot(xs, mean, "r-", label="FixedMean")
ax.fill_between(xs, lo, hi, color="C3", alpha=0.3, label="корректный 95% коридор")
ax.plot(xs, corner_lo, ":", color="gray", lw=2, label="«угловой» коридор (некорректно)")
ax.plot(xs, corner_hi, ":", color="gray", lw=2)
ax.legend(loc="best")
plt.show()
"""
)

md(
    """#### Модель со случайным смещением и наклоном

Модель стала лучше, но на остатках всё ещё видна проблема: у части испытуемых остатки систематически меняются с днями — скорость нарастания времени реакции у каждого своя. Добавим случайный наклон:

$$Reaction_{ij} = \\beta_0 + \\beta_1 Days_{ij} + b_i + c_{i} Days_{ij} + \\varepsilon_{ij},$$

где

$$b_i \\sim N(0, \\sigma_b^2), \\qquad c_i \\sim N(0, \\sigma_c^2), \\qquad \\varepsilon_{ij} \\sim N(0, \\sigma^2).$$

В R модель записывается так:
```R
mod2 = lmer(Reaction ~ Days + (1 + Days | Subject), data = data)
```

В statsmodels другой синтаксис: случайные наклоны добавляются через `re_formula`. При этом сохраняются все правила формирования формул.
""",
    """#### Random intercept and random slope

The model has improved, but the residuals still show a problem: for some subjects the residuals trend systematically with days — each person has their own rate of slowing. Let us add a random slope:

$$Reaction_{ij} = \\beta_0 + \\beta_1 Days_{ij} + b_i + c_{i} Days_{ij} + \\varepsilon_{ij},$$

where

$$b_i \\sim N(0, \\sigma_b^2), \\qquad c_i \\sim N(0, \\sigma_c^2), \\qquad \\varepsilon_{ij} \\sim N(0, \\sigma^2).$$

In R the model is written as
```R
mod2 = lmer(Reaction ~ Days + (1 + Days | Subject), data = data)
```

In statsmodels the syntax is different: random slopes are added via `re_formula`. All the usual formula conventions still apply.
""",
)

code(
    """lm_mixed = smf.mixedlm("Reaction ~ Days", data, groups=data["Subject"], re_formula="~ Days").fit()
print(lm_mixed.summary())
"""
)

code(
    """print("Теперь у каждого испытуемого два параметра: собственный сдвиг и собственная чувствительность к дням недосыпа")

re = pd.DataFrame(lm_mixed.random_effects)
display(re)
re.T.plot(kind='bar');
"""
)

md(
    """Когда случайных эффектов несколько, возникает вопрос об их связи: если смещение и наклон коррелируют, нужно моделировать ковариацию между ними. Стандартная смешанная модель задаёт для пары $(b_i, c_i)$ двумерное нормальное распределение с корреляцией $\\rho$.

Стоит уточнить распространённое заблуждение: statsmodels с `re_formula="~ Days"` **оценивает** полную ковариационную матрицу случайных эффектов — в summary она видна как `Group x Days Cov`. То есть корреляцию $\\rho$ statsmodels оценивает так же, как `lme4` с формулой `(1 + Days | Subject)`. Ограничение другое: statsmodels не сообщает стандартную ошибку или доверительный интервал для $\\rho$ и других компонент дисперсии.
""",
    """With several random effects, the question of their association arises: if the intercept shift and the slope correlate, the covariance between them has to be modeled. The standard mixed model gives the pair $(b_i, c_i)$ a bivariate normal distribution with correlation $\\rho$.

A common misconception is worth clarifying: statsmodels with `re_formula="~ Days"` **does estimate** the full random-effects covariance matrix — in the summary it appears as `Group x Days Cov`. That is, statsmodels estimates the correlation $\\rho$ just as `lme4` does with the formula `(1 + Days | Subject)`. The limitation lies elsewhere: statsmodels provides no standard error or confidence interval for $\\rho$ or for the other variance components.
""",
)

code(
    """# Корреляция случайных эффектов из модели (а не из диаграммы рассеяния BLUP)
cov_re = lm_mixed.cov_re
print("Оценённая ковариационная матрица случайных эффектов:")
print(cov_re.round(2))

corr_model = cov_re.iloc[0, 1] / np.sqrt(cov_re.iloc[0, 0] * cov_re.iloc[1, 1])
print(f"\\nМодельная оценка корреляции rho = {corr_model:.3f} (без стандартной ошибки)")
print(f"Выборочная корреляция BLUP-оценок (для сравнения) = {re.T['Group'].corr(re.T['Days']):.3f}")
print("BLUP — это условные моды, а не оценки параметров; их выборочная корреляция смешивает сжатие с истинной rho.")
"""
)

code(
    """display(pg.corr(re.T['Group'], re.T['Days']))
sns.scatterplot(x='Group', y = 'Days', data=re.T)
plt.show()
"""
)

code(
    """print("Хоть и видно отклонение от нормальности, но только за счёт нескольких выбросов, а в остальном остатки сильно лучше")
fig, axs = plt.subplots(1, 3, figsize = (10, 3))
axs[0].hist(lm_mixed.resid)
qqplot(lm_mixed.resid, line="s", ax=axs[1], markersize=3)
axs[2].scatter(range(len(lm_mixed.resid)), lm_mixed.resid, s=8)
plt.show()
pg.normality(lm_mixed.resid)
"""
)

code(
    """print("Посмотрим, как устроены остатки модели по дням и субъектам: теперь всё сильно лучше")
data['resids'] = lm_mixed.resid
fig, axs = plt.subplots(1, 2, figsize = (10, 3))
sns.boxplot(x = 'Days', y = 'resids', data=data, ax=axs[0])
sns.boxplot(x = 'Subject', y = 'resids', data=data, ax=axs[1])
axs[1].tick_params(axis="x", labelrotation=90)
plt.show()
"""
)

code(
    """print("Доверительный коридор изменился, в том числе по дням — это важно для выводов")
mean, lo, hi = fixed_effect_ci_band(lm_mixed, xs)
corner_lo = conf_ints.loc['Intercept', 0] + conf_ints.loc['Days', 0] * xs
corner_hi = conf_ints.loc['Intercept', 1] + conf_ints.loc['Days', 1] * xs

fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(data.Days, data.Reaction, color="C0", ec="black", alpha=0.35, s=18, label="data")
ax.plot(xs, mean, "r-", label="FixedMean")
ax.fill_between(xs, lo, hi, color="C3", alpha=0.3, label="корректный 95% коридор")
ax.plot(xs, corner_lo, ":", color="gray", lw=2, label="«угловой» коридор (некорректно)")
ax.plot(xs, corner_hi, ":", color="gray", lw=2)
ax.legend(loc="best")
plt.show()
"""
)

md(
    """## Как тестировать гипотезы и выбирать модели?

- Про выбор моделей — это критерии, основанные на правдоподобии (само правдоподобие, AIC, BIC).
- Наличие эффекта — хочется иметь доверительный интервал и t-статистику, и с этим есть проблемы: получить их для смешанных моделей можно, но оценки приближённые. В пакете `lme4`, чтобы не вводить в заблуждение, уровни значимости даже не приводятся.

Что делать? Есть параметрический бутстреп.

Еще можно использовать байесовские модели: там задача ставится иначе (Монте-Карло или вариационный вывод), и на выходе получаются апостериорные распределения на параметры.
""",
    """## How to test hypotheses and choose models?

- For model choice, these are likelihood-based criteria (the likelihood itself, AIC, BIC).
- For the presence of an effect, one wants a confidence interval and a t-statistic — and here there are problems: they can be obtained for mixed models, but the approximations are rough. To avoid misleading users, the `lme4` package does not report significance levels at all.

What can be done? There is the parametric bootstrap.

One can also use Bayesian models: there the task is posed differently (Monte Carlo or variational inference), and the output consists of posterior distributions over the parameters.
""",
)

md(
    """## Байесовская иерархическая модель

Теория — вне рамок курса; если интересно, см. Richard McElreath (https://xcelab.net/rm/statistical-rethinking/): у него есть лекции, книга, видео. Мы сейчас посмотрим только на то, как это делается на практике.

Благодаря появлению пакета https://bambinos.github.io/bambi/ строить такие модели стало очень просто. Работает та же история с формулами, и синтаксис очень похож на используемый в `lme4`.

Основная идея: все параметры — и фиксированные эффекты, и дисперсии случайных эффектов — получают априорные распределения, а в результате сэмплирования мы получаем апостериорные распределения для каждого из них.

Один нюанс, о котором стоит знать: в Bambi групповые эффекты в формуле вида `(Days | Subject)` по умолчанию считаются **независимыми** — их корреляция не оценивается (в отличие от `lme4` и от смешанной модели statsmodels выше). Для этих данных это не критично: мы уже видели, что модельная оценка корреляции мала (ρ ≈ 0.07). Если корреляция сама по себе — предмет исследования, модель с корреляцией пишут вручную в PyMC.
""",
    """## Bayesian hierarchical model

The theory is outside the scope of the course; if you are curious, see Richard McElreath (https://xcelab.net/rm/statistical-rethinking/): he has lectures, a book, videos. Here we only look at how it is done in practice.

Thanks to the https://bambinos.github.io/bambi/ package, building such models has become very easy. Formulas work the same way, and the syntax closely resembles `lme4`.

The main idea: all parameters — fixed effects and the variances of random effects alike — receive prior distributions, and sampling yields a posterior distribution for each of them.

One caveat worth knowing: in Bambi, group effects in a formula like `(Days | Subject)` are by default **independent** — their correlation is not estimated (unlike `lme4` and unlike the statsmodels mixed model above). For these data this is not critical: we have already seen that the model estimate of the correlation is small (ρ ≈ 0.07). If the correlation itself is the object of study, the correlated model is written by hand in PyMC.
""",
)

code(
    """model1 = bmb.Model("Reaction ~ 1 + Days + (1 | Subject)", data, categorical="Subject")
model1
"""
)

code(
    """idata1 = model1.fit(draws=1000, tune=1000, chains=4, random_seed=RANDOM_SEED,
                    idata_kwargs={"log_likelihood": True})
"""
)

code(
    """az.summary(idata1)
"""
)

code(
    """# Прежняя версия тетради печатала здесь WAIC; в актуальном ArviZ остался критерий PSIS-LOO
print(az.loo(idata1))
"""
)

code(
    """model2 = bmb.Model("Reaction ~ 1 + Days + (Days | Subject)", data, categorical="Subject")
model2
"""
)

code(
    """idata2 = model2.fit(draws=1000, tune=1000, chains=4, random_seed=RANDOM_SEED,
                    idata_kwargs={"log_likelihood": True})
"""
)

code(
    """print(az.loo(idata2))
"""
)

code(
    """models_dict = {
    "model1": idata1,
    "model2": idata2
}
df_compare = az.compare(models_dict)
df_compare
"""
)

code(
    """az.plot_compare(df_compare)
plt.show()
"""
)

code(
    """#проверка коррелированности параметров
_, ax = plt.subplots()

post_ds = idata2.posterior.to_dataset()
post_ds.plot.scatter(
    x="1|Subject", y="Days|Subject",
    hue="Subject__factor_dim",
    add_colorbar=False,
    add_legend=False,
    cmap="tab20",
    s=4, alpha=0.15,
)
ax.set_xlabel("апостериорный сдвиг интерсепта (мс)")
ax.set_ylabel("апостериорный сдвиг наклона (мс/день)")
plt.show()
"""
)

code(
    """# Апостериорные интерсепты и наклоны каждого испытуемого с 94% интервалами
subjects = list(data["Subject"].unique())
b_draws = post_ds["1|Subject"].values.reshape(-1, len(subjects))
c_draws = post_ds["Days|Subject"].values.reshape(-1, len(subjects))

def forest(ax, draws, xlabel):
    order = np.argsort(draws.mean(axis=0))
    for pos, idx in enumerate(order):
        lo, hi = az.hdi(draws[:, idx])
        ax.hlines(pos, lo, hi, color="C0", lw=2, alpha=0.8)
        ax.plot(draws[:, idx].mean(), pos, "o", color="C3", ms=4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([str(subjects[i]) for i in order], fontsize=7)
    ax.set_xlabel(xlabel)

fig, axs = plt.subplots(1, 2, figsize = (10, 6))
forest(axs[0], b_draws, "сдвиг интерсепта (мс)")
axs[0].set_title("Интерсепт каждого испытуемого")
forest(axs[1], c_draws, "сдвиг наклона (мс/день)")
axs[1].set_title("Наклон по дням каждого испытуемого")
plt.tight_layout()
plt.show()
"""
)

md(
    """Обратите внимание на разницу с классической моделью: у каждого испытуемого теперь не точечная оценка (BLUP), а целое апостериорное распределение — со своей неопределённостью. То же самое верно для стандартных отклонений случайных эффектов: классический метод даёт одно число без интервала, байесовский — распределение.

# Домашнее задание
Постройте смешанную модель для данных по росту свиней. Эти данные и результаты можно найти во всех пакетах: https://www.rdocumentation.org/packages/doBy/versions/4.6-2/topics/data-dietox. Но в тьюториалах рассматривается только зависимость от времени, а вы посмотрите и остальные переменные, потестируйте разные модели.

Описание данных:

This data frame contains the following columns:

- Weight: Weight
- Feed: Cumulated feed intake
- Time: Time (in weeks) in the experiment
- Pig: Id of each pig
- Evit: Vitamin E dose
- Cu: Copper dose
- Start: Start weight in experiment, i.e. weight at week 1
- Litter: Id of litter of each pig

Details: Data contains weight of slaughter pigs measured weekly for 12 weeks. Data also contains the start weight (i.e. the weight at week 1). The treatments are 3 different levels of Evit = vitamin E (dose: 0, 100, 200 mg dl-alpha-tocopheryl acetat /kg feed) in combination with 3 different levels of Cu = copper (dose: 0, 35, 175 mg/kg feed) in the feed. The cumulated feed intake is also recorded. The pigs are littermates.
""",
    """Note the difference from the classical model: each subject now has not a point estimate (a BLUP) but a full posterior distribution — with its own uncertainty. The same holds for the standard deviations of the random effects: the classical method gives one number without an interval, the Bayesian one gives a distribution.

# Homework
Build a mixed model for the pig growth data. These data and results can be found in most packages: https://www.rdocumentation.org/packages/doBy/versions/4.6-2/topics/data-dietox. Tutorials only consider the dependence on time, but you should also look at the other variables and try different models.

Data description:

This data frame contains the following columns:

- Weight: Weight
- Feed: Cumulated feed intake
- Time: Time (in weeks) in the experiment
- Pig: Id of each pig
- Evit: Vitamin E dose
- Cu: Copper dose
- Start: Start weight in experiment, i.e. weight at week 1
- Litter: Id of litter of each pig

Details: Data contains weight of slaughter pigs measured weekly for 12 weeks. Data also contains the start weight (i.e. the weight at week 1). The treatments are 3 different levels of Evit = vitamin E (dose: 0, 100, 200 mg dl-alpha-tocopheryl acetat /kg feed) in combination with 3 different levels of Cu = copper (dose: 0, 35, 175 mg/kg feed) in the feed. The cumulated feed intake is also recorded. The pigs are littermates.
""",
)

code(
    """# Данные dietox — каноническая копия в репозитории курса
PIGS_PATH = "data/dietox.csv"
if not os.path.exists(PIGS_PATH):
    url_p = "https://raw.githubusercontent.com/iknyazeva/bayes-cogsci-book/main/data/dietox.csv"
    urllib.request.urlretrieve(url_p, PIGS_PATH)
data_pigs = pd.read_csv(PIGS_PATH)
"""
)

code(
    """data_pigs.head()
"""
)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build() -> None:
    for lang in ["ru", "en"]:
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
