"""Build the Lesson 10 Going Further workbook: a faithful transfer of the legacy
course notebook `BayesianModelsMaterials/notebooks/11_MixedModel(Logistic).ipynb`.

The Russian workbook reproduces the original notebook cell-by-cell (original
texts, original plots, homework). Only documented corrections are applied:
- title: "mixed LINEAR models for binary data" -> mixed models for binary
  outcomes (GLMM);
- Depression.dat is loaded from the book repository (the original
  ./file_examples/ path no longer exists; the file is restored from a Wayback
  snapshot of Agresti's original source);
- ANES is loaded from the book repository (clinton/trump subset, the same rows
  the original notebook filters to);
- the Clinton fit: draws 200 -> 1000, chains=4, fixed seed;
- az.waic -> az.loo (ArviZ 1.x no longer ships WAIC);
- the model3 summary is restricted to population-level parameters (the full
  table of 340 case effects is not informative);
- the unverifiable "starfish" homework pointer is replaced by the chimpanzees
  task (the dataset is the Lesson 10 capstone of this book).

Run from the book root with the pymc_env environment:
    /opt/homebrew/Caskroom/miniforge/base/envs/pymc_env/bin/python scripts/build_lesson10_mixed_models_notebooks.py

Outputs:
    notebooks/colab/10_mixed_models_logistic.ipynb        (English)
    notebooks/colab/10_mixed_models_logistic_ru.ipynb     (Russian, primary transfer)
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
    "en": "10_mixed_models_logistic.ipynb",
    "ru": "10_mixed_models_logistic_ru.ipynb",
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
    f"""# Смешанные модели для бинарных данных (GLMM)

[![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)]({GITHUB}{FILES['ru']})
*English version: [{FILES['en']}]({GITHUB}{FILES['en']})*

Эта рабочая тетрадь — приложение к занятию 10 «Бинарные исходы и обобщённые линейкие модели» и странице «Классические смешанные модели и байесовская иерархия». Мы разберём продольные бинарные данные из клинического испытания лечения депрессии: от обычной логистической регрессии к смешанной модели со случайным интерсептом пациента, сначала в statsmodels, затем в Bambi, и сравним модели по PSIS-LOO. В конце — пример о неоднородных группах без повторений: голосование на выборах ANES 2016.
""",
    f"""# Mixed models for binary outcomes (GLMM)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({GITHUB}{FILES['en']})
*Russian version: [{FILES['ru']}]({GITHUB}{FILES['ru']})*

This workbook accompanies Lesson 10 (Binary Outcomes and Generalized Linear Models) and the "Classical Mixed Models vs. Hierarchical Bayes" page. We work through longitudinal binary data from a depression drug trial: from ordinary logistic regression to a mixed model with a patient random intercept — first in statsmodels, then in Bambi — and compare the models by PSIS-LOO. At the end comes an example of heterogeneous groups without repetition: voting in the ANES 2016 survey.
""",
)

# ---------------------------------------------------------------------------
# Cell 0: title/intro of the original notebook (cells 0-1), data description
# ---------------------------------------------------------------------------
md(
    """## Продольное исследование лечения депрессии

Table 9.1 refers to a longitudinal study comparing a new drug with a standard drug for treating subjects suffering mental depression. Subjects were classified into two groups according to whether the initial severity of depression was mild or severe. In each group, subjects were randomly assigned to one of the two drugs. Following 1 week, 2 weeks, and 4 weeks of treatment, each subject's extent of suffering from mental depression was classified as normal or abnormal.
- $y_t$ = 1 for normal and 0 for abnormal at time $t$.

Пусть $s$ обозначает исходную тяжесть состояния: $s = 1$ — тяжёлая, $s = 0$ — лёгкая. Пусть $d$ обозначает препарат: $d = 1$ — новый, $d = 0$ — стандартный. Пусть $t$ — момент измерения. Когда шкала времени отражает накопленную дозу препарата, на логит-шкале эффект часто приблизительно линеен по логарифму времени. Используются баллы $(0, 1, 2)$ — логарифмы номеров недель $(1, 2, 4)$ по основанию 2. С самими номерами недель получаются похожие содержательные результаты.

(Источник данных: Agresti, *Categorical Data Analysis*, таблица 9.1. Файл `Depression.dat` входит в репозиторий курса; оригинал распространялся по адресу users.stat.ufl.edu/~aa/cat/data/Depression.dat.)
""",
    """## A longitudinal study of depression treatment

Table 9.1 refers to a longitudinal study comparing a new drug with a standard drug for treating subjects suffering mental depression. Subjects were classified into two groups according to whether the initial severity of depression was mild or severe. In each group, subjects were randomly assigned to one of the two drugs. Following 1 week, 2 weeks, and 4 weeks of treatment, each subject's extent of suffering from mental depression was classified as normal or abnormal.
- $y_t$ = 1 for normal and 0 for abnormal at time $t$.

Let $s$ denote the initial severity of depression, with $s = 1$ for severe and $s = 0$ for mild. Let $d$ denote the drug, with $d = 1$ for new and $d = 0$ for standard. Let $t$ denote the time of measurement. When the time metric reflects cumulative drug dosage, a logit scale often has an approximate linear effect for the logarithm of time. We use scores $(0, 1, 2)$, the logs to base 2 of the week numbers $(1, 2, 4)$. Similar substantive results occur using the week numbers themselves.

(Data source: Agresti, *Categorical Data Analysis*, Table 9.1. The `Depression.dat` file ships with the course repository; the original was distributed at users.stat.ufl.edu/~aa/cat/data/Depression.dat.)
""",
)

code(
    """import os, urllib.request
import pandas as pd
import arviz as az
import seaborn as sns
import bambi as bmb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

az.style.use("arviz-darkgrid")

RANDOM_SEED = 2026

# Данные — из репозитория курса (оригинальный ./file_examples/ больше не существует)
DEPR_PATH = "data/Depression.dat"
os.makedirs("data", exist_ok=True)
if not os.path.exists(DEPR_PATH):
    url = "https://raw.githubusercontent.com/iknyazeva/bayes-cogsci-book/main/data/Depression.dat"
    urllib.request.urlretrieve(url, DEPR_PATH)
"""
)

code(
    """data_depr = pd.read_csv(DEPR_PATH, sep=r"\\s+")
data_depr.head()
"""
)

code(
    """agg_data = data_depr.groupby(['drug', 'severity', 'time'])['outcome'].mean().reset_index()
agg_data
"""
)

code(
    """sns.catplot(x="time", y="outcome", hue="drug", col='severity', data=agg_data, kind="bar",
            palette="rocket", legend_out=False)
plt.show()
"""
)

code(
    """basic = smf.logit(formula = 'outcome ~ drug', data = data_depr).fit()
print(basic.summary())
"""
)

code(
    """fig, axs = plt.subplots(1, 2, figsize = (10, 3))
axs[0].scatter(range(len(data_depr)), basic.resid_generalized)
axs[1].scatter(basic.fittedvalues, basic.resid_generalized)
plt.show()
"""
)

md(
    """У модели только с препаратом остатки структурированы: модель не учитывает ни тяжесть состояния, ни время. Добавим в модель все имеющиеся предикторы и взаимодействие препарата со временем.
""",
    """With the drug-only model, the residuals are structured: the model ignores both the severity of the condition and time. Let us add all available predictors and the drug-by-time interaction.
""",
)

code(
    """basic = smf.logit(formula = 'outcome ~ drug + time + severity + drug:time', data = data_depr).fit()
print(basic.summary())
"""
)

code(
    """fig, axs = plt.subplots(1, 2, figsize = (10, 3))
axs[0].scatter(range(len(data_depr)), basic.resid_generalized)
axs[1].scatter(basic.fittedvalues, basic.resid_generalized)
plt.show()
"""
)

md(
    """Обратите внимание: в полной модели главный эффект препарата почти нулевой ($\\approx -0.06$), а взаимодействие препарата со временем большое ($\\approx +1.02$). Это содержательный результат: на старте препараты не различаются, различие нарастает с накопленной дозой. Главное влияние без взаимодействия здесь ничего не значит.

У бинарных данных остатки всегда образуют две полосы (возможные значения $y - p$ ограничены), поэтому сам по себе такой график малоинформативен; важно, что плоская модель в принципе не учитывает повторяемость наблюдений одного пациента — 1020 строк ведут себя как 1020 независимых людей, и стандартные ошибки занижены.
""",
    """Note: in the full model the main effect of the drug is nearly zero ($\\approx -0.06$), while the drug-by-time interaction is large ($\\approx +1.02$). This is a substantive result: at baseline the drugs do not differ, and the difference grows with cumulative dose. A main effect without its interaction means nothing here.

For binary data the residuals always form two bands (the possible values of $y - p$ are bounded), so by itself this plot is not very informative; the key point is that the flat model does not account for the repetition of observations within a patient — the 1020 rows behave like 1020 independent people, and the standard errors are too small.
""",
)

code(
    """from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
"""
)

code(
    """random = {'a': '0 + C(case)', }
mixed = BinomialBayesMixedGLM.from_formula(
    'outcome ~ drug + time + severity + drug:time', random, data_depr)
result = mixed.fit_vb()
"""
)

code(
    """print(result.summary())
"""
)

code(
    """result.random_effects().head()
"""
)

md(
    """Если посмотреть на точечные оценки, то кажется, что всё одинаковое. Проблема в том, что не оценить качество, и выводы сложно делать какие-то, особенно количественные. Насколько? Какой размер эффекта? Какая неопределённость? Вариационное приближение в statsmodels даёт приближённые средние и стандартные отклонения, но не даёт ни сэмплов апостериорного распределения, ни оценок его качества, ни распределения для дисперсии случайных эффектов — только одно число.
""",
    """Looking at the point estimates, everything seems the same. The problem is that quality cannot be assessed, and quantitative conclusions are hard to draw. By how much? What effect size? What uncertainty? The variational approximation in statsmodels returns approximate means and standard deviations, but no posterior samples, no assessment of its own quality, and no distribution for the random-effects variance — just one number.
""",
)

md(
    """## Байесовские смешанные модели в Bambi

Теперь те же модели в Bambi: сначала плоская модель только с препаратом, затем полная плоская модель, затем полная модель со случайным интерсептом пациента `(1 | case)`. Сравним их по PSIS-LOO (в прежней версии тетради использовался WAIC; в актуальном ArviZ остался критерий LOO).
""",
    """## Bayesian mixed models in Bambi

Now the same models in Bambi: first the flat drug-only model, then the full flat model, then the full model with a patient random intercept `(1 | case)`. We compare them by PSIS-LOO (the earlier version of this notebook used WAIC; current ArviZ ships only LOO).
""",
)

code(
    """model1 = bmb.Model("outcome ~ drug", data_depr, family="bernoulli")
model1
"""
)

code(
    """idata1 = model1.fit(draws=1000, tune=1000, chains=4, random_seed=RANDOM_SEED,
                     idata_kwargs={"log_likelihood": True})
"""
)

code(
    """print(az.loo(idata1))
az.summary(idata1)
"""
)

code(
    """model2 = bmb.Model("outcome ~ drug + time + severity + drug:time", data_depr, family="bernoulli")
idata2 = model2.fit(draws=1000, tune=1000, chains=4, random_seed=RANDOM_SEED,
                    idata_kwargs={"log_likelihood": True})
print(az.loo(idata2))
az.summary(idata2)
"""
)

code(
    """model3 = bmb.Model("outcome ~ drug + time + severity + drug:time + (1 | case)",
                    data_depr, family="bernoulli")
idata3 = model3.fit(draws=1000, tune=1000, chains=4, random_seed=RANDOM_SEED,
                    target_accept=0.95,
                    idata_kwargs={"log_likelihood": True})
print(az.loo(idata3))
# Полная сводка по 340 пациентским эффектам неинформативна — смотрим популяционные параметры
az.summary(idata3, var_names=["Intercept", "drug", "time", "severity", "drug:time", "1|case_sigma"])
"""
)

code(
    """model3
"""
)

code(
    """models_dict = {
    "model1: drug": idata1,
    "model2: полная плоская": idata2,
    "model3: + (1|case)": idata3,
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

md(
    """Как читать этот результат. Полная плоская модель и смешанная модель практически неразличимы по предсказательной точности (разница меньше одной единицы ELPD). Причина понятна: на каждого пациента приходится всего три бинарных наблюдения, и данные не позволяют отличить «пациенты не различаются» от «различия есть, но небольшие». Апостериорное распределение `1|case_sigma` сосредоточено около нуля (среднее $\\approx 0.3$), то есть частичное объединение вырождается в полное. Это не значит, что иерархические модели бесполезны: на данных `sleepstudy` из тетради занятия 9, где на каждого испытуемого приходится десять непрерывных наблюдений, иерархическая структура выигрывала уверенно. Иерархия окупается тогда, когда кластеры информативны.

Отдельно стоит сказать про графики для содержательного вывода: коэффициенты логистической регрессии живут на шкале лог-шансов, и их нужно переводить в вероятности при явно названных условиях. Например, для типичного пациента к четвёртой неделе новый препарат повышает вероятность нормальной классификации примерно на $+0.23$ для исходно лёгкой формы и на $+0.43$ для тяжёлой (94% интервалы $[+0.17, +0.29]$ и $[+0.34, +0.51]$). Соответствующие расчёты есть на странице «Классические смешанные модели и байесовская иерархия».
""",
    """How to read this result. The full flat model and the mixed model are practically indistinguishable in predictive accuracy (the difference is below one ELPD unit). The reason is clear: each patient contributes only three binary observations, and the data cannot tell "patients do not differ" apart from "there are differences, but small ones". The posterior of `1|case_sigma` concentrates near zero (mean $\\approx 0.3$), so partial pooling degenerates into complete pooling. This does not mean hierarchical models are useless: on the `sleepstudy` data of the Lesson 9 workbook, where each subject contributes ten continuous observations, the hierarchical structure won decisively. Hierarchy pays off when clusters are informative.

A separate note about plots for substantive conclusions: logistic-regression coefficients live on the log-odds scale and have to be translated into probabilities at explicitly named conditions. For example, for a typical patient by week 4 the new drug raises the probability of a normal classification by about $+0.23$ in the initially mild group and $+0.43$ in the severe group (94% intervals $[+0.17, +0.29]$ and $[+0.34, +0.51]$). The corresponding computations are on the "Classical Mixed Models vs. Hierarchical Bayes" page.
""",
)

md(
    """## Пример без повторений: голосование на выборах ANES 2016

Не все бинарные данные продольные. Смешанно-модельный аппарат полезен и для неоднородных групп в одном срезе. В данных ANES 2016 моделируем голосование за Клинтон: зависимая переменная `vote['clinton']` означает, что успехом считается голос за Клинтона. Формула `vote['clinton'] ~ party_id + party_id:age` даёт каждой партии собственный наклон возраста: взаимодействие `party_id:age` разворачивается в три коэффициента (по одному на демократов, независимых и республиканцев), поэтому отдельный главный эффект возраста не нужен и не пропущен.
""",
    """## An example without repetition: voting in ANES 2016

Not all binary data are longitudinal. The mixed-model toolkit is also useful for heterogeneous groups in a single cross-section. In the ANES 2016 data we model the Clinton vote: the response `vote['clinton']` means that a Clinton vote is counted as success. The formula `vote['clinton'] ~ party_id + party_id:age` gives every party its own age slope: the `party_id:age` interaction expands into three coefficients (one for democrats, independents and republicans), so a separate main effect of age is not needed and not missing.
""",
)

code(
    """# Подмножество ANES 2016 (голоса clinton/trump) — из репозитория курса
ANES_PATH = "data/anes_clinton.csv"
if not os.path.exists(ANES_PATH):
    url_a = "https://raw.githubusercontent.com/iknyazeva/bayes-cogsci-book/main/data/anes_clinton.csv"
    urllib.request.urlretrieve(url_a, ANES_PATH)
data = pd.read_csv(ANES_PATH)
data.head()
"""
)

code(
    """clinton_data = data.loc[data["vote"].isin(["clinton", "trump"]), :]
clinton_data.head()
"""
)

code(
    """clinton_model = bmb.Model("vote['clinton'] ~ party_id + party_id:age", clinton_data, family="bernoulli")
clinton_fitted = clinton_model.fit(
    draws=1000, tune=1000, chains=4, random_seed=RANDOM_SEED, target_accept=0.85,
    idata_kwargs={"log_likelihood": True}
)
"""
)

code(
    """az.summary(clinton_fitted)
"""
)

code(
    """# Наклоны возраста по партиям на шкале лог-шансов
post = clinton_fitted.posterior.to_dataset()
party_levels = ["democrat", "independent", "republican"]
slopes = {p: post["party_id:age"].sel(**{"party_id:age_dim": p}).values.flatten()
          for p in party_levels}

fig, ax = plt.subplots(figsize=(7, 3.8))
for i, p in enumerate(party_levels):
    ax.hist(slopes[p], bins=60, alpha=0.55, density=True, label=p, color=f"C{i}")
ax.set_xlabel("наклон возраста (лог-шансы за год)")
ax.legend()
plt.show()

# Те же эффекты на шкале вероятностей: изменение вероятности за десять лет возраста
# при собственной базовой вероятности каждой партии
for p in party_levels:
    eta = post["Intercept"].values.flatten()
    if p != "democrat":   # democrat — референтный уровень, входит в интерсепт
        eta = eta + post["party_id"].sel(**{"party_id_dim": p}).values.flatten()
    p_hat = 1 / (1 + np.exp(-eta))
    delta10 = slopes[p] * 10 * p_hat * (1 - p_hat)
    lo, hi = az.hdi(delta10, prob=0.94)
    print(f"{p:12s}: изменение P(голос за Клинтон) за 10 лет возраста = {delta10.mean():+.3f}, "
          f"94% HDI [{lo:+.3f}, {hi:+.3f}]")
"""
)

md(
    """Наклоны у партий почти не пересекаются: с возрастом вероятность голоса за Клинтона заметно падает у республиканцев и независимых, а у демократов близка к нулю. Один «средний» эффект возраста скрыл бы всю картину — взаимодействия делают неоднородность видимой.

## Домашнее задание

Примените разобранный сегодня подход к данным о просоциальных шимпанзе (капстоун занятия 10, файл `data/chimpanzees.csv`):

- постройте плоскую логистическую модель `pulled_left ~ prosoc_left * condition`, игнорируя акторов и блоки;
- постройте смешанную модель `pulled_left ~ prosoc_left * condition + (1 | actor) + (1 | block)`;
- сравните модели по PSIS-LOO и отчитайтесь о популяционном эффекте просоциальной опции с неопределённостью;
- одно предложение: что даёт включение `(1 | block)` и поддерживает ли его таблица LOO.
""",
    """The parties' slopes barely overlap: with age, the probability of a Clinton vote falls markedly among republicans and independents, while among democrats it is close to zero. A single "average" age effect would have hidden the whole picture — interactions make heterogeneity visible.

## Homework

Apply today's approach to the prosocial chimpanzees data (the Lesson 10 capstone, file `data/chimpanzees.csv`):

- fit the flat logistic model `pulled_left ~ prosoc_left * condition`, ignoring actors and blocks;
- fit the mixed model `pulled_left ~ prosoc_left * condition + (1 | actor) + (1 | block)`;
- compare the models by PSIS-LOO and report the population-level effect of the prosocial option with its uncertainty;
- one sentence: what does including `(1 | block)` buy, and does the LOO table support it?
""",
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
