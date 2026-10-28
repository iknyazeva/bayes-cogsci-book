"""Build the Lesson 10 Going Further workbook: GLMMs for binary longitudinal data.

Full transfer of the legacy course notebook `11_MixedModel(Logistic).ipynb`
(BayesianModelsMaterials) after the 2026-10-28 revision: the Agresti depression
drug trial (flat logit -> full logit -> statsmodels VB mixed GLMM -> Bambi
hierarchical GLMM with PSIS-LOO comparison), the ANES 2016 Clinton/Trump vote
model with party-specific age slopes, and an optional chimpanzees homework
cross-linked to the Lesson 10 capstone.

Run from the book root with the pymc_env environment:
    /opt/homebrew/Caskroom/miniforge/base/envs/pymc_env/bin/python scripts/build_lesson10_mixed_models_notebooks.py

Outputs:
    notebooks/colab/10_mixed_models_logistic.ipynb
    notebooks/colab/10_mixed_models_logistic_ru.ipynb
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


def md(en: str, ru: str | None = None) -> None:
    CELLS.append(("markdown", {"en": en, "ru": ru if ru is not None else en}))


def code(en: str, ru: str | None = None) -> None:
    CELLS.append(("code", {"en": en, "ru": ru if ru is not None else en}))


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
md(
    f"""# Lesson 10 (Going Further): Mixed Models for Binary Data — GLMMs
### Student Laboratory Workbook
*Course: Bayesian Analysis of Empirical Data (2026)*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({GITHUB}{FILES['en']})
*Russian version: [{FILES['ru']}]({GITHUB}{FILES['ru']})*

Lesson 10 taught one binary model per observation. Real binary data, however, is often *longitudinal*: the same patient answers at several visits, the same citizen is tracked across waves. This companion to the *Mixed Models vs. Hierarchical Bayes* page adds the random-effects machinery to the logistic link, on two datasets: the classic **depression drug trial** from Agresti's *Categorical Data Analysis* (340 patients, three follow-ups) and the **ANES 2016** vote data. We climb the same ladder as in Lesson 9's linear workbook — flat model, naive flat model, classical GLMM, Bayesian hierarchical GLMM — and let PSIS-LOO arbitrate.
""",
    f"""# Занятие 10 (дополнительно): смешанные модели для бинарных данных — GLMM
### Рабочая тетрадь студента
*Курс: Байесовский анализ эмпирических данных (2026)*

[![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)]({GITHUB}{FILES['ru']})
*English version: [{FILES['en']}]({GITHUB}{FILES['en']})*

На занятии 10 мы строили одну бинарную модель на наблюдение. Реальные бинарные данные часто *продольные*: один пациент отвечает на нескольких визитах, один избиратель наблюдается на нескольких волнах. Это приложение к странице «Классические смешанные модели и байесовская иерархия» добавляет аппарат случайных эффектов к логистической связи на двух датасетах: классическое **исследование лечения депрессии** из книги Агрести *Categorical Data Analysis* (340 пациентов, три визита) и данные голосования **ANES 2016**. Мы пройдём ту же лестницу, что и в линейной тетради занятия 9 — плоская модель, наивная плоская модель, классический GLMM, байесовский иерархический GLMM, — и позволим PSIS-LOO рассудить.
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

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

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

def fetch(fname, url):
    # Canonical copy in the book repository; auto-download in Colab or when missing.
    path = os.path.join("data", fname)
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(path):
        import urllib.request
        urllib.request.urlretrieve(url, path)
    return path

BASE = "https://raw.githubusercontent.com/iknyazeva/bayes-cogsci-book/main/data/"
DEPR_PATH = fetch("Depression.dat", BASE + "Depression.dat")
ANES_PATH = fetch("anes_clinton.csv", BASE + "anes_clinton.csv")

print("Environment ready: statsmodels", sm.__version__, "| Bambi", bmb.__version__, "| ArviZ", az.__version__)
"""
)

# ---------------------------------------------------------------------------
# Section 1: the depression trial data
# ---------------------------------------------------------------------------
md(
    """## 1. A Longitudinal Binary Outcome: the Depression Drug Trial

### The Idea
The study (distributed as Table 9.1 in Agresti's *Categorical Data Analysis* materials) is a longitudinal clinical trial comparing a **new drug** with a **standard** drug for subjects suffering mental depression. Subjects were classified by initial severity — *mild* or *severe* — randomly assigned to one of the two drugs, and their status was classified as *normal* (1) or *abnormal* (0) after 1, 2 and 4 weeks of treatment.

A modeling detail that matters: when the time metric reflects **cumulative dosage**, a logit model often behaves approximately linearly in the **logarithm of time**. Weeks 1, 2, 4 give the tidy scores $t = \\log_2(\\text{week}) = 0, 1, 2$. Agresti reports similar substantive results with raw week numbers; the log scores are the theoretically motivated default.

### Example
""",
    """## 1. Продольный бинарный исход: исследование лечения депрессии

### Идея
Исследование (распространяется как таблица 9.1 в материалах книги Агрести *Categorical Data Analysis*) — продольное клиническое испытание, сравнивающее **новый препарат** со **стандартным** у пациентов с депрессией. Пациентов классифицировали по исходной тяжести — *лёгкая* или *тяжёлая* — случайно назначали один из двух препаратов и классифицировали состояние как *норма* (1) или *патология* (0) после 1, 2 и 4 недель лечения.

Важная деталь моделирования: когда шкала времени отражает **накопленную дозу**, логит-модель часто ведёт себя приблизительно линейно по **логарифму времени**. Недели 1, 2, 4 дают аккуратные баллы $t = \\log_2(\\text{неделя}) = 0, 1, 2$. Агрести отмечает, что с номерами недель получаются похожие содержательные результаты; лог-баллы — теоретически мотивированный выбор по умолчанию.

### Пример
""",
)

code(
    """data_depr = pd.read_csv(DEPR_PATH, sep=r"\\s+")
print(data_depr.head())
print("\\nShape:", data_depr.shape,
      "| cases:", data_depr["case"].nunique(),
      "| rows per case:", sorted(data_depr.groupby("case").size().unique()))
print("severity: 0 = mild, 1 = severe | drug: 0 = standard, 1 = new | time: log2(week) in {0,1,2}")

# Descriptive proportion of 'normal' responses by cell
agg = data_depr.groupby(["severity", "drug", "time"])["outcome"].mean().reset_index()
print("\\nShare of normal outcomes by cell:")
print(agg.pivot(index=["severity", "drug"], columns="time", values="outcome").round(3))

sns.catplot(x="time", y="outcome", hue="drug", col="severity", data=agg, kind="bar",
            palette="rocket", legend_out=False, height=3.6, aspect=1.1)
plt.suptitle("Proportion normal by week, drug and baseline severity", y=1.05)
plt.show()
"""
)

md(
    """### What to Watch For
Read the four panels as a $2 \\times 2$ factorial story:
- Both arms start near their baseline severity level (mild $\\approx 0{.}5$, severe $\\approx 0{.}2$).
- Under the **standard** drug improvement is slow and modest in both severity groups.
- Under the **new** drug the curves take off: by week 4 (t=2) 97% of initially mild and 83% of initially severe patients are classified normal.
- The gap between drugs *grows with time* — a first hint at a **drug × time interaction** on the logit scale, which the flat models below will formalize.

And the structural point: the same `case` id appears three times. These 1020 rows are not 1020 independent people; they are 340 people measured thrice.
""",
    """### На что обратить внимание
Читаем четыре панели как факторную историю $2 \\times 2$:
- Обе группы начинают около уровня своей исходной тяжести (лёгкая $\\approx 0{,}5$, тяжёлая $\\approx 0{,}2$).
- На **стандартном** препарате улучшение медленное и скромное в обеих группах тяжести.
- На **новом** препарате кривые взлетают: к 4-й неделе (t=2) 97% исходно лёгких и 83% исходно тяжёлых пациентов классифицированы как норма.
- Разрыв между препаратами *растёт со временем* — первый намёк на **взаимодействие препарат × время** на логит-шкале, которое формализуют плоские модели ниже.

И структурный момент: один и тот же `case` встречается трижды. Эти 1020 строк — не 1020 независимых людей; это 340 человек, измеренных трижды.
""",
)

# ---------------------------------------------------------------------------
# Section 2: flat logistic models
# ---------------------------------------------------------------------------
md(
    """## 2. Flat Logistic Models: the Naive Baselines

### The Idea
Pretend all rows are independent and fit ordinary logistic regressions — first with `drug` only, then with the full fixed structure `drug + time + severity + drug:time`. This is exactly the "complete pooling" extreme from Lesson 9, now on a binary outcome. The comparison between the two fits already tells the substantive story; the residual plots show why "residuals" for binary data must be read with care.

### Example
""",
    """## 2. Плоские логистические модели: наивные базовые уровни

### Идея
Сделаем вид, что все строки независимы, и подгоним обычные логистические регрессии — сначала только с `drug`, затем с полной фиксированной структурой `drug + time + severity + drug:time`. Это ровно крайность «полного объединения» из занятия 9, теперь на бинарном исходе. Сравнение двух подгонок уже рассказывает содержательную историю; графики остатков показывают, почему «остатки» бинарных данных нужно читать осторожно.

### Пример
""",
)

code(
    """basic1 = smf.logit(formula="outcome ~ drug", data=data_depr).fit(disp=0)
print(basic1.summary())
print("\\nOdds ratio for drug:", round(float(np.exp(basic1.params["drug"])), 2))
"""
)

code(
    """basic2 = smf.logit(formula="outcome ~ drug + time + severity + drug:time", data=data_depr).fit(disp=0)
print(basic2.summary())

# Binned residuals: for binary y, raw residuals always form two bands — bin them first.
def binned_residuals(res, data, x="fitted", n_bins=10, ax=None):
    pred = res.fittedvalues if x == "fitted" else data[x]
    df = pd.DataFrame({"x": pred, "r": res.resid_generalized})
    if df["x"].nunique() <= n_bins:
        df["bin"] = df["x"]          # discrete designs (e.g. drug-only): bin by fitted value
    else:
        df["bin"] = pd.qcut(df["x"], n_bins)
    binned = df.groupby("bin", observed=True).mean(numeric_only=True)
    ax.plot(binned["x"], binned["r"], "o-", color="C3")
    ax.axhline(0, color="gray", lw=0.8)
    return binned

fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
axs[0].scatter(basic1.fittedvalues, basic1.resid_generalized, s=8, alpha=0.5, color="C0")
axs[0].set_title("drug-only: raw residuals vs fitted")
binned_residuals(basic1, data_depr, ax=axs[1])
axs[1].set_title("drug-only: binned mean residuals")
for ax in axs: ax.set_xlabel("fitted probability")
plt.tight_layout(); plt.show()

fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
axs[0].scatter(basic2.fittedvalues, basic2.resid_generalized, s=8, alpha=0.5, color="C0")
axs[0].set_title("full flat model: raw residuals vs fitted")
binned_residuals(basic2, data_depr, ax=axs[1])
axs[1].set_title("full flat model: binned mean residuals")
for ax in axs: ax.set_xlabel("fitted probability")
plt.tight_layout(); plt.show()
"""
)

md(
    """### What to Watch For
1. In the drug-only model the new drug looks uniformly better ($\\widehat{\\operatorname{OR}} \\approx 2$). In the full model the **main effect of `drug` collapses to $\\approx -0.06$** while `drug:time` explodes to $\\approx +1.02$: the drugs are similar at baseline and diverge with cumulative dosage. A main effect without its interaction is a *weighted average over time* — a meaningless summary when the interaction is real.
2. The raw-residual panels are two bands by construction ($y - p$ given $y \\in \\{0,1\\}$); they carry no diagnostic message. The *binned* mean residuals are the useful version: for the drug-only model they curve systematically — structure is missing. (Binned residual plots are the binary-outcome analogue of residual-vs-fitted checks, following Gelman & Hill.)
3. Every number here is computed under the fiction of independent rows, so all standard errors are too small — the price of ignoring `case`.
""",
    """### На что обратить внимание
1. В модели только с препаратом новый препарат выглядит равномерно лучше ($\\widehat{\\operatorname{OR}} \\approx 2$). В полной модели **главный эффект `drug` схлопывается до $\\approx -0{,}06$**, а `drug:time` взлетает до $\\approx +1{,}02$: препараты похожи на старте и расходятся с накопленной дозой. Главный эффект без своего взаимодействия — это *средневзвешенное по времени*, бессмысленная сводка, когда взаимодействие реально.
2. Панели сырых остатков — две полосы по построению ($y - p$ при $y \\in \\{0,1\\}$); они не несут диагностического сообщения. Полезная версия — *бинованные* средние остатки: для модели только с препаратом они систематически изгибаются — структуры не хватает. (Бинованные графики остатков — бинарный аналог проверок residual-vs-fitted по Гельману и Хиллу.)
3. Каждое число здесь вычислено при фикции независимых строк, поэтому все стандартные ошибки занижены — плата за игнорирование `case`.
""",
)

# ---------------------------------------------------------------------------
# Section 3: classical GLMM via VB
# ---------------------------------------------------------------------------
md(
    """## 3. The Classical GLMM Attempt: `BinomialBayesMixedGLM`

### The Idea
Add a random intercept per patient: `outcome ~ drug + time + severity + drug:time + (1 | case)`. Exact likelihood for a logistic GLMM requires numerical integration over 340 patient effects, which classical software sidesteps with approximations. `statsmodels` offers a *variational Bayes* approximation (`fit_vb`): it returns approximate posterior means and standard deviations for the fixed effects, but **no posterior samples, no convergence diagnostics, no full posterior for the variance component** — a Gaussian approximation around a mode, not the real thing.

### Example
""",
    """## 3. Классическая попытка GLMM: `BinomialBayesMixedGLM`

### Идея
Добавим случайный интерсепт на пациента: `outcome ~ drug + time + severity + drug:time + (1 | case)`. Точное правдоподобие для логистического GLMM требует численного интегрирования по 340 эффектам пациентов, что классический софт обходит приближениями. `statsmodels` предлагает *вариационное байесовское* приближение (`fit_vb`): оно возвращает приближённые апостериорные средние и стандартные отклонения фиксированных эффектов, но **без сэмплов апостериорного распределения, без диагностик сходимости, без полной апостериорной оценки компоненты дисперсии** — гауссово приближение вокруг моды, а не настоящая вещь.

### Пример
""",
)

code(
    """random_effects_spec = {"a": "0 + C(case)"}
mixed_vb = BinomialBayesMixedGLM.from_formula(
    "outcome ~ drug + time + severity + drug:time", random_effects_spec, data_depr)
result_vb = mixed_vb.fit_vb()
print(result_vb.summary())

fe = pd.DataFrame({"VB mean": result_vb.fe_mean,
                   "VB sd": result_vb.fe_sd,
                   "flat logit": basic2.params.values},
                  index=["Intercept", "drug", "time", "severity", "drug:time"])
print("\\nFixed effects: VB mixed GLMM vs flat logit MLE")
print(fe.round(3))

re_vb = result_vb.random_effects()
print("\\nFirst patient effects (VB):")
print(re_vb.head().round(3))
print("Estimated SD of case effects (VB):", round(float(result_vb.vc_mean[0]), 3))
"""
)

md(
    """### What to Watch For
The VB point estimates land close to the flat MLE — the fixed-effect story is stable. What the table cannot give you:
- a posterior distribution for the case-level standard deviation (only one number `vc_mean`);
- a predictive distribution for a *new* patient;
- any way to check whether the variational approximation is good (the ELBO improved, but by how much is it off?).

The legacy notebook said it bluntly: with point estimates alone "it is difficult to assess quality and make quantitative claims — by how much, which effect size, what uncertainty?" The Bayesian hierarchical model below answers exactly those questions.
""",
    """### На что обратить внимание
VB-оценки попадают близко к плоской MLE — история фиксированных эффектов устойчива. Чего таблица дать не может:
- апостериорного распределения для стандартного отклонения на уровне пациентов (только одно число `vc_mean`);
- предсказательного распределения для *нового* пациента;
- способа проверить качество вариационного приближения (ELBO вырос, но насколько он ошибается?).

Прежняя тетрадь говорила прямо: с одними точечными оценками «сложно оценить качество и делать количественные выводы — насколько, какой размер эффекта, какая неопределённость?» Байесовская иерархическая модель ниже отвечает ровно на эти вопросы.
""",
)

# ---------------------------------------------------------------------------
# Section 4: Bayesian hierarchical GLMM + LOO
# ---------------------------------------------------------------------------
md(
    """## 4. The Bayesian Hierarchical GLMM in Bambi + PSIS-LOO

### The Idea
The same formula, now with a full posterior: every patient gets an intercept $b_{\\text{case}}$ drawn from $\\operatorname{Normal}(0, \\sigma_{\\text{case}})$, and $\\sigma_{\\text{case}}$ itself is a parameter with a posterior distribution. We fit three nested models and let PSIS-LOO (the course criterion; the legacy notebook printed WAIC, which ArviZ 1.x no longer ships) rank them by out-of-sample predictive accuracy:
1. `drug` only — the naive pooling baseline;
2. `drug + time + severity + drug:time` — the full flat model;
3. the same plus `(1 | case)` — the hierarchical GLMM.

### Example
""",
    """## 4. Байесовский иерархический GLMM в Bambi и PSIS-LOO

### Идея
Та же формула, теперь с полным апостериорным распределением: каждый пациент получает интерсепт $b_{\\text{case}}$, извлечённый из $\\operatorname{Normal}(0, \\sigma_{\\text{case}})$, а $\\sigma_{\\text{case}}$ — сам параметр с апостериорным распределением. Подгоним три вложенные модели и позволим PSIS-LOO (критерий курса; прежняя тетрадь печатала WAIC, которого в ArviZ 1.x больше нет) ранжировать их по вневыборочной предсказательной точности:
1. только `drug` — наивный базовый уровень с полным объединением;
2. `drug + time + severity + drug:time` — полная плоская модель;
3. то же плюс `(1 | case)` — иерархический GLMM.

### Пример
""",
)

code(
    """FIXED = "drug + time + severity + drug:time"

model1 = bmb.Model("outcome ~ drug", data_depr, family="bernoulli")
idata1 = model1.fit(draws=1000, tune=1000, chains=4, random_seed=RANDOM_SEED,
                    idata_kwargs={"log_likelihood": True})
print(az.summary(idata1, round_to=2))
"""
)

code(
    """model2 = bmb.Model(f"outcome ~ {FIXED}", data_depr, family="bernoulli")
idata2 = model2.fit(draws=1000, tune=1000, chains=4, random_seed=RANDOM_SEED,
                    idata_kwargs={"log_likelihood": True})
print(az.summary(idata2, round_to=2))
print("\\nPSIS-LOO (ELPD):", az.loo(idata2).elpd.round(1))
"""
)

code(
    """model3 = bmb.Model(f"outcome ~ {FIXED} + (1 | case)", data_depr, family="bernoulli")
print(model3)
idata3 = model3.fit(draws=1000, tune=1000, chains=4, random_seed=RANDOM_SEED,
                    target_accept=0.95, idata_kwargs={"log_likelihood": True})
print(az.summary(idata3, var_names=["Intercept", "drug", "time", "severity",
                                    "drug:time", "1|case_sigma"], round_to=2))
print("\\nPSIS-LOO (ELPD):", az.loo(idata3).elpd.round(1))
"""
)

code(
    """cmp_df = az.compare({"M1: drug only (flat)": idata1,
                        "M2: full fixed (flat)": idata2,
                        "M3: + (1|case) GLMM": idata3})
print(cmp_df.round(1))
az.plot_compare(cmp_df)
plt.show()
"""
)

md(
    """### What to Watch For
1. **Trust checks first**: `r_hat` $\\le 1.01$ and healthy ESS in the summaries above. Even at `target_accept=0.95` the GLMM reports a handful of divergences: with only three binary observations per patient, $\\sigma_{\\text{case}}$ is weakly identified, its posterior piles up near zero, and the sampler feels the funnel geometry of Lesson 6 — the same boundary territory as the "singular fit" of `lme4` on the book page.
2. **The LOO verdict is a tie**: the full flat model and the GLMM differ by about one ELPD (dse $< 1$). With three observations per patient the data cannot tell "no patient heterogeneity" from "small patient heterogeneity": partial pooling degenerates toward complete pooling, and out-of-sample prediction does not care. Contrast this with `sleepstudy` in the Lesson 9 workbook, where ten continuous observations per subject let the hierarchical structure win decisively. **Hierarchy earns its keep when clusters are informative — not merely because rows repeat.**
3. The posterior of `1|case_sigma` (mean $\\approx 0.3$, mass concentrated near zero) is precisely the answer the VB table could only hint at: the full uncertainty statement about between-patient heterogeneity that the classical fit could not make — and here it says the heterogeneity is small.
4. Pareto-$k$ warnings above $0.7$ are common for hierarchical models (each row leans on its patient's effect); treat them as a prompt to consider leave-one-*patient*-out CV rather than a verdict of failure.

The next cell reports the model where it matters — on the **observable probability scale**, with named baselines (the course protocol: report contrasts at stated baseline values, never bare log-odds):
""",
    """### На что обратить внимание
1. **Сначала проверки доверия**: `r_hat` $\\le 1{,}01$ и здоровый ESS в сводках выше. Даже при `target_accept=0.95` GLMM сообщает о нескольких дивергенциях: при всего трёх бинарных наблюдениях на пациента $\\sigma_{\\text{case}}$ слабо идентифицируем, его апостериорное распределение скапливается у нуля, и сэмплер чувствует геометрию воронки из занятия 6 — та же приграничная территория, что и «сингулярная подгонка» `lme4` со страницы книги.
2. **Вердикт LOO — ничья**: полная плоская модель и GLMM различаются примерно на один ELPD (dse $< 1$). С тремя наблюдениями на пациента данные не могут отличить «нет гетерогенности пациентов» от «небольшой гетерогенности»: частичное объединение вырождается в полное, и вневыборочному предсказанию всё равно. Сравните со `sleepstudy` из тетради занятия 9, где десять непрерывных наблюдений на испытуемого позволили иерархии выиграть уверенно. **Иерархия окупается, когда кластеры информативны, — а не просто потому, что строки повторяются.**
3. Апостериорное распределение `1|case_sigma` (среднее $\\approx 0{,}3$, масса у нуля) — ровно тот ответ, на который VB-таблица могла лишь намекнуть: полное утверждение о неопределённости межпациентской гетерогенности, недоступное классическому фиту, — и здесь оно говорит, что гетерогенность мала.
4. Предупреждения Парето-$k$ выше $0{,}7$ обычны для иерархических моделей (каждая строка опирается на эффект своего пациента); считайте их поводом рассмотреть кросс-валидацию с исключением целого *пациента*, а не приговором.

Следующая ячейка отчитывается там, где это важно, — на **наблюдаемой шкале вероятностей**, с названными базовыми уровнями (протокол курса: контрасты при указанных базовых значениях, никогда не голые лог-одды):
""",
)

code(
    """def cell_probability(idata, severity, drug, t, var_names):
    # Posterior of P(normal) for a TYPICAL patient (case effect = 0)
    post = idata.posterior.to_dataset()
    eta = (post["Intercept"] + drug * post["drug"] + t * post["time"]
           + severity * post["severity"] + drug * t * post["drug:time"])
    return 1 / (1 + np.exp(-eta)).values.flatten()

rows = []
for sev, sev_name in [(0, "mild"), (1, "severe")]:
    for dr, dr_name in [(0, "standard"), (1, "new")]:
        for t, week in [(0, 1), (1, 2), (2, 4)]:
            p = cell_probability(idata3, sev, dr, t, None)
            lo, hi = az.hdi(p, prob=0.94)
            rows.append({"severity": sev_name, "drug": dr_name, "week": week,
                         "P(normal) mean": p.mean().round(3),
                         "94% HDI": f"[{lo:.2f}, {hi:.2f}]"})
report = pd.DataFrame(rows)
print("M3 (hierarchical GLMM), typical patient, posterior P(normal):")
print(report.to_string(index=False))

# Probability-scale drug effect at week 4 for each severity
for sev, sev_name in [(0, "mild"), (1, "severe")]:
    p_new = cell_probability(idata3, sev, 1, 2, None)
    p_std = cell_probability(idata3, sev, 0, 2, None)
    delta = p_new - p_std
    lo, hi = az.hdi(delta, prob=0.94)
    print(f"week 4, {sev_name}: P(new) - P(standard) = {delta.mean():+.3f}, 94% HDI [{lo:+.2f}, {hi:+.2f}]")
"""
)

# ---------------------------------------------------------------------------
# Section 5: ANES
# ---------------------------------------------------------------------------
md(
    """## 5. Cross-Sectional Interactions: ANES 2016 Vote

### The Idea
Not all binary data is longitudinal, but the same machinery handles *heterogeneous groups* in cross-sections too. The ANES 2016 example (from the Bambi documentation) models the Clinton/Trump vote with **party-specific age slopes**. One coding subtlety: the formula `vote['clinton'] ~ party_id + party_id:age` expands the interaction into an age slope **for every party** (three slopes: democrat, independent, republican), so the model is full-rank — the age main effect is absorbed into the per-party slopes rather than omitted. `vote['clinton']` tells Bambi which level of the response is the modeled "success".

### Example
""",
    """## 5. Поперечные взаимодействия: голосование ANES 2016

### Идея
Не все бинарные данные продольные, но тот же аппарат обрабатывает и *неоднородные группы* в поперечных срезах. Пример ANES 2016 (из документации Bambi) моделирует голосование Клинтон/Трамп с **партийно-специфичными наклонами возраста**. Одна тонкость кодирования: формула `vote['clinton'] ~ party_id + party_id:age` разворачивает взаимодействие в наклон возраста **для каждой партии** (три наклона: демократы, независимые, республиканцы), так что модель полного ранга — главный эффект возраста поглощён партийными наклонами, а не выброшен. `vote['clinton']` сообщает Bambi, какой уровень отклика является моделируемым «успехом».

### Пример
""",
)

code(
    """clinton_data = pd.read_csv(ANES_PATH)
print(clinton_data.head())
print("\\nRows:", len(clinton_data), "| votes:", clinton_data["vote"].value_counts().to_dict(),
      "| parties:", sorted(clinton_data["party_id"].unique()))

clinton_model = bmb.Model("vote['clinton'] ~ party_id + party_id:age", clinton_data,
                          family="bernoulli")
print(clinton_model)
idata_anes = clinton_model.fit(draws=1000, tune=1000, chains=4, random_seed=RANDOM_SEED,
                               target_accept=0.85, idata_kwargs={"log_likelihood": True})
print(az.summary(idata_anes, round_to=2))
"""
)

code(
    """# Report the age effects on the observable scale: probability change per decade of age,
# evaluated at each party's own baseline (mean age within party).
post = idata_anes.posterior.to_dataset()
party_levels = ["democrat", "independent", "republican"]
base_age = {p: clinton_data.loc[clinton_data["party_id"] == p, "age"].mean()
            for p in party_levels}
slopes = {p: post["party_id:age"].sel(**{"party_id:age_dim": p}).values.flatten()
          for p in party_levels}

fig, ax = plt.subplots(figsize=(7, 3.8))
for i, p in enumerate(party_levels):
    ax.hist(slopes[p], bins=60, alpha=0.55, density=True, label=p, color=f"C{i}")
ax.set_xlabel("posterior log-odds slope per year of age")
ax.set_title("Party-specific age slopes (ANES 2016, Clinton vote)")
ax.legend()
plt.tight_layout(); plt.show()

print("Per-decade effect on P(vote Clinton), at each party's mean age (divide-by-4 rule as a sanity check):")
for p in party_levels:
    s10 = slopes[p] * 10
    eta = post["Intercept"].values.flatten()
    if p != "democrat":   # 'democrat' is the reference level, absorbed into the Intercept
        eta = eta + post["party_id"].sel(**{"party_id_dim": p}).values.flatten()
    p_hat = 1 / (1 + np.exp(-eta))
    delta_prob = s10 * p_hat * (1 - p_hat)   # logit-approximation of the probability change
    lo, hi = az.hdi(delta_prob, prob=0.94)
    print(f"  {p:12s} baseline age {base_age[p]:.0f}: "
          f"DeltaP/decade = {delta_prob.mean():+.3f}, 94% HDI [{lo:+.3f}, {hi:+.3f}]")
"""
)

md(
    """### What to Watch For
1. The three posterior slope distributions barely overlap: age *negatively* predicts a Clinton vote among republicans and independents, while among democrats the slope is much closer to zero. An "average age effect" would have hidden the entire story — interactions are how heterogeneity becomes visible.
2. We translated slopes to the probability scale **at each party's own baseline probability** (the course's golden rule from Lesson 10): the same logit slope means different probability changes for democrats ($p \\approx 0.9$) and republicans ($p \\approx 0.1$).
3. The source notebook used `draws=200, target_accept=0.85`; we raised the draws to 1000 and kept `target_accept=0.85` — always check the r_hat column before quoting the numbers.
""",
    """### На что обратить внимание
1. Три апостериорных распределения наклонов почти не перекрываются: возраст *негативно* предсказывает голос за Клинтон у республиканцев и независимых, а у демократов наклон близок к нулю. «Средний эффект возраста» скрыл бы всю историю — взаимодействия делают гетерогенность видимой.
2. Мы перевели наклоны на шкалу вероятностей **при собственной базовой вероятности каждой партии** (золотое правило курса из занятия 10): один и тот же наклон логита означает разные изменения вероятности для демократов ($p \\approx 0{,}9$) и республиканцев ($p \\approx 0{,}1$).
3. Прежняя тетрадь использовала `draws=200, target_accept=0.85`; мы подняли число сэмплов до 1000 и сохранили `target_accept=0.85` — всегда проверяйте колонку r_hat, прежде чем цитировать числа.
""",
)

# ---------------------------------------------------------------------------
# Section 6: optional homework
# ---------------------------------------------------------------------------
md(
    """## 6. Optional Homework: the Prosocial Chimpanzees, Revisited as a GLMM

The legacy notebook pointed to "starfish and chimpanzees GLM examples" from other course materials. The starfish pointer had no attached dataset; the chimpanzees are already in this book as the Lesson 10 capstone — so the homework makes the connection explicit.

**Tasks:**
1. Load the chimpanzees data from the book repository (`data/chimpanzees.csv`, see the Lesson 10 workbook) and recall the design: `pulled_left`, `prosoc_left`, `condition`, `actor`, `block`.
2. Fit the flat logistic model `pulled_left ~ prosoc_left * condition` ignoring actors and blocks.
3. Fit the GLMM `pulled_left ~ prosoc_left * condition + (1 | actor) + (1 | block)` in Bambi.
4. Compare the two with PSIS-LOO. Report the population-level effect of the prosocial option on P(pull left) with its uncertainty, and one actor-level posterior contrast of your choice.
5. One sentence on what `(1 | block)` buys you — and check whether the LOO table supports including it.
""",
    """## 6. Необязательное домашнее задание: просоциальные шимпанзе как GLMM

Прежняя тетрадь ссылалась на «примеры GLM про морские звёзды и шимпанзе» из других материалов курса. Указатель на морских звёзд не имел присоединённых данных; шимпанзе уже есть в этой книге как капстоун занятия 10 — поэтому домашнее задание делает связь явной.

**Задания:**
1. Загрузите данные шимпанзе из репозитория книги (`data/chimpanzees.csv`, см. тетрадь занятия 10) и вспомните дизайн: `pulled_left`, `prosoc_left`, `condition`, `actor`, `block`.
2. Подгоните плоскую логистическую модель `pulled_left ~ prosoc_left * condition`, игнорируя акторов и блоки.
3. Подгоните GLMM `pulled_left ~ prosoc_left * condition + (1 | actor) + (1 | block)` в Bambi.
4. Сравните обе модели по PSIS-LOO. Отчитайтесь о популяционном эффекте просоциальной опции на P(потянуть левый рычаг) с неопределённостью и об одном контрасте на уровне акторов по вашему выбору.
5. Одно предложение о том, что даёт `(1 | block)` — и проверьте, поддерживает ли таблица LOO его включение.
""",
)

code(
    """# Homework data starter (no solutions below — this one is yours to build)
CHIMP_PATH = fetch("chimpanzees.csv", BASE + "chimpanzees.csv")
chimp = pd.read_csv(CHIMP_PATH, sep=";")
print(chimp.head())
print("actors:", chimp["actor"].nunique(), "| blocks:", chimp["block"].nunique(),
      "| trials:", len(chimp))
"""
)

md(
    """## Summary

- Longitudinal binary data has the same clustered structure as longitudinal continuous data; the fix is the same — random effects — wrapped around a different link function.
- The flat logistic ladder already carries the substantive story (main effect of the drug collapses once the interaction with time enters), but its standard errors are fiction.
- `statsmodels`' variational GLMM gives point estimates and approximate SDs; the Bayesian hierarchical GLMM gives the full posterior — including the between-patient SD — and PSIS-LOO arbitration between the flat and hierarchical structures. In this trial the arbitration called it a tie (three binary rows per patient carry too little cluster information); in `sleepstudy` hierarchy won decisively.
- Reporting belongs on the probability scale at named baselines: "at week 4, initially mild patients: +X probability points under the new drug, 94% HDI [a, b]".
- Reference literature: Agresti, *Categorical Data Analysis*, the depression clinical trial example; Gelman & Hill, *Data Analysis Using Regression and Multilevel/Hierarchical Models*, binned residuals and varying-intercept logistic models; the Bambi documentation for the ANES case.

**Back to the book:** the Lesson 10 capstone (Prosocial Chimpanzees) builds a hierarchical logit by hand; the *Mixed Models vs. Hierarchical Bayes* page condenses this ladder; Lesson 11 adds the full diagnostic arsenal.
""",
    """## Итоги

- Продольные бинарные данные имеют ту же кластерную структуру, что и продольные непрерывные; лечение то же — случайные эффекты, — только вокруг другой функции связи.
- Плоская логистическая лестница уже несёт содержательную историю (главный эффект препарата схлопывается, как только входит взаимодействие со временем), но её стандартные ошибки — фикция.
- Вариационный GLMM из `statsmodels` даёт точечные оценки и приближённые SD; байесовский иерархический GLMM даёт полное апостериорное распределение — включая SD между пациентами — и арбитраж PSIS-LOO между плоской и иерархической структурами. В этом испытании арбитраж назвал ничью (три бинарные строки на пациента несут слишком мало кластерной информации); на `sleepstudy` иерархия выиграла уверенно.
- Отчётность принадлежит шкале вероятностей при названных базовых уровнях: «к 4-й неделе у исходно лёгких пациентов: +X процентных пунктов на новом препарате, 94% HDI [a, b]».
- Литература: Agresti, *Categorical Data Analysis*, пример с клиническим испытанием лечения депрессии; Gelman & Hill, *Data Analysis Using Regression and Multilevel/Hierarchical Models*, бинованные остатки и логистические модели со случайным интерсептом; документация Bambi для случая ANES.

**Назад к книге:** капстоун занятия 10 (просоциальные шимпанзе) строит иерархический логит вручную; страница «Классические смешанные модели и байесовская иерархия» сжимает эту лестницу; занятие 11 добавляет полный диагностический арсенал.
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
