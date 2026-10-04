"""Build the Lesson 6 student workbooks (English and Russian) from one source.

Run from the book root with the pymc_env environment:

    conda run -n pymc_env python scripts/build_lesson6_notebooks.py

Outputs:
    notebooks/colab/06_running_a_sampler.ipynb
    notebooks/colab/06_running_a_sampler_ru.ipynb

Every cell is defined once with an English and a Russian variant. The code works
with PyMC 5+ / ArviZ 0.x (Colab) and PyMC 6 / ArviZ 1.x (course environment):
draws are read with ``idata.posterior[var].values`` and diagnostics with
``az.rhat`` / ``az.ess`` / ``az.summary``.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks" / "colab"
GITHUB = "https://colab.research.google.com/github/iknyazeva/bayes-cogsci-book/blob/main/notebooks/colab/"
FILES = {"en": "06_running_a_sampler.ipynb", "ru": "06_running_a_sampler_ru.ipynb"}
BOOK = "https://iknyazeva.github.io/bayes-cogsci-book/"

CELLS: list[tuple[str, dict[str, str]]] = []


def md(en: str, ru: str | None = None) -> None:
    CELLS.append(("markdown", {"en": en, "ru": ru if ru is not None else en}))


def code(en: str, ru: str | None = None) -> None:
    CELLS.append(("code", {"en": en, "ru": ru if ru is not None else en}))


# ---------------------------------------------------------------------------
md(
    f"""# Lesson 6: Running a Sampler — Convergence, Trust Checks and Approximations
### Student workbook
*Course: Bayesian Analysis of Empirical Data (2026)*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({GITHUB}{FILES['en']})

Russian version: [{FILES['ru']}]({GITHUB}{FILES['ru']})
""",
    f"""# Занятие 6: Запускаем сэмплер — сходимость, проверки доверия и приближения
### Рабочая тетрадь студента
*Курс: Байесовский анализ эмпирических данных (2026)*

[![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)]({GITHUB}{FILES['ru']})

English version: [{FILES['en']}]({GITHUB}{FILES['en']})
""",
)

md(
    """## 1. Before class
* **One-minute goal:** run a sampler, check whether its draws can be trusted (R-hat, ESS, divergences), and recognise two fast shortcuts (Laplace and variational inference).
* **Example:** the housing-subsidy survey: 58 of 100 lower-income and 40 of 100 higher-income households support the subsidy; Beta(2, 2) priors. Exact answers from Lesson 4: P(gap > 5 points) = 0.964.
* **How to work:** predict first, change one number, rerun. PyMC cells take a few seconds each.
""",
    """## 1. Перед занятием
* **Цель в одну минуту:** запустить сэмплер, проверить, можно ли доверять его выборке (R-hat, ESS, расхождения), и узнать два быстрых приближения (Лапласа и вариационный вывод).
* **Пример:** опрос о жилищной субсидии: 58 из 100 домохозяйств с низким доходом и 40 из 100 с высоким доходом поддерживают субсидию; априорные Beta(2, 2). Точный ответ из занятия 4: P(разрыв > 5 п.п.) = 0,964.
* **Как работать:** прогноз → изменить одно число → запустить снова. Каждая ячейка с PyMC выполняется несколько секунд.
""",
)

code(
    """import sys, subprocess
IS_COLAB = "google.colab" in sys.modules
try:
    import pymc as pm
except ImportError:                                   # install PyMC if it is missing
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pymc"])
    import pymc as pm
import arviz as az
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import logging, warnings
logging.getLogger("pymc").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
plt.rcParams.update({"figure.figsize": (9, 3.8), "axes.grid": True, "grid.alpha": 0.3})
EXACT_LOW = stats.beta(60, 44)                        # Lesson 4 posterior, lower-income
xs = np.linspace(0.3, 0.85, 400)
print("PyMC", pm.__version__, "| ArviZ", az.__version__)
""",
    """import sys, subprocess
IS_COLAB = "google.colab" in sys.modules
try:
    import pymc as pm
except ImportError:                                   # установить PyMC, если его нет
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pymc"])
    import pymc as pm
import arviz as az
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import logging, warnings
logging.getLogger("pymc").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
plt.rcParams.update({"figure.figsize": (9, 3.8), "axes.grid": True, "grid.alpha": 0.3})
EXACT_LOW = stats.beta(60, 44)                        # апостериорное из занятия 4, низкий доход
xs = np.linspace(0.3, 0.85, 400)
print("PyMC", pm.__version__, "| ArviZ", az.__version__)
""",
)

md(
    """## 2. Predict before you run (on paper)
1. A walker accepts 98% of its proposals. Is that good?
2. Four chains start at 20%, 40%, 70% and 90% and take tiny steps for 400 steps. Will they agree?
3. A bell (Normal) is fitted to a posterior based on only 10 households, 1 of whom supports the subsidy. What can go wrong?
""",
    """## 2. Прогноз до запуска (на бумаге)
1. «Путник» принимает 98% предложений. Это хорошо?
2. Четыре цепи стартуют с 20%, 40%, 70% и 90% и делают крошечные шаги 400 раз. Согласятся ли они?
3. К апостериорному, построенному всего по 10 домохозяйствам (1 «за»), подгоняют колокол (нормальное распределение). Что может пойти не так?
""",
)

# ---------------------------------------------------------------------------
md(
    """## 3. Classwork 1 — Walker settings: step size and warm-up
This is the NumPy walker from Lesson 5. **Change** `STEP` (0.003, 0.07, 0.6) and `START` (0.05, 0.45). Look at the acceptance rate, the trace and the **ESS** (how many independent draws the chain is worth).
""",
    """## 3. Классная работа 1 — Настройки «путника»: размер шага и разогрев
Это «путник» на NumPy из занятия 5. **Измените** `STEP` (0.003, 0.07, 0.6) и `START` (0.05, 0.45). Посмотрите на долю принятых шагов, трассировку и **ESS** — сколько независимых значений «стоит» цепь.
""",
)

code(
    """def log_shape(theta):
    if not 0 < theta < 1:
        return -np.inf
    return stats.beta.logpdf(theta, 2, 2) + stats.binom.logpmf(58, 100, theta)

def walker(start, step, n_steps, rng):
    chain = np.empty(n_steps); cur, cur_lh = start, log_shape(start); acc = 0
    for i in range(n_steps):
        prop = cur + rng.normal(0, step); prop_lh = log_shape(prop)
        if np.log(rng.random()) < prop_lh - cur_lh:
            cur, cur_lh, acc = prop, prop_lh, acc + 1
        chain[i] = cur
    return chain, acc / n_steps

STEP, START = 0.07, 0.45                             # <- change me
chain, acc = walker(START, STEP, 2000, rng)
ess = float(np.asarray(az.ess(chain[None, :])))     # shape (1 chain, n draws)
print(f"accepted {acc:.0%} | 2,000 steps are worth about {ess:.0f} independent draws")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(chain, lw=0.7); ax[0].set_xlabel("step"); ax[0].set_ylabel("support")
ax[1].hist(chain[200:], bins=40, density=True, alpha=0.5); ax[1].plot(xs, EXACT_LOW.pdf(xs), "k--")
ax[1].set_xlim(0.3, 0.85); plt.tight_layout(); plt.show()
""",
    """def log_shape(theta):
    if not 0 < theta < 1:
        return -np.inf
    return stats.beta.logpdf(theta, 2, 2) + stats.binom.logpmf(58, 100, theta)

def walker(start, step, n_steps, rng):
    chain = np.empty(n_steps); cur, cur_lh = start, log_shape(start); acc = 0
    for i in range(n_steps):
        prop = cur + rng.normal(0, step); prop_lh = log_shape(prop)
        if np.log(rng.random()) < prop_lh - cur_lh:
            cur, cur_lh, acc = prop, prop_lh, acc + 1
        chain[i] = cur
    return chain, acc / n_steps

STEP, START = 0.07, 0.45                             # <- измените
chain, acc = walker(START, STEP, 2000, rng)
ess = float(np.asarray(az.ess(chain[None, :])))     # форма (1 цепь, n значений)
print(f"принято {acc:.0%} | 2000 шагов стоят примерно {ess:.0f} независимых значений")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(chain, lw=0.7); ax[0].set_xlabel("шаг"); ax[0].set_ylabel("поддержка")
ax[1].hist(chain[200:], bins=40, density=True, alpha=0.5); ax[1].plot(xs, EXACT_LOW.pdf(xs), "k--")
ax[1].set_xlim(0.3, 0.85); plt.tight_layout(); plt.show()
""",
)

md(
    """**Question 1.** Which step size gives the highest acceptance? Which gives the highest ESS? With `START = 0.05`, how many steps does the walker need to reach the hill, i.e. how long should the warm-up be?
""",
    """**Вопрос 1.** При каком шаге доля принятых предложений максимальна? А ESS? При `START = 0.05` за сколько шагов «путник» добирается до холма — какой длины должен быть разогрев?
""",
)

# ---------------------------------------------------------------------------
md(
    """## 4. Classwork 2 — Four chains and R-hat
Run four walkers from different starting points. **R-hat** compares the chains with each other; values ≤ 1.01 mean they agree. **Change** `STEP` to 0.002 and `N_STEPS` to 400 to make them fail.
""",
    """## 4. Классная работа 2 — Четыре цепи и R-hat
Запустим четырёх «путников» из разных точек. **R-hat** сравнивает цепи между собой; значения ≤ 1,01 означают согласие. **Измените** `STEP` на 0.002 и `N_STEPS` на 400, чтобы они не сошлись.
""",
)

code(
    """STEP, N_STEPS, WARMUP = 0.07, 1200, 200            # <- change me
starts = [0.2, 0.4, 0.7, 0.9]
chains = np.array([walker(s, STEP, N_STEPS, rng)[0] for s in starts])
kept = chains[:, WARMUP:] if N_STEPS > WARMUP else chains
r_hat = float(np.asarray(az.rhat(kept)))
ess = float(np.asarray(az.ess(kept)))
print(f"R-hat = {r_hat:.3f}  |  ESS = {ess:.0f}  ->", "chains agree" if r_hat <= 1.01 else "chains DISAGREE")
for c in kept:
    plt.plot(c, lw=0.7)
plt.xlabel("step"); plt.ylabel("support"); plt.ylim(0, 1); plt.show()
""",
    """STEP, N_STEPS, WARMUP = 0.07, 1200, 200            # <- измените
starts = [0.2, 0.4, 0.7, 0.9]
chains = np.array([walker(s, STEP, N_STEPS, rng)[0] for s in starts])
kept = chains[:, WARMUP:] if N_STEPS > WARMUP else chains
r_hat = float(np.asarray(az.rhat(kept)))
ess = float(np.asarray(az.ess(kept)))
print(f"R-hat = {r_hat:.3f}  |  ESS = {ess:.0f}  ->", "цепи согласны" if r_hat <= 1.01 else "цепи НЕ согласны")
for c in kept:
    plt.plot(c, lw=0.7)
plt.xlabel("шаг"); plt.ylabel("поддержка"); plt.ylim(0, 1); plt.show()
""",
)

# ---------------------------------------------------------------------------
md(
    """## 5. Classwork 3 — PyMC does the walking
We write the model (prior + likelihood); `pm.sample()` runs NUTS with 4 chains. `tune` is the warm-up and `draws` are the kept steps.
""",
    """## 5. Классная работа 3 — PyMC ходит за нас
Мы записываем модель (априорное + правдоподобие); `pm.sample()` запускает NUTS с 4 цепями. `tune` — это разогрев, `draws` — сохраняемые шаги.
""",
)

code(
    """with pm.Model() as survey_model:
    support = pm.Beta("support", alpha=2, beta=2, shape=2)            # [lower-income, higher-income]
    pm.Binomial("supporters", n=[100, 100], p=support, observed=[58, 40])
    idata = pm.sample(draws=1000, tune=1000, chains=4, random_seed=RANDOM_SEED, progressbar=False)

az.summary(idata)
""",
    """with pm.Model() as survey_model:
    support = pm.Beta("support", alpha=2, beta=2, shape=2)            # [низкий доход, высокий доход]
    pm.Binomial("supporters", n=[100, 100], p=support, observed=[58, 40])
    idata = pm.sample(draws=1000, tune=1000, chains=4, random_seed=RANDOM_SEED, progressbar=False)

az.summary(idata)
""",
)

md(
    """**The trust-check card.** Run it before interpreting any number: all three lights must be green.
""",
    """**Карточка проверки доверия.** Запускайте её до того, как интерпретировать любые числа: все три огня должны быть зелёными.
""",
)

code(
    """def trust_check(idata, var):
    r = float(np.max(np.asarray(az.rhat(idata, var_names=[var])[var])))
    e = float(np.min(np.asarray(az.ess(idata, var_names=[var])[var])))
    d = int(np.asarray(idata.sample_stats["diverging"]).sum())
    lights = ["🟢" if r <= 1.01 else "🔴", "🟢" if e >= 400 else "🔴", "🟢" if d == 0 else "🔴"]
    ok = all(l == "🟢" for l in lights)
    print(f"{lights[0]} R-hat {r:.3f} | {lights[1]} ESS {e:.0f} | {lights[2]} divergences {d}  ->",
          "may interpret" if ok else "DO NOT interpret")
    return ok

trust_check(idata, "support")
""",
    """def trust_check(idata, var):
    r = float(np.max(np.asarray(az.rhat(idata, var_names=[var])[var])))
    e = float(np.min(np.asarray(az.ess(idata, var_names=[var])[var])))
    d = int(np.asarray(idata.sample_stats["diverging"]).sum())
    lights = ["🟢" if r <= 1.01 else "🔴", "🟢" if e >= 400 else "🔴", "🟢" if d == 0 else "🔴"]
    ok = all(l == "🟢" for l in lights)
    print(f"{lights[0]} R-hat {r:.3f} | {lights[1]} ESS {e:.0f} | {lights[2]} расхождения {d}  ->",
          "можно интерпретировать" if ok else "НЕ интерпретировать")
    return ok

trust_check(idata, "support")
""",
)

md(
    """Now use the draws **exactly as in Lessons 4–5**: compute the quantity for every draw, then count or average.
""",
    """Теперь используйте выборку **точно как на занятиях 4–5**: вычислите величину для каждого значения, затем посчитайте долю или среднее.
""",
)

code(
    """draws = idata.posterior["support"].values.reshape(-1, 2)          # 4,000 rows: [lower, higher]
gap = draws[:, 0] - draws[:, 1]
print("P(gap > 5 points):", round(np.mean(gap > 0.05), 3), "  exact (Lesson 4): 0.964")
print("95% interval for the gap:", np.quantile(gap, [0.025, 0.975]).round(3))

plt.hist(draws[:, 0], bins=40, density=True, alpha=0.5, label="PyMC draws")
plt.plot(xs, EXACT_LOW.pdf(xs), "k--", label="exact Beta(60, 44)")
plt.xlabel("lower-income support"); plt.legend(); plt.show()
""",
    """draws = idata.posterior["support"].values.reshape(-1, 2)          # 4000 строк: [низкий, высокий]
gap = draws[:, 0] - draws[:, 1]
print("P(разрыв > 5 п.п.):", round(np.mean(gap > 0.05), 3), "  точно (занятие 4): 0.964")
print("95% интервал для разрыва:", np.quantile(gap, [0.025, 0.975]).round(3))

plt.hist(draws[:, 0], bins=40, density=True, alpha=0.5, label="выборка PyMC")
plt.plot(xs, EXACT_LOW.pdf(xs), "k--", label="точное Beta(60, 44)")
plt.xlabel("поддержка: низкий доход"); plt.legend(); plt.show()
""",
)

md(
    """**Question 2.** Look at the `mcse_mean` column of `az.summary`. With this Monte Carlo error, how many decimal places of the mean are worth reporting?
""",
    """**Вопрос 2.** Посмотрите на столбец `mcse_mean` в `az.summary`. Сколько знаков после запятой у среднего имеет смысл сообщать при такой ошибке Монте-Карло?
""",
)

# ---------------------------------------------------------------------------
md(
    """## 6. Classwork 4 — Gallery of failed fits
Run each model and give it a verdict with `trust_check`. **Do not interpret** any model with a red light.

**(a) Too few draws.** Only 25 kept draws per chain.
""",
    """## 6. Классная работа 4 — Галерея неудачных подгонок
Запустите каждую модель и вынесите вердикт с помощью `trust_check`. **Не интерпретируйте** модель, если горит хотя бы один красный огонь.

**(a) Слишком мало значений.** Всего 25 сохраняемых шагов на цепь.
""",
)

code(
    """with survey_model:
    idata_few = pm.sample(draws=25, tune=1000, chains=4, random_seed=3, progressbar=False)
trust_check(idata_few, "support")
""",
)

md(
    """**(b) Two hills.** The data (mean about 4) tell us only the *size* of an effect μ through μ², not its sign. So μ = +2 and μ = −2 are equally plausible. We start two chains at −2 and two at +2.
""",
    """**(b) Два холма.** Данные (среднее около 4) говорят только о *величине* эффекта μ через μ², но не о его знаке. Поэтому μ = +2 и μ = −2 одинаково правдоподобны. Запускаем две цепи из −2 и две из +2.
""",
)

code(
    """y_two_hills = np.array([4.6, 3.1, 4.9, 3.8, 4.4, 2.9, 4.1, 3.6, 5.0, 3.6])
with pm.Model():
    mu = pm.Normal("mu", 0, 3)
    pm.Normal("y", mu=mu**2, sigma=1, observed=y_two_hills)
    idata_two = pm.sample(1000, tune=1000, chains=4, random_seed=RANDOM_SEED, progressbar=False,
                          initvals=[{"mu": -2.0}, {"mu": 2.0}, {"mu": -2.0}, {"mu": 2.0}])
trust_check(idata_two, "mu")
for c in idata_two.posterior["mu"].values:
    plt.plot(c, lw=0.6)
plt.xlabel("step"); plt.ylabel("mu"); plt.show()
""",
    """y_two_hills = np.array([4.6, 3.1, 4.9, 3.8, 4.4, 2.9, 4.1, 3.6, 5.0, 3.6])
with pm.Model():
    mu = pm.Normal("mu", 0, 3)
    pm.Normal("y", mu=mu**2, sigma=1, observed=y_two_hills)
    idata_two = pm.sample(1000, tune=1000, chains=4, random_seed=RANDOM_SEED, progressbar=False,
                          initvals=[{"mu": -2.0}, {"mu": 2.0}, {"mu": -2.0}, {"mu": 2.0}])
trust_check(idata_two, "mu")
for c in idata_two.posterior["mu"].values:
    plt.plot(c, lw=0.6)
plt.xlabel("шаг"); plt.ylabel("mu"); plt.show()
""",
)

md(
    """**(c) Divergences: the eight-schools funnel.** A classic hierarchical model (Rubin, 1981). Here you only need the verdict; how to fix it comes in later lessons. Red dots are divergent draws.
""",
    """**(c) Расхождения: «воронка» восьми школ.** Классическая иерархическая модель (Rubin, 1981). Пока нужен только вердикт; как это исправлять — на следующих занятиях. Красные точки — расходящиеся шаги.
""",
)

code(
    """y_schools = np.array([28., 8., -3., 7., -1., 1., 18., 12.])
se_schools = np.array([15., 10., 16., 11., 9., 11., 10., 18.])
with pm.Model():
    mu = pm.Normal("mu", 0, 5)
    tau = pm.HalfCauchy("tau", 5)
    theta = pm.Normal("theta", mu, tau, shape=8)
    pm.Normal("y", theta, se_schools, observed=y_schools)
    idata_div = pm.sample(1000, tune=1000, chains=4, random_seed=RANDOM_SEED, progressbar=False)
trust_check(idata_div, "tau")

tau_d = idata_div.posterior["tau"].values.ravel()
th0 = idata_div.posterior["theta"].values[..., 0].ravel()
div = np.asarray(idata_div.sample_stats["diverging"]).ravel().astype(bool)
plt.scatter(th0[~div], np.log(tau_d[~div]), s=3, alpha=0.3)
plt.scatter(th0[div], np.log(tau_d[div]), s=12, color="red")
plt.xlabel("effect in school 1"); plt.ylabel("log(spread between schools)"); plt.show()
""",
    """y_schools = np.array([28., 8., -3., 7., -1., 1., 18., 12.])
se_schools = np.array([15., 10., 16., 11., 9., 11., 10., 18.])
with pm.Model():
    mu = pm.Normal("mu", 0, 5)
    tau = pm.HalfCauchy("tau", 5)
    theta = pm.Normal("theta", mu, tau, shape=8)
    pm.Normal("y", theta, se_schools, observed=y_schools)
    idata_div = pm.sample(1000, tune=1000, chains=4, random_seed=RANDOM_SEED, progressbar=False)
trust_check(idata_div, "tau")

tau_d = idata_div.posterior["tau"].values.ravel()
th0 = idata_div.posterior["theta"].values[..., 0].ravel()
div = np.asarray(idata_div.sample_stats["diverging"]).ravel().astype(bool)
plt.scatter(th0[~div], np.log(tau_d[~div]), s=3, alpha=0.3)
plt.scatter(th0[div], np.log(tau_d[div]), s=12, color="red")
plt.xlabel("эффект в школе 1"); plt.ylabel("log(разброс между школами)"); plt.show()
""",
)

md(
    """**Question 3.** For each failed fit, would running 10 times more draws fix it? Why or why not?
""",
    """**Вопрос 3.** Исправит ли каждую из неудачных подгонок запуск в 10 раз большего числа шагов? Почему?
""",
)

# ---------------------------------------------------------------------------
md(
    """## 7. Classwork 5 — Shortcuts: Laplace and variational inference
For the same question we compare:
* **Laplace:** `pm.find_MAP()` finds the top of the hill; a Normal bell is placed there with the hill's curvature;
* **VI:** `pm.fit()` (ADVI) optimises a bell, maximising the ELBO;
* **MCMC:** `pm.sample()`.

**Change** `SUPPORTERS, HOUSEHOLDS` from `58, 100` (full survey) to `1, 10` (tiny pilot).
""",
    """## 7. Классная работа 5 — Приближения: Лаплас и вариационный вывод
Для одного и того же вопроса сравним:
* **Лаплас:** `pm.find_MAP()` находит вершину холма; там ставится нормальный колокол с той же кривизной;
* **VI:** `pm.fit()` (ADVI) оптимизирует колокол, максимизируя ELBO;
* **MCMC:** `pm.sample()`.

**Измените** `SUPPORTERS, HOUSEHOLDS` с `58, 100` (полный опрос) на `1, 10` (крошечный пилот).
""",
)

code(
    """SUPPORTERS, HOUSEHOLDS = 58, 100                    # <- change to 1, 10
a, b = 2 + SUPPORTERS, 2 + HOUSEHOLDS - SUPPORTERS   # exact posterior Beta(a, b)

with pm.Model():
    s = pm.Beta("support", 2, 2)
    pm.Binomial("supporters", n=HOUSEHOLDS, p=s, observed=SUPPORTERS)
    map_value = float(pm.find_MAP(progressbar=False)["support"])
    vi_draws = np.asarray(pm.fit(n=30_000, method="advi", random_seed=RANDOM_SEED, progressbar=False)
                          .sample(4000, random_seed=RANDOM_SEED).posterior["support"]).ravel()
    mc_draws = pm.sample(1000, tune=1000, chains=4, random_seed=RANDOM_SEED, progressbar=False).posterior["support"].values.ravel()

# Laplace: curvature of the log posterior at the top (second derivative, by finite differences)
logp = lambda t: stats.beta.logpdf(t, 2, 2) + stats.binom.logpmf(SUPPORTERS, HOUSEHOLDS, t)
h = 1e-4
curvature = -(logp(map_value + h) - 2 * logp(map_value) + logp(map_value - h)) / h**2
laplace_sd = curvature ** -0.5

rows = {"exact": stats.beta.ppf([0.025, 0.975], a, b),
        "MCMC": np.quantile(mc_draws, [0.025, 0.975]),
        "Laplace": map_value + np.array([-1.96, 1.96]) * laplace_sd,
        "VI": np.quantile(vi_draws, [0.025, 0.975])}
for name, (lo, hi) in rows.items():
    print(f"{name:>8}: 95% interval {lo:6.1%} – {hi:6.1%}")

grid = np.linspace(-0.2, 1, 600)
plt.plot(grid, np.where((grid > 0) & (grid < 1), stats.beta.pdf(np.clip(grid, 1e-9, 1 - 1e-9), a, b), 0), "k", lw=2, label="exact")
plt.hist(mc_draws, bins=50, density=True, alpha=0.3, label="MCMC")
plt.plot(grid, stats.norm.pdf(grid, map_value, laplace_sd), "--", lw=2, label="Laplace")
plt.hist(vi_draws, bins=50, density=True, histtype="step", lw=2, label="VI")
plt.axvline(0, color="red", lw=1); plt.legend(); plt.xlabel("support"); plt.show()
""",
    """SUPPORTERS, HOUSEHOLDS = 58, 100                    # <- замените на 1, 10
a, b = 2 + SUPPORTERS, 2 + HOUSEHOLDS - SUPPORTERS   # точное апостериорное Beta(a, b)

with pm.Model():
    s = pm.Beta("support", 2, 2)
    pm.Binomial("supporters", n=HOUSEHOLDS, p=s, observed=SUPPORTERS)
    map_value = float(pm.find_MAP(progressbar=False)["support"])
    vi_draws = np.asarray(pm.fit(n=30_000, method="advi", random_seed=RANDOM_SEED, progressbar=False)
                          .sample(4000, random_seed=RANDOM_SEED).posterior["support"]).ravel()
    mc_draws = pm.sample(1000, tune=1000, chains=4, random_seed=RANDOM_SEED, progressbar=False).posterior["support"].values.ravel()

# Лаплас: кривизна логарифма апостериорного в вершине (вторая производная конечными разностями)
logp = lambda t: stats.beta.logpdf(t, 2, 2) + stats.binom.logpmf(SUPPORTERS, HOUSEHOLDS, t)
h = 1e-4
curvature = -(logp(map_value + h) - 2 * logp(map_value) + logp(map_value - h)) / h**2
laplace_sd = curvature ** -0.5

rows = {"точно": stats.beta.ppf([0.025, 0.975], a, b),
        "MCMC": np.quantile(mc_draws, [0.025, 0.975]),
        "Лаплас": map_value + np.array([-1.96, 1.96]) * laplace_sd,
        "VI": np.quantile(vi_draws, [0.025, 0.975])}
for name, (lo, hi) in rows.items():
    print(f"{name:>8}: 95% интервал {lo:6.1%} – {hi:6.1%}")

grid = np.linspace(-0.2, 1, 600)
plt.plot(grid, np.where((grid > 0) & (grid < 1), stats.beta.pdf(np.clip(grid, 1e-9, 1 - 1e-9), a, b), 0), "k", lw=2, label="точно")
plt.hist(mc_draws, bins=50, density=True, alpha=0.3, label="MCMC")
plt.plot(grid, stats.norm.pdf(grid, map_value, laplace_sd), "--", lw=2, label="Лаплас")
plt.hist(vi_draws, bins=50, density=True, histtype="step", lw=2, label="VI")
plt.axvline(0, color="red", lw=1); plt.legend(); plt.xlabel("поддержка"); plt.show()
""",
)

md(
    """**Question 4.** With the tiny pilot, which method gives an impossible interval? Which one has a right tail that is too long? Which one matches the exact answer?
""",
    """**Вопрос 4.** Какой метод даёт невозможный интервал на крошечном пилоте? У какого слишком длинный правый хвост? Какой совпадает с точным ответом?
""",
)

md(
    f"""## 8. Exit ticket
1. Write the **computational status** sentence for the survey model (Classwork 3), filling in your numbers:
   > *Posterior draws were obtained with NUTS in PyMC (4 chains, 1,000 warm-up and 1,000 retained draws each). All reported quantities had R-hat ≤ … and bulk and tail ESS ≥ …, and there were … divergent transitions.*
2. Name one failure that more draws **can** fix and one that they **cannot**.
3. In one sentence each: what do Laplace and VI do, and when are they risky?

The lesson page with all interactive figures: [Lesson 6]({BOOK}en/runningsamplertrustchecks/)

---
*Reproducibility:* `RANDOM_SEED = 42`; PyMC and ArviZ versions are printed in the setup cell. MCMC and VI results vary slightly between versions and runs.
""",
    f"""## 8. Итоговые вопросы
1. Напишите предложение о **вычислительном статусе** для модели опроса (классная работа 3), подставив свои числа:
   > *Апостериорная выборка получена с помощью NUTS в PyMC (4 цепи, по 1000 шагов разогрева и 1000 сохранённых шагов). Для всех сообщаемых величин R-hat ≤ …, bulk и tail ESS ≥ …, расходящихся переходов: ….*
2. Назовите одну проблему, которую **можно** исправить увеличением числа шагов, и одну, которую **нельзя**.
3. По одному предложению: что делают Лаплас и VI, и когда они рискованны?

Страница занятия со всеми интерактивными рисунками: [Занятие 6]({BOOK}ru/runningsamplertrustchecks/)

---
*Воспроизводимость:* `RANDOM_SEED = 42`; версии PyMC и ArviZ печатаются в ячейке настройки. Результаты MCMC и VI немного различаются между версиями и запусками.
""",
)


def build(lang: str) -> None:
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["colab"] = {"provenance": []}
    for kind, text in CELLS:
        src = text[lang]
        nb.cells.append(new_markdown_cell(src) if kind == "markdown" else new_code_cell(src))
    out = OUT_DIR / FILES[lang]
    nbformat.write(nb, out)
    print("wrote", out.relative_to(ROOT), f"({len(nb.cells)} cells)")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build("en")
    build("ru")
