"""Build the Lesson 5 student workbooks (English and Russian) from one source.

Run from the book root with the pymc_env environment:

    conda run -n pymc_env python scripts/build_lesson5_notebooks.py

Outputs:
    notebooks/colab/05_monte_carlo_to_mcmc.ipynb
    notebooks/colab/05_monte_carlo_to_mcmc_ru.ipynb

Every cell is defined once with an English and a Russian variant so the two
workbooks stay structurally identical. The workbook uses NumPy, SciPy and
Matplotlib only (no PyMC yet).
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks" / "colab"
GITHUB = "https://colab.research.google.com/github/iknyazeva/bayes-cogsci-book/blob/main/notebooks/colab/"
FILES = {"en": "05_monte_carlo_to_mcmc.ipynb", "ru": "05_monte_carlo_to_mcmc_ru.ipynb"}
BOOK = "https://iknyazeva.github.io/bayes-cogsci-book/"

CELLS: list[tuple[str, dict[str, str]]] = []


def md(en: str, ru: str | None = None) -> None:
    CELLS.append(("markdown", {"en": en, "ru": ru if ru is not None else en}))


def code(en: str, ru: str | None = None) -> None:
    CELLS.append(("code", {"en": en, "ru": ru if ru is not None else en}))


# ---------------------------------------------------------------------------
md(
    f"""# Lesson 5: Why Is a Posterior Hard to Compute? From Monte Carlo to MCMC
### Student workbook
*Course: Bayesian Analysis of Empirical Data (2026)*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({GITHUB}{FILES['en']})

Russian version: [{FILES['ru']}]({GITHUB}{FILES['ru']})
""",
    f"""# Занятие 5: Почему апостериорное распределение трудно вычислить? От Монте-Карло к MCMC
### Рабочая тетрадь студента
*Курс: Байесовский анализ эмпирических данных (2026)*

[![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)]({GITHUB}{FILES['ru']})

English version: [{FILES['en']}]({GITHUB}{FILES['en']})
""",
)

md(
    """## 1. Before class
* **One-minute goal:** a posterior is "prior × likelihood divided by an area". The area is easy with one unknown and impossible with many. Random draws, and in particular a Markov chain walker, give us the posterior without ever computing that area.
* **Example:** the Lesson 4 housing-subsidy survey: 58 of 100 lower-income households support the subsidy; prior Beta(2, 2); exact posterior Beta(60, 44).
* **How to work:** in every task, **predict first**, then change **one number** and rerun.
* **Tools:** NumPy, SciPy and Matplotlib only. PyMC comes in Lesson 6.
""",
    """## 1. Перед занятием
* **Цель в одну минуту:** апостериорное распределение — это «априорное × правдоподобие, делённое на площадь». С одним неизвестным площадь вычислить легко, с многими — невозможно. Случайные выборки, а особенно «блуждающая» марковская цепь, дают апостериорное распределение, ни разу не вычисляя эту площадь.
* **Пример:** опрос о жилищной субсидии из занятия 4: 58 из 100 домохозяйств с низким доходом поддерживают субсидию; априорное Beta(2, 2); точное апостериорное Beta(60, 44).
* **Как работать:** в каждом задании **сначала сделайте прогноз**, затем измените **одно число** и запустите ячейку снова.
* **Инструменты:** только NumPy, SciPy и Matplotlib. PyMC появится на занятии 6.
""",
)

code(
    """import sys
import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

IS_COLAB = "google.colab" in sys.modules
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
plt.rcParams.update({"figure.figsize": (8, 4), "axes.grid": True, "grid.alpha": 0.3})

# The running example (Lesson 4)
SUPPORTERS, HOUSEHOLDS = 58, 100
PRIOR_A, PRIOR_B = 2, 2
EXACT = stats.beta(PRIOR_A + SUPPORTERS, PRIOR_B + HOUSEHOLDS - SUPPORTERS)   # Beta(60, 44)

LBL = {"support": "support among lower-income households", "density": "density",
       "step": "step", "share": "share of nights", "island": "island", "exact": "exact posterior"}
print("Exact P(support > 50%):", round(EXACT.sf(0.5), 3))
""",
    """import sys
import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

IS_COLAB = "google.colab" in sys.modules
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
plt.rcParams.update({"figure.figsize": (8, 4), "axes.grid": True, "grid.alpha": 0.3})

# Сквозной пример (занятие 4)
SUPPORTERS, HOUSEHOLDS = 58, 100
PRIOR_A, PRIOR_B = 2, 2
EXACT = stats.beta(PRIOR_A + SUPPORTERS, PRIOR_B + HOUSEHOLDS - SUPPORTERS)   # Beta(60, 44)

LBL = {"support": "поддержка среди домохозяйств с низким доходом", "density": "плотность",
       "step": "шаг", "share": "доля ночей", "island": "остров", "exact": "точное апостериорное"}
print("Точная P(поддержка > 50%):", round(EXACT.sf(0.5), 3))
""",
)

md(
    """## 2. Predict before you run (5 minutes, on paper)
1. With **20 unknowns** and 100 grid points per unknown, how many grid cells are needed?
2. You throw 1,000 random darts at a square containing a quarter circle. Roughly what share lands inside?
3. A walker always moves uphill and sometimes downhill. Where will it spend most of its time?
""",
    """## 2. Прогноз до запуска (5 минут, на бумаге)
1. Сколько ячеек сетки понадобится для **20 неизвестных** при 100 точках на каждое?
2. Вы бросаете 1000 случайных дротиков в квадрат с вписанной четвертью круга. Какая примерно доля попадёт внутрь?
3. «Путник» всегда идёт в гору и иногда — под гору. Где он проведёт больше всего времени?
""",
)

# ---------------------------------------------------------------------------
md(
    """## 3. Classwork 1 — The area problem
Bayes' theorem gives the posterior **shape**: prior × likelihood. To turn the shape into a distribution we divide by its total **area** (the evidence). With one unknown, a grid of bars computes it.

**Change** `N_BARS` (5, 10, 20, 100) and compare with the exact area.
""",
    """## 3. Классная работа 1 — Проблема площади
Теорема Байеса задаёт **форму** апостериорного распределения: априорное × правдоподобие. Чтобы получить распределение, форму нужно разделить на её общую **площадь** (обоснованность, evidence). С одним неизвестным её можно вычислить сеткой столбиков.

**Измените** `N_BARS` (5, 10, 20, 100) и сравните с точной площадью.
""",
)

code(
    """def shape(theta):
    \"\"\"Prior x likelihood: the posterior shape WITHOUT the denominator.\"\"\"
    return stats.beta.pdf(theta, PRIOR_A, PRIOR_B) * stats.binom.pmf(SUPPORTERS, HOUSEHOLDS, theta)

N_BARS = 10                                   # <- change me
edges = np.linspace(0, 1, N_BARS + 1)
mids = (edges[:-1] + edges[1:]) / 2
area_grid = (shape(mids) * (1 / N_BARS)).sum()
area_exact = stats.betabinom.pmf(SUPPORTERS, HOUSEHOLDS, PRIOR_A, PRIOR_B)
print(f"{N_BARS} bars: area = {area_grid:.5f}   exact area = {area_exact:.5f}")

xs = np.linspace(0, 1, 500)
plt.bar(mids, shape(mids), width=1 / N_BARS * 0.95, alpha=0.4)
plt.plot(xs, shape(xs), lw=2)
plt.xlabel(LBL["support"]); plt.ylabel("prior × likelihood"); plt.show()
""",
    """def shape(theta):
    \"\"\"Априорное × правдоподобие: форма апостериорного БЕЗ знаменателя.\"\"\"
    return stats.beta.pdf(theta, PRIOR_A, PRIOR_B) * stats.binom.pmf(SUPPORTERS, HOUSEHOLDS, theta)

N_BARS = 10                                   # <- измените
edges = np.linspace(0, 1, N_BARS + 1)
mids = (edges[:-1] + edges[1:]) / 2
area_grid = (shape(mids) * (1 / N_BARS)).sum()
area_exact = stats.betabinom.pmf(SUPPORTERS, HOUSEHOLDS, PRIOR_A, PRIOR_B)
print(f"{N_BARS} столбиков: площадь = {area_grid:.5f}   точная площадь = {area_exact:.5f}")

xs = np.linspace(0, 1, 500)
plt.bar(mids, shape(mids), width=1 / N_BARS * 0.95, alpha=0.4)
plt.plot(xs, shape(xs), lw=2)
plt.xlabel(LBL["support"]); plt.ylabel("априорное × правдоподобие"); plt.show()
""",
)

md(
    """Now count the cost of a grid when the model has more unknowns. **Change** `POINTS_PER_UNKNOWN` and the list of unknowns.
""",
    """Теперь посчитаем стоимость сетки, когда неизвестных больше. **Измените** `POINTS_PER_UNKNOWN` и список числа неизвестных.
""",
)

code(
    """POINTS_PER_UNKNOWN = 100                    # <- change me
CELLS_PER_SECOND = 1e9                       # a fast computer
SECONDS_PER_YEAR = 3.156e7

for d in [1, 2, 3, 5, 10, 20]:
    cells = POINTS_PER_UNKNOWN ** d
    years = cells / CELLS_PER_SECOND / SECONDS_PER_YEAR
    print(f"{d:>2} unknowns: {cells:.1e} cells, {years:.1e} years")
""",
    """POINTS_PER_UNKNOWN = 100                    # <- измените
CELLS_PER_SECOND = 1e9                       # быстрый компьютер
SECONDS_PER_YEAR = 3.156e7

for d in [1, 2, 3, 5, 10, 20]:
    cells = POINTS_PER_UNKNOWN ** d
    years = cells / CELLS_PER_SECOND / SECONDS_PER_YEAR
    print(f"{d:>2} неизвестных: {cells:.1e} ячеек, {years:.1e} лет")
""",
)

md(
    """**Question 1.** How many unknowns can a grid handle in about one minute? What does this mean for a model with one parameter per participant?
""",
    """**Вопрос 1.** Сколько неизвестных сетка обработает примерно за минуту? Что это значит для модели, где у каждого участника свой параметр?
""",
)

# ---------------------------------------------------------------------------
md(
    """## 4. Classwork 2 — Darts compute areas (Monte Carlo)
**Part A.** Throw darts at the unit square. The share inside the quarter circle estimates its area, π/4. **Change** `N_DARTS` (10, 100, 1,000, 100,000) and run the cell several times. How much does the estimate wobble?
""",
    """## 4. Классная работа 2 — Дротики вычисляют площадь (Монте-Карло)
**Часть A.** Бросаем дротики в единичный квадрат. Доля попавших в четверть круга оценивает её площадь, π/4. **Измените** `N_DARTS` (10, 100, 1000, 100 000) и запустите ячейку несколько раз. Насколько «гуляет» оценка?
""",
)

code(
    """N_DARTS = 1000                               # <- change me
x, y = rng.uniform(0, 1, N_DARTS), rng.uniform(0, 1, N_DARTS)
inside = x**2 + y**2 <= 1
print(f"{N_DARTS} darts: pi is about {4 * inside.mean():.4f}  (true value {math.pi:.4f})")
""",
    """N_DARTS = 1000                               # <- измените
x, y = rng.uniform(0, 1, N_DARTS), rng.uniform(0, 1, N_DARTS)
inside = x**2 + y**2 <= 1
print(f"{N_DARTS} дротиков: pi примерно {4 * inside.mean():.4f}  (истинное значение {math.pi:.4f})")
""",
)

md(
    """**Part B — rejection sampling (von Neumann, 1951).** Throw darts at a box around the posterior *shape* and keep those under the curve. The kept positions are draws from the posterior. We never use the denominator.
""",
    """**Часть B — выборка с отклонением (фон Нейман, 1951).** Бросаем дротики в прямоугольник вокруг *формы* апостериорного и оставляем те, что попали под кривую. Положения оставленных дротиков — выборка из апостериорного. Знаменатель нам не нужен.
""",
)

code(
    """N_DARTS = 20_000                             # <- change me
box_height = shape(np.linspace(0, 1, 2001)).max() * 1.01
x = rng.uniform(0, 1, N_DARTS)
u = rng.uniform(0, box_height, N_DARTS)
kept = x[u < shape(x)]

print(f"kept {len(kept)} of {N_DARTS} darts ({len(kept) / N_DARTS:.1%})")
print(f"P(support > 50%): darts {np.mean(kept > 0.5):.3f}   exact {EXACT.sf(0.5):.3f}")
print(f"95% interval: darts {np.quantile(kept, [0.025, 0.975]).round(3)}   exact {EXACT.ppf([0.025, 0.975]).round(3)}")

plt.hist(kept, bins=40, density=True, alpha=0.5, label="kept darts")
plt.plot(xs, EXACT.pdf(xs), "k--", label=LBL["exact"])
plt.xlim(0.3, 0.85); plt.xlabel(LBL["support"]); plt.legend(); plt.show()
""",
    """N_DARTS = 20_000                             # <- измените
box_height = shape(np.linspace(0, 1, 2001)).max() * 1.01
x = rng.uniform(0, 1, N_DARTS)
u = rng.uniform(0, box_height, N_DARTS)
kept = x[u < shape(x)]

print(f"оставлено {len(kept)} из {N_DARTS} дротиков ({len(kept) / N_DARTS:.1%})")
print(f"P(поддержка > 50%): дротики {np.mean(kept > 0.5):.3f}   точно {EXACT.sf(0.5):.3f}")
print(f"95% интервал: дротики {np.quantile(kept, [0.025, 0.975]).round(3)}   точно {EXACT.ppf([0.025, 0.975]).round(3)}")

plt.hist(kept, bins=40, density=True, alpha=0.5, label="оставленные дротики")
plt.plot(xs, EXACT.pdf(xs), "k--", label=LBL["exact"])
plt.xlim(0.3, 0.85); plt.xlabel(LBL["support"]); plt.legend(); plt.show()
""",
)

md(
    """**Question 2.** What share of darts is wasted? Using Section 5 of the lesson, what would happen with 20 unknowns?
""",
    """**Вопрос 2.** Какая доля дротиков пропадает зря? Что произойдёт при 20 неизвестных (см. раздел 5 занятия)?
""",
)

# ---------------------------------------------------------------------------
md(
    """## 5. Classwork 3 — King Markov's islands
Seven islands in a ring with populations 1, 2, …, 7 (thousand). Each night the king proposes a neighbour at random. If it is bigger he moves. If it is smaller he moves with probability = (its population) ÷ (current population).

**Change** `START_ISLAND` (1–7) and `NIGHTS` (10, 100, 1,000, 50,000).
""",
    """## 5. Классная работа 3 — Острова короля Маркова
Семь островов по кругу с населением 1, 2, …, 7 (тыс. человек). Каждую ночь король случайно выбирает соседний остров. Если он больше — переезжает. Если меньше — переезжает с вероятностью (его население) ÷ (текущее население).

**Измените** `START_ISLAND` (1–7) и `NIGHTS` (10, 100, 1000, 50 000).
""",
)

code(
    """POPULATION = np.arange(1, 8)
START_ISLAND = 4                             # <- change me (1-7)
NIGHTS = 1000                                # <- change me

def king_markov(start, nights, rng):
    pos = start - 1
    visits = np.zeros(7)
    for _ in range(nights):
        visits[pos] += 1
        proposal = (pos + rng.choice([-1, 1])) % 7
        if rng.random() < POPULATION[proposal] / POPULATION[pos]:   # bigger: ratio > 1, always move
            pos = proposal
    return visits / nights

shares = king_markov(START_ISLAND, NIGHTS, rng)
target = POPULATION / POPULATION.sum()
plt.bar(POPULATION, shares, alpha=0.6, label="nights spent")
plt.scatter(POPULATION, target, color="darkorange", marker="_", s=900, lw=3, label="population share")
plt.xlabel(LBL["island"]); plt.ylabel(LBL["share"]); plt.legend(); plt.show()
print("share on island 7:", round(shares[6], 3), " target:", round(target[6], 3))
""",
    """POPULATION = np.arange(1, 8)
START_ISLAND = 4                             # <- измените (1-7)
NIGHTS = 1000                                # <- измените

def king_markov(start, nights, rng):
    pos = start - 1
    visits = np.zeros(7)
    for _ in range(nights):
        visits[pos] += 1
        proposal = (pos + rng.choice([-1, 1])) % 7
        if rng.random() < POPULATION[proposal] / POPULATION[pos]:   # больше: отношение > 1, переезд всегда
            pos = proposal
    return visits / nights

shares = king_markov(START_ISLAND, NIGHTS, rng)
target = POPULATION / POPULATION.sum()
plt.bar(POPULATION, shares, alpha=0.6, label="проведённые ночи")
plt.scatter(POPULATION, target, color="darkorange", marker="_", s=900, lw=3, label="доля населения")
plt.xlabel(LBL["island"]); plt.ylabel(LBL["share"]); plt.legend(); plt.show()
print("доля ночей на острове 7:", round(shares[6], 3), " цель:", round(target[6], 3))
""",
)

md(
    """**Question 3.** Does the starting island matter after 50,000 nights? After 10 nights? The king never knew the total population of all islands. Which number in the rule plays the role of the "unknown area"?
""",
    """**Вопрос 3.** Важен ли стартовый остров после 50 000 ночей? После 10 ночей? Король никогда не знал общего населения всех островов. Какое число в правиле играет роль «неизвестной площади»?
""",
)

# ---------------------------------------------------------------------------
md(
    """## 6. Classwork 4 — The Metropolis walker on the survey hill
The same rule on a continuous hill. The walker uses only the **logarithm of prior × likelihood**: logarithms avoid tiny numbers, and a ratio of heights becomes a difference of logarithms.

**Change** `STEP` (0.005, 0.05, 0.5) and `START` (0.05, 0.4, 0.9). The first `WARMUP` steps are discarded (more on this in Lesson 6).
""",
    """## 6. Классная работа 4 — «Путник» Метрополиса на холме опроса
То же правило на непрерывном холме. «Путник» использует только **логарифм априорного × правдоподобия**: логарифмы избавляют от крошечных чисел, а отношение высот превращается в разность логарифмов.

**Измените** `STEP` (0.005, 0.05, 0.5) и `START` (0.05, 0.4, 0.9). Первые `WARMUP` шагов отбрасываются (подробнее — на занятии 6).
""",
)

code(
    """def log_shape(theta):
    if not 0 < theta < 1:
        return -np.inf                       # impossible values have zero height
    return stats.beta.logpdf(theta, PRIOR_A, PRIOR_B) + stats.binom.logpmf(SUPPORTERS, HOUSEHOLDS, theta)

def metropolis(log_height, start, step, n_steps, rng):
    chain = np.empty(n_steps)
    current, current_lh = start, log_height(start)
    accepted = 0
    for i in range(n_steps):
        proposal = current + rng.normal(0, step)                       # 1. propose a nearby step
        proposal_lh = log_height(proposal)
        if np.log(rng.random()) < proposal_lh - current_lh:            # 2-3. uphill: always; downhill: sometimes
            current, current_lh = proposal, proposal_lh
            accepted += 1
        chain[i] = current                                             # 4. if rejected: stay and count again
    return chain, accepted / n_steps

STEP, START, N_STEPS, WARMUP = 0.05, 0.4, 5000, 500                    # <- change STEP and START
chain, acc = metropolis(log_shape, START, STEP, N_STEPS, rng)
draws = chain[WARMUP:]
print(f"accepted {acc:.0%} of proposals")
print(f"P(support > 50%): walker {np.mean(draws > 0.5):.3f}   exact {EXACT.sf(0.5):.3f}")
print(f"mean: walker {draws.mean():.3f}   exact {EXACT.mean():.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(chain, lw=0.7); ax[0].axvspan(0, WARMUP, color="red", alpha=0.1)
ax[0].set_xlabel(LBL["step"]); ax[0].set_ylabel(LBL["support"][:7])
ax[1].hist(draws, bins=40, density=True, alpha=0.5); ax[1].plot(xs, EXACT.pdf(xs), "k--")
ax[1].set_xlim(0.3, 0.85); ax[1].set_xlabel(LBL["support"]); plt.tight_layout(); plt.show()
""",
    """def log_shape(theta):
    if not 0 < theta < 1:
        return -np.inf                       # у невозможных значений нулевая высота
    return stats.beta.logpdf(theta, PRIOR_A, PRIOR_B) + stats.binom.logpmf(SUPPORTERS, HOUSEHOLDS, theta)

def metropolis(log_height, start, step, n_steps, rng):
    chain = np.empty(n_steps)
    current, current_lh = start, log_height(start)
    accepted = 0
    for i in range(n_steps):
        proposal = current + rng.normal(0, step)                       # 1. предложить соседний шаг
        proposal_lh = log_height(proposal)
        if np.log(rng.random()) < proposal_lh - current_lh:            # 2-3. в гору — всегда; под гору — иногда
            current, current_lh = proposal, proposal_lh
            accepted += 1
        chain[i] = current                                             # 4. если отказ — остаться и посчитать ещё раз
    return chain, accepted / n_steps

STEP, START, N_STEPS, WARMUP = 0.05, 0.4, 5000, 500                    # <- измените STEP и START
chain, acc = metropolis(log_shape, START, STEP, N_STEPS, rng)
draws = chain[WARMUP:]
print(f"принято {acc:.0%} предложений")
print(f"P(поддержка > 50%): путник {np.mean(draws > 0.5):.3f}   точно {EXACT.sf(0.5):.3f}")
print(f"среднее: путник {draws.mean():.3f}   точно {EXACT.mean():.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(chain, lw=0.7); ax[0].axvspan(0, WARMUP, color="red", alpha=0.1)
ax[0].set_xlabel(LBL["step"]); ax[0].set_ylabel("поддержка")
ax[1].hist(draws, bins=40, density=True, alpha=0.5); ax[1].plot(xs, EXACT.pdf(xs), "k--")
ax[1].set_xlim(0.3, 0.85); ax[1].set_xlabel(LBL["support"]); plt.tight_layout(); plt.show()
""",
)

md(
    """**The denominator really is never needed.** Multiply the shape by 1,000 (add log 1000 to its logarithm) and rerun the walker with the same random seed. Compare the two chains.
""",
    """**Знаменатель действительно не нужен.** Умножим форму на 1000 (прибавим log 1000 к её логарифму) и запустим «путника» с тем же начальным значением генератора. Сравните две цепи.
""",
)

code(
    """chain_a, _ = metropolis(log_shape, 0.4, 0.05, 3000, np.random.default_rng(1))
chain_b, _ = metropolis(lambda t: log_shape(t) + np.log(1000), 0.4, 0.05, 3000, np.random.default_rng(1))
print("Identical chains:", np.allclose(chain_a, chain_b))
""",
    """chain_a, _ = metropolis(log_shape, 0.4, 0.05, 3000, np.random.default_rng(1))
chain_b, _ = metropolis(lambda t: log_shape(t) + np.log(1000), 0.4, 0.05, 3000, np.random.default_rng(1))
print("Цепи совпадают:", np.allclose(chain_a, chain_b))
""",
)

md(
    """**Question 4.** With `STEP = 0.005`, what goes wrong? With `STEP = 0.5`? With `START = 0.05` and `WARMUP = 0`? Write one sentence for each. These are the problems Lesson 6 teaches you to detect.
""",
    """**Вопрос 4.** Что идёт не так при `STEP = 0.005`? При `STEP = 0.5`? При `START = 0.05` и `WARMUP = 0`? Напишите по одному предложению. Именно эти проблемы мы научимся обнаруживать на занятии 6.
""",
)

# ---------------------------------------------------------------------------
md(
    """## 7. 🚀 Stretch — two unknowns at once
Walk over **both** groups together: lower-income (58 of 100) and higher-income (40 of 100). The walker proposes a step in both directions at once. Estimate P(gap > 5 points) and compare with Lesson 4 (exact: 0.964).
""",
    """## 7. 🚀 Задание повышенной сложности — два неизвестных сразу
«Путник» ходит сразу по **двум** группам: с низким доходом (58 из 100) и с высоким (40 из 100). Шаг предлагается сразу в обоих направлениях. Оцените P(разрыв > 5 п.п.) и сравните с занятием 4 (точно: 0,964).
""",
)

code(
    """def log_shape_2(t):
    lo, hi = t
    if not (0 < lo < 1 and 0 < hi < 1):
        return -np.inf
    return (stats.beta.logpdf(lo, 2, 2) + stats.binom.logpmf(58, 100, lo)
            + stats.beta.logpdf(hi, 2, 2) + stats.binom.logpmf(40, 100, hi))

current = np.array([0.5, 0.5]); current_lh = log_shape_2(current)
chain2 = np.empty((20_000, 2))
for i in range(len(chain2)):
    proposal = current + rng.normal(0, 0.05, size=2)
    proposal_lh = log_shape_2(proposal)
    if np.log(rng.random()) < proposal_lh - current_lh:
        current, current_lh = proposal, proposal_lh
    chain2[i] = current
draws2 = chain2[2000:]
gap = draws2[:, 0] - draws2[:, 1]
print("P(gap > 5 points):", round(np.mean(gap > 0.05), 3), "  exact: 0.964")
plt.plot(draws2[::10, 0], draws2[::10, 1], ".", ms=2, alpha=0.4)
plt.xlabel("lower-income support"); plt.ylabel("higher-income support"); plt.show()
""",
    """def log_shape_2(t):
    lo, hi = t
    if not (0 < lo < 1 and 0 < hi < 1):
        return -np.inf
    return (stats.beta.logpdf(lo, 2, 2) + stats.binom.logpmf(58, 100, lo)
            + stats.beta.logpdf(hi, 2, 2) + stats.binom.logpmf(40, 100, hi))

current = np.array([0.5, 0.5]); current_lh = log_shape_2(current)
chain2 = np.empty((20_000, 2))
for i in range(len(chain2)):
    proposal = current + rng.normal(0, 0.05, size=2)
    proposal_lh = log_shape_2(proposal)
    if np.log(rng.random()) < proposal_lh - current_lh:
        current, current_lh = proposal, proposal_lh
    chain2[i] = current
draws2 = chain2[2000:]
gap = draws2[:, 0] - draws2[:, 1]
print("P(разрыв > 5 п.п.):", round(np.mean(gap > 0.05), 3), "  точно: 0.964")
plt.plot(draws2[::10, 0], draws2[::10, 1], ".", ms=2, alpha=0.4)
plt.xlabel("поддержка: низкий доход"); plt.ylabel("поддержка: высокий доход"); plt.show()
""",
)

md(
    """## 8. 🚀 Stretch — why blind darts fail in many dimensions
Throw random points into a d-dimensional box and count how many land inside the ball, i.e. in the region where the probability is. **Change** the list of dimensions.
""",
    """## 8. 🚀 Задание повышенной сложности — почему слепые дротики не работают в многомерии
Бросаем случайные точки в d-мерный куб и считаем, сколько попало в шар — туда, где сосредоточена вероятность. **Измените** список размерностей.
""",
)

code(
    """for d in [2, 3, 5, 10, 15]:
    pts = rng.uniform(-1, 1, size=(200_000, d))
    print(f"{d:>2} unknowns: {np.mean((pts**2).sum(axis=1) <= 1):.5%} of points are useful")
""",
    """for d in [2, 3, 5, 10, 15]:
    pts = rng.uniform(-1, 1, size=(200_000, d))
    print(f"{d:>2} неизвестных: полезны {np.mean((pts**2).sum(axis=1) <= 1):.5%} точек")
""",
)

md(
    f"""## 9. Exit ticket
Answer in two or three sentences each:
1. Why does computing a posterior mean computing an area, and why does a grid fail for many unknowns?
2. Why does the Metropolis walker never need the denominator?
3. A rejected proposal makes the walker stay. Why must the repeated value be kept?

The lesson page with all interactive figures: [Lesson 5]({BOOK}en/montecarlotomcmc/)

---
*Reproducibility:* NumPy random generator with `RANDOM_SEED = 42`. Results change slightly if you rerun cells in a different order; that wobble is the Monte Carlo error.
""",
    f"""## 9. Итоговые вопросы
Ответьте в двух-трёх предложениях на каждый:
1. Почему вычисление апостериорного распределения — это вычисление площади, и почему сетка не работает при многих неизвестных?
2. Почему «путнику» Метрополиса никогда не нужен знаменатель?
3. При отказе «путник» остаётся на месте. Почему повторённое значение нужно сохранить?

Страница занятия со всеми интерактивными рисунками: [Занятие 5]({BOOK}ru/montecarlotomcmc/)

---
*Воспроизводимость:* генератор NumPy с `RANDOM_SEED = 42`. При запуске ячеек в другом порядке результаты немного меняются — это и есть ошибка Монте-Карло.
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
