"""Build the advanced computation workbook (English and Russian) from one source.

This is the optional lab linked from Lesson 5's further reading (Monte Carlo
estimation, Markov chains, Metropolis–Hastings, Gibbs sampling): Metropolis from
scratch, step size, the correlated ridge, and a PyMC/NUTS benchmark.

Run from the book root with the pymc_env environment:

    conda run -n pymc_env python scripts/build_bayesian_computation_notebooks.py

Outputs:
    notebooks/colab/06_bayesian_computation.ipynb
    notebooks/colab/06_bayesian_computation_ru.ipynb

The code works with PyMC 5+ / ArviZ 0.x (Colab) and PyMC 6 / ArviZ 1.x (course
environment): diagnostics use ``az.rhat`` / ``az.ess`` and ``az.summary`` is
printed without selecting version-specific columns.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks" / "colab"
GITHUB = "https://colab.research.google.com/github/iknyazeva/bayes-cogsci-book/blob/main/notebooks/colab/"
FILES = {"en": "06_bayesian_computation.ipynb", "ru": "06_bayesian_computation_ru.ipynb"}
BOOK = "https://iknyazeva.github.io/bayes-cogsci-book/"

CELLS: list[tuple[str, dict[str, str]]] = []


def md(en: str, ru: str | None = None) -> None:
    CELLS.append(("markdown", {"en": en, "ru": ru if ru is not None else en}))


def code(en: str, ru: str | None = None) -> None:
    CELLS.append(("code", {"en": en, "ru": ru if ru is not None else en}))


# ---------------------------------------------------------------------------
md(
    rf"""# Bayesian Computation Lab — From Metropolis to PyMC & NUTS
### Optional workbook for Lessons 5–6
*Course: Bayesian Analysis of Empirical Data (2026). Author: Irina Knyazeva*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({GITHUB}{FILES['en']})

Russian version: [{FILES['ru']}]({GITHUB}{FILES['ru']})

In realistic research the normalising denominator in Bayes' rule — the marginal likelihood $p(y) = \int p(y \mid \theta) p(\theta)\, d\theta$ — is an intractable high-dimensional integral. We cannot solve it analytically, and naive rejection sampling wastes almost every draw.

This lab walks through the computational ladder of modern Bayesian statistics:
1. **Direct Monte Carlo vs. MCMC:** independent draws vs. dependent Markov chains.
2. **Metropolis–Hastings from scratch:** an unnormalised-posterior sampler in about 15 lines of Python.
3. **Step size:** why the proposal scale trades acceptance rate against exploration speed.
4. **The correlated 2D ridge:** why random-walk proposals struggle when parameters move together.
5. **PyMC/NUTS benchmark:** checking Hamiltonian Monte Carlo against the exact $\operatorname{{Beta}}(50, 34)$ posterior.
6. **Trust checks:** $\widehat{{R}}$, bulk/tail ESS, rank plots and divergences before any interpretation.

Book pages: [Lesson 5]({BOOK}en/montecarlotomcmc/) · [Monte Carlo estimation]({BOOK}en/montecarloestimation/) · [Metropolis–Hastings]({BOOK}en/metropolishastings/) · [HMC and NUTS]({BOOK}en/hamiltonianmontecarlonuts/) · [Lesson 6]({BOOK}en/runningsamplertrustchecks/) · [Convergence diagnostics]({BOOK}en/assessingconvergence/)
""",
    rf"""# Лаборатория байесовских вычислений: от Метрополиса к PyMC и NUTS
### Дополнительная тетрадь к занятиям 5–6
*Курс: Байесовский анализ эмпирических данных (2026). Автор: Ирина Князева*

[![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)]({GITHUB}{FILES['ru']})

English version: [{FILES['en']}]({GITHUB}{FILES['en']})

В реальных исследованиях знаменатель теоремы Байеса — маргинальное правдоподобие $p(y) = \int p(y \mid \theta) p(\theta)\, d\theta$ — это невычислимый многомерный интеграл. Аналитически его не взять, а наивный метод отклонения выбрасывает почти все сэмплы.

Эта лаборатория проходит по «лестнице» вычислений современной байесовской статистики:
1. **Прямой Монте-Карло и MCMC:** независимые сэмплы и зависимые марковские цепи.
2. **Метрополис–Гастингс с нуля:** сэмплер ненормированного апостериорного распределения примерно в 15 строк Python.
3. **Размер шага:** почему масштаб предложения — это компромисс между долей принятых шагов и скоростью исследования.
4. **Коррелированный двумерный гребень:** почему случайное блуждание не справляется, когда параметры меняются вместе.
5. **Эталон PyMC/NUTS:** проверка гамильтонова Монте-Карло по точному апостериорному $\operatorname{{Beta}}(50, 34)$.
6. **Проверки доверия:** $\widehat{{R}}$, bulk/tail ESS, ранговые графики и расходящиеся переходы — до любой интерпретации.

Страницы книги: [Занятие 5]({BOOK}ru/montecarlotomcmc/) · [Оценивание методом Монте-Карло]({BOOK}ru/montecarloestimation/) · [Метрополис–Гастингс]({BOOK}ru/metropolishastings/) · [HMC и NUTS]({BOOK}ru/hamiltonianmontecarlonuts/) · [Занятие 6]({BOOK}ru/runningsamplertrustchecks/) · [Диагностика сходимости]({BOOK}ru/assessingconvergence/)
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
import matplotlib.pyplot as plt
import scipy.stats as stats
import logging, warnings
logging.getLogger("pymc").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

plt.rcParams.update({"figure.autolayout": True, "axes.grid": True, "grid.alpha": 0.3})
rng = np.random.default_rng(2026)
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
import matplotlib.pyplot as plt
import scipy.stats as stats
import logging, warnings
logging.getLogger("pymc").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

plt.rcParams.update({"figure.autolayout": True, "axes.grid": True, "grid.alpha": 0.3})
rng = np.random.default_rng(2026)
print("PyMC", pm.__version__, "| ArviZ", az.__version__)
""",
)

# ---------------------------------------------------------------------------
md(
    r"""---
## 1. Direct Monte Carlo: integration by simulation

When independent draws $\theta^{(1)}, \dots, \theta^{(S)}$ can be taken directly from the posterior $p(\theta \mid y)$, any posterior expectation $\mathbb{E}[g(\theta) \mid y]$ is estimated by the sample average:

$$\widehat{\mathbb{E}}[g(\theta)] = \frac{1}{S} \sum_{s=1}^S g(\theta^{(s)}).$$

The numerical accuracy of this finite simulation is the **Monte Carlo standard error (MCSE)**:

$$\operatorname{MCSE}(\bar{g}) = \frac{\operatorname{SD}(g(\theta))}{\sqrt{S}}.$$

> **Posterior uncertainty vs. Monte Carlo error.** A credible interval reflects real uncertainty about $\theta$ given the data and prior. MCSE measures only simulation noise: going from $S$ to $4S$ draws halves the MCSE but does **not** make the posterior narrower.
""",
    r"""---
## 1. Прямой Монте-Карло: интегрирование моделированием

Если независимые сэмплы $\theta^{(1)}, \dots, \theta^{(S)}$ можно получать прямо из апостериорного распределения $p(\theta \mid y)$, любое апостериорное ожидание $\mathbb{E}[g(\theta) \mid y]$ оценивается выборочным средним:

$$\widehat{\mathbb{E}}[g(\theta)] = \frac{1}{S} \sum_{s=1}^S g(\theta^{(s)}).$$

Численную точность такого конечного моделирования измеряет **стандартная ошибка Монте-Карло (MCSE)**:

$$\operatorname{MCSE}(\bar{g}) = \frac{\operatorname{SD}(g(\theta))}{\sqrt{S}}.$$

> **Апостериорная неопределённость и ошибка Монте-Карло.** Байесовский интервал отражает реальную неопределённость относительно $\theta$ при данных и априорном распределении. MCSE измеряет только шум моделирования: переход от $S$ к $4S$ сэмплам вдвое уменьшает MCSE, но **не** сужает апостериорное распределение.
""",
)

code(
    """# Exact conjugate benchmark: Beta(2, 2) prior + 48 of 80 -> Beta(50, 34)
alpha_post, beta_post = 50, 34
exact_mean = alpha_post / (alpha_post + beta_post)          # 50 / 84 = 0.5952
exact_eti = stats.beta.ppf([0.025, 0.975], alpha_post, beta_post)

S = 4000                                                     # independent draws
direct_draws = rng.beta(alpha_post, beta_post, size=S)
mc_mean = direct_draws.mean()
mcse = direct_draws.std(ddof=1) / np.sqrt(S)

print(f"Exact mean:              {exact_mean:.5f}")
print(f"Monte Carlo mean:        {mc_mean:.5f} ± {mcse:.5f} (MCSE)")
print(f"Exact 95% interval:      [{exact_eti[0]:.4f}, {exact_eti[1]:.4f}]")
print(f"Sampled 95% quantiles:   [{np.percentile(direct_draws, 2.5):.4f}, {np.percentile(direct_draws, 97.5):.4f}]")
""",
    """# Точный сопряжённый эталон: априорное Beta(2, 2) + 48 из 80 -> Beta(50, 34)
alpha_post, beta_post = 50, 34
exact_mean = alpha_post / (alpha_post + beta_post)          # 50 / 84 = 0.5952
exact_eti = stats.beta.ppf([0.025, 0.975], alpha_post, beta_post)

S = 4000                                                     # независимые сэмплы
direct_draws = rng.beta(alpha_post, beta_post, size=S)
mc_mean = direct_draws.mean()
mcse = direct_draws.std(ddof=1) / np.sqrt(S)

print(f"Точное среднее:          {exact_mean:.5f}")
print(f"Среднее Монте-Карло:     {mc_mean:.5f} ± {mcse:.5f} (MCSE)")
print(f"Точный 95% интервал:     [{exact_eti[0]:.4f}, {exact_eti[1]:.4f}]")
print(f"Квантили по сэмплам:     [{np.percentile(direct_draws, 2.5):.4f}, {np.percentile(direct_draws, 97.5):.4f}]")
""",
)

# ---------------------------------------------------------------------------
md(
    r"""---
## 2. A Metropolis–Hastings sampler from scratch

In complex models we cannot draw independent samples because the denominator $p(y)$ is unknown. But we can evaluate the **unnormalised posterior**

$$\widetilde{p}(\theta \mid y) = p(y \mid \theta) p(\theta).$$

The **Metropolis–Hastings algorithm** builds a Markov chain whose stationary distribution is exactly the posterior:
1. From the current state $\theta^{(t)}$, propose $\theta^* \sim q(\theta^* \mid \theta^{(t)})$.
2. Compute the acceptance probability
   $$\alpha = \min\left(1, \; \frac{\widetilde{p}(\theta^* \mid y)}{\widetilde{p}(\theta^{(t)} \mid y)} \cdot \frac{q(\theta^{(t)} \mid \theta^*)}{q(\theta^* \mid \theta^{(t)})}\right).$$
   For a symmetric proposal the $q$ terms cancel, and the unknown $p(y)$ cancels too.
3. Draw $u \sim \operatorname{Uniform}(0, 1)$. If $u < \alpha$, accept $\theta^{(t+1)} = \theta^*$; otherwise **stay**: $\theta^{(t+1)} = \theta^{(t)}$.
""",
    r"""---
## 2. Сэмплер Метрополиса–Гастингса с нуля

В сложных моделях независимые сэмплы получить нельзя: знаменатель $p(y)$ неизвестен. Но мы можем вычислить **ненормированное апостериорное распределение**

$$\widetilde{p}(\theta \mid y) = p(y \mid \theta) p(\theta).$$

**Алгоритм Метрополиса–Гастингса** строит марковскую цепь, стационарное распределение которой в точности равно апостериорному:
1. Из текущего состояния $\theta^{(t)}$ предлагаем $\theta^* \sim q(\theta^* \mid \theta^{(t)})$.
2. Вычисляем вероятность принятия
   $$\alpha = \min\left(1, \; \frac{\widetilde{p}(\theta^* \mid y)}{\widetilde{p}(\theta^{(t)} \mid y)} \cdot \frac{q(\theta^{(t)} \mid \theta^*)}{q(\theta^* \mid \theta^{(t)})}\right).$$
   Для симметричного предложения члены с $q$ сокращаются, а неизвестное $p(y)$ сокращается тоже.
3. Сэмплируем $u \sim \operatorname{Uniform}(0, 1)$. Если $u < \alpha$, принимаем $\theta^{(t+1)} = \theta^*$; иначе **остаёмся**: $\theta^{(t+1)} = \theta^{(t)}$.
""",
)

code(
    '''def unnormalized_log_posterior(theta, y=48, n=80, a_prior=2, b_prior=2):
    """Log joint density log p(y, theta) = log p(y | theta) + log p(theta)."""
    if theta <= 0.0 or theta >= 1.0:
        return -np.inf                                      # outside the parameter space
    return stats.beta.logpdf(theta, a_prior, b_prior) + stats.binom.logpmf(y, n, theta)


def run_metropolis(log_target, theta_start, n_draws, prop_sd, seed=2026):
    """Random-walk Metropolis sampler."""
    gen = np.random.default_rng(seed)
    chain = np.zeros(n_draws)
    current, current_logp = theta_start, log_target(theta_start)
    accepted = 0
    for t in range(n_draws):
        proposal = current + gen.normal(0, prop_sd)         # Gaussian random-walk proposal
        proposal_logp = log_target(proposal)
        if np.log(gen.uniform()) < proposal_logp - current_logp:   # log acceptance ratio
            current, current_logp = proposal, proposal_logp
            accepted += 1
        chain[t] = current                                  # rejected: repeat the current state
    return chain, accepted / n_draws


chain_test, acc_test = run_metropolis(unnormalized_log_posterior, 0.5, 5000, 0.15)
print(f"Test run: acceptance {acc_test:.0%}, mean after warm-up {chain_test[500:].mean():.4f} (exact {exact_mean:.4f})")
''',
    '''def unnormalized_log_posterior(theta, y=48, n=80, a_prior=2, b_prior=2):
    """Логарифм совместной плотности log p(y, theta) = log p(y | theta) + log p(theta)."""
    if theta <= 0.0 or theta >= 1.0:
        return -np.inf                                      # вне пространства параметров
    return stats.beta.logpdf(theta, a_prior, b_prior) + stats.binom.logpmf(y, n, theta)


def run_metropolis(log_target, theta_start, n_draws, prop_sd, seed=2026):
    """Сэмплер Метрополиса со случайным блужданием."""
    gen = np.random.default_rng(seed)
    chain = np.zeros(n_draws)
    current, current_logp = theta_start, log_target(theta_start)
    accepted = 0
    for t in range(n_draws):
        proposal = current + gen.normal(0, prop_sd)         # гауссово предложение
        proposal_logp = log_target(proposal)
        if np.log(gen.uniform()) < proposal_logp - current_logp:   # логарифм отношения
            current, current_logp = proposal, proposal_logp
            accepted += 1
        chain[t] = current                                  # отказ: повторяем текущее состояние
    return chain, accepted / n_draws


chain_test, acc_test = run_metropolis(unnormalized_log_posterior, 0.5, 5000, 0.15)
print(f"Пробный запуск: принято {acc_test:.0%}, среднее после разогрева {chain_test[500:].mean():.4f} (точно {exact_mean:.4f})")
''',
)

# ---------------------------------------------------------------------------
md(
    r"""---
## 3. The step-size experiment

In random-walk Metropolis the proposal standard deviation $\sigma_{\text{prop}}$ controls efficiency:
* **Too small ($\sigma = 0.01$):** proposals are accepted almost every time, but the chain explores by slow diffusion.
* **Too large ($\sigma = 2.50$):** most proposals land in very low-density regions; the chain rejects and "sticks".
* **Tuned ($\sigma = 0.15$):** balances movement and acceptance (for a 1-D Normal-like target, about 44% acceptance is efficient).

**Predict first:** which chain will have the highest acceptance rate? Which will have the fastest-decaying autocorrelation?
""",
    r"""---
## 3. Эксперимент с размером шага

В алгоритме Метрополиса со случайным блужданием эффективность определяет стандартное отклонение предложения $\sigma_{\text{prop}}$:
* **Слишком мал ($\sigma = 0.01$):** предложения принимаются почти всегда, но цепь исследует пространство медленной диффузией.
* **Слишком велик ($\sigma = 2.50$):** большинство предложений попадает в области очень малой плотности; цепь отказывается и «залипает».
* **Подобран ($\sigma = 0.15$):** баланс движения и принятия (для одномерной цели, похожей на нормальную, эффективна доля принятия около 44%).

**Сначала предскажите:** у какой цепи будет наибольшая доля принятия? У какой автокорреляция затухает быстрее всего?
""",
)

code(
    r'''N_DRAWS = 5000
scales = [0.01, 0.15, 2.5]
labels = ["Too small (σ = 0.01)", "Tuned (σ = 0.15)", "Too large (σ = 2.50)"]
colors = ["tab:orange", "tab:blue", "tab:red"]
results = [run_metropolis(unnormalized_log_posterior, 0.1, N_DRAWS, s) for s in scales]

fig, axes = plt.subplots(3, 2, figsize=(13, 8))
for i, ((chain, acc), label, col) in enumerate(zip(results, labels, colors)):
    ess = float(np.asarray(az.ess(chain[None, 500:])))
    axes[i, 0].plot(chain[:1000], lw=0.8, color=col)
    axes[i, 0].axhline(exact_mean, color="black", ls="--", lw=1, alpha=0.7)
    axes[i, 0].set_title(f"{label}: accepted {acc:.0%}, ESS ≈ {ess:.0f}")
    axes[i, 0].set_ylabel(r"$\theta$")
    axes[i, 0].set_ylim(0, 1)
    acf = [1.0] + [np.corrcoef(chain[500:-lag], chain[500 + lag:])[0, 1] for lag in range(1, 40)]
    axes[i, 1].stem(range(40), acf, basefmt=" ")
    axes[i, 1].set_title("Autocorrelation (lags 0–39)")
    axes[i, 1].set_ylim(-0.1, 1.05)
axes[2, 0].set_xlabel("Iteration")
axes[2, 1].set_xlabel("Lag")
plt.show()
''',
    r'''N_DRAWS = 5000
scales = [0.01, 0.15, 2.5]
labels = ["Слишком мал (σ = 0.01)", "Подобран (σ = 0.15)", "Слишком велик (σ = 2.50)"]
colors = ["tab:orange", "tab:blue", "tab:red"]
results = [run_metropolis(unnormalized_log_posterior, 0.1, N_DRAWS, s) for s in scales]

fig, axes = plt.subplots(3, 2, figsize=(13, 8))
for i, ((chain, acc), label, col) in enumerate(zip(results, labels, colors)):
    ess = float(np.asarray(az.ess(chain[None, 500:])))
    axes[i, 0].plot(chain[:1000], lw=0.8, color=col)
    axes[i, 0].axhline(exact_mean, color="black", ls="--", lw=1, alpha=0.7)
    axes[i, 0].set_title(f"{label}: принято {acc:.0%}, ESS ≈ {ess:.0f}")
    axes[i, 0].set_ylabel(r"$\theta$")
    axes[i, 0].set_ylim(0, 1)
    acf = [1.0] + [np.corrcoef(chain[500:-lag], chain[500 + lag:])[0, 1] for lag in range(1, 40)]
    axes[i, 1].stem(range(40), acf, basefmt=" ")
    axes[i, 1].set_title("Автокорреляция (лаги 0–39)")
    axes[i, 1].set_ylim(-0.1, 1.05)
axes[2, 0].set_xlabel("Итерация")
axes[2, 1].set_xlabel("Лаг")
plt.show()
''',
)

# ---------------------------------------------------------------------------
md(
    rf"""---
## 4. The correlated 2D ridge: why random walks struggle

With more parameters, they are often correlated (an intercept and a slope, or variance components). A random-walk proposal steps in a round ball, so most proposals cut across a narrow diagonal ridge and are rejected.

This motivates **Hamiltonian Monte Carlo (HMC)**, which uses the gradient of $\log p(\theta \mid y)$ to glide along the ridge — the "skateboarder" of Lesson 5 ([HMC and NUTS]({BOOK}en/hamiltonianmontecarlonuts/)).
""",
    rf"""---
## 4. Коррелированный двумерный гребень: почему случайное блуждание не справляется

Когда параметров больше, они часто коррелированы (свободный член и наклон, компоненты дисперсии). Случайное блуждание предлагает шаги в круглом шаре, поэтому большинство предложений пересекает узкий диагональный гребень и отклоняется.

Это мотивирует **гамильтонов Монте-Карло (HMC)**: он использует градиент $\log p(\theta \mid y)$, чтобы скользить вдоль гребня, — «скейтбордист» из занятия 5 ([HMC и NUTS]({BOOK}ru/hamiltonianmontecarlonuts/)).
""",
)

code(
    r'''# A correlated 2D Gaussian posterior ridge (correlation 0.92)
mu_2d = np.array([0.0, 0.0])
cov_2d = np.array([[1.0, 0.92], [0.92, 1.0]])
inv_cov_2d = np.linalg.inv(cov_2d)
log_target_2d = lambda theta: -0.5 * theta @ inv_cov_2d @ theta

n_steps, prop_sd_2d = 1500, 0.25
rw_chain = np.zeros((n_steps, 2))
curr = np.array([-2.0, -2.0])
for t in range(n_steps):
    prop = curr + rng.normal(0, prop_sd_2d, size=2)
    if np.log(rng.uniform()) < log_target_2d(prop) - log_target_2d(curr):
        curr = prop
    rw_chain[t] = curr

x_grid, y_grid = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
z_density = stats.multivariate_normal(mu_2d, cov_2d).pdf(np.dstack((x_grid, y_grid)))
plt.figure(figsize=(7, 6))
plt.contour(x_grid, y_grid, z_density, levels=7, cmap="Blues_r", alpha=0.7)
plt.plot(rw_chain[:400, 0], rw_chain[:400, 1], color="crimson", alpha=0.6, lw=0.9, marker=".", markersize=3, label="random walk (400 steps)")
plt.scatter(*rw_chain[0], color="black", s=80, zorder=5, label="start (-2, -2)")
plt.title("A random walk on a correlated ridge (ρ = 0.92)")
plt.xlabel(r"$\theta_1$"); plt.ylabel(r"$\theta_2$"); plt.legend()
plt.show()
''',
    r'''# Коррелированный двумерный нормальный гребень (корреляция 0.92)
mu_2d = np.array([0.0, 0.0])
cov_2d = np.array([[1.0, 0.92], [0.92, 1.0]])
inv_cov_2d = np.linalg.inv(cov_2d)
log_target_2d = lambda theta: -0.5 * theta @ inv_cov_2d @ theta

n_steps, prop_sd_2d = 1500, 0.25
rw_chain = np.zeros((n_steps, 2))
curr = np.array([-2.0, -2.0])
for t in range(n_steps):
    prop = curr + rng.normal(0, prop_sd_2d, size=2)
    if np.log(rng.uniform()) < log_target_2d(prop) - log_target_2d(curr):
        curr = prop
    rw_chain[t] = curr

x_grid, y_grid = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
z_density = stats.multivariate_normal(mu_2d, cov_2d).pdf(np.dstack((x_grid, y_grid)))
plt.figure(figsize=(7, 6))
plt.contour(x_grid, y_grid, z_density, levels=7, cmap="Blues_r", alpha=0.7)
plt.plot(rw_chain[:400, 0], rw_chain[:400, 1], color="crimson", alpha=0.6, lw=0.9, marker=".", markersize=3, label="случайное блуждание (400 шагов)")
plt.scatter(*rw_chain[0], color="black", s=80, zorder=5, label="старт (-2, -2)")
plt.title("Случайное блуждание на коррелированном гребне (ρ = 0.92)")
plt.xlabel(r"$\theta_1$"); plt.ylabel(r"$\theta_2$"); plt.legend()
plt.show()
''',
)

# ---------------------------------------------------------------------------
md(
    r"""---
## 5. The PyMC & NUTS benchmark

Now the production tool: **PyMC** with the **No-U-Turn Sampler (NUTS)**. We fit the same Beta–Binomial problem ($y = 48$, $n = 80$, Beta(2, 2) prior), whose exact answer $\operatorname{Beta}(50, 34)$ we know.
""",
    r"""---
## 5. Эталон PyMC и NUTS

Теперь рабочий инструмент: **PyMC** с сэмплером **NUTS** (No-U-Turn Sampler). Подгоняем ту же бета-биномиальную задачу ($y = 48$, $n = 80$, априорное Beta(2, 2)), точный ответ которой — $\operatorname{Beta}(50, 34)$ — нам известен.
""",
)

code(
    """with pm.Model() as benchmark_model:
    theta = pm.Beta("theta", alpha=2, beta=2)                      # prior
    pm.Binomial("y_obs", n=80, p=theta, observed=48)               # likelihood: 48 of 80
    idata = pm.sample(draws=1000, tune=1000, chains=4, random_seed=2026, progressbar=False)
print("Sampling complete")
""",
    """with pm.Model() as benchmark_model:
    theta = pm.Beta("theta", alpha=2, beta=2)                      # априорное
    pm.Binomial("y_obs", n=80, p=theta, observed=48)               # правдоподобие: 48 из 80
    idata = pm.sample(draws=1000, tune=1000, chains=4, random_seed=2026, progressbar=False)
print("Сэмплирование завершено")
""",
)

md(
    r"""---
## 6. The trust checks

Before interpreting any model, run the course checks:

| Diagnostic | Course rule | What it tests |
|:---|:---|:---|
| $\widehat{R}$ | $\widehat{R} \le 1.01$ | Do the chains explore the same distribution? |
| Bulk ESS | $\ge 400$ | Enough effective draws for means and medians? |
| Tail ESS | $\ge 400$ | Are interval endpoints (2.5%, 97.5%) numerically stable? |
| Divergences | exactly 0 | Did the sampler meet regions it could not follow? |
| Rank plots | roughly uniform for every chain | Are the chains well mixed across the whole support? |
""",
    r"""---
## 6. Проверки доверия

Прежде чем интерпретировать модель, пройдите проверки курса:

| Диагностика | Правило курса | Что проверяет |
|:---|:---|:---|
| $\widehat{R}$ | $\widehat{R} \le 1.01$ | Исследуют ли цепи одно и то же распределение? |
| Bulk ESS | $\ge 400$ | Достаточно ли эффективных сэмплов для средних и медиан? |
| Tail ESS | $\ge 400$ | Устойчивы ли численно концы интервала (2,5%, 97,5%)? |
| Расходящиеся переходы | ровно 0 | Встретил ли сэмплер области, которые не смог пройти? |
| Ранговые графики | примерно равномерные для каждой цепи | Хорошо ли перемешаны цепи по всему носителю? |
""",
)

code(
    """print(az.summary(idata, var_names=["theta"]))

r_hat = float(np.asarray(az.rhat(idata, var_names=["theta"])["theta"]))
ess_bulk = float(np.asarray(az.ess(idata, var_names=["theta"])["theta"]))
ess_tail = float(np.asarray(az.ess(idata, var_names=["theta"], method="tail")["theta"]))
n_divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
print(f"\\nR-hat {r_hat:.3f} | bulk ESS {ess_bulk:.0f} | tail ESS {ess_tail:.0f} | divergences {n_divergences}")

az.plot_rank(idata, var_names=["theta"])
plt.show()
""",
    """print(az.summary(idata, var_names=["theta"]))

r_hat = float(np.asarray(az.rhat(idata, var_names=["theta"])["theta"]))
ess_bulk = float(np.asarray(az.ess(idata, var_names=["theta"])["theta"]))
ess_tail = float(np.asarray(az.ess(idata, var_names=["theta"], method="tail")["theta"]))
n_divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
print(f"\\nR-hat {r_hat:.3f} | bulk ESS {ess_bulk:.0f} | tail ESS {ess_tail:.0f} | расходящихся переходов {n_divergences}")

az.plot_rank(idata, var_names=["theta"])
plt.show()
""",
)

md("### Checking NUTS against the exact answer", "### Сверка NUTS с точным ответом")

code(
    r'''posterior_theta = idata.posterior["theta"].values.ravel()        # 4,000 pooled draws
x_vals = np.linspace(0.35, 0.85, 300)
plt.figure(figsize=(9, 4.5))
plt.hist(posterior_theta, bins=40, density=True, alpha=0.5, color="teal", label=f"PyMC NUTS, mean {posterior_theta.mean():.4f}")
plt.plot(x_vals, stats.beta.pdf(x_vals, alpha_post, beta_post), color="darkred", lw=2.2, label=f"exact Beta(50, 34), mean {exact_mean:.4f}")
plt.axvline(exact_mean, color="darkred", ls="--", lw=1)
plt.title("PyMC NUTS reproduces the exact posterior")
plt.xlabel(r"$\theta$ (success probability)"); plt.ylabel("density"); plt.legend()
plt.show()
''',
    r'''posterior_theta = idata.posterior["theta"].values.ravel()        # 4000 сэмплов всех цепей
x_vals = np.linspace(0.35, 0.85, 300)
plt.figure(figsize=(9, 4.5))
plt.hist(posterior_theta, bins=40, density=True, alpha=0.5, color="teal", label=f"PyMC NUTS, среднее {posterior_theta.mean():.4f}")
plt.plot(x_vals, stats.beta.pdf(x_vals, alpha_post, beta_post), color="darkred", lw=2.2, label=f"точное Beta(50, 34), среднее {exact_mean:.4f}")
plt.axvline(exact_mean, color="darkred", ls="--", lw=1)
plt.title("PyMC NUTS воспроизводит точное апостериорное распределение")
plt.xlabel(r"$\theta$ (вероятность успеха)"); plt.ylabel("плотность"); plt.legend()
plt.show()
''',
)

# ---------------------------------------------------------------------------
md(
    r"""---
## 7. Practice and reflection

### Challenge: Monte Carlo error for a specific estimand
Compute the posterior probability that $\theta > 0.65$ from the PyMC draws,

$$p = \Pr(\theta > 0.65 \mid y) = \mathbb{E}[\mathbb{I}(\theta > 0.65) \mid y],$$

and its Monte Carlo standard error from the indicator variance and the tail ESS:

$$\operatorname{MCSE}(p) \approx \sqrt{\frac{p(1 - p)}{\text{ESS}_{\text{tail}}}}.$$
""",
    r"""---
## 7. Практика и размышление

### Задание: ошибка Монте-Карло для конкретной величины
Вычислите по сэмплам PyMC апостериорную вероятность того, что $\theta > 0.65$,

$$p = \Pr(\theta > 0.65 \mid y) = \mathbb{E}[\mathbb{I}(\theta > 0.65) \mid y],$$

и её стандартную ошибку Монте-Карло по дисперсии индикатора и tail ESS:

$$\operatorname{MCSE}(p) \approx \sqrt{\frac{p(1 - p)}{\text{ESS}_{\text{tail}}}}.$$
""",
)

code(
    """indicator_draws = (posterior_theta > 0.65).astype(float)
prob_est = indicator_draws.mean()
exact_prob = stats.beta.sf(0.65, alpha_post, beta_post)
mcse_prob = np.sqrt(prob_est * (1.0 - prob_est) / ess_tail)

print("Target: Pr(θ > 0.65 | y)")
print(f"Exact probability:  {exact_prob:.4f}")
print(f"PyMC estimate:      {prob_est:.4f} ± {mcse_prob:.4f} (MCSE)")
print(f"Absolute error:     {abs(prob_est - exact_prob):.4f}")
""",
    """indicator_draws = (posterior_theta > 0.65).astype(float)
prob_est = indicator_draws.mean()
exact_prob = stats.beta.sf(0.65, alpha_post, beta_post)
mcse_prob = np.sqrt(prob_est * (1.0 - prob_est) / ess_tail)

print("Цель: Pr(θ > 0.65 | y)")
print(f"Точная вероятность:   {exact_prob:.4f}")
print(f"Оценка PyMC:          {prob_est:.4f} ± {mcse_prob:.4f} (MCSE)")
print(f"Абсолютная ошибка:    {abs(prob_est - exact_prob):.4f}")
""",
)

md(
    r"""### Reflection questions

1. **Why does a high acceptance rate not imply efficient sampling?** In Section 3 the $\sigma = 0.01$ chain accepted almost every proposal, but its autocorrelation decayed very slowly and its ESS was small. Why?
2. **Why do we need four chains?** Can a single chain reveal multimodality or a convergence failure through $\widehat{R}$?
3. **What does one divergence mean?** A fit reports $\widehat{R} = 1.00$ and a large ESS, but 3 divergences. May you interpret the posterior?
""",
    r"""### Вопросы для размышления

1. **Почему высокая доля принятия не означает эффективного сэмплирования?** В разделе 3 цепь с $\sigma = 0.01$ принимала почти все предложения, но её автокорреляция затухала очень медленно, а ESS был мал. Почему?
2. **Зачем нужны четыре цепи?** Может ли одна цепь выявить многомодальность или отсутствие сходимости через $\widehat{R}$?
3. **Что означает хотя бы одно расхождение?** Подгонка сообщает $\widehat{R} = 1.00$ и большой ESS, но 3 расходящихся перехода. Можно ли интерпретировать апостериорное распределение?
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
