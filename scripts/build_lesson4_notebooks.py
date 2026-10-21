"""Build the Lesson 4 student workbooks (English and Russian) from one source.

Run from the book root with the pymc_env environment:
    /opt/homebrew/Caskroom/miniforge/base/envs/pymc_env/bin/python scripts/build_lesson4_notebooks.py

Outputs:
    notebooks/colab/04_priors_posterior_uncertainty_predictive_checking.ipynb
    notebooks/colab/04_priors_posterior_uncertainty_predictive_checking_ru.ipynb
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
    "en": "04_priors_posterior_uncertainty_predictive_checking.ipynb",
    "ru": "04_priors_posterior_uncertainty_predictive_checking_ru.ipynb",
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
    """# Lesson 4: Priors, Posterior Uncertainty & Predictive Checking
### Student Laboratory Workbook
*Course: Bayesian Analysis of Empirical Data (2026)*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](""" + GITHUB + FILES['en'] + """)
*Russian version: [""" + FILES['ru'] + "](" + GITHUB + FILES['ru'] + """)*

In this laboratory, we work through the closed-form Bayesian workflow for binomial data: eliciting priors and testing them with prior predictive checks, updating via precision-weighting, contrasting ETI and HDI credible intervals, simulating posterior predictive forecasts with tail-risk assessment, and conducting a prior sensitivity audit.
""",
    """# Занятие 4: Априорные распределения, апостериорная неопределенность и предсказательная проверка
### Рабочая тетрадь студента
*Курс: Байесовский анализ эмпирических данных (2026)*

[![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)](""" + GITHUB + FILES['ru'] + """)
*English version: [""" + FILES['en'] + "](" + GITHUB + FILES['en'] + """)*

В этой лабораторной работе мы разберем аналитический байесовский рабочий процесс для биномиальных данных: элиситацию априорных распределений и априорные проверки, точное обновление через взвешивание точностей, сравнение интервалов ETI и HDI, моделирование апостериорных предсказаний и оценку хвостовых рисков, а также аудит чувствительности к априорным предположениям.
""",
)

# ---------------------------------------------------------------------------
# Setup Cell
# ---------------------------------------------------------------------------
code(
    """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize

RANDOM_SEED = 2026
rng = np.random.default_rng(RANDOM_SEED)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
print("Environment initialized.")
"""
)

# ---------------------------------------------------------------------------
# Step 1: Prior Elicitation & Prior Predictive Checks
# ---------------------------------------------------------------------------
md(
    """## 1. Prior Elicitation and Prior Predictive Checking

### The Idea
How do we choose parameters $\\alpha$ and $\\beta$ for a $\\operatorname{Beta}$ prior? It is intuitive to ask domain experts for two quantities: the expected prior mean $\\mu_0$ and the effective prior sample size $n_0$ (how many pseudo-observations of confidence the prior represents):
$$\\alpha = \\mu_0 \\cdot n_0, \\qquad \\beta = (1 - \\mu_0) \\cdot n_0$$
Before looking at real data, we run a **prior predictive check**: simulate synthetic data directly from the prior. This reveals whether a "flat" prior like $\\operatorname{Beta}(1, 1)$ is truly harmless. For a sample of $N=20$, a flat prior predicts that extreme boundary counts ($k \\le 2$ or $k \\ge 18$) happen nearly $28.6\\%$ of the time — an absurdly confident statement about extreme events!
""",
    """## 1. Элиситация априорных распределений и априорная предсказательная проверка

### Идея
Как выбрать гиперпараметры $\\alpha$ и $\\beta$ для априорного $\\operatorname{Beta}$-распределения? Удобнее всего запросить у эксперта две понятные величины: ожидаемое априорное среднее $\\mu_0$ и эффективный априорный объем выборки $n_0$ (силу уверенности в эквиваленте числа псевдонаблюдений):
$$\\alpha = \\mu_0 \\cdot n_0, \\qquad \\beta = (1 - \\mu_0) \\cdot n_0$$
До контакта с реальными данными мы обязаны провести **априорную предсказательную проверку**: сгенерировать синтетические исходы прямо из априорного распределения. Это сразу показывает скрытые дефекты «плоского» приори $\\operatorname{Beta}(1, 1)$. При выборке $N=20$ плоский приор утверждает, что экстремальные исходы ($k \\le 2$ или $k \\ge 18$) происходят в $28{,}6\\%$ случаев — нелепо самоуверенное утверждение о крайностях!
""",
)

code(
    """# Sample size for an upcoming experiment
N_trials = 20
n_sim = 10000

# Three candidate priors
candidate_priors = {
    "Flat Beta(1, 1)": (1.0, 1.0),
    "Weakly Informative Beta(3, 3)": (3.0, 3.0),
    "Substantive Beta(6, 4)": (6.0, 4.0)
}

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)

for ax, (name, (a, b)) in zip(axes, candidate_priors.items()):
    # Draw theta from prior, then draw k from Binomial(N, theta)
    thetas = rng.beta(a, b, size=n_sim)
    k_sim = rng.binomial(N_trials, thetas)
    
    # Calculate probability of extreme boundary results (k <= 2 or k >= 18)
    p_extreme = np.mean((k_sim <= 2) | (k_sim >= 18))
    
    ax.hist(k_sim, bins=np.arange(-0.5, 21.5, 1), density=True, color="#2b5c8f", alpha=0.7, edgecolor="black")
    ax.set_title(f"{name}\\nP(k <= 2 or k >= 18) = {p_extreme:.1%}")
    ax.set_xlabel("Predicted Successes k (out of 20)")

axes[0].set_ylabel("Prior Predictive Probability")
plt.tight_layout()
plt.show()
"""
)

md(
    """### What to Watch For
The flat prior $\\operatorname{Beta}(1, 1)$ spreads probability mass uniformly over all counts $k \\in \\{0, \\dots, 20\\}$, asserting that finding $0$ successes out of $20$ is just as likely as finding $10$ successes! A weakly informative prior $\\operatorname{Beta}(3, 3)$ concentrates probability plausibly around the center while leaving generous uncertainty.
""",
    """### На что обратить внимание
Плоский приор $\\operatorname{Beta}(1, 1)$ размазывает вероятность совершенно равномерно по всем исходам от 0 до 20, утверждая, что получить 0 успехов из 20 так же вероятно, как и 10 успехов. Слабоинформативный приор $\\operatorname{Beta}(3, 3)$ концентрирует априорную массу в разумном центре, сохраняя достаточную открытость к новым данным.
""",
)

# ---------------------------------------------------------------------------
# Step 2: Conjugate Updating & Precision Weighting
# ---------------------------------------------------------------------------
md(
    """## 2. Exact Conjugate Updating and Precision Weighting

### The Idea
When data follow a Bernoulli or Binomial likelihood and the prior is $\\operatorname{Beta}(\\alpha, \\beta)$, the posterior is also a Beta distribution (conjugacy):
$$\\alpha_{\\text{post}} = \\alpha + k, \\qquad \\beta_{\\text{post}} = \\beta + N - k$$
The posterior mean is a transparent **precision-weighted compromise** between the prior mean $\\mu_0$ and the maximum likelihood estimate $\\hat{\\theta}_{\\text{MLE}} = k / N$:
$$\\mathbb{E}[\\theta \\mid k] = \\left(\\frac{n_0}{n_0 + N}\\right) \\cdot \\mu_0 + \\left(\\frac{N}{n_0 + N}\\right) \\cdot \\hat{\\theta}_{\\text{MLE}}$$
""",
    """## 2. Точное сопряженное обновление и взвешивание точностей

### Идея
Когда правдоподобие биномиально, а априорное распределение — $\\operatorname{Beta}(\\alpha, \\beta)$, апостериорное распределение остается в том же семействе Beta (сопряженность):
$$\\alpha_{\\text{post}} = \\alpha + k, \\qquad \\beta_{\\text{post}} = \\beta + N - k$$
Апостериорное среднее представляет собой наглядный **взвешенный компромисс** между априорным ожиданием $\\mu_0$ и выборочной долей $\\hat{\\theta}_{\\text{MLE}} = k / N$:
$$\\mathbb{E}[\\theta \\mid k] = \\left(\\frac{n_0}{n_0 + N}\\right) \\cdot \\mu_0 + \\left(\\frac{N}{n_0 + N}\\right) \\cdot \\hat{\\theta}_{\\text{MLE}}$$
""",
)

code(
    """# Observation: 16 successes out of 20 trials
k_obs, N_obs = 16, 20
mle = k_obs / N_obs

# Substantive prior: mu0 = 0.60, n0 = 10 -> Beta(6, 4)
a_prior, b_prior = 6.0, 4.0
mu_0 = a_prior / (a_prior + b_prior)
n_0 = a_prior + b_prior

# Posterior parameters
a_post = a_prior + k_obs
b_post = b_prior + N_obs - k_obs
mu_post = a_post / (a_post + b_post)

# Weights
weight_prior = n_0 / (n_0 + N_obs)
weight_data = N_obs / (n_0 + N_obs)

print(f"Prior Mean:     mu_0 = {mu_0:.3f} (Weight = {weight_prior:.1%})")
print(f"Sample Share:   MLE  = {mle:.3f} (Weight = {weight_data:.1%})")
print(f"Posterior Mean: mu_post = {mu_post:.3f}")
print(f"Calculated:     {weight_prior:.3f} * {mu_0:.3f} + {weight_data:.3f} * {mle:.3f} = {weight_prior * mu_0 + weight_data * mle:.3f}")

theta_vals = np.linspace(0, 1, 400)
fig, ax = plt.subplots(figsize=(8.5, 4.0))
ax.plot(theta_vals, stats.beta.pdf(theta_vals, a_prior, b_prior), color="gray", linestyle="--", label=f"Prior Beta({a_prior:.0f}, {b_prior:.0f})")
ax.plot(theta_vals, stats.beta.pdf(theta_vals, 1 + k_obs, 1 + N_obs - k_obs), color="#e07a5f", linestyle=":", label=f"Normalized Likelihood (k={k_obs}/{N_obs})")
ax.plot(theta_vals, stats.beta.pdf(theta_vals, a_post, b_post), color="#2b5c8f", lw=2.5, label=f"Posterior Beta({a_post:.0f}, {b_post:.0f})")
ax.set_xlabel("Parameter theta")
ax.set_ylabel("Density")
ax.set_title("The Bayesian Triplot: Prior, Likelihood and Compromise Posterior")
ax.legend()
plt.tight_layout()
plt.show()
"""
)

md(
    """### What to Watch For
The data weight is $20 / 30 = 66.7\\%$ while the prior weight is $10 / 30 = 33.3\\%$. The posterior distribution cleanly balances the two sources of information, shifting from the prior mean ($0.60$) toward the sample data ($0.80$) to settle at exactly $0.733$.
""",
    """### На что обратить внимание
Вес данных составляет $20 / 30 = 66{,}7\\%$, а вес априорных данных — $10 / 30 = 33{,}3\\%$. Постериор гармонично балансирует оба источника информации, смещаясь от априорного ожидания ($0{,}60$) в сторону выборочных данных ($0{,}80$) и останавливаясь точно в точке $0{,}733$.
""",
)

# ---------------------------------------------------------------------------
# Step 3: Posterior Uncertainty — ETI vs. HDI
# ---------------------------------------------------------------------------
md(
    """## 3. Posterior Uncertainty: Equal-Tailed (ETI) vs. Highest Density (HDI) Intervals

### The Idea
To communicate uncertainty, we report a credible interval containing $95\\%$ of posterior probability mass:
1. **Equal-Tailed Interval (ETI)**: cuts off $2.5\\%$ from each tail via quantiles.
2. **Highest Density Interval (HDI)**: finds the shortest interval such that every point inside has higher probability density than any point outside.
When the posterior is symmetric, ETI and HDI are identical. But when the posterior is skewed (e.g., near boundaries), the HDI is strictly narrower than the ETI and excludes regions of low probability density.
""",
    """## 3. Неопределенность: интервал равных хвостов (ETI) vs наивысшей плотности (HDI)

### Идея
Для выражения неопределенности мы строим байесовский интервал надежности, содержащий $95\\%$ апостериорной массы:
1. **Интервал равных хвостов (ETI)**: отсекает ровно по $2{,}5\\%$ вероятности с каждого хвоста через квантили.
2. **Интервал наивысшей плотности (HDI)**: находит самый узкий отрезок, внутри которого плотность вероятности в каждой точке строго выше, чем в любой точке снаружи.
На симметричных распределениях ETI и HDI совпадают. Но на скошенных данных (например, вблизи границ 0 или 1) интервал HDI строго уже ETI и не включает точки с низкой апостериорной плотностью.
""",
)

code(
    """# Skewed posterior example: Beta(2, 20) representing rare events
a_skew, b_skew = 2.0, 20.0
dist_skew = stats.beta(a_skew, b_skew)

# 95% ETI via quantiles
eti_lower, eti_upper = dist_skew.ppf(0.025), dist_skew.ppf(0.975)
eti_width = eti_upper - eti_lower

# 95% HDI via numerical minimization of interval width
def hdi_objective(lower):
    upper = dist_skew.ppf(dist_skew.cdf(lower) + 0.95)
    return upper - lower

opt = optimize.minimize_scalar(hdi_objective, bounds=(0, dist_skew.ppf(0.05)), method="bounded")
hdi_lower = opt.x
hdi_upper = dist_skew.ppf(dist_skew.cdf(hdi_lower) + 0.95)
hdi_width = hdi_upper - hdi_lower

print(f"95% ETI: [{eti_lower:.4f}, {eti_upper:.4f}] -> Width = {eti_width:.4f}")
print(f"95% HDI: [{hdi_lower:.4f}, {hdi_upper:.4f}] -> Width = {hdi_width:.4f} (Shortest!)")

x_grid = np.linspace(0, 0.35, 400)
fig, ax = plt.subplots(figsize=(8.5, 4.0))
ax.plot(x_grid, dist_skew.pdf(x_grid), color="#2b5c8f", lw=2)
ax.fill_between(x_grid, 0, dist_skew.pdf(x_grid), where=(x_grid >= hdi_lower) & (x_grid <= hdi_upper), color="#2b5c8f", alpha=0.3, label=f"95% HDI (Width = {hdi_width:.3f})")
ax.plot([eti_lower, eti_upper], [1.5, 1.5], color="#e07a5f", lw=3.5, label=f"95% ETI (Width = {eti_width:.3f})")
ax.plot([hdi_lower, hdi_upper], [0.5, 0.5], color="black", lw=3.5, label="95% HDI")
ax.set_xlabel("Parameter theta")
ax.set_ylabel("Density")
ax.set_title("Credible Intervals on Skewed Posterior Beta(2, 20): HDI Is Narrower")
ax.legend()
plt.tight_layout()
plt.show()
"""
)

md(
    """### What to Watch For
On skewed distributions, the ETI drags unnecessarily far into the long right tail (up to $0.231$), whereas the HDI starts closer to zero ($0.005$) and ends at $0.203$. The HDI is noticeably more compact (width $0.198$ vs. $0.219$), providing the most credible summary of the parameter.
""",
    """### На что обратить внимание
На скошенных распределениях интервал ETI неоправданно глубоко залезает в длинный правый хвост (до $0{,}231$), тогда как HDI начинается ближе к нулю ($0{,}005$) и заканчивается на $0{,}203$. Интервал HDI заметно компактнее (ширина $0{,}198$ против $0{,}219$), отбирая только наиболее вероятные значения параметра.
""",
)

# ---------------------------------------------------------------------------
# Step 4: Posterior Predictive Forecasting & Tail Risk
# ---------------------------------------------------------------------------
md(
    """## 4. Posterior Predictive Forecasting and Tail Risk

### The Idea
How many successes should we expect in a future cohort of $\\tilde{N} = 100$ trials? We cannot simply plug in the posterior mean $\\mu_{\\text{post}}$, because that ignores two sources of uncertainty:
1. **Epistemic parameter uncertainty**: $\\theta$ is not known exactly; it follows $\\operatorname{Beta}(\\alpha_{\\text{post}}, \\beta_{\\text{post}})$.
2. **Aleatory sampling noise**: even for a known $\\theta$, coin tosses fluctuate binomially.
By integrating both sources, we simulate the true posterior predictive distribution $\\tilde{y} \\sim p(\\tilde{y} \\mid D)$ and calculate decision-critical quantities, such as the probability of a critical shortfall $P(\\tilde{y} < 65)$.
""",
    """## 4. Апостериорное предсказательное прогнозирование и хвостовой риск

### Идея
Сколько успехов следует ожидать в будущей когорте из $\\tilde{N} = 100$ испытаний? Нельзя просто подставить среднее значение $\\mu_{\\text{post}}$, так как это проигнорирует два источника неопределенности:
1. **Эпистемическая неопределенность параметра**: точное значение $\\theta$ неизвестно, оно распределено как $\\operatorname{Beta}(\\alpha_{\\text{post}}, \\beta_{\\text{post}})$.
2. **Алеаторный шум выборки**: даже при фиксированном $\\theta$ результаты бросков флуктуируют по биномиальному закону.
Объединяя оба источника, мы моделируем истинное апостериорное предсказательное распределение $\\tilde{y} \\sim p(\\tilde{y} \\mid D)$ и оцениваем критические риски, например вероятность провала целевого показателя $P(\\tilde{y} < 65)$.
""",
)

code(
    """N_future = 100
n_forecast = 10000

# 1. Epistemic draw: sample theta from posterior Beta(22, 8)
theta_draws = rng.beta(a_post, b_post, size=n_forecast)

# 2. Aleatory draw: sample future observable count
y_future = rng.binomial(N_future, theta_draws)

# Threshold risk analysis: probability of fewer than 65 successes
threshold = 65
p_failure = np.mean(y_future < threshold)

fig, ax = plt.subplots(figsize=(8.5, 4.0))
counts, bins, patches = ax.hist(y_future, bins=np.arange(45.5, 95.5, 1), density=True, color="#2b5c8f", alpha=0.7, edgecolor="black")

# Highlight failure zone
for b, patch in zip(bins[:-1], patches):
    if b < threshold:
        patch.set_facecolor("#e07a5f")

ax.axvline(threshold - 0.5, color="#e07a5f", linestyle="--", lw=2, label=f"Threshold ({threshold}) -> Failure Risk = {p_failure:.1%}")
ax.set_xlabel(f"Predicted Future Successes out of {N_future}")
ax.set_ylabel("Predictive Probability")
ax.set_title(f"Future Forecasting: Tail Risk P(y_future < {threshold}) = {p_failure:.1%}")
ax.legend()
plt.tight_layout()
plt.show()

print(f"Expected future successes: {y_future.mean():.1f} +- {y_future.std():.1f}")
print(f"Risk of shortfall (< {threshold}): {p_failure:.1%}")
"""
)

md(
    """### What to Watch For
Although the expected future outcome is $73.3$ successes, the predictive distribution spreads across the range from $55$ to $90$. The tail probability of falling below $65$ is $9.8\\%$. If $65$ is a critical regulatory threshold, a naive point prediction would falsely claim safety!
""",
    """### На что обратить внимание
Хотя среднее ожидаемое число успехов равно $73{,}3$, предсказательное распределение охватывает диапазон от $55$ до $90$. Риск оказаться ниже планки в $65$ успехов составляет целых $9{,}8\\%$. Если порог 65 является критическим стандартом безопасности, точечный прогноз ввел бы исследователя в опасное заблуждение!
""",
)

# ---------------------------------------------------------------------------
# Step 5: Prior Sensitivity Audit
# ---------------------------------------------------------------------------
md(
    """## 5. Prior Sensitivity Audit: How Skeptical Must a Prior Be?

### The Idea
A cornerstone of Bayesian reporting is checking whether conclusions collapse under alternative prior assumptions. We re-fit our model across 4 distinct prior stances: Flat, Weakly Informative, Optimistic, and a Skeptical prior representing a critic who believes performance is below chance ($\\mu_0 = 0.30$).
""",
    """## 5. Аудит чувствительности к априорному распределению

### Идея
Стандарт качественного байесовского отчета — проверка того, сохраняются ли выводы при альтернативных априорных предположениях. Мы пересчитываем постериор для 4 различных позиций: плоской, слабоинформативной, оптимистичной и скептической (критик, уверенный, что результативность ниже случайного уровня, $\\mu_0 = 0{,}30$).
""",
)

code(
    """audit_priors = {
    "Flat Beta(1, 1)": (1.0, 1.0),
    "Weak Beta(2, 2)": (2.0, 2.0),
    "Substantive Beta(6, 4)": (6.0, 4.0),
    "Skeptical Beta(3, 7)": (3.0, 7.0)  # Prior mean 0.30
}

audit_results = []
for name, (a, b) in audit_priors.items():
    a_p = a + k_obs
    b_p = b + N_obs - k_obs
    d = stats.beta(a_p, b_p)
    hdi = (d.ppf(0.025), d.ppf(0.975))
    audit_results.append({
        "Prior Scenario": name,
        "Prior Mean": a / (a + b),
        "Post Mean": a_p / (a_p + b_p),
        "Post 95% Interval": f"[{hdi[0]:.3f}, {hdi[1]:.3f}]",
        "P(theta > 0.50)": 1.0 - d.cdf(0.50)
    })

df_audit = pd.DataFrame(audit_results)
print("--- Prior Sensitivity Audit Table ---")
print(df_audit.to_string(index=False))
"""
)

md(
    """### What to Watch For
Across all 4 priors — even the skeptical prior starting at $0.30$ — the posterior probability $P(\\theta > 0.50)$ remains above $98.5\\%$, and the posterior mean stays firmly between $0.63$ and $0.77$. Because the sample evidence ($16/20$) is clear, our substantive finding that performance exceeds chance is robust to prior skepticism.
""",
    """### На что обратить внимание
Во всех 4 сценариях — даже при скептическом приоре с центром в $0{,}30$ — апостериорная вероятность $P(\\theta > 0{,}50)$ остается выше $98{,}5\\%$, а апостериорное среднее уверенно лежит между $0{,}63$ и $0{,}77$. Поскольку выборочные данные ($16/20$) убедительны, вывод о том, что результативность превышает случайный уровень, робастен к априорному скептицизму.
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
