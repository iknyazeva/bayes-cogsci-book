"""Build the Lesson 3 student workbooks (English and Russian) from one source.

Run from the book root with the pymc_env environment:

    conda run -n pymc_env python scripts/build_lesson3_notebooks.py

Outputs:
    notebooks/colab/03_generative_models_likelihood_prediction.ipynb
    notebooks/colab/03_generative_models_likelihood_prediction_ru.ipynb

Every cell is defined once with an English and a Russian variant so the two
workbooks stay structurally identical (same cell order, same code logic).
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks" / "colab"
GITHUB = "https://colab.research.google.com/github/iknyazeva/bayes-cogsci-book/blob/main/notebooks/colab/"
FILES = {
    "en": "03_generative_models_likelihood_prediction.ipynb",
    "ru": "03_generative_models_likelihood_prediction_ru.ipynb",
}

# Each entry: (cell_type, {"en": text, "ru": text}); when "ru" is missing the
# English text (typically pure code) is reused.
CELLS: list[tuple[str, dict[str, str]]] = []


def md(en: str, ru: str | None = None) -> None:
    CELLS.append(("markdown", {"en": en, "ru": ru if ru is not None else en}))


def code(en: str, ru: str | None = None) -> None:
    CELLS.append(("code", {"en": en, "ru": ru if ru is not None else en}))


# ---------------------------------------------------------------------------
# 0. Title
# ---------------------------------------------------------------------------
md(
    f"""# Lesson 3: Generative Models, Likelihood and Prediction
### Student Laboratory Workbook
*Course: Bayesian Analysis of Empirical Data (2026)*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({GITHUB}{FILES['en']})

Russian version: [{FILES['ru']}]({GITHUB}{FILES['ru']})
""",
    f"""# Занятие 3: Генеративные модели, правдоподобие и предсказание
### Рабочая тетрадь студента
*Курс: Байесовский анализ эмпирических данных (2026)*

[![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)]({GITHUB}{FILES['ru']})

English version: [{FILES['en']}]({GITHUB}{FILES['en']})
""",
)

# ---------------------------------------------------------------------------
# 1. Before class
# ---------------------------------------------------------------------------
md(
    """## 1. Before class
* **One-minute goal**: A Bayesian model is a generative hypothesis. We run it forward to simulate data, read the same model backwards as a likelihood, and distinguish variation in the data from uncertainty about the parameter.
* **Prerequisites**: Distinguish a parameter from observed data (Lesson 2).
* **Expected runtime**: ~90 minutes.
""",
    """## 1. Перед занятием
* **Цель в одну минуту**: Байесовская модель — это генеративная гипотеза. Мы запускаем её «вперёд», чтобы симулировать данные, читаем ту же модель «назад» как функцию правдоподобия, и отличаем изменчивость данных от неопределённости относительно параметра.
* **Что нужно знать**: отличать параметр от наблюдаемых данных (занятие 2).
* **Ожидаемое время**: ~90 минут.
""",
)

# ---------------------------------------------------------------------------
# 2. Make your copy
# ---------------------------------------------------------------------------
md(
    """## 2. Make your copy
To save your progress, click **File $\\to$ Save a copy in Drive**. Alternatively, download the notebook as a `.ipynb` file to run locally.
""",
    """## 2. Сделайте свою копию
Чтобы сохранять изменения, нажмите **Файл $\\to$ Сохранить копию на Диске** (File → Save a copy in Drive). Можно также скачать блокнот как файл `.ipynb` и запускать локально.
""",
)

# ---------------------------------------------------------------------------
# 3. Environment check
# ---------------------------------------------------------------------------
md(
    """## 3. Environment check
Initialize the scientific computing libraries and configure Plotly for Google Colab.
""",
    """## 3. Проверка окружения
Импортируем библиотеки и настраиваем Plotly для Google Colab.
""",
)
code(
    """import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import trapezoid
import plotly.graph_objects as go
from plotly.subplots import make_subplots

IS_COLAB = "google.colab" in sys.modules

if IS_COLAB:
    print("⚡ Running in Google Colab environment.")
    import plotly.io as pio
    pio.renderers.default = "colab"
else:
    print("💻 Running in local environment.")

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
print(f"✅ Environment initialized. NumPy random seed set to {RANDOM_SEED}.")
""",
    """import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import trapezoid
import plotly.graph_objects as go
from plotly.subplots import make_subplots

IS_COLAB = "google.colab" in sys.modules

if IS_COLAB:
    print("⚡ Работаем в Google Colab.")
    import plotly.io as pio
    pio.renderers.default = "colab"
else:
    print("💻 Работаем в локальном окружении.")

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
print(f"✅ Окружение готово. Зерно генератора NumPy: {RANDOM_SEED}.")
""",
)

# ---------------------------------------------------------------------------
# 4. Data source and ethics
# ---------------------------------------------------------------------------
md(
    """## 4. Data source and ethics
* **Provenance**: All data are synthetic and generated with NumPy inside this workbook. No real survey is being reported; the numbers illustrate how a model behaves.
* **Licence/Status**: Open / Teaching Extract.
* **Unit of observation**: a simulated citizen answering a survey.
* **Privacy note**: No personal data of any kind, and nothing is downloaded.
""",
    """## 4. Источник данных и этика
* **Происхождение**: все данные синтетические и генерируются NumPy внутри этой тетради. Никакой реальный опрос здесь не описывается; числа иллюстрируют поведение модели.
* **Лицензия/статус**: открытые учебные материалы.
* **Единица наблюдения**: симулированный респондент опроса.
* **Приватность**: персональных данных нет, ничего не скачивается.
""",
)

# ---------------------------------------------------------------------------
# 5. Learning targets
# ---------------------------------------------------------------------------
md(
    """## 5. Learning targets
By completing this workbook, you will be able to:
1. Describe and run a simple generative model using supplied code.
2. Distinguish variability at a fixed parameter from uncertainty about that parameter.
3. Identify implausible simulated observations and relate them to a model assumption.
4. Separate prior predictive from posterior predictive simulation.
""",
    """## 5. Учебные цели
Выполнив эту тетрадь, вы сможете:
1. Описать и запустить простую генеративную модель по готовому коду.
2. Отличать изменчивость данных при фиксированном параметре от неопределённости относительно самого параметра.
3. Находить неправдоподобные симулированные наблюдения и связывать их с допущением модели.
4. Различать априорное и апостериорное предсказательное моделирование.
""",
)

# ---------------------------------------------------------------------------
# 6. Classwork 0: predict before running
# ---------------------------------------------------------------------------
md(
    """## 6. Classwork 0: predict before running
> ✍ **WRITE**:
> **A. Survey.** An opinion poll of $N = 20$ citizens yields $k = 13$ supporters of a policy.
> 1. Which value of the support parameter $\\theta \\in [0, 1]$ makes the observed data most likely?
> 2. Will the likelihood function $\\mathcal{L}(\\theta \\mid k=13)$ integrate to $1.0$ over $\\theta \\in [0, 1]$? Why or why not?
>
> **B. Prediction.** Before any calculation:
> 3. If we did not fix $\\theta$ but drew it from a prior first, would the simulated counts be more or less spread out than with $\\theta$ fixed at 0.65?
> 4. After observing $k = 13$, would a prediction for the *next* survey be narrower or wider than the posterior for $\\theta$ itself?
""",
    """## 6. Классная работа 0: предскажите до запуска
> ✍ **НАПИШИТЕ**:
> **A. Опрос.** В опросе $N = 20$ человек $k = 13$ поддержали некоторую меру.
> 1. При каком значении параметра поддержки $\\theta \\in [0, 1]$ наблюдаемые данные наиболее вероятны?
> 2. Будет ли интеграл функции правдоподобия $\\mathcal{L}(\\theta \\mid k=13)$ по $\\theta \\in [0, 1]$ равен $1{,}0$? Почему?
>
> **B. Предсказание.** До всяких вычислений:
> 3. Если бы мы не фиксировали $\\theta$, а сначала извлекали его из априорного распределения, симулированные числа были бы разбросаны сильнее или слабее, чем при фиксированном $\\theta = 0{,}65$?
> 4. После наблюдения $k = 13$ предсказание для *следующего* опроса будет уже или шире, чем апостериорное распределение самого $\\theta$?
""",
)

# ---------------------------------------------------------------------------
# 7. Classwork 1: reproduce
# ---------------------------------------------------------------------------
md(
    """## 7. Classwork 1: reproduce
### A. Worked example: simulating survey samples from a known model
▶ **RUN TOGETHER**: We begin in the generative world. Assume the true support for a policy is $\\theta_0 = 0.65$. We simulate $S = 1{,}000$ independent survey organizations, each polling $N = 20$ voters.
""",
    """## 7. Классная работа 1: воспроизведите
### A. Разобранный пример: симуляция опросов из известной модели
▶ **ЗАПУСКАЕМ ВМЕСТЕ**: начинаем в «генеративном мире». Пусть истинная доля поддержки $\\theta_0 = 0{,}65$. Симулируем $S = 1000$ независимых опросных организаций, каждая опрашивает $N = 20$ человек.
""",
)
code(
    """N_trials = 20
theta_true = 0.65
S_sims = 1000

# Forward simulation: draws from Binomial(N=20, theta=0.65)
simulated_k = rng.binomial(n=N_trials, p=theta_true, size=S_sims)

print(f"Simulated {S_sims} survey replications:")
print(f"  Theoretical mean:   {N_trials * theta_true:.2f}")
print(f"  Empirical mean:     {simulated_k.mean():.2f}")
print(f"  Theoretical SD:     {np.sqrt(N_trials * theta_true * (1 - theta_true)):.2f}")
print(f"  Empirical SD:       {simulated_k.std():.2f}")

counts, bins = np.histogram(simulated_k, bins=np.arange(-0.5, N_trials + 1.5, 1))
fig_sim = go.Figure(go.Bar(
    x=np.arange(0, N_trials + 1),
    y=counts / S_sims,
    marker_color='#2b6cb0',
    hovertemplate='<b>k = %{x}</b><br>Simulated frequency: %{y:.3f}<extra></extra>'
))
fig_sim.update_layout(
    title=f'Forward simulation: Binomial(N={N_trials}, θ={theta_true}) across {S_sims} replications',
    xaxis_title='Simulated supporters k (out of 20)',
    yaxis_title='Relative frequency',
    template='plotly_white',
    height=380
)
fig_sim.show()
""",
    """N_trials = 20
theta_true = 0.65
S_sims = 1000

# Прямая симуляция: выборки из Binomial(N=20, theta=0.65)
simulated_k = rng.binomial(n=N_trials, p=theta_true, size=S_sims)

print(f"Симулировано {S_sims} повторений опроса:")
print(f"  Теоретическое среднее: {N_trials * theta_true:.2f}")
print(f"  Эмпирическое среднее:  {simulated_k.mean():.2f}")
print(f"  Теоретическое SD:      {np.sqrt(N_trials * theta_true * (1 - theta_true)):.2f}")
print(f"  Эмпирическое SD:       {simulated_k.std():.2f}")

counts, bins = np.histogram(simulated_k, bins=np.arange(-0.5, N_trials + 1.5, 1))
fig_sim = go.Figure(go.Bar(
    x=np.arange(0, N_trials + 1),
    y=counts / S_sims,
    marker_color='#2b6cb0',
    hovertemplate='<b>k = %{x}</b><br>Частота в симуляции: %{y:.3f}<extra></extra>'
))
fig_sim.update_layout(
    title=f'Прямая симуляция: Binomial(N={N_trials}, θ={theta_true}), {S_sims} повторений',
    xaxis_title='Число сторонников k (из 20)',
    yaxis_title='Относительная частота',
    template='plotly_white',
    height=380
)
fig_sim.show()
""",
)

# ---------------------------------------------------------------------------
# 8. STOP 1
# ---------------------------------------------------------------------------
md(
    """## 8. STOP 1
### 🛑 STOP: Synchronization checkpoint
Discuss: How does the distribution of the sample proportion $\\hat{p} = k/N$ change as the sample size increases? Will every simulated survey contain exactly 13 supporters?
""",
    """## 8. СТОП 1
### 🛑 СТОП: точка синхронизации
Обсудите: как меняется распределение выборочной доли $\\hat{p} = k/N$ с ростом объёма выборки? Будет ли в каждом симулированном опросе ровно 13 сторонников?
""",
)

# ---------------------------------------------------------------------------
# 9. Classwork 2: modify
# ---------------------------------------------------------------------------
md(
    """## 9. Classwork 2: modify
### B. Evaluating the likelihood: data fixed, parameter varying
🧪 **CHANGE ONE THING**: First change the sample size in the forward simulation, then hold the observed data fixed and let the parameter vary.
""",
    """## 9. Классная работа 2: измените
### B. Правдоподобие: данные фиксированы, параметр меняется
🧪 **ИЗМЕНИТЕ ОДНО**: сначала измените объём выборки в прямой симуляции, затем зафиксируйте наблюдаемые данные и дайте меняться параметру.
""",
)
code(
    """# 🧪 STUDENT SANDBOX: forward simulation
# Modify the parameters below and run to test your hypotheses.

sandbox_N = 200        # Try: 20, 50, 200, 1000
sandbox_theta = 0.65   # Try: 0.04 (rare events), 0.50 (fair), 0.65 (majority)
sandbox_sims = 1000

sandbox_k = rng.binomial(n=sandbox_N, p=sandbox_theta, size=sandbox_sims)
sandbox_prop = sandbox_k / sandbox_N

print(f"Results for N={sandbox_N}, θ={sandbox_theta}:")
print(f"  Mean proportion: {sandbox_prop.mean():.4f} (true: {sandbox_theta:.4f})")
print(f"  SD proportion:   {sandbox_prop.std():.4f} (theoretical SE: {np.sqrt(sandbox_theta*(1-sandbox_theta)/sandbox_N):.4f})")
print(f"  Min k: {sandbox_k.min()}, Max k: {sandbox_k.max()}")
""",
    """# 🧪 ПЕСОЧНИЦА: прямая симуляция
# Измените параметры ниже и запустите, чтобы проверить свои гипотезы.

sandbox_N = 200        # Попробуйте: 20, 50, 200, 1000
sandbox_theta = 0.65   # Попробуйте: 0.04 (редкие события), 0.50 (поровну), 0.65 (большинство)
sandbox_sims = 1000

sandbox_k = rng.binomial(n=sandbox_N, p=sandbox_theta, size=sandbox_sims)
sandbox_prop = sandbox_k / sandbox_N

print(f"Результаты для N={sandbox_N}, θ={sandbox_theta}:")
print(f"  Средняя доля: {sandbox_prop.mean():.4f} (истинная: {sandbox_theta:.4f})")
print(f"  SD доли:      {sandbox_prop.std():.4f} (теоретическая SE: {np.sqrt(sandbox_theta*(1-sandbox_theta)/sandbox_N):.4f})")
print(f"  Min k: {sandbox_k.min()}, Max k: {sandbox_k.max()}")
""",
)
code(
    """k_obs = 13
N_obs = 20
theta_grid = np.linspace(0.001, 0.999, 500)

# Likelihood: L(theta | k=13, N=20) = binom.pmf(13, 20, theta)
likelihood = stats.binom.pmf(k_obs, N_obs, theta_grid)

mle_idx = np.argmax(likelihood)
mle_theta = theta_grid[mle_idx]

# Two competing candidate values
h1_val = 0.65
h2_val = 0.50
l_h1 = stats.binom.pmf(k_obs, N_obs, h1_val)
l_h2 = stats.binom.pmf(k_obs, N_obs, h2_val)
lr_h1_h2 = l_h1 / l_h2

fig_lik = go.Figure()
fig_lik.add_trace(go.Scatter(
    x=theta_grid, y=likelihood, mode='lines',
    line=dict(color='#d97706', width=2.5), fill='tozeroy', fillcolor='rgba(217, 119, 6, 0.12)',
    name=f'Likelihood L(θ | k={k_obs})'
))
fig_lik.add_trace(go.Scatter(
    x=[h2_val, h1_val], y=[l_h2, l_h1], mode='markers+text',
    text=[f'H2: θ={h2_val}<br>(L={l_h2:.4f})', f'H1: θ={h1_val}<br>(L={l_h1:.4f})'],
    textposition=['bottom left', 'top right'],
    marker=dict(size=10, color=['#c53030', '#276749']), name='Candidates'
))
fig_lik.update_layout(
    title=f'Likelihood L(θ | k={k_obs}, N={N_obs})   [likelihood ratio H1/H2 = {lr_h1_h2:.2f}]',
    xaxis_title='Candidate parameter θ (support probability)',
    yaxis_title='Likelihood L(θ)',
    template='plotly_white', height=420
)
fig_lik.show()

print(f"Maximum likelihood estimate: θ_hat = {mle_theta:.3f} (sample proportion {k_obs}/{N_obs} = {k_obs/N_obs:.3f})")
print(f"Likelihood at H1 (θ={h1_val}): {l_h1:.4f}")
print(f"Likelihood at H2 (θ={h2_val}): {l_h2:.4f}")
print(f"Likelihood ratio: {lr_h1_h2:.3f} -> the data are {lr_h1_h2:.2f}x more compatible with H1 than with H2.")
""",
    """k_obs = 13
N_obs = 20
theta_grid = np.linspace(0.001, 0.999, 500)

# Правдоподобие: L(theta | k=13, N=20) = binom.pmf(13, 20, theta)
likelihood = stats.binom.pmf(k_obs, N_obs, theta_grid)

mle_idx = np.argmax(likelihood)
mle_theta = theta_grid[mle_idx]

# Два конкурирующих значения параметра
h1_val = 0.65
h2_val = 0.50
l_h1 = stats.binom.pmf(k_obs, N_obs, h1_val)
l_h2 = stats.binom.pmf(k_obs, N_obs, h2_val)
lr_h1_h2 = l_h1 / l_h2

fig_lik = go.Figure()
fig_lik.add_trace(go.Scatter(
    x=theta_grid, y=likelihood, mode='lines',
    line=dict(color='#d97706', width=2.5), fill='tozeroy', fillcolor='rgba(217, 119, 6, 0.12)',
    name=f'Правдоподобие L(θ | k={k_obs})'
))
fig_lik.add_trace(go.Scatter(
    x=[h2_val, h1_val], y=[l_h2, l_h1], mode='markers+text',
    text=[f'H2: θ={h2_val}<br>(L={l_h2:.4f})', f'H1: θ={h1_val}<br>(L={l_h1:.4f})'],
    textposition=['bottom left', 'top right'],
    marker=dict(size=10, color=['#c53030', '#276749']), name='Кандидаты'
))
fig_lik.update_layout(
    title=f'Правдоподобие L(θ | k={k_obs}, N={N_obs})   [отношение правдоподобий H1/H2 = {lr_h1_h2:.2f}]',
    xaxis_title='Кандидат θ (вероятность поддержки)',
    yaxis_title='Правдоподобие L(θ)',
    template='plotly_white', height=420
)
fig_lik.show()

print(f"Оценка максимального правдоподобия: θ_hat = {mle_theta:.3f} (выборочная доля {k_obs}/{N_obs} = {k_obs/N_obs:.3f})")
print(f"Правдоподобие при H1 (θ={h1_val}): {l_h1:.4f}")
print(f"Правдоподобие при H2 (θ={h2_val}): {l_h2:.4f}")
print(f"Отношение правдоподобий: {lr_h1_h2:.3f} -> данные в {lr_h1_h2:.2f} раза лучше согласуются с H1, чем с H2.")
""",
)
code(
    """# 🧪 STUDENT SANDBOX: your own likelihood scenarios
k_sandbox = 65       # Try: 65 (large sample), 0 (zero events), 7
N_sandbox = 100      # Try: 100, 20, 10
hypo_1 = 0.65        # Try: 0.65, 0.80
hypo_2 = 0.50        # Try: 0.50, 0.40

lik_curve = stats.binom.pmf(k_sandbox, N_sandbox, theta_grid)
l_h1_sb = stats.binom.pmf(k_sandbox, N_sandbox, hypo_1)
l_h2_sb = stats.binom.pmf(k_sandbox, N_sandbox, hypo_2)
lr_sb = l_h1_sb / l_h2_sb if l_h2_sb > 0 else np.inf

print(f"Sandbox experiment (k={k_sandbox}/{N_sandbox}):")
print(f"  MLE:                    {theta_grid[np.argmax(lik_curve)]:.3f}")
print(f"  Likelihood at H1:       {l_h1_sb:.6e}")
print(f"  Likelihood at H2:       {l_h2_sb:.6e}")
print(f"  Likelihood ratio H1/H2: {lr_sb:.2f}x")
""",
    """# 🧪 ПЕСОЧНИЦА: свои сценарии для правдоподобия
k_sandbox = 65       # Попробуйте: 65 (большая выборка), 0 (ни одного события), 7
N_sandbox = 100      # Попробуйте: 100, 20, 10
hypo_1 = 0.65        # Попробуйте: 0.65, 0.80
hypo_2 = 0.50        # Попробуйте: 0.50, 0.40

lik_curve = stats.binom.pmf(k_sandbox, N_sandbox, theta_grid)
l_h1_sb = stats.binom.pmf(k_sandbox, N_sandbox, hypo_1)
l_h2_sb = stats.binom.pmf(k_sandbox, N_sandbox, hypo_2)
lr_sb = l_h1_sb / l_h2_sb if l_h2_sb > 0 else np.inf

print(f"Эксперимент в песочнице (k={k_sandbox}/{N_sandbox}):")
print(f"  ОМП:                          {theta_grid[np.argmax(lik_curve)]:.3f}")
print(f"  Правдоподобие при H1:         {l_h1_sb:.6e}")
print(f"  Правдоподобие при H2:         {l_h2_sb:.6e}")
print(f"  Отношение правдоподобий H1/H2: {lr_sb:.2f}x")
""",
)

# ---------------------------------------------------------------------------
# 10. STOP 2
# ---------------------------------------------------------------------------
md(
    """## 10. STOP 2
### 🛑 STOP: Compare results
▶ **RUN TOGETHER**: Let's show numerically why the likelihood is NOT a density by computing the area under the curve.
""",
    """## 10. СТОП 2
### 🛑 СТОП: сравните результаты
▶ **ЗАПУСКАЕМ ВМЕСТЕ**: покажем численно, почему правдоподобие — НЕ плотность, вычислив площадь под кривой.
""",
)
code(
    """area_lik = trapezoid(likelihood, theta_grid)
theoretical_area = 1.0 / (N_obs + 1)

print(f"Numerical area under L(θ | k=13, N=20): {area_lik:.5f}")
print(f"Exact area 1/(N+1) = 1/21:              {theoretical_area:.5f}")
print(f"Is the likelihood a probability density over θ? {'YES' if np.isclose(area_lik, 1.0) else 'NO (area ≠ 1.0)'}")
""",
    """area_lik = trapezoid(likelihood, theta_grid)
theoretical_area = 1.0 / (N_obs + 1)

print(f"Численная площадь под L(θ | k=13, N=20): {area_lik:.5f}")
print(f"Точная площадь 1/(N+1) = 1/21:           {theoretical_area:.5f}")
print(f"Является ли правдоподобие плотностью по θ? {'ДА' if np.isclose(area_lik, 1.0) else 'НЕТ (площадь ≠ 1.0)'}")
""",
)

# ---------------------------------------------------------------------------
# 11. Classwork 3: prior and posterior prediction
# ---------------------------------------------------------------------------
md(
    """## 11. Classwork 3: interpret — prior and posterior prediction
### C. Where do predictions come from? (preview of Lesson 4)
▶ **RUN TOGETHER**: So far $\\theta$ was fixed. Instead of fixing $\\theta$, draw it from a prior $\\operatorname{Beta}(2, 2)$ before generating counts; then repeat with the posterior $\\operatorname{Beta}(2 + 13, 2 + 7)$ after observing $k = 13$. This makes both sources of uncertainty visible at once.
""",
    """## 11. Классная работа 3: интерпретируйте — априорное и апостериорное предсказание
### C. Откуда берутся предсказания? (анонс занятия 4)
▶ **ЗАПУСКАЕМ ВМЕСТЕ**: до сих пор $\\theta$ был фиксирован. Вместо фиксированного $\\theta$ извлеките его из априорного распределения $\\operatorname{Beta}(2, 2)$ перед генерацией чисел; затем повторите с апостериорным $\\operatorname{Beta}(2 + 13, 2 + 7)$ после наблюдения $k = 13$. Так оба источника неопределённости видны одновременно.
""",
)
code(
    """# 1. Prior: Beta(2, 2)
prior_a, prior_b = 2, 2
theta_prior_draws = rng.beta(prior_a, prior_b, size=2000)
y_prior_pred = rng.binomial(n=N_obs, p=theta_prior_draws)      # prior predictive counts

# 2. Posterior: Beta(2 + 13, 2 + 7) = Beta(15, 9)
post_a = prior_a + k_obs
post_b = prior_b + (N_obs - k_obs)
theta_post_draws = rng.beta(post_a, post_b, size=2000)
y_post_pred = rng.binomial(n=N_obs, p=theta_post_draws)        # posterior predictive counts

fig_pred = make_subplots(rows=1, cols=2, subplot_titles=[
    '<b>Prior predictive</b><br><span style="font-size:11px;color:#64748b">before seeing data</span>',
    '<b>Posterior predictive</b><br><span style="font-size:11px;color:#64748b">after conditioning on k = 13</span>'])
fig_pred.add_trace(go.Histogram(x=y_prior_pred, histnorm='probability', marker_color='#94a3b8'), row=1, col=1)
fig_pred.add_trace(go.Histogram(x=y_post_pred, histnorm='probability', marker_color='#2563eb'), row=1, col=2)
fig_pred.add_vline(x=k_obs, line_dash='dash', line_color='#dc2626', annotation_text='observed k = 13', row=1, col=2)
fig_pred.update_layout(template='plotly_white', height=420, showlegend=False)
fig_pred.update_xaxes(title_text='Simulated k (out of 20)', range=[-0.5, 20.5])
fig_pred.update_yaxes(title_text='Probability')
fig_pred.show()

print(f"Prior predictive mean:     {y_prior_pred.mean():.2f} (SD = {y_prior_pred.std():.2f})")
print(f"Posterior predictive mean: {y_post_pred.mean():.2f} (SD = {y_post_pred.std():.2f})")
print(f"Posterior parameter mean:  {theta_post_draws.mean():.4f} (SD = {theta_post_draws.std():.4f})")
""",
    """# 1. Априорное: Beta(2, 2)
prior_a, prior_b = 2, 2
theta_prior_draws = rng.beta(prior_a, prior_b, size=2000)
y_prior_pred = rng.binomial(n=N_obs, p=theta_prior_draws)      # априорное предсказательное

# 2. Апостериорное: Beta(2 + 13, 2 + 7) = Beta(15, 9)
post_a = prior_a + k_obs
post_b = prior_b + (N_obs - k_obs)
theta_post_draws = rng.beta(post_a, post_b, size=2000)
y_post_pred = rng.binomial(n=N_obs, p=theta_post_draws)        # апостериорное предсказательное

fig_pred = make_subplots(rows=1, cols=2, subplot_titles=[
    '<b>Априорное предсказательное</b><br><span style="font-size:11px;color:#64748b">до наблюдения данных</span>',
    '<b>Апостериорное предсказательное</b><br><span style="font-size:11px;color:#64748b">после наблюдения k = 13</span>'])
fig_pred.add_trace(go.Histogram(x=y_prior_pred, histnorm='probability', marker_color='#94a3b8'), row=1, col=1)
fig_pred.add_trace(go.Histogram(x=y_post_pred, histnorm='probability', marker_color='#2563eb'), row=1, col=2)
fig_pred.add_vline(x=k_obs, line_dash='dash', line_color='#dc2626', annotation_text='наблюдение k = 13', row=1, col=2)
fig_pred.update_layout(template='plotly_white', height=420, showlegend=False)
fig_pred.update_xaxes(title_text='Симулированное k (из 20)', range=[-0.5, 20.5])
fig_pred.update_yaxes(title_text='Вероятность')
fig_pred.show()

print(f"Среднее априорного предсказания:     {y_prior_pred.mean():.2f} (SD = {y_prior_pred.std():.2f})")
print(f"Среднее апостериорного предсказания: {y_post_pred.mean():.2f} (SD = {y_post_pred.std():.2f})")
print(f"Апостериорное среднее параметра:     {theta_post_draws.mean():.4f} (SD = {theta_post_draws.std():.4f})")
""",
)
md(
    """> ✍ **WRITE**: Why does the posterior predictive distribution of a new survey count have a larger spread than the posterior distribution of $\\theta$ itself?
""",
    """> ✍ **НАПИШИТЕ**: почему апостериорное предсказательное распределение нового результата опроса имеет больший разброс, чем апостериорное распределение самого $\\theta$?
""",
)

# ---------------------------------------------------------------------------
# 12. STOP 3
# ---------------------------------------------------------------------------
md(
    """## 12. STOP 3
### 🛑 STOP: Compare with your neighbour
1. Which did you predict in Classwork 0, and did the simulation agree?
2. The prior predictive is much wider than the posterior predictive. Which part of that width is uncertainty about $\\theta$, and which part is variation between surveys?
3. A larger future survey makes the predicted *proportion* more stable. Does it make an individual respondent's answer more predictable?
""",
    """## 12. СТОП 3
### 🛑 СТОП: сравните с соседом
1. Что вы предсказали в классной работе 0 и согласилась ли с вами симуляция?
2. Априорное предсказание заметно шире апостериорного. Какая часть этой ширины — неопределённость относительно $\\theta$, а какая — разброс между опросами?
3. Опрос большего объёма делает предсказанную *долю* устойчивее. Делает ли он предсказуемее ответ отдельного человека?
""",
)

# ---------------------------------------------------------------------------
# 13. Exit record
# ---------------------------------------------------------------------------
md(
    """## 13. Exit record
> ✍ **WRITE**:
> 1. **Estimand**: Which quantity did the likelihood in Classwork 2 describe, and what was held fixed while it was evaluated?
> 2. **Likelihood vs density**: In your own words, why is the likelihood not a probability distribution over the parameter?
> 3. **Two kinds of uncertainty**: Give one example from today of variation at a fixed parameter and one example of uncertainty about the parameter.
> 4. **One limitation**: Name one assumption of the survey model you would check first with real data.
""",
    """## 13. Итоговая запись
> ✍ **НАПИШИТЕ**:
> 1. **Оцениваемая величина**: какую величину описывало правдоподобие в классной работе 2 и что было зафиксировано при его вычислении?
> 2. **Правдоподобие vs плотность**: своими словами — почему правдоподобие не является распределением вероятностей по параметру?
> 3. **Два вида неопределённости**: приведите по одному сегодняшнему примеру изменчивости при фиксированном параметре и неопределённости относительно параметра.
> 4. **Одно ограничение**: назовите одно допущение модели опроса, которое вы проверили бы на реальных данных первым.
""",
)

# ---------------------------------------------------------------------------
# 14. Optional homework
# ---------------------------------------------------------------------------
md(
    """## 14. Optional homework
### 🏠 OPTIONAL HOMEWORK: counts that a Poisson model cannot produce
So far the outcome was bounded by $N$. Many social outcomes are unbounded counts instead: calls to a municipal helpline in a day, complaints filed, events attended. The natural first model is Poisson, which forces **variance = mean**.

Real counts rarely obey that. One generative story for why: each day has its own rate $\\lambda_t$ around a baseline (a weekday pattern × random day-to-day variation), and the count is Poisson given $\\lambda_t$. Marginally this is a Negative Binomial. Simulate a year of days and watch the dispersion index (variance ÷ mean) leave the range a genuine Poisson process produces.
""",
    """## 14. Необязательное домашнее задание
### 🏠 ДОМАШНЕЕ ЗАДАНИЕ (по желанию): счётные данные, которых Пуассон не порождает
До сих пор исход был ограничен сверху числом $N$. Но многие социальные исходы — неограниченные счётчики: число обращений в городскую горячую линию за день, число поданных жалоб, число посещённых мероприятий. Естественная первая модель — Пуассон, а он жёстко требует **дисперсия = среднее**.

Реальные счётчики этому почти никогда не подчиняются. Одна из генеративных историй: у каждого дня своя интенсивность $\\lambda_t$ вокруг базового уровня (паттерн дня недели × случайная вариация от дня ко дню), а число событий при данном $\\lambda_t$ — пуассоновское. Маргинально это отрицательное биномиальное распределение. Симулируйте год дней и посмотрите, как индекс дисперсии (дисперсия ÷ среднее) выходит за пределы того, что даёт настоящий пуассоновский процесс.
""",
)
code(
    """baseline_rate = 8.0                 # expected events per day
n_days = 365
alpha_disp = 4.0                    # Try: 0.5 (very overdispersed), 4 (moderate), 50 (nearly Poisson)
weekday_mult = np.array([0.9, 0.9, 0.95, 1.0, 1.2, 1.3, 0.9])   # Mon..Sun; set all to 1.0 to remove the pattern


def dispersion_index(v):
    return v.var(ddof=1) / v.mean()


# What does the index look like when the counts really ARE Poisson?
ref = np.array([dispersion_index(rng.poisson(lam=baseline_rate, size=n_days)) for _ in range(500)])
lo, hi = np.percentile(ref, [2.5, 97.5])

weekday = np.arange(n_days) % 7
lam_day = baseline_rate * weekday_mult[weekday] * rng.gamma(shape=alpha_disp, scale=1.0 / alpha_disp, size=n_days)
y_year = rng.poisson(lam=lam_day)

print(f"A year of simulated daily counts (alpha = {alpha_disp}):")
print(f"  mean {y_year.mean():.2f}, variance {y_year.var(ddof=1):.2f}, dispersion index {dispersion_index(y_year):.2f}")
print(f"  a genuine Poisson process over {n_days} days gives an index in [{lo:.2f}, {hi:.2f}]")
print(f"  mean by weekday (Mon..Sun): {np.round([y_year[weekday == d].mean() for d in range(7)], 2)}")

fig_hw = go.Figure()
fig_hw.add_trace(go.Histogram(x=rng.poisson(lam=baseline_rate, size=n_days), xbins=dict(size=1), histnorm='probability',
                              opacity=0.6, marker_color='#2b6cb0', name=f'Poisson({baseline_rate:.0f})'))
fig_hw.add_trace(go.Histogram(x=y_year, xbins=dict(size=1), histnorm='probability',
                              opacity=0.6, marker_color='#e53e3e', name=f'Weekday × Gamma-Poisson (α={alpha_disp})'))
fig_hw.update_layout(template='plotly_white', height=380, barmode='overlay',
                     xaxis_title='events per day', yaxis_title='relative frequency')
fig_hw.show()
""",
    """baseline_rate = 8.0                 # ожидаемое число событий в день
n_days = 365
alpha_disp = 4.0                    # Попробуйте: 0.5 (сильная сверхдисперсия), 4 (умеренная), 50 (почти Пуассон)
weekday_mult = np.array([0.9, 0.9, 0.95, 1.0, 1.2, 1.3, 0.9])   # Пн..Вс; поставьте все единицы, чтобы убрать паттерн


def dispersion_index(v):
    return v.var(ddof=1) / v.mean()


# Каким бывает индекс, когда счётчики ДЕЙСТВИТЕЛЬНО пуассоновские?
ref = np.array([dispersion_index(rng.poisson(lam=baseline_rate, size=n_days)) for _ in range(500)])
lo, hi = np.percentile(ref, [2.5, 97.5])

weekday = np.arange(n_days) % 7
lam_day = baseline_rate * weekday_mult[weekday] * rng.gamma(shape=alpha_disp, scale=1.0 / alpha_disp, size=n_days)
y_year = rng.poisson(lam=lam_day)

print(f"Год симулированных дневных счётчиков (alpha = {alpha_disp}):")
print(f"  среднее {y_year.mean():.2f}, дисперсия {y_year.var(ddof=1):.2f}, индекс дисперсии {dispersion_index(y_year):.2f}")
print(f"  настоящий пуассоновский процесс за {n_days} дней даёт индекс в [{lo:.2f}, {hi:.2f}]")
print(f"  среднее по дням недели (Пн..Вс): {np.round([y_year[weekday == d].mean() for d in range(7)], 2)}")

fig_hw = go.Figure()
fig_hw.add_trace(go.Histogram(x=rng.poisson(lam=baseline_rate, size=n_days), xbins=dict(size=1), histnorm='probability',
                              opacity=0.6, marker_color='#2b6cb0', name=f'Пуассон({baseline_rate:.0f})'))
fig_hw.add_trace(go.Histogram(x=y_year, xbins=dict(size=1), histnorm='probability',
                              opacity=0.6, marker_color='#e53e3e', name=f'День недели × Гамма-Пуассон (α={alpha_disp})'))
fig_hw.update_layout(template='plotly_white', height=380, barmode='overlay',
                     xaxis_title='событий в день', yaxis_title='Относительная частота')
fig_hw.show()
""",
)
# ---------------------------------------------------------------------------
# 15. Reproducibility footer
# ---------------------------------------------------------------------------
md(
    """## 15. Reproducibility footer
▶ **RUN TOGETHER**: Run this cell to stamp the environment details for reproducibility.
""",
    """## 15. Воспроизводимость
▶ **ЗАПУСКАЕМ ВМЕСТЕ**: запустите ячейку, чтобы зафиксировать параметры окружения.
""",
)
code(
    """import datetime
import scipy
import plotly
print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"SciPy: {scipy.__version__}")
print(f"Plotly: {plotly.__version__}")
print(f"Random seed: {RANDOM_SEED}")
print("Data: synthetic, generated in this notebook (no downloads)")
""",
    """import datetime
import scipy
import plotly
print(f"Время запуска: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"SciPy: {scipy.__version__}")
print(f"Plotly: {plotly.__version__}")
print(f"Зерно генератора: {RANDOM_SEED}")
print("Данные: синтетические, сгенерированы в этой тетради (ничего не скачивается)")
""",
)


# ---------------------------------------------------------------------------
# Assemble notebooks
# ---------------------------------------------------------------------------
def build(lang: str) -> Path:
    nb = new_notebook()
    for cell_type, text in CELLS:
        src = text[lang].rstrip("\n")
        nb.cells.append(new_markdown_cell(src) if cell_type == "markdown" else new_code_cell(src))
    title = {
        "en": "Generative Models, Likelihood and Prediction",
        "ru": "Генеративные модели, правдоподобие и предсказание",
    }[lang]
    nb.metadata = {
        "bayesbook": {
            "assessment_id": None,
            "data_status": "synthetic",
            "estimated_classwork_minutes": 90,
            "graded": False,
            "language": lang,
            "lesson": 3,
            "seed": 42,
            "title": title,
        },
        "colab": {"name": FILES[lang], "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11",
        },
    }
    out = OUT_DIR / FILES[lang]
    nbformat.write(nb, out)
    return out


if __name__ == "__main__":
    for lang in FILES:
        path = build(lang)
        print(f"wrote {path.relative_to(ROOT)} ({len(CELLS)} cells)")
