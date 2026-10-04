"""Create Russian versions of the interactive figures for Lessons 1-6.

Run from the book root (after the English figures have been generated):

    conda run -n pymc_env python scripts/translate_figures_ru.py           # translate all
    conda run -n pymc_env python scripts/translate_figures_ru.py --check   # also list untranslated English

For every ``_static/<name>.html`` in FIGURES this writes ``_static/<name>_ru.html``.
Plotly-generated figures are translated string by string inside their JSON (so
escaping stays valid); hand-written JavaScript figures and HTML text nodes are
translated by phrase replacement. The plotly.js library itself is never touched.

Terminology follows the course glossary (see the Russian lesson pages) and the
Russian usage of S. Nikolenko's lectures: априорное / апостериорное распределение,
правдоподобие, нормировочная константа, порождающая модель, предсказательное
распределение, сэмплы, проклятие размерности, лапласовская аппроксимация,
вариационная нижняя оценка (ELBO), KL-дивергенция.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "_static"

FIGURES = [
    # Lesson 1
    "bayes_freq_demo", "clt_demo",
    # Lesson 2
    "monty_hall_demo", "interactive_base_rates", "interactive_prevalence_curve", "fig2_3_sequential_bernoulli_triad",
    "fig2_4_likelihood_vs_density", "interactive_sequential_updating",
    # Lesson 3
    "dgp_binomial_pmf", "interactive_likelihood_ratio",
    # Lesson 4
    "lesson4_eti_hdi", "lesson4_cred_vs_conf", "lesson4_rope", "lesson4_prediction",
    # Lesson 5
    "lesson5_shape_not_size", "lesson5_answers_are_areas", "lesson5_grid_area", "lesson5_grid_explosion", "lesson5_empty_box",
    "lesson5_darts_pi", "lesson5_pile_of_draws", "lesson5_rejection", "lesson5_weight_vs_place", "lesson5_ratio_only",
    "lesson5_king_markov", "lesson5_forget_start", "lesson5_balanced_traffic", "lesson5_walker_hill", "lesson5_gibbs_staircase",
    "lesson5_hmc_vs_walk", "lesson5_timeline",
    # Lesson 5, further reading (6.MonteCarloEstimation, 6.MarkovChains, 6.MetropolisHastings, 6.GibbsSampling)
    "fig6_1_mc_running_mean", "fig6_rejection_sampling", "fig6_5_transition_convergence", "fig6_6_autocorrelation_ess",
    "fig6_8_proposal_scale", "fig6_11_gibbs_path",
    # Lesson 5, further reading (5.HamiltonianMonteCarloNUTS)
    "fig6_13_hmc_vs_rw",
    # Lesson 6
    "lesson6_step_size", "lesson6_warmup", "lesson6_four_chains", "lesson6_sticky_ess", "lesson6_mcse_wobble",
    "lesson6_coverage_two_hills", "lesson6_pymc_vs_exact", "lesson6_broken_fits", "lesson6_laplace", "lesson6_vi_fit",
    "lesson6_meanfield", "lesson6_method_map",
]

N = r"([-−]?[\d][\d,.]*(?:e[+-]?\d+)?)"  # a number as printed in the figures

# Regex rules with numbers (applied first, in order).
REGEX = [
    # ---------- Lesson 1 ----------
    (r"Study (\d+)", r"Исследование \1"),
    (r"Mean: ", "Среднее: "),
    (r"95% CI: ", "95% ДИ: "),
    (r"Prior: Normal\(" + N + r", SD=" + N + r"\); selected sample mean=" + N,
     r"Априорное: Normal(\1, SD=\2); выборочное среднее выделенного исследования = \3"),
    (r"Synthetic Normal observations: n=(\d+), known SD=" + N + r"<br><sup>Generating mean=" + N + r"; (\d+)/(\d+) intervals cover it in this run\. "
     r"Blue interval supplies the data for the posterior\.</sup>",
     r"Нормальные наблюдения: n=\1, известное SD=\2<br><sup>Истинное среднее \3; его накрывают \4 из \5 интервалов; синий — данные для апостериорного.</sup>"),
    (r"Means from ([\d,]+) simulated studies", r"Средние \1 исследований"),
    (r"Exponential\(mean=(\d+)\)", r"Экспоненциальное(среднее=\1)"),
    (r"Poisson\(mean=(\d+)\)", r"Пуассона(среднее=\1)"),
    (r"Uniform\(0, 1\)", "Равномерное(0, 1)"),
    # ---------- Lesson 2 ----------
    (r"Step 0: Initial Prior", "Шаг 0: исходное априорное"),
    (r"<b>Step (\d) \(Initial Baseline\):</b> Initial Prior θ ~ Beta\((\d+), (\d+)\) \[Mean = " + N + r"\]\. No new data yet\.",
     r"<b>Шаг \1 (исходное состояние):</b> априорное θ ~ Beta(\2, \3) [среднее = \4]. Данных пока нет."),
    (r"<b>Step (\d) \(Flip (\d) = Heads\):</b>", r"<b>Шаг \1 (бросок \2 = орёл):</b>"),
    (r"<b>Step (\d) \(Flip (\d) = Tails\):</b>", r"<b>Шаг \1 (бросок \2 = решка):</b>"),
    (r"Prior Beta\((\d+),(\d+)\) \[from Step (\d)\]", r"априорное Beta(\1,\2) [из шага \3]"),
    (r"Prior Beta\((\d+),(\d+)\) ×", r"априорное Beta(\1,\2) ×"),
    (r"× Likelihood L\(θ\)", "× правдоподобие L(θ)"),
    (r"➔ Updated Posterior", "➔ обновлённое апостериорное"), (r"➔ Final Posterior", "➔ итоговое апостериорное"),
    (r"\[Mean = ", "[среднее = "), (r"\[from Step (\d)\]", r"[из шага \1]"),
    (r"New Evidence: (y\S+) = (\d) \(Heads\)", r"Новые данные: \1 = \2 (орёл)"),
    (r"New Evidence: (y\S+) = (\d) \(Tails\)", r"Новые данные: \1 = \2 (решка)"),
    (r"Switching doubles your winning probability!", "смена двери удваивает шанс!"),
    (r"Flip (\d+): H \(\+1\)", r"Бросок \1: орёл (+1)"),
    (r"Flip (\d+): T \(\+0\)", r"Бросок \1: решка (+0)"),
    (r"Car at Door %\{x\}", "Машина за дверью %{x}"),
    (r"Pick Door (\d) · Host opens (\d)", r"Выбрана дверь \1 · ведущий открыл \2"),
    (r"<b>Stay at Door (\d):</b>", r"<b>Остаться (\1):</b>"),
    (r"<b>Switch to Door (\d):</b>", r"<b>Сменить (\1):</b>"),
    (r"<b>DOOR (\d)</b>", r"<b>ДВЕРЬ \1</b>"),
    (r"Door (\d)", r"Дверь \1"),
    (r"<b>Rare: " + N + r"% Prev → " + N + r"% Post</b>", r"<b>Редкое: распространённость \1% → апостериорная \2%</b>"),
    (r"<b>Moderate: " + N + r"% Prev → " + N + r"% Post</b>", r"<b>Умеренное: распространённость \1% → апостериорная \2%</b>"),
    (r"Prior Beta\((\d+),(\d+)\) \[Mean=" + N + r"\]", r"Априорное Beta(\1,\2) [среднее=\3]"),
    (r"Posterior Beta\((\d+),(\d+)\) \[Mean=" + N + r"\]", r"Апостериорное Beta(\1,\2) [среднее=\3]"),
    (r"Post Beta\((\d+),(\d+)\) \[Mean=" + N + r"\]", r"Апостер. Beta(\1,\2) [среднее=\3]"),
    (r"Prior Beta\((\d+),(\d+)\)<br>", r"Априорное Beta(\1,\2)<br>"),
    (r"Posterior Beta\((\d+),(\d+)\)<br>", r"Апостериорное Beta(\1,\2)<br>"),
    (r"Likelihood L\(θ \| k=(\d+)\)", r"Правдоподобие L(θ | k=\1)"),
    (r"Posterior p\(θ \| k=(\d+)\) \[Beta\((\d+),(\d+)\) Prior\]", r"Апостериорное p(θ | k=\1) [априорное Beta(\2,\3)]"),
    (r"Posterior p\(θ \| k=(\d+)\)", r"Апостериорное p(θ | k=\1)"),
    (r"<b>Observed Data</b><br>k = (\d+) \(P = " + N + r"\)", r"<b>Наблюдённые данные</b><br>k = \1 (P = \2)"),
    (r"<b>Likelihood Peak = " + N + r"</b><br>Area under L\(θ\) ≈ " + N + r"<br><i>\(NOT a probability density!\)</i>",
     r"<b>Пик правдоподобия = \1</b><br>Площадь под L(θ) ≈ \2<br><i>(это НЕ плотность вероятности!)</i>"),
    (r"<b>Posterior Mode = " + N + r"</b><br>Area under p\(θ\|D\) = <b>1\.00</b><br><i>\(Normalized via Bayes Rule\)</i>",
     r"<b>Мода апостериорного = \1</b><br>Площадь под p(θ|D) = <b>1.00</b><br><i>(нормировано по теореме Байеса)</i>"),
    (r"<b>Likelihood Peak at θ = " + N + r"</b><br>L\(θ=" + N + r"\) = " + N, r"<b>Пик правдоподобия при θ = \1</b><br>L(θ=\2) = \3"),
    (r"<b>Likelihood Peak at θ = " + N + r"</b><br>Maximized near boundary", r"<b>Пик правдоподобия при θ = \1</b><br>максимум у границы"),
    (r"<b>Posterior Mode = " + N + r"</b><br>Beta\((\d+), (\d+)\) Density \(Area = 1\.0\)",
     r"<b>Мода апостериорного = \1</b><br>плотность Beta(\2, \3) (площадь = 1.0)"),
    (r"<b>Posterior Mode = " + N + r"</b><br><i>Skeptical prior pulls peak leftward!</i><br>",
     r"<b>Мода апостериорного = \1</b><br><i>Скептическое априорное сдвигает пик влево!</i><br>"),
    (r"k = (\d+)/(\d+) \(Uniform Prior\)", r"k = \1/\2 (равномерное)"),
    (r"k = (\d+)/(\d+) \+ Beta\((\d+),(\d+)\) Prior", r"k = \1/\2 + Beta(\3,\4)"),
    (r"<b>Data: k = (\d+)</b><br>Sample rate = (\d+)%", r"<b>Данные: k = \1</b><br>доля в выборке = \2%"),
    (r"<b>Likelihood peak = " + N + r"</b><br>\(Data alone\)", r"<b>Пик правдоподобия = \1</b><br>(только данные)"),
    (r"Fixed θ = " + N + r", outcome k varies \(Sums to 1\.0\)", r"θ = \1 фикс., k меняется (сумма = 1.0)"),
    (r"Fixed k = (\d+), candidate θ varies \(Area = 1\.0 vs " + N + r"\)", r"k = \1 фикс., θ меняется (площади 1.0 и \2)"),
    (r"Probability Mass P\(k \| θ = " + N + r"\)", r"Вероятность P(k | θ = \1)"),
    # ---------- Lesson 3 ----------
    (r"Mean \(θ=" + N + r"\)", r"Среднее (θ=\1)"),
    (r"Number of Successes k \(out of N=(\d+) trials\)", r"Число успехов k (из N=\1 испытаний)"),
    (r"Likelihood L\(θ \| k=(\d+), N=(\d+)\)", r"Правдоподобие L(θ | k=\1, N=\2)"),
    (r"<b>Hypothesis H₁ \(MLE\)</b>", "<b>Гипотеза H₁ (МП-оценка)</b>"),
    (r"<b>Hypothesis H₂ \(Even Split\)</b>", "<b>Гипотеза H₂ (поровну)</b>"),
    (r"H₁: θ = " + N + r" \(MLE, L = " + N + r"\)", r"H₁: θ = \1 (МП-оценка, L = \2)"),
    (r"<b>H₁: θ = " + N + r" \(MLE\)</b>", r"<b>H₁: θ = \1 (МП-оценка)</b>"),
    (r"H₂: θ = " + N + r" \(Fair, L = " + N + r"\)", r"H₂: θ = \1 (честная, L = \2)"),
    (r"<b>Likelihood Ratio:</b>", "<b>Отношение правдоподобий:</b>"),
    (r"<b>Binomial Likelihood Function</b> L\(θ \| k=(\d+), N=(\d+)\) and Hypothesis Evaluation",
     r"<b>Биномиальная функция правдоподобия</b> L(θ | k=\1, N=\2) и сравнение гипотез"),
    # ---------- Lesson 4 ----------
    (r"ETI \(Wider, pushed to tail\): ", "ETI (шире, сдвинут к хвосту): "),
    (r"HDI \(Shorter, captures peak\): ", "HDI (короче, захватывает пик): "),
    (r"Area under curve = P\(Δ > " + N + r" \| Data\) = " + N + "%", r"Площадь под кривой = P(Δ > \1 | данные) = \2%"),
    # ---------- Lesson 5 ----------
    (r"The blue bars have the right shape but a total area of " + N + r", not 1\.<br>Dividing by that area \(the evidence\) turns the shape into a distribution\.",
     r"У синих столбиков правильная форма, но общая площадь \1, а не 1.<br>Делим на эту площадь (нормировочную константу) и получаем распределение."),
    (r"Orange area = <b>" + N + r"%</b><br>= P\(support > 50%\)", r"Оранжевая площадь = <b>\1%</b><br>= P(поддержка > 50%)"),
    (r"Middle 95% of the area:<br>", "Средние 95% площади:<br>"),
    (r"Average = <b>" + N + r"%</b>:<br>the area balances here", r"Среднее = <b>\1%</b>:<br>здесь площадь уравновешена"),
    (r"<b>(\d+) bars</b><br>area ≈ " + N + r"<br>exact = " + N, r"<b>\1 столбиков</b><br>площадь ≈ \2<br>точно = \3"),
    (r"^(\d+) bars$", r"\1 столбиков"),
    (r"^([\d,]+)<br>years$", r"\1<br>лет"),
    (r"<b>2 unknowns</b>: a 100 × 100 grid already has 10,000 cells", "<b>2 неизвестных</b>: сетка 100 × 100 — уже 10 000 ячеек"),
    (r"<b>2 unknowns</b>: 79% of random points land in the useful circle", "<b>2 неизвестных</b>: 79% случайных точек попадает в полезный круг"),
    (r"<b>([\d,]+) darts</b><br>π ≈ 4 × share inside<br>= <b>" + N + "</b>", r"<b>\1 дротиков</b><br>π ≈ 4 × доля внутри<br>= <b>\2</b>"),
    (r"<b>([\d,]+) draws</b><br>orange share = <b>" + N + r"%</b><br>exact = " + N + "%",
     r"<b>\1 сэмплов</b><br>оранжевая доля = <b>\2%</b><br>точно = \3%"),
    (r"^([\d,]+) draws$", r"\1 сэмплов"),
    (r"<b>Darts in a box: keep those under the curve</b> \(kept (\d+)%\)", r"<b>Дротики в прямоугольнике: оставляем те, что под кривой</b> (оставлено \1%)"),
    (r"It needs only the shape of the curve, but (\d+)% of the darts are wasted\.",
     r"Нужна только форма кривой, но \1% дротиков пропадает впустую."),
    (r"curve <b>(.*?)</b><br>height at 50%: (.*?)<br>height at 58%: (.*?)<br><b>ratio = " + N + r"</b> \(always\)",
     r"кривая <b>\1</b><br>высота при 50%: \2<br>высота при 58%: \3<br><b>отношение = \4</b> (всегда)"),
    (r"<b>Night ([\d,]+)</b>", r"<b>Ночь \1</b>"),
    (r"starts on island (\d)", r"старт на острове \1"),
    (r"target (\d+)%", r"цель \1%"),
    (r"share on island (\d)", r"доля на острове \1"),
    (r"<b>A</b><br>([\d,]+)<br>people<br>(\d+)%<br>of nights", r"<b>A</b><br>\1<br>жителей<br>\2%<br>ночей"),
    (r"<b>Island B</b><br>([\d,]+) people<br>(\d+)% of nights", r"<b>Остров B</b><br>\1 жителей<br>\2% ночей"),
    (r"Visits (\d+) : (\d+) \(= populations\)", r"Ночи \1 : \2 (= населению)"),
    (r"Visits (\d+) : (\d+)", r"Ночи \1 : \2"),
    (r"<b>Step ([\d,]+)</b>", r"<b>Шаг \1</b>"),
    (r"<b>Random walk</b>: (\d+) steps", r"<b>Случайное блуждание</b>: \1 шагов"),
    (r"<b>Skateboarder \(HMC\)</b>: (\d+) glides", r"<b>Скейтбордист (HMC)</b>: \1 скольжений"),
    (r"· Hastings</b>", "· Гастингс</b>"), (r"· Neal</b>", "· Нил</b>"), (r"· Laplace</b>", "· Лаплас</b>"),
    (r"<b>as computed</b>", "<b>как есть</b>"),
    # ---------- Lesson 5, further reading ----------
    (r"\(a\) Running Means \(S=(\d+)\)", r"(a) Текущие средние (S=\1)"),
    (r"Accepted \(([\d,]+) draws, " + N + r"% rate\)", r"Принято (\1 сэмплов, доля \2%)"),
    (r"Target p\(θ\) = Beta\((.*?)\)", r"Цель p(θ) = Beta(\1)"),
    (r"Envelope M · q\(θ\) \(M = " + N + r"\)", r"Огибающая M · q(θ) (M = \1)"),
    (r"S = ([\d,]+) \(Small\)", r"S = \1 (мало)"), (r"S = ([\d,]+) \(Medium\)", r"S = \1 (средне)"),
    (r"S = ([\d,]+) \(Large\)", r"S = \1 (много)"),
    (r"\(a\) Initial State (\d) Convergence", r"(a) Сходимость из состояния \1"),
    (r"\(b\) Initial State (\d) Convergence", r"(b) Сходимость из состояния \1"),
    (r"Independent \(ρ=0, ESS=(\d+)\)", r"Независимые (ρ=0, ESS=\1)"),
    (r"Moderate \(ρ=" + N + r", ESS=(\d+)\)", r"Умеренная (ρ=\1, ESS=\2)"),
    (r"High \(ρ=" + N + r", ESS=(\d+)\)", r"Высокая (ρ=\1, ESS=\2)"),
    (r"\(a\) σ=" + N + r" \(Too Small\)", r"(a) σ=\1 (слишком мал)"),
    (r"\(b\) σ=" + N + r" \(Optimal\)", r"(b) σ=\1 (подходящий)"),
    (r"\(c\) σ=" + N + r" \(Too Large\)", r"(c) σ=\1 (слишком велик)"),
    (r"Gibbs Path \(ρ=" + N + r"\)", r"Путь Гиббса (ρ=\1)"),
    (r"\(a\) Low Correlation \(ρ=" + N + r"\)", r"(a) Слабая корреляция (ρ=\1)"),
    (r"\(b\) High Correlation \(ρ=" + N + r"\)", r"(b) Сильная корреляция (ρ=\1)"),
    # ---------- Lesson 6 ----------
    (r"<b>Tiny steps</b> \(" + N + r"\): accepted (\d+)%, worth ≈ <b>([\d,]+)</b> independent draws",
     r"<b>Крошечные шаги</b> (\1): принято \2%, стоят ≈ <b>\3</b> независимых сэмплов"),
    (r"<b>Huge steps</b> \(" + N + r"\): accepted (\d+)%, worth ≈ <b>([\d,]+)</b> independent draws",
     r"<b>Огромные шаги</b> (\1): принято \2%, стоят ≈ <b>\3</b> независимых сэмплов"),
    (r"<b>Good steps</b> \(" + N + r"\): accepted (\d+)%, worth ≈ <b>([\d,]+)</b> independent draws",
     r"<b>Хорошие шаги</b> (\1): принято \2%, стоят ≈ <b>\3</b> независимых сэмплов"),
    (r"<b>Four chains</b>: R-hat = <b>" + N + r"</b>, ESS = ([\d,]+) → 🟢 they agree",
     r"<b>Четыре цепи</b>: R-hat = <b>\1</b>, ESS = \2 → 🟢 согласны"),
    (r"<b>Four chains</b>: R-hat = <b>" + N + r"</b>, ESS = ([\d,]+) → 🔴 they disagree: do not interpret",
     r"<b>Четыре цепи</b>: R-hat = <b>\1</b>, ESS = \2 → 🔴 не согласны: не интерпретировать"),
    (r"<b>First 300 of 2,000 draws</b>: worth ≈ <b>([\d,]+)</b> independent draws 🔴 \(below 400\)",
     r"<b>Первые 300 из 2000 сэмплов</b>: стоят ≈ <b>\1</b> независимых 🔴 (меньше 400)"),
    (r"<b>First 300 of 2,000 draws</b>: worth ≈ <b>([\d,]+)</b> independent draws 🟢",
     r"<b>Первые 300 из 2000 сэмплов</b>: стоят ≈ <b>\1</b> независимых 🟢"),
    (r"Very sticky \(" + N + r"\)", r"Очень «липкие» (\1)"),
    (r"^Sticky \(" + N + r"\)$", r"«Липкие» (\1)"),
    (r"Lively \(" + N + r"\)", r"Подвижные (\1)"),
    (r"<b>Width</b>: 100 vs 10,000 draws have the same spread", "<b>Ширина</b>: у 100 и 10 000 сэмплов одинаковый разброс"),
    (r"<b>Four chains from different starts</b>: R-hat = " + N + " 🔴", r"<b>Четыре цепи из разных стартов</b>: R-hat = \1 🔴"),
    (r"<b>Gap between the groups</b>: P\(gap > 5 points\) = " + N + r" \(exact " + N + r"\)",
     r"<b>Разрыв между группами</b>: P(разрыв > 5 п. п.) = \1 (точно \2)"),
    (r"(🟢|🔴) R-hat " + N + r" · (🟢|🔴) ESS ([\d,]+) · (🟢|🔴) divergences (\d+) → <b>may interpret</b>",
     r"\1 R-hat \2 · \3 ESS \4 · \5 расхождений \6 → <b>можно интерпретировать</b>"),
    (r"(🟢|🔴) R-hat " + N + r" · (🟢|🔴) ESS ([\d,]+) · (🟢|🔴) divergences (\d+) → <b>do not interpret</b>",
     r"\1 R-hat \2 · \3 ESS \4 · \5 расхождений \6 → <b>не интерпретировать</b>"),
    (r"<b>95% interval</b><br>exact: (.*?)<br>Laplace: (.*?)<br>", r"<b>95% интервал</b><br>точно: \1<br>Лаплас: \2<br>"),
    (r"<b>(\d+)%</b> of the bell<br>is below 0%: impossible", r"<b>\1%</b> колокола<br>ниже 0%: невозможно"),
    (r"^step (\d+) · remaining KL = " + N, r"шаг \1 · осталось KL = \2"),
    (r"<b>The bell moves and changes width</b>", "<b>Колокол сдвигается и меняет ширину</b>"),
    (r"Two unknowns that move together<br>\(correlation " + N + r"\)\. Spread of<br>unknown 1 in the draws: <b>" + N + "</b>",
     r"Два неизвестных, которые меняются вместе<br>(корреляция \1). Разброс<br>неизвестного 1 в сэмплах: <b>\2</b>"),
    (r"Mean-field: an independent bell<br>per unknown\. Spread: <b>" + N + r"</b>,<br>less than half: ",
     r"Среднее поле: независимый колокол<br>на каждое неизвестное. Разброс: <b>\1</b>,<br>меньше половины: "),
    (r"Full-rank VI lets the bell tilt:<br>spread <b>" + N + r"</b>, like the draws,<br>but it costs more computation\.",
     r"Полноранговый VI позволяет колоколу наклониться:<br>разброс <b>\1</b>, как у сэмплов,<br>но вычислений больше."),
    (r"<b>95% intervals</b><br>exact: (.*?)<br>MCMC: (.*?)<br>Laplace: (.*?)<br>VI: (.*?)<br>MAP: ",
     r"<b>95% интервалы</b><br>точно: \1<br>MCMC: \2<br>Лаплас: \3<br>VI: \4<br>MAP: "),
]

# Rules that apply to one figure only (checked before the general rules).
FIG_RULES = {
    "lesson4_cred_vs_conf": [(r"95% CI: ", "95% байес. интервал: ")],
    # the long axis titles are clipped in the narrow two-panel layout
    "fig6_13_hmc_vs_rw": [(r"Parameter θ", "θ")],
}
CURRENT_FIG = {"name": ""}

# Literal phrases (applied after REGEX, longest first).
LITERAL = {
    # ---------- shared ----------
    "▶ Play": "▶ Пуск", "❚❚ Pause": "❚❚ Пауза",
    # ---------- fig6_13 (HMC vs random walk) ----------
    "HMC Physics Trajectory": "Траектория HMC",
    "Random Walk (Slow Diffusion)": "Случайное блуждание (медленная диффузия)",
    "(a) Random-Walk Metropolis": "(a) Метрополис со случайным блужданием",
    "(b) Hamiltonian Dynamics (HMC)": "(b) Гамильтонова динамика (HMC)",
    "Probability Density p(θ)": "Плотность вероятности p(θ)",
    "Posterior Density": "Апостериорная плотность",
    "Parameter Value": "Значение параметра",
    "Parameter θ": "Параметр θ",
    "Density: ": "Плотность: ", "Density": "Плотность", "density": "плотность",
    "Probability: ": "Вероятность: ",
    "Likelihood: ": "Правдоподобие: ", "Prior: ": "Априорное: ", "Posterior: ": "Апостериорное: ",
    # ---------- Lesson 1 ----------
    "95% posterior interval": "95% апостериорный интервал",
    "Estimate and confidence interval": "Оценка и доверительный интервал",
    "Simulated study": "Смоделированное исследование",
    "Unknown population mean": "Неизвестное среднее в популяции",
    "40 repeated studies: 95% confidence intervals": "40 исследований: 95% ДИ",
    "Highlighted study: prior and posterior": "Одно исследование",
    "Individual observations": "Отдельные наблюдения",
    "Sample means": "Выборочные средние",
    "Normal approximation": "Нормальное приближение",
    "Individual value": "Отдельное значение",
    "Mean within one study": "Среднее в одном исследовании",
    "What becomes approximately Normal?": "Что становится примерно нормальным?",
    # ---------- Lesson 2 ----------
    "<b>Hypothesized car location</b>": "<b>Где может быть машина</b>",
    "<b>Probability</b>": "<b>Вероятность</b>",
    "<b>P(Host opens | Car)</b>": "<b>P(ведущий открывает | машина)</b>",
    "<b>Prior</b>: P(Car at Door)": "<b>Априорное</b>: P(машина)",
    "<b>Likelihood</b>: P(Host Opens | Car)": "<b>Правдоподобие</b>: P(открыл | машина)",
    "<b>Posterior</b>: P(Car at Door | Action)": "<b>Апостериорное</b>: P(машина | открыл)",
    "<b>YOUR PICK · closed</b>": "<b>ВАШ ВЫБОР · закрыта</b>",
    "<b>HOST OPENS · goat</b>": "<b>ВЕДУЩИЙ ОТКРЫЛ · коза</b>",
    "<b>SWITCH · closed</b>": "<b>СМЕНИТЬ · закрыта</b>",
    "<b>PRIZE LEGEND</b>": "<b>ОБОЗНАЧЕНИЯ</b>",
    "<b>Monty Hall Paradox</b>": "<b>Парадокс Монти Холла</b>",
    "Monty Hall probability explorer": "Парадокс Монти Холла: вероятности",
    "Prevalence: ": "Распространённость: ",
    "Posterior P(C|+): ": "Апостериорная P(C|+): ",
    "Posterior Curve": "Апостериорная кривая",
    "🔴 Rare (0.5%)": "🔴 Редкое (0.5%)", "🟠 Moderate (5%)": "🟠 Умеренное (5%)",
    "🟢 Common (20%)": "🟢 Частое (20%)", "🔵 High (50%)": "🔵 Высокое (50%)",
    "Count": "Число",
    "Prior Prevalence (%)": "Априорная распространённость (%)",
    "Posterior P(C | +) (%)": "Апостериорная P(C | +) (%)",
    "<b>(a) Counts (per 10,000 tested)</b>": "<b>(a) Число на 10 000 тестов</b>",
    "<b>(b) Posterior P(Condition | +)</b>": "<b>(b) P(болезнь | +)</b>",
    "Prevalence Scenario:": "Сценарий распространённости:",
    "PREVALENCE SCENARIO:": "СЦЕНАРИЙ РАСПРОСТРАНЁННОСТИ:",
    "True (+)<br>Condition": "Истинно (+)<br>болен",
    "False (+)<br>Healthy": "Ложно (+)<br>здоров",
    "P(D|+): ": "P(Б|+): ",
    "P(Disease | Positive)": "P(болезнь | положительный тест)",
    "Rare (0.1%)": "Редкое (0.1%)", "Moderate (5%)": "Умеренное (5%)",
    "Population Disease Prevalence (%)": "Распространённость болезни в популяции (%)",
    "Posterior P(Disease | Test +) (%)": "Апостериорная P(болезнь | тест +) (%)",
    "(a) Prior: ": "(a) Априорное: ", "(b) New Evidence: None": "(b) Новые данные: нет",
    "(b) New Evidence: ": "(b) Новые данные: ", "(c) Posterior: ": "(c) Апостериорное: ",
    "Likelihood": "Правдоподобие",
    "<b>Likelihood</b>": "<b>Правдоподобие</b>",
    "<b>Posterior Density</b>": "<b>Апостериорная плотность</b>",
    "<b>k = %{x}</b><br>Probability: ": "<b>k = %{x}</b><br>Вероятность: ",
    "Data Outcome k (Number of Successes)": "Исход k (число успехов)",
    "Hypothesized Parameter θ (True Success Rate)": "Гипотетический параметр θ (истинная доля успехов)",
    "Function Value (Likelihood vs. Density)": "Значение (правдоподобие или плотность)",
    "<b>(a) Generative: P(Data | Model)</b>": "<b>(a) Порождающая модель: P(k | θ)</b>",
    "<b>(b) Inferential: L(Model | Data) vs. Posterior</b>": "<b>(b) Правдоподобие и апостериорное</b>",
    "Coin Heads Probability Parameter θ": "Вероятность орла θ",
    "Choose Prior:": "Выберите априорное:",
    "🔵 Uniform Beta(1,1)": "🔵 Равномерное Beta(1,1)", "🟢 Fair Beta(10,10)": "🟢 «Честная» Beta(10,10)",
    "🟠 Heads-Bias Beta(8,2)": "🟠 Перекос к орлу Beta(8,2)", "🔴 Tails-Bias Beta(2,8)": "🔴 Перекос к решке Beta(2,8)",
    "🌈 Compare All 4": "🌈 Сравнить все 4",
    # ---------- Lesson 3 ----------
    "Successes k: ": "Успехов k: ",
    "Probability P(k | N, θ)": "Вероятность P(k | N, θ)",
    "🔵 Symmetric (θ=0.5)": "🔵 Симметричное (θ=0.5)", "🟢 High Rate (θ=0.8)": "🟢 Высокая доля (θ=0.8)",
    "🔴 Low Rate (θ=0.2)": "🔴 Низкая доля (θ=0.2)", "🌈 Overlay All": "🌈 Все вместе",
    "Likelihood L(θ): ": "Правдоподобие L(θ): ",
    "Success Probability Parameter θ": "Вероятность успеха θ",
    "Likelihood L(θ | Data)": "Правдоподобие L(θ | данные)",
    # ---------- Lesson 4 ----------
    "95% Credible Interval": "95% байесовский интервал",
    "100 Simulated Samples": "100 смоделированных выборок",
    "<b>Bayesian Credible Interval</b>": "<b>Байесовский интервал</b>",
    "95% Probability mass (Data is fixed)": "95% вероятности (данные фиксированы)",
    "<b>Frequentist Confidence Intervals</b>": "<b>Доверительные интервалы</b>",
    "95% Coverage across repeated samples": "95% покрытия в повторных выборках",
    "True θ (Fixed)": "Истинное θ (фиксировано)",
    "95% Equal-Tailed Interval (ETI)": "95% интервал с равными хвостами (ETI)",
    "95% Highest Density Interval (HDI)": "95% интервал наибольшей плотности (HDI)",
    "<b>ETI vs HDI on a Skewed Posterior</b>": "<b>ETI и HDI для скошенного апостериорного</b>",
    "The HDI shifts left to capture the most probable values. ETI guarantees 2.5% in each tail.":
        "HDI сдвигается влево к самым вероятным значениям. ETI оставляет по 2,5% в каждом хвосте.",
    "<b>Expected Mean (μ)</b><br>Narrow Epistemic Uncertainty": "<b>Ожидаемое среднее (μ)</b><br>узкая неопределённость параметра",
    "<b>Future Observation (y)</b><br>Wide Aleatory Uncertainty": "<b>Новое наблюдение (y)</b><br>широкая неопределённость исхода",
    "<b>Parameter Uncertainty vs. Predictive Uncertainty</b>": "<b>Неопределённость параметра и предсказания</b>",
    "Outcome Scale": "Шкала исхода",
    "Posterior Difference": "Апостериорная разность",
    "Prob > 0.05": "Вероятность > 0.05",
    "Threshold (5 points)": "Порог (5 п. п.)",
    "<b>Hypothesis Testing: Probability that Difference > 5 points</b>": "<b>Проверка гипотезы: вероятность того, что разность > 5 п. п.</b>",
    "Difference in Support (Group A - Group B)": "Разность в поддержке (группа A − группа B)",
    # ---------- Lesson 5 ----------
    "<b>Prior</b>": "<b>Априорное</b>", "<b>× Likelihood</b>": "<b>× Правдоподобие</b>", "<b>= Posterior shape</b>": "<b>= Форма апостериорного</b>",
    "support among lower-income households": "поддержка среди домохозяйств с низким доходом",
    "<b>Bayes' theorem gives a shape, not a size</b>": "<b>Теорема Байеса даёт форму, а не размер</b>",
    "<b>Every answer is an area under the posterior</b>": "<b>Любой ответ — это площадь под апостериорным</b>",
    "Click a question. In Lesson 4 the Beta formula computed these areas for us.":
        "Нажмите на вопрос. На занятии 4 эти площади за нас считала формула бета-распределения.",
    "support (lower-income)": "поддержка (низкий доход)",
    "P(support > 50%)": "P(поддержка > 50%)", "Middle 95%": "Средние 95%", "Average": "Среднее",
    "<b>With one unknown, a grid computes the area</b>": "<b>С одним неизвестным площадь считает сетка</b>",
    "Move the slider: a few dozen bars already give the exact area (the denominator).":
        "Двигайте ползунок: несколько десятков столбиков уже дают точную площадь (знаменатель).",
    "prior × likelihood": "априорное × правдоподобие",
    "longer than<br>the universe": "дольше возраста<br>Вселенной",
    "support, lower-income": "поддержка, низкий доход", "support,<br>higher-income": "поддержка,<br>высокий доход",
    "number of unknowns": "число неизвестных", "grid cells": "ячеек сетки",
    "<b>Cells needed</b> (100 per unknown), time at a billion per second": "<b>Нужно ячеек</b> (по 100 на неизвестное), время при миллиарде в секунду",
    "<b>The grid explodes when a model has more unknowns</b>": "<b>Сетка «взрывается», когда неизвестных больше</b>",
    "Each extra unknown multiplies the number of cells by 100.<br>Real models often have dozens of unknowns.":
        "Каждое новое неизвестное умножает число ячеек на 100.<br>В реальных моделях неизвестных часто десятки (проклятие размерности).",
    "useful share": "полезная доля",
    "<b>Useful share</b> of the box as the number of unknowns grows": "<b>Полезная доля</b> куба при росте числа неизвестных",
    "<b>In many dimensions almost the whole box is empty</b>": "<b>В большой размерности куб почти весь пуст</b>",
    "Points placed blindly (at random or on a grid) almost never land<br>where the posterior is.":
        "Точки, поставленные вслепую (случайно или по сетке), почти никогда<br>не попадают туда, где апостериорное распределение.",
    "number of darts": "число дротиков", "<b>Darts on a square</b>": "<b>Дротики в квадрате</b>", "<b>Estimate of π</b>": "<b>Оценка π</b>",
    "<b>Monte Carlo: let randomness compute an area</b>": "<b>Монте-Карло: пусть площадь посчитает случайность</b>",
    "The share of darts inside the quarter circle estimates its area, π/4.<br>Move the slider: the wobble shrinks as darts accumulate.":
        "Доля дротиков внутри четверти круга оценивает её площадь, π/4.<br>Двигайте ползунок: с ростом числа дротиков колебания уменьшаются.",
    "darts: ": "дротиков: ",
    "<b>A pile of draws can stand in for the distribution</b>": "<b>Куча сэмплов может заменить распределение</b>",
    "Each draw is one plausible value of support. Questions become counting:<br>what share of the draws is above 50% (orange)?":
        "Каждый сэмпл — одно правдоподобное значение поддержки. Вопросы превращаются в подсчёт:<br>какая доля сэмплов выше 50% (оранжевые)?",
    "height": "высота",
    "<b>The kept darts are draws from the posterior</b>": "<b>Оставленные дротики — сэмплы из апостериорного</b>",
    "<b>Rejection sampling (von Neumann, 1951)</b>": "<b>Метод отклонения (фон Нейман, 1951)</b>",
    "With twenty unknowns almost every dart would be rejected.": "При двадцати неизвестных отклонялся бы почти каждый дротик.",
    "<b>Old way</b>: spread points evenly, weight each by its height<br>(most points are almost weightless)":
        "<b>Старый способ</b>: расставить точки равномерно и взвесить по высоте<br>(у большинства точек вес почти нулевой)",
    "<b>Metropolis et al. (1953)</b>: place points where the probability is<br>(every point counts equally)":
        "<b>Метрополис и др. (1953)</b>: ставить точки туда, где вероятность<br>(каждая точка весит одинаково)",
    "<b>The key idea of 1953</b>": "<b>Главная идея 1953 года</b>",
    "Instead of weighting points by probability,<br>choose points with that probability.":
        "Вместо того чтобы взвешивать точки по вероятности,<br>выбирать точки с этой вероятностью.",
    "<b>The walker only compares two heights</b>": "<b>Путник сравнивает только две высоты</b>",
    "Multiply the curve by any number: the picture and the ratio stay the same;<br>only the axis numbers change.":
        "Умножьте кривую на любое число: картинка и отношение не меняются;<br>меняются только числа на оси.",
    "as computed": "как есть", "here": "здесь", "proposed": "предложено",
    "island (1 = smallest, 7 = largest)": "остров (1 — самый маленький, 7 — самый большой)",
    "share of nights": "доля ночей",
    "<b>Seven islands in a ring</b> (circle size = population)": "<b>Семь островов по кругу</b> (размер круга = население)",
    "<b>Share of nights spent on each island</b>": "<b>Доля ночей на каждом острове</b>",
    "<b>King Markov's rule produces the target distribution</b>": "<b>Правило короля Маркова даёт целевое распределение</b>",
    "Press ▶ Play. Blue bars: where the king actually slept.<br>Orange marks: each island's share of the population.":
        "Нажмите ▶ Пуск. Синие столбики: где король действительно ночевал.<br>Оранжевые метки: доля населения каждого острова.",
    "<b>Wherever the king starts, the long run is the same</b>": "<b>Откуда бы король ни начал, в итоге всё одинаково</b>",
    "Share of nights on the largest island, for three starting islands.": "Доля ночей на самом большом острове при трёх разных стартах.",
    "night": "ночь",
    "<b>Why it works: balanced traffic</b>": "<b>Почему это работает: баланс потоков</b>",
    "The king proposes the other island half of the time. He always moves to a<br>bigger island and moves to a smaller one with probability 2/6.":
        "Король предлагает другой остров в половине случаев. На больший остров он<br>переезжает всегда, на меньший — с вероятностью 2/6.",
    "More travellers go A → B than back:<br>visits to B will grow.": "Из A в B переезжают чаще, чем обратно:<br>доля ночей на B будет расти.",
    "Traffic is balanced: the shares no longer change.<br>This is the target distribution.":
        "Потоки уравновешены: доли больше не меняются.<br>Это и есть целевое распределение.",
    "uphill: move": "в гору: идём", "downhill: coin says stay": "под гору: монетка — остаться",
    "downhill: coin says move": "под гору: монетка — идти",
    "step": "шаг", "support": "поддержка",
    "<b>The hill</b> (posterior shape) and the walker": "<b>Холм</b> (форма апостериорного) и путник",
    "<b>Where the walker has been</b> vs the exact posterior (dashed)": "<b>Где побывал путник</b> и точное апостериорное (пунктир)",
    "<b>Trace</b>: position at each step": "<b>Трасса</b>: положение на каждом шаге",
    "<b>The Metropolis walker on our survey posterior</b>": "<b>Путник Метрополиса на апостериорном распределении опроса</b>",
    "Press ▶ Play. Uphill: always move. Downhill: move with probability = height ratio.<br>Rejected: stay and count this place again.":
        "Нажмите ▶ Пуск. В гору — идём всегда. Под гору — с вероятностью, равной отношению высот.<br>Отказ — остаёмся и считаем это место ещё раз.",
    "start": "старт", "unknown 1": "неизвестное 1", "unknown 2": "неизвестное 2",
    "<b>Gibbs sampling (1984): one unknown at a time</b>": "<b>Сэмплирование по Гиббсу (1984): по одному неизвестному</b>",
    "Fix one unknown, draw the other from its exact distribution,<br>then swap: the path is a staircase.":
        "Фиксируем одно неизвестное, второе сэмплируем из его точного распределения,<br>потом меняем местами: путь — лесенка.",
    "<b>Why modern software uses Hamiltonian Monte Carlo</b>": "<b>Почему современные программы используют гамильтонов Монте-Карло</b>",
    "In a narrow valley small random steps cover only part of it.<br>The skateboarder uses the slope and covers all of it.":
        "В узкой долине мелкие случайные шаги покрывают только её часть.<br>Скейтбордист использует уклон и проходит её целиком.",
    # timeline (labels and hover stories)
    "<b>Two roads around the hard integral, 1763 – today</b>": "<b>Две дороги в обход трудного интеграла: с 1763 года до наших дней</b>",
    "One event per row. Hover over (or tap) a point for the story.": "Одно событие в строке. Наведите (или нажмите) на точку, чтобы прочитать историю.",
    "<b>Simulate</b>": "<b>Сэмплировать</b>", "<b>Approximate</b>": "<b>Аппроксимировать</b>",
    "<b>Computers &<br>software</b>": "<b>Компьютеры<br>и программы</b>",
    "<i>Los Alamos</i>": "<i>Лос-Аламос</i>", "<i>MCMC revolution</i>": "<i>MCMC-революция</i>",
    "Bayes & Price": "Байес и Прайс", "Laplace": "Лаплас", "Buffon's needle": "Игла Бюффона",
    "'Student' (Gosset)": "«Стьюдент» (Госсет)", "Fermi (1930s)": "Ферми (1930-е)", "Ulam & von Neumann": "Улам и фон Нейман",
    "'Monte Carlo' named": "Название «Монте-Карло»", "rejection sampling": "метод отклонения",
    "Kullback–Leibler": "Кульбак–Лейблер", "Metropolis et al.": "Метрополис и др.", "Hastings": "Гастингс",
    "Gibbs sampler": "Сэмплирование по Гиббсу", "Tierney & Kadane": "Тирни и Кадейн", "Hamiltonian MC": "Гамильтонов MC",
    "Gelfand & Smith": "Гельфанд и Смит", "R-hat (Gelman & Rubin)": "R-hat (Гельман и Рубин)", "Neal": "Нил",
    "Variational inference": "Вариационный вывод", "new R-hat": "новый R-hat", "modern PyMC": "современный PyMC",
    "Bayes's essay (published by Price): the area under the posterior curve is the hard part; they could only bound it.":
        "Эссе Байеса (опубликовал Прайс): трудная часть — площадь под апостериорной кривой; её удалось лишь оценить сверху и снизу.",
    "Laplace's method: approximate the area by a bell curve placed at the peak.":
        "Метод Лапласа: заменить площадь площадью под колоколом, поставленным в пик.",
    "Buffon: dropping needles has a probability that involves π, so chance can compute geometry.":
        "Бюффон: вероятность того, что брошенная игла пересечёт линию, содержит π, — случайность может вычислять геометрию.",
    "W. S. Gosset shuffled 3,000 cards with prisoners' measurements to check his t distribution by simulation.":
        "У. Госсет перемешивал 3000 карточек с измерениями заключённых, чтобы проверить своё t-распределение моделированием.",
    "Enrico Fermi used hand-computed random sampling for neutron problems, but did not publish the method.":
        "Энрико Ферми вручную использовал случайные выборки в задачах о нейтронах, но метод не опубликовал.",
    "Ulam, playing solitaire while ill, asks: why not estimate the chance of winning by playing many games? With von Neumann he turns it into a method for neutron physics.":
        "Улам, раскладывая пасьянс во время болезни, спрашивает: почему бы не оценить шанс выигрыша, просто сыграв много партий? Вместе с фон Нейманом он превращает это в метод нейтронной физики.",
    "One of the first electronic computers; first Monte Carlo runs in 1948 (programmed by Klára Dán von Neumann with Nicholas Metropolis).":
        "Один из первых электронных компьютеров; первые расчёты Монте-Карло в 1948 году (программировали Клара Дан фон Нейман и Николас Метрополис).",
    "Metropolis & Ulam publish 'The Monte Carlo Method'. Metropolis suggested the name, after the casino.":
        "Метрополис и Улам публикуют статью «Метод Монте-Карло». Название, в честь казино, предложил Метрополис.",
    "Von Neumann: throw darts under a box and keep those below the curve.":
        "Фон Нейман: бросать дротики в прямоугольник и оставлять те, что под кривой.",
    "A measure of distance between distributions, later the heart of variational inference.":
        "Мера расхождения между распределениями — позже основа вариационного вывода.",
    "The Los Alamos computer on which the Metropolis algorithm was first run.":
        "Компьютер в Лос-Аламосе, на котором впервые запустили алгоритм Метрополиса.",
    "Metropolis, Arianna & Marshall Rosenbluth, Augusta & Edward Teller: the walker. MCMC is born.":
        "Метрополис, Арианна и Маршалл Розенблют, Августа и Эдвард Теллер: путник. Рождение MCMC.",
    "W. K. Hastings generalises the walker to lopsided proposals: Metropolis–Hastings.":
        "У. К. Гастингс обобщает путника на несимметричные предложения: алгоритм Метрополиса–Гастингса.",
    "Geman & Geman restore noisy images by updating one unknown at a time.":
        "Братья Джеман восстанавливают зашумлённые изображения, обновляя по одному неизвестному за раз.",
    "Accurate Laplace approximations for posterior summaries.": "Точные лапласовские аппроксимации для апостериорных характеристик.",
    "Duane, Kennedy, Pendleton & Roweth: momentum and slope let the walker glide (particle physics).":
        "Дуэйн, Кеннеди, Пендлтон и Роуэт: импульс и уклон позволяют путнику скользить (физика частиц).",
    "The BUGS project (Cambridge) lets applied researchers write a model and get Gibbs samples.":
        "Проект BUGS (Кембридж): прикладной исследователь записывает модель и получает сэмплы по Гиббсу.",
    "Sampling solves everyday Bayesian problems: the 'MCMC revolution' begins.":
        "Сэмплирование решает повседневные байесовские задачи: начинается «MCMC-революция».",
    "Run several chains and check whether they agree.": "Запускать несколько цепей и проверять, согласны ли они.",
    "Radford Neal brings Hamiltonian Monte Carlo into statistics and machine learning.":
        "Рэдфорд Нил приносит гамильтонов Монте-Карло в статистику и машинное обучение.",
    "Jordan, Ghahramani, Jaakkola & Saul: approximate the posterior by the closest simple distribution, found by optimisation.":
        "Джордан, Гахрамани, Яаккола и Сол: заменить апостериорное ближайшим простым распределением, найденным оптимизацией.",
    "PyMC begins as a Python library for Bayesian modelling.": "Появляется PyMC — библиотека Python для байесовского моделирования.",
    "Rue, Martino & Chopin: fast nested Laplace approximations.": "Рю, Мартино и Шопен: быстрые вложенные лапласовские аппроксимации.",
    "Automatic Hamiltonian Monte Carlo for any differentiable model.": "Автоматический гамильтонов Монте-Карло для любой дифференцируемой модели.",
    "Hoffman & Gelman: the No-U-Turn Sampler decides by itself how far to glide.":
        "Хоффман и Гельман: сэмплер NUTS («без разворотов») сам решает, как далеко скользить.",
    "NUTS and automatic variational inference in Python.": "NUTS и автоматический вариационный вывод в Python.",
    "Kucukelbir et al.: automatic differentiation variational inference (pm.fit in PyMC).":
        "Кучукельбир и др.: автоматический вариационный вывод с автоматическим дифференцированием (pm.fit в PyMC).",
    "Vehtari, Gelman, Simpson, Carpenter & Bürkner: today's default convergence checks.":
        "Вехтари, Гельман, Симпсон, Карпентер и Бюркнер: современные стандартные проверки сходимости.",
    "PyMC rewritten (versions 4 and later): the library used in this course.":
        "PyMC переписан (версии 4 и новее) — библиотека, которой мы пользуемся в курсе.",
    # ---------- Lesson 6 ----------
    "<b>Goldilocks: the size of the walker's steps</b>": "<b>Не слишком мало и не слишком много: размер шага путника</b>",
    "Left: the trace. Right: the histogram of the 1,000 steps vs the exact posterior.<br>A high acceptance rate is not the goal: look at how many draws the chain is worth.":
        "Слева — трасса. Справа — гистограмма 1000 шагов и точное апостериорное.<br>Цель — не высокая доля принятых шагов: смотрите, сколько независимых сэмплов стоит цепь.",
    "all steps": "все шаги", "after warm-up": "после разогрева", "exact posterior": "точное апостериорное",
    "<b>Trace</b>: the walker starts far away, at 5%": "<b>Трасса</b>: путник стартует далеко, с 5%",
    "<b>Histogram</b>: with and without the warm-up steps": "<b>Гистограмма</b>: с шагами разогрева и без них",
    "warm-up: thrown away": "разогрев: отбрасываем",
    "<b>Warm-up: the walker first has to find the hill</b>": "<b>Разогрев: сначала путнику надо найти холм</b>",
    "PyMC uses these first steps to find the hill and tune its step size,<br>then discards them.":
        "PyMC использует первые шаги, чтобы найти холм и настроить размер шага,<br>а потом отбрасывает их.",
    "step (after warm-up)": "шаг (после разогрева)", "<b>Each chain's histogram</b>": "<b>Гистограмма каждой цепи</b>",
    "<b>Several walkers: do they tell the same story?</b>": "<b>Несколько путников: рассказывают ли они одно и то же?</b>",
    "R-hat compares the chains with each other. Course rule: R-hat ≤ 1.01.<br>“Not converged”: tiny steps from 20%, 40%, 70% and 90%.":
        "R-hat сравнивает цепи между собой. Правило курса: R-hat ≤ 1.01.<br>«Не сошлись»: крошечные шаги из 20%, 40%, 70% и 90%.",
    "Healthy fit": "Здоровая подгонка", "Not converged": "Не сошлись",
    "draw": "сэмпл", "distance between draws (lag)": "расстояние между сэмплами (лаг)", "similarity": "сходство",
    "<b>How similar is a draw to the next ones?</b> (autocorrelation)": "<b>Насколько сэмпл похож на следующие?</b> (автокорреляция)",
    "<b>Sticky draws: the effective sample size (ESS)</b>": "<b>«Липкие» сэмплы: эффективный размер выборки (ESS)</b>",
    "Neighbouring draws from a walker are similar, so 2,000 of them carry less<br>information than 2,000 independent draws. Course rule: ESS ≥ 400.":
        "Соседние сэмплы путника похожи, поэтому 2000 таких сэмплов несут меньше<br>информации, чем 2000 независимых. Правило курса: ESS ≥ 400.",
    "wobble of the average": "колебания среднего", "wobble of the 97.5% point": "колебания 97,5% квантиля",
    "width of the posterior": "ширина апостериорного", "number of draws": "число сэмплов", "spread (log scale)": "разброс (лог. шкала)",
    "<b>Wobble</b>: how much the estimate changes from run to run": "<b>Колебания</b>: насколько оценка меняется от запуска к запуску",
    "<b>More draws remove wobble, not uncertainty</b>": "<b>Больше сэмплов — меньше колебаний, но не меньше неопределённости</b>",
    "Monte Carlo standard error (MCSE) measures the wobble.<br>Tail quantities, such as interval ends, wobble more.":
        "Стандартная ошибка Монте-Карло (MCSE) измеряет колебания.<br>Хвостовые величины, например концы интервалов, колеблются сильнее.",
    "true posterior": "истинное апостериорное", "chain(s)": "цепь(и)",
    "effect (only its size, not its sign, is identified)": "эффект (определяется только его величина, но не знак)",
    "<b>One chain</b>: it finds one hill and looks perfectly healthy": "<b>Одна цепь</b>: находит один холм и выглядит совершенно здоровой",
    "<b>Coverage: does the sample cover the whole posterior?</b>": "<b>Покрытие: покрывает ли выборка всё апостериорное?</b>",
    "A single chain can miss a whole region and still look fine.<br>Chains started in different places reveal it.":
        "Одна цепь может пропустить целую область и выглядеть нормально.<br>Цепи, запущенные из разных мест, это выявляют.",
    "gap (lower-income minus higher-income)": "разрыв (низкий доход минус высокий)",
    "<b>Lower-income support</b>": "<b>Поддержка: низкий доход</b>", "<b>Higher-income support</b>": "<b>Поддержка: высокий доход</b>",
    "<b>PyMC reproduces the Lesson 4 answer</b>": "<b>PyMC воспроизводит ответ занятия 4</b>",
    "Bars: 4,000 draws from PyMC (NUTS). Dashed lines: the exact answers from Lesson 4.<br>Orange: draws with a gap larger than 5 points.":
        "Столбики: 4000 сэмплов из PyMC (NUTS). Пунктир: точные ответы занятия 4.<br>Оранжевые: сэмплы с разрывом больше 5 п. п.",
    "<b>Gallery: what failed fits look like</b>": "<b>Галерея: как выглядят неудачные подгонки</b>",
    "Click through the cases. Any red light means: stop, do not report the numbers,<br>fix the model or ask for help.":
        "Переключайте случаи. Любой красный огонь означает: стоп, числа не сообщаем,<br>исправляем модель или просим помощи.",
    "Too few draws": "Мало сэмплов", "Chains disagree": "Цепи не согласны", "Divergences": "Расхождения",
    "<b>The funnel</b>: red dots are divergent draws": "<b>Воронка</b>: красные точки — расходящиеся переходы",
    "log(spread)": "log(разброс)", "effect in school 1": "эффект в школе 1", "effect": "эффект",
    "Laplace bell": "колокол Лапласа", "top of the hill (MAP)": "вершина холма (MAP)",
    "<b>Laplace approximation (1774): a bell at the top of the hill</b>": "<b>Лапласовская аппроксимация (1774): колокол на вершине холма</b>",
    "Find the highest point (MAP), match how sharply the hill bends there,<br>and draw a Normal bell.":
        "Находим самую высокую точку (MAP), измеряем, как круто там изгибается холм,<br>и рисуем нормальный колокол.",
    "almost perfect fit": "почти идеальное совпадение",
    "Full survey: 58 of 100": "Полный опрос: 58 из 100", "Tiny pilot: 1 of 10": "Крошечный пилот: 1 из 10",
    "posterior": "апостериорное", "VI bell": "колокол VI", "best possible": "лучше невозможно",
    "optimisation step": "шаг оптимизации",
    "<b>Fit score (ELBO)</b> at each optimisation step": "<b>Качество подгонки (ELBO)</b> на каждом шаге оптимизации",
    "<b>Variational inference: the bell that adjusts itself</b>": "<b>Вариационный вывод: колокол, который подстраивается сам</b>",
    "Press ▶ Play. The optimiser raises the ELBO, which is the same as shrinking<br>the KL distance between the bell and the posterior.":
        "Нажмите ▶ Пуск. Оптимизатор увеличивает ELBO — это то же самое, что уменьшать<br>KL-дивергенцию между колоколом и апостериорным.",
    "draws (MCMC)": "сэмплы (MCMC)", "mean-field VI": "VI среднего поля", "full-rank VI": "полноранговый VI",
    "<b>When the bell is too simple</b>": "<b>Когда колокол слишком прост</b>",
    "Mean-field VI ignores links between unknowns<br>and can be far too sure of itself.":
        "VI среднего поля игнорирует связи между неизвестными<br>и может быть слишком самоуверенным.",
    "overconfident": "излишняя уверенность", "Draws": "Сэмплы",
    "+ mean-field VI": "+ VI среднего поля", "+ full-rank VI": "+ полноранговый VI",
    "MCMC (pm.sample)": "MCMC (pm.sample)", "exact": "точно", "grid": "сетка", "VI (pm.fit)": "VI (pm.fit)",
    "<b>One question, five ways to get the posterior</b>": "<b>Один вопрос — пять способов получить апостериорное</b>",
    "With plenty of data every method agrees. With very little data the<br>shortcuts drift from the exact answer; MCMC does not.":
        "Когда данных много, все методы согласны. Когда данных очень мало,<br>приближения уходят от точного ответа, а MCMC — нет.",
    "Prior": "Априорное", "Posterior": "Апостериорное",
    "±1.96 MCSE Theory": "±1.96 MCSE (теория)", "Sample Size S": "Число сэмплов S", "Running Mean": "Текущее среднее",
    "Monte Carlo SE": "Ошибка Монте-Карло (SE)", "(b) MCSE Rate Decay (1/√S)": "(b) Убывание MCSE (1/√S)",
    "<b>Accepted Draw</b>": "<b>Принятый сэмпл</b>", "Accepted Draw": "Принятый сэмпл",
    "<b>Rejected Draw</b>": "<b>Отклонённый сэмпл</b>", "Rejected Draw": "Отклонённый сэмпл",
    "<b>Bin Range</b>: ": "<b>Интервал</b>: ", "Target p(θ)": "Цель p(θ)", "Candidate Parameter θ": "Кандидат θ",
    "Proposal Height u ~ Uniform(0, M)": "Высота u ~ Uniform(0, M)", "Accepted Parameter θ": "Принятое значение θ",
    "Empirical Density": "Эмпирическая плотность",
    "<b>(a) The Rejection Geometry (Proposals & Boundary)</b>": "<b>(a) Геометрия метода отклонения</b>",
    "<b>(b) Target Density Recovery (Accepted Draws)</b>": "<b>(b) Восстановленная плотность</b>",
    "State 1 (Urban)": "Состояние 1 (центр)", "State 2 (Suburban)": "Состояние 2 (ближний пригород)",
    "State 3 (Rural)": "Состояние 3 (дальний пригород)", "Step n": "Шаг n", "State Probability": "Вероятность состояния",
    "Iteration Index": "Номер итерации", "Sample Value": "Значение", "Lag k": "Лаг k",
    "Autocorrelation ACF(k)": "Автокорреляция ACF(k)", "(a) AR(1) Simulated Chains": "(a) Смоделированные цепи AR(1)",
    "(b) Autocorrelation ACF(k)": "(b) Автокорреляция ACF(k)", "Iteration": "Итерация", "State θ": "Состояние θ",
    "Parameter θ₁": "Параметр θ₁", "Parameter θ₂": "Параметр θ₂",
    "0.1 µs": "0,1 мкс", "10 µs": "10 мкс", "1 ms": "1 мс", "10 s": "10 с", "1930s": "1930-е",
}

_LIT = sorted(LITERAL.items(), key=lambda kv: -len(kv[0]))
_WHOLE = {"Prior", "Posterior", "Likelihood", "Density", "density", "Count", "Average", "step", "support", "night", "draw", "start",
          "here", "proposed", "as computed", "exact", "grid", "posterior", "effect", "height", "similarity", "Draws", "Laplace", "Neal",
          "Hastings", "Iteration", "0.1 µs", "10 µs", "1 ms", "10 s", "1930s"}


def translate_text(s: str) -> str:
    """Translate one visible string."""
    if s in LITERAL and s in _WHOLE:
        return LITERAL[s]
    for pat, rep in FIG_RULES.get(CURRENT_FIG["name"], []):
        s = re.sub(pat, rep, s)
    for pat, rep in REGEX:
        s = re.sub(pat, rep, s)
    for en, ru in _LIT:
        if en in _WHOLE:
            continue  # single short words are only translated as whole strings
        if en in s:
            s = s.replace(en, ru)
    return s


STR_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


def _encode_like_plotly(s: str) -> str:
    out = json.dumps(s, ensure_ascii=False)[1:-1]
    return out.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026").replace("/", "\\u002f")


TEXT_KEYS_RE = re.compile(
    r'("(?:text|name|label|hovertemplate|prefix|suffix|hovertext|ticktext|x|y)"\s*:\s*)(\[(?:[^\[\]"]|"(?:[^"\\]|\\.)*")*\]|"(?:[^"\\]|\\.)*")')


def translate_json_strings(code: str) -> str:
    """Translate only the values of text-bearing keys in Plotly JSON (never keys or enums)."""
    def one(m):
        raw = m.group(1)
        try:
            val = json.loads('"' + raw + '"')
        except Exception:
            return m.group(0)
        new = translate_text(val)
        return m.group(0) if new == val else '"' + _encode_like_plotly(new) + '"'

    def pair(m):
        return m.group(1) + STR_RE.sub(one, m.group(2))

    return TEXT_KEYS_RE.sub(pair, code)


def translate_quoted_literals(code: str) -> str:
    """Decode double-quoted JS/JSON literals (not keys), translate, re-encode."""
    def one(m):
        if code[m.end():m.end() + 40].lstrip().startswith(":"):
            return m.group(0)  # an object key
        raw = m.group(1)
        try:
            val = json.loads('"' + raw + '"')
        except Exception:
            return m.group(0)
        new = translate_text(val)
        if new == val:
            return m.group(0)
        return '"' + json.dumps(new, ensure_ascii=False)[1:-1].replace("</", "<\\/") + '"'

    return re.sub(r'"((?:[^"\\\n]|\\.)*)"', one, code)


def translate_raw(code: str) -> str:
    """Phrase replacement for hand-written JavaScript and HTML text."""
    code = translate_quoted_literals(code)
    for pat, rep in REGEX:
        code = re.sub(pat, rep, code, flags=re.M)
    for en, ru in _LIT:
        if en in _WHOLE:
            continue
        code = code.replace(en, ru)
    # whole-word labels that hand-written code keeps in quotes (never object keys)
    for en in _WHOLE:
        code = re.sub(r"(['\"`])" + re.escape(en) + r"\1(?!\s*:)", lambda m, en=en: m.group(1) + LITERAL[en] + m.group(1), code)
        code = code.replace(f">{en}<", f">{LITERAL[en]}<")
    return code


def translate_html(html: str) -> str:
    parts = []
    pos = 0
    for m in re.finditer(r"(<script[^>]*>)(.*?)(</script>)", html, flags=re.S):
        parts.append(translate_raw_text_nodes(html[pos:m.start()]))
        body = m.group(2)
        if "plotly.js v" in body[:3000]:
            new = body
        elif "PLOTLYENV" in body[:400]:
            new = translate_json_strings(body)
        else:
            new = translate_raw(body)
        parts.append(m.group(1) + new + m.group(3))
        pos = m.end()
    parts.append(translate_raw_text_nodes(html[pos:]))
    out = "".join(parts)
    return out.replace('<html lang="en">', '<html lang="ru">')


def translate_raw_text_nodes(fragment: str) -> str:
    def repl(m):
        return ">" + translate_text(m.group(1)) + "<"
    fragment = re.sub(r"<title>(.*?)</title>", lambda m: "<title>" + translate_text(m.group(1)) + "</title>", fragment)
    return re.sub(r">([^<>]*[A-Za-z][^<>]*)<", repl, fragment)


ALLOW = {"Beta", "Normal", "PyMC", "PyMC3", "NUTS", "HMC", "MCMC", "ESS", "MCSE", "R-hat", "ELBO", "KL", "VI", "ADVI", "MAP", "INLA", "BUGS",
         "Stan", "ENIAC", "MANIAC", "ETI", "HDI", "SD", "CI", "pm", "sample", "fit", "find", "Python", "local", "MathJaxConfig", "font",
         "size", "color", "gray", "span", "style", "sup", "extra", "monty", "hall", "btn", "group", "button", "active", "status", "box",
         "title", "prior", "like", "post", "plotly", "white", "plot", "lines", "dash", "step", "goat", "car", "door", "max", "min"}


def leftovers(name: str) -> list[str]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from importlib import import_module  # noqa: F401
    html = (STATIC / f"{name}_ru.html").read_text(encoding="utf-8")
    found = []
    for m in re.finditer(r"<script[^>]*>(.*?)</script>", html, flags=re.S):
        body = m.group(1)
        if "plotly.js v" in body[:3000]:
            continue
        if "PLOTLYENV" in body[:400]:
            for k in re.finditer(r'"(?:text|name|label|hovertemplate|prefix|title|hovertext|ticktext)"\s*:\s*(\[[^\]]*\]|"(?:[^"\\]|\\.)*")', body):
                found += [json.loads('"' + x + '"') for x in STR_RE.findall(k.group(1))]
        else:
            found += re.findall(r"'((?:[^'\\\n]|\\.){3,})'", body)
    body_html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.S)
    body_html = re.sub(r"<style[^>]*>.*?</style>", " ", body_html, flags=re.S)
    found += [t.strip() for t in re.findall(r">([^<>]{2,})<", body_html) if t.strip()]
    bad = []
    for s in dict.fromkeys(found):
        words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", re.sub(r"<[^>]+>|%\{[^}]*\}|&[a-z]+;", " ", s))
        if any(w not in ALLOW for w in words):
            bad.append(s)
    return bad


def main(check: bool) -> None:
    for name in FIGURES:
        CURRENT_FIG["name"] = name
        src = STATIC / f"{name}.html"
        if not src.exists():
            print("missing", name)
            continue
        (STATIC / f"{name}_ru.html").write_text(translate_html(src.read_text(encoding="utf-8")), encoding="utf-8")
        if check:
            bad = leftovers(name)
            if bad:
                print(f"--- {name}: {len(bad)} strings still contain English")
                for b in bad[:25]:
                    print("    ", b[:160])
    print("translated", len(FIGURES), "figures")


if __name__ == "__main__":
    main(check="--check" in sys.argv)
