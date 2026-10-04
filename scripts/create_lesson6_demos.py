"""Interactive figures for Lesson 6: running a sampler, convergence, trust checks and approximations.

Run from the book root with the pymc_env environment:

    conda run -n pymc_env python scripts/create_lesson6_demos.py

Writes standalone Plotly pages to ``_static/lesson6_*.html``. Set ``PREVIEW_DIR``
(and optionally ``PREVIEW_WIDTH``, e.g. 700 = the book's content column) to also
write PNG snapshots. PyMC models are small and run in a few seconds each.

Layout rule: every figure must stay readable in the ~650-750 px book column, so
panels are stacked vertically, subtitles are at most two short lines, buttons sit
in their own row under the title, and the verdict of a case is shown in the title
of the top panel.

Running example (Lesson 4): 58 of 100 lower-income and 40 of 100 higher-income
households support a housing subsidy; Beta(2, 2) priors. A "tiny pilot" with 1
supporter out of 10 households shows where approximations fail.
"""

from __future__ import annotations

import logging
import os
import warnings

import arviz as az
import numpy as np
import plotly.graph_objects as go
import pymc as pm
from plotly.subplots import make_subplots
from scipy import stats
from scipy.special import expit, logit

warnings.filterwarnings("ignore")
logging.getLogger("pymc").setLevel(logging.ERROR)
os.makedirs("_static", exist_ok=True)
PREVIEW_DIR = os.environ.get("PREVIEW_DIR")

BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
GREY = "#a0aec0"
RED = "#e53e3e"
GREEN = "#38a169"
PURPLE = "#805ad5"
DARK = "#2d3748"
CHAIN_COLORS = [BLUE, ORANGE, GREEN, PURPLE]

POST = stats.beta(60, 44)
XZ = np.linspace(0.30, 0.85, 500)


def title(main: str, sub: str = "") -> dict:
    text = f"<b>{main}</b>" if not sub else f"<b>{main}</b><br><span style='font-size:12px;color:gray'>{sub}</span>"
    return dict(text=text, x=0.01, xanchor="left", y=1.0, yref="container", yanchor="top", pad=dict(t=16), font=dict(size=16))


def save(fig: go.Figure, name: str, width: int = 900, height: int | None = None) -> None:
    fig.write_html(f"_static/{name}.html", include_plotlyjs="cdn", auto_play=False, config={"displayModeBar": False, "responsive": True})
    if PREVIEW_DIR:
        os.makedirs(PREVIEW_DIR, exist_ok=True)
        w = int(os.environ.get("PREVIEW_WIDTH", width))
        if fig.layout.width:
            w = min(w, int(fig.layout.width))
        fig.write_image(f"{PREVIEW_DIR}/{name}.png", width=w, height=height or fig.layout.height or 450)
    print("wrote", name)


def layout(fig: go.Figure, height: int, t: int, b: int = 55, l: int = 55, r: int = 20, **kw) -> None:
    fig.update_layout(template="plotly_white", height=height, margin=dict(t=t, b=b, l=l, r=r), **kw)


def subplot_title_dicts(fig: go.Figure, size: int = 13) -> list[dict]:
    out = []
    for a in fig.layout.annotations:
        d = a.to_plotly_json()
        d["font"] = dict(size=size, color=DARK)
        out.append(d)
    fig.layout.annotations = out
    return out


def retitle(base: list[dict], idx: int, text: str, color: str = DARK) -> list[dict]:
    out = [dict(a) for a in base]
    out[idx] = {**out[idx], "text": text, "font": dict(size=13, color=color)}
    return out


def button_row(buttons, height: int, t: int, b: int, px_above: int = 38) -> list[dict]:
    """Buttons in their own row just above the plot area (px_above pixels above it)."""
    y = 1 + px_above / (height - t - b)
    return [dict(type="buttons", direction="right", x=0.0, y=y, xanchor="left", yanchor="bottom", showactive=True,
                 pad=dict(t=0, b=0), font=dict(size=12), buttons=buttons)]


def logpost_survey(t):
    t = np.asarray(t, dtype=float)
    out = np.full(t.shape, -np.inf)
    ok = (t > 0) & (t < 1)
    out[ok] = stats.beta.logpdf(t[ok], 2, 2) + stats.binom.logpmf(58, 100, t[ok])
    return out


def metropolis(logp, start, step, n, seed):
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    cur = start
    lp = logp(np.array(cur))
    acc = 0
    for i in range(n):
        prop = cur + rng.normal(0, step)
        lpp = logp(np.array(prop))
        if np.log(rng.random()) < lpp - lp:
            cur, lp = prop, lpp
            acc += 1
        x[i] = cur
    return x, acc / n


def ess(chains):
    return float(np.asarray(az.ess(np.atleast_2d(np.asarray(chains)))))


def rhat(chains):
    return float(np.asarray(az.rhat(np.asarray(chains))))


def density_line(x, edges):
    c, _ = np.histogram(x, bins=edges)
    w = edges[1] - edges[0]
    return (edges[:-1] + edges[1:]) / 2, c / (len(x) * w)


# ---------------------------------------------------------------------------
# 1. Step size
# ---------------------------------------------------------------------------
def step_size():
    cases = [("Tiny steps", 0.003), ("Huge steps", 0.6), ("Good steps", 0.07)]
    runs = [metropolis(logpost_survey, 0.45, step, 1000, seed=10 + r) for r, (_, step) in enumerate(cases, start=1)]
    titles = []
    for (name, step), (x, acc) in zip(cases, runs):
        titles += [f"<b>{name}</b> ({step}): accepted {acc:.0%}, worth ≈ <b>{ess(x[None, :]):,.0f}</b> independent draws", ""]
    fig = make_subplots(rows=3, cols=2, column_widths=[0.66, 0.34], vertical_spacing=0.11, horizontal_spacing=0.04,
                        subplot_titles=titles)
    for a in fig.layout.annotations:
        a.update(font=dict(size=12, color=DARK), x=0.0, xanchor="left")
    edges = np.arange(0.30, 0.8501, 0.0125)
    for r, (x, _) in enumerate(runs, start=1):
        col = CHAIN_COLORS[r - 1]
        fig.add_trace(go.Scatter(y=x, mode="lines", line=dict(color=col, width=1.2)), r, 1)
        m, d = density_line(x, edges)
        fig.add_trace(go.Bar(y=m, x=d, orientation="h", width=0.012, marker_color=col), r, 2)
        fig.add_trace(go.Scatter(y=XZ, x=POST.pdf(XZ), line=dict(color=DARK, width=1.5, dash="dash"), hoverinfo="skip"), r, 2)
        fig.update_yaxes(range=[0.35, 0.8], tickformat=".0%", row=r, col=1)
        fig.update_yaxes(range=[0.35, 0.8], showticklabels=False, row=r, col=2)
        fig.update_xaxes(showticklabels=False, row=r, col=2)
    fig.update_xaxes(title_text="step", row=3, col=1)
    layout(fig, 720, t=100, showlegend=False, bargap=0,
           title=title("Goldilocks: the size of the walker's steps",
                       "Left: the trace. Right: the histogram of the 1,000 steps vs the exact posterior.<br>"
                       "A high acceptance rate is not the goal: look at how many draws the chain is worth."))
    save(fig, "lesson6_step_size", width=700, height=720)


# ---------------------------------------------------------------------------
# 2. Warm-up
# ---------------------------------------------------------------------------
def warmup():
    x, _ = metropolis(logpost_survey, 0.05, 0.03, 1200, seed=7)
    W = 200
    edges = np.arange(0.0, 0.8501, 0.0125)
    fig = make_subplots(rows=2, cols=1, row_heights=[0.5, 0.5], vertical_spacing=0.16, subplot_titles=[
        "<b>Trace</b>: the walker starts far away, at 5%", "<b>Histogram</b>: with and without the warm-up steps"])
    subplot_title_dicts(fig)
    fig.add_vrect(x0=0, x1=W, fillcolor="rgba(229,62,62,0.10)", line_width=0, row=1, col=1,
                  annotation_text="warm-up: thrown away", annotation_position="bottom right", annotation_font_color=RED)
    fig.add_trace(go.Scatter(y=x, mode="lines", line=dict(color=BLUE, width=1.3), showlegend=False), 1, 1)
    m, d_all = density_line(x, edges)
    _, d_keep = density_line(x[W:], edges)
    fig.add_trace(go.Scatter(x=m, y=d_all, mode="lines", line=dict(color=RED, width=2, shape="hvh"), name="all steps"), 2, 1)
    fig.add_trace(go.Scatter(x=m, y=d_keep, mode="lines", line=dict(color=BLUE, width=2.5, shape="hvh"), name="after warm-up"), 2, 1)
    fig.add_trace(go.Scatter(x=XZ, y=POST.pdf(XZ), line=dict(color=DARK, width=1.5, dash="dash"), name="exact posterior"), 2, 1)
    fig.update_yaxes(range=[0, 0.85], tickformat=".0%", title_text="support", row=1, col=1)
    fig.update_xaxes(title_text="step", row=1, col=1)
    fig.update_xaxes(range=[0, 0.85], tickformat=".0%", title_text="support", row=2, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    layout(fig, 640, t=100, b=90, legend=dict(orientation="h", x=0, y=-0.16, font=dict(size=12)),
           title=title("Warm-up: the walker first has to find the hill",
                       "PyMC uses these first steps to find the hill and tune its step size,<br>then discards them."))
    save(fig, "lesson6_warmup", width=700)


# ---------------------------------------------------------------------------
# 3. Four chains and R-hat
# ---------------------------------------------------------------------------
def four_chains():
    starts = [0.2, 0.4, 0.7, 0.9]
    good = np.array([metropolis(logpost_survey, s, 0.07, 1200, seed=20 + i)[0][200:] for i, s in enumerate(starts)])
    bad = np.array([metropolis(logpost_survey, s, 0.002, 400, seed=30 + i)[0] for i, s in enumerate(starts)])
    edges = np.arange(0.0, 1.0001, 0.0125)
    fig = make_subplots(rows=2, cols=1, row_heights=[0.55, 0.45], vertical_spacing=0.15,
                        subplot_titles=["trace", "<b>Each chain's histogram</b>"])
    base = subplot_title_dicts(fig)
    for chains in (good, bad):
        for i, ch in enumerate(chains):
            fig.add_trace(go.Scatter(y=ch, mode="lines", line=dict(color=CHAIN_COLORS[i], width=1), opacity=0.8, visible=chains is good), 1, 1)
        for i, ch in enumerate(chains):
            m, d = density_line(ch, edges)
            fig.add_trace(go.Scatter(x=m, y=d, mode="lines", line=dict(color=CHAIN_COLORS[i], width=2, shape="hvh"),
                                     visible=chains is good), 2, 1)

    def verdict(chains, ok):
        r, e = rhat(chains), ess(chains)
        txt = (f"<b>Four chains</b>: R-hat = <b>{r:.2f}</b>, ESS = {e:,.0f} → "
               + ("🟢 they agree" if ok else "🔴 they disagree: do not interpret"))
        return retitle(base, 0, txt, GREEN if ok else RED)

    a_good, a_bad = verdict(good, True), verdict(bad, False)
    n = 8
    H, T, B = 640, 135, 55
    layout(fig, H, t=T, b=B, showlegend=False, annotations=a_good,
           title=title("Several walkers: do they tell the same story?",
                       "R-hat compares the chains with each other. Course rule: R-hat ≤ 1.01.<br>"
                       "“Not converged”: tiny steps from 20%, 40%, 70% and 90%."),
           yaxis=dict(title="support", tickformat=".0%", range=[0, 1]), xaxis=dict(title="step (after warm-up)"),
           xaxis2=dict(tickformat=".0%", range=[0, 1], title="support"), yaxis2=dict(showticklabels=False),
           updatemenus=button_row([
               dict(label="Healthy fit", method="update", args=[{"visible": [True] * n + [False] * n}, {"annotations": a_good}]),
               dict(label="Not converged", method="update", args=[{"visible": [False] * n + [True] * n}, {"annotations": a_bad}]),
           ], H, T, B))
    save(fig, "lesson6_four_chains", width=700)


# ---------------------------------------------------------------------------
# 4. Sticky draws and ESS
# ---------------------------------------------------------------------------
def acf(x, lags=40):
    x = x - x.mean()
    v = (x * x).mean()
    return np.array([1.0] + [(x[:-k] * x[k:]).mean() / v for k in range(1, lags + 1)])


def sticky_ess():
    cases = [("Very sticky", 0.005), ("Sticky", 0.02), ("Lively", 0.07)]
    fig = make_subplots(rows=2, cols=1, row_heights=[0.5, 0.5], vertical_spacing=0.17,
                        subplot_titles=["trace", "<b>How similar is a draw to the next ones?</b> (autocorrelation)"])
    base = subplot_title_dicts(fig)
    anns = []
    for i, (name, step) in enumerate(cases):
        x, _ = metropolis(logpost_survey, 0.578, step, 2000, seed=40 + i)
        e = ess(x[None, :])
        fig.add_trace(go.Scatter(y=x[:300], mode="lines+markers", marker=dict(size=3), line=dict(color=CHAIN_COLORS[i], width=1),
                                 visible=i == 0), 1, 1)
        fig.add_trace(go.Bar(x=np.arange(41), y=acf(x), marker_color=CHAIN_COLORS[i], visible=i == 0), 2, 1)
        ok = e >= 400
        anns.append(retitle(base, 0, f"<b>First 300 of 2,000 draws</b>: worth ≈ <b>{e:,.0f}</b> independent draws "
                                     + ("🟢" if ok else "🔴 (below 400)"), GREEN if ok else RED))
    buttons = []
    for i, (name, step) in enumerate(cases):
        vis = [False] * 6
        vis[2 * i] = vis[2 * i + 1] = True
        buttons.append(dict(label=f"{name} ({step})", method="update", args=[{"visible": vis}, {"annotations": anns[i]}]))
    H, T, B = 620, 135, 55
    layout(fig, H, t=T, b=B, showlegend=False, annotations=anns[0],
           title=title("Sticky draws: the effective sample size (ESS)",
                       "Neighbouring draws from a walker are similar, so 2,000 of them carry less<br>"
                       "information than 2,000 independent draws. Course rule: ESS ≥ 400."),
           yaxis=dict(range=[0.4, 0.75], tickformat=".0%", title="support"), xaxis=dict(title="draw"),
           xaxis2=dict(title="distance between draws (lag)"), yaxis2=dict(title="similarity", range=[-0.2, 1.05]),
           updatemenus=button_row(buttons, H, T, B))
    save(fig, "lesson6_sticky_ess", width=700)


# ---------------------------------------------------------------------------
# 5. Width vs wobble: Monte Carlo error, and the tails
# ---------------------------------------------------------------------------
def mcse_wobble():
    rng = np.random.default_rng(2026)
    edges = np.arange(0.30, 0.8501, 0.0125)
    fig = make_subplots(rows=2, cols=1, row_heights=[0.45, 0.55], vertical_spacing=0.17, subplot_titles=[
        "<b>Width</b>: 100 vs 10,000 draws have the same spread",
        "<b>Wobble</b>: how much the estimate changes from run to run"])
    subplot_title_dicts(fig)
    for S, col in [(100, ORANGE), (10_000, BLUE)]:
        m, d = density_line(rng.beta(60, 44, S), edges)
        fig.add_trace(go.Scatter(x=m, y=d, mode="lines", line=dict(color=col, width=2.5, shape="hvh"), name=f"{S:,} draws"), 1, 1)
    Ss = np.array([10, 30, 100, 300, 1000, 3000, 10000])
    wob_mean, wob_tail = [], []
    for S in Ss:
        runs = rng.beta(60, 44, size=(300, S))
        wob_mean.append(runs.mean(1).std())
        wob_tail.append(np.quantile(runs, 0.975, axis=1).std())
    fig.add_trace(go.Scatter(x=Ss, y=wob_mean, mode="lines+markers", line=dict(color=BLUE, width=2.5), name="wobble of the average"), 2, 1)
    fig.add_trace(go.Scatter(x=Ss, y=wob_tail, mode="lines+markers", line=dict(color=RED, width=2.5), name="wobble of the 97.5% point"), 2, 1)
    fig.add_trace(go.Scatter(x=[Ss[0], Ss[-1]], y=[POST.std()] * 2, mode="lines+text", line=dict(color=DARK, dash="dash"),
                             text=["", "width of the posterior"], textposition="bottom left", showlegend=False), 2, 1)
    fig.update_xaxes(tickformat=".0%", range=[0.3, 0.85], row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(type="log", title_text="number of draws", tickvals=[10, 100, 1000, 10000], ticktext=["10", "100", "1,000", "10,000"], row=2, col=1)
    fig.update_yaxes(type="log", title_text="spread (log scale)", row=2, col=1)
    layout(fig, 680, t=100, b=110, legend=dict(orientation="h", x=0, y=-0.17, font=dict(size=12)),
           title=title("More draws remove wobble, not uncertainty",
                       "Monte Carlo standard error (MCSE) measures the wobble.<br>Tail quantities, such as interval ends, wobble more."))
    save(fig, "lesson6_mcse_wobble", width=700)


# ---------------------------------------------------------------------------
# 6. Does the sample cover the whole posterior? (two hills)
# ---------------------------------------------------------------------------
Y_TWO_HILLS = np.array([4.6, 3.1, 4.9, 3.8, 4.4, 2.9, 4.1, 3.6, 5.0, 3.6])  # observed values whose mean is about 4


def logpost_two_hills(mu):
    mu = np.asarray(mu, dtype=float)
    return stats.norm.logpdf(mu, 0, 3) + stats.norm.logpdf(Y_TWO_HILLS[:, None] if mu.ndim else Y_TWO_HILLS, mu ** 2, 1).sum(axis=0)


def coverage_two_hills():
    g = np.linspace(-3, 3, 1200)
    lp = np.array([logpost_two_hills(v) for v in g])
    dens = np.exp(lp - lp.max())
    dens /= dens.sum() * (g[1] - g[0])
    one, _ = metropolis(logpost_two_hills, 2.0, 0.15, 2000, seed=1)
    four = np.array([metropolis(logpost_two_hills, s, 0.15, 2000, seed=50 + i)[0] for i, s in enumerate([-2.0, 2.0, -2.0, 2.0])])
    edges = np.arange(-3, 3.0001, 0.05)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14, subplot_titles=[
        "<b>One chain</b>: it finds one hill and looks perfectly healthy",
        f"<b>Four chains from different starts</b>: R-hat = {rhat(four):.2f} 🔴"])
    subplot_title_dicts(fig)
    for r in (1, 2):
        fig.add_trace(go.Scatter(x=g, y=dens, fill="tozeroy", fillcolor="rgba(160,174,192,0.25)", line=dict(color=GREY, width=1.5),
                                 name="true posterior", showlegend=r == 1), r, 1)
    m, d = density_line(one, edges)
    fig.add_trace(go.Scatter(x=m, y=d, mode="lines", line=dict(color=BLUE, width=2.5, shape="hvh"), name="chain(s)"), 1, 1)
    for i, ch in enumerate(four):
        m, d = density_line(ch, edges)
        fig.add_trace(go.Scatter(x=m, y=d, mode="lines", line=dict(color=CHAIN_COLORS[i], width=2, shape="hvh"), showlegend=False), 2, 1)
    fig.update_xaxes(title_text="effect (only its size, not its sign, is identified)", range=[-3, 3], row=2, col=1)
    fig.update_yaxes(showticklabels=False)
    layout(fig, 600, t=100, b=60, legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
           title=title("Coverage: does the sample cover the whole posterior?",
                       "A single chain can miss a whole region and still look fine.<br>Chains started in different places reveal it."))
    save(fig, "lesson6_coverage_two_hills", width=700)


# ---------------------------------------------------------------------------
# PyMC models used by several figures
# ---------------------------------------------------------------------------
def fit_survey(draws=1000, tune=1000, seed=2026):
    with pm.Model():
        theta = pm.Beta("theta", alpha=2, beta=2, shape=2)
        pm.Binomial("y", n=[100, 100], p=theta, observed=[58, 40])
        return pm.sample(draws=draws, tune=tune, chains=4, random_seed=seed, progressbar=False)


def fit_two_hills(seed=2026):
    with pm.Model():
        mu = pm.Normal("mu", 0, 3)
        pm.Normal("y", mu=mu ** 2, sigma=1, observed=Y_TWO_HILLS)
        return pm.sample(1000, tune=1000, chains=4, random_seed=seed, progressbar=False,
                         initvals=[{"mu": -2.0}, {"mu": 2.0}, {"mu": -2.0}, {"mu": 2.0}])


def fit_eight_schools_centered(seed=2026):
    y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
    with pm.Model():
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfCauchy("tau", 5)
        theta = pm.Normal("theta", mu, tau, shape=8)
        pm.Normal("y", theta, sigma, observed=y)
        return pm.sample(1000, tune=1000, chains=4, random_seed=seed, progressbar=False)


def checks(idata, var):
    r = np.atleast_1d(np.asarray(az.rhat(idata, var_names=[var])[var])).max()
    e = np.atleast_1d(np.asarray(az.ess(idata, var_names=[var])[var])).min()
    div = int(np.asarray(idata.sample_stats["diverging"]).sum())
    return float(r), float(e), div


def verdict(r, e, div):
    flags = [("R-hat", f"{r:.2f}", r <= 1.01), ("ESS", f"{e:,.0f}", e >= 400), ("divergences", f"{div}", div == 0)]
    ok_all = all(ok for *_, ok in flags)
    txt = " · ".join(f"{'🟢' if ok else '🔴'} {n} {v}" for n, v, ok in flags)
    return txt + (" → <b>may interpret</b>" if ok_all else " → <b>do not interpret</b>"), ok_all


# ---------------------------------------------------------------------------
# 7. PyMC versus the exact answer
# ---------------------------------------------------------------------------
def pymc_vs_exact(idata):
    post = idata.posterior["theta"].values.reshape(-1, 2)
    gap = post[:, 0] - post[:, 1]
    rng = np.random.default_rng(1)
    exact_gap = rng.beta(60, 44, 10 ** 6) - rng.beta(42, 62, 10 ** 6)
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.17, horizontal_spacing=0.06,
                        specs=[[{}, {}], [{"colspan": 2}, None]],
                        subplot_titles=["<b>Lower-income support</b>", "<b>Higher-income support</b>",
                                        f"<b>Gap between the groups</b>: P(gap > 5 points) = {(gap > 0.05).mean():.3f} (exact {(exact_gap > 0.05).mean():.3f})"])
    subplot_title_dicts(fig)
    edges = np.arange(0.2, 0.8001, 0.0125)
    for c, (k, a, b) in enumerate([(0, 60, 44), (1, 42, 62)], start=1):
        m, d = density_line(post[:, k], edges)
        fig.add_trace(go.Bar(x=m, y=d, width=0.012, marker_color="rgba(43,108,176,0.55)"), 1, c)
        xs = np.linspace(0.2, 0.8, 400)
        fig.add_trace(go.Scatter(x=xs, y=stats.beta.pdf(xs, a, b), line=dict(color=DARK, width=2.5, dash="dash")), 1, c)
        fig.update_xaxes(tickformat=".0%", range=[0.2, 0.8], row=1, col=c)
    ge = np.arange(-0.1, 0.45, 0.0125)
    m, d = density_line(gap, ge)
    fig.add_trace(go.Bar(x=m, y=d, width=0.012, marker_color=[ORANGE if v > 0.05 else GREY for v in m]), 2, 1)
    m2, d2 = density_line(exact_gap, ge)
    fig.add_trace(go.Scatter(x=m2, y=d2, line=dict(color=DARK, width=2.5, dash="dash", shape="spline")), 2, 1)
    fig.update_xaxes(tickformat=".0%", title_text="gap (lower-income minus higher-income)", row=2, col=1)
    fig.update_yaxes(showticklabels=False)
    layout(fig, 600, t=100, showlegend=False,
           title=title("PyMC reproduces the Lesson 4 answer",
                       "Bars: 4,000 draws from PyMC (NUTS). Dashed lines: the exact answers from Lesson 4.<br>"
                       "Orange: draws with a gap larger than 5 points."))
    save(fig, "lesson6_pymc_vs_exact", width=700)


# ---------------------------------------------------------------------------
# 8. Gallery of broken fits
# ---------------------------------------------------------------------------
def broken_fits(idata_ok):
    idata_few = fit_survey(draws=25, tune=1000, seed=3)
    idata_two = fit_two_hills()
    idata_div = fit_eight_schools_centered()
    fig = make_subplots(rows=2, cols=1, row_heights=[0.55, 0.45], vertical_spacing=0.16,
                        subplot_titles=["verdict", "<b>Each chain's histogram</b>"])
    base = subplot_title_dicts(fig)
    states = []
    for name, idata, var, idx in [("Healthy fit", idata_ok, "theta", 0), ("Too few draws", idata_few, "theta", 0),
                                  ("Chains disagree", idata_two, "mu", None)]:
        arr = idata.posterior[var].values
        arr = arr[..., idx] if idx is not None else arr
        txt, ok = verdict(*checks(idata, var))
        start = len(fig.data)
        lo, hi = np.quantile(arr, [0.001, 0.999])
        edges = np.linspace(lo, hi, 40)
        for i in range(arr.shape[0]):
            fig.add_trace(go.Scatter(y=arr[i], mode="lines", line=dict(color=CHAIN_COLORS[i], width=1), opacity=0.8, visible=False), 1, 1)
        for i in range(arr.shape[0]):
            m, d = density_line(arr[i], edges)
            fig.add_trace(go.Scatter(x=m, y=d, mode="lines", line=dict(color=CHAIN_COLORS[i], width=2, shape="hvh"), visible=False), 2, 1)
        label = "support" if var == "theta" else "effect"
        states.append((name, list(range(start, len(fig.data))), retitle(base, 0, txt, GREEN if ok else RED),
                       {"yaxis.title.text": label, "xaxis2.title.text": label, "yaxis2.title.text": "", "yaxis2.showticklabels": False,
                        "annotations": None}))
    # divergences: trace of log(tau) and the funnel
    start = len(fig.data)
    tau = idata_div.posterior["tau"].values
    th0 = idata_div.posterior["theta"].values[..., 0]
    divm = np.asarray(idata_div.sample_stats["diverging"]).astype(bool)
    for i in range(tau.shape[0]):
        fig.add_trace(go.Scatter(y=np.log(tau[i]), mode="lines", line=dict(color=CHAIN_COLORS[i], width=1), opacity=0.8, visible=False), 1, 1)
    fig.add_trace(go.Scatter(x=th0[~divm], y=np.log(tau[~divm]), mode="markers", marker=dict(size=3, color=GREY), visible=False), 2, 1)
    fig.add_trace(go.Scatter(x=th0[divm], y=np.log(tau[divm]), mode="markers", marker=dict(size=6, color=RED), visible=False), 2, 1)
    txt, ok = verdict(*checks(idata_div, "tau"))
    funnel_titles = retitle(retitle(base, 0, txt, RED), 1, "<b>The funnel</b>: red dots are divergent draws")
    states.append(("Divergences", list(range(start, len(fig.data))), funnel_titles,
                   {"yaxis.title.text": "log(spread)", "xaxis2.title.text": "effect in school 1", "yaxis2.title.text": "log(spread)",
                    "yaxis2.showticklabels": True}))
    total = len(fig.data)
    buttons = []
    for name, idxs, anns, axes in states:
        lay = {k: v for k, v in axes.items() if k != "annotations"}
        lay.update({"annotations": anns, "xaxis.autorange": True, "yaxis.autorange": True, "xaxis2.autorange": True, "yaxis2.autorange": True})
        buttons.append(dict(label=name, method="update", args=[{"visible": [i in idxs for i in range(total)]}, lay]))
    for i in states[0][1]:
        fig.data[i].visible = True
    H, T, B = 640, 135, 55
    layout(fig, H, t=T, b=B, showlegend=False, annotations=states[0][2],
           title=title("Gallery: what failed fits look like",
                       "Click through the cases. Any red light means: stop, do not report the numbers,<br>"
                       "fix the model or ask for help."),
           xaxis=dict(title="step"), yaxis=dict(title="support"), xaxis2=dict(title="support"), yaxis2=dict(showticklabels=False),
           updatemenus=button_row(buttons, H, T, B))
    save(fig, "lesson6_broken_fits", width=700)


# ---------------------------------------------------------------------------
# 9. Laplace approximation
# ---------------------------------------------------------------------------
def laplace_parts(a, b):
    mode = (a - 1) / (a + b - 2)
    curv = (a - 1) / mode ** 2 + (b - 1) / (1 - mode) ** 2
    return mode, curv ** -0.5


def laplace():
    cases = [("Full survey: 58 of 100", 60, 44, (0.3, 0.85)), ("Tiny pilot: 1 of 10", 3, 11, (-0.25, 0.75))]
    fig = go.Figure()
    anns = []
    for k, (name, a, b, (lo, hi)) in enumerate(cases):
        xs = np.linspace(lo, hi, 800)
        mode, sd = laplace_parts(a, b)
        exact = np.where((xs > 0) & (xs < 1), stats.beta.pdf(np.clip(xs, 1e-9, 1 - 1e-9), a, b), 0)
        bell = stats.norm.pdf(xs, mode, sd)
        neg = xs < 0
        vis = k == 0
        fig.add_trace(go.Scatter(x=xs[neg], y=bell[neg], fill="tozeroy", fillcolor="rgba(229,62,62,0.35)", line=dict(width=0), visible=vis,
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=xs, y=exact, line=dict(color=BLUE, width=3), name="exact posterior", visible=vis))
        fig.add_trace(go.Scatter(x=xs, y=bell, line=dict(color=ORANGE, width=3, dash="dash"), name="Laplace bell", visible=vis))
        fig.add_trace(go.Scatter(x=[mode, mode], y=[0, stats.norm.pdf(mode, mode, sd) * 1.05], mode="lines",
                                 line=dict(color=DARK, width=1.5, dash="dot"), name="top of the hill (MAP)", visible=vis))
        e_lo, e_hi = stats.beta.ppf([0.025, 0.975], a, b)
        p_neg = stats.norm.cdf(0, mode, sd)
        txt = f"<b>95% interval</b><br>exact: {e_lo:.1%} – {e_hi:.1%}<br>Laplace: {mode - 1.96 * sd:.1%} – {mode + 1.96 * sd:.1%}"
        if p_neg > 0.001:
            txt += f"<br><span style='color:{RED}'><b>{p_neg:.0%}</b> of the bell<br>is below 0%: impossible</span>"
        else:
            txt += "<br><span style='color:#38a169'>almost perfect fit</span>"
        anns.append([dict(x=0.99, y=0.97, xref="paper", yref="paper", xanchor="right", yanchor="top", align="right", showarrow=False,
                          text=txt, font=dict(size=12, color=DARK), bgcolor="rgba(255,255,255,0.85)")])
    buttons = []
    for k, (name, *_r, rng_) in enumerate(cases):
        vis = [False] * 8
        for j in range(4):
            vis[4 * k + j] = True
        buttons.append(dict(label=name, method="update", args=[{"visible": vis}, {"annotations": anns[k], "xaxis.range": list(rng_), "yaxis.autorange": True}]))
    H, T, B = 500, 120, 110
    layout(fig, H, t=T, b=B, annotations=anns[0],
           title=title("Laplace approximation (1774): a bell at the top of the hill",
                       "Find the highest point (MAP), match how sharply the hill bends there,<br>and draw a Normal bell."),
           xaxis=dict(title="support", tickformat=".0%", range=list(cases[0][3])), yaxis=dict(title="density"),
           legend=dict(orientation="h", x=0, y=-0.22, font=dict(size=12)),
           updatemenus=button_row(buttons, H, T, B, px_above=10))
    save(fig, "lesson6_laplace", width=700)


# ---------------------------------------------------------------------------
# 10. Variational inference: the bell that adjusts itself
# ---------------------------------------------------------------------------
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(60)
GH_W = GH_W / GH_W.sum()
LOG_EVIDENCE = np.log(stats.betabinom.pmf(58, 100, 2, 2))


def log_target_z(z):
    t = expit(z)
    return stats.beta.logpdf(t, 2, 2) + stats.binom.logpmf(58, 100, t) + np.log(t) + np.log1p(-t)


def elbo(m, log_s):
    s = np.exp(log_s)
    z = m + s * GH_X
    return float((GH_W * log_target_z(z)).sum() + log_s + 0.5 * np.log(2 * np.pi * np.e))


def vi_fit():
    m, ls = logit(0.25), np.log(0.6)
    path = [(m, ls, elbo(m, ls))]
    lr, h = 0.01, 1e-5
    for _ in range(80):
        gm = (elbo(m + h, ls) - elbo(m - h, ls)) / (2 * h)
        gs = (elbo(m, ls + h) - elbo(m, ls - h)) / (2 * h)
        m, ls = m + lr * gm, ls + lr * 0.5 * gs
        path.append((m, ls, elbo(m, ls)))
    xs = np.linspace(0.05, 0.9, 600)

    def q_theta(m, ls):
        return stats.norm.pdf(logit(xs), m, np.exp(ls)) / (xs * (1 - xs))

    its = np.arange(len(path))
    el = np.array([p[2] for p in path])
    fig = make_subplots(rows=2, cols=1, row_heights=[0.55, 0.45], vertical_spacing=0.16, subplot_titles=[
        "step 0", "<b>Fit score (ELBO)</b> at each optimisation step"])
    base = subplot_title_dicts(fig)
    fig.add_trace(go.Scatter(x=xs, y=POST.pdf(xs), line=dict(color=BLUE, width=3), name="posterior"), 1, 1)
    fig.add_trace(go.Scatter(x=xs, y=q_theta(*path[0][:2]), line=dict(color=PURPLE, width=3, dash="dash"), name="VI bell"), 1, 1)
    fig.add_trace(go.Scatter(x=its, y=el, mode="lines", line=dict(color=GREY, width=1.5), showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=[0], y=[el[0]], mode="markers", marker=dict(color=PURPLE, size=12), showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=[0, its[-1]], y=[LOG_EVIDENCE] * 2, mode="lines+text", line=dict(color=DARK, dash="dot"),
                             text=["", "best possible"], textposition="bottom left", showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=[0.89], y=[9.4], mode="text", text=[f"step 0 · remaining KL = {LOG_EVIDENCE - el[0]:.2f}"],
                             textposition="middle left", textfont=dict(size=13, color=DARK), hoverinfo="skip", showlegend=False), 1, 1)

    def step_titles(k, ek):
        return retitle(base, 0, "<b>The bell moves and changes width</b>")

    shown = [0, 1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 60, 80]
    static = list(fig.data)
    fig.frames = [go.Frame(name=str(k), data=[static[0], go.Scatter(x=xs, y=q_theta(*path[k][:2])), static[2],
                                              go.Scatter(x=[k], y=[path[k][2]]), static[4],
                                              go.Scatter(x=[0.89], y=[9.4], text=[f"step {k} · remaining KL = {LOG_EVIDENCE - path[k][2]:.2f}"])],
                           traces=[0, 1, 2, 3, 4, 5]) for k in shown]
    steps = [dict(method="animate", label=str(k) if k in (0, 10, 30, 80) else "",
                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))]) for k in shown]
    layout(fig, 700, t=100, b=110, annotations=step_titles(0, el[0]), legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
           title=title("Variational inference: the bell that adjusts itself",
                       "Press ▶ Play. The optimiser raises the ELBO, which is the same as shrinking<br>the KL distance between the bell and the posterior."),
           xaxis=dict(title="support", tickformat=".0%", range=[0.05, 0.9]), yaxis=dict(title="density", range=[0, 10]),
           xaxis2=dict(title="optimisation step"), yaxis2=dict(title="ELBO"),
           updatemenus=[dict(type="buttons", showactive=False, x=0.0, y=-0.12, xanchor="left", yanchor="top", direction="left", buttons=[
               dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=400, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
               dict(label="❚❚", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])])],
           sliders=[dict(active=0, x=0.24, y=-0.12, len=0.76, xanchor="left", yanchor="top", currentvalue=dict(visible=False),
                         font=dict(size=10), steps=steps)])
    save(fig, "lesson6_vi_fit", width=700)


# ---------------------------------------------------------------------------
# 11. Mean-field VI is too narrow when unknowns move together
# ---------------------------------------------------------------------------
def ellipse(cov, k=2.0, n=200):
    vals, vecs = np.linalg.eigh(cov)
    t = np.linspace(0, 2 * np.pi, n)
    pts = vecs @ (np.sqrt(vals)[:, None] * np.stack([np.cos(t), np.sin(t)])) * k
    return pts[0], pts[1]


def meanfield():
    rho = 0.9
    cov = np.array([[1, rho], [rho, 1]])
    rng = np.random.default_rng(3)
    draws = rng.multivariate_normal([0, 0], cov, 2000)
    mf_sd = np.sqrt(1 - rho ** 2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=draws[:, 0], y=draws[:, 1], mode="markers", marker=dict(size=3, color=BLUE, opacity=0.45), name="draws (MCMC)"))
    ex, ey = ellipse(np.diag([mf_sd ** 2, mf_sd ** 2]))
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=PURPLE, width=3), fill="toself", fillcolor="rgba(128,90,213,0.15)",
                             name="mean-field VI", visible=False))
    fx, fy = ellipse(cov)
    fig.add_trace(go.Scatter(x=fx, y=fy, mode="lines", line=dict(color=GREEN, width=3), fill="toself", fillcolor="rgba(56,161,105,0.12)",
                             name="full-rank VI", visible=False))

    def ann(t):
        return [dict(x=0.01, y=0.99, xref="paper", yref="paper", xanchor="left", yanchor="top", align="left", showarrow=False,
                     text=t, font=dict(size=12, color=DARK), bgcolor="rgba(255,255,255,0.9)")]

    a0 = ann("Two unknowns that move together<br>(correlation 0.9). Spread of<br>unknown 1 in the draws: <b>1.00</b>")
    a1 = ann(f"Mean-field: an independent bell<br>per unknown. Spread: <b>{mf_sd:.2f}</b>,<br>"
             f"less than half: <span style='color:{RED}'>overconfident</span>")
    a2 = ann("Full-rank VI lets the bell tilt:<br>spread <b>1.00</b>, like the draws,<br>but it costs more computation.")
    H, T, B = 560, 125, 90
    layout(fig, H, t=T, b=B, l=50, annotations=a0,
           title=title("When the bell is too simple",
                       "Mean-field VI ignores links between unknowns<br>and can be far too sure of itself."),
           xaxis=dict(title="unknown 1", range=[-4, 4], constrain="domain"), yaxis=dict(title="unknown 2", range=[-4, 4], scaleanchor="x"),
           legend=dict(orientation="h", x=0, y=-0.18, font=dict(size=12)),
           updatemenus=button_row([
               dict(label="Draws", method="update", args=[{"visible": [True, False, False]}, {"annotations": a0}]),
               dict(label="+ mean-field VI", method="update", args=[{"visible": [True, True, False]}, {"annotations": a1}]),
               dict(label="+ full-rank VI", method="update", args=[{"visible": [True, True, True]}, {"annotations": a2}])], H, T, B, px_above=10))
    save(fig, "lesson6_meanfield", width=700)


# ---------------------------------------------------------------------------
# 12. Method map: one question, five ways
# ---------------------------------------------------------------------------
def method_map():
    cases = [("Full survey: 58 of 100", 58, 100, (0.3, 0.85)), ("Tiny pilot: 1 of 10", 1, 10, (-0.2, 0.75))]
    fig = go.Figure()
    anns = []
    for k, (name, yobs, n, (lo, hi)) in enumerate(cases):
        a, b = 2 + yobs, 2 + n - yobs
        with pm.Model():
            th = pm.Beta("theta", 2, 2)
            pm.Binomial("y", n=n, p=th, observed=yobs)
            idata = pm.sample(1000, tune=1000, chains=4, random_seed=11, progressbar=False)
            approx = pm.fit(n=30_000, method="advi", random_seed=11, progressbar=False)
            vi = np.asarray(approx.sample(8000, random_seed=11).posterior["theta"]).ravel()
            mp = float(pm.find_MAP(progressbar=False)["theta"])
        mc = idata.posterior["theta"].values.ravel()
        xs = np.linspace(lo, hi, 800)
        exact = np.where((xs > 0) & (xs < 1), stats.beta.pdf(np.clip(xs, 1e-9, 1 - 1e-9), a, b), 0)
        mode, sd = laplace_parts(a, b)
        grid = np.linspace(0.0125, 0.9875, 40)
        gd = stats.beta.pdf(grid, 2, 2) * stats.binom.pmf(yobs, n, grid)
        gd = gd / (gd.sum() * (grid[1] - grid[0]))
        edges = np.linspace(max(lo, 0), min(hi, 1), 45)
        m_mc, d_mc = density_line(mc, edges)
        m_vi, d_vi = density_line(vi, edges)
        vis = k == 0
        fig.add_trace(go.Bar(x=m_mc, y=d_mc, width=(edges[1] - edges[0]) * 0.95, marker_color="rgba(43,108,176,0.35)", name="MCMC (pm.sample)", visible=vis))
        fig.add_trace(go.Scatter(x=xs, y=exact, line=dict(color=DARK, width=3), name="exact", visible=vis))
        fig.add_trace(go.Scatter(x=grid, y=gd, mode="markers", marker=dict(color=GREY, size=7, symbol="square"), name="grid", visible=vis))
        fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, mode, sd), line=dict(color=ORANGE, width=3, dash="dash"), name="Laplace", visible=vis))
        fig.add_trace(go.Scatter(x=m_vi, y=d_vi, mode="lines", line=dict(color=PURPLE, width=3, dash="dot", shape="spline"), name="VI (pm.fit)", visible=vis))
        rows = [("exact", *stats.beta.ppf([0.025, 0.975], a, b)), ("MCMC", *np.quantile(mc, [0.025, 0.975])),
                ("Laplace", mode - 1.96 * sd, mode + 1.96 * sd), ("VI", *np.quantile(vi, [0.025, 0.975]))]
        txt = "<b>95% intervals</b><br>" + "<br>".join(f"{r[0]}: {r[1]:.1%} – {r[2]:.1%}" for r in rows) + f"<br>MAP: {mp:.1%}"
        anns.append(([dict(x=0.99, y=0.97, xref="paper", yref="paper", xanchor="right", yanchor="top", align="right", showarrow=False,
                           text=txt, font=dict(size=12, color=DARK), bgcolor="rgba(255,255,255,0.9)")], (lo, hi)))
    nt = 5
    buttons = []
    for k, (name, *_r) in enumerate(cases):
        vis = [False] * (2 * nt)
        for j in range(nt):
            vis[nt * k + j] = True
        buttons.append(dict(label=name, method="update", args=[{"visible": vis}, {"annotations": anns[k][0], "xaxis.range": list(anns[k][1]), "yaxis.autorange": True}]))
    H, T, B = 540, 125, 110
    layout(fig, H, t=T, b=B, annotations=anns[0][0], barmode="overlay",
           title=title("One question, five ways to get the posterior",
                       "With plenty of data every method agrees. With very little data the<br>shortcuts drift from the exact answer; MCMC does not."),
           xaxis=dict(title="support", tickformat=".0%", range=list(cases[0][3])), yaxis=dict(title="density"),
           legend=dict(orientation="h", x=0, y=-0.2, font=dict(size=12)),
           updatemenus=button_row(buttons, H, T, B, px_above=10))
    save(fig, "lesson6_method_map", width=700)


if __name__ == "__main__":
    step_size()
    warmup()
    four_chains()
    sticky_ess()
    mcse_wobble()
    coverage_two_hills()
    idata_ok = fit_survey()
    pymc_vs_exact(idata_ok)
    broken_fits(idata_ok)
    laplace()
    vi_fit()
    meanfield()
    method_map()
