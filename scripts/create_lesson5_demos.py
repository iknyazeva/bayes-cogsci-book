"""Interactive figures for Lesson 5: why a posterior is hard to compute, Monte Carlo and MCMC.

Run from the book root with the pymc_env environment:

    conda run -n pymc_env python scripts/create_lesson5_demos.py

Writes standalone Plotly pages to ``_static/lesson5_*.html``. Set ``PREVIEW_DIR``
(and optionally ``PREVIEW_WIDTH``, e.g. 700 = the book's content column) to also
write PNG snapshots of the first frame.

Layout rule: every figure must stay readable in the ~650-750 px book column, so
panels that need width are stacked vertically, subtitles are at most two short
lines, and slider labels are sparse.

Running example (Lesson 4): 58 of 100 lower-income households support a housing
subsidy; prior Beta(2, 2); posterior Beta(60, 44).
"""

from __future__ import annotations

import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.special import gammaln

os.makedirs("_static", exist_ok=True)
PREVIEW_DIR = os.environ.get("PREVIEW_DIR")

BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
GREY = "#a0aec0"
RED = "#e53e3e"
GREEN = "#38a169"
PURPLE = "#805ad5"
DARK = "#2d3748"
GOLD = "#d69e2e"

A_POST, B_POST = 60, 44  # Beta(2,2) prior + 58 of 100
POST = stats.beta(A_POST, B_POST)
X = np.linspace(0.0, 1.0, 1001)
XZ = np.linspace(0.30, 0.85, 600)
SUPPORT = "support (lower-income)"


def title(main: str, sub: str = "") -> dict:
    text = f"<b>{main}</b>" if not sub else f"<b>{main}</b><br><span style='font-size:12px;color:gray'>{sub}</span>"
    return dict(text=text, x=0.01, xanchor="left", y=1.0, yref="container", yanchor="top", pad=dict(t=16), font=dict(size=16))


def unnormalised(theta):
    """Prior x likelihood for the survey (no denominator)."""
    return stats.beta.pdf(theta, 2, 2) * stats.binom.pmf(58, 100, theta)


def base_layout(fig: go.Figure, height: int, t: int = 95, b: int = 55, l: int = 55, r: int = 20, **kw) -> None:
    fig.update_layout(template="plotly_white", height=height, margin=dict(t=t, b=b, l=l, r=r), **kw)


def save(fig: go.Figure, name: str, width: int = 900, height: int | None = None) -> None:
    fig.write_html(f"_static/{name}.html", include_plotlyjs="cdn", auto_play=False, config={"displayModeBar": False, "responsive": True})
    if PREVIEW_DIR:
        os.makedirs(PREVIEW_DIR, exist_ok=True)
        w = int(os.environ.get("PREVIEW_WIDTH", width))
        if fig.layout.width:
            w = min(w, int(fig.layout.width))
        fig.write_image(f"{PREVIEW_DIR}/{name}.png", width=w, height=height or fig.layout.height or 450)
    print("wrote", name)


def shrink_subplot_titles(fig: go.Figure, size: int = 13) -> None:
    for a in fig.layout.annotations:
        a.font = dict(size=size, color=DARK)


def play_controls(frame_names, labels, duration=350, y=-0.16):
    """Play/Pause buttons on the left, a slider with sparse labels on the right."""
    steps = [dict(method="animate", label=lab,
                  args=[[n], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
             for n, lab in zip(frame_names, labels)]
    menus = [dict(type="buttons", showactive=False, x=0.0, y=y, xanchor="left", yanchor="top", direction="left",
                  pad=dict(t=0, r=4), buttons=[
                      dict(label="▶ Play", method="animate",
                           args=[None, dict(frame=dict(duration=duration, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                      dict(label="❚❚", method="animate",
                           args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))]),
                  ])]
    sliders = [dict(active=0, x=0.24, y=y, len=0.76, xanchor="left", yanchor="top", pad=dict(t=0, b=0),
                    font=dict(size=10), currentvalue=dict(visible=False), ticklen=3, steps=steps)]
    return menus, sliders


def sparse(names, keep):
    return [f"{n:,}" if n in keep else "" for n in names]


# ---------------------------------------------------------------------------
# 1. Bayes' theorem gives a shape, not a size
# ---------------------------------------------------------------------------
def shape_not_size():
    grid = np.linspace(0.01, 0.99, 50)
    prior = stats.beta.pdf(grid, 2, 2)
    lik = stats.binom.pmf(58, 100, grid)
    prod = prior * lik
    evidence = stats.betabinom.pmf(58, 100, 2, 2)
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.06,
                        subplot_titles=["<b>Prior</b>", "<b>× Likelihood</b>", "<b>= Posterior shape</b>"])
    shrink_subplot_titles(fig, 14)
    w = grid[1] - grid[0]
    fig.add_trace(go.Bar(x=grid, y=prior, width=w * 0.9, marker_color=GREY), 1, 1)
    fig.add_trace(go.Bar(x=grid, y=lik, width=w * 0.9, marker_color=ORANGE), 1, 2)
    fig.add_trace(go.Bar(x=grid, y=prod, width=w * 0.9, marker_color=BLUE), 1, 3)
    for c in (1, 2, 3):
        fig.update_xaxes(range=[0, 1], tickvals=[0, 0.5, 1], ticktext=["0%", "50%", "100%"], row=1, col=c)
    fig.update_yaxes(showticklabels=False)
    fig.add_annotation(x=0.5, y=-0.17, xref="paper", yref="paper", showarrow=False, text="support among lower-income households",
                       font=dict(size=12, color=DARK))
    base_layout(fig, 370, t=115, b=60, showlegend=False, bargap=0.05,
                title=title("Bayes' theorem gives a shape, not a size",
                            f"The blue bars have the right shape but a total area of {evidence:.4f}, not 1.<br>"
                            "Dividing by that area (the evidence) turns the shape into a distribution."))
    save(fig, "lesson5_shape_not_size")


# ---------------------------------------------------------------------------
# 2. Every answer is an area
# ---------------------------------------------------------------------------
def answers_are_areas():
    y = POST.pdf(XZ)
    lo, hi = POST.ppf([0.025, 0.975])
    mean = POST.mean()
    p_maj = POST.sf(0.5)

    def band(mask, color):
        xs, ys = XZ[mask], y[mask]
        return go.Scatter(x=np.r_[xs, xs[::-1]], y=np.r_[ys, np.zeros_like(ys)], fill="toself",
                          fillcolor=color, line=dict(width=0), hoverinfo="skip", visible=False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=XZ, y=y, line=dict(color=BLUE, width=3), hoverinfo="skip"))
    fig.add_trace(band(XZ > 0.5, "rgba(221,107,32,0.55)"))
    fig.add_trace(band((XZ >= lo) & (XZ <= hi), "rgba(43,108,176,0.35)"))
    fig.add_trace(go.Scatter(x=[mean, mean], y=[0, POST.pdf(mean)], mode="lines",
                             line=dict(color=DARK, width=3, dash="dash"), visible=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[mean], y=[-0.35], mode="markers", marker=dict(symbol="triangle-up", size=20, color=DARK),
                             visible=False, hoverinfo="skip"))
    fig.data[1].visible = True

    def ann(text, color):
        return [dict(x=0.99, y=0.97, xref="paper", yref="paper", xanchor="right", yanchor="top", align="right", showarrow=False,
                     text=text, font=dict(size=14, color=color), bgcolor="rgba(255,255,255,0.85)")]

    ann_maj = ann(f"Orange area = <b>{p_maj:.1%}</b><br>= P(support > 50%)", ORANGE)
    ann_int = ann(f"Middle 95% of the area:<br><b>{lo:.1%} – {hi:.1%}</b>", BLUE)
    ann_mean = ann(f"Average = <b>{mean:.1%}</b>:<br>the area balances here", DARK)
    base_layout(fig, 440, t=130, showlegend=False, annotations=ann_maj,
                title=title("Every answer is an area under the posterior",
                            "Click a question. In Lesson 4 the Beta formula computed these areas for us."),
                xaxis=dict(title=SUPPORT, tickformat=".0%", range=[0.3, 0.85]),
                yaxis=dict(title="density", range=[-0.7, 9.5]),
                updatemenus=[dict(type="buttons", direction="right", x=0.0, y=1.02, xanchor="left", yanchor="bottom", showactive=True,
                                  pad=dict(t=0, b=4), font=dict(size=12),
                                  buttons=[
                                      dict(label="P(support > 50%)", method="update",
                                           args=[{"visible": [True, True, False, False, False]}, {"annotations": ann_maj}]),
                                      dict(label="Middle 95%", method="update",
                                           args=[{"visible": [True, False, True, False, False]}, {"annotations": ann_int}]),
                                      dict(label="Average", method="update",
                                           args=[{"visible": [True, False, False, True, True]}, {"annotations": ann_mean}]),
                                  ])])
    save(fig, "lesson5_answers_are_areas")


# ---------------------------------------------------------------------------
# 3. In one dimension a grid is enough
# ---------------------------------------------------------------------------
def grid_area():
    evidence = stats.betabinom.pmf(58, 100, 2, 2)
    counts = [5, 10, 20, 100]
    xs = np.linspace(0, 1, 800)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=unnormalised(xs), line=dict(color=BLUE, width=3), hoverinfo="skip"))
    states = []
    for n in counts:
        edges = np.linspace(0, 1, n + 1)
        mids = (edges[:-1] + edges[1:]) / 2
        h = unnormalised(mids)
        states.append((mids, h, 1 / n, h.sum() / n))
    mids, h, w, est = states[1]
    fig.add_trace(go.Bar(x=mids, y=h, width=w * 0.97, marker=dict(color="rgba(43,108,176,0.30)", line=dict(color=BLUE, width=1))))

    def ann(n, est):
        return [dict(x=0.01, y=0.97, xref="paper", yref="paper", xanchor="left", yanchor="top", showarrow=False, align="left",
                     text=f"<b>{n} bars</b><br>area ≈ {est:.5f}<br>exact = {evidence:.5f}", font=dict(size=13, color=DARK))]

    steps = [dict(method="update", label=f"{n} bars",
                  args=[{"x": [xs, m], "y": [unnormalised(xs), hh], "width": [None, ww * 0.97]}, {"annotations": ann(n, e)}])
             for n, (m, hh, ww, e) in zip(counts, states)]
    base_layout(fig, 440, b=100, showlegend=False, annotations=ann(counts[1], states[1][3]),
                title=title("With one unknown, a grid computes the area",
                            "Move the slider: a few dozen bars already give the exact area (the denominator)."),
                xaxis=dict(title=SUPPORT, tickformat=".0%"), yaxis=dict(title="prior × likelihood"),
                sliders=[dict(active=1, steps=steps, x=0.05, len=0.9, y=-0.2, currentvalue=dict(visible=False), font=dict(size=11))])
    save(fig, "lesson5_grid_area")


# ---------------------------------------------------------------------------
# 4. The grid explodes
# ---------------------------------------------------------------------------
def grid_explosion():
    g = np.linspace(0.25, 0.75, 100)
    z = np.outer(stats.beta.pdf(g, 42, 62), stats.beta.pdf(g, 60, 44))
    dims = [1, 2, 3, 5, 10, 20]
    cells = [100.0 ** d for d in dims]
    labels = ["0.1 µs", "10 µs", "1 ms", "10 s", "3,200<br>years", "longer than<br>the universe"]
    fig = make_subplots(rows=2, cols=1, row_heights=[0.42, 0.58], vertical_spacing=0.14, subplot_titles=[
        "<b>2 unknowns</b>: a 100 × 100 grid already has 10,000 cells",
        "<b>Cells needed</b> (100 per unknown), time at a billion per second"])
    shrink_subplot_titles(fig)
    fig.add_trace(go.Heatmap(x=g, y=g, z=z, colorscale="Blues", showscale=False, xgap=0.3, ygap=0.3, hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Bar(x=[str(d) for d in dims], y=cells, text=labels, textposition="outside", textfont=dict(size=12),
                         marker_color=[BLUE, BLUE, BLUE, ORANGE, RED, RED], cliponaxis=False), 2, 1)
    fig.update_xaxes(title_text="support, lower-income", tickformat=".0%", constrain="domain", row=1, col=1)
    fig.update_yaxes(title_text="support,<br>higher-income", tickformat=".0%", scaleanchor="x", row=1, col=1)
    fig.update_xaxes(title_text="number of unknowns", type="category", row=2, col=1)
    fig.update_yaxes(type="log", title_text="grid cells", range=[0, 49], row=2, col=1,
                     tickvals=[1e2, 1e10, 1e20, 1e30, 1e40],
                     ticktext=["10<sup>2</sup>", "10<sup>10</sup>", "10<sup>20</sup>", "10<sup>30</sup>", "10<sup>40</sup>"])
    base_layout(fig, 780, t=100, showlegend=False,
                title=title("The grid explodes when a model has more unknowns",
                            "Each extra unknown multiplies the number of cells by 100.<br>Real models often have dozens of unknowns."))
    save(fig, "lesson5_grid_explosion")


# ---------------------------------------------------------------------------
# 5. Almost all of the box is empty
# ---------------------------------------------------------------------------
def empty_box():
    rng = np.random.default_rng(42)
    pts = rng.uniform(-1, 1, size=(1500, 2))
    inside = (pts ** 2).sum(1) <= 1
    dims = [2, 3, 5, 10, 20]
    frac = [np.exp(d / 2 * np.log(np.pi) - gammaln(d / 2 + 1) - d * np.log(2)) for d in dims]
    txt = ["79%", "52%", "16%", "0.25%", "0.0000025%"]
    fig = make_subplots(rows=2, cols=1, row_heights=[0.45, 0.55], vertical_spacing=0.13, subplot_titles=[
        "<b>2 unknowns</b>: 79% of random points land in the useful circle",
        "<b>Useful share</b> of the box as the number of unknowns grows"])
    shrink_subplot_titles(fig)
    fig.add_trace(go.Scatter(x=pts[~inside, 0], y=pts[~inside, 1], mode="markers", marker=dict(color=GREY, size=3)), 1, 1)
    fig.add_trace(go.Scatter(x=pts[inside, 0], y=pts[inside, 1], mode="markers", marker=dict(color=ORANGE, size=3)), 1, 1)
    t = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(x=np.cos(t), y=np.sin(t), mode="lines", line=dict(color=DARK, width=2)), 1, 1)
    fig.add_trace(go.Bar(x=[str(d) for d in dims], y=frac, text=txt, textposition="outside", textfont=dict(size=12), cliponaxis=False,
                         marker_color=[ORANGE, ORANGE, ORANGE, RED, RED]), 2, 1)
    fig.update_xaxes(range=[-1.05, 1.05], showticklabels=False, constrain="domain", row=1, col=1)
    fig.update_yaxes(range=[-1.05, 1.05], showticklabels=False, scaleanchor="x", row=1, col=1)
    fig.update_xaxes(title_text="number of unknowns", type="category", row=2, col=1)
    fig.update_yaxes(type="log", range=[-8.7, 0.7], title_text="useful share", row=2, col=1,
                     tickvals=[1, 1e-2, 1e-4, 1e-6, 1e-8], ticktext=["100%", "1%", "0.01%", "10<sup>-6</sup>", "10<sup>-8</sup>"])
    base_layout(fig, 760, t=100, showlegend=False,
                title=title("In many dimensions almost the whole box is empty",
                            "Points placed blindly (at random or on a grid) almost never land<br>where the posterior is."))
    save(fig, "lesson5_empty_box")


# ---------------------------------------------------------------------------
# 6. Monte Carlo: darts estimate pi
# ---------------------------------------------------------------------------
def darts_pi():
    rng = np.random.default_rng(1946)
    N = 10_000
    pts = rng.uniform(0, 1, size=(N, 2))
    inside = (pts ** 2).sum(1) <= 1
    run = 4 * np.cumsum(inside) / np.arange(1, N + 1)
    idx = np.unique(np.geomspace(1, N, 400).astype(int)) - 1
    ns = [10, 100, 1000, 10000]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55], horizontal_spacing=0.12,
                        subplot_titles=["<b>Darts on a square</b>", "<b>Estimate of π</b>"])
    shrink_subplot_titles(fig)
    sub_titles = [a.to_plotly_json() for a in fig.layout.annotations]

    def sub(n):
        p, ins = pts[:n], inside[:n]
        return p[ins], p[~ins]

    t = np.linspace(0, np.pi / 2, 100)
    pin, pout = sub(ns[1])
    fig.add_trace(go.Scatter(x=pin[:, 0], y=pin[:, 1], mode="markers", marker=dict(color=ORANGE, size=4)), 1, 1)
    fig.add_trace(go.Scatter(x=pout[:, 0], y=pout[:, 1], mode="markers", marker=dict(color=GREY, size=4)), 1, 1)
    fig.add_trace(go.Scatter(x=np.cos(t), y=np.sin(t), mode="lines", line=dict(color=DARK, width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=idx + 1, y=run[idx], mode="lines", line=dict(color=BLUE, width=2)), 1, 2)
    fig.add_trace(go.Scatter(x=[1, N], y=[np.pi, np.pi], mode="lines", line=dict(color=DARK, dash="dash")), 1, 2)
    fig.add_trace(go.Scatter(x=[ns[1]], y=[run[ns[1] - 1]], mode="markers", marker=dict(color=ORANGE, size=13)), 1, 2)

    def ann(n):
        return [dict(x=0.98, y=0.97, xref="x2 domain", yref="y2 domain", xanchor="right", yanchor="top", align="right", showarrow=False,
                     text=f"<b>{n:,} darts</b><br>π ≈ 4 × share inside<br>= <b>{run[n - 1]:.3f}</b>", font=dict(size=12, color=DARK),
                     bgcolor="rgba(255,255,255,0.85)")]

    steps = []
    for n in ns:
        a, b = sub(n)
        steps.append(dict(method="update", label=f"{n:,}",
                          args=[{"x": [a[:, 0], b[:, 0], np.cos(t), idx + 1, [1, N], [n]],
                                 "y": [a[:, 1], b[:, 1], np.sin(t), run[idx], [np.pi, np.pi], [run[n - 1]]]},
                                {"annotations": sub_titles + ann(n)}]))
    fig.update_xaxes(range=[0, 1], constrain="domain", row=1, col=1)
    fig.update_yaxes(range=[0, 1], scaleanchor="x", row=1, col=1)
    fig.update_xaxes(type="log", title_text="number of darts", tickvals=[10, 100, 1000, 10000], ticktext=["10", "100", "1k", "10k"], row=1, col=2)
    fig.update_yaxes(range=[2.4, 4.0], row=1, col=2)
    for a in ann(ns[1]):
        fig.add_annotation(**a)
    base_layout(fig, 460, t=115, b=110, showlegend=False,
                title=title("Monte Carlo: let randomness compute an area",
                            "The share of darts inside the quarter circle estimates its area, π/4.<br>Move the slider: the wobble shrinks as darts accumulate."),
                sliders=[dict(active=1, steps=steps, x=0.05, len=0.9, y=-0.2, currentvalue=dict(prefix="darts: ", font=dict(size=12)),
                              font=dict(size=11))])
    save(fig, "lesson5_darts_pi")


# ---------------------------------------------------------------------------
# 7. A pile of draws answers questions
# ---------------------------------------------------------------------------
def pile_of_draws():
    rng = np.random.default_rng(2026)
    draws = rng.beta(A_POST, B_POST, size=10_000)
    edges = np.arange(0.30, 0.8501, 0.01)
    mids = (edges[:-1] + edges[1:]) / 2
    ns = [10, 100, 1000, 10000]

    def hist(n):
        c, _ = np.histogram(draws[:n], bins=edges)
        d = c / (n * 0.01)
        left = mids < 0.5
        return d * left, d * ~left, (draws[:n] > 0.5).mean()

    fig = go.Figure()
    l, r, share = hist(ns[1])
    fig.add_trace(go.Bar(x=mids, y=l, width=0.0095, marker_color=GREY))
    fig.add_trace(go.Bar(x=mids, y=r, width=0.0095, marker_color=ORANGE))
    fig.add_trace(go.Scatter(x=XZ, y=POST.pdf(XZ), line=dict(color=BLUE, width=3), hoverinfo="skip"))

    def ann(n, share):
        return [dict(x=0.99, y=0.97, xref="paper", yref="paper", xanchor="right", yanchor="top", showarrow=False, align="right",
                     text=f"<b>{n:,} draws</b><br>orange share = <b>{share:.1%}</b><br>exact = {POST.sf(0.5):.1%}",
                     font=dict(size=13, color=DARK), bgcolor="rgba(255,255,255,0.85)")]

    steps = []
    for n in ns:
        l, r, s = hist(n)
        steps.append(dict(method="update", label=f"{n:,} draws", args=[{"y": [l, r, POST.pdf(XZ)]}, {"annotations": ann(n, s)}]))
    base_layout(fig, 450, t=115, b=100, showlegend=False, barmode="overlay", annotations=ann(ns[1], share),
                title=title("A pile of draws can stand in for the distribution",
                            "Each draw is one plausible value of support. Questions become counting:<br>what share of the draws is above 50% (orange)?"),
                xaxis=dict(title=SUPPORT, tickformat=".0%", range=[0.3, 0.85]), yaxis=dict(title="density", range=[0, 14]),
                sliders=[dict(active=1, steps=steps, x=0.05, len=0.9, y=-0.2, currentvalue=dict(visible=False), font=dict(size=11))])
    save(fig, "lesson5_pile_of_draws")


# ---------------------------------------------------------------------------
# 8. Rejection sampling (von Neumann, 1951)
# ---------------------------------------------------------------------------
def rejection():
    rng = np.random.default_rng(1951)
    M = POST.pdf((A_POST - 1) / (A_POST + B_POST - 2)) * 1.02
    n = 4000
    xs = rng.uniform(0, 1, n)
    us = rng.uniform(0, M, n)
    keep = us < POST.pdf(xs)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, row_heights=[0.58, 0.42], subplot_titles=[
        f"<b>Darts in a box: keep those under the curve</b> (kept {keep.mean():.0%})",
        "<b>The kept darts are draws from the posterior</b>"])
    shrink_subplot_titles(fig)
    fig.add_trace(go.Scatter(x=xs[~keep], y=us[~keep], mode="markers", marker=dict(color=GREY, size=3, opacity=0.5)), 1, 1)
    fig.add_trace(go.Scatter(x=xs[keep], y=us[keep], mode="markers", marker=dict(color=BLUE, size=3.5)), 1, 1)
    fig.add_trace(go.Scatter(x=X, y=POST.pdf(X), line=dict(color=DARK, width=2.5), hoverinfo="skip"), 1, 1)
    edges = np.arange(0.30, 0.8501, 0.0125)
    c, _ = np.histogram(xs[keep], bins=edges)
    fig.add_trace(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=c / (keep.sum() * 0.0125), width=0.012, marker_color=BLUE), 2, 1)
    fig.add_trace(go.Scatter(x=X, y=POST.pdf(X), line=dict(color=DARK, width=2.5), hoverinfo="skip"), 2, 1)
    fig.update_xaxes(range=[0, 1], tickformat=".0%", row=2, col=1, title_text=SUPPORT)
    fig.update_yaxes(title_text="height", row=1, col=1)
    fig.update_yaxes(title_text="density", row=2, col=1)
    base_layout(fig, 610, t=115, showlegend=False,
                title=title("Rejection sampling (von Neumann, 1951)",
                            f"It needs only the shape of the curve, but {1 - keep.mean():.0%} of the darts are wasted.<br>"
                            "With twenty unknowns almost every dart would be rejected."))
    save(fig, "lesson5_rejection")


# ---------------------------------------------------------------------------
# 9. The 1953 idea: place points where the probability is
# ---------------------------------------------------------------------------
def weight_vs_place():
    n = 60
    even = np.linspace(0.02, 0.98, n)
    h = POST.pdf(even)
    placed = POST.ppf((np.arange(n) + 0.5) / n)
    rng = np.random.default_rng(3)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.18, subplot_titles=[
        "<b>Old way</b>: spread points evenly, weight each by its height<br>(most points are almost weightless)",
        "<b>Metropolis et al. (1953)</b>: place points where the probability is<br>(every point counts equally)"])
    shrink_subplot_titles(fig)
    for r in (1, 2):
        fig.add_trace(go.Scatter(x=X, y=POST.pdf(X) / POST.pdf(X).max(), line=dict(color="rgba(43,108,176,0.25)", width=2),
                                 fill="tozeroy", fillcolor="rgba(43,108,176,0.07)", hoverinfo="skip"), r, 1)
    fig.add_trace(go.Scatter(x=even, y=np.full(n, 0.5), mode="markers",
                             marker=dict(size=3 + 22 * h / h.max(), color=ORANGE, line=dict(color=DARK, width=0.5))), 1, 1)
    fig.add_trace(go.Scatter(x=placed, y=0.5 + rng.uniform(-0.25, 0.25, n), mode="markers",
                             marker=dict(size=8, color=BLUE, line=dict(color=DARK, width=0.5))), 2, 1)
    fig.update_yaxes(showticklabels=False, range=[0, 1.1])
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_xaxes(title_text=SUPPORT, row=2, col=1)
    base_layout(fig, 520, t=120, showlegend=False,
                title=title("The key idea of 1953",
                            "Instead of weighting points by probability,<br>choose points with that probability."))
    save(fig, "lesson5_weight_vs_place")


# ---------------------------------------------------------------------------
# 10. Only the ratio of heights matters
# ---------------------------------------------------------------------------
def ratio_only():
    xs = np.linspace(0.3, 0.85, 500)
    f = unnormalised(xs)
    a, b = 0.50, 0.58
    fa, fb = unnormalised(a), unnormalised(b)
    ratio = fb / fa
    scales = [("as computed", 1.0), ("× 1,000", 1e3), ("÷ 1,000", 1e-3), ("× 1,000,000", 1e6)]

    def traces(c):
        return ([xs, [a, a], [b, b], [a, b]], [f * c, [0, fa * c], [0, fb * c], [fa * c, fb * c]])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=f, line=dict(color=BLUE, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[a, a], y=[0, fa], mode="lines", line=dict(color=ORANGE, width=8)))
    fig.add_trace(go.Scatter(x=[b, b], y=[0, fb], mode="lines", line=dict(color=GREEN, width=8)))
    fig.add_trace(go.Scatter(x=[a, b], y=[fa, fb], mode="markers", marker=dict(color=[ORANGE, GREEN], size=14)))

    def ann(label, c):
        return [dict(x=0.01, y=0.97, xref="paper", yref="paper", xanchor="left", yanchor="top", showarrow=False, align="left",
                     text=f"curve <b>{label}</b><br>height at 50%: {fa * c:.3g}<br>height at 58%: {fb * c:.3g}<br>"
                          f"<b>ratio = {ratio:.2f}</b> (always)",
                     font=dict(size=13, color=DARK), bgcolor="rgba(255,255,255,0.85)"),
                dict(x=a, y=fa * c, yanchor="bottom", yshift=10, xanchor="right", text="here", showarrow=False, font=dict(color=ORANGE, size=13)),
                dict(x=b, y=fb * c, yanchor="bottom", yshift=10, xanchor="left", text="proposed", showarrow=False, font=dict(color=GREEN, size=13))]

    steps = []
    for label, c in scales:
        xx, yy = traces(c)
        steps.append(dict(method="update", label=label, args=[{"x": xx, "y": yy}, {"annotations": ann(label, c)}]))
    base_layout(fig, 450, t=115, b=100, showlegend=False, annotations=ann(*scales[0]),
                title=title("The walker only compares two heights",
                            "Multiply the curve by any number: the picture and the ratio stay the same;<br>only the axis numbers change."),
                xaxis=dict(title=SUPPORT, tickformat=".0%"),
                yaxis=dict(title="prior × likelihood", exponentformat="power", rangemode="tozero"),
                sliders=[dict(active=0, steps=steps, x=0.05, len=0.9, y=-0.22, currentvalue=dict(visible=False), font=dict(size=11))])
    save(fig, "lesson5_ratio_only")


# ---------------------------------------------------------------------------
# 11-13. King Markov's islands
# ---------------------------------------------------------------------------
POPS = np.arange(1, 8)


def king_chain(start, nights, seed):
    rng = np.random.default_rng(seed)
    pos = start
    path = np.empty(nights, dtype=int)
    for t in range(nights):
        path[t] = pos
        prop = (pos + rng.choice([-1, 1])) % 7
        if rng.random() < min(1.0, POPS[prop] / POPS[pos]):
            pos = prop
    return path


def king_markov():
    path = king_chain(start=3, nights=20_000, seed=42)
    target = POPS / POPS.sum()
    frame_nights = list(range(1, 16)) + [25, 50, 100, 200, 500, 1000, 2000, 5000, 20000]
    isl_x = np.arange(1, 8)
    fig = make_subplots(rows=2, cols=1, row_heights=[0.3, 0.7], vertical_spacing=0.1, subplot_titles=[
        "<b>Seven islands in a ring</b> (circle size = population)", "<b>Share of nights spent on each island</b>"])
    shrink_subplot_titles(fig)
    sub_titles = [a.to_plotly_json() for a in fig.layout.annotations]
    fig.add_trace(go.Scatter(x=isl_x, y=np.zeros(7), mode="markers+text", text=[f"{p}k" for p in POPS], textposition="bottom center",
                             marker=dict(size=8 + 6 * POPS, color="rgba(56,161,105,0.35)", line=dict(color=GREEN, width=2)),
                             hoverinfo="skip"), 1, 1)

    def shares(n):
        return np.bincount(path[:n], minlength=7) / n

    fig.add_trace(go.Scatter(x=[isl_x[path[0]]], y=[0.0], mode="markers",
                             marker=dict(symbol="star", size=24, color=GOLD, line=dict(color=DARK, width=1.5))), 1, 1)
    fig.add_trace(go.Bar(x=isl_x, y=shares(1), marker_color=BLUE, width=0.6), 2, 1)
    fig.add_trace(go.Scatter(x=isl_x, y=target, mode="markers", marker=dict(symbol="line-ew", size=36, line=dict(color=ORANGE, width=4))), 2, 1)

    def night_ann(n):
        return [dict(x=1.0, y=1.0, xref="paper", yref="paper", xanchor="right", yanchor="bottom", showarrow=False,
                     text=f"<b>Night {n:,}</b>", font=dict(size=15, color=DARK))]

    fig.add_trace(go.Scatter(x=[7.6], y=[0.62], mode="text", text=["<b>Night 1</b>"], textposition="middle left",
                             textfont=dict(size=15, color=DARK), hoverinfo="skip"), 1, 1)  # counter (data, so it animates)
    islands, target_marks = fig.data[0], fig.data[3]
    fig.frames = [go.Frame(name=str(n), data=[islands, go.Scatter(x=[isl_x[path[n - 1]]], y=[0.0]), go.Bar(x=isl_x, y=shares(n)), target_marks,
                                              go.Scatter(x=[7.6], y=[0.62], text=[f"<b>Night {n:,}</b>"])],
                           traces=[0, 1, 2, 3, 4], layout=dict(yaxis2=dict(range=[0, 1.05] if n < 50 else [0, 0.4])))
                  for n in frame_nights]
    menus, sliders = play_controls([str(n) for n in frame_nights], sparse(frame_nights, {1, 10, 100, 1000, 20000}), duration=450, y=-0.17)
    fig.update_yaxes(visible=False, range=[-0.9, 0.7], row=1, col=1)
    fig.update_xaxes(visible=False, range=[0.4, 7.6], row=1, col=1)
    fig.update_xaxes(title_text="island (1 = smallest, 7 = largest)", tickvals=list(isl_x), range=[0.4, 7.6], row=2, col=1)
    fig.update_yaxes(title_text="share of nights", range=[0, 1.05], tickformat=".0%", row=2, col=1)
    base_layout(fig, 630, t=125, b=130, showlegend=False, updatemenus=menus, sliders=sliders,
                title=title("King Markov's rule produces the target distribution",
                            "Press ▶ Play. Blue bars: where the king actually slept.<br>Orange marks: each island's share of the population."))
    save(fig, "lesson5_king_markov", height=630)


def forget_start():
    fig = go.Figure()
    nights = np.arange(1, 20_001)
    for start, col, seed in [(0, PURPLE, 1), (3, BLUE, 2), (6, ORANGE, 3)]:
        path = king_chain(start=start, nights=20_000, seed=seed)
        share7 = np.cumsum(path == 6) / nights
        idx = np.unique(np.geomspace(1, len(nights), 400).astype(int)) - 1
        fig.add_trace(go.Scatter(x=nights[idx], y=share7[idx], mode="lines", line=dict(color=col, width=2.5),
                                 name=f"starts on island {start + 1}"))
    fig.add_trace(go.Scatter(x=[1, 20000], y=[0.25, 0.25], mode="lines+text", line=dict(color=DARK, dash="dash"),
                             text=["", "target 25%"], textposition="top left", showlegend=False))
    base_layout(fig, 440, t=95, b=110,
                title=title("Wherever the king starts, the long run is the same",
                            "Share of nights on the largest island, for three starting islands."),
                xaxis=dict(type="log", title="night", tickvals=[1, 10, 100, 1000, 10000], ticktext=["1", "10", "100", "1,000", "10,000"]),
                yaxis=dict(title="share on island 7", tickformat=".0%", range=[0, 1.02]),
                legend=dict(orientation="h", x=0.0, y=-0.25, font=dict(size=12)))
    save(fig, "lesson5_forget_start")


def balanced_traffic():
    def state(va, vb):
        ab = va * 0.5 * 1.0
        ba = vb * 0.5 * (2 / 6)
        unbalanced = ab > ba + 1e-9
        verdict = ("More travellers go A → B than back:<br>visits to B will grow." if unbalanced
                   else "Traffic is balanced: the shares no longer change.<br>This is the target distribution.")
        col = RED if unbalanced else GREEN
        return [
            dict(x=0.15, y=0.5, text=f"<b>A</b><br>2,000<br>people<br>{va:.0%}<br>of nights", showarrow=False, font=dict(size=12)),
            dict(x=0.85, y=0.5, text=f"<b>Island B</b><br>6,000 people<br>{vb:.0%} of nights", showarrow=False, font=dict(size=13)),
            dict(x=0.62, y=0.8, ax=0.32, ay=0.8, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3,
                 arrowwidth=max(1.5, 24 * ab), arrowcolor=BLUE, text=""),
            dict(x=0.47, y=0.93, text=f"A → B: {va:.0%} × ½ × 1 = <b>{ab:.3f}</b>", showarrow=False, font=dict(color=BLUE, size=12)),
            dict(x=0.32, y=0.2, ax=0.62, ay=0.2, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3,
                 arrowwidth=max(1.5, 24 * ba), arrowcolor=ORANGE, text=""),
            dict(x=0.47, y=0.07, text=f"B → A: {vb:.0%} × ½ × 2/6 = <b>{ba:.3f}</b>", showarrow=False, font=dict(color=ORANGE, size=12)),
            dict(x=0.5, y=-0.12, text=verdict, showarrow=False, font=dict(color=col, size=13)),
        ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0.15, 0.85], y=[0.5, 0.5], mode="markers",
                             marker=dict(size=[80, 135], color="rgba(56,161,105,0.25)", line=dict(color=GREEN, width=2)), hoverinfo="skip"))
    s1, s2 = state(0.5, 0.5), state(0.25, 0.75)
    base_layout(fig, 480, t=115, b=70, showlegend=False, annotations=s1,
                xaxis=dict(visible=False, range=[0, 1]), yaxis=dict(visible=False, range=[-0.25, 1.02]),
                title=title("Why it works: balanced traffic",
                            "The king proposes the other island half of the time. He always moves to a<br>bigger island and moves to a smaller one with probability 2/6."),
                updatemenus=[dict(type="buttons", direction="right", x=0.5, y=-0.04, xanchor="center", yanchor="top", font=dict(size=12),
                                  buttons=[dict(label="Visits 50 : 50", method="relayout", args=[{"annotations": s1}]),
                                           dict(label="Visits 25 : 75 (= populations)", method="relayout", args=[{"annotations": s2}])])])
    save(fig, "lesson5_balanced_traffic")


# ---------------------------------------------------------------------------
# 14. The walker on the survey hill (Metropolis)
# ---------------------------------------------------------------------------
def walker_hill():
    rng = np.random.default_rng(1953)
    n = 5000
    step = 0.05
    logf = lambda t: stats.beta.logpdf(t, 2, 2) + stats.binom.logpmf(58, 100, t) if 0 < t < 1 else -np.inf
    pos = np.empty(n + 1)
    prop = np.empty(n)
    acc = np.empty(n, dtype=bool)
    pos[0] = 0.40
    for t in range(n):
        p = pos[t] + rng.normal(0, step)
        prop[t] = p
        acc[t] = np.log(rng.random()) < logf(p) - logf(pos[t])
        pos[t + 1] = p if acc[t] else pos[t]
    xs = np.linspace(0.3, 0.85, 400)
    peak = POST.pdf(xs).max()
    rel = lambda t: POST.pdf(t) / peak if 0 < t < 1 else 0.0
    edges = np.arange(0.30, 0.8501, 0.01)
    mids = (edges[:-1] + edges[1:]) / 2

    fig = make_subplots(rows=3, cols=1, row_heights=[0.36, 0.3, 0.34], vertical_spacing=0.1, subplot_titles=[
        "<b>The hill</b> (posterior shape) and the walker", "<b>Where the walker has been</b> vs the exact posterior (dashed)",
        "<b>Trace</b>: position at each step"])
    shrink_subplot_titles(fig)
    sub_titles = [a.to_plotly_json() for a in fig.layout.annotations]
    fig.add_trace(go.Scatter(x=xs, y=POST.pdf(xs) / peak, line=dict(color=BLUE, width=3), hoverinfo="skip"), 1, 1)  # 0
    fig.add_trace(go.Scatter(x=[pos[0]], y=[rel(pos[0])], mode="markers", marker=dict(size=16, color=BLUE, line=dict(color=DARK, width=2))), 1, 1)  # 1
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers+text", textposition="top center",
                             marker=dict(size=14, symbol="circle-open", line=dict(width=3))), 1, 1)  # 2
    fig.add_trace(go.Bar(x=mids, y=np.zeros_like(mids), width=0.0095, marker_color="rgba(43,108,176,0.55)"), 2, 1)  # 3
    fig.add_trace(go.Scatter(x=xs, y=POST.pdf(xs), line=dict(color=DARK, width=2, dash="dash"), hoverinfo="skip"), 2, 1)  # 4
    fig.add_trace(go.Scatter(x=[0], y=[pos[0]], mode="lines", line=dict(color=BLUE, width=1.2)), 3, 1)  # 5
    fig.add_trace(go.Scatter(x=[0.85], y=[1.28], mode="text", text=["<b>Step 1</b>"], textposition="middle left",
                             textfont=dict(size=15, color=DARK), hoverinfo="skip"), 1, 1)  # 6: counter

    def step_ann(k):
        return [dict(x=1.0, y=1.0, xref="paper", yref="paper", xanchor="right", yanchor="bottom", showarrow=False,
                     text=f"<b>Step {k:,}</b>", font=dict(size=15, color=DARK))]

    def frame(k, show_prop):
        c, _ = np.histogram(pos[1:k + 1], bins=edges)
        dens = c / (max(k, 1) * 0.01)
        data = [go.Scatter(x=[pos[k]], y=[rel(pos[k])])]
        if show_prop:
            p, here = prop[k - 1], pos[k - 1]
            up = rel(p) >= rel(here)
            label = "uphill: move" if up else ("downhill: coin says move" if acc[k - 1] else "downhill: coin says stay")
            col = GREEN if acc[k - 1] else RED
            data.append(go.Scatter(x=[p], y=[rel(p)], text=[label], textfont=dict(color=col, size=12),
                                   marker=dict(size=14, symbol="circle-open", line=dict(width=3, color=col))))
        else:
            data.append(go.Scatter(x=[], y=[], text=[]))
        data.append(go.Bar(x=mids, y=dens))
        data.append(exact_curve)
        data.append(go.Scatter(x=np.arange(0, k + 1), y=pos[:k + 1]))
        data.append(go.Scatter(x=[0.85], y=[1.28], text=[f"<b>Step {k:,}</b>"]))
        return go.Frame(name=str(k), data=[hill_curve] + data, traces=[0, 1, 2, 3, 4, 5, 6],
                        layout=dict(xaxis3=dict(range=[0, 30] if k <= 30 else [0, k])))

    names = list(range(1, 26)) + [50, 100, 250, 500, 1000, 2500, 5000]
    hill_curve, exact_curve = fig.data[0], fig.data[4]
    fig.frames = [frame(k, k <= 25) for k in names]
    for trace_idx, d in zip([1, 2, 3, 5], [fig.frames[0].data[i] for i in (1, 2, 3, 5)]):
        fig.data[trace_idx].update(x=d.x, y=d.y)
        if trace_idx == 2:
            fig.data[2].update(text=d.text, textfont=d.textfont, marker=d.marker)
    menus, sliders = play_controls([str(k) for k in names], sparse(names, {1, 10, 20, 1000}), duration=600, y=-0.1)
    fig.update_xaxes(range=[0.3, 0.85], tickformat=".0%", row=1, col=1)
    fig.update_yaxes(range=[0, 1.35], title_text="height", row=1, col=1)
    fig.update_xaxes(range=[0.3, 0.85], tickformat=".0%", row=2, col=1)
    fig.update_yaxes(range=[0, 12], showticklabels=False, row=2, col=1)
    fig.update_xaxes(range=[0, 30], title_text="step", row=3, col=1)
    fig.update_yaxes(range=[0.3, 0.85], tickformat=".0%", title_text="support", row=3, col=1)
    base_layout(fig, 830, t=125, b=110, showlegend=False, updatemenus=menus, sliders=sliders,
                title=title("The Metropolis walker on our survey posterior",
                            "Press ▶ Play. Uphill: always move. Downhill: move with probability = height ratio.<br>Rejected: stay and count this place again."))
    save(fig, "lesson5_walker_hill", height=830)


# ---------------------------------------------------------------------------
# 15-16. Gibbs staircase and the HMC "skateboarder"
# ---------------------------------------------------------------------------
def gauss_contours(rho, lim=3.0):
    g = np.linspace(-lim, lim, 120)
    xx, yy = np.meshgrid(g, g)
    z = np.exp(-(xx ** 2 - 2 * rho * xx * yy + yy ** 2) / (2 * (1 - rho ** 2)))
    return go.Contour(x=g, y=g, z=z, contours=dict(coloring="lines", start=0.05, end=0.95, size=0.15),
                      line=dict(width=1.5), colorscale=[[0, "#bee3f8"], [1, BLUE]], showscale=False, hoverinfo="skip")


def gibbs_staircase():
    rho = 0.8
    rng = np.random.default_rng(1984)
    x, y = -2.2, 2.0
    px, py = [x], [y]
    sd = np.sqrt(1 - rho ** 2)
    for _ in range(25):
        x = rng.normal(rho * y, sd)
        px.append(x); py.append(y)
        y = rng.normal(rho * x, sd)
        px.append(x); py.append(y)
    fig = go.Figure()
    fig.add_trace(gauss_contours(rho))
    fig.add_trace(go.Scatter(x=px, y=py, mode="lines+markers", line=dict(color=ORANGE, width=2), marker=dict(size=5, color=ORANGE)))
    fig.add_trace(go.Scatter(x=[px[0]], y=[py[0]], mode="markers+text", text=["start"], textposition="top left",
                             marker=dict(size=12, color=DARK)))
    base_layout(fig, 520, t=105, showlegend=False,
                title=title("Gibbs sampling (1984): one unknown at a time",
                            "Fix one unknown, draw the other from its exact distribution,<br>then swap: the path is a staircase."),
                xaxis=dict(title="unknown 1", range=[-3, 3], constrain="domain"), yaxis=dict(title="unknown 2", range=[-3, 3], scaleanchor="x"))
    save(fig, "lesson5_gibbs_staircase", width=700)


def hmc_vs_walk():
    rho = 0.95
    prec = np.linalg.inv(np.array([[1, rho], [rho, 1]]))
    logp = lambda q: -0.5 * q @ prec @ q
    grad = lambda q: -prec @ q
    rng = np.random.default_rng(1987)
    q = np.array([-2.0, -2.0])
    rw = [q.copy()]
    for _ in range(150):
        prop = q + rng.normal(0, 0.25, 2)
        if np.log(rng.random()) < logp(prop) - logp(q):
            q = prop
        rw.append(q.copy())
    rw = np.array(rw)
    q = np.array([-2.0, -2.0])
    trajs, ends = [], [q.copy()]
    eps, L = 0.12, 22
    for _ in range(10):
        p = rng.normal(size=2)
        qn, pn = q.copy(), p.copy()
        path = [qn.copy()]
        pn = pn + 0.5 * eps * grad(qn)
        for i in range(L):
            qn = qn + eps * pn
            if i < L - 1:
                pn = pn + eps * grad(qn)
            path.append(qn.copy())
        pn = pn + 0.5 * eps * grad(qn)
        if np.log(rng.random()) < (-logp(q) + 0.5 * p @ p) - (-logp(qn) + 0.5 * pn @ pn):
            q = qn
        trajs.append(np.array(path))
        ends.append(q.copy())
    ends = np.array(ends)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.06,
                        subplot_titles=["<b>Random walk</b>: 150 steps", "<b>Skateboarder (HMC)</b>: 10 glides"])
    shrink_subplot_titles(fig)
    for c in (1, 2):
        fig.add_trace(gauss_contours(rho), 1, c)
    fig.add_trace(go.Scatter(x=rw[:, 0], y=rw[:, 1], mode="lines+markers", line=dict(color=ORANGE, width=1.5), marker=dict(size=3)), 1, 1)
    for tr in trajs:
        fig.add_trace(go.Scatter(x=tr[:, 0], y=tr[:, 1], mode="lines", line=dict(color="rgba(128,90,213,0.55)", width=2)), 1, 2)
    fig.add_trace(go.Scatter(x=ends[:, 0], y=ends[:, 1], mode="markers", marker=dict(size=8, color=PURPLE, line=dict(color=DARK, width=1))), 1, 2)
    for c in (1, 2):
        fig.update_xaxes(range=[-3, 3], showticklabels=False, row=1, col=c)
        fig.update_yaxes(range=[-3, 3], showticklabels=False, row=1, col=c)
    base_layout(fig, 440, t=115, b=30, l=20, showlegend=False,
                title=title("Why modern software uses Hamiltonian Monte Carlo",
                            "In a narrow valley small random steps cover only part of it.<br>The skateboarder uses the slope and covers all of it."))
    save(fig, "lesson5_hmc_vs_walk", width=700)


# ---------------------------------------------------------------------------
# 17. Timeline of ideas (vertical: one event per row, readable at any width)
# ---------------------------------------------------------------------------
EVENTS = [
    (1763, "A", "Bayes & Price", "Bayes's essay (published by Price): the area under the posterior curve is the hard part; they could only bound it."),
    (1774, "A", "Laplace", "Laplace's method: approximate the area by a bell curve placed at the peak."),
    (1777, "S", "Buffon's needle", "Buffon: dropping needles has a probability that involves π, so chance can compute geometry."),
    (1908, "S", "'Student' (Gosset)", "W. S. Gosset shuffled 3,000 cards with prisoners' measurements to check his t distribution by simulation."),
    (1934, "S", "Fermi (1930s)", "Enrico Fermi used hand-computed random sampling for neutron problems, but did not publish the method."),
    (1946, "S", "Ulam & von Neumann", "Ulam, playing solitaire while ill, asks: why not estimate the chance of winning by playing many games? With von Neumann he turns it into a method for neutron physics."),
    (1946, "M", "ENIAC", "One of the first electronic computers; first Monte Carlo runs in 1948 (programmed by Klára Dán von Neumann with Nicholas Metropolis)."),
    (1949, "S", "'Monte Carlo' named", "Metropolis & Ulam publish 'The Monte Carlo Method'. Metropolis suggested the name, after the casino."),
    (1951, "S", "rejection sampling", "Von Neumann: throw darts under a box and keep those below the curve."),
    (1951, "A", "Kullback–Leibler", "A measure of distance between distributions, later the heart of variational inference."),
    (1952, "M", "MANIAC", "The Los Alamos computer on which the Metropolis algorithm was first run."),
    (1953, "S", "Metropolis et al.", "Metropolis, Arianna & Marshall Rosenbluth, Augusta & Edward Teller: the walker. MCMC is born."),
    (1970, "S", "Hastings", "W. K. Hastings generalises the walker to lopsided proposals: Metropolis–Hastings."),
    (1984, "S", "Gibbs sampler", "Geman & Geman restore noisy images by updating one unknown at a time."),
    (1986, "A", "Tierney & Kadane", "Accurate Laplace approximations for posterior summaries."),
    (1987, "S", "Hamiltonian MC", "Duane, Kennedy, Pendleton & Roweth: momentum and slope let the walker glide (particle physics)."),
    (1989, "M", "BUGS", "The BUGS project (Cambridge) lets applied researchers write a model and get Gibbs samples."),
    (1990, "S", "Gelfand & Smith", "Sampling solves everyday Bayesian problems: the 'MCMC revolution' begins."),
    (1992, "S", "R-hat (Gelman & Rubin)", "Run several chains and check whether they agree."),
    (1996, "S", "Neal", "Radford Neal brings Hamiltonian Monte Carlo into statistics and machine learning."),
    (1999, "A", "Variational inference", "Jordan, Ghahramani, Jaakkola & Saul: approximate the posterior by the closest simple distribution, found by optimisation."),
    (2003, "M", "PyMC", "PyMC begins as a Python library for Bayesian modelling."),
    (2009, "A", "INLA", "Rue, Martino & Chopin: fast nested Laplace approximations."),
    (2012, "M", "Stan", "Automatic Hamiltonian Monte Carlo for any differentiable model."),
    (2014, "S", "NUTS", "Hoffman & Gelman: the No-U-Turn Sampler decides by itself how far to glide."),
    (2016, "M", "PyMC3", "NUTS and automatic variational inference in Python."),
    (2017, "A", "ADVI", "Kucukelbir et al.: automatic differentiation variational inference (pm.fit in PyMC)."),
    (2021, "S", "new R-hat", "Vehtari, Gelman, Simpson, Carpenter & Bürkner: today's default convergence checks."),
    (2022, "M", "modern PyMC", "PyMC rewritten (versions 4 and later): the library used in this course."),
]


def timeline():
    lanes = {"S": (0, BLUE, "Simulate"), "A": (1, ORANGE, "Approximate"), "M": (2, GREEN, "Computers &<br>software")}
    ev = sorted(EVENTS, key=lambda e: e[0])
    n = len(ev)
    fig = go.Figure()
    for xv in (0, 1, 2):
        fig.add_trace(go.Scatter(x=[xv, xv], y=[0.5, -n + 0.5], mode="lines", line=dict(color="#e2e8f0", width=2), hoverinfo="skip"))
    for key, (xv, col, name) in lanes.items():
        rows = [(i, e) for i, e in enumerate(ev) if e[1] == key]
        fig.add_trace(go.Scatter(
            x=[xv] * len(rows), y=[-i for i, _ in rows], mode="markers+text", text=[e[2] for _, e in rows], textposition="middle right",
            textfont=dict(size=12, color=col), marker=dict(size=12, color=col, line=dict(color="white", width=1.5)),
            hovertext=[f"<b>{e[0]} · {e[2]}</b><br>{e[3]}" for _, e in rows], hoverinfo="text"))
    for era, y0, y1 in [("Los Alamos", 1946, 1953), ("MCMC revolution", 1984, 1992)]:
        idx = [i for i, e in enumerate(ev) if y0 <= e[0] <= y1]
        fig.add_shape(type="rect", x0=-0.3, x1=3.05, y0=-max(idx) - 0.5, y1=-min(idx) + 0.5, fillcolor="rgba(214,158,46,0.10)",
                      line=dict(width=0), layer="below")
        fig.add_annotation(x=3.05, y=-(min(idx) + max(idx)) / 2, text=f"<i>{era}</i>", textangle=90, showarrow=False,
                           font=dict(size=11, color=GOLD), xanchor="right")
    for key, (xv, col, name) in lanes.items():
        fig.add_annotation(x=xv - 0.05, y=0.9, text=f"<b>{name}</b>", showarrow=False, font=dict(size=13, color=col),
                           xanchor="left", yanchor="bottom", align="left")
    height = 30 * n + 190
    base_layout(fig, height, t=105, b=20, l=55, r=10, showlegend=False, hovermode="closest",
                title=title("Two roads around the hard integral, 1763 – today",
                            "One event per row. Hover over (or tap) a point for the story."),
                xaxis=dict(visible=False, range=[-0.3, 3.1], fixedrange=True),
                yaxis=dict(tickvals=[-i for i in range(n)], ticktext=["1930s" if e[0] == 1934 else str(e[0]) for e in ev],
                           range=[-n + 0.4, 2.3], showgrid=False, zeroline=False, fixedrange=True, tickfont=dict(size=12)))
    save(fig, "lesson5_timeline", width=700, height=height)


if __name__ == "__main__":
    shape_not_size()
    answers_are_areas()
    grid_area()
    grid_explosion()
    empty_box()
    darts_pi()
    pile_of_draws()
    rejection()
    weight_vs_place()
    ratio_only()
    king_markov()
    forget_start()
    balanced_traffic()
    walker_hill()
    gibbs_staircase()
    hmc_vs_walk()
    timeline()
