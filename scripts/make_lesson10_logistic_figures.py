"""Generate the figures for the Lesson 10 (binary outcomes / logistic regression) pages.

Runs the Bangladesh arsenic-wells logistic regression from the chapter with real
data (data/wells.csv) and saves the plots under figures/lesson10_logistic/.
Also builds the linear-probability failure demo and the separation demo.

Run from the book root with the pymc_env environment:
    /opt/homebrew/Caskroom/miniforge/base/envs/pymc_env/bin/python scripts/make_lesson10_logistic_figures.py
"""

from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "lesson10_logistic"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 2026
plt.style.use("seaborn-v0_8-whitegrid")


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


df = pd.read_csv(ROOT / "data" / "wells.csv")
switch = df["switch"].values
dist100 = df["dist100"].values
arsenic = df["arsenic"].values

# ===========================================================================
# fig 1: the data — switch rate by distance and by arsenic
# ===========================================================================
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
bins = np.quantile(dist100, np.linspace(0, 1, 11))
mids, rates = [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    sel = (dist100 >= lo) & (dist100 <= hi)
    mids.append((lo + hi) / 2)
    rates.append(switch[sel].mean())
axs[0].plot(mids, rates, "o-", color="C0")
axs[0].axhline(switch.mean(), color="gray", ls="--", lw=1, label=f"overall rate {switch.mean():.2f}")
axs[0].set_xlabel("distance to the nearest safe well (100 m)")
axs[0].set_ylabel("share of households that switched")
axs[0].set_title("Farther safe wells -> fewer switches")
axs[0].legend()

bins_a = np.quantile(arsenic, np.linspace(0, 1, 11))
mids_a, rates_a = [], []
for lo, hi in zip(bins_a[:-1], bins_a[1:]):
    sel = (arsenic >= lo) & (arsenic <= hi)
    mids_a.append((lo + hi) / 2)
    rates_a.append(switch[sel].mean())
axs[1].plot(mids_a, rates_a, "o-", color="C3")
axs[1].axhline(switch.mean(), color="gray", ls="--", lw=1)
axs[1].set_xlabel("arsenic in the current well (µg/L)")
axs[1].set_ylabel("share of households that switched")
axs[1].set_title("More contamination -> more switches")
fig.suptitle(f"Bangladesh wells: N = {len(df)} households, {int(switch.sum())} switched "
             f"({switch.mean():.0%})", fontsize=12)
save(fig, "wells_data")

# ===========================================================================
# fig 2: why a straight line fails (linear probability model vs logistic)
# ===========================================================================
import statsmodels.api as sm
import statsmodels.formula.api as smf

lin_fit = smf.ols("switch ~ dist100 + arsenic", data=df).fit()
print("linear probability model:", dict(zip(lin_fit.params.index, lin_fit.params.round(3))))
print("out-of-range predictions:", int((lin_fit.fittedvalues > 1).sum()), "above 1,",
      int((lin_fit.fittedvalues < 0).sum()), "below 0")

logit_fit = smf.logit("switch ~ dist100 + arsenic", data=df).fit(disp=0)

fig, ax = plt.subplots(figsize=(9, 4.5))
xs_a2 = np.linspace(arsenic.min(), arsenic.max(), 200)
for d_val, color, ls in [(0.37, "C3", "-"), (0.10, "C1", "-")]:
    y_lin = (lin_fit.params["Intercept"] + lin_fit.params["dist100"] * d_val
             + lin_fit.params["arsenic"] * xs_a2)
    p_log = logit_fit.predict(pd.DataFrame({"dist100": np.full_like(xs_a2, d_val),
                                            "arsenic": xs_a2}))
    ax.plot(xs_a2, y_lin, ls, color=color, lw=1.6,
            label=f"linear model, dist = {d_val*100:.0f} m")
    ax.plot(xs_a2, p_log, "--", color=color, lw=1.6,
            label=f"logistic model, dist = {d_val*100:.0f} m")
ax.axhline(1, color="gray", lw=0.8, ls=":")
ax.axhline(0, color="gray", lw=0.8, ls=":")
ax.fill_between(xs_a2, 1, 1.3, color="C3", alpha=0.10, label="impossible: p > 1")
ax.set_xlabel("arsenic in the current well (µg/L)")
ax.set_ylabel("P(switch = 1)")
ax.set_ylim(-0.1, 1.32)
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.set_title("The straight line predicts impossible probabilities; the logistic curve does not")
save(fig, "linear_vs_logistic")

# ===========================================================================
# fig 3: the Bayesian logistic model — priors, fit, posterior
# ===========================================================================
with pm.Model() as wells_model:
    alpha = pm.Normal("alpha", mu=0.0, sigma=1.5)
    beta_dist = pm.Normal("beta_dist", mu=0.0, sigma=1.0)
    beta_ars = pm.Normal("beta_ars", mu=0.0, sigma=1.0)
    eta = alpha + beta_dist * dist100 + beta_ars * arsenic
    p = pm.Deterministic("p", pm.math.invlogit(eta))
    y = pm.Bernoulli("y", p=p, observed=switch)
    idata = pm.sample(draws=2000, tune=1000, chains=4, random_seed=SEED,
                      target_accept=0.95, progressbar=False)

summ = az.summary(idata, var_names=["alpha", "beta_dist", "beta_ars"], round_to=2)
print(summ)

az.plot_forest(idata, var_names=["alpha", "beta_dist", "beta_ars"], combined=True)
plt.title("Posterior of the wells model coefficients (log-odds scale)")
save(plt.gcf(), "wells_posterior")

# ===========================================================================
# fig 4: fitted probability curves with uncertainty
# ===========================================================================
post = idata.posterior
a_s = post["alpha"].values.flatten()
bd_s = post["beta_dist"].values.flatten()
ba_s = post["beta_ars"].values.flatten()
rng = np.random.default_rng(SEED)
idx = rng.choice(len(a_s), size=800, replace=False)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
xs_d = np.linspace(0, 3.4, 100)
for ars_val, color in [(0.5, "C7"), (1.66, "C0"), (3.0, "C3")]:
    P = 1 / (1 + np.exp(-(a_s[idx, None] + bd_s[idx, None] * xs_d[None, :] + ba_s[idx, None] * ars_val)))
    m, lo, hi = P.mean(0), np.percentile(P, 3, axis=0), np.percentile(P, 97, axis=0)
    axs[0].plot(xs_d, m, color=color, label=f"arsenic = {ars_val}")
    axs[0].fill_between(xs_d, lo, hi, color=color, alpha=0.15)
axs[0].set_xlabel("distance to the nearest safe well (100 m)")
axs[0].set_ylabel("P(switch)")
axs[0].set_title("By distance, for three arsenic levels")
axs[0].legend(fontsize=9)

xs_a = np.linspace(0.5, 9.0, 100)
for d_val, color in [(0.1, "C0"), (0.5, "C3")]:
    P = 1 / (1 + np.exp(-(a_s[idx, None] + bd_s[idx, None] * d_val + ba_s[idx, None] * xs_a[None, :])))
    m, lo, hi = P.mean(0), np.percentile(P, 3, axis=0), np.percentile(P, 97, axis=0)
    axs[1].plot(xs_a, m, color=color, label=f"dist = {d_val:.0f} ({d_val*100:.0f} m)")
    axs[1].fill_between(xs_a, lo, hi, color=color, alpha=0.15)
axs[1].set_xlabel("arsenic in the current well (µg/L)")
axs[1].set_ylabel("P(switch)")
axs[1].set_title("By arsenic, for two distances")
axs[1].legend(fontsize=9)
fig.suptitle("Fitted switching probability with 94% posterior bands", fontsize=12)
save(fig, "wells_curves")

# ===========================================================================
# fig 5: probability-scale contrasts with HDI
# ===========================================================================
def p_switch(a, bd, ba, d, ars):
    return 1 / (1 + np.exp(-(a + bd * d + ba * ars)))

mean_ars = arsenic.mean()
c1 = p_switch(a_s, bd_s, ba_s, 0.0, mean_ars) - p_switch(a_s, bd_s, ba_s, 1.0, mean_ars)
mean_dist = dist100.mean()
c2 = p_switch(a_s, bd_s, ba_s, mean_dist, 1.0) - p_switch(a_s, bd_s, ba_s, mean_dist, 0.5)

fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, contrast, label in [
        (axs[0], c1, "distance 0 → 100 m\n(at mean arsenic)"),
        (axs[1], c2, "arsenic 0.5 → 1.0\n(at mean distance)")]:
    ax.hist(contrast, bins=60, density=True, color="C0", alpha=0.75, edgecolor="white")
    lo, hi = az.hdi(contrast, prob=0.94)
    ax.axvline(lo, color="C3", ls="--")
    ax.axvline(hi, color="C3", ls="--")
    ax.set_xlabel(f"ΔP(switch)\nmean = {contrast.mean():+.3f}, 94% HDI [{lo:+.3f}, {hi:+.3f}]")
    ax.set_title(label, fontsize=10)
fig.suptitle("Substantive effects on the probability scale", fontsize=12)
save(fig, "wells_contrasts")

def num(name, draws):
    lo, hi = az.hdi(draws, prob=0.94)
    print(f"{name}: mean = {draws.mean():.3f}, 94% HDI [{lo:.2f}, {hi:.2f}]")

print("key numbers:")
num("alpha", post["alpha"].values.flatten())
num("beta_dist", post["beta_dist"].values.flatten())
num("beta_ars", post["beta_ars"].values.flatten())
print("contrast dist 0->100:", c1.mean().round(3), az.hdi(c1, prob=0.94).round(3))
print("contrast arsenic 0.5->1.0:", c2.mean().round(3), az.hdi(c2, prob=0.94).round(3))

# ===========================================================================
# fig 6: complete separation — classical divergence vs Bayesian regularization
# ===========================================================================
x_sep = np.array([-3, -2.5, -2, -1.5, -1.5, -0.7, 1, 1.5, 2, 2.5, 3, 3.2])
y_sep = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

try:
    logit_sep = smf.logit("y ~ x", pd.DataFrame({"x": x_sep, "y": y_sep})).fit(disp=0)
    classical_slope = float(logit_sep.params["x"])
    classical_note = f"ML estimate: {classical_slope:.1f}"
except Exception as e:
    classical_slope = None
    classical_note = "ML fit fails / diverges"

with pm.Model() as sep_model:
    a = pm.Normal("a", 0, 1.5)
    b = pm.Normal("b", 0, 1.0)
    p = pm.math.invlogit(a + b * x_sep)
    pm.Bernoulli("y", p=p, observed=y_sep)
    idata_sep = pm.sample(draws=2000, tune=1000, chains=4, random_seed=SEED,
                          progressbar=False)

b_post = idata_sep.posterior["b"].values.flatten()
a_post = idata_sep.posterior["a"].values.flatten()

xs_s = np.linspace(-3.6, 3.8, 200)
fig, ax = plt.subplots(figsize=(8, 4.2))
rng2 = np.random.default_rng(SEED)
ax.scatter(x_sep + rng2.normal(0, 0.03, len(x_sep)), y_sep + rng2.normal(0, 0.03, len(x_sep)),
           s=60, color="C0", zorder=5)
sel = rng2.choice(len(b_post), 200, replace=False)
for i in sel:
    ax.plot(xs_s, 1 / (1 + np.exp(-(a_post[i] + b_post[i] * xs_s))), color="C2", alpha=0.05, lw=1)
P = 1 / (1 + np.exp(-(a_post[sel, None] + b_post[sel, None] * xs_s[None, :])))
ax.plot(xs_s, P.mean(0), color="C2", lw=2, label="Bayesian posterior mean (Normal(0,1) prior)")
if classical_slope is not None and np.isfinite(classical_slope):
    lo_b, hi_b = az.hdi(b_post, prob=0.94)
    ax.set_title(f"Separated data: ML slope = {classical_slope:.0f} (effectively divergent); "
                 f"Bayesian slope = {b_post.mean():.2f}, 94% HDI [{lo_b:.2f}, {hi_b:.2f}]")
else:
    ax.set_title(f"Separated data: {classical_note}; Bayesian slope stays finite")
ax.set_xlabel("x")
ax.set_ylabel("P(y = 1)")
ax.legend(loc="upper left", fontsize=9)
save(fig, "separation")


# ===========================================================================
# fig 7: interaction model — does the distance effect depend on arsenic?
# ===========================================================================
with pm.Model() as wells_int_model:
    a_i = pm.Normal("a", mu=0.0, sigma=1.5)
    bd_i = pm.Normal("bd", mu=0.0, sigma=1.0)
    ba_i = pm.Normal("ba", mu=0.0, sigma=1.0)
    bda = pm.Normal("bda", mu=0.0, sigma=1.0)
    p_i = pm.math.invlogit(a_i + bd_i * dist100 + ba_i * arsenic + bda * dist100 * arsenic)
    pm.Bernoulli("y", p=p_i, observed=switch)
    idata_int = pm.sample(draws=2000, tune=1000, chains=4, random_seed=SEED,
                          target_accept=0.95, progressbar=False)

po = idata_int.posterior
aI = po["a"].values.flatten(); bdI = po["bd"].values.flatten()
baI = po["ba"].values.flatten(); bdaI = po["bda"].values.flatten()
lo_, hi_ = az.hdi(bdaI, prob=0.94)
print(f"bda: {bdaI.mean():.3f} [{lo_:.2f}, {hi_:.2f}]")

def P_int(d, ar):
    # d, ar: scalars or arrays broadcastable against (n_draws, 1) grids
    return 1 / (1 + np.exp(-(aI[:, None] + bdI[:, None] * d + baI[:, None] * ar
                             + bdaI[:, None] * d * ar)))

rng3 = np.random.default_rng(SEED)
sel3 = rng3.choice(len(aI), 800, replace=False)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
xs_d = np.linspace(0, 3.4, 100)
for ar_val, color in [(0.5, "C7"), (1.66, "C0"), (3.0, "C3")]:
    Pm = P_int(xs_d[None, :], ar_val)[sel3]
    m, lo2, hi2 = Pm.mean(0), np.percentile(Pm, 3, 0), np.percentile(Pm, 97, 0)
    axs[0].plot(xs_d, m, color=color, label=f"arsenic = {ar_val}")
    axs[0].fill_between(xs_d, lo2, hi2, color=color, alpha=0.15)
axs[0].set_xlabel("distance to the nearest safe well (100 m)")
axs[0].set_ylabel("P(switch)")
axs[0].set_title("Interaction model: curves by arsenic level")
axs[0].legend(fontsize=9)

for ar_val, x in [(0.5, 0), (1.66, 1), (3.0, 2)]:
    c = P_int(0.0, ar_val).flatten() - P_int(1.0, ar_val).flatten()
    l2, h2 = az.hdi(c, prob=0.94)
    axs[1].errorbar(x, c.mean(), yerr=[[c.mean() - l2], [h2 - c.mean()]],
                    marker="o", color="C0", capsize=4)
    axs[1].annotate(f"{c.mean():.2f}\n[{l2:.2f}, {h2:.2f}]", (x, c.mean()),
                    textcoords="offset points", xytext=(12, -5), fontsize=9)
axs[1].set_xticks([0, 1, 2]); axs[1].set_xticklabels(["0.5", "1.66", "3.0"])
axs[1].set_xlabel("arsenic level")
axs[1].set_ylabel("ΔP(switch) for 0 → 100 m")
axs[1].set_title("The distance effect grows with contamination")
axs[1].set_ylim(0.0, 0.32)
save(fig, "wells_interaction")

# ===========================================================================
# fig 8: posterior predictive check
# ===========================================================================
with wells_int_model:
    ppc = pm.sample_posterior_predictive(idata_int, var_names=["y"],
                                         random_seed=SEED, progressbar=False)
yrep = ppc.posterior_predictive["y"].values.reshape(-1, len(switch))

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
tot = yrep.mean(1)
axs[0].hist(tot, bins=50, color="C0", alpha=0.75, edgecolor="white", density=True)
axs[0].axvline(switch.mean(), color="C3", lw=2,
               label=f"observed {switch.mean():.3f}")
l3, h3 = az.hdi(tot, prob=0.94)
axs[0].axvline(l3, color="gray", ls="--"); axs[0].axvline(h3, color="gray", ls="--")
axs[0].set_xlabel("replicated share of switchers")
axs[0].set_title("Total switch rate: model vs data")
axs[0].legend(fontsize=9)

q = np.quantile(dist100, np.linspace(0, 1, 11))
mids_b, obs_b, lo_b, hi_b = [], [], [], []
for lo4, hi4 in zip(q[:-1], q[1:]):
    sel = (dist100 >= lo4) & (dist100 <= hi4)
    mids_b.append((lo4 + hi4) / 2)
    obs_b.append(switch[sel].mean())
    rep_b = yrep[:, sel].mean(1)
    l4, h4 = az.hdi(rep_b, prob=0.94)
    lo_b.append(l4); hi_b.append(h4)
mean_b = [(yrep[:, (dist100 >= lo4) & (dist100 <= hi4)]).mean(1).mean()
          for lo4, hi4 in zip(q[:-1], q[1:])]
axs[1].errorbar(mids_b, mean_b,
                yerr=[np.array(mean_b) - np.array(lo_b),
                      np.array(hi_b) - np.array(mean_b)],
                marker="s", color="C0", capsize=3, label="model (94% band)")
axs[1].plot(mids_b, obs_b, "o-", color="C3", label="observed")
axs[1].set_xlabel("distance to the nearest safe well (100 m)")
axs[1].set_ylabel("share of switchers")
axs[1].set_title("By distance decile: aggregate fits, edges bend")
axs[1].legend(fontsize=9)
fig.suptitle("Posterior predictive checks of the interaction model", fontsize=12)
save(fig, "wells_ppc")

print("PPC total: mean", tot.mean().round(4), "94% HDI", az.hdi(tot, prob=0.94).round(4),
      "observed", switch.mean().round(4))

print("separation slope: ML =", classical_slope, "| Bayesian mean =", b_post.mean().round(2),
      "HDI", az.hdi(b_post, prob=0.94).round(2))

print("\nAll figures saved to", OUT)
