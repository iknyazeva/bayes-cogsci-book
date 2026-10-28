"""Generate the figures for the Part III "Mixed Models vs. Hierarchical Bayes" pages.

Recreates the key plots of the two Going Further workbooks
(09_mixed_models_linear, 10_mixed_models_logistic) and saves them as PNG files
under figures/part3_mixed/. The figures use English labels and are shared by the
English and Russian pages (the same convention as the interactive _static figures).

Run from the book root with the pymc_env environment:
    /opt/homebrew/Caskroom/miniforge/base/envs/pymc_env/bin/python scripts/make_part3_mixed_figures.py
"""

from __future__ import annotations

from pathlib import Path

import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.graphics.gofplots import qqplot

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "part3_mixed"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 2026
plt.style.use("seaborn-v0_8-whitegrid")


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ===========================================================================
# Part 1: sleepstudy (linear mixed models)
# ===========================================================================
data = pd.read_csv(ROOT / "data" / "sleepstudy.csv")
data["Subject"] = data["Subject"].astype("category")
subjects = list(data["Subject"].unique())

# fig 1: subject panel
fig, axes = plt.subplots(2, 9, figsize=(16, 7), sharey=True, sharex=True)
for ax, subj in zip(axes.ravel(), subjects):
    sub = data[data["Subject"] == subj]
    ax.scatter(sub["Days"], sub["Reaction"], color="C0", ec="black", alpha=0.7, s=22)
    m, b = np.polyfit(sub["Days"], sub["Reaction"], 1)
    ax.plot(sub["Days"], b + m * sub["Days"], color="C3", lw=1.5)
    ax.set_title(str(subj), fontsize=9)
fig.suptitle("sleepstudy: reaction time vs days of sleep deprivation (OLS line per subject)", fontsize=13)
fig.supxlabel("Days")
fig.supylabel("Reaction (ms)")
save(fig, "sleep_panel")

# fig 2: residuals of the pooled OLS model
lm_basic = smf.ols("Reaction ~ Days", data=data).fit()
fig, axs = plt.subplots(1, 3, figsize=(11, 3))
axs[0].hist(lm_basic.resid, bins=25, color="C0", alpha=0.75, edgecolor="white")
axs[0].set_title("residual histogram")
qqplot(np.asarray(lm_basic.resid), line="s", ax=axs[1], markersize=3)
axs[1].set_title("QQ-plot vs Normal")
axs[2].scatter(range(len(lm_basic.resid)), lm_basic.resid, s=8, alpha=0.6, color="C0")
axs[2].set_title("residuals in row order")
fig.suptitle("Pooled OLS (Model 1): residuals")
save(fig, "ols_resid_diagnostics")

# fig 3: OLS prediction bands (correct labels)
pred = lm_basic.get_prediction().summary_frame()
order = np.argsort(data["Days"].values)
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.scatter(data["Days"], data["Reaction"], color="C0", ec="black", alpha=0.5, s=20, label="data")
ax.plot(data["Days"].values[order], pred["mean"].values[order], "r-", label="OLS fit")
ax.plot(data["Days"].values[order], pred["mean_ci_lower"].values[order], "r--", lw=1, label="95% CI for the mean")
ax.plot(data["Days"].values[order], pred["mean_ci_upper"].values[order], "r--", lw=1)
ax.plot(data["Days"].values[order], pred["obs_ci_lower"].values[order], "g--", lw=1, label="95% interval for a new obs")
ax.plot(data["Days"].values[order], pred["obs_ci_upper"].values[order], "g--", lw=1)
ax.set_xlabel("Days"); ax.set_ylabel("Reaction (ms)"); ax.legend()
save(fig, "ols_prediction_bands")

# fig 4: subject-dummy model fits
lm_sub = smf.ols("Reaction ~ Days + C(Subject)", data=data).fit()
data["fitted_sub"] = lm_sub.fittedvalues
fig, ax = plt.subplots(figsize=(11, 5))
ax.scatter(data["Days"], data["Reaction"], color="C0", ec="black", alpha=0.4, s=18)
sns.lineplot(x="Days", y="fitted_sub", hue="Subject", data=data, ax=ax, palette="tab20", lw=1.2, legend=False)
ax.set_title("Model 2 (subject dummies): parallel per-subject lines")
save(fig, "dummy_fits")

# fig 5: RI model
lm_ri = smf.mixedlm("Reaction ~ Days", data, groups=data["Subject"]).fit(reml=True)
re_ri = pd.DataFrame(lm_ri.random_effects).T
re_ri.columns = ["b_i"]
fig, ax = plt.subplots(figsize=(11, 3.5))
re_ri["b_i"].plot(kind="bar", ax=ax, color="C0", edgecolor="black")
ax.axhline(0, color="C3", lw=1)
ax.set_ylabel("BLUP (ms)")
ax.set_title("Random-intercept model: predicted subject deviations")
save(fig, "ri_blups")

data["fitted_ri"] = lm_ri.fittedvalues
fig, ax = plt.subplots(figsize=(11, 5))
ax.scatter(data["Days"], data["Reaction"], color="C0", ec="black", alpha=0.4, s=18)
sns.lineplot(x="Days", y="fitted_ri", hue="Subject", data=data, ax=ax, palette="tab20", lw=1.2, legend=False)
ax.set_title("Random-intercept model: common slope, subject offsets")
save(fig, "ri_fits")

data["resid_ri"] = lm_ri.resid
fig, axs = plt.subplots(1, 2, figsize=(12, 3.5))
sns.boxplot(x="Days", y="resid_ri", data=data, ax=axs[0], color="C0")
axs[0].set_title("residuals by day (trend remains)")
sns.boxplot(x="Subject", y="resid_ri", data=data, ax=axs[1], color="C0")
axs[1].tick_params(axis="x", labelrotation=90)
axs[1].set_title("residuals by subject")
save(fig, "ri_resid_boxes")

# fig 6: RIS model
lm_ris = smf.mixedlm("Reaction ~ Days", data, groups=data["Subject"], re_formula="~ Days").fit(reml=True)
cov_re = lm_ris.cov_re
re_ris = pd.DataFrame(lm_ris.random_effects).T
re_ris.columns = ["b_i", "c_i"]
corr_model = cov_re.iloc[0, 1] / np.sqrt(cov_re.iloc[0, 0] * cov_re.iloc[1, 1])

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.scatter(re_ris["b_i"], re_ris["c_i"], color="C0", ec="black", alpha=0.8)
for subj, row in re_ris.iterrows():
    ax.annotate(str(subj), (row["b_i"], row["c_i"]), fontsize=7, alpha=0.7)
ax.set_xlabel("BLUP intercept deviation (ms)")
ax.set_ylabel("BLUP slope deviation (ms/day)")
ax.set_title(f"BLUPs: sample corr = {np.corrcoef(re_ris.iloc[:,0], re_ris.iloc[:,1])[0,1]:.2f}, "
             f"model rho = {corr_model:.2f}")
save(fig, "ris_blup_scatter")

data["resid_ris"] = lm_ris.resid
fig, axs = plt.subplots(1, 2, figsize=(12, 3.5))
sns.boxplot(x="Days", y="resid_ris", data=data, ax=axs[0], color="C0")
axs[0].set_title("residuals by day (improved)")
sns.boxplot(x="Subject", y="resid_ris", data=data, ax=axs[1], color="C0")
axs[1].tick_params(axis="x", labelrotation=90)
axs[1].set_title("residuals by subject (improved)")
save(fig, "ris_resid_boxes")

# fig 7: correct vs corner CI band
def fixed_effect_ci_band(res, xs):
    X = np.column_stack([np.ones_like(xs), xs])
    beta = res.fe_params.values[:2]
    V = res.cov_params().loc[["Intercept", "Days"], ["Intercept", "Days"]].values
    mean = X @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", X, V, X))
    crit = stats.norm.ppf(0.975)
    return mean, mean - crit * se, mean + crit * se

xs = np.linspace(0, 9, 100)
mean, lo, hi = fixed_effect_ci_band(lm_ris, xs)
ci = lm_ris.conf_int().loc[["Intercept", "Days"]]
corner_lo = ci.loc["Intercept", 0] + ci.loc["Days", 0] * xs
corner_hi = ci.loc["Intercept", 1] + ci.loc["Days", 1] * xs
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.scatter(data["Days"], data["Reaction"], color="C0", ec="black", alpha=0.35, s=18, label="data")
ax.plot(xs, mean, "r-", label="fixed-effects mean")
ax.fill_between(xs, lo, hi, color="C3", alpha=0.35, label="correct pointwise 95% CI")
ax.plot(xs, corner_lo, ":", color="gray", lw=2, label="corner-plugging band (wrong)")
ax.plot(xs, corner_hi, ":", color="gray", lw=2)
ax.set_xlabel("Days"); ax.set_ylabel("Reaction (ms)"); ax.legend()
ax.set_title("Population line: correct pointwise CI vs the corner shortcut")
save(fig, "ci_band")

# fig 8-10: Bambi fits
model1 = bmb.Model("Reaction ~ 1 + Days + (1 | Subject)", data, categorical="Subject")
idata1 = model1.fit(draws=1000, tune=1000, chains=4, random_seed=SEED, progressbar=False,
                    idata_kwargs={"log_likelihood": True})
model2 = bmb.Model("Reaction ~ 1 + Days + (Days | Subject)", data, categorical="Subject")
idata2 = model2.fit(draws=1000, tune=1000, chains=4, random_seed=SEED, progressbar=False,
                    idata_kwargs={"log_likelihood": True})

cmp_df = az.compare({"RI: (1|Subject)": idata1, "RIS: (Days|Subject)": idata2})
az.plot_compare(cmp_df)
save(plt.gcf(), "loo_compare_linear")

post_ds = idata2.posterior.to_dataset()
_, ax = plt.subplots(figsize=(5.5, 4.5))
post_ds.plot.scatter(x="1|Subject", y="Days|Subject", hue="Subject__factor_dim",
                     add_colorbar=False, add_legend=False, cmap="tab20", s=4, alpha=0.15, ax=ax)
ax.set_xlabel("posterior intercept deviation (ms)")
ax.set_ylabel("posterior slope deviation (ms/day)")
ax.set_title("Joint posterior of subject effects (RIS model)")
save(plt.gcf(), "posterior_effects")

b_draws = post_ds["1|Subject"].values.reshape(-1, len(subjects))
c_draws = post_ds["Days|Subject"].values.reshape(-1, len(subjects))

def forest(ax, draws, xlabel):
    order = np.argsort(draws.mean(axis=0))
    for pos, idx in enumerate(order):
        l, h = az.hdi(draws[:, idx])
        ax.hlines(pos, l, h, color="C0", lw=2, alpha=0.8)
        ax.plot(draws[:, idx].mean(), pos, "o", color="C3", ms=4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([str(subjects[i]) for i in order], fontsize=7)
    ax.set_xlabel(xlabel)

fig, axs = plt.subplots(1, 2, figsize=(10, 6))
forest(axs[0], b_draws, "posterior intercept deviation (ms)")
axs[0].set_title("Posterior subject intercepts (94% HDI)")
forest(axs[1], c_draws, "posterior slope deviation (ms/day)")
axs[1].set_title("Posterior subject slopes (94% HDI)")
save(fig, "forest_effects")

# ===========================================================================
# Part 2: depression trial (binary GLMM)
# ===========================================================================
data_depr = pd.read_csv(ROOT / "data" / "Depression.dat", sep=r"\s+")
agg = data_depr.groupby(["drug", "severity", "time"])["outcome"].mean().reset_index()

sns.catplot(x="time", y="outcome", hue="drug", col="severity", data=agg, kind="bar",
            palette="rocket", legend_out=False, height=3.6, aspect=1.1)
plt.suptitle("Proportion normal by week, drug and baseline severity", y=1.05)
save(plt.gcf(), "depr_props")

basic1 = smf.logit("outcome ~ drug", data=data_depr).fit(disp=0)
basic2 = smf.logit("outcome ~ drug + time + severity + drug:time", data=data_depr).fit(disp=0)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
axs[0].scatter(basic1.fittedvalues, basic1.resid_generalized, s=8, alpha=0.5, color="C0")
axs[0].set_title("drug-only model: residuals vs fitted")
axs[1].scatter(basic2.fittedvalues, basic2.resid_generalized, s=8, alpha=0.5, color="C0")
axs[1].set_title("full flat model: residuals vs fitted")
for ax in axs:
    ax.set_xlabel("fitted probability")
save(fig, "depr_resid_flat")

m1 = bmb.Model("outcome ~ drug", data_depr, family="bernoulli")
i1 = m1.fit(draws=1000, tune=1000, chains=4, random_seed=SEED, progressbar=False,
            idata_kwargs={"log_likelihood": True})
m2 = bmb.Model("outcome ~ drug + time + severity + drug:time", data_depr, family="bernoulli")
i2 = m2.fit(draws=1000, tune=1000, chains=4, random_seed=SEED, progressbar=False,
            idata_kwargs={"log_likelihood": True})
m3 = bmb.Model("outcome ~ drug + time + severity + drug:time + (1 | case)", data_depr,
               family="bernoulli")
i3 = m3.fit(draws=1000, tune=1000, chains=4, random_seed=SEED, target_accept=0.95,
            progressbar=False, idata_kwargs={"log_likelihood": True})

cmp2 = az.compare({"M1: drug only (flat)": i1,
                   "M2: full fixed (flat)": i2,
                   "M3: + (1|case) GLMM": i3})
az.plot_compare(cmp2)
save(plt.gcf(), "depr_loo")

# posterior of the case-level SD
sig = i3.posterior.to_dataset()["1|case_sigma"].values.flatten()
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(sig, bins=40, color="C0", alpha=0.75, edgecolor="white", density=True)
l, h = az.hdi(sig)
ax.axvline(l, color="C3", ls="--", label=f"94% HDI [{l:.2f}, {h:.2f}]")
ax.axvline(h, color="C3", ls="--")
ax.set_xlabel("1|case_sigma (logit scale)")
ax.set_title("Posterior of the patient-level SD (depression GLMM)")
ax.legend()
save(fig, "depr_sigma_case")

# probability-scale report
def cell_probability(idata, severity, drug, t):
    post = idata.posterior.to_dataset()
    eta = (post["Intercept"] + drug * post["drug"] + t * post["time"]
           + severity * post["severity"] + drug * t * post["drug:time"])
    return (1 / (1 + np.exp(-eta))).values.flatten()

rows = []
for sev, sev_name in [(0, "mild"), (1, "severe")]:
    for dr, dr_name in [(0, "standard"), (1, "new")]:
        for t, week in [(0, 1), (1, 2), (2, 4)]:
            p = cell_probability(i3, sev, dr, t)
            lo_, hi_ = az.hdi(p, prob=0.94)
            rows.append({"severity": sev_name, "drug": dr_name, "week": week,
                         "p_mean": p.mean(), "lo": lo_, "hi": hi_})
rep = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, sev, name in zip(axes, [0, 1], ["initially mild", "initially severe"]):
    sub = rep[rep["severity"] == ("mild" if name == "initially mild" else "severe")]
    for dr, color, label in [(0, "C0", "standard"), (1, "C3", "new")]:
        ss = sub[sub["drug"] == label]
        ax.errorbar(ss["week"], ss["p_mean"],
                    yerr=[ss["p_mean"] - ss["lo"], ss["hi"] - ss["p_mean"]],
                    marker="o", color=color, capsize=3, label=label)
    ax.set_title(name)
    ax.set_xlabel("week")
    ax.set_xticks([1, 2, 4])
axes[0].set_ylabel("P(normal | typical patient)")
axes[0].legend()
fig.suptitle("Depression GLMM: posterior P(normal) by drug and week, 94% HDI")
save(fig, "depr_probs")

# print the numbers for the page text
print("\nP(normal) report:")
print(rep.round(3).to_string(index=False))
for sev, name in [(0, "mild"), (1, "severe")]:
    p_new = cell_probability(i3, sev, 1, 2)
    p_std = cell_probability(i3, sev, 0, 2)
    d = p_new - p_std
    lo_, hi_ = az.hdi(d, prob=0.94)
    print(f"week4 {name}: delta(new - standard) = {d.mean():+.3f} [{lo_:+.2f}, {hi_:+.2f}]")

# ===========================================================================
# Part 3: ANES
# ===========================================================================
clinton = pd.read_csv(ROOT / "data" / "anes_clinton.csv")
cm = bmb.Model("vote['clinton'] ~ party_id + party_id:age", clinton, family="bernoulli")
ic = cm.fit(draws=1000, tune=1000, chains=4, random_seed=SEED, target_accept=0.85,
            progressbar=False, idata_kwargs={"log_likelihood": True})

post_c = ic.posterior.to_dataset()
levels = ["democrat", "independent", "republican"]
slopes = {p: post_c["party_id:age"].sel(**{"party_id:age_dim": p}).values.flatten() for p in levels}
fig, ax = plt.subplots(figsize=(7, 3.8))
for i, p in enumerate(levels):
    ax.hist(slopes[p], bins=60, alpha=0.55, density=True, label=p, color=f"C{i}")
ax.set_xlabel("posterior age slope (log-odds per year)")
ax.set_title("Party-specific age slopes (ANES 2016, Clinton vote)")
ax.legend()
save(fig, "anes_slopes")

print("\nANES per-decade probability effects:")
for p in levels:
    eta = post_c["Intercept"].values.flatten()
    if p != "democrat":
        eta = eta + post_c["party_id"].sel(**{"party_id_dim": p}).values.flatten()
    p_hat = 1 / (1 + np.exp(-eta))
    d10 = slopes[p] * 10 * p_hat * (1 - p_hat)
    lo_, hi_ = az.hdi(d10, prob=0.94)
    print(f"  {p:12s}: {d10.mean():+.3f} [{lo_:+.3f}, {hi_:+.3f}]")

print("\nAll figures saved to", OUT)
