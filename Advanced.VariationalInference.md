# Variational inference: fast posterior approximation with an accuracy cost

## 1. The computational problem

For parameters or latent quantities $\theta$ and observed data $y$, Bayesian inference targets

$$
p(\theta\mid y)
=\frac{p(y,\theta)}{p(y)}.
$$

MCMC constructs draws whose long-run distribution is the target posterior. Variational inference (VI) instead chooses a tractable distribution $q_\phi(\theta)$ from an approximating family and adjusts its parameters $\phi$ so that $q_\phi$ is close to the posterior.

The computational problem becomes

$$
q_{\phi^*}(\theta)
=\arg\min_{q_\phi\in\mathcal Q}
\operatorname{KL}\!\left(q_\phi(\theta)\,\|\,p(\theta\mid y)\right).
$$

This is an optimization problem. Samples drawn afterward come from the optimized approximation $q_{\phi^*}$, not directly from the exact posterior.

## 2. KL divergence and the evidence lower bound

The usual VI objective uses

$$
\operatorname{KL}(q\|p)
=\mathbb E_q\!\left[
\log\frac{q_\phi(\theta)}{p(\theta\mid y)}
\right].
$$

For discrete distributions this is

$$
D_{\mathrm{KL}}(q\|p)
=\sum_{\theta}q(\theta)
\log\frac{q(\theta)}{p(\theta)},
$$

with the sum replaced by an integral for continuous densities. KL divergence is non-negative and is zero only when the distributions agree almost everywhere. It is **not a distance metric**: in general,

$$
D_{\mathrm{KL}}(q\|p)\ne D_{\mathrm{KL}}(p\|q),
$$

and it does not satisfy the symmetry or triangle requirements of a metric.

### Why the direction matters in VI

The expectation in $D_{\mathrm{KL}}(q\|p)$ is taken under the approximation $q$. Consequently:

- putting appreciable $q$ mass where the target posterior $p$ is nearly zero is very costly;
- failing to cover a region where $p$ has mass can be comparatively cheap when $q$ assigns that region almost no mass;
- among several separated posterior modes, a simple $q$ may settle around one mode rather than spread probability across low-density space;
- posterior tails and interval width can therefore be underestimated.

The reverse direction, $D_{\mathrm{KL}}(p\|q)$, averages under $p$ and strongly penalizes an approximation that assigns too little density to regions visited by the target. It is often described as more *mass covering*. Neither direction is universally superior; each defines a different approximation problem.

:::{admonition} Translating the linked KL article to variational inference
:class: note
The [Towards Data Science visual explanation](https://towardsdatascience.com/understanding-kl-divergence-f3ddc8dff254/) is useful for seeing that swapping the reference and comparison distributions changes the answer. Its application is production-data drift, however. In this chapter the target $p(\theta\mid y)$ is the posterior and $q_\phi(\theta)$ is its computational approximation. Also retain the mathematical term **divergence**, not “distance metric.”
:::

Because the marginal likelihood $p(y)$ is generally unknown, VI maximizes the evidence lower bound (ELBO):

$$
\mathcal L(\phi)
=\mathbb E_{q_\phi}[\log p(y,\theta)]
-\mathbb E_{q_\phi}[\log q_\phi(\theta)].
$$

The identity

$$
\log p(y)
=\mathcal L(\phi)
+\operatorname{KL}\!\left(q_\phi(\theta)\,\|\,p(\theta\mid y)\right)
$$

shows why maximizing the ELBO minimizes the KL divergence: $\log p(y)$ does not depend on $\phi$.

The first ELBO term rewards agreement with the joint model; the second rewards entropy and resists collapse. A larger ELBO indicates a better solution **within the chosen approximating family, parameterization, target model, and optimization setup**. Stochastic estimates of the ELBO may be noisy. It does not prove that the approximation is scientifically adequate or justify using ELBO as a universal model-selection score.

## 3. Mean-field and full-rank approximations

In a mean-field approximation,

$$
q_\phi(\theta)=\prod_{j=1}^{J}q_{\phi_j}(\theta_j),
$$

so posterior dependence is removed in the approximation. This can be fast and memory-efficient, but it is risky when the estimand depends on correlations, funnels, ridges, multimodality, or uncertain group-level scales.

A full-rank Gaussian approximation retains a covariance matrix. It can represent linear dependence but costs more and still cannot represent every skewed, heavy-tailed, bounded, or multimodal posterior.

| Feature | Mean-field VI | Full-rank VI | NUTS/MCMC target |
|---|---|---|---|
| Marginal location | Often useful | Often useful | Asymptotically exact under valid sampling |
| Posterior correlations | Suppressed | Linear dependence retained | Retained |
| Skew or heavy tails | Limited | Limited | Retained if explored |
| Multiple separated modes | Commonly missed | Commonly missed | Possible, but also computationally difficult |
| Speed on large models | Often strongest | Intermediate | Often slowest |
| Standard MCMC diagnostics | Not applicable | Not applicable | Required |

The direction $\operatorname{KL}(q\|p)$ penalizes placing mass where the target has little mass more strongly than failing to cover every target region. This can produce a mode-seeking approximation with intervals that are too narrow.

## 4. Automatic differentiation variational inference

Automatic differentiation variational inference (ADVI) automates transformations to an unconstrained space, differentiates the variational objective, and optimizes it with stochastic gradients. The same probabilistic model can therefore be paired with either MCMC or VI.

Mean-field factorization does not imply that every implementation uses textbook coordinate-ascent updates. PyMC's ADVI uses gradient-based stochastic optimization and the reparameterization idea; random initialization and gradient noise can lead different runs to different solutions. A fixed seed makes a particular computation reproducible, not the objective globally solved.

Current PyMC exposes mean-field ADVI, full-rank ADVI, and the `pm.fit` interface. The exact API should be checked against the [current PyMC documentation](https://www.pymc.io/projects/docs/en/stable/api/vi.html) when the notebook is implemented.

```python
import pymc as pm

with model:
    approximation = pm.fit(
        n=30_000,
        method="advi",
        random_seed=2026,
    )
    idata_vi = approximation.sample(
        draws=2_000,
        random_seed=2026,
    )
```

Do not copy legacy `pymc3` imports from older courses. The model definition, predictive quantities, dimensions, and coordinates should follow the same current-PyMC conventions as the rest of this book.

## 5. VI versus MCMC is not “fast versus slow” alone

Prefer VI when:

- the model or dataset makes full MCMC impractical;
- approximate predictions are sufficient for the declared use;
- the important posterior functionals have been validated;
- rapid iteration or an initialization/prototyping stage has real value.

Prefer a trusted MCMC fit when:

- tail probabilities, interval coverage, correlations, group-level scales, or multiple modes are central;
- decisions are sensitive to underestimated uncertainty;
- the model is small enough for reliable NUTS sampling;
- the analysis is confirmatory and approximation error is not negligible.

VI may also be used for exploration or initialization followed by a complete MCMC fit for the final report. This is a workflow decision, not a claim that one algorithm is universally superior. MCMC does not ordinarily “refine only the tails” of a VI fit: unless a specialized valid method is constructed, the full Markov chain must target the full posterior.

## 6. Validation protocol for VI

The final ELBO and an attractive posterior plot are insufficient. Validate the approximation in layers:

1. **Known-target check:** start with a conjugate or simulated model whose posterior is known.
2. **Repeated optimization:** run multiple random initializations and compare the achieved objectives and substantive summaries.
3. **Reference inference:** compare key marginals, correlations, tail probabilities, and predictions with a trusted MCMC fit on the full data or a manageable subset.
4. **Prior and posterior predictive checks:** verify observable implications as for any other fitted model.
5. **Simulation calibration:** across repeated fake datasets, evaluate bias and interval coverage for the actual estimand.
6. **Decision sensitivity:** recompute the intended action under VI and the reference posterior.

Do not report MCMC $\widehat R$, MCMC ESS, or divergence counts as evidence that VI is accurate. VI needs optimization and approximation checks. Posterior predictive adequacy alone also cannot show that parameter uncertainty is correct.

## 7. Cognitive-science example

Consider a trial-level response-time model with crossed participant and item effects. Mean-field VI may estimate the population mean efficiently while understating uncertainty in participant/item scales or ignoring posterior dependence between intercepts and slopes.

The comparison should therefore focus on named quantities:

- the population condition contrast;
- participant and item scale parameters;
- a tail probability for a new response;
- prediction for a new participant and a new item;
- the decision derived from those predictions, if any.

“The VI posterior looks similar” is not enough. State which estimands and predictive distributions agree, by how much, and which discrepancies remain.

## 8. Common mistakes

- Describing VI draws as exact posterior draws.
- Assuming that a stable ELBO proves convergence to the best approximation.
- Comparing only posterior means while ignoring scale, tails, dependence, and decisions.
- Treating mean-field independence as a harmless computational detail.
- Calling VI deterministic or assuming one optimization run is sufficient.
- Using ELBO values from unrelated models as automatic scientific rankings.
- Applying MCMC diagnostics to VI output and interpreting them in the usual way.
- Choosing VI only because it is available, without a computational need or validation plan.

## 9. Exercise

Fit the Session 6 Beta–Binomial benchmark with ADVI and NUTS. Compare the posterior mean, standard deviation, 95% interval, $P(\theta>0.65\mid y)$, and posterior predictive distribution. Then repeat ADVI with at least three random starts.

:::{dropdown} What matters most in the comparison?
Agreement of posterior means is the weakest check. Tail probabilities, interval width, predictive uncertainty, and stability across starts reveal errors that a mean-only comparison can hide. Because the exact Beta posterior is known, this exercise checks the approximation rather than merely comparing two opaque algorithms.
:::

## Sources

- Kucukelbir et al. (2017), [Automatic Differentiation Variational Inference](https://www.jmlr.org/papers/v18/16-107.html).
- Blei, Kucukelbir, and McAuliffe (2017), [Variational Inference: A Review for Statisticians](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1285773).
- Kullback and Leibler (1951), [On Information and Sufficiency](https://doi.org/10.1214/aoms/1177729694).
- [PyMC variational-inference API](https://www.pymc.io/projects/docs/en/stable/api/vi.html).
- Dhinakaran (2023), [Understanding KL Divergence](https://towardsdatascience.com/understanding-kl-divergence-f3ddc8dff254/) (supplementary visual intuition; drift-monitoring context).
