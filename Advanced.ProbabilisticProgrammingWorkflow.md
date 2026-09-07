# Probabilistic programming and the Bayesian workflow

## 1. What probabilistic programming adds

A probabilistic program expresses a generative model in executable form. The analyst specifies random variables, deterministic relationships, observed data, and derived quantities; an inference engine then applies an algorithm such as NUTS or variational inference.

Probabilistic programming separates, but does not disconnect:

- **model specification:** what observations and latent quantities mean;
- **inference:** how the posterior is approximated;
- **criticism and use:** whether the fitted model is adequate for the scientific and decision task.

Successful execution proves only that code ran. It does not establish that the construct, likelihood, prior, hierarchy, approximation, causal interpretation, or deployment decision is valid.

## 2. The workflow is iterative

A complete Bayesian workflow is not a one-way “prior + likelihood = posterior” pipeline:

```text
question and measurement
        ↓
estimand and prediction target
        ↓
generative model and prior predictive simulation
        ↓
computation and algorithm-specific diagnostics
        ↓
posterior understanding and posterior predictive checks
        ↓
sensitivity, comparison, expansion, or approximation audit
        ↓
prediction, decision, communication, and reproducibility
        ↖──────────────────────────── revise when needed
```

Model revision is expected. The audit trail should retain why each revision occurred and which data features motivated it.

## 3. How Sessions 1–14 already implement the workflow

| Workflow question | Course owner |
|---|---|
| What is measured, on which unit, for which population? | Session 1 |
| What probability statements and updates are required? | Session 2 |
| How could the observations be generated? | Session 3 |
| What do the priors imply before observing outcomes? | Session 4 |
| Can the planned design recover and calibrate the estimand? | Session 5 |
| Can posterior expectations be computed reliably? | Session 6 |
| Which likelihood, predictor structure, link, and hierarchy are needed? | Sessions 7–10 |
| Does computation work, does the model fit, and how does it predict? | Session 11 |
| What causal, missingness, and generalization assumptions limit use? | Sessions 12–13 |
| Can another researcher reproduce and defend the analysis? | Session 14 |

The Coursera workflow module is therefore best used as a synthesis check. Its coin-bias example is pedagogically useful but too simple to reveal participant/item dependence, measurement problems, or prediction for new groups. This book uses a small crossed participant–item example for the advanced audit.

## 4. One PyMC model, two inference engines

The model should be defined once using named dimensions and transformed quantities:

```python
import pymc as pm

coords = {
    "participant": participant_ids,
    "item": item_ids,
    "observation": range(len(data)),
}

with pm.Model(coords=coords) as model:
    participant_idx = pm.Data("participant_idx", p_idx, dims="observation")
    item_idx = pm.Data("item_idx", i_idx, dims="observation")
    condition = pm.Data("condition", x, dims="observation")

    alpha = pm.Normal("alpha", 0, 1)
    beta = pm.Normal("beta", 0, 0.5)
    sigma_p = pm.Exponential("sigma_p", 1)
    sigma_i = pm.Exponential("sigma_i", 1)
    z_p = pm.Normal("z_p", 0, 1, dims="participant")
    z_i = pm.Normal("z_i", 0, 1, dims="item")

    eta = alpha + beta * condition + sigma_p * z_p[participant_idx] + sigma_i * z_i[item_idx]
    response = pm.Bernoulli("response", logit_p=eta, observed=y, dims="observation")
```

NUTS and VI can be applied to this same model, but their validation evidence differs. MCMC requires chain and geometry diagnostics. VI requires optimization stability and approximation validation. Both require prior and posterior predictive checking.

## 5. Workflow record for probabilistic programs

Every reported program should make the following inspectable:

1. data provenance and preprocessing;
2. coordinates, observational units, and grouping levels;
3. likelihood support and link function;
4. prior rationale and prior predictive checks;
5. named estimands and draw-wise transformations;
6. inference algorithm and settings;
7. algorithm-appropriate diagnostics;
8. posterior predictive checks tied to possible model failures;
9. prediction target and validation split;
10. model, prior, and approximation sensitivity;
11. action/loss specification when a decision is made;
12. frozen environment, seeds, versions, and saved inference artifacts.

## 6. End-to-end audit exercise

Take one existing project model and answer four questions before changing any code:

1. **Meaning:** What construct, indicator, unit, and estimand does each important variable represent?
2. **Generative adequacy:** Which observable pattern would contradict the likelihood or hierarchy?
3. **Computational adequacy:** What evidence is required for the chosen inference engine?
4. **Use:** Is the output a description, prediction, causal claim, or decision—and what additional assumptions does that use require?

Then execute or inspect the workflow in this order:

```text
fake-data recovery → prior prediction → fit → computational checks
→ posterior summaries → targeted PPC → sensitivity/comparison
→ predictive decision → reproducibility record
```

The deliverable is a short workflow map plus one revision justified by evidence. Complexity is not rewarded by itself.

## 7. Strengths and limitations

### Strengths

- the model is explicit and generative;
- derived estimands can be computed from draws;
- hierarchical structure can match the design;
- inference engines can be exchanged while retaining the model;
- predictive simulation becomes a routine part of analysis.

### Limitations

- a flexible language permits scientifically incoherent models;
- default priors and sampler settings are not automatically appropriate;
- inference may fail silently or approximately;
- reproducible code can reproduce a biased measurement or design;
- model comparison cannot create causal identification or transportability.

## 8. Course integration decision

Do not create a second required “Bayesian workflow” lecture that repeats every session. Instead:

- Session 6 links forward to VI as an alternative inference engine;
- Session 11 links forward to decision theory after predictive comparison;
- Session 14 uses this page as the optional final workflow audit;
- notebooks use current `pymc` syntax rather than legacy `pymc3` syntax.

## Sources

- Gelman et al., [Bayesian Workflow](https://arxiv.org/abs/2011.01808).
- [PyMC documentation](https://www.pymc.io/projects/docs/en/stable/).
- University of Pittsburgh, [Advanced Bayesian Methods and Applications](https://www.coursera.org/learn/advanced-bayesian-methods-and-applications-odc/), Module 4 public outline.
