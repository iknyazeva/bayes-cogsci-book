# Optional advanced block: inference, decisions, and Bayesian workflow

:::{admonition} Independent-study extension
:class: note
This block is optional and sits outside the required 14-session, 56-hour course. It assumes familiarity with posterior simulation, PyMC, computational diagnostics, posterior prediction, and model checking through Session 11.
:::

## Organizing question

> **What changes when exact posterior computation is too expensive, predictions must be converted into actions, and an entire analysis must be expressed as a reproducible probabilistic program?**

The block connects three topics:

1. **Variational inference:** approximate a posterior through optimization when full MCMC is too costly.
2. **Bayesian decision theory and prediction:** combine posterior or posterior-predictive uncertainty with an explicit loss or utility function.
3. **Probabilistic programming and Bayesian workflow:** express the generative model in code and carry it through simulation, computation, criticism, comparison, and use.

These are related but not interchangeable. Variational inference changes the **computational approximation**. Decision theory changes how uncertainty is converted into an **action**. Probabilistic programming changes how the model and workflow are **represented and executed**.

## Learning outcomes

After completing the block, students should be able to:

- explain VI as posterior approximation by optimization rather than sampling from the exact posterior;
- derive the evidence lower bound and explain the direction of the usual KL divergence;
- distinguish mean-field and full-rank approximating families;
- identify posterior features that VI may miss and design a validation strategy;
- define actions, uncertain states, losses, posterior expected loss, and a Bayes action;
- explain why a predictive distribution is not itself a decision;
- derive common point predictions from squared, absolute, and asymmetric loss;
- implement a complete PyMC workflow without confusing successful computation with scientific adequacy.

## Where the block attaches to the course

| Required course material | Advanced extension |
|---|---|
| Session 5: simulation and calibration | Simulation-based assessment of approximation error and decision performance |
| Session 6: MCMC, HMC, and NUTS | {doc}`Advanced.VariationalInference` |
| Sessions 7–10: applied and hierarchical models | A realistic model whose posterior may be expensive to approximate |
| Session 11: PPC, cross-validation, and model uncertainty | {doc}`Advanced.DecisionTheoryPrediction` |
| Session 14: complete analysis protocol | {doc}`Advanced.ProbabilisticProgrammingWorkflow` |

## Audit of the Coursera source

The design was informed by the University of Pittsburgh Coursera course [Advanced Bayesian Methods and Applications](https://www.coursera.org/learn/advanced-bayesian-methods-and-applications-odc/). Its public outline and the instructor-provided local copies of the Module 1–3 transcripts and guided notebooks were reviewed on 3 September 2026.

| Coursera module | Publicly listed content | BayesBook decision |
|---|---|---|
| **1. Advanced Bayesian Inference** | Why VI is needed; KL divergence; VI core; mean-field approximation; VI versus MCMC | Create a dedicated VI chapter after Session 6 and add stronger approximation-validation cautions |
| **2. Bayesian Decision Theory & Prediction** | Loss functions; realistic and multi-objective loss; prediction as a decision problem; connection with machine learning | Create a decision chapter after predictive comparison and before the capstone decision statement |
| **4. Probabilistic Programming and Bayesian Workflow** | Definition of probabilistic programming; PyMC; Bayesian workflow; end-to-end coin-bias example; strengths and limitations | Treat as synthesis rather than a new inferential method; use current PyMC syntax and a repeated-measures example |

### What was retained, corrected, or deferred

| Course-material idea | Editorial decision |
|---|---|
| VI changes posterior computation into optimization and trades some accuracy for speed | Retain, but require a named computational need and approximation validation |
| KL divergence is asymmetric | Expand the explanation: $D_{\mathrm{KL}}(q\|p)$ is not a metric and its direction helps explain mode-seeking undercoverage |
| VI gives the same answer every time | Correct: ADVI commonly uses stochastic gradients and can reach different solutions across seeds or initializations |
| Mean-field VI can be updated by coordinate ascent | Present as one classical algorithm, not as the implementation rule for PyMC ADVI |
| MCMC can refine only selected tails or modes after VI | Qualify: a valid MCMC run ordinarily targets the full posterior; VI may initialize or precede it |
| Bayesian decisions minimize expected loss | Retain, while distinguishing posterior expected loss from the formal prior-averaged meaning of Bayes risk |
| Multi-objective loss can use weighted sums or a Pareto frontier | Retain both; require transparent weights, constraints, and sensitivity analysis |
| The Module 2 policy notebook | Use as a code-audit example because it passes a posterior variance where SciPy expects a standard deviation |

The course's Module 3 transcripts and Gaussian-process notebook were also reviewed. Gaussian processes and Dirichlet processes are not imported into this block: each needs its own scientific motivation, kernel or partition-prior implications, computational checks, and applied cognitive-science case. The supplied Gaussian-process notebook is a useful sketch, but it uses `scikit-learn`, labels an uncertainty band as a confidence interval, and does not demonstrate a PyMC Bayesian workflow. It should seed a later optional chapter rather than be inserted as if the topic were already covered.

## Recommended route

1. Read {doc}`Advanced.VariationalInference` and compare VI with NUTS on a model whose exact posterior or trusted MCMC fit is available.
2. Read {doc}`Advanced.DecisionTheoryPrediction` and make one decision from the same posterior predictive distribution under two different loss functions.
3. Use {doc}`Advanced.ProbabilisticProgrammingWorkflow` to audit the entire analysis from construct and estimand to final action and reproducibility record.

## Assessment boundary

This block is ungraded by default. A defensible optional submission contains:

- a declared estimand and prediction target;
- a trusted reference fit or known simulated target;
- a VI approximation with repeated starts and explicit validation;
- one posterior-predictive decision under a justified loss function;
- a short account of which conclusions change across inference methods or losses.

Fast computation, a smooth loss curve, or elaborate model code cannot substitute for this evidence.

## Source boundary

The Coursera outline supplies the topic sequence. Mathematical and implementation claims in the subchapters are checked against the [PyMC variational-inference API](https://www.pymc.io/projects/docs/en/stable/api/vi.html), Kucukelbir et al.'s [ADVI paper](https://www.jmlr.org/papers/v18/16-107.html), Blei et al.'s [review of variational inference](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1285773), *Bayesian Data Analysis* Chapter 9, and Gelman et al.'s [Bayesian workflow](https://arxiv.org/abs/2011.01808).
