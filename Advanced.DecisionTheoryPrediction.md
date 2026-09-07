# Bayesian decision theory and prediction

## 1. A posterior is not yet a decision

A posterior distribution represents uncertainty about unknown quantities. A posterior predictive distribution represents uncertainty about unobserved outcomes. Neither specifies what should be done.

A decision problem requires:

- a set of possible actions $a\in\mathcal A$;
- uncertain states or future outcomes $\omega$ under each action;
- a loss $L(a,\omega)$, or equivalently a utility $U(a,\omega)$;
- a posterior or posterior predictive distribution describing uncertainty about $\omega$.

In the general case, an action can change the distribution of the outcome. The posterior expected loss of action $a$ is therefore

$$
\rho(a\mid y)
=\mathbb E[L(a,\omega)\mid y]
=\int L(a,\omega)p(\omega\mid a,y)\,d\omega.
$$

When the action merely reports an estimate or class label and cannot change the state, $p(\omega\mid a,y)=p(\omega\mid y)$. For treatment, policy, or intervention decisions, the action-specific distribution is essential and usually requires causal assumptions in addition to a predictive model.

The Bayes action is

$$
a^*(y)=\arg\min_{a\in\mathcal A}\rho(a\mid y).
$$

When utility is used, choose the action with the largest posterior expected utility. This framework separates factual uncertainty encoded by the model from value judgments encoded by losses, utilities, constraints, and available actions.

:::{admonition} Terminology: posterior expected loss is not always “Bayes risk”
:class: note
Some introductory accounts use *Bayes risk* for the conditional quantity $\rho(a\mid y)$. In formal statistical decision theory, *risk* usually averages loss over possible data at a fixed state, and *Bayes risk* then averages that risk over the prior. To avoid ambiguity, this chapter uses **posterior expected loss** for the quantity minimized after observing $y$.
:::

## 2. Prediction as a decision problem

Suppose the uncertain future outcome is $\widetilde y$ and the action is a point prediction $a$.

| Loss | Bayes point prediction |
|---|---|
| Squared error $(a-\widetilde y)^2$ | Posterior predictive mean |
| Absolute error $|a-\widetilde y|$ | Posterior predictive median |
| Asymmetric absolute error | A posterior predictive quantile determined by relative costs |
| Zero–one classification loss | Most probable class |

There is therefore no universally optimal point prediction. The same posterior predictive distribution can imply different actions under different consequences of over- and under-prediction.

For asymmetric loss

$$
L(a,\widetilde y)
=c_{\text{under}}(\widetilde y-a)_+
+c_{\text{over}}(a-\widetilde y)_+,
$$

the optimal prediction is the quantile

$$
F^{-1}_{\widetilde y\mid y}\!\left(
\frac{c_{\text{under}}}{c_{\text{under}}+c_{\text{over}}}
\right).
$$

## 3. Classification thresholds follow consequences

Let $p=P(\widetilde y=1\mid y)$, with false-negative cost $c_{FN}$ and false-positive cost $c_{FP}$. Predict or act as if positive when

$$
c_{FP}(1-p)<c_{FN}p,
$$

or equivalently

$$
p>\frac{c_{FP}}{c_{FP}+c_{FN}}.
$$

A threshold of $0.5$ is optimal only under a particular symmetric loss. A screening problem with costly missed impairment may justify a lower action threshold, while unnecessary invasive follow-up may push it upward.

This does not authorize the statistical model to define the costs. Ethical, clinical, institutional, and distributional judgments must be supplied and defended separately.

## 4. From posterior draws to expected loss

Decision analysis can usually be implemented directly with posterior predictive draws $\widetilde y^{(s)}$:

$$
\widehat\rho(a\mid y)
=\frac{1}{S}\sum_{s=1}^{S}L(a,\widetilde y^{(s)}).
$$

```python
import numpy as np

actions = np.array(["do_not_intervene", "intervene"])

risk_no = false_negative_cost * predictive_probability
risk_yes = false_positive_cost * (1 - predictive_probability)

chosen_action = actions[np.argmin([risk_no, risk_yes])]
```

For continuous or multi-stage decisions, evaluate the loss draw by draw for every admissible action. Report the distribution of consequences as well as the expected value when tail risk matters.

## 5. Multi-objective decisions

Real decisions may involve predictive accuracy, money, time, burden, fairness, safety, and scientific information. A weighted loss such as

$$
L(a,\omega)
=w_1L_1(a,\omega)+w_2L_2(a,\omega)
$$

is coherent only when the components and weights have an interpretable scale. Hiding contested values inside arbitrary weights does not make a decision objective.

If losses remain vector valued, action $a_1$ **dominates** $a_2$ when it is no worse on every objective and better on at least one. The **Pareto frontier** contains the non-dominated actions. It exposes trade-offs but does not select one action without additional preferences or constraints. A weighted sum, or *scalarization*, makes that choice explicit but should be accompanied by sensitivity analysis over defensible weights.

Required practice:

1. list the affected parties and feasible actions;
2. define each consequence and its time horizon;
3. justify any conversion to a common scale;
4. inspect sensitivity across defensible weights;
5. identify actions that are dominated across the relevant range;
6. record constraints that cannot be traded away.

## 6. Decision analysis versus model comparison

Session 11 compares predictive distributions. Decision theory asks how predictions will be used.

| Question | Appropriate tool |
|---|---|
| Which model better predicts a declared held-out unit? | Cross-validation or external validation |
| Where does a model contradict observable data? | Posterior predictive checking |
| Which action has the smallest expected consequence? | Posterior expected loss |
| Is more information worth collecting? | Expected value of information |

A small ELPD advantage need not change the preferred action. Conversely, two models with similar aggregate prediction scores may imply different actions in the region where losses are concentrated. Compare decisions under all scientifically credible models rather than feeding only the top-ranked model into the loss function.

## 7. Notebook audit: a variance–standard-deviation trap

The downloaded course lab uses a Normal prior for a policy effect and a Normal likelihood for a sample mean. With prior $\theta\sim\mathcal N(\mu_0,v_0)$, known observation standard deviation $\sigma$, sample mean $\bar y$, and sample size $n$,

$$
v_n=\left(\frac{1}{v_0}+\frac{n}{\sigma^2}\right)^{-1},
\qquad
\mu_n=v_n\left(\frac{\mu_0}{v_0}+\frac{n\bar y}{\sigma^2}\right).
$$

The first expression is the posterior **variance**, not its standard deviation. SciPy's `norm` expects the standard deviation in `scale`, so the safe implementation is:

```python
post_var = 1 / (1 / prior_var + n / observation_sd**2)
post_mean = post_var * (
    prior_mean / prior_var + n * sample_mean / observation_sd**2
)
post_sd = np.sqrt(post_var)
posterior = stats.norm(loc=post_mean, scale=post_sd)
```

Passing `post_var` as `scale` makes the posterior too narrow and alters expected-loss calculations. This is exactly why decision code should be checked with fake posterior draws, unit tests for parameterization, and sensitivity analysis—not only by whether it produces a plausible recommendation.

For the lab values $\mu_0=5$, $v_0=16$, $\bar y=3$, $\sigma=3$, and $n=15$, the corrected posterior is approximately

$$
\theta\mid y\sim\mathcal N(3.07,\,0.76^2).
$$

The example also assumes that policy benefit is linear in the true effect and that all implementation consequences have been captured by one fixed cost. Those are decision-model assumptions, not conclusions supplied by Bayes' theorem.

## 8. Applied exercise: a cognitive-screening policy

A clinic must choose whether a positive screen triggers an additional assessment. Use posterior predictive draws for true impairment and future screening outcomes.

Compare at least two policies under:

- false-negative harm;
- false-positive burden;
- capacity cost of follow-up assessment;
- uncertainty in prevalence and test performance;
- an explicit equity constraint or subgroup audit.

Report the action threshold, expected loss for every policy, sensitivity to the cost ratio, and the consequences of model misspecification. Do not present the result as a clinical recommendation without domain authority and external validation.

## 9. Common mistakes

- Calling a posterior mean “the Bayesian decision” without naming a loss.
- Treating an arbitrary 0.5 probability threshold as universal.
- Using $p(\omega\mid y)$ when the action itself changes outcomes and $p(\omega\mid a,y)$ is required.
- Optimizing predictive accuracy when the real cost is asymmetric or subgroup-specific.
- Encoding ethical choices as hidden numerical weights.
- Ignoring uncertainty in the loss function or deployment population.
- Selecting a model first and concealing decision sensitivity across plausible models.
- Reporting only expected loss when rare catastrophic consequences matter.

## 10. Protocol addition

An optional decision section in the final project should state:

1. action set and decision maker;
2. uncertain future state and time horizon;
3. posterior predictive quantity used;
4. loss/utility function and source of its values;
5. expected loss for each action;
6. sensitivity to model, prior, population, and loss assumptions;
7. ethical constraints and parties bearing each error.

## Sources

- Gelman et al., *Bayesian Data Analysis*, 3rd ed., Chapter 9: decision analysis using posterior and posterior predictive distributions.
- University of Pittsburgh, [Advanced Bayesian Methods and Applications](https://www.coursera.org/learn/advanced-bayesian-methods-and-applications-odc/), Module 2 public outline.
