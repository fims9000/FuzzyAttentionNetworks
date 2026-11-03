# PD-ConceptFAN v4: train-only canonical signed concept head

**Executed:** 2026-11-02, GitHub Actions
[Run 41501300941 — successful](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41501300941).
Full aggregate results in its `spectral-task-head-v4-1` artifact:
`task_head_v4_per_run.csv` (45 fitted models),
`task_head_v4_pairwise.csv` (90 independent-model pair comparisons),
`task_head_v4_summary.csv` (all 9 cells),
`train_only_task_head_receipts.json` and `STATUS.json`.
CI also executed focused mathematical regression tests successfully.

## Central experimental question

In v3, the **observable dynamic dimension** and even frozen spectral
concept definitions were repeatable, but the *signed relevance*
`beta_k` independently learned in each model varied substantially.
The same learned factor may receive a positive coefficient in one fit
and a negative coefficient in another. Thus equal concept definitions
do **not** imply stable patient-specific explanations.

v4 tests an additional, **training-time** scientific assumption:
the same signed, task-specific coefficients can be fit ONCE on train
patient labels and train-only spectral concept measurements, then be
shared across independent neural fits. This does not require expert
concept labels, and unlike post-hoc importance scores the shared head
is used directly in the model forward/loss for every training step.

All methods see the same train patient traces, train outcomes, sensor
dictionary and detected rank K. The held-out *test* is not used.

The three methods:
- `spectral_finetuned`: unconstrained independently trained
  signed decision coefficients, fine-tunable spectral operator basis.
- `task_head_prior`: fit a common L2 logistic head from **train-only**
  concept summaries and labels, initialize each network with it,
  regularize deviations; observed concept definitions remain
  trainable.
- `task_head_fixed`: use the identical canonical signed head and
  fixed observed spectral feature definitions across the 5 seeded
  fits; learn concept thresholds, widths and temporal dynamics.
  This method has **less capacity** than the fully free arm.

The common head is mathematically explicit:

```text
q_k(x) = sparse train-derived signed sensor operator
c_k(x) = sigmoid((q_k(x) - tau_k) / width_k) * observability
beta_star, bias_star = L2LogisticRegression(TRAIN named sensor concept summaries, TRAIN outcome)
logit(x) = bias_star + SUM_k(beta_star_k * (c_k(x) - 0.5))
```

This is an exact logit decomposition, a meaningful intervention on
**the mathematical model**, but not on medical treatment/causal
physiology.

## Executed result: 45 models, 3 fresh synthetic generators

Four-digit values below are direct means from
`task_head_v4_summary.csv`. Five different training initialization
seeds per arm. All patient-specific agreement values are corrected
against an independent-patient null that preserves each fitted
concept's empirical case frequencies. They are **not** p-values.

| Planted K (audit only) | Method | Mean outer-val AUPRC | Std AUPRC across 5 seeds | Concept recovery, coverage-adjusted abs corr | Case-specific Spearman excess | Case-specific sign excess |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 3 | spectral_finetuned | 0.442391 | 0.039969 | 0.452221 | 0.053537 | 0.000716 |
| 3 | task_head_fixed | 0.440082 | 0.001670 | 0.475803 | 0.101453 | 0.101208 |
| 3 | task_head_prior | 0.439678 | 0.000880 | 0.468812 | 0.101040 | 0.115336 |
| 5 | spectral_finetuned | 0.435323 | 0.034536 | 0.327082 | 0.070526 | 0.036033 |
| 5 | task_head_fixed | 0.421795 | 0.000359 | 0.340416 | 0.645349 | 0.109792 |
| 5 | task_head_prior | 0.428679 | 0.000977 | 0.334186 | 0.658296 | 0.111128 |
| 7 | spectral_finetuned | 0.486140 | 0.051701 | 0.323443 | -0.032861 | 0.002542 |
| 7 | task_head_fixed | 0.411851 | 0.000105 | 0.322676 | 0.502314 | 0.257161 |
| 7 | task_head_prior | 0.405763 | 0.000709 | 0.324081 | 0.507426 | 0.261043 |

**Nontrivial result:** canonical signed coefficients can remove a large
source of independent-run interpretability instability. For planted
K5 and K7 the mean rank agreement beyond unrelated patients rises
from about 0.071/-0.033 to about 0.645/0.502. The normalized
definition similarity was already >0.99 in the free baseline, so
stabilizing *what a concept means for the task* is an additional
requirement beyond aligning sensor definitions.

**Cost:** AUPRC decreases, most notably on K7 from 0.4861 to
0.4119 fixed or 0.4058 regularized. This invalidates an assertion
of universally superior predictive performance. Free head has
more degrees of freedom, so the reduction in outcome variability
from fixed-head constraints partly follows from the architecture.

**No confirmatory significance:** only 5 initial seeds per method
for one generator per K; crossing within-generator patient metrics
does not create independent trials. v4 was designed after inspecting
earlier runs, so it is exploratory. The common head trains on the
target **outcome**, not ground-truth concept annotations; concepts
are thus task-shaped but not medically named or causally identified.

## Scientific constraints and next acceptance test

The next step is not to increase loss weights until the CI excludes zero.
We must establish **both** (a) genuine reproducible, patient-dependent
concept effects and (b) no unacceptable degradation in AU(P)RC, with
independently collected/generator-separated test cohorts.

For real ICU data, train-only predictive rank is unstable across 1h,
4h and 6h aggregation and was initially not identifiable within
independent train halves due to irregularly observed measurements.
External clinical diagnosis labels or interventions would be needed to
justify clinical names or causal mechanisms. Report honest unknown
semantics rather than inferring that a spectral rank component is
"renal failure" or "respiratory dysfunction" from high loading.

**Conclusion:** v4 makes a bounded, reproducible mathematical
contribution to fixing *model-level signed effects*, but does not
demonstrate full automatic medically valid concept creation or
cross-hospital interpretability.
