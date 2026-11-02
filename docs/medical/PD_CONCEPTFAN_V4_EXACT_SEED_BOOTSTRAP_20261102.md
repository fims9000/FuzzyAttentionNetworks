# PD-ConceptFAN v4: dependent-pair uncertainty corrected by exact model-index bootstrap

**Frozen data:** 45 training runs from
[GitHub Actions 41501300941](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41501300941),
`spectral-task-head-v4-1` artifact.

**Independent verifier:** [GitHub Actions 41502755029](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41502755029),
`exact-v4-model-bootstrap-1` artifact. Tests passed. It includes
`exact_five_model_bootstrap_v4.csv` (63 full method/metric/K results)
and `full_bootstrap_receipt.json`.

## Why the first table is not a statistically independent 10-pair trial

For each K and method, five independently initialized fitted models
produce (inom{5}{2}=10) model-pair similarity values. Every model
belongs to four pairs; ten pairwise values do not create ten independent
trained models. A naive 10-pair bootstrap would be pseudo-replication.

The exact algorithm enumerates all (5^5=3125) ordered model-resampling
draws with replacement, computes weights
(w_{ij}=n_in_j) for **distinct** original models, and excludes five
all-identical bootstrap replicates without any distinct model pair.
The same sampled five model IDs are used in both model arms when
computing *paired* differences. For AUPRC, bootstrap original
seed-matched per-model metrics directly (all 3125 draws included).

This is **model-only bootstrap conditional on fixed patients and one
synthetic generator**. It is not a clinical confidence interval or
a confirmatory p-value. Five models offer limited inference.

## Corrected results for shared train-only signed classifier vs free readout

Differences refer to shared head MINUS freely fitted head. Each
number comes from exact enumeration of the original five source models,
not invented patient-level attributions.

| True latent K used ONLY for audit | Head | Δ case-specific signed Spearman agreement | 95% model-bootstrap interval | Δ AUPRC | 95% model-bootstrap interval |
| ---: | --- | ---: | --- | ---: | --- |
| 3 | frozen | +0.047916 | [-0.095720, +0.137679] | -0.002309 | [-0.032932, +0.029307] |
| 3 | prior regularized | +0.047503 | [-0.093735, +0.137679] | -0.002712 | [-0.032841, +0.028906] |
| 5 | frozen | +0.574823 | [+0.495720, +0.681090] | -0.013527 | [-0.044540, +0.008180] |
| 5 | prior regularized | +0.587770 | [+0.511230, +0.693990] | -0.006643 | [-0.037020, +0.014590] |
| 7 | frozen | +0.535176 | [+0.408800, +0.646030] | -0.074289 | [-0.113140, -0.032110] |
| 7 | prior regularized | +0.540287 | [+0.409220, +0.651720] | -0.080377 | [-0.119680, -0.038090] |

Rounded to six digits; the verified artifact retains full precision.

- At K3, adding common signed weights does **not** give an
  unambiguous improvement in case-specific rank agreement; the
  interval crosses zero.
- At K5, stability improves, while prediction performance loss
  is not resolved by five-model bootstrap.
- At K7, case-specific rank agreement improves *alongside a
  substantial AUPRC decrease*. This is a real scientific
  tradeoff, not a universally better architecture.
- The head was fitted on TRAIN outcome labels, so concepts remain
  **task-dependent** but are *not* supervised by human concept
  labels or synthetic ground-truth latent factors.
- The frozen head/readout and frozen source directions reduce
  capacity and stabilize some parameters **by construction**.
  The case-specific chance-corrected agreement is more informative
  than raw agreement, but does not establish physiological semantics
  or real treatment effects.

## Relation to existing ConceptFAN paper and inferential boundaries

The original paper's negative conclusion (exact attribution does not
imply semantic completeness) remains valid. These are new
experimental models and must not silently replace the accepted
frozen ConceptFAN results.

No new medical test set has been inspected, no external hospital
replication has occurred, and no uniquely identified physiological
concept or clinical intervention effect is claimed.
