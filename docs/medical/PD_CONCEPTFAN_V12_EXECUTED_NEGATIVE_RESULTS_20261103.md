# ConceptFAN V12: executed falsification and reproducibility audit (2026-11-03)

> **Scientific verdict:** V12 is an implemented and unit-verified architectural
> hypothesis, **not** a demonstrated improvement in functional concept
> identifiability. All original PIERE/V10/V11 findings remain untouched.
> The Set B PhysioNet cohort was consumed in V11's one-shot 41710356212 and
> was **never** read, scored or tuned on by V12.

## 1. Frozen code, provenance, and scientific scope

- New separate research [PR #8](https://github.com/fims9000/FuzzyAttentionNetworks/pull/8).
- [**Synthetic 3/5/7 generator run 41720328037: SUCCESS**](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41720328037).
  Immutable aggregate artifact `conceptfan-v12-new-seed-oracle-3-5-7-1`:
  36 fitted models, 4 methods × 3 seeds × 3 factor-count regimes.
- [V12 mathematical unit/regression CI 41720254720: SUCCESS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41720254720):
  37 focused tests and **197 full-discovery tests** passed, one warning.
- 3/5/7 different new generator seeds were selected before scoring;
  **96?** (see source certificate for exact patient bootstrap counts) and
  199 train-only label permutations were performed on their training data.
  Oracle hidden trajectories were stripped from the training object and
  opened only for scoring learned concept summaries on heldout synthetic
  validation cases; their true count did not guide computational K.
- Historic PhysioNet **Set A DEV is repeatedly inspected** and is NOT
  a new external validation. The first full 13-fit run
  [41719424100](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41719424100)
  completed model fits and printed all AUPRCs but failed at the aggregate
  serialization stage because of a Python variable-shadowing typo.
  Fixed in separate commit with scientific model/loss/config Git blob SHA
  held unchanged. A second A-ONLY run completes the report, without
  using Set B. The rerun is NOT an independent cohort.
- **Independent MAGNETS-style method:** self-developed mask/time
  aggregation + a binary classification concept bottleneck, *inspired*
  by Forest/Wei/Fink's regression MAGNETS; it is **NOT** their released
  reference implementation or an official reproduced MAGNETS score.
  Comparisons should be described as `MAGNETS-style binary adaptation`,
  not as a refutation or reproduction of their published regression model.

## 2. New-seed planted-factor synthetic results

**Metric definitions:** predictive AUPRC is on heldout generated validation
episodes; `oracle |rho|` is an *optimistically Hungarian-matched*, mean
absolute Spearman agreement of learned episode concept summaries with
known last-six-step latent-factor means. The matching is performed
**only at evaluation time** and **overestimates** naive semantic matching
quality; scrambled-patient matched controls are reported alongside.
Every model uses exactly the rank computed from TRAIN observations,
not planted K.

| True latent factors | Model (3 seeds) | Mean validation AUPRC | Optimistic mean abs oracle rho | Scrambled-patient matched control |
|---:|---|---:|---:|---:|
| 3 | V11 projector + group | 0.3551 | 0.6415 | 0.0990 |
| 3 | V12 observable anchor | 0.3749 | 0.6485 | 0.0971 |
| 3 | V12 anchor + signed contributions | 0.3746 | 0.6487 | 0.0973 |
| 3 | MAGNETS-style binary adaptation | 0.3891 | 0.0625 | 0.0198 |
| 5 | V11 projector + group | 0.4032 | 0.5109 | 0.1056 |
| 5 | V12 observable anchor | 0.4238 | 0.5088 | 0.1050 |
| 5 | V12 anchor + signed contributions | 0.4231 | 0.5088 | 0.1050 |
| 5 | MAGNETS-style binary adaptation | 0.4227 | 0.0204 | 0.0225 |
| 7 | V11 projector + group | 0.4243 | 0.4219 | 0.1132 |
| 7 | V12 observable anchor | 0.4251 | 0.4249 | 0.1140 |
| 7 | V12 anchor + signed contributions | 0.4247 | 0.4249 | 0.1140 |
| 7 | MAGNETS-style binary adaptation | 0.4478 | 0.1048 | 0.0341 |

Detected computational predictive source rank: **3, 5, 7**, matching
planted count respectively in these controlled regimes. This does **not**
establish general automated identifiability of clinical latent factors.
MAGNETS-style masks may encode localized patterns not aligned to
last-six-step latent means; poor matched `rho` here does not prove the
official MAGNETS method lacks coherent temporal concepts.

## 3. The primary failure criterion: signed functional singleton count

Require **every one** of the 3 independent seed pairs to meet
same-patient signed effect Spearman >=0.5 and learned concept-definition
projector overlap >=0.75, for a TRAIN-only structurally resolved axis.
Reject a stable group as proof of the identities of its individual axes.
No patient-specific signed-effect test uses an artificially sign-corrected
Hungarian assignment.

| Planted factor count | V11 | V12 anchor only | V12 anchor + signed |
|---:|---:|---:|---:|
| 3 | 1 | 1 | 1 |
| 5 | **1** | **0** | **0** |
| 7 | 0 individual + 1 stable group | 0 individual + 1 stable group | 0 individual + 1 stable group |

**Result: no increase in independent, reproducible signed singleton effects.**
AUPRC improvements at K=3/5 must not be advertised as concept identity
improvement. No new individual medical or physical concepts were
identified. Unresolved SVD groups can rotate and are only group-level
computational constructs.

### Train-only signed risk anchor: all null

The train-only risk-sign gate uses:
- orientation of *certified singleton axes* only;
- 96 stratified patient bootstrap resamples and 199 label permutations;
- 90% bootstrap interval excluding zero, two-sided Bonferroni p threshold,
  and absolute training class-difference >=0.02.

**Zero certified risk signs** for K=3, K=5, and K=7. The training penalty
against risk coefficient sign therefore had **exactly no active signed
slots** in this experiment; the `V12 anchor + signed` arm differs from
`anchor-only` principally because of the direct two-view contribution
consistency, NOT because clinical or synthetic task directions were
statistically identified. E.g., K=3 strongest observed class difference
0.02126 had unadjusted permutation p=0.15 and cannot be certified.

This is a meaningful negative control, not a missing result: forcing
risk-sign labels from weak sample associations would give apparently
stable signs **by construction** and invalidate the scientific question.

## 4. What this proves / fails to prove

**Established in code and falsification:**
- Native, train-time concept bottleneck; no direct encoder-to-logit
  shortcut or post-hoc explanatory network.
- Fixed measured causal-past operator dictionary, observed masks,
  train-only source dimension, explicit sensor/temporal definitions.
- Gradients pass from new direct-effect terms to native learned concept
  and output parameters; masking/no-history controls are tested.
- When source axes cannot be empirically resolved, individual identities
  and task signs ABSTAIN instead of being artificially fixed.
- Random-label/no-signal fixtures do not automatically manufacture
  robust risk-sign certificates.

**Not established:**
- Increased count of reproducible signed individual effects versus V11;
  the actual synthetic study contradicts this target.
- Strict superiority or equality of predictive performance under
  three seeds and one synthetic generator family.
- Unsupervised uniqueness of nonlinear latent factors or clinically
  meaningful factor names. True K on a controlled simulator is not
  general patient-level identifiability.
- Transfer across hospitals, datasets, calendar shifts or causal
  interventions, let alone clinical readiness.

**Research correction:** A stronger hypothesis would seek *task-relevant
effect-identifiable* functional directions rather than impose
orientations and signs on covariance singular vectors. Additional
independent interventions or multiple environments may be required to
distinguish correlated physiological factors. New experiments should
remain in a separate version/branch, and **must not use Set B again
as a fresh holdout**.

## 5. Evidence and provenance

- [Synthetic 3/5/7 confirmed run](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41720328037)
- `src/conceptfan_discovery/oracle_synthetic_v12.py`,
  `src/conceptfan_discovery/signed_effect_v12.py`,
  `src/conceptfan_discovery/train_signed_effect_v12.py`,
  `src/conceptfan_discovery/magnets_binary_baseline_v12.py`.
- Train-only generator rank groups, learning vs oracle vs shuffled
  patient control, all per-seed results are in the archived ZIP,
  not inferred from the PIERE article.
- The accepted PIERE manuscript must not be retrospectively changed to
  claim these V12 experiments or improvement.
