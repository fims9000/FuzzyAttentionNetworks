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
  **96 stratified training-patient bootstraps** and **199 train-only label
  permutations** were performed for singleton risk-sign testing in each regime.
  Oracle hidden trajectories were stripped from the training object and
  opened only for scoring learned concept summaries on heldout synthetic
  validation cases; their true count did not guide computational K.
- Historic PhysioNet **Set A DEV is repeatedly inspected** and is NOT
  a new external validation. The first full 13-fit run
  [41719424100](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41719424100)
  completed model fits and printed all AUPRCs but failed at the aggregate
  serialization stage because of a Python variable-shadowing typo.
  Fixed in separate commit with scientific model/loss/config Git blob SHA
  held unchanged. [**A-only report repair 41721295743: SUCCESS**](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41721295743),
  artifact `conceptfan-v12-seta-development-report-fixed-1`, completed
  every 13-fit aggregate and strict signed-effect certificate, **without
  using Set B**. The rerun is NOT an independent cohort.
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

## 4. Matched Set A historical DEV: an exploratory but different signal

[**Successfully completed full report-only rerun 41721295743**](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41721295743).
Models/losses/data configs and 3 independent seeds were byte-identical to
the first execution; only the local Python variable shadowing in the report
writer was fixed. The same 2,400 Set A TRAIN / previously seen **300 DEV**
records were reused. **No Set B patient records or outcomes were accessed.**

| Method | Mean AUPRC original DEV | AUROC | MCAR AUPRC | MAR AUPRC | Independent fits |
|---|---:|---:|---:|---:|---:|
| V11 group | **0.368722** | 0.727021 | 0.313020 | 0.376609 | 3 |
| V12 observable anchor | 0.355270 | 0.722591 | 0.300449 | 0.374383 | 3 |
| V12 anchor + signed effects | 0.353549 | 0.720715 | 0.301294 | 0.369991 | 3 |
| MAGNETS-style binary adaptation | **0.404660** | **0.781561** | **0.357957** | **0.384526** | 3 |
| HistGradientBoosting | 0.313649 | 0.746493 | 0.317102 | 0.294133 | 1 |

V12 signed vs V11 original AUPRC difference **-0.015173**,
descriptive paired patient/model bootstrap 95% interval
**[-0.064502,+0.024792]**, spanning zero. V12 signed versus
MAGNETS-style: **-0.051112**, descriptive interval
**[-0.168427,+0.057507]**. Repeatedly inspected DEV plus 3
initializations do **NOT** support population-level superiority,
inferiority, noninferiority or a p-value claim.

### Strict source identity and signed effects, on same earlier DEV cases

| Method | Certified reproducible individual singleton effects | Multi-axis group effects | Abstained groups |
|---|---:|---:|---:|
| V11 group | **1** | 0 | 6 |
| V12 observable anchor | **2** | 0 | 5 |
| V12 anchor + signed effects | **2** | 0 | 5 |

The surviving **source slots 3 and 4** pass all 3 pair comparisons
for both V12 methods, while V11 passed only slot 4.
For the full V12 signed arm, minimum signed same-patient Spearman
across its 3 seed pairs was **0.9197** on slot 3 and **0.9004**
on slot 4; minimum learned definition-projector overlap was
**0.8586** and **0.8951**, respectively. This gain over V11 was
observed **ONLY on the repeatedly inspected historical Set A DEV** and
was **NOT reproduced** in the different new-seed synthetic regimes
above (at K=5, V12 had fewer robust singletons than V11).

On this Set A TRAIN sample the Bonferroni patient-label tests selected
three risk coefficient signs for source slots **0:-, 4:-, 7:+**.
**Slot 3 was not given a forced task sign**, despite passing the DEV
strict effect test; conversely, the forced 0/7 signs did not make those
slots pass the joint DEV direct-effect/definition screen.
Therefore even the positive historical DEV effect cannot be
interpreted as a mathematical guarantee from forced beta signs.

The independent MAGNETS-style binary adaptation attained mean
signed, train-only Hungarian matched DEV same-patient Spearman
**0.8807** across 24 matched slot/pair comparisons (min **0.7634**);
all 24 correlations were positive. **These 24 scores share only
three trained model instances**. Strong cross-model effect correlation
after matching is NOT proof of distinct concepts, genuine semantic
factor recovery, clinical validity, low redundancy, or a direct
comparison with official regression MAGNETS. The new-seed synthetic
oracle correlation of the adapted method (0.020–0.105) was weak even
when some AUPRCs were competitive.

This creates a clear contradiction worth reporting, not erasing:
**on the old DEV, V12 has two rather than one stable mathematical
effect; on unseen synthetic factor regimes, it has no advantage
and sometimes performs worse.** The valid conclusion is that V12
remains exploratory and the central improvement hypothesis has
NOT generalized.

## 5. What this proves / fails to prove

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

## 6. Evidence and provenance

- [Synthetic 3/5/7 confirmed run](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41720328037)
- [Completed Set A previous DEV report](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41721295743)
- `src/conceptfan_discovery/oracle_synthetic_v12.py`,
  `src/conceptfan_discovery/signed_effect_v12.py`,
  `src/conceptfan_discovery/train_signed_effect_v12.py`,
  `src/conceptfan_discovery/magnets_binary_baseline_v12.py`.
- Train-only generator rank groups, learning vs oracle vs shuffled
  patient control, all per-seed results are in the archived ZIP,
  not inferred from the PIERE article.
- The accepted PIERE manuscript must not be retrospectively changed to
  claim these V12 experiments or improvement.
