# Measurement missingness is not a physiological concept: executed v5-v6 evidence

**Date:** 2026-11-02. **Status:** exploratory, validated software and actual
train/validation experiments; not independent clinical discovery or medical
concept identification. Previous accepted PIERE/ConceptFAN research remains
unchanged. No changes to historical test or calibration splits.

## Mathematical defect fixed in v6

The old ConceptFAN formula computed a native concept at time t by
`C_old[t,k] = gate[k] * reliability[t,k] * sigmoid((signed measured score - threshold)/width)`,
then subtracted the same neutral baseline `0.5*gate[k]` from its
direct per-concept task contribution. If no sensor provided evidence,
`reliability=0`, `C_old=0`, and the logit contained the fixed,
**nonzero** direct term `-0.5 * gate[k] * beta[k]`.

That is a semantic error: no observable physiology was interpreted as a
positive or negative task concept, even though the task head was exact.

v6 uses the explicitly unknown-neutral concept state

```text
C_new[t,k] =
  gate[k] * (0.5 + reliability[t,k]
                   * (sigmoid((signed_score-threshold)/width) - 0.5))
direct_task_logit[k] = beta[k] * (summary_k - 0.5*gate[k])
```

At zero observed support the direct contribution is **exactly zero**
for all beta and thresholds; UNKNOWN is not necessarily healthy or
clinically normal. The forecast/denoising and mortality heads still
depend on training-time native concept trajectories: no post-hoc
concept generator or bypass classifier exists.

Optional dual-thinning regularization compares the same native concept
trajectories for independently masked versions of each observed
training episode, weighted by reliability present in both views.

## Synthetic v5: when is K robust to changed measurement regimes?

[Actions 41505537320](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41505537320)
validated 3/5/7 planted dynamic factors across original sampling, MCAR,
hour/channel-specific MAR and separately diagnostic value-conditioned MNAR.
Both pooled lag covariance and equal-time-stratified lag covariance
passed a predeclared rank invariance gate for K3/K5, and abstained
for K7 as independent train halves gave inconsistent dimensions.
The time-stratified estimator did not improve the pass rate relative
to the pooled comparator; robustness of **abstention**, not
rank-estimator superiority, was established.

## Real PhysioNet v5: independently chosen availability mask sensors

[Actions 41506176316](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41506176316).

- 2,400 official historical TRAIN patients, 600 of them for mask-only
  sensor availability selection with no outcomes or validation,
  1,800 others (900/900) for rank inference.
- 12 measured channels could be reliably retained under the
  predeclared conservative pilot measurement rule.
- None of the 1h, 4h, and 6h resolutions produced a rank stable under
  train-half replication and observation policy.
- Final gate: `ABSTAIN_CLINICAL_K_UNSTABLE`.

**No number of clinical physiological concepts is identified.**
A time-resolution-specific numerical estimate must not be elevated
to true latent pathology count or allowed to choose the number of
clinical concepts without declaring its assumptions.

## Synthetic v6, 36 models

[Actions 41506778924](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41506778924),
four initialization seeds each in three methods on independently
generated 3/5/7-factor systems:

- `multiplicative_original`: old absent-measurement behavior.
- `unknown_neutral`: physically unknown evidence gives zero direct logit effect.
- `unknown_neutral_consistency`: neutral plus dual-view train regularizer.

Compared with original, the neutral family improved patient-specific
cross-training Spearman excess over unrelated patient null:
K3 -0.049→+0.267, K5 +0.045→+0.797,
K7 -0.043→+0.225, but generally reduced outcome AUPRC for K5/K7.
Regularizing dual views added a smaller marginal benefit than
semantic neutrality of the unknown state itself.

[Exact 4^4 bootstrap 41507243707](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41507243707)
confirmed the K5 effect conditional on original seed and fixed cohort:
neutral-consistency-minus-old cross-training Spearman
+0.757, 95% model-only interval [+0.612,+0.873],
AUPRC -0.0234 [-0.0443,-0.0025].
K7 crossfit improvement interval covers zero:
+0.259 [-0.074,+0.746], with AUPRC loss
-0.0492 [-0.0653,-0.0317].
This is **not** a multi-generator or patient-bootstrap confidence interval.

## Real PhysioNet v6, 9 models

[Actions 41507973119](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41507973119),
three initialization seeds per method, official train only fitting,
repeatedly inspected historical validation for exploratory scoring,
test and calibration sealed, **no human clinical concept labels**.
The pilot-retained 12 measured channels allow an 8-slot computational
*budget*; earlier strict K gate remains ABSTAIN, so eight are not
identified clinical physiological processes.

| Method | Original outer validation AUPRC | AUROC | Absolute task contribution when zero measurements | Cross-seed case-specific Spearman excess | Mean time-channel-MAR direct contribution shift |
|---|---:|---:|---:|---:|---:|
| old multiplicative | 0.279848 | 0.631291 | 0.355106 | 0.008508 | 0.287489 |
| unknown-neutral | 0.286336 | 0.667743 | **0.000000** | 0.272021 | 0.159890 |
| unknown-neutral + consistency | 0.280521 | 0.667651 | **0.000000** | 0.272099 | 0.150208 |

The new architecture is consistently superior here on
**direct-contribution robustness** and cross-seed attribution agreement
without clear AUPRC loss under historical validation. Mean absolute
probability shift under simulated timeMAR is **not** smaller
(0.01803 old vs 0.02059 neutral); avoid saying all aspects of
prediction robustness improve.

No significance inference from three init seeds or repeated old validation.
Clinically true concept identities, treatments and causal interventions
are **not** established.

## Reproducible and native concept passports

[Native formula reproduction 41507540396](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41507540396)
retrained nine frozen initializations and reproduced AUPRC exactly,
exporting actual learned signed operator mixtures, threshold/width,
observation reliability, occurrence, exact signed beta and logit
contribution, WITHOUT patient-level examples. Mathematical
model-level concept interventions do not identify physical causal
effects.

## Next conceptual separation: V7 observation versus physiological evidence

Clinician measurement decisions are potentially informative about
patient risk. Forcing measurement absence to be neutral in the
**physiological** concept family removes semantic confusion but may
discard a useful independent prediction signal.

The predeclared V7 hypothesis therefore uses **two explicit
separate, jointly trained** concept families:

1. Measurement-value / physiological-evidence concepts:
   unknown-neutral when no measurement is available; measured signed
   operators, thresholds, signed direct outcome contributions,
   future prediction and denoising via these concepts only.
2. *Care observation-process* concepts:
   a mask-only signed operator bank (measurement frequency, recency,
   staleness, transitions), with separate own learned fuzzy
   thresholds/widths and direct coefficients. These must always be
   called observation/sampling concepts, **not** real diseases.

Exact combined task logit is bias+SUM of signed physiological
contributions+SUM of care-process contributions.
During deliberate denoising dropout the care-process head must
still receive the **original** recorded mask, not the artificial
training corruption. A learned mask-only task effect may describe
measurement policy, not latent pathology or treatment causality.

The claim is falsifiable by AUPRC of full/physical-only/process-only
heads, mask contribution magnitude share, behavior under missingness
and exact additivity. A model that derives most of its prediction
from measurement policy does NOT thereby validate its physiological
concepts.

## Hard external scientific limits

Clinical concept identification still requires external ground-truth
medical measurements, actual independent sites/environments, and
assumptions about informative missingness or interventions. A
test set already explored in an earlier paper cannot become new
untouched prospective confirmation of post-selected architecture.
