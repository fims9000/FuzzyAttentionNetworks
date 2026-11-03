# Evidence receipt: spectral predictive rank + native concept model v3

**Date:** 2026-11-02. **Scientific status:** exploratory, not confirmed
clinical concept discovery. The original PIERE/ConceptFAN tag is immutable.

## Source runs

1. [Train-only rank benchmark — PASS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41498657019).
   Outputs in artifact `spectral-predictive-rank-v3-1`, including
   `spectral_rank_external_oracle_audit.csv` and spectra/permutation diagnostics.
2. [Native spectral concepts vs learned-10 — PASS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41499541576).
   Outputs in artifact `native-spectral-concept-v3-2`: 36 training runs,
   9 row summary, all paired run comparisons, complete formulas, train-only
   spectral rank provenance, 13 focused unit tests passed.
3. Earlier [masked-denoising v2 — PASS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41494889902)
   remains a negative reference: all three system types collapsed to K=1
   after greedy validation-based pruning.

## Rank detection on independent generator seeds

| True hidden K used only by external evaluator | Detected TRAIN K (generator 20261221) | Detected TRAIN K (generator 20261237) |
| ---: | ---: | ---: |
| 3 | 3 | 3 |
| 5 | 5 | 5 |
| 7 | 7 | 7 |

- All 6 ranks exactly matched external synthetic truth with
  positive-lag cross covariance and a **familywise max-singular-value null**
  from 120 patient-permutation resamples.
- The existing square-block-Hankel comparator also matched all six
  cases in this particular experiment, so we **cannot claim spectral
  rank estimation itself is new or superior**.
- 50 patient-bootstrap resamples per case gave intervals [K,K]
  conditional on that frozen null threshold. That is **not a general
  proof of unique latent-factor identifiability**.
- No outcome labels, latent factor values, or evaluation patient records
  were given to the spectral rank estimator.

## End-to-end concept learning on *new* generator seeds

Four independently initialized training runs per arm for each K.
All three arms train via the bottleneck, without any manually defined or
synthetic ground-truth concept labels:

- `learned_10`: original end-to-end v2, K selected on inner validation.
- `spectral_frozen`: detected TRAIN K; normalized signed observed
  sensor-level definitions fixed, but fuzzy thresholds, widths and
  decision/dynamic heads learned.
- `spectral_finetuned`: same TRAIN K and operator initialization;
  definition weights may adapt under alignment and orthogonality losses.

| External planted K | Method | Selected K | Outer AUPRC | Oracle coverage-adjusted abs correlation | Cross-fit case-specific Spearman excess over independent-patient null |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3 | learned_10 | 1 | 0.386829 | 0.059815 | undefined (single slot) |
| 3 | spectral_frozen | 3 | 0.414918 | 0.468774 | 0.080196 |
| 3 | spectral_finetuned | 3 | 0.413610 | 0.465331 | 0.094871 |
| 5 | learned_10 | 1 | 0.443440 | 0.073131 | undefined |
| 5 | spectral_frozen | 5 | 0.426049 | 0.386884 | 0.036546 |
| 5 | spectral_finetuned | 5 | 0.432321 | 0.385729 | 0.040634 |
| 7 | learned_10 | 1 | 0.398277 | 0.051677 | undefined |
| 7 | spectral_frozen | 7 | 0.361130 | 0.377140 | -0.025957 |
| 7 | spectral_finetuned | 7 | 0.365057 | 0.374189 | -0.029618 |

**Interpretation, explicitly separated:**

- The *dimension* of observable dynamics was identified reliably for
  these synthetic generator families, with train-only data.
- For the spectral arms, **100% oracle-factor match coverage is mostly
  a consequence of choosing K equal to planted K and maintaining
  K nonconstant slots**. It is *not* independently proof of correct
  concept identity or medical semantics. Coverage-adjusted correlations
  provide the nontrivial measurement of correspondence.
- Compared with the collapsed one-slot v2 baseline, spectral concepts
  correlate better with planted latent variables, but not perfectly.
  They are signed *sensor combinations*, not automatically named
  physiological diagnoses or uniquely recovered causal states.
- AUPRC improves for planted K=3 but becomes worse for K=5 and K=7
  compared with learned-10 baseline; no all-task predictive superiority
  can be claimed.
- The **case-specific cross-training agreement** is modest and for K=7
  actually slightly below the patient-independent null. **Stable
  operator definitions by construction are not stable individual
  explanations.** This is the principal remaining scientific problem.

## What is directly auditable

Each case-specific model logit is exactly:
`bias + sum_k observed_sensor_concept_k * signed_decision_coefficient_k`.
There is no hidden classifier path bypassing the concept layer.
The concept formulas, thresholds, widths, signed sensor contributions and
example event boundaries can be exported without giving them disease names.

New regression tests reject *constant explanations* that appear stable
but provide no additional episode-specific information. The
`evaluation.py` module now reports learned signed beta and contribution
quantiles in machine-readable passports and compares cross-fit sign
agreement against empirical sign-frequency nulls.

## Next validity gate

A clinical train-only rank diagnostic is run separately on official
PhysioNet without the manually assigned five concept labels. In the
initial 1-hour audit, the strict covariance estimator correctly refused
many channel pairs because **too few jointly timed observations exist**.
The revised version openly tests 1h, 4h, and 6h aggregation and reports
which sensor subsets pass 20-sample per-pair support, rather than
falsifying a medical factor count from insufficient observations.

Even if rank exists on coarsened clinical measurements, its transfer
to pathology names and causal medicine requires external validation and
new observational assumptions, not optimization pressure.

## Prior art and new contribution scope

- Classical system identification / subspace-Hankel SVD introduced the
  measurement-rank idea. Do **not** label it as invented here.
- MAGNETS (Forest, Wei, Fink; 2026,
  https://doi.org/10.1007/s10618-026-01267-y) already learns
  annotation-free time-series concepts directly in the predictive
  pipeline, so that property is **not** by itself novel.
- The **hypothesis** to investigate is whether train-only dimensionality
  detection with an explicit signed, measured operator bottleneck
  improves reproducible *patient-specific* concept explanations
  without unacceptable predictive degradation. Current experiments
  have **not yet established** that improvement.
