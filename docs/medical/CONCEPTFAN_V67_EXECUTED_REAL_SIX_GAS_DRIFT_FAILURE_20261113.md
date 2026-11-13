# V67 EXECUTED: six real pure-gas classes under 36-month sensor drift — anonymous fuzzy sources not stable physical identities

**Status: full successful new-data-family test, negative outcome for concept completeness and gas identity.** The experimental protocol was committed **before reading UCI270 source row labels and before any V67 score**: `CONCEPTFAN_V67_FROZEN_UNSEEN_SIX_GAS_DRIFT_PROTOCOL_20261113.md`. [GitHub Actions SUCCESS #43485497291](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43485497291), executed source SHA `c8091134ddd08d90aae8b730c96a1b9f36f18bfa`, **25 checks passed**. Retained original `conceptfan-v67-real-unseen-six-gas-drift-source-semantics` artifact ID **14066952423**, containing `V67_UNSEEN_DRIFT_EVIDENCE/STATUS.json`, `METRICS.csv` (24 predeclared cohort×method results), immutable source commit and aggregate log. Official archive SHA256 **98fe3a30981a222dd4518fbcc3dddd45d5c0ce9b03ef6dc6fe5cf7a04cfbff5e**.

## Genuine new real measurement regime

Official [UCI Gas Sensor Array Drift at Different Concentrations (ID270)](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+at+different+concentrations), research-only original measurements. **13,910 physical single-gas exposures**, 16 measured sensors, each summarized by **8 engineered response descriptors** (2 steady-state, 6 transient EMAs) → **128 numeric features**. Classes and concentration metadata are present in the original files but NEVER passed into unsupervised source fitting. Six gas names: 1 ethanol, 2 ethylene, 3 ammonia, 4 acetaldehyde, 5 acetone, 6 toluene. Original ten batches span ~36 months in ONE installation; no independent lab/hardware. No simultaneous source mixture and no 128-point time trajectories.

Unchanged chronological physical drift split:
- TRAIN original batches **1–7, 9,546 exposures**, all 6 classes.
- VALID historical batches **8–9, 764 exposures**, all 6 classes.
- **TEST untouched batch 10: 3,600 exposures, 600 per gas class**.
- Source-normalization medians/IQR, sensor-view grouping, ICA directions, source K and 3 fuzzy memberships per statistical axis learned entirely on TRAIN responses. Supervised gas classification heads trained TRAIN labels only. VALID gas labels used strictly to match already-frozen anonymous axes to gas references. No TEST selection or rematching.

**Important scope:** Because 8 engineered features per sensor are different types of measurement descriptors, not chronologically successive samples, this is a **modality-correct STATIC test** of label-free ConceptFAN source grammar and exact signed additive head; it does **NOT** directly validate the V65 recurrent 120-second architecture. No fake time axis was created.

## Source construction evidence

- 16 sensors × (steady-response mean of 2 normalized descriptors, transient-response mean of remaining 6) = **32 pooled measured features**.
- TRAIN-only PCA 95% rank **K=4 (uncapped K=4)**, capturing **95.8797%** of TRAIN pooled 32-dimensional variance; PCA count is an observable statistical dimension, not count of distinct chemicals.
- TRAIN-only FastICA on 32-dimensional sensor summary, seed20261967, **converged after 12 iterations**; no physical gas labels in ICA. Four anonymous signed ICA coordinates projected on all new samples.
- Each ICA coordinate has TRAIN-only Q25/Q50/Q75 fuzzy thresholds and positive widths → **12 bounded soft membership STATES**. One six-way multinomial logistic head with TRAIN-only state standardization; coefficients transformed back into original fuzzy coordinates for exact signed class-wise additive logit. Machine-precision probability reconstruction max delta **3.775×10^-15**, zero recorded training convergence warnings. This proves ONLY functional fidelity, not stable chemical identity.

## Strong baselines and held-out TEST outcome

All models trained with the same 9,546 TRAIN gas-class labels; only fuzzy source and linear ICA use restricted unlabeled source representation. Original six-class test is **balanced**, so accuracy equals balanced accuracy.

| Frozen method | TEST accuracy | TEST macro-F1 | TEST macro OvR AUROC | TEST log loss |
|---|---:|---:|---:|---:|
| 12 fuzzy concept states + exact additive logistic | **.4833** | **.4451** | .8021 | 2.9307 |
| 4 linear ICA source coordinates + logistic | .5567 | .5090 | .8468 | 1.9665 |
| 32 pooled physical sensor descriptor summary + logistic | .6186 | .6034 | .8778 | 2.3198 |
| Original 128 descriptors + LogisticRegression | **.6817** | **.6837** | .8989 | 2.2005 |
| Original 128 descriptors + ExtraTrees (300) | .6444 | .6442 | **.9017** | **.9803** |
| Majority TRAIN class | .1667 | .0476 | .5000 | 30.0364 |

The best test macro-F1 belongs to raw 128-feature logistic, exceeding the interpretable fuzzy concept head by **0.2386** (23.86 percentage points). The large raw-versus-concept gap must remain visible. ExtraTrees has lower test macro-F1 than raw logistic but much better reported log loss .9803 vs 2.2005, so neither raw model uniformly dominates across metrics; concept head is inferior on all listed metrics.

## VALID vs final TEST; no hyperparameter selection

| Model | VALID batches 8–9 macro-F1 | TEST batch 10 macro-F1 |
|---|---:|---:|
| 12 fuzzy sources | .5589 | .4451 |
| 4 ICA linear sources | .5079 | .5090 |
| 32 pooled measured descriptors | **.8704** | .6034 |
| Raw 128 logistic | .6954 | **.6837** |
| Raw 128 ExtraTrees | .7919 | .6442 |

The decline from VALID to TEST in several models is consistent with drift/batch regime changes; this is not proof that individual features are chemically causal. The VALID pooled32 model looked especially promising at .8704 but fell sharply to .6034 TEST, demonstrating why one must not tune or report only earlier periods.

## Critical new physical semantics falsification

After ICA and fuzzy source fit, VALID gas class labels (batches8–9) were used to construct ONE-TO-ONE maximum separation matching from 4 anonymous axes to six chemical classes. Mapping frozen before batch10. Two gas classes necessarily **ABSTAIN** because K=4<6. No TEST axis rematching.

| Physical gas class | VALID-oriented OvR AUC of assigned axis | batch10 TEST AUC, *fixed same axis and sign* |
|---|---:|---:|
| 1 Ethanol | **1.0000** | **.6070** |
| 2 Ethylene | **.9931** | **.6627** |
| 3 Ammonia | .9706 | .8566 |
| 4 Acetaldehyde | **ABSTAIN** | ABSTAIN |
| 5 Acetone | **ABSTAIN** | ABSTAIN |
| 6 Toluene | .9153 | .8907 |

Two near-perfect VALID associations (ethanol, ethylene) **collapse under sensor drift** on the actual future batch10: AUC ~.607/.663. Ammonia and toluene remain moderately discriminative, but that is *association*, not unique chemical interpretation. TEST shuffled-label controls on the fixed mapping are around .48–.51 AUC; their failure to discriminate labels does not establish physical causality.

Within each matched TEST gas class, source-axis/concentration Spearman can be large (e.g. ammonia +.967; ethanol −.718), illustrating that a high concept–gas association may depend on **concentration** in the pure-gas regime. Thus a high one-vs-rest gas-label association is not automatically gas-specific independent of dose/batch.

## What this genuinely establishes

**Reproducible positive engineering:** all 10 original UCI270 batch files were parsed at 128 observed engineered features/exposure without padding or imaginary timestamps, 16-sensor two-view source code runs without gas labels, selection rankK4 and 12 calibrated fuzzy states are deterministic, complete train/VALID/TEST head and classwise predictions execute, and probabilities are exactly decomposable into original fuzzy-state contributions. Raw data and licenses remain external to the artifact.

**Strong negative finding:** in a NEW independent dataset family and unseen future measurement period, automatically discovered anonymous PCA/ICA source axes are **not reliably equal to distinct pure-gas concepts**. Under drift, near-perfect earlier-class associations can fail, and concept-only classification is much less accurate than raw sensor classifiers. The valid-label-informed assignment is *postfit supervised semantic auditing*, not automatic discovery of chemical names.

**No overclaims:** This is six different **single-compound** exposures, not six simultaneously mixed gas sources; no independent second lab, no clinical or instrument-fault validation, and the engineered features carry implicit temporal summaries but no original time series. Stable chemical source identifiability and automatically generated human-readable gas concepts have NOT been proved. This new family can test transfer of statistical concept *operators*, not V65 recurrent dynamics or independent chemical causal mechanisms.

**Scientific implication:** Further increasing fuzzy memberships or adding a transformer is not a grounded fix. The missing capability is source-specific independent reference anchoring and cross-environment concept invariance with an abstention option, evaluated on prospectively held-out physical sessions. This cannot be retroactively demonstrated by optimizing on the already-seen UCI270 batch10. Preserve negative evidence, keep PR DRAFT and accepted IOI/PIERE papers unchanged.
