# ConceptFAN V53–V55: executed architecture and scientific evidence audit

Date: 13 November 2026. Status: **executed, exploratory, synthetic-data only**. This file summarizes pinned source, implementation, and completed GitHub Actions; it does not claim clinical validation, a novel latent-rank theorem, or semantic concept identification.

## Reproducible evidence

| Stage | Verified run | Implementation boundary |
|---|---|---|
| V53, all-sensor measured state + surprise prototypes | [Actions SUCCESS #43435625688](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43435625688) | No manually supplied initial K2; learn groups from TRAIN observation only; explicit temporal state/surprise |
| V54, robust group median and spectral observable rank | [Actions SUCCESS #43436861118](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43436861118) | Medians resist one faulty channel; existing positive-lag rank supports otherwise ungrouped mixtures |
| V55, automatic label-independent route between two operators | [Actions SUCCESS #43438469688](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43438469688) | Single source vocabulary/risk API; independent fresh group-world draws plus mixed-loading family; 19 focused tests passed |

Original protocols committed before inspecting each stage's new experimental outcomes:
- `CONCEPTFAN_V53_FROZEN_AUTONOMOUS_TEMPORAL_PROTOCOL_20261113.md`
- `CONCEPTFAN_V54_FROZEN_ROBUST_SPECTRAL_PROTOCOL_20261113.md`
- `CONCEPTFAN_V55_FROZEN_ADAPTIVE_VOCABULARY_PROTOCOL_20261113.md`

Complete V55 Actions artifact: `conceptfan-v55-adaptive-concepts`; source commit `afd76ceda2aa18519f9b7f56488b4a40d812b4fe`; `V55_EVIDENCE/STATUS.json`, `GROUP_RESULTS.csv`, `MIXED_RESULTS.csv`, and TRAIN-only model passports.

## Implemented model and exact contract

V55 `AdaptiveConceptFAN` takes all observed TRAIN series and their masks and chooses its vocabulary with no outcome labels:
1. Require at least 2 complete signed-correlation source groups covering >=50% of channels. Groups are found in three disjoint TRAIN measurement folds (no manually specified original concepts). Within each >=3-channel group, apply signed-standardized observed-value *median* with TRAIN-derived fuzzy threshold/width; otherwise measured mean.
2. If that rule fails, apply the repository's established positive-lag predictive-rank method (lags 1/2/3; 60 TRAIN-only patient permutation nulls, 20 bootstrap repetitions) and form unnamed, signed fuzzy concepts from the recovered observable subspace. The rank can be 0; then abstain from creating source slots.
3. Freeze vocabulary. Only now fit the same TRAIN-label additive logistic head: `logit=bias + sum_k beta_k*(mean_t c_k-0.5)`. No target gradient to concept definitions, no oracle factor labels, and no bypass predictor. Source manifest hash and exact logit reconstruction are audited.
4. Explicitly distinguish `MEASURED_GROUPS`, `PREDICTIVE_SUBSPACE` and `OBSERVATION_ABSTAIN`. No method assigns physical or medical concept names automatically.

This is a **two-stage train-calibrated statistical concept module**, not a newly trained native differentiable Transformer/Type-3 system. The original goal of end-to-end *learned* and physically verified concept identities is still open.

## New V55 result: separate fresh source-group draws

Two independent episode draws from the **same nonlinear V30 extended synthetic family**, seeds 20261331 and 20261333. Each uses TRAIN=1,400, unused VALID=600, TEST=700, 12 channels and 16 time steps. The route found 5 coherent measured groups on both new draws, independently of all three outcome laws.

**PHYSICAL_EXTRA synthetic target; AUPRC**:

| Draw/perturbation | V53 mean-group | V53 whole-group reject | V55 adaptive robust group | Raw logistic |
|---|---:|---:|---:|---:|
| 20261331, normal MCAR20 | .8308 | .8305 | .8307 | .8292 |
| 20261333, normal MCAR20 | .8208 | .8185 | .8217 | .8030 |
| 20261331, TEST-only flip of one source sensor | .8000 | .7522 | **.8215** | .8019 |
| 20261333, TEST-only flip of one source sensor | .7744 | .7118 | **.8050** | .7110 |
| 20261331, simultaneous flip of ALL 3 source sensors | .5617 | .5650 | .5623 | .5675 |
| 20261333, simultaneous flip of ALL 3 source sensors | .5320 | .5247 | .5327 | .5189 |

Under single-sensor flips, paired TEST-subject descriptive NLL improvement for V55 compared with V53 mean-group:
- draw 20261331: +.02055 [95% .01357, .02783];
- draw 20261333: +.02559 [95% .01903, .03129].

These within-TEST intervals are descriptive; generator-family replication is only TWO draws. V55 robust pooling offers effectively unchanged clean-data prediction but does not solve simultaneous common-mode corruption. Under intact condition NLL is slightly *worse* than V53 mean-group (about .00045/.00016), and other classifiers may win. Do not label this globally superior.

## Different synthetic dynamical family: mixed-loading observations

On a separate planted variable-factor generator with 8 sensors and 36 historical time steps, V53 strict correlation cliques produced no stable source groups. The V55 label-free selector routed all three conditions to **PREDICTIVE_SUBSPACE**, with TRAIN-estimated ranks 3, 5, 7 for evaluator-only hidden dimensions 3, 5, 7. Patient-bootstrapped TRAIN intervals [3,3], [5,5], [7,7]. The original sealed TEST was **not opened**; the following AUPRC uses DEVELOPMENT VALID only:

| Evaluator-only true K | Detected predictive rank | V55 spectral concept head | Raw logistic | PCA auto-K |
|---|---:|---:|---:|---:|
| 3 | 3 | **.5488** | .4736 | .5320 |
| 5 | 5 | .4486 | **.5431** | .5279 |
| 7 | 7 | .4073 | .4206 | **.4369** |

Dimension recovery under this synthetic observable generator is meaningful, but **not a unique physical concept decomposition**. The original V3 already recovered rank under favorable synthetic assumptions, so V55 does NOT invent the rank method. Task ranking can be worse than simpler baselines even when recovered rank matches planted K.

## Main falsifiers and next scientific gate

- V53 temporal surprise features were not a robust AUPRC improvement over state-only. More temporal operators do not automatically improve explanation quality.
- A static correlation clique may hallucinate a semantic source: the outcome-predictive recording-artifact group remains observationally plausible, not medically certified.
- Median group state helps one gross outlier when >=3 readings exist; it cannot resolve 2-sensor conflicts or common-mode shift.
- Spectral axes are rotation/gauge dependent; even exact observable K is weaker than unique factor identity. Mixed-source tests are development validation, not an independent sealed external confirmation.
- On genuine sparse PhysioNet data previous rank varied greatly with channel support and temporal aggregation. V55 has **not** overcome this clinical source-observability limitation.

**Further engineering target:** native masked-time-series concept discovery with TRAIN-only forecast/denoising and independent source-identity anchors, followed by across-mechanism validation and human-measured concept references. Keep output labels out of the concept-formation gradients; compare raw/PCA/CBM with label and computation budgets controlled. Do not promote this exploratory work into the accepted PIERE/IOI manuscript without separate editorial/scientific approval.
