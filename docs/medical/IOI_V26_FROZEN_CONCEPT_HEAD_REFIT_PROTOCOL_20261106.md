# IOI ConceptFAN V26 — frozen concept-only decision-head diagnostic (exploratory follow-up)

**Date:** 2026-11-06. **Status:** post-V25 exploratory control. Results from V25 TEST have already been seen; this V26 analysis on the SAME synthetic generator and TEST is NOT an independent confirmation or a preregistered test for the journal.

## Question

V25 trained 18 real native temporal ConceptFAN models with a K2 or privileged K3 dictionary (1 Transformer layer, 14 epochs, AdamW 0.002, concept trajectory supervision). Canonical fuzzy decision heads underperformed a raw-sensor logistic baseline and showed substantial seed sensitivity despite low synthetic concept reconstruction MSE. Separate two mechanisms:

- weak concepts that cannot represent predictive signal;
- an underoptimized/overcomplicated fuzzy risk head when useful concept evidence is already present.

## Frozen split and models

Reuse exactly V25's synthetic generator, indexes 1600 TRAIN / 400 untouched CAL / 800 TEST, seeds 11/23/37, K2 vs privileged K3, 6 noisy measured channels, 16 time steps and MCAR20 TRAIN masks. Reuse *identical* 14-epoch model fitting and initial parameter seeds (no tuning on results). Stress only frozen `MCAR20`, `MCAR40`, `MNAR` masks with the exact V25 random seeds. Instrument-artifact family is bitwise the same observed dataset and fitted weights as omitted-factor family; no extra pseudo-replications.

After full native model training, freeze encoder, projector and temporal aggregation:

1. **Native fuzzy risk head:** original model untouched.
2. **Positive Platt calibration:** fit TRAIN-only positive affine transform of original logits; ranking must not change (AUPRC, AUROC invariant), exact original additive contributions can be rescaled. This tests calibration vs ranking without modifying concept representation.
3. **Simple additive linear concept risk head:** fit a standardized logistic model on **TRAIN-only frozen native concept summaries** `q_k` and TRAIN outcome labels. Export equivalent coefficients in ORIGINAL concept coordinates, so risk is exactly `b + Σ beta_k q_k`, with no encoder/raw sensor bypass. This is a **classic concept bottleneck baseline, not original architectural novelty**.
4. **Oracle concept ceiling:** logistic classifier fitted on privileged generator-known TRUE concepts (TRAIN) then evaluated with TRUE concepts (TEST). It is an audit only, not a feasible real-data model.
5. **Raw input logistic baseline:** exact same direct measured channel statistics as V25, TRAIN-only.

The extra post-training head optimization reuses the same TRAIN labels but entails additional compute; therefore comparisons are **not optimizer/compute matched**. Model selection is not performed on TEST. K3 concepts have privileged synthetic TRAIN concept labels and cannot be described as automatically discovered.

## Endpoints

All 3 neural initializations, both distinct label worlds and all TEST stress conditions: AUPRC, AUROC, NLL, Brier, fitted concept MSE, risk logit decomposition residual, original and calibrated source state hashes, and fitted head coefficient logs. Paired TEST subjects must not be treated as independent NN seeds; results are descriptive, and no post-hoc significance threshold is implied.

### Interpretation

- If concept-only linear refit improves AUPRC relative to fuzzy head, the *head optimization/parameterization* is one major bottleneck of this particular training recipe; it does not establish that fuzzy memberships are always harmful.
- If positive Platt only improves NLL/Brier (not AUPRC), separate calibration error from representation quality.
- If K2 concept-only linear refit still lags oracle-K2 and raw logistic, model concept estimation is a further predictive limitation.
- If oracle-K3 improves only in omitted/artifact worlds, this indicates privileged extra-factor value but NOT semantic identification.

All evidence in a new branch/PR, no edits to V25 frozen protocol, previous PIERE results or clinical TEST.
