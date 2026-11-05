# Frozen experiment — temporal concept evidence rather than episode means (2026-11-06)

**Freeze before implementation, generator execution or any TEST inspection.** Branch \`experiment/ioi-temporal-source-bank-v31-20261106\`, parent PR #27. This is a new controlled simulator family and a comparison of existing temporal summary operators, **not a new theorem, causal identification, or faithful reproduction of CIBM / Res-SCBM**.

## Scientific bottleneck identified

Previous PR26/PR27 automatic group discovery and risk validation computed each new concept using the sigmoid of an episode-average of observed sensor readings. An early positive pulse followed by a late negative pulse can have *zero overall mean*, while the **ordering of those pulses** matters to the outcome. A grouping algorithm may correctly detect mutually corroborating sensors, yet the summary \`q_static\` discards the relevant temporal information. Merely using a Temporal ConceptFAN encoder for already-known K2 concepts does NOT repair this defect in the newly added sources.

This experiment asks whether a **TRAIN-frozen source-agnostic temporal statistic** (early-minus-late signed contrast) can recover that useful information, while preserving existing K2 concept values and coefficients, rejecting an irrelevant source, and failing honestly under previously unseen sign changes.

The sequential/independent/joint CBM protocols are known since Koh et al., ICML 2020 (https://proceedings.mlr.press/v119/koh20a.html); leakage vs incompleteness was studied by Havasi et al., NeurIPS 2022 (https://github.com/dtak/addressing-leakage); ICLR2026 Information Bottleneck CBMs have public authors' code (https://github.com/dsb-ifi/cibm), and Residual Stochastic CBMs (Marcou et al., arxiv:2610.23421) were proposed October2026. The temporal contrast and two-environment admission are conventional baselines, **not new named published methods**. Faithful authors' code reproduction is outside this frozen single study; report this gap.

## Distinct genuinely temporal simulator family

Generator seed **20261116**, n=1500 TRAIN / 500 VALID1 / 500 VALID2 / 800 TEST independent 16-step episodes, 12 noisy measured channels:
- known concept1 and concept2: distinct independent low-frequency *non-AR Gaussian* waveform sources with episode-level offsets, sparse impulses and nonlinear sinusoidal waves. Sensors0/1 are redundant readouts of first source; sensors2/3 of second. They are external KNOWN groups for K2, not inferred clinical semantic concepts.
- third source: independent random signed **first-half + amplitude, second-half - amplitude** waveform, with small per-time zero-mean perturbations and independent measurement noise in channels4/5/6. Its full-history mean is uninformative about its signed amplitude in expectation, while early-minus-late contrast is informative.
- independent nonpredictive nuisance waveform duplicated in sensors7/8, independent recording/measurement artifact phase source duplicated in sensors9/10, single-channel white-noise distractor sensor11.
- three outcome laws sampled with frozen per-subject random uniforms: COMPLETE (Y depends only on K2), PHYSICAL (Y additionally depends on third-source signed pulse amplitude), ARTIFACT (Y additionally depends on similarly pulsed *recording artifact* amplitude). Predictive artifact may be selected and MUST never be called physiological; instrument and physiological latent source semantics are supplied by simulator provenance only.

Important: this pulse-order simulator is distinct from the nonlinear switching/Laplace AR-like generator V30-PR27. Demonstrate with a unit test that two signals with equal full-history means but reversed half ordering have opposite true outcome contribution, while mean-only candidate is invariant.

TRAIN measured mask MCAR20 seed20269301; VALID1 MCAR20 seed20269302; VALID2 MCAR20 seed20269303. TEST missingness complete/MCAR20/MCAR40/MNAR/block8 seeds20269900+ordinal. All TEST fault transformations apply to same 800 subjects and no target labels are used to generate missingness. Three actual native ConceptFAN K2 random seeds **11,23,37**, 14 epochs, AdamW .002, batch128, stage-one supervised ONLY by available measured sensor0/1 and2/3 proxies. Core input_dim=8 (4 known channels and mask), no candidate channel into its concept pathway, no outcome gradients to stage-one concepts. Original K2 risk head trained on TRAIN Y and then fully frozen for both methods.

## Unlabelled candidate GROUPS, identical for both methods

TRAIN-only signed-correlation grouping of unknown channels4:11 over three disjoint patient folds, |rho|>=.60 per fold, min300 co-recorded cells/fold, sign-stable clique criterion, max5 groups. Do NOT reveal true groups or class labels to discovery. If no group / ambiguous grouping, fail and retain recorded negative. Group mean, std, sign alignment are TRAIN-only. Candidate grouping and original K2 states are SHARED across all operator comparisons.

For EACH discovered group generate exactly TWO candidate statistics, fixed **before** training labels/test:
1. **static_mean**: preexisting \`measured_group_concept\`, sigmoid of signed TRAIN-normalized average of actually observed readings, with full-episode availability shrinkage. 
2. **half_contrast**: split 16 time steps at t=8; compute the signed TRAIN-normalized mean of observed readings separately in first and second half; logit statistic \`(mean_first - mean_last)/2\`; candidate \`q=.5+min(support_first,support_second)*(sigmoid(statistic)-.5)\`. When either half has support<0.50, mark this *candidate for that episode* unsupported and ensure fallback to unchanged K2 logit. No missing payload, target label, external source assignment, or oracle true pulse amplitude used. This is a classical temporal summary, not a learned auto-semantic concept.
- Same group count for static and half-contrast (<=5 each). Separate gamma fitted ONLY with TRAIN labels; same TRAIN outcomes and same native K2 baseline head. Target-outcome gradients must not affect candidate grouping / statistics.
- Gates (fixed) need BOTH disjoint VALID1 500 and VALID2 500 cohorts, MCAR20, lower 1% paired TEST-independent bootstrap from 399 resamples >0, mean paired NLL reduction>=.003 in each, >=75% source-supported episodes, one-source-at-most. Keep selection log for ALL groups and both temporal operators. Max 10 evaluated candidates; 1% percentile screening is only an empirical conservative heuristic and does not guarantee exact familywise error.
- VALID2 additional **observation environments**: (A) benign gain 1.10 and offset+.05 applied on unknown channels (target laws unchanged); (B) common-mode pulse source sign inversion in VALID2 and TEST for PHYSICAL (fault seen); (C) analogous artifact sensors9/10 inversion for ARTIFACT (fault seen); (D) benign VALID2 but unseen inversion only on TEST for PHYSICAL (expected FAIL); (E) COMPLETE law benign drift/no extra factor.
- The **same existing TRAIN and VALID subjects**, no extra outcome labels for temporal vs static; compare with no-shift IID gate on same VALID2 labels when useful.

## Comparisons, budget and observability

All base methods get same labelled TRAIN1500 + two VALID500+500 =2500 subject outcome labels **available**. Methods actually using them are marked:
- Native frozen K2 baseline.
- Auto-STATIC source only (grouping, gamma and two-environment gate as in PR27).
- Auto-HALF-CONTRAST source only (same groups, TRAIN coefficient, VALID gate).
- Auto-BANK (both operators among <=10 proposed, max one VALID-approved candidate).
- Strong logistic on measured 12-channel summaries including **first and last half means and observed mask fractions** fit with TRAIN1500 labels. Crucially, this baseline is allowed the same temporal information and is not an artificially weak whole-history-mean model.
- Same raw temporal logistic fitted on ALL2500 TRAIN+VALID labels (with VALID2 shifted as prespecified). This shares TOTAL label budget, not optimizer steps or allocation.
- Standard additive CBM head on **all discovered temporal group summaries plus K2** fitted with TRAIN1500; also repeat with ALL2500 labels under shifted VALID2, thereby allocating all available supervision to the baseline rather than only admission. This baseline can change original K2 weights, document it.
- Optional oracle TRUE-K2 + TRUE physical/recording pulse for posthoc audit only. It is privileged and not a deployable baseline.
- No IB-CBM or Res-SCBM described as "reproduced" until those original published approaches are actually run with their corresponding losses and intervention operators. No equality of compute across all reference families claimed.

Report TEST AUPRC, AUROC, NLL, Brier, candidate usage/abstentions, source group and operator chosen, per-VALID gate gains/bootstrap bands, loss of useful source after shifts, and exact additive reconstruction. Test that K2 neural state SHA and original K2 head coefficients never change when new source selected or rejected. For the primary clean PHYSICAL/MCAR20 contrast (half vs static; half vs K2) compute **paired patient bootstrap 95% intervals** (399 resamples on same 800 TEST episodes and three frozen NN seeds) and disclose shared-sample, conditional-on-fits nature.

## Outcomes to report even when unfavourable

- If full mean static summary accidentally captures pulse information through sampling bias, report it; don't tune generator after inspecting TEST.
- If temporal contrast loses to a raw temporal logistic or all-group CBM, state no predictive superiority. AUC/PR results do not prove source semantics.
- If artifact temporal contrast is predictive and selected, record this as semantic failure of unsupervised concept naming.
- If unseen source inversion fools gate or block8 missingness causes high abstention, retain failure.
- If no genuine new cross-family advantage is demonstrated, do not claim architectural novelty on this synthetic exercise.

All weights and outputs frozen by source commit in GitHub Actions; do not edit prior accepted IEEE PIERE or IITI manuscripts or alter previous V30/PR26/PR27 TEST data.
