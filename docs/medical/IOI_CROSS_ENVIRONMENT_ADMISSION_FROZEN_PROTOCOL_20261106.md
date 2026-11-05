# IOI automatic measured concepts: frozen out-of-environment admission test

**Prespecified 2026-11-06 before generating or inspecting outcomes of this experiment.** Branch \`experiment/ioi-cross-environment-source-admission-20261106\`, based on successfully executed automatic-proposals PR #26. This is a NEW RANDOM DRAW (seed 20261114) of the SAME extended nonlinear V30 synthetic simulator family, not external clinical validation or a novel causal source-identification result. The previously inspected PR #26 TEST may NOT be used for training, policy tuning, thresholds or stopping.

## Research hypothesis

PR #26 formed 3 candidate groups without oracle group identities or labels and admitted a predictive physical source OR a predictive recording artifact using independent VALID subjects, but **both labeled VALID cohorts had the same measurement regime**. Here one validation cohort is an explicitly different device regime. Can risk-based independent source admission reject *known* device mapping inversions while permitting a healthy source under a different benign regime, and how far can it generalize to an *unseen* mapping failure? The latter should be an explicit failure mode, not hidden by a retrospective threshold adjustment.

This is a transport/risk-screen acceptance criterion, not identification of underlying physiological semantics. A selected recording artifact remains a recording artifact, whatever the AUPRC gain. A two-environment gate is existing robust validation practice, not a new theorem.

## Data budget, models, partition

- **Data-generation seed = 20261114**, 16 steps, 12 measured channels, same synthetic nonlinear switching-shock family as PR #26, **TRAIN1700 / VALID1 600 / VALID2 600 / TEST900** independent episode identities. Three native K2 ConceptFAN seeds 11/23/37. No TEST outcome labels used in proposal, model training, calibration, admission, comparisons or thresholds.
- Native known K2 encoder input remains HARD restricted to channels0:4 and observation masks. Known concept surrogate targets use TRAIN-only observed pair means for sensors0:1 and2:3, with target support masking. No latent oracle state in K2 training; no outcome gradients into K2. 14 AdamW epochs, LR=.002, identical prior code and optimizers. K2 decision weights and bias fit TRAIN1700 outcome labels via same canonical additive logistic head and then frozen.
- Automatic unknown-source proposals formed using TRAIN (1700) observation-only signed correlation in three disjoint subject folds, max5 groups, original PR26 thresholds |rho|>=0.6 and 300 joint cells per fold, complete signed cliques, no oracle/VALID/TEST source hints. Reuse existing PR26 module unchanged. Record every proposal even when rejected.
- Candidate additive risk coefficient gamma fit from TRAIN1700 outcome labels only (L2 .002). Candidate q formed using TRAIN-only normalizations and currently observed sensor readings. Gate includes two disjoint VALID600 cohorts. Per-env all-patient paired NLL improvement >= .003, subject-level bootstrap 399, lower **1% quantile >0**, >=75% source eligible, per-patient support>=50%, max one source. Keep original non-formal FWER disclaimer.
- VALID1 remains ordinary MCAR20 (seed20268202). VALID2 is predeclared shifted observation environment (MCAR20, seed20268203) with changing sensors according to the tested scenario below. The two VALID cohort labels and latent target law remain unchanged. TEST missingness seeds20268900 + original STRESS ordinal for complete,MCAR20,MCAR40,MNAR,block8; common TEST subjects across all stress modes and scenarios, not five independent cohorts.
- The same **TRAIN1700, VALID1 600, VALID2 600** labelled cases are available to ALL methods. Baselines explicitly include raw logistic with 1700 TRAIN labels and separately raw logistic with all 2900 TRAIN+VALID labels and sensor-shifted VALID2; show both, no false claim equal optimizer compute. Extra VALID labels are required for candidate admission, so comparing only TRAIN1700 raw is not enough.

## Environments fixed before evaluation

Exactly 5 registered conditions, one 900-episode TEST cohort each (same underlying TEST source episodes, interventions change sensor readings only):
1. \`physical_benign_shift\`: physical source predicts Y, VALID2 channels 4..11 multiply by 1.15 and shift +.10 (benign scale+offset drift); TEST unshifted. Expected to retain candidate for group 4/5/6 if NLL remains helpful; report if it fails.
2. \`physical_known_flip\`: physical source predicts Y, VALID2 sensors4..6 sign flipped; same flip also in TEST. Expected risk gate rejection; if it accepts, report failure.
3. \`artifact_known_flip\`: ARTIFACT source group9/10 predicts Y, VALID2 sensors9/10 sign flipped; same flip in TEST. A robust risk gate should reject this *predictively useful but nonphysiological* source. Admission must not be mistaken for proving physiology.
4. \`physical_UNSEEN_flip\`: same physical outcome law, VALID2 benign drift (same as #1), but TEST sensors4..6 sign flipped. A VALID-gate using known environments CANNOT guarantee protection; expected admission and TEST harm are not a surprise and must be kept in results.
5. \`complete_benign_shift\`: complete K2 outcome law, VALID2 benign drift, TEST unshifted; extra source should be refused unless a false positive occurs.

For each scenario report auto-selector under SHIFTED-VALID gate and a deliberate **same-distribution/unshifted-VALID control** built from the exact same train / VALID patient IDs but with unshifted VALID2 measurements (NO additional labels). This isolates the effect of observing an environment shift. The no-shift gate is a legitimate internal control **not** a second independent data cohort. Also report a no-guard predictor using the highest validation-supported candidate from unshifted VALID, if available. Do not count TEST as independent from earlier scenarios.

## Baselines on identical access budget

- Frozen known K2, no candidate;
- \`AUTO_INVARIANT\`: automatic candidates with shifted-VALID2 admission, preserving original K2 head;
- \`AUTO_IID_GATE\`: same candidate proposals and gamma but original unchanged VALID2 for admission (negative control);
- \`MANUAL_PAIR\`: privileged known channel pair4/5 under the SAME shifted-VALID admission rule, not a realistic automatic proposal;
- \`PCA_UNKNOWN\`: TRAIN-only PCA source of unknown channels4..11 under the SAME shifted-VALID admission rule;
- \`ALL_GROUPS_REFIT\`: known K2+all unsupervised groups, refit ordinary logistic on TRAIN1700 (this changes original K2 coefficients, report that);
- \`RAW_TRAIN1700\`: StandardScaler+LogisticRegression on recorded 12-sensor summaries TRAIN1700 only;
- \`RAW_ALL2900\`: same raw classifier fit on the union TRAIN1700+VALID1 600+shifted VALID2 600, i.e. equal TOTAL labelled cases available to gate (but DIFFERENT label-use and compute);
- privileged true-K2 and true-relevant-extra-source logistic controls on synthetic oracle state for audit only, never model fitting/selection.

Report full held-out TEST AUROC, AUPRC, NLL, Brier, source use/refusal, per-env VALID gain/confidence, source assignment, no hidden bypass and EXACT additive head. Include raw fit and all-model calibration. For major paired comparisons AUTO_INVARIANT vs frozen K2 and vs AUTO_IID_GATE at TEST/MCAR20, calculate **paired patient bootstrap CI95** (399 fixed re-samples; aggregate mean AP difference over all 3 trained seeds for each sampled patient cohort). This is uncertainty from the 900 shared TEST subjects **conditional on the 3 fitted models**, not CI over new training draws/generators.

## Interpretation / stop criteria

- If physical_benign_shift does NOT admit or does not improve predictive quality, report failure; do not adjust thresholds on TEST.
- If physical_known_flip or artifact_known_flip STILL admit, report failure; do not post-hoc change VALID shifts.
- In physical_UNSEEN_flip, any admitted source that harms TEST is an important negative result.
- COMPLETE should normally refuse candidates, but a false admission cannot be censored.
- Even if AUTO_INVARIANT outperforms AUTO_IID_GATE under a known shift, this is a distinction between **predictive association under observed environments**, NOT physical/causal identification. A benign instrumental artifact can still pass.
- If RAW_ALL2900 or ALL_GROUPS_REFIT outperform our K2-frozen source in clean conditions, explicitly report it. Need full published CBM/IB-CBM/Residual CBM baseline implementation and externally validated concept semantics before a novel-method article claim.
- Old PR #26 and V30 empirical results, accepted PIERE manuscript, IITI papers and sealed PhysioNet test cohorts remain unchanged.

**Reproducibility:** All policy choices in this protocol frozen before experiment outputs. GitHub Actions CPU pinned, full smoke+3 neural seeds, negative TEST and shifted validation receipts plus immutable artifact by commit SHA. No silent selection of a favorable subset of environments.
