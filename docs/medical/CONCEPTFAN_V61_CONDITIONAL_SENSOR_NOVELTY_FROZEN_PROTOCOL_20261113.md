# ConceptFAN V61 — conditional sensor novelty beyond the dominant PULSE channel

**Frozen before V61 test execution, 13 November 2026.** Base commit: V60 draft PR #69. Branch \`experiment/conceptfan-v61-conditional-novelty-20261113\`. V53–V60 findings and original code remain unchanged. This is a **new exploratory architecture test** on previously inspected hospitals, NOT prospective external validation.

## Falsifiable architecture objective
V60 generated 20–21 measurable programs but **all six selected sources depended on PULSE**. Its improvement could therefore be a nonlinear pulse-only score. V61 tests a strictly stronger requirement:

> A newly named measured concept is admitted only if its contribution to an already trained, frozen **PULSE-only concept head** is independently useful and its value is **not fully predictable from PULSE-derived programs**.

"Independent" means *conditional statistical information and residual task prediction*, NOT anatomical independence, causal discovery, trusted measurement identity or clinical utility.

## Two real future (non-contemporaneous) tasks
1. **Primary HR+60s:** mean ECG-derived monitor HR ≥100 bpm over four samples starting **60s after** the input window ends. Predictor: 16s of PPG PULSE, RESP and SpO2. The independent-source candidate must use **only RESP, SpO2, or their pair**, never PULSE. Strong existing monitor PULSE-only controls remain.
2. **Secondary RESP+60s:** mean measured monitor RESP ≥20 breaths/min across four samples 60s after input end. Input and concepts use only **PULSE and SpO2**; RESP is **hard-excluded** from *all* input sensor values, preprocessing statistics, operator discovery, source calibration and baseline comparisons. The target RESP monitor uses impedance in BIDMC vs capnography in VitalDB; this is a substantial source-modality shift, not a clinical ground truth adjudication. Do not choose a new threshold from validation/test prevalence. If any split is one class, report unestimable AUPRC without changing the split, task or 20 bpm threshold.

Two sites: official BIDMC PhysioNet 53 original recordings and real MIMIC patient split **27 TRAIN /6 VALID1 /6 VALID2 /7 TEST** inherited from V58, and independent hospital VitalDB **40 actual cases** from frozen V59 source-availability enrollment. Both sites already inspected in previous research cycles and V60, so both outcomes remain **exploratory**.

Sixteen-second causal measured windows, 8s stride. Target starts 60 seconds after input end; no future inputs, target HR/RESP never in the new program compiler, risk model or TRAIN normalization. Preserve actual missingness, input-only extra MCAR20, complete PULSE outage and input-only PULSE +25bpm shift; all derived sources recomputed from input sensor values. No TEST refit, threshold search, target-label leakage, or domain adaptation.

## Outcome-blind typed grammar
Use exact V60 24-program grammar and fold coverage/variance/redundancy controls, but add a **hard raw-channel allowlist before TRAIN statistics and operator generation**:
- HR task: PULSE (0), RESP (1), SpO2 (2) allowed.
- RESP task: PULSE (0), SpO2 (2) allowed; RESP is excluded even from quantile fitting, covariance screening and discovery ranking.

Among surviving candidates, **PULSE-only six unary programs** define the dominant concept head, fitted as native V60 \`PURE\` PyTorch signed bounded-risk head on TRAIN labels with two fixed seeds 11/23. Its contributions and risk score remain frozen thereafter. Non-PULSE candidates are programs involving only the allowed raw non-PULSE channels; any program involving PULSE, including PULSE−RESP/SpO2 pair interactions, is prohibited from being claimed as independent.

## Source-only residualizer (no target labels)
Given TRAIN PULSE-only program values/support (\`C\`: six values + six availability support indicators) and each non-PULSE program \`Z_j\`, fit **Ridge(alpha=25, fit_intercept=True)** from \`C\` to \`Z_j\` on measured TRAIN episodes only, using equal total weight per true TRAIN patient. No Y or TEST values in Ridge. Residual \`R_j=Z_j−E_TRAIN[Z_j|C]\`, robustly centered/scaled with **TRAIN observed** median and IQR (floor 0.25), clipped to [−8,8]. A residual is included only when all three subject folds have ≥25 supported real episodes, ≥50% observed coverage, and normalized residual std ≥0.15; up to 14 raw non-PULSE programs. The residual is an **observational conditional innovation**, not established causal information. If no measurable residual sources survive, return frozen PULSE-only baseline.

If PULSE measurements fail (<50% input context observed) or non-PULSE source mask unavailable, its residual contribution must be **exactly zero** and the model must return frozen PULSE-only prediction. Never silently impute missing source values as measured. Source candidates preserve operator passports, permitted channel provenance, TRAIN Ridge coefficients, TRAIN normalization, and frozen source gamma.

## Training and dual-VALID admission
Use native V60 \`fit_program_head_v60\`, but train only **RESIDUAL**, not a second pure head:
\[
z_{\mathrm{V61}}=z_{\mathrm{PULSE\,only}}+
\sum_{j\in S,\ |S|\le6}\gamma_j(q_j(R_j)-0.5).
\]
Each q is bounded/availability-masked and was built from original named measurement programs. 220 TRAIN-only Adam steps +80 TRAIN-only gamma refit, group-equal weights, same V60 regularizers. Gamma magnitudes and selected program definitions are never selected on TEST. The **native PULSE-only head stays frozen** while the new signed residual head is fitted.

Allow deployment only if both VALID1 and VALID2 demonstrate:
- ≥4 genuine patient groups; patient-macro NLL improvement of at least **0.005**, 99% one-sided bootstrap lower bound >0 with **399 whole-patient** resamples (never overlapping windows);
- source coverage ≥50% of episodes;
- **conditional null**: at least **49 predeclared within-patient, PULSE-score-quartile, block-row permutations** of the entire multi-program source vector, performed on VALID with fixed seeds; the real patient-macro NLL gain must exceed the empirical **95th percentile** of shuffled gains by ≥0.003 in **each** VALID cohort, with ≥25% source rows actually reassigned within their patient and PULSE quartile. This shuffle control can be conservative and does not provide an exact test under autocorrelation/overlapping windows; report the achieved swap rate and its limitation. If too few rows can be shuffled or any one check fails, **abstain with bitwise exact PULSE-only risk**.

Evaluate proposed V61 and deployed V61 separately; never retroactively promote rejected models because TEST looks good. Negative controls: additional MCAR20, all PULSE missing, +25bpm PULSE input bias, and null-permuted source-only inference. In TEST, a separately labeled **diagnostic** null shuffling is allowed only to quantify association, not select/adapt the model.

## Strong baselines and metrics
- original V60 PULSE-only logistic on current PULSE monitor mean,
- improved **same-program PULSE-only logistic** on the six PULSE program values+availability,
- **same-program PULSE+non-PULSE logistic** on allowed values+availability (no Y from holdout), as the strongest simple non-neural baseline;
- native baseline pulse-concept-only head; proposed/gated conditional-residual ProgramFAN. Baselines must receive equal allowed source information, equal TRAIN labels and no target-channel leakage.
- Report all AUPRC (when estimable), pooled NLL, **patient-macro NLL**, episode source support, TRAIN residual R² per source, 99% patient-VALID gates, within-patient stratified null test gains, same-patient TEST paired 399-bootstrap intervals. Replicate two native seeds **without treating them as independent patients**.
- In TEST at most 40 external patients, and in BIDMC TEST only 7: uncertainty and prevalence shifts must be explicit. Compare against same-program logistic even if it outperforms V61.

## Integrity, privacy and scientific limits
- No original raw vital signs, clinical case identifiers, free-text patient details, per-person predictions or vital time series uploaded to GitHub artifact; private per-window predictions stay on runner only for independent case-level audit, which validates per-program signed decomposition, exact pulse fallback and patient-macro NLL.
- The only exported patient grouping metadata is **aggregate counts and public BIDMC/VitalDB record ordinals**. Sources are monitor-derived signals without authenticated independent physical assay. Conditional residualization can be confounded by patient condition, channel algorithms, temporal correlation and device changes; "novelty" is statistical, not biological/causal.
- If no non-PULSE source passes dual patient VALID or conditional null, **this is an informative failure**: do not lower thresholds, fabricate new measures or silently switch deployment to ungated model.
- Positive novelty on these already inspected datasets would be exploratory and **requires a new, untouched dataset and independent reference** before any SOTA/clinical/physical concept claim.
