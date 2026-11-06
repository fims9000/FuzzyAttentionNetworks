# ConceptFAN V43 — preregistered live instrument monitoring and matched-physical-reference raw regression

**Frozen on 6 Nov 2026, BEFORE ANY V43 TEST outcomes were generated or inspected.** Parent: [V42 PR #39](https://github.com/fims9000/FuzzyAttentionNetworks/pull/39) (synthetic, reproducible, 79–84% wrong physical concept estimates among accepted under unseen OFF drift). No retuning V42 on its TEST. This study tests new EXTERNAL physical evidence, not a new general identification theorem.

## What changes, physically

V42 had a historic independent noisy physical-energy reference R = E + noise only for TRAIN300/VALID1 200/VALID2 200; its correctness could deteriorate at deployment without a current reference. V43 adds an independent **contemporaneous nuisance sensor** measuring the instrumentation background \`U_on(t)\` and \`U_off(t)\`, each with iid Gaussian noise std 0.06 in each of 16 temporal positions, with known unit gain and *no direct physical slow source S*. These are two genuinely NEW measurements supplied to TEST runtime, not inferred from hidden S or used as TEST Y or TEST R. They are **simulated from the nuisance U by the generator**, so the study does NOT establish real hardware feasibility. Existing channel10 was intentionally blind to the aligned artifact; the new control is a different, privileged experimental device.

Under matched background U_on = U_off the measured control difference has only difference noise. Under TEST-only OFF drift U_off = U_on +1.4 * AR(t), this control generally reveals drift. It does **NOT** detect (i) OFF-path source leakage without changing U, (ii) A-only sensor inversion, (iii) an absent/broken monitor, (iv) a monitor that hides the actual OFF drift, (v) nonlinear \`g(H S + H_0 U) - g(H_0 U)\` dependence on even equal U. All counterexamples must be tested and reported.

## Pre-TEST fixed monitoring rule and validity receipt

Let \`c_on[n,t]\` and \`c_off[n,t]\` be **actual observed control samples**, with separate strict BOOL masks; require ≥8 jointly observed time points per patient. Define
\[
D_i=\left(\frac1{n_i}\sum_{t\in\Omega_i}(c_{\mathrm{on},i,t}-c_{\mathrm{off},i,t})^2\right)^{1/2}.
\]
On 1000 healthy TRAIN devices episodes with full monitor, freeze \(\tau=Q_{0.99}(D_\mathrm{TRAIN})\). Independently verify on 400 healthy VALID1 and 400 healthy VALID2 that false alarm fraction ≤3%, and at least 90% episodes have sufficient observed monitor points. If these checks fail, **do not issue a runtime certificate** for that device. No Y, R, S, H or TEST response used in this measurement rule. If TEST monitor missing/insufficient, **fail closed** for an extra physical estimate; never infer U from observed source sensors as a substitute. When \`D_i>\tau\`, suppress the physical-energy output as \`REJECT_LIVE_BACKGROUND_MISMATCH\`. A pass is only a conditional certificate of matching monitored U, NOT proof of fully correct source isolation or sensor health.

Existing V42 has exactly the same TRAIN-only V37 source directions, V41 pair agreement and V42 300+200+200 external R calibration; no hyperparameters of source extraction or physical estimate changed. V43 source output = V42 physical source estimate if and only if live monitor certificate passes, otherwise explicit null. Historical R-calibrated interval for NOISY R cannot be upgraded to unconditional truth-E coverage.

## Fresh simulated devices and registered TEST faults

New independent device random seeds never used V35–V42:
- two linear mixing / AR devices \`20261707\`, \`20261709\`
- two nonlinear switching/heavy-tail + pre-transfer nonlinear instrument devices \`20261721\`, \`20261723\`.

Each device: TRAIN1000, VALID1=400, VALID2=400, TEST600, original 16×12 ON/OFF sensors plus two 16-long nuisance control arrays; fresh physical-energy reference R available **only** in 300 TRAIN +200 VALID1 +200 VALID2 records. Four TEST missingness modes \`complete\`, \`MCAR20\`, \`MCAR40\`, \`missing_view_A\` with original source ON/OFF masks shared across all fault worlds per device.

Ten frozen worlds (same patient S, outcome Y, initial source/background U for paired comparisons):
1. \`healthy\`: true correct matched ON/OFF background, independent current monitor correct.
2. \`matched_U8\`: huge common aligned background U×8 in ON and OFF; sentinel should pass. Linear exact subtraction guaranteed, nonlinear not.
3. \`OFF_drift\`: future-only U_off ≠ U_on, independently measured sentinel detects mismatch when measurable.
4. \`OFF_source_leak\`: 50% physical source remains in OFF; sentinel only monitors U, expected to miss.
5. \`OFF_missing\`: no actual OFF source readings (0 paired source).
6. \`VALID2_known_drift\`: damaged V2 historical reference input; V42 physical mapping should refuse before TEST.
7. \`monitor_blind_drift\`: actual U_off drifts, but current monitor erroneously reports matched background (adversarial device fault); live check may fail.
8. \`monitor_false_alarm\`: source ON/OFF healthy but nuisance monitor OFF erroneously reports drift; false refusals/coverage loss required.
9. \`monitor_missing\`: no current nuisance measurements; V43 refuses, V42 history may still predict but is not current certified.
10. \`A_ON_inversion\`: one original source view changes sign; independent nuisance monitor correctly says U matched; semantic source may still be wrong.

No model/threshold/pair may inspect true S/H, TEST E, TEST R or TEST Y. True E is used in a separate frozen audit after all predictions are computed.

## FAIR physical-energy regression baselines

Use exactly the *same observed source sensors* as ConceptFAN: actual ON channels 0..3, actual measured paired ON−OFF difference channels 4..9, strict Boolean masks, classical v40 190ish time-domain lag/quadratic statistics (fixed beforehand). No simulator oracle H/S or extra true reference.
- \`Ridge300\` = StandardScaler + Ridge(alpha=10), fit TRAIN300 noisy independent R labels.
- \`HGB300\` = HistGradientBoostingRegressor(max_leaf_nodes=15,max_iter=120,learning_rate=.065,max_depth=4,min_samples_leaf=20,l2_regularization=1,early_stopping=False,random_state=20264301), fit TRAIN300 noisy R labels.
- \`Ridge700\`, \`HGB700\`: same frozen algorithms fit TRAIN300+VALID1 200+VALID2 200 noisy R labels (explicit extra fit-label budget). Historic V42 also **consumes all 700 R** (300 fit + VALID1 interval calibration + VALID2 approval), so RAW700 is a better comparison of total reference-label budget, though usage pattern/compute unequal.
- Each raw predictor is reported (i) **un-gated** (all TEST, for prediction-quality controls), (ii) independently **sentinel-gated using same current nuisance evidence** and ≥8 actual paired source sensor readings, and (iii) on the exact **intersection** of TEST subjects accepted by both V43 and the raw method. Report physical MAE and error>0.5 TRAIN-IQR E among accepted, plus total coverage, not just AUPRC of Y.

Raw HGB/Ridge may outperform V43; report negative result. Raw regressors are privileged R-supervised physical predictors, not automatically interpretable physical concept discovery.

## Predeclared scientific outcomes

Per device×fault×stress×method: acceptance % of ALL TEST, accepted-subject normalized MAE, wrong accepted fraction threshold \(|\hat E-E| > 0.5\,IQR_\mathrm{TRAIN}(E)\) (this uses simulator truth ONLY for audit); true E in old R nominal interval for V42/V43; nuisance monitor sensitivity under OFF drift, false-alarm rate under healthy, false acceptance when monitor blind, and V43 loss of coverage under corrupt monitor. Explicit single-pair/2+ multi-pair acceptances and source pair identity.

**Primary safety quantity:** \`wrong_and_accepted_per_all_TEST\` = (wrong among issued physical estimates)/600. This penalizes BOTH high error and unnecessary acceptance; report together with correct-and-accepted and rejection rate to prevent the trivial "always reject" policy from being called superior.

Preregistered paired TEST comparisons for \`MCAR20\` and \`MCAR40\`:
- historical V42 vs V43 live monitor: wrong-and-accepted fractions for healthy, OFF drift, monitor-blind drift, false alarm, leakage;
- V43 vs **HGB300**/**HGB700** with the same current control, on intersection subjects and raw entire accepted subject set: normalized physical MAE;
- V43 healthy vs V43 OFF drift: source coverage and wrong accepted fraction;
- V43 healthy vs V43 monitor-blind drift: residual vulnerability.
Use 399 fixed-patient paired bootstrap resamples, 4 devices. **Rejection alone is not proof of physical correctness**. CIs conditional on four generator devices, not across real-world instruments.

## Verification requirements and non-claims

- Frozen candidate source fit never sees independent nuisance monitor values (they only gate physical estimates); can show conceptual provenance but not intrinsic current robustness.
- Synthetic sentinel is privileged side information. No real hardware or clinical interpretation. Cannot assert "self-calibrating autonomous concepts".
- Test dead/saturated/misreporting sentinel, OFF source leak, nonlinear transfer, full OFF loss, single-pair support. JSON user-path save/reload and batch-1 parity if implementable.
- Match same sensor mask and TEST subjects across all worlds. Save every TEST accepted mask/physical estimate/monitor score and frozen true E AUDIT_ONLY, + all baseline forecasts and training provenance; independently recompute all summaries from raw TEST arrays and preregistered confidence intervals.
- Keep PR DRAFT, do not merge master, do not update any V42 frozen TEST result or choose new thresholds after looking at V43 TEST.
