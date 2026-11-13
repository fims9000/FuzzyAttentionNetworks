# ConceptFAN V54: independently verified nonlinear measured operator discovery

**13 November 2026 · EXECUTED EXPLORATORY**. Draft [PR #56](https://github.com/fims9000/FuzzyAttentionNetworks/pull/56). Frozen protocol: `CONCEPTFAN_V54_NONLINEAR_OPERATOR_FROZEN_PROTOCOL_20261113.md`. Verified CI [run 43468880863](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43468880863), source commit `cd93a722826d32b2532b8030a38f67f53263c23c`, artifact `conceptfan-v54-nonlinear-operator-library` (ID `14059588175`, SHA256 `0569d26b32188be10bed5b230330de87f7c50efaa797f9378b46c1a71caab461`).

## New architectural capability
- Unsupervised TRAIN-only operator discovery from previously unassigned masked channels, three subject folds, lags 1–3, explicit **SIGNED_LAG** and **EVEN_ENERGY_LAG** families, same-time correlation controls, independent source passports. The latter is *nonlinear and sign invariant* because it squares each observed sensor before scoring.
- Source uses only sensor[i,t−lag] and sensor[j,t], no lookahead; missingness is neutral. At most three discovered pairs compete, exactly one can be admitted by two independent VALID cohorts.
- Real native `TemporalConceptFANModel` is used with a signed concept-only risk head; source q flows through **native torch K3 forward** rather than an approximate external addition. Refused/unavailable source gives **bitwise exact frozen K2**.
- Falsifying modes: (1) outcome-relevant hidden nonlinear synthetic energy, (2) irrelevant delays, (3) predictive **recording artifact**. Both meaningful energy and device artifact can be accepted; no genuine physical semantics follows.

## Registered synthetic protocol and case-level verification
New 16-step 12-channel synthetic generator, 950 TRAIN, 400 VALID1, 400 VALID2, 650 shared TEST cases per mode, seeds 11/23 (6 K2 native models, 30 TEST prediction arrays). TEST stresses: complete, MCAR20, MCAR40, sign flip, and unseen 1.5× paired sensor gain. All 30 arrays independently rechecked for saved hashes, identical patient labels across seeds, AUPRC/NLL, signed source decomposition, and exact fallback. 15 scenario-level posthoc paired patient bootstraps use 399 resamples; descriptive only.

### MCAR20 (mean of two native seeds)

| Mode | K2 | V53 signed-only | V54 operator library | raw logistic | engineered energy logistic |
|---|---:|---:|---:|---:|---:|
| Useful nonlinear source AUPRC | 0.59937 | 0.59937 | **0.75092** | 0.58393 | **0.79664** |
| Irrelevant delays AUPRC | 0.73958 | 0.73958 | **0.73958** | 0.74029 | 0.73112 |
| Predictive recording artifact AUPRC | 0.61463 | 0.61463 | **0.77692** | 0.59561 | **0.82732** |
| Useful nonlinear source NLL | 0.67259 | 0.67259 | 0.64249 | 0.67907 | **0.52573** |
| Predictive artifact NLL | 0.67191 | 0.67191 | 0.63888 | 0.65867 | **0.48542** |

Selected family for both useful and artifact modes: EVEN_ENERGY_LAG (channels 4/5 and 8/9 respectively); both seeds refuse all new concepts in the irrelevant mode.

Posthoc paired 95% descriptive intervals:
- Useful source V54−K2 AUPRC **+0.15155 [ +0.11248, +0.19298 ]**.
- Useful source V54−engineered logistic **−0.04572 [−0.07586, −0.01761]**.
- Predictive artifact V54−K2 **+0.16229 [ +0.12455, +0.20009 ]**.
- Predictive artifact V54−engineered logistic **−0.05040 [−0.07509, −0.02113]**.
- Irrelevant source V54−K2 **exactly zero** (all test cases, both seeds).

On a TEST-only sign flip of both selected measured channels, **the EVEN source's AUPRC and NLL are exactly unchanged** (0.75092 / 0.64249 on useful source), reflecting algebraic parity, not physical robustness. On a TEST-only 1.5× gain change, V54 useful-source AUPRC 0.75468 and NLL 0.63513; engineered logistic's NLL worsens 0.52573→1.12722. These are descriptive on this specific generator, not a universal drift claim.

## Critical scientific limitations / next experiment
- A strong equal-label **energy-engineered logistic** still beats the native FAN on normal MCAR20 in both predictive modes, and FAN NLL is appreciably worse. V54 demonstrates **automatic measured-concept formation and selective admission**, not superior risk prediction.
- Concept meanings remain observational statistical associations. A predictive **device artifact** is accepted, with no independent assay. V50 rotational nonidentifiability remains unresolved. Physical and clinical naming prohibited.
- A frozen, finite operator library is neither universal nonlinear discovery nor an end-to-end learned new concept definition; V54 trains only its TRAIN risk coefficient, not the operator itself. There is no external multi-hospital clinical validation.
- All bootstrap intervals unadjusted for multiplicity; both initializations share TEST episodes; source family and generator were designed for this scientific hypothesis, so external transfer remains open.

**Next coherent architecture requirement:** jointly learn a *constrained* new operator calibration and source-risk coefficient from TRAIN while anchoring it to the discovered measurement program, then gate on dual VALID and compare it on a fresh independent generator against V54 and equal-label engineered baselines. Keep exact refusal, operator passport and separate semantic assurance.
