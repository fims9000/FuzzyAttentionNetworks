# ConceptFAN V53 — executed native lagged-concept discovery, 13 November 2026

**Status: EXECUTED EXPLORATORY; NOT external or physical/causal validation.** Draft [PR #51](https://github.com/fims9000/FuzzyAttentionNetworks/pull/51). Frozen design: [V53 protocol](./CONCEPTFAN_V53_LAGGED_NATIVE_FROZEN_PROTOCOL_20261113.md). Successful full [GitHub Actions run 43437349058](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43437349058), source commit `01d7fd243e2e7f180693868f35a484bf1accd973`, result artifact `conceptfan-v53-lagged-native-research` ID `14043164857`, SHA256 `ed0e36511a6726ea0043494af07bf9a8c911c9d85ff8cb36c255230227e379eb`.

## 1. Genuine architectural change

For a new scalar observed-value concept, the *previously unseen source definition* is constructed without predefined physical concept targets or outcomes: three disjoint TRAIN subject folds must agree on a signed delayed pair at lag 1–3, with |r_lag| >= .65 and > maximum contemporaneous |r| by >=.30. Select disjoint pairs and record TRAIN-only centers/scales. The **temporally causal-as-a-filter** activation at t uses only source sensor at t-lag and readout sensor at t with strictly observed cells; missing signal yields neutral q=.5. Temporal ordering is NOT physiological intervention/causal direction.

Existing native `TemporalConceptFANModel` encoder inside `AnchoredJointConceptBottleneckV47` trains real K2, on observed two-sensor proxies (not hidden oracle factor targets). Only then the signed risk coefficient for each new source is fitted using TRAIN Y; two separate VALID cohorts admit at most one new slot. Risk logit is the exact known-K2 signed sum plus gamma*(q-.5), only for covered and approved source. No raw-to-risk bypass. An unused source yields bitwise **exact frozen K2**. This is dynamic **source/decision expansion**, not joint end-to-end training of the new concept's operator definition or confirmation of physical identity.

Auto proposals on this generator (three TRAIN stable delayed pairs, each lag2): channels 4/5, 6/7 and 8/9. Source count **K=2 or K=3** according to VALID evidence; the number 3 is not forced.

## 2. Real executions and uncertainty

Single 16-step, 12-channel *new synthetic* measurement generator, subject splits TRAIN950 / VALID1 400 / VALID2 400 / TEST650, scenario modes PHYSICAL_EXTRA, IRRELEVANT_SOURCE, PREDICTIVE_ARTIFACT; two native independent initializations per target, total **six trained models**. Each examined under complete, MCAR20, MCAR40 and unseen TEST-only flipping source/readout sensor sign (24 test rows). Ordinary raw logistic regression uses same TRAIN subject/label budget; two native seeds **share exactly the same 650 TEST people per scenario**. 11 new targeted architecture/negative tests plus existing measured-concept tests passed in CI.

### MCAR20 heldout AUPRC, mean of the two native models

| Scenario | Frozen K2 | New lagged auto K3 | Raw logistic |
|---|---:|---:|---:|
| New predictive synthetic source | **.658006** | **.770354** | **.774115** |
| Outcome-independent delayed source | **.750723** | **.750723** | **.742031** |
| Predictive *recording artifact* | **.699357** | **.771313** | **.778560** |

Independent recomputation from **all 24 individual TEST prediction arrays** `y,k2_logits,k3_logits,raw_logits,used,q,measured_support` and exact fallback passed. Posthoc paired patient bootstrap with 399 draws (both seeds share the same patient index resample), unadjusted 95% descriptive intervals for ΔAUPRC:
- PHYSICAL_EXTRA auto-K3 minus K2 **+.11235**, **[+.07210,+.15134]**; auto-K3 minus raw logistic **−.00376**, **[−.02498,+.01775]**.
- IRRELEVANT_SOURCE auto-K3 minus K2 **exactly 0**, CI **[0,0]** due to refusal on both seeds.
- PREDICTIVE_ARTIFACT auto-K3 minus K2 **+.07196**, **[+.03813,+.10003]**; auto-K3 minus raw logistic **−.00725**, **[−.02941,+.01326]**.

**The model does NOT show a significant predictive advantage over raw logistic.** In MCAR20 PHYSICAL_EXTRA NLL is **.63891 for K3 vs .52358 raw logistic**; for predictive artifact **.63245 vs .50903**. AUPRC gains over K2 must not be sold as better calibrated medical risk.

### Critical TEST-only inversion negative control

Under unknown sign flip of both currently selected sensors:
- PHYSICAL_EXTRA: K2 AUPRC **.658006**, auto-K3 **.484120**, paired Δ **−.17389**, 95% [−.20293,−.14136].
- PREDICTIVE_ARTIFACT: K2 **.699357**, auto-K3 **.484240**, paired Δ **−.21512**, 95% [−.24983,−.17860].
- Source proposal and TRAIN/VALID acceptance were identical; the shift was TEST-only. This violates the desideratum of a reliable concept and **must remain a negative result**.

The changes of sign of both paired channels preserve internal zero-lag and lagged absolute correlations under a symmetric source law. Thus **without an independently oriented reference** a source-internal correlation monitor cannot generally identify this shift. A new signal or expensive trusted external assay is an *information requirement*, not a missing threshold that one can fix by TEST fitting.

## 3. Exactly what has/has not been solved

**Completed and checked:**
- actual training and inference through the native ConceptFAN exact signed concept bottleneck;
- truly dynamic train-only *operator* definitions for time-lagged observed sources, masks and source passports, not manually predeclared pair 4/5;
- K2/K3 structural expansion gated solely by separate TRAIN and VALID, no hidden true generator sources, exact fallback and refusal;
- case-level reproducibility and independent numeric audit of each Y/probability, plus paired patient bootstrap;
- comparison with strong raw logistic and explicit instrument-artifact and unseen-sign-shift failures.

**Unresolved and not claimed:**
- automatic identification of **physical** or **clinical** concept semantics (V50 observational-twin counterexample still applies);
- unsupervised *nonlinear* source discovery without a transparent operator library; jointly learning new concept definitions + risk in V53 (gamma-only risk updates);
- robustness to sign-inversion without independent reference, external hospitals, cross-generator transfer or diverse real data;
- a superior AUPRC/NLL to raw baseline or clinical adoption.

### Next meaningful architecture step (separate new protocol)
Connect automatically proposed operator-based source to an independently **attested observation reference** (V50/V51/V52), and train its native signed decision head with exactly the same source membership used at inference. Test on new synthetic devices/generator families and real clinical data where physically verified references exist. A reliable `UNKNOWN_STATISTICAL_SOURCE` must remain the only naming mode without reference; failing source availability/measurement provenance must produce exact K2. Do not treat already inspected V53 TEST as tuning data. Avoid claiming physical causal validity from a high prediction metric.
