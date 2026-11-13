# ConceptFAN V60 — global typed-program architecture on two real hospital datasets

**Executed 13 November 2026; research-only, exploratory.** [Draft PR #69](https://github.com/fims9000/FuzzyAttentionNetworks/pull/69). Frozen pre-outcome protocol: `CONCEPTFAN_V60_PROGRAM_COMPILER_FROZEN_PROTOCOL_20261113.md`. Successful original independent CI runs: [BIDMC #43482375248](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43482375248) and [BIDMC→VitalDB #43482639703](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43482639703). All 32 private per-window TEST prediction files were checked by an independent runner-local script for source hashes, exact additive native PyTorch risk, exact K2 fallback, patient-group-macro loss and scalar metrics. **Private vital signs, patient IDs and risk arrays are not uploaded**; the artifact exposes aggregated metrics and program passports only.

## Major architectural change
Earlier V54 source discovery required a delayed cross-sensor correlation to exceed the simultaneous association by ≥0.30. That produced **zero** proposals from real BIDMC. V60 replaces this mechanism and the fixed K2-only concept choices with an explicit outcome-blind **typed program compiler** and a trainable sparse **native concept-only risk head**.

The finite grammar contains 24 original-channel measurement programs: last-four and 16-second levels, trailing trend, volatility, differential energy, above-TRAIN-quantile events, and same-time pair contrast/coactivity. Measured values and causal masks only; no ECG HR labels, future observations, simulation oracle, or raw-to-risk network shortcut. Three independent TRAIN patient folds screen coverage and variation, redundant expressions are removed, program values scaled using TRAIN median and IQR. Exactly **21 programs** survived on BIDMC simultaneous windows and **20** on the new +60s HR task; each native risk head admitted at most **six active programs** after TRAIN-only sparse fitting.

Two separate native PyTorch alternatives are fitted: pure compiled ProgramFAN and a residual correction to frozen native K2. Model variant chosen only by TRAIN patient-macro NLL. Deployment uses a strict one-sided whole-**patient** bootstrap rule on both separate VALID cohorts (six patients each), not the previous overlapping-window bootstrap. Failed admission means **bitwise exact original K2**, even when the rejected proposed model later looks good on TEST. Every signed source contribution is accessible and audited.

## Real BIDMC and cross-hospital VitalDB TEST

Real BIDMC dataset from PhysioNet (DOI 10.13026/C2208R), 53 official files and 46 identifiable patient groups, split TRAIN 27 / VALID1 6 / VALID2 6 / TEST 7 distinct patients. TEST windows: 472 for simultaneous HR and 408 for +60s forecast. The separate VitalDB external test reused the frozen V59 cohort of **40** patients, 2313 simultaneous windows / 2010 +60s future windows. **These sites and their outcome trends were inspected in V58/V59 before designing V60, so this is not an untouched confirmatory validation.** VitalDB RESP is capnography-based rather than BIDMC impedance: domain shift is not a controlled single-factor experiment.

Both outcomes are *monitor-derived threshold tasks*, ECG/HR ≥100 bpm; in +60s task the target begins 60 seconds after all input samples, with no future feature leakage. Neither is a clinically adjudicated outcome, medical device or evidence for causal physiology.

### Future +60s HR, natural missingness: AUPRC↑ / NLL↓

| Model | BIDMC TEST 7 patients | VitalDB TEST 40 patients |
|---|---:|---:|
| Original native FAN K2 | **.0797 / .5606** | **.4370 / .5557** |
| Pure new compiled ProgramFAN, ungated | **.1802 / .0994** | **.7507 / .2720** |
| **Dual-VALID gated ProgramFAN** | **.1628 / .1027** | **.7496 / .2750** |
| Logistic on original input features | .2096 / .1054 | .7506 / .4940 |
| Logistic on engineered input features | .2115 / .0972 | .7528 / .5004 |
| Strong PULSE-only logistic | **.2416 / .0841** | **.7552 / .3886** |

Numbers averaged over two fixed K2 seeds 11 and 23. **Same test people across both seeds; no pseudo-doubling of sample size**. Both +60s native models passed the preregistered dual VALID patient-cluster rule. This demonstrates a real cross-site improvement against old K2 and competitive descriptive ranking with simple logistic, but NOT superiority over the PULSE-only ranking baseline. ProgramFAN's external NLL is smaller than the logistic baselines on this exploratory site.

### Simultaneous HR: crucial negative gate outcome

| Model | BIDMC AUPRC / NLL | VitalDB AUPRC / NLL |
|---|---:|---:|
| Native K2 and deployed gated FAN | .0728 / .5557 | .5615 / .5451 |
| **Ungated pure ProgramFAN** | .1740 / .0945 | .9606 / .0825 |
| PULSE-only logistic | .2297 / .0811 | .9602 / .1325 |

Although the proposed alternative performed well retrospectively, it was **rejected after checking both BIDMC VALID groups**: VALID1 passed, but VALID2's 99%-one-sided patient-group lower bound crossed zero. **No threshold was adjusted** on inspected TEST scores. The deployed model therefore remained exact K2 on both sites; claiming deployed AUPRC .9606 would be false.

### Stress: VitalDB +60s, gated model

| Perturbation | ProgramFAN AUPRC | ProgramFAN NLL |
|---|---:|---:|
| Natural missingness | .7496 | .2750 |
| Additional MCAR20 on inputs | .7636 | .2285 |
| Entire PULSE channel missing | .1358 | .5631 |
| PULSE monitor input biased +25 bpm | .7362 | **1.1898** |

The simulated per-window MCAR perturbation does **not** model a physical sensor failure. Sensor absence and unseen bias cause severe loss of reliability; model is not deployment-ready. On the original BIDMC +60s TEST there is only **one patient with positive events among seven**; AUPRC uncertainty is large despite hundreds of overlapping windows.

## Critical concept-level finding
Selected +60s programs (same six in both native seeds): last-four PULSE level, full-window PULSE level, PULSE high-event fraction, PULSE−RESP contrast, PULSE−SpO2 contrast, PULSE/SpO2 simultaneous coactivity. **All six read PULSE.** V60 proves configurable outcome-blind **statistical concept generation**, not independently discovered physiological entities. Its success may largely be a flexible interpretable monitor-PULSE score. This is why explicit comparison with a simple PULSE-only predictor is indispensable.


## Independent posthoc source-dependence and PULSE-only patient audit

A **separately labeled posthoc** diagnostic was added only **after** the primary V60 BIDMC/VitalDB TEST metrics had been inspected. It reruns the exact frozen TRAIN/VALID architecture and checks only previously saved TEST predictions, with **no model, operator, threshold, or admission update**. Successful [BIDMC CI #43483484969](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43483484969) and [BIDMC→VitalDB CI #43483601257](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43483601257), with `posthoc_v60_source_dependence.json` in the aggregate-only artifact. The baseline comparison resamples **entire patient groups** (399 repetitions), not overlapping windows. No adjustments for multiple experiments are applied; the sites were already inspected, so these are descriptive associations, not confirmatory hypothesis tests.

- **All six active programs in both tasks and both native seeds include the PULSE sensor.** This is a fundamental negative outcome for *independent physiological source discovery*: the learned head is a transparent but PULSE-dominated risk function, not new evidence for distinct autonomic/mechanical source identities.
- VitalDB **future +60s, deployed model minus PULSE-only logistic**: ΔAUPRC around **−0.0056**, posthoc 95% patient-cluster interval **[−.033,+.033]** (seed11; seed23 [−.035,+.029]). The rank difference is not statistically resolved. Conversely, patient-macro NLL difference interval is **[−.252,−.020]** (seed11) and **[−.253,−.012]** (seed23); V60 has **lower descriptive patient-weighted NLL** on this particular inspected external site.
- VitalDB **simultaneous ungated PURE** vs PULSE-only logistic: ΔAUPRC **+.00039**; interval approximately **[−.004,+.003]**, again no demonstrated rank advantage. The *deployed* gated model was rejected on BIDMC VALID2 and remains exact K2; it performs far worse than the PULSE baseline. Its apparently strong alternative cannot be claimed as an accepted model after looking at TEST.
- A model with the same accuracy-ranking as a simple monitor-based score can still be useful for explicit provenance, component-level auditing, refusing unsupported concepts, and constraining direct shortcuts. **None of this is evidence of a newly discovered causal physical variable or guaranteed robustness**.

An additional cross-workflow audit compared the independent BIDMC-only and BIDMC+VitalDB CI executions. The frozen program-bank SHA fingerprints and all reported AUPRC values for BIDMC matched exactly; patient loss summaries varied by <1e−4 due to nondeterministic floating-point CPU training. We do **not** claim bitwise reproducibility of PyTorch training weights, only case-level audit and numerical outcome reproducibility within this tolerance.

## What has and has not been solved
**Implemented and verified:** coherent measured-program grammar, TRAIN-only program passports and robust scaling, per-patient stability and group-weighted risk fitting, two distinct sparse native concepts-only heads, exact additive attribution, protected K2 fallback, patient-level dual admission, a non-simultaneous 60-second target, real-site evaluation on two hospitals and independent case-level audit.

**Not established:** physically or medically identifiable concepts; novel independent sensor information beyond PULSE; generalizable causal explanation; superiority to engineered baselines; clinical prospective diagnosis or early warning; stability under device bias, missing sensors or external assay inconsistency. V60 used two already-inspected hospital datasets — a genuinely new, untouched reference source and a non-tautological multi-signal target are still needed before confirmatory claims. **Keep this PR in Draft; do not merge into master or retrofit previously submitted articles.**
