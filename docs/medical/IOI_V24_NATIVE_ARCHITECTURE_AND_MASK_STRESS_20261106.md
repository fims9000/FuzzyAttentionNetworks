# IOI journal / ConceptFAN V24 — native integration and an explicit support-audit boundary

**Research status (2026-11-06): architecture test + simulator diagnostic, not a validated journal method.**

## Audit of earlier branches and publication overlap

- **IITI 2026 / concept-latent alignment**: prior architectural idea (not new journal novelty).
- **PIERE 2026 / 30 independent fits**: exact concept logit decomposition and *cross-run signed contribution reproducibility*, including numerical 0.0217/0.0035 Spearman results (not a new journal finding).
- **V6 + V7**: already have UNKNOWN-neutral physical concepts and a separate missingness/care-observation branch. Do not claim invented handling of missing values.
- **V10**: already has native measured-operator availability, multiscale concept pooling and exact additive logit.
- **V17–V22**: train-only residual source proposals, optional expert feedback, task-relevance screen and artifact falsification. On synthetic test, accurate source extraction and risk ranking do **not** imply physiology/causality. V22d equal 800-label comparison is not compute-matched.
- **V23**: separate real-PhysioNet *frozen protocol* for train-only probability recalibration (positive affine logit). It is not a published validation result and does not prove completeness.

These sources mean the only defensible planned IOI journal emphasis is **conditional predictive completeness of a prescribed dictionary and an explicit right to refuse unsupported physiological explanations**. Conditional predictive gain alone may describe an instrument artifact or model misspecification.

## Two independent code paths (do not conflate)

1. `fan.concept.ObservationAwareConceptFAN` is an **optional PIERE-style temporal ConceptFAN prototype**. It masks weakly observed contributions inside its own exact additive computation. It must be retrained and compared fairly before being considered an improvement.
2. `conceptfan_discovery.NativeV10SelectiveExplanationAudit` is a **new non-destructive V10/V19 wrapper**, built on the latest exploratory native *physical plus observation-process* architecture. It **never changes the original model prediction or native signed contributions**. It reads the original physical operator-availability tensor and the native physical-risk weights; descriptive-only sources do not force abstention. It reports a physiological-explanation abstention flag (not a rejection of all model forecasts).

The physiological concepts and the sampling-process concepts are mathematically different paths. An artificially hidden measurement may reduce *physical* reliability, but the process branch can receive the unchanged **original recording mask**. Sending the artificially thinned mask as if it were a clinician sampling decision would contaminate the interpretation.

## Verification results

- Local contract tests: **9 passed, 1 integration skipped** when canonical V10 source was absent in the standalone package.
- Real canonical GitHub CI (initial V24 run): [Actions 42110460699](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/42110460699) **31 passed, zero skipped**, including the original V10 model integration, the earlier PIERE-style multi-set core integration and legacy regression tests. No patient data needed.
- The four-condition oracle-completeness probe (complete/oracle-known versus added predictive source vs instrument artifact) checks **associative predictive information**, not automatic physical identification.

## Missingness-simulation sanity: baseline versus *selective reporting*, not a neural-model comparison

Executed `scripts/run_selective_missingness_protocol_v24.py`: 3 independent synthetic generator/estimator seeds, per seed **2400 TRAIN** and **1200 TEST** sequences, 24 time steps, three directly observed concept surrogates. A regularized logistic model is trained only under TRAIN MCAR20. Its concept values are observation-masked recent means. Test labels are generated from synthetic AR source states; generator truth is used only to measure diagnostic concept-estimation error. The original observation schedule remains fixed. A support threshold is selected from TRAIN coverage alone.

| Evaluation mechanism | TEST coverage | TEST overall log-loss | TEST selected-only log-loss | Avg. measured concept estimation error, accepted / refused |
|---|---:|---:|---:|---:|
| Fully observed | 100.0% | 0.545 | 0.545 | 0.091 / n.a. |
| MCAR 10% | 99.6% | 0.551 | 0.551 | 0.124 / 0.250 |
| MCAR 20% | 94.7% | 0.552 | 0.551 | 0.155 / 0.237 |
| MCAR 40% | 55.5% | 0.561 | **0.566 (worse!)** | 0.199 / 0.281 |
| MAR conditional on measured covariate | 52.9% | 0.561 | 0.539 | 0.150 / 0.228 |
| MNAR dependent on unobserved value | 26.3% | 0.602 | 0.582 | 0.281 / 0.463 |
| 12-step contiguous blocks | 40.0% | 0.587 | 0.553 | 0.163 / 0.346 |

Means across three independent simulator seeds; **these are not confidence intervals or confirmatory neural ConceptFAN results**. All probabilities come from the **same trained predictor** whether or not the support flag abstains. In particular, selection can worsen log-loss (MCAR40) despite lower average concept-estimation error on selected episodes. Thus a coverage/accuracy tradeoff must be reported honestly; this gate is not a universal accuracy improvement.

The simulator tests code paths and experimental metrics. It does NOT establish that V24 improves the original V10, PIERE, MIMIC-II or PhysioNet patients; cannot validate semantic or causal physical concept identity.

## Required full experiment before a scientific effectiveness claim

1. Freeze one native V10/V19 model family and a *separate* independent synthetic generator with: complete dictionary; known missing outcome factor; predictive measurement artifact; non-predictive artifact. Freeze train/validation/test and seeds **before evaluation**.
2. Compare (A) native V10, (B) the exact same frozen native V10 with V24 audit (must give identical logits), (C) label-budget and compute-matched V10+V17 residual proposal plus V22 train-only task screen. Use original V6/V7 reliability in all arms. Include standard CBM and non-CBM predictive references. No reuse of PIERE 30-fit metrics as novelty.
3. Train concept dictionaries only on TRAIN; expert/device provenance may inform training but may not be synthesized into a fake clinical certificate. Test oracle concepts are **audit only**.
4. Stress frozen TEST once with MCAR 10/20/40%, MAR, MNAR and blocked missingness. Keep actual observation-policy mask separate from artificial thinning mask. Report full-test AUPRC/AUROC/Brier/NLL, selected-only versions **with coverage denominators**, concept-estimation error on synthetic oracle, false-support/abstention, and exact additive decomposition.
5. Select/refine a threshold using separate TRAIN calibration or validation only. If the full-test predictive metric is unchanged with V24, report it as unchanged; report altered *selective* metric only alongside rejected counts and matched-subset baseline scores.
6. Treat V23's 300 already-inspected PhysioNet development cases as **exploratory** and preserve the separate 400 CALIBRATION + 600 TEST seals; do not imply external clinical validation from them. Require independent cohort, provenance and medical review for substantive clinical interpretation.
7. Report all training runs and failures, with patient/model as the independent units. Do not reinterpret 435 overlapping PIERE fit pairs as 435 independent seeds.

**Decision:** V24 fixes an observability *reporting boundary* on top of already developed V10, while V22d gives a promising, but compute-confounded, predictive architecture lead. The *new publication claim* cannot be defended until a controlled omitted-factor versus estimation-error experiment on the actual native model is executed.