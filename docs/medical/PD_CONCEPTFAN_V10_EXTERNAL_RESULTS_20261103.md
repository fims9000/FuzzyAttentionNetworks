# Executed V10 versus independent external prediction baselines — 2026-11-03

**Run:** https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41626717843
**Outcome:** SUCCESS, including focused code contracts and complete refits.
**Artifact:** `conceptfan-v10-vs-external-methods-1`.

The user requested a comparison for **ConceptFAN V10 only** against
architectures outside this project. V7, V8 and V9 are **not** present
as model rows in this benchmark. V10 is the pre-existing 1h/4h/12h/24h
physiological concept model with the cross-view consistency training term.

## Data and controlled training

- Original PhysioNet/CinC Challenge 2012 Set A, 2,400 original TRAIN
  patients and the **same 300 previously examined development**
  validation patients. The 400 original calibration and 600 original
  test patients are never inspected.
- Every method uses **exactly the same 12 train-selected observed
  variables**, with the same V10 train-only median/IQR scaling:
  DiasABP, HR, MAP, NIDiasABP, NIMAP, NISysABP, SysABP, Temp,
  Urine, Weight, RespRate, GCS.
- Each receives **36 historical hours**, not 48. The last 12h
  measurements are *training targets only for V10's self-supervised
  future-observation auxiliary task*, never classifier inputs.
- All classifiers can use original measurement-event masks as well
  as visible measurement masks, including during synthetic additional
  missingness. No model receives additional future patient data.
- Three initialization seeds for V10, GRU-D-style and Transformer
  (11, 23, 37), each 18 epochs with same train order, batch size,
  AdamW / class-balancing and observed-measurement 20% dropout.
  Logistic and HGB are each trained once with 1 augmented observed view
  per original training patient. This does **not** create extra people.
- External GRU-D-style network reproduces central train-only
  forward fill / input decay / hidden decay principles from
  Che et al. (2018, DOI: 10.1038/s41598-018-24271-9), **not** official
  trained weights or fully reproduced hyperparameters.
  Transformer is a fresh 2-layer, 4-head, hidden-64 implementation;
  no pretrained foundation weights. HGB is **scikit-learn
  HistGradientBoosting, not XGBoost**.

## Original historical-observation results

Values are mean across *distinct trained models* where applicable.
Uncalibrated Brier scores are reported only as descriptive scores,
because all methods were trained with class-imbalance weighting.

| Method | Models | AUPRC ↑ | AUROC ↑ | Brier ↓ |
|---|---:|---:|---:|---:|
| Time-series Transformer | 3 | **0.424756** | **0.787806** | 0.178954 |
| **ConceptFAN V10** | 3 | **0.379405** | 0.732681 | 0.214083 |
| Logistic history | 1 | 0.321336 | 0.757014 | 0.198810 |
| Histogram Gradient Boosting | 1 | 0.313649 | 0.746493 | **0.136049** |
| GRU-D-style | 3 | 0.297408 | 0.699120 | 0.212102 |

Transformer leads by 0.045352 AUPRC (11.95% relative to V10) on
the historical development cohort. Consequently **V10 cannot claim
predictive superiority to competing nonconcept architectures**.
Logistic and boosting achieve higher AUROC than V10 despite a lower AUPRC.

The theoretical constant predictor set to the **observed
development positive prevalence** of 0.14 would have Brier 0.1204,
below any of the raw class-weighted model scores. This is a diagnostic
calculation, **not a trained validation baseline**. It demonstrates why
separate out-of-sample calibration is necessary before medical risk
interpretation. AUC/AUPRC rank comparisons are unaffected by monotonic
recalibration; Brier values can change substantially.

## Robustness to *additional artificial* omissions

| Method | Original AUPRC | MCAR (35% further reduction) | Time/channel MAR |
|---|---:|---:|---:|
| Transformer | **0.424756** | **0.369060** | **0.406499** |
| ConceptFAN V10 | 0.379405 | 0.327838 | 0.387322 |
| Logistic | 0.321336 | 0.227444 | 0.257189 |
| HGB | 0.313649 | 0.317102 | 0.294133 |
| GRU-D-style | 0.297408 | 0.204438 | 0.271627 |

These are nested perturbations of the **same 300 people**. Additional
missingness is not an independent clinical policy or external hospital.
The HGB MCAR AUPRC happens to increase slightly; do not present it
as a cross-hospital robustness improvement.

## Finite sample and seed uncertainty

For the original view, V10-minus-external AUPRC mean differences,
and *descriptive* matched patient + available model-initialization
stratified bootstrap 2.5/97.5 percentile intervals (800 draws):

| External comparator | Δ AUPRC: V10 minus comparator | Descriptive interval |
|---|---:|---:|
| Transformer | **-0.045352** | [-0.152643, +0.079502] |
| Logistic | +0.058069 | [-0.054284, +0.152808] |
| HGB | +0.065756 | [-0.051873, +0.162826] |
| GRU-D-style | +0.081997 | [-0.006118, +0.172275] |

All **original-scenario intervals contain zero**. Only three
initializations support each neural architecture, and one model
supports each deterministic tabular method. The bootstrap does not
correct for repeated development reuse, method design after inspecting
previous results, hospital-source shift or larger data collection.
**Do not convert these intervals into confirmatory hypothesis tests.**

Per-initialization AUPRC:
- V10: seed 11 = 0.366577; seed 23 = 0.380552;
  seed 37 = 0.391084. Across-seed SD = 0.012294.
- Transformer: 0.460356, 0.362242, 0.451672;
  across-seed SD = 0.054313. This is **suggestive**, not proof,
  of smaller V10 predictor variation.
- GRU-D: 0.295568, 0.295393, 0.301263.

## Model complexity and CPU computation

| Neural model | Trainable parameters | Mean 18-epoch fit time | 300-case inference |
|---|---:|---:|---:|
| **ConceptFAN V10** | **1,410** | **15.82 s** | 0.0317 s |
| GRU-D-style | 20,505 | 8.76 s | 0.0242 s |
| Transformer | 72,705 | 26.46 s | 0.0379 s |

**V10 has approximately 51.6× fewer parameters than Transformer**
and a roughly 40% shorter mean CPU fit time under this specific
hosted-runner implementation, for an observed 0.0454 AUPRC
disadvantage. It is not globally the cheapest classifier:
Logistic trains in 0.23 s and HGB in 4.43 s.

A small interpretable predictive bottleneck is therefore the
defensible *model-size/explicit-decomposition* advantage of V10,
not top predictive accuracy. It still lacks independent evidence
that its physiological concepts correspond to true clinical
mechanisms; a formal explanation-quality comparison to an external
concept method would be a different experiment.

## Scientific decision

1. **Reject claim:** "ConceptFAN V10 is the best predictor among standard
   time-series models." Transformer currently has higher mean AUPRC
   and AUROC, but not proven significantly on a new cohort.
2. **Retain qualified claim:** V10 is a much more compact model with
   an explicitly inspectable concept decomposition and lower
   mean AUPRC variation over three starts.
3. **New next gate:** freeze V10 vs Transformer hyperparameters,
   use an independent patient cohort/hospital, more initialization
   seeds, and preregister the ranking and concept-fidelity metrics.
   Additionally test a real external concept-discovery model
   (e.g., MAGNETS) for stability/semantics rather than treating
   conventional predictive architectures as interpretability baselines.
4. **Do not migrate these exploratory results to accepted PIERE camera-ready.**

Reference:
https://www.nature.com/articles/s41598-018-24271-9

Canonical numerical evidence:
`v10_vs_external_summary.csv`,
`v10_vs_external_all_run_metrics.csv`,
`v10_vs_external_descriptive_paired_bootstrap.csv`,
`v10_vs_external_training_runtime.csv`,
and `STATUS.json` in the successful run artifact.
