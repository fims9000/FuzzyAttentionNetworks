# ConceptFAN V20 — executed optimizer attribution and distinct-family generalization (2026-11-04)

## Scope and evidence
V19e had improved predictive AUPRC through TRAIN-only optimization of **the existing native decision weights and bias** while leaving concept trajectories, forecasts, reconstructions and mask-only observation process unchanged. But its optimizer, pos-weighting, and optimization budget differed from original joint training. V20a split two of these factors. V20b tested the same native-head correction on different nonlinear, regime and missing-observation generating processes, with a new fixed synthetic TEST cohort. V20c adds a conservative fit-disjoint acceptance gate as an optional safety policy; see `CONCEPTFAN_V20C_FROZEN_TRAIN_ONLY_REFIT_GATE_20261104.md`.

All results below come from executed GitHub Actions and complete-denominator CSV/JSON data stored in `docs/medical/evidence_v20/head_factorial/` and `docs/medical/evidence_v20/distinct_generators/`. The original accepted PIERE paper was not touched. These are **synthetic benchmarks**, not physical concept identity or hospital validation.

## A. V20a: optimizer vs class weighting, same native concept summaries
[Actions run 41837565372](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41837565372) succeeded with 9 original pretrained models and 36 *native* risk-head refits, all 45 opportunities stored. All optimizer arms began from identical prefit ConceptFAN K3 coefficients, used the exact same frozen concept summaries C and original mask-only care offsets, and modified only `physical.decision_raw_weight` and `physical.decision_bias`. No student/teacher predictor or raw-to-logit bypass.

| Experiment (mean validation AUPRC, 3 seeds) | Positive outcome | Negative outcome | Independent null target |
|---|---:|---:|---:|
| Original end-to-end K2→K3 native | 0.587819 | 0.604346 | 0.453344 |
| LBFGS, ordinary BCE | **0.650434** | **0.686267** | 0.432619 |
| LBFGS, original TRAIN pos-weighted BCE | 0.650407 | 0.686167 | 0.432755 |
| AdamW, ordinary BCE | 0.650418 | 0.686257 | 0.432737 |
| AdamW, original TRAIN pos-weighted BCE | 0.650383 | 0.686158 | 0.432861 |

Optimizer choice and pos-weighting have negligible differences on this deliberately *roughly class-balanced* generator. In the positive-outcome task, TRAIN prevalence was 0.51625, thus pos_weight was only 0.93705; this setup cannot support a general claim that class weighting never matters in imbalanced real ICU prediction. The native beta heads did not approach the saturation bound ±3 in any of these arm results, weakening a weight-range saturation hypothesis for these runs.

**Mechanistic interpretation:** Additional head-only optimization of fixed representations is responsible for most of V19e's improvement under the current settings, not any unique magic of LBFGS. It does **not** cleanly distinguish few-joint-epoch budget, concept feature drift or auxiliary-loss conflicts; those causes still require same-budget comparison with longer joint training. In the null outcome, both methods degrade heldout AUPRC, cautioning against default extra fitting without a separate task-relevance check.

## B. V20b: distinct generator families, new heldout synthetic TEST
[Actions run 41838565224](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41838565224) succeeded: **12 original native networks, 24 refitted variants, 36 complete model test evaluations and 4 raw-feature logistic predictive controls**. TRAIN/validation/test generated using independent predefined seeds 305011/305012/305013, with 1000/300/300 episodes per family. TRAIN-only medians/IQR; K3 fixed *observed sensor-coordinate directions* to isolate downstream learning, **not** automatically discovered or clinically grounded concepts. No artificial human annotation and no latent source states in the trained data objects.

| Generator family (synthetic TEST mean AUPRC over 3 NN seeds) | Original native K3 | Unweighted LBFGS native refit | Original-weighted LBFGS native refit | Plain logistic on raw observed historical summaries |
|---|---:|---:|---:|---:|
| Nonlinear delayed | 0.468453 | **0.545155** | 0.545136 | 0.566756 |
| Regime switching | 0.485549 | **0.630408** | 0.630226 | 0.598530 |
| State-informative measurement missingness | 0.448520 | **0.513495** | 0.513321 | 0.615722 |
| Outcome independent of measurements (null) | 0.447302 | **0.416827 (worse)** | 0.416775 | 0.435957 |

All three nonnull families improved in synthetic TEST AUPRC and Brier with direct native-head refit. In the regime-switch family, the chosen native concept approach outranked the simple **prediction-only** logistic feature baseline on this particular TEST partition, but not on the nonlinear or informative-sampling families. In the informative missingness case, the prediction gap versus logistic remains large. On the null task, refit decreased TEST AUPRC despite a small Brier change in the favorable direction. Do not turn three neural seeds on the same 300 test episodes into an unjustified population significance claim.

This is an improved optimization procedure within ConceptFAN's exact additive concept bottleneck, not a proven new semantic discovery algorithm or a benchmark win over a published automatic concept model.

## C. Mandatory conceptual/clinical limits
The new generator fixes three known **measured sensor axes** before model training, so success cannot be interpreted as automatic physiological concept discovery. Earlier V18 falsification showed that *outcome-predictive temporally colored sensor noise* passed both independent source proposal and predictive screening in 3/3 cases. Signed coefficients may be useful for observational prediction, but have no license to be called causal clinical mechanisms. The risk screen V18 detects some task-irrelevant candidates but cannot identify whether a source is physical or instrumental without additional provenance or observations.

## Engineering state
- `native_head_factorial_v20.py` trains only original native ConceptFAN decision parameters by AdamW or LBFGS with controlled TRAIN BCE weighting. Does not add an inference model.
- `distinct_synthetic_v20b.py` adds three distinct dynamical observation families + outcome-null, seeded independent train/dev/test partitions.
- `experiment_head_factorial_v20.py` and `experiment_distinct_families_v20b.py` execute all frozen ablations and commit machine-readable receipt CSV/JSON.
- `native_head_refit_gate_v20c.py` implements optional independent fit-disjoint TRAIN acceptance with risk-prediction one-sided permutation and bootstrap gain tests; do not deploy before its own executed empirical evidence.
- All modifications are in Draft PR #17 based on the V19 Draft experiment branch; current mainline architecture is unchanged.

## What remains to determine
1. Whether longer **joint** training with the original V10 optimization objective closes the same gap (iso-epoch/budget control), or whether freezing concepts is essential.
2. Whether a fit-disjoint TRAIN acceptance rule can reject false-positive null refits without too many false refusals of genuine predictive improvements. That requires explicitly reporting all gate decisions, not hand-selecting.
3. Independent clinician/instrument provenance grounding, real ICU missingness/transfer and literature-matched intrinsic concept baselines. Until those exist, do not claim a medically meaningful new physical law or accurate physical interpretation of learned concepts.

Original PIERE camera-ready remains unchanged. The reports retain all positive and negative experiment outcomes.
