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


## D. V20c–V20d: independent TRAIN gate and corrected unused TEST seed

**Protocol integrity correction:** V20c reused synthetic TEST seed 305013 previously seen in V20b. This was an oversight in *claiming the TEST was novel*, not leakage into the gate: original model TRAIN (seed 305011, 800 episodes), gate (305014, 300 independently generated episodes) and test (305013, 300 episodes) were distinct *within V20c*, and no test labels informed its trained head or gate decisions. Nevertheless, the TEST had been inspected in the prior experiment. **We must not cite V20c's TEST as fresh confirmatory evidence.** V20c all-case evidence remains unchanged and available in `evidence_v20/refit_gate/`.

Frozen corrective protocol `CONCEPTFAN_V20D_FROZEN_FRESH_TEST_SEED_20261104.md` was written **before viewing V20d TEST metrics**. It keeps training, gate and the predeclared acceptance thresholds unchanged, but scores a genuinely unused synthetic TEST seed **305115**. GitHub Actions [41840551546 — SUCCESS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41840551546), all 12 gate decisions and all 36 TEST arm evaluations completed, verified in `evidence_v20/fresh_test_305115/`.

Train-only acceptance: independent fit-disjoint 300-episode GATE; one-sided candidate risk permutation p<=0.01 on 499 permutations, mean per-patient gate logloss improvement >=0.005, bootstrap 90% CI lower >0 across 499 resamples. If the gate refuses, the deployed choice remains EXACTLY the original native ConceptFAN. This is a conservative *predictive improvement selector*, not a clinical concept validation.

### Full TEST results after the unchanged gate (AUPRC, 3 neural initializations)

| Structural generator | Gate acceptances | Original V10 | Ungated native refit candidate | **Gated selected native model** |
|---|---:|---:|---:|---:|
| Nonlinear delayed | 2/3 | 0.559604 | 0.617174 | **0.618430** |
| Regime switch | 3/3 | 0.546188 | 0.696175 | **0.696175** |
| Informative state-dependent missingness | 2/3 | 0.558358 | 0.609861 | **0.611266** |
| Independent null target | **0/3** | 0.575557 | **0.552238 (worse)** | **0.575557 (original preserved)** |

For the null generator, the gate rejected all three refit candidates despite them reaching lower core TRAIN losses; on the freshly unused synthetic TEST seed they indeed decreased both AUPRC and Brier (Brier rose from 0.25010 original to 0.25382 candidate). The policy retained the original predictions exactly, preventing that degradation in this controlled study. It also rejected 2/9 non-null refits, underlining the false-refusal cost of conservative criteria.

Although V20d uses a previously unevaluated **TEST random draw**, its simulator families and acceptance policy had already been inspected and modified during prior V20 research. It is therefore independent test-seed evidence **within the same known simulation families**, **not a prospective clinical or new-family confirmation**, and not grounds for a population-level false-positive guarantee. All three neural models per family are evaluated on the same test episodes, hence initialization results are dependent.

### Decision and non-claims

- **Retain as optional train-native capability:** V20 native head-only reoptimization on full TRAIN observations, followed by a fit-disjoint, TRAIN-only ACCEPT/KEEP_ORIGINAL policy; no raw-input bypass or extra inference head.
- **Do not enable automatically for arbitrary tasks** without independent gate data, since label-null control and weakly prognostic tasks may be harmed.
- **Do not cite class-weighting irrelevance generally:** current TRAIN prevalence was near balanced and V20a was conducted on a narrow family.
- **Do not claim physiological source identity, expert-validated semantics or real-world safety:** the gate tests only risk-prediction improvement. Earlier V18 showed predictive instrument artifacts survive a task association check.
- **Outstanding scientific research:** isolate joint-training epoch/optimizer budget under more balanced budgets, official intrinsic-concept baselines, negative controls with colored sensor artifacts on the new generator, a real clinician feedback trial, and independent clinical cohort external validation. None is solved by V20.


## E. V20e: larger budget of the ORIGINAL joint objective, not just a head-only optimization comparison

Frozen preregistration: `CONCEPTFAN_V20E_FROZEN_ADDITIONAL_JOINT_EPOCHS_20261104.md`. [GitHub Actions 41841521716 — SUCCESS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41841521716), 36 completed native model evaluations (3 outcome modes × 3 seeds × 4 arms); machine-readable full denominator `docs/medical/evidence_v20/joint_epochs/{ALL_36_JOINT_VERSUS_HEAD_FITS.csv,SUMMARY.csv,STATUS.json,COMMIT_SHA.txt}`.

All arms start from the **same originally trained** 6-epoch K2 + 6-epoch native K3 model on the repeatedly inspected V18 synthetic family. The additional joint-training arms continue exactly the original V10 combined supervised task, future forecast, masked reconstruction, operator regularization and entropy loss for 6 or 12 extra epochs. Joint training uses explicit new random-order and masking seeds and **restarts the AdamW optimizer every six-epoch block**; optimization cost is not matched to head-only AdamW. The comparison refits only the *actual* native decision weights/bias on frozen concept features for 300 full-batch AdamW steps, with the native concept bank, forecast and process head identical.

| Arm | Positive-source AUPRC | Negative-source AUPRC | Null-target AUPRC |
|---|---:|---:|---:|
| Original 6 K2 + 6 K3 | 0.587819 | 0.604346 | 0.453344 |
| +6 more joint epochs | 0.610107 | 0.635028 | 0.442300 |
| +12 more joint epochs | 0.627088 | 0.652216 | 0.437485 |
| Same native model, 300 head-only AdamW steps | **0.650418** | **0.686257** | **0.432737 (most harm)** |

**Interpretation:** the original native model was partly short-trained: more joint epochs improve AUPRC on task-relevant sources. However this **does not close the gap** to head-only optimization under the fixed studied budgets. Concept summaries drift during further joint training (mean absolute displacement positive 0.00883 / negative 0.01150 after 12 extra epochs), while head-only refit leaves them **bit-for-bit unchanged**. Results support a separate head optimization phase as a useful practical design at these budgets, but do **not** isolate why (loss coupling, optimization path, weight dynamics, exposure counts). The null-target deterioration shows why the independent TRAIN acceptance gate is not optional.

## Research decision after V20e

The largest *demonstrated practical limitation* of the original prediction graph is no longer simply lack of extra concept capacity or temporal attention. The proposed **optional** training process is:
1. Train measurable native concept trajectories and their auxiliary forecast/denoising tasks jointly (unchanged V10 concept graph).
2. On TRAIN core only, optionally refit **only native decision coefficients and bias** using frozen concepts; the original model architecture is preserved.
3. On an independent fit-disjoint labelled **TRAIN gate** accept only if a preregistered predictive association and logloss-improvement rule passes; otherwise **retain original model**.
4. On external TEST, report all attempted changes and negative controls. Do not call source identity physical or causal without independent measurement provenance/expert confirmation.

**No clinician-proven concept semantics or hidden physical law identified.** A V20b/c/d family with fixed observed sensor-coordinate axes is not automatic concept discovery. The next substantive scientific goal is to evaluate end-to-end concept correctness, human-editable positive/negative concept annotations and automatic novel-concept admission against external, explicitly **physically annotated** independent data. Without those labels/provenance one cannot substantiate the intended scientific claim, regardless of predictive gains.
