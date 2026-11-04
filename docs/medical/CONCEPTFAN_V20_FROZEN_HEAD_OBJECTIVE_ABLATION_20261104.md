# V20 — frozen head-objective and optimizer factorial ablation (2026-11-04)

## Main scientific question

V19d found that TRAIN-only logistic regression using **the frozen ConceptFAN K3 native summary** can achieve AUPRC 0.65096/0.68565, vs 0.58782/0.60435 from the jointly fitted native decision layer on a previously inspected synthetic generator (positive/negative outcome modes). V19e directly refitted the actual ConceptFAN coefficient/bias with **LBFGS + UNWEIGHTED BCE + L2=0.001**, obtaining AUPRC 0.65043/0.68623 without any post-hoc inference path. However V19e changed multiple things at once: class weighting, optimizer, and optimization budget. Do NOT attribute improvement solely to auxiliary-task competition.

This experiment isolates the effect of the class-weighted objective and optimization method on the **same exact frozen concept trajectories** and the **same initialized original model**.

## Frozen protocol

Data, architecture and source discovery: the same previously inspected V18 5-channel synthetic data (generator seed 20411204), 800 TRAIN / 200 validation / 80 sealed test. Three outcome modes positive, negative and null. Fixed K2→K3 model with V17 unsupervised TRAIN-only discovered measurement direction on 350 TRAIN episodes, disjoint TRAIN-only risk screen on 300 episodes, same 6-epoch K2 + 6-epoch K3 fit and train-screened raw ±0.05 initial coefficient, three seeds (11,23,37). For null, force training only as a documented negative control; real deployment must ABSTAIN.

**No extra concept model, no raw-feature prediction bypass, no oracle factor or patient-specific export.** Physical concepts C, forecast, denoising and mask-only observation process are held fixed during every refit. Before fitting each variant restore the same original native physical decision_raw_weight and bias from the initial fitted model. Only update those two original parameters, and check all others are exactly unchanged after training.

Five optimizer/weighting arms, each on full original TRAIN only (no early stopping, no validation labels in training):
1. `no_refit`: 12-epoch joint native model as fitted.
2. `lbfgs_unweighted`: unweighted BCE + effective-beta L2=0.001, 120 L-BFGS iterations, strong Wolfe. Matches V19e.
3. `lbfgs_weighted`: **pos_weight = TRAIN negative / TRAIN positive** from the original multitask native objective; same 120 L-BFGS iterations and L2=0.001.
4. `adamw_unweighted`: same unweighted BCE + L2=0.001, AdamW head-only 300 full-data steps at LR 0.03 (no weight decay), same starting coefficients.
5. `adamw_weighted`: pos-weighted BCE + L2=0.001, AdamW full-data steps 300 at LR 0.03 (no weight decay).

3 outcome modes × 3 seeds × 5 arms = **45 opportunities**, including baseline copies. New optimizer code never sees any validation labels or sealed test data. Source labels from TRAIN are the only supervised signals. The optimizer comparison is not exactly compute-matched (iterations and algorithm differ), but all models use the same frozen inputs and initial native parameters.

Metrics: AUPRC/AUROC/Brier, TRAIN weighted and unweighted BCE objective before/after, mean absolute direct native K3 signed contribution, native coefficient magnitude/saturation (3*tanh bound), max exact-additive logit error, strict equality of frozen C and non-head parameters. Report all per-seed and aggregate rows, not only wins. No repeated validation tuning, no p-values based on three seeds. Distinguish *difference in fitted objective* from putative improvement in physical concept identity.

**Decision criterion:** If weighted-LBFGS and weighted-AdamW recover most of V19e gain, optimizer budget/optimization is relevant; if unweighted-LBFGS and unweighted-AdamW outperform weighted counterparts with comparable convergence, class weighting is implicated. If strong residual gap remains under matched heads, revisit score-parameter range and concept sufficiency. Multiple factors may be active. Negative/null outcomes must be reported even if performance decreases.

## Boundaries
This is a **mechanism-disentanglement diagnostic**, not a confirmatory hospital experiment. A target-predictive temporally correlated sensor artifact can still be admitted as observational pattern; that does not grant physical meaning. Added signed orientation is derived from TRAIN and does not prove natural cross-run explanation reproducibility. Original IEEE PIERE article remains untouched.

When a plausible training correction has been isolated, evaluate on a genuinely distinct synthetic data-generating family with different temporal dynamics and noise, not just a new random seed of the existing AR model.
