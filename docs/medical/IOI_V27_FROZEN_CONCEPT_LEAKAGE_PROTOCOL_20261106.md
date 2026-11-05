# IOI ConceptFAN V27 — FROZEN fresh-draw concept leakage vs staged concept training

**Freeze date:** 2026-11-06. **Status:** protocol fixed before examining *V27* generated labels, fitted models or TEST metrics. This study follows V25/V26 and uses the **same previously inspected simulator family**, but a distinct **fresh random generator draw with seed 20261107**; it is *not* an external generalization or clinical validation.

## New research question

V26 showed that a direct linear head on K2 learned concept summaries could outperform an oracle linear predictor supplied with the two true concept summaries in an omitted-factor synthetic world. This suggests (does not prove) **label-driven task information leakage into the named K2 concepts**. A core architectural question separate from accepted PIERE cross-run reproducibility: Does isolating concept formation from outcome gradients reduce extra predictive information beyond a predefined concept dictionary?

### Fresh data and controlled worlds

Use the existing fixed V25 `frozen_world(seed=20261107, n_train=1600,n_cal=400,n_test=800)`, with exactly 16 time steps, 6 noisy sensors, 3 independent AR factors and binary outcome laws:
1. COMPLETE: outcome depends only on true first two concept means; third measured factor is irrelevant.
2. OMITTED: outcome additionally depends on third true factor, absent from K2 dictionary.
3. ARTIFACT: observationally identical OMITTED data/outcomes, but third sensor direction is an instrumental proxy for an unobserved prognostic driver rather than an identified physical factor. Reuse OMITTED results, never count additional model fits.

Use the same fixed synthetic INPUT and concept supervision for COMPLETE vs OMITTED; only outcome laws differ. Train corruption MCAR20 with seed **20262101**; TEST corruptions are `MCAR20`, `MCAR40`, `MNAR` with independent seeds **20263300 + index in V25 STRESS tuple**. No TEST data/labels used to calibrate or choose anything. Keep all 400 calibration subjects unused.

### Matched architecture and training schedule

Native canonical K=2 `MultiSetAdditiveTemporalConceptFANModel`: encoder1/latent24/heads2/FFN48/dropout0, 3 sigmoid fuzzy membership sets and native attention pool; exact same initialization seeds 11,23,37, AdamW LR0.002/weight_decay0.001, batch128, **14 epochs** and same per-epoch batch order for both arms.

- **JOINT:** native training objective `BCE(Y,z_native)+5*MSE(C_hat,C*_TRAIN)`; outcome gradients update the *concept path*, temporal pooling, fuzzy membership and output head.
- **STAGED:** for the exact same first 14 epochs and batch steps, optimize `5*MSE(C_hat,C*_TRAIN)` ONLY; outcome gradients do not enter concept formation. After training, freeze the concept path and fit the exact same simple additive `logit=b+sum_k w_k q_k` decision head on **the same 1600 TRAIN outcomes**.
- In JOINT, after the 14 epochs freeze the concept path and fit the **same** linear head with same sklearn settings and TRAIN labels. Thus both arms receive the same postfit risk-head solver, TRAIN labels and TEST evaluation; training step count equal but outcome label gradient exposure and computational graphs differ as the experimental treatment. Do NOT claim fully equal FLOPs/label-gradient exposure.
- K2 concept gold supplied only in synthetic TRAIN to supervise both. In real ICU data these privileged concept targets are generally absent; no automatic discovery claim.

### Evaluation

For each NN seed and outcome family, compare on exactly the same fresh TEST episodes and MCAR20/40/MNAR: AUPRC, AUROC, Brier, NLL, concept trajectory MSE, exact additive logit reconstruction discrepancy and immutable core state hashes. Include TRAIN-only raw-sensor logistic benchmark and oracle true K2/K3 logistic ceiling (oracle generator labels supplied for evaluation).

**Primary semantic-leakage diagnostic**, conditional on true K2 concepts on fresh TEST:
1. TRAIN logistic: `Y | C*_1,C*_2`.
2. TRAIN logistic: `Y | C*_1,C*_2, C_hat1,C_hat2`, where `C_hat` is from the trained JOINT or STAGED encoder with MCAR20 masks. No leaked TEST labels.
3. Paired 499-subject bootstrap of TEST NLL reduction for model2 versus model1. In COMPLETE a positive residual gain would be a concerning false positive (potential probe misspecification/finite sample), so report it.
4. Train-only Ridge regression predicting `C_hat` from true K2, then test association of residual concept errors with **oracle third** true factor (evaluation-only); strong residual association in OMITTED may indicate an unapproved third source stored inside supposedly K2 concepts. This is an associative audit, not a causal source identification.

**Hypothesis falsifiers:** separated training may show equally high conditional extra information, worse MSE/forecast, or no practical improvement; any such result is reported. Higher AUPRC under JOINT is not automatically desirable if it comes from injecting undisclosed extra factor into named concepts; lower AUPRC in STAGED is not an automatic proof of semantic purity. Statistical/causal interpretation explicitly limited to controlled generator family.

**Publication separation:** Do not reuse PIERE 30-fit Spearman values as new. This experiment concerns supervised concept identity leakage and conditional dictionary completeness, not cross-training explanation agreement. No real clinical cohort or tests are executed and V25/V26 previous data are not overwritten. Results stored as immutable GitHub CI artifact keyed by commit SHA.
