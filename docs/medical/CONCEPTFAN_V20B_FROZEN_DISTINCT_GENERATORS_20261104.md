# V20b — frozen distinct-generator native risk refit stress test (2026-11-04)

## Purpose
V18–V20 optimizer experiments reuse a simple linear autoregressive generator with a few seeds and a repeatedly inspected validation cohort. That is useful for diagnosis but **not independent algorithmic generalization evidence**. V20b freezes a separate non-linear and non-stationary simulator with different observation processes, no learned source discovery and no clinician concept targets. It tests only the robustness of native **train-only decision-head refitting** from V19e/V20.

## New generator / independent partitions
Four predefined generator families, each generated with *independent random numbers per train/validation/test split*, each split 1000 TRAIN / 300 VAL / 300 synthetic TEST:
- `nonlinear_delayed`: coupled non-linear latent recurrence, lag-dependent outcome.
- `regime_switch`: abrupt change in latent persistence and observation mixing inside each episode, with regime variation across episodes.
- `informative_sampling`: non-linear dynamics with observation masks dependent on measured state (non-random missingness); physical concept UNKNOWN-neutral behavior is retained and mask-only care branch is kept separate.
- `outcome_null`: the same dynamic measurement family but binary target independent of state; detects label overfitting and spurious risk promotion.

Synthetic seeds fixed in source before execution: TRAIN 305011, VALIDATION 305012, TEST 305013. These are **not** new independent hospitals or real physical mechanisms. Hidden generator states are only used internally to generate measurements/outcome and are not included in model input/benchmark data objects. All train normalization medians/IQR computed using TRAIN and observed values exclusively. No labels or measurements from TEST enter training, hypothesis selection, normalization, risk screening or hyperparameter adjustment. Test is scored exactly once per frozen variant; no test-based tuning.

There are **predefined three observable measurement directions** (sensor_0, sensor_1, sensor_2), deliberately not an automatically-discovered or expert-certified set. This isolates downstream native risk-head optimization from upstream concept-source identifiability. Unlike V18, there is no independent discovery/risk sign gate; therefore no claim that any concept represents physiological cause or can be deployed as a clinical diagnosis.

## Frozen baselines and training
Three neural seeds 11/23/37. Each model: V10 K3 native physical concept layer, 12 epochs of identical current V10 joint task / forecast / denoising objective, exact additive native decision, original care-only observation-process concepts, V16 native feature-gradient bridge. All models see same TRAIN episodes and batch ordering. Refit two copies of each baseline using same frozen native concept trajectories and original head state:
1. `no_refit`: exact original joint trained native head.
2. `lbfgs_unweighted`: original actual decision weights/bias refit TRAIN-only using BCE+effective-beta L2 .001, 120 LBFGS steps.
3. `lbfgs_original_class_weight`: same frozen concepts/starting parameters, original TRAIN neg/pos class weight in BCE, same L2 and optimizer.

Four families × 3 seeds × 3 arms = **36 native-model evaluations**, with 12 initial joint trained networks and 24 actual-native-refit variants. Fit one additional external **predictive** LogisticRegression baseline per family on explicit observed recency level, early-vs-recent trend, variation and missingness summary (TRAIN only, fixed C=1). This baseline is *not* an intrinsic concept model; it is here to quantify prediction opportunity cost.

Evaluate all three arms on fixed VALIDATION and TEST using AUPRC, AUROC, Brier. Record TRAIN outcome prevalence and TEST outcome prevalence, mean native concept signed effect magnitude, maximum additive-logit reconstruction error, and identical other model parameters under head-only refit; report all fits, every null case and all 4 logistic predictive controls. Do not select model arms after viewing any TEST results. Decision criterion predeclared: on at least two of three non-null generator families, both absolute AUPRC improvement over native non-refit and Brier change must be reported; improvement isn't declared universal unless every family/test direction agrees.

## Epistemic limits
Even a stronger task result is **not** evidence that concepts are physically meaningful or robust under sensor artifacts. V18 established a predictive colored instrument artifact can pass task relevance screening. Missingness-aware future observation-process features are not biomedical concepts. A model can give useful predictions with the wrong underlying physical story. Before clinical or publication claims, add independent real expert annotations and external instrument metadata, and compare official concept bottleneck methods.

Accepted PIERE camera-ready is not modified; this remains a Draft exploratory PR.
