# ConceptFAN V22e — frozen *external predictive model* comparison (2026-11-04)

## Scientific motivation
V22a/d demonstrated that a train-native ConceptFAN concept layer with independently risk-screened source slots can improve its own predictive AUPRC on synthetic dense multisensor mixtures, including with the same 800 TRAIN labels as a longer original native joint-training model. But **improvement against one's own architecture is not proof of superiority against established predictive baselines**. V22e therefore directly compares **external non-concept predictor models** on the same outcomes/test subjects. These are *prediction* controls, not fair concept-fidelity or official CBM reference architectures.

V22e protocol is committed **before** obtaining outcomes on brand-new synthetic TEST seed **522323** (n=300 per family). Never re-examine 522203/522313 to choose hyperparameters for V22e. Already studied structural generator families are not new.

## Frozen independent input/training
Four V22 families (distinct source variances, proportional variance, sensor gain drift, instrument-only variance), synthetic TRAIN generator seed **522201**, n=800, all five observable sensor channels, historical length=24, 6-step future unused by external predictors. TRAIN outcomes available for all 800. For ConceptFAN, first 600 TRAIN histories only fit pooled PCA whitened input and environment BSS axes; last 200 independent TRAIN labels only determine source outcome relevance signs and a structural no-risk source mask. Native K2 concept models train for exactly 10 epochs, batch 96, original V10 joint task+forecast+reconstruction objective, model seeds 11/23/37, no extra head-only optimization. Unscreened BSS method ABSTAINS in proportional variance and unstable calibration drift cases.

**External predictive controls, using the SAME 800 TRAIN outcomes with no oracle truth, no environment ID and no validation tuning:**

1. `logistic_30_temporal_summaries`: for each of five observable sensors, the original/raw 24-hour history produces six predeclared scalars: last-6 mean, previous-6 mean, full-24 mean, last observed value, difference between last-6 and previous-6 means, and standard deviation of last-12; 5×6=30 features. Pipeline `StandardScaler()` (TRAIN-fit only) and `LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', class_weight=None)`.
2. `hgb_30_temporal_summaries`: identical 30 raw history features, sklearn `HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=0.1, random_state=11)`. All tuning frozen without looking at TEST seed 522323.
3. `native_pca_600fit_200screen`: existing V22d intrinsic two-concept risk architecture using PCA whitened source axes, exactly same 600+200 training-label budget.
4. `native_environment_600fit_200screen`: same native architecture, source axes from established covariance BSS where TRAIN identifiable; otherwise explicit ABSTAIN, **no silent PCA fallback**.

No source oracle/clinical labels used by any model. Outcome labels are visible for supervised external controls only on TRAIN. Validation seed 522202 may be used by original V10 fit for logging but cannot select/tune any arm; TEST seed 522323 must not be instantiated until all **18 native models + eight external predictors** have fitted and all source/risk decisions are finalized.

## Metrics and evidence
Per family × method, TEST AUPRC, AUROC, Brier and logloss. Native: three independent model initializations on one shared 300-case TEST and mean/range; external logistic/HGB: one deterministic TRAIN fit (not three artificial seeds). Record source-vs-instrument oracle source alignment for native slots only, **after all fitting**; external predictive models do not claim an interpretable native source, so mark source alignment unavailable. Record all 24 registered native opportunities, 18 trained / 6 source refusals, exactly eight external model fits and 26 TEST evaluations. Native model SHA256 must be invariant before/after TEST scoring; external sklearn estimators must be immutable on prediction.

The comparison deliberately tests **prediction utility**, not whether a model recovers physical concepts. HGB/logistic see engineered raw sensor summaries, while native models see two whitened latent measurement components; they have different hypothesis classes and differing input representations and are not architectural equivalents. This is nevertheless essential as a concrete check against simple established non-concept prediction methods and has no official CBM/MAGNETS baseline claim.

## Expected scientific outcome handling
If external models outperform ConceptFAN, report that directly; do not tune on TEST or relabel a negative result as a concept-fidelity victory. If ConceptFAN outranks them, note small synthetic cohorts, limited 30-feature baselines and lack of strong deep temporal baselines. Regardless of predictive wins, sensor artifact/physical causal meaning remains unresolved; no clinical accuracy, physiology discovery, physician annotation, hospital transfer or principled novel BSS theorem is established.

This extends Draft PR #19; PR #18 and accepted IEEE PIERE camera-ready remain unchanged.
