# ConceptFAN V40 — preregistered head-to-head equal-label difference baselines and nonlinear device family

**Frozen on 2026-11-06 BEFORE drawing/evaluating any V40 TEST outcomes.** Parent: [V39 completed PR #36](https://github.com/fims9000/FuzzyAttentionNetworks/pull/36). V39 showed paired ON−OFF cancellation of source-aligned instrument noise conditional on a real isolated OFF reference, but **did NOT compare a strong raw learner trained on ON−OFF differences only**. Do not retrofit V39 thresholds or select a V40 variant by inspected TEST.

## Hypotheses to falsify

1. **H1:** under genuine linear mixing and shared source-aligned corruption U on both ON and OFF, V39's measured paired-source concept remains stable; a non-concept learner given the **same measured ON−OFF difference and original K2 ON channels** can learn source energy too. No prediction-superiority claim should survive if raw matched-feature competitor performs at least as well.
2. **H2 (critical negative):** when the instrument applies a nonlinear transfer \`g(z)=z+0.035*z*abs(z)\` *before readout*, \`g(H*S+U)-g(U)\` is generally **not** \`H*S\`. Consequently the same physical ON/OFF source isolation **cannot algebraically cancel** common U; V39 linear argument fails and performance may deteriorate under U8. Do not relabel this as evidence for more fuzzy types or use TEST to tune g.
3. **H3 (critical negative):** if OFF reference has unobserved drift or leaks part of S, any paired-difference source can become misleading even in a linear instrument; sensor agreement and independent VALID are not dynamic physical guarantees.
4. **H4 (prediction/semantics):** a strong raw difference classifier can outperform the explicitly auditable concept model. Compare **both AP and proper scoring (NLL/Brier)** and source coverage. Record these negative results without post-hoc hyperparameter tuning. Raw prediction quality alone does not assign a correct physical name to its features.

## Fresh independent data

**Four newly drawn devices/cohorts never used by V36–V39:**

- linear Gaussian AR mixing, seeds **20261407, 20261409**
- nonlinear switching / heavy-tailed source dynamics and nonlinear instrument transfer, seeds **20261421, 20261423**

Per device: **TRAIN1000**, **VALID1=400**, **VALID2=400**, **TEST600**, 16 ordered time positions, 12 sensor channels. Same 2 independent known K2 factors through channels0:3 as previous native K2 architecture, 3 independently initialized **genuine Temporal ConceptFAN K2 networks seeds11/23/37**, 10 concept-only supervised TRAIN epochs with **no outcome Y gradient** into K2 encoder. Unknown source slow and competing fast source mixed by three independent unknown 2×2 matrices into A=(4,5), B=(6,7), C=(8,9). Matrix H is simulation-only; learner never receives H or latent source S. TRAIN original source-aligned common U has amplitude .35, invisible to external channel10. OFF switch physically removes both sources; ON contains both. Readout noise independent across views, SD=.19. For nonlinear family the source dynamics have tanh feedback, rare stochastic shocks and regime-volatility change, fast source non-Gaussian; \`g\` is applied to ON and OFF **before** noise, so even unchanged U does not in general disappear after subtraction. This is a different simulator family but does not prove external validation.

Outcome labels generated from sigmoid of original K2 true concepts plus true **demeaned slow-source energy** only in \`energy\` law; \`complete\` law has no extra source contribution. Source truth is used **only by simulator / post-fit audit**, not for source extraction or prediction.

## Registered TEST faults and missingness

Nine worlds on exactly the **same TEST patient IDs and outcomes** per device:

1. \`healthy_energy\`: no new fault;
2. \`common_U8_matched_ON_OFF\`: additional 8× physical source-aligned shared U applied BEFORE nonlinear transfer, equal ON and OFF;
3. \`common_U4_matched_ON_OFF\`: similar 4×;
4. \`OFF_only_U4_drift\`: only OFF reference shifts; condition fails;
5. \`OFF_source_leak_50pct\`: measured OFF contains 50% of source0; condition fails;
6. \`OFF_reference_unavailable\`: no OFF measurements of channels4:9;
7. \`constant_offset_both_2\`: after-transfer constant +2 on ON/OFF, should cancel for any g;
8. \`K2_complete_no_extra_source\`: original K2 Y law, new source should be refused;
9. \`VALID2_known_OFF_drift\`: OFF-only drift in VALID2 and TEST.

Five TEST observation modes: \`complete\`, \`MCAR20\`, \`MCAR40\`, \`synchronized_block8\`, \`missing_reference\`. Both ON and OFF have real masks, OFF channels4..9 baseline MCAR5%, paired difference available ONLY at **ON_mask & OFF_mask**, never imputed. Original K2 first4 test missingness controlled identically for **all methods/cases/stresses per device** (shared pre-registered MCAR20 known mask). Missing OFF->exact frozen K2 risk for concept approaches (raw classifiers may still produce a prediction from mask/known inputs, explicitly report that tradeoff).

## Main models / fair label budgets

Native K2 Transformer trained with observed K2 proxy labels only, frozen. Source measured through classical V37 masked, centered cross-view lag eigendecomposition on **TRAIN only** and projected through trained scalar signed gamma (TRAIN1000 outcome Y). Candidate definitions:
- \`ON_AB\`: original centered 2-view without OFF;
- \`DIFF_AB\`: masked paired ON−OFF centered 2-view;
- \`DIFF_ABC\`: masked paired ON−OFF 3-view and TRAIN-only view disagreement threshold.

If a source eigensystem cannot be identified under frozen V37 eigengap/bootstrap criteria, **refuse and report** rather than adjust eigengap or recover source from oracle H. Each source risk contribution must pass **two independent 400-case VALID cohorts**, same risk gate \`min_mean_gain=.003\`, \`min_support_fraction=.25\`, \`min_subject_support=.50\`, 399 bootstrap resamples. On a rejected episode, baseline is EXACT same frozen K2 probability (no hallucinated OFF values).

**Strong, truly information-matched raw baselines:**
- \`RAW_ON_LOGIT_TRAIN1000\`: original 0:3 ON + 4:9 ON measured temporal quadratic/lag statistics, 1000 TRAIN outcome labels.
- \`RAW_ON_LOGIT_ALL1800\`: same feature policy, 1000 TRAIN+400 VALID1+400 VALID2 labels (explicit *higher* outcome-label budget, not apples-to-apples).
- \`RAW_DIFF_LOGIT_TRAIN1000\`: original 0:3 ON + only the **actual ON−OFF difference channels4:9**, with missingness indicators and temporal means/variances/cross-products, 1000 TRAIN outcome labels.
- \`RAW_DIFF_LOGIT_ALL1800\`: identical features, 1800 outcome labels.
- \`RAW_DIFF_HGB_TRAIN1000\`: histogram gradient boosting classifier, max_leaf_nodes=15, max_iter=100, learning_rate=.065, l2_regularization=1, max_depth=4, min_samples_leaf=25, early_stopping=False, random_state=20264001, 1000 TRAIN labels.
- \`RAW_DIFF_HGB_ALL1800\`: identical algorithm and features, 1800 outcome labels.

Logistic: StandardScaler+LogisticRegression(C=1, max_iter=1500), no hyperparameter selection on TEST or VALID. HistGradientBoostingClassifier predict_proba uses only TRAIN labels for TRAIN1000 variant. All six baselines receive **only measured inputs and Boolean masks** and never a hidden source H/S or target energy. Known K2 inputs are given to ALL arms, so DIFF-only models are not penalized by removal of baseline prognostic information; raw model uses actual sensor readings rather than the K2 output. Raw models do **not** get semantic concept certification / abstention policy. Compute budgets (especially native K2 vs sklearn) are NOT matched; report this confound honestly.

Four devices × nine worlds × five masks × ten methods, three neural seeds for 4 methods gives **3240 per-model TEST evaluations** and **1800 aggregate rows**, **324 dual VALID source gate receipts** (4 devices × 9 worlds × 3 seeds × 3 sources), six fixed neural network risk/concept comparator variants? Actual NN fits =12. Save every original TEST patient probability, each model trained state, CSV/JSON source gate receipts, per-subject exact abstention & source-use, and 32 fixed paired TEST bootstrap AP contrasts (8/device, 399 resamples; 95% intervals conditional on test cohort and fixed neural fits, not device population).

Pre-registered AP contrast families for MCAR20:
A \`DIFF_AB−RAW_DIFF_LOGIT_TRAIN1000\` healthy;
B \`DIFF_AB−RAW_DIFF_HGB_TRAIN1000\` healthy;
C \`DIFF_AB−RAW_DIFF_HGB_ALL1800\` healthy;
D \`DIFF_AB−K2\` healthy;
E \`DIFF_AB−ON_AB\` common_U8;
F \`DIFF_AB−RAW_DIFF_HGB_TRAIN1000\` common_U8;
G \`DIFF_AB−ON_AB\` OFF_only_U4_drift;
H \`DIFF_AB−ON_AB\` OFF_source_leak_50pct.

If V40 fails to recover sources under nonlinear family, record refusal. Do not turn new TEST cases into tuning data. Any new stronger baseline after results must be a **new cohort/protocol**, not post-hoc cherry-picking.

## Negative scientific guarantees

- Classical SOBI source directions are identifiable only under additional linear mixing and source independence conditions. Nonlinear view g can invalidate exact directions.
- Physical OFF switch semantics, equality of U and sensor independence are external/experimental assumptions. No autonomous physical identity can be certified without them.
- Two sensor groups are not generally worse than three; extra stringency reduces usable coverage.
- A powerful classifier can predict better even if it cannot explain via a physical source. Report this as it is.
- No claiming superiority over original published CBM, interventional CRL, sensor-drift, or nonlinear ICA implementations unless faithfully benchmarked.
- No real patient data, no deployed clinical likelihood calibration, no causal effect claims, no master merge.

**Stop condition:** if a fatal source eigensystem failure occurs, run all strong raw baselines and record explicit abstention where possible, without modifying the frozen algorithm. PR Draft.
