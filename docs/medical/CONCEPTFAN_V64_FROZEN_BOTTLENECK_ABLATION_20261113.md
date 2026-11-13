# V64 frozen post-V63 diagnostic: where does concept information disappear?

Registered on 2026-11-13 **AFTER viewing the V63 UCI322 TEST scores**. This is an **exploratory mechanism ablation on the same previously exposed TEST**, not new evidence of physical source identity, independent confirmation, or a new architecture with an outcome-guided K.

## Preceding falsifier

On new real UCI322 dynamic mixtures V63, the automatically fitted unlabeled ICA/PCA innovation rank was K3. Continuous concentration inference through the V56 fuzzy-native and static concept-only heads was largely ineffective (R² about -0.43 to +0.04); raw 16-sensor ridge scored +0.535 to +0.696. The core question is WHERE the information is lost: subtracting initial sensor levels, reducing to three innovation axes, bounding states through sigmoid, or temporal pooling.

## Frozen exact ablation — no hyperparameter tuning

Same official source archive, source SHA, 120s windows, 120s temporal embargo, TRAIN414/VALID134/TEST136, two original continuous recording sessions, identical TRAIN-only 16-channel median/IQR and V63 ICA/PCA loading matrices and source K=3 determined from the TRAIN 95% innovation variance criterion (NOT gas labels).

For each source basis PCA and ICA, compare the following Ridge(alpha=1) heads fitted only from TRAIN setpoints, separately per gas and recording:
1. `raw_absolute_32`: 16 full-window means + 16 last-30s means of original normalized sensors.
2. `raw_initial_differenced_32`: the identical summaries after subtracting EACH window's first observed second (a strict information-loss control).
3. `absolute_pca_K`: TRAIN-only PCA (with K fixed to the V63 innovation-selected K) on 32 absolute-sensor features; supervised Ridge in K-dimensional PCA feature space.
4. `projected_absolute_linear_K`: V63 frozen ICA/PCA loadings projected on the **absolute** standardized sensor values; concat time-mean and last30 mean of K source axes (2K features) for supervised Ridge. This removes the baseline subtraction, but does not relearn any source basis.
5. `projected_delta_linear_K`: same 2K summaries of the original V63 bank's baseline-differenced linear source levels.
6. `projected_delta_fuzzy_all_K`: V63 original static fuzzy concept mean over all 120 seconds (K features).
7. `projected_delta_fuzzy_late_K`: V63 static fuzzy concept last30 mean (K features).

All operations use original TRAIN-only sensor and source geometry; source K, ICA/PCA directions, fuzzy thresholds, comparison labels, and Ridge regularization remain unchanged. No controller gas setpoint enters unsupervised source axes. Raw features and derived masked features are computed only from genuinely observed sensors; no time interpolation.

Report VALID and **previously inspected TEST** R², MAE ppm, rank correlation and negative R². No new model gets to select a winning representation using TEST. A more accurate new representation is a *posthoc diagnostic, not a fresh confirmed improvement*. Record the full eight-by-four matrix even if all variants fail.

## Diagnostic interpretations to falsify

- If `raw_initial_differenced_32` collapses similarly to V63, initial baseline subtraction is itself a major information-destruction mechanism.
- If `raw_initial_differenced_32` remains strong while `projected_delta_linear_K` collapses, reduced ICA/PCA innovation subspace likely loses task-relevant dimensions.
- If `projected_delta_linear_K` is strong but `fuzzy_all_K` fails, sigmoid saturation or episode-mean pooling is implicated.
- If `fuzzy_late_K` beats `fuzzy_all_K`, full-window averaging discards temporally local current gas information.

None proves chemical identity, independent calibration or transfer across physical sensors. The uncontrolled within-session and time drift effects remain, and only two real recording sessions were collected.

The actual research objective is a falsifiable diagnosis of the current architecture's failure, not an invented victory against raw predictors. Original accepted manuscripts and main branch unchanged.
