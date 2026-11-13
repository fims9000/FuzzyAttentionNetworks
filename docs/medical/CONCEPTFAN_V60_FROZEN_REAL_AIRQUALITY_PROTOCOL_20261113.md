# V60 preregistered real chemical-sensor/reference-analyzer experiment

Registered 2026-11-13 **before reading any V60 chronological VALID/TEST numerical outcomes**. This is exploratory relative to the V53–V59 research program, not confirmatory work or a claim of physical-source identification.

## External real-world dataset and semantic controls

UCI Air Quality (id=360), DOI **10.24432/C59K5F**, donor S. De Vito; official source https://archive.ics.uci.edu/dataset/360/air+quality . About 9358 hourly observations from 2004–2005 in one Italian city, five low-cost metal-oxide PT08 sensor responses plus CO and NOx concentrations from a co-located **certified reference analyzer**. Known source cross-sensitivity, drift and missingness tag -200.

- Sensor inputs: ONLY `PT08.S1(CO)`, `PT08.S2(NMHC)`, `PT08.S3(NOx)`, `PT08.S4(NO2)`, `PT08.S5(O3)`. Do not use analyzer values, temperature, humidity, dates, clock time, future values, local trends based on future examples, or hidden target measurements in concept fitting.
- External reference channels: `CO(GT)` and `NOx(GT)` for independent **concurrent source-association audit**; the same certified-analyzer channels provide **future next-hour target labels** for supervised, post-concept prediction. The use of analyzer labels in the downstream task head does NOT establish semantic concept identification.
- Missing numeric -200 (and NaN) are absent. Select and scale channels with TRAIN-only medians/IQR (no test/validation normalization), keep binary observed masks. Never treat missing values as real negative signals.
- Parse official CSV decimal comma and semicolon delimiters. Exclude empty rows and rows without valid timestamps. Require chronological monotonicity; detect missing/duplicate hours; do not interpolate.
- Non-overlapping **13-hour candidate blocks**: 12 hours of sensor history and the 13th, next hour certified-analyzer reference for target. Require all 13 timestamps be hourly contiguous. No window may overlap another, including across temporal split boundaries. This is forecasting *next-hour certified concentration exceedance*, not same-hour reference reconstruction.
- Split the sequence of non-overlapping valid blocks **chronologically** as first 60% TRAIN, next 20% VALID, last 20% TEST, with no test-driven fold changes. Time-series autocorrelation and sensor drift remain; blocks are not independent hospitals or sites.
- TRAIN-only target definition for each pollutant: **median of valid future reference concentration in TRAIN blocks**. High equals `next_hour_reference > train_median`; no physical regulatory exceedance claim. Invalid future analyzer rows are excluded from corresponding supervised metric but may remain in unsupervised source fitting. Report target coverage and class prevalence by split. The signed head for CO and NOx gets the same TRAIN label budget as raw comparators.

## Architecture and baselines (all hyperparameters frozen)

1. V56 `TrainOnlySourceBank` uses all five normalized TRAIN MOX channels, no manual mapping, no analyzer labels; if at least two measurement groups cover >=50% of channels it selects robust groups, else preexisting positive-lag observable spectral rank (lags1/2/3, 60 permutations, 20 bootstraps). If no source rank, return bias-only; do not invent slots.
2. V56 native per-source differentiable causal fuzzy concept encoder, **one unsupervised TRAIN-only fit** (seed=11, 8 epochs, 20% random observed-channel dropout, denoise+forecast). No CO/NOx labels/GT in source or fuzzy training; freeze network hash.
3. V55 static automatic measured/spectral concept bank is independently fit on the same unlabeled TRAIN inputs with the same source selector seed. V55 and V56 each fit a simple logistic binary risk head using TRAIN-only target outcomes and only their own concept summaries, then evaluate unchanged on chronological VALID and TEST. Their risk logit has exact bias+signed concept contributions, no raw or reference bypass.
4. Raw feature baselines on the **same** five original MOX channels and missingness mask: standardized full/last-6 observed means + channel mask fraction; LogisticRegression(C=1), PCA(max(1,min(V56_K,15))) + logistic; HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,l2_regularization=1). Standard hyperparameters for all, no tuning against TEST.
5. For semantic source validation, on TEST compute **episode-level masked actual analyzer history mean** and separately the V56/V55 mean of fuzzy concept c_k(t). Report the complete K-by-2 Spearman association matrix and max abs correlation with CO and NOx per method, with exact non-missing counts. Also report raw sensor/analyzer Spearman associations for fair reference. Pick no single k for re-fitting on TEST, and do not give physical names to automatically discovered factors.
6. Negative control: shuffle TEST analyzer *history* labels using a frozen RNG while leaving all model predictions unchanged. Report max absolute Spearman of shuffled reference against the **same frozen** concepts. No training or threshold selection based on this control.
7. Report all results on VALID and TEST, no optimistic hyperparameter selection. Show any negative result. Aggregate-only artifacts; do not package the raw copyrighted data or individual timestamps.

## Falsifiers and limits

- If strict sensor-source grouping cannot find credible groups, a spectral observable subspace is a **basis**, not uniquely identifiable pollutant sources. Correct predictive rank or strong correlation does not prove specific CO/NOx concept identity due to correlated pollutants and sensor cross-sensitivity.
- If V56 predicts next-hour analyzer high concentration worse than raw logistic/gradient boosting, report that directly. Train losses, exact additive head and a calibrated fuzzy membership do not certify external validity.
- The reference is an independently collected, certified analyzer, but **fault labels, separately documented source interventions and calibration failure logs are not available**. V57–V58 reference-fault-repair claims cannot be tested directly on UCI Air Quality.
- The dataset comes from one station in one city over one year; chronological TEST is a drift/transfer challenge **within site**, not independent geographic or hardware validation.
- Use a fixed input representation; abstain on too few valid target records. Never synthesize outcome, copy reference as a model sensor, or tune labels/threshold to obtain a desired win.
- No accepted IOI, PIERE or clinical paper may be modified on the basis of an exploratory V60 run without independent scientific review.

## Publication boundary

The tested methods (correlation clique / lagged SVD, logistic CBM, neural causal scalar recurrences, reference-analyzer sensor calibration) are established pieces. V60's prospective contribution is a **real-world falsification and provenance audit** of the integrated automatic ConceptFAN program, not discovery of novel atmospheric source physics.
