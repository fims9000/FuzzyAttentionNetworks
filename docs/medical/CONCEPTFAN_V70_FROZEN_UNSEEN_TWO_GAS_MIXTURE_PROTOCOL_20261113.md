# V70 frozen independent mixed-source experiment — UCI308 acetone + ethanol

**Registered 2026-11-13 BEFORE retrieving any UCI308 actual sensor samples, reference rows, outcomes or fit results.** This is a different real physical dataset family from UCI309/322/270 and tests **two separately dosed chemical sources in mixtures**, unlike UCI270's six pure compounds. A small heldout physical test may inform engineering but cannot prove unique molecular concept semantics. The V65/V69 architecture families were selected on prior datasets; V70 is a new-data-family prospective challenge for this frozen hypothesis, not a registered new-chemistry study.

## Source and genuine physical independence

Official [UCI Gas sensor array under flow modulation (ID308)](https://archive.ics.uci.edu/dataset/308/gas+sensor+array+under+flow+modulation), DOI `10.24432/C5BG7G`, original **58 separate 5-minute physical experimental trials**, each with 16 gas sensors and ~25Hz measurements over ~300s. Two independently controlled physical analytes: acetone and ethanol, including 8 air-control trials and several single-source and binary-mixture conditions. **Five separately acquired session/day batches** (19, 10, 10, 11, 8 real trials). Controller concentration fields are experimental targets, not independent live chemical analyzer time trajectories; record their provenance accordingly. No patient data, diagnosis, or device-fault ground truth.

Raw `rawdata.csv.gz` is recorded as 16 successive physical sensor rows per trial, about 7,500 true time points per channel; `features.csv` provides one set of metadata and conventional engineered descriptors per **original trial**. These are two representations of the SAME 58 physical trials, not 58 + 928 independent trials. The parser must assert exact trial-to-metadata identity, physical sensor order and source fields. A source-schema/column parsing patch is permitted after inspecting filenames and the FIRST nonoutcome row *only*; do not use it to choose K, target, threshold, results, head or data partition.

## Chronological independent-trial split locked

Order five original batch/measurement periods chronologically as documented (not by gas class or labels):
- **TRAIN first three acquisition batches: 39 whole physical trials** (19+10+10).
- **VALID fourth batch: 11 whole trials**.
- **TEST fifth batch: 8 whole trials**.
No trial appears in two groups. No derived window/sensor measurement from a trial crosses group boundaries. The 16 sensor channels and ~7500 time samples ARE NOT independent physical experiments; regression evaluation nTEST=8 and nVALID=11, even if curves have millions of readings. If parsing cannot substantiate 5 genuinely ordered groups and per-trial metadata, ABSTAIN / fail rather than reassign to favor outcomes.

Use 60 equally sized contiguous **five-second** bins per trial, median of the actual sensor samples in each bin, no interpolation and no treating sensor index as time. If exact source sample count is not 7500, bin by true sample time (target nominal 25Hz) and report measured duration / missingness rather than arbitrarily reshaping 58×16×7500.

## Frozen unsupervised source architecture

- TRAIN-only robust median/IQR scale for each of the 16 measured sensors; no concentration/label is passed to source fit or membership calibration; clip [-8,8].
- Per original TRAIN trial form a 16-dimensional **innovation** vector `median(sensor, last 25% trial) - median(sensor, first 25% trial)`; fit TRAIN-only PCA, select smallest K explaining at least **95%** innovation variance, cap **4**; state K (may equal 1) is a statistical dimension, NEVER the true number of gas chemicals.
- Use the K normalized signed PCA loadings as anonymous directions. Do not fit an unstable high-dimensional ICA to only 39 independent trials; this is a smaller-sample falsifier of the **V65 dual absolute + delta** source principle.
- For every [real physical trial, 60 true time bins], form **absolute** source projection and **causal change from the first actually measured bin** for each of K source directions. Calibrate each stream with TRAIN Q25/Q50/Q75 and positive width `max(0.15, TRAIN-IQR/1.35)`. Generate 6K static fuzzy membership states and two time summaries per state (whole and LAST **15** bins) → 12K signed interpretable features, no neural recurrent training because only 39 original independent TRAIN trials. This is **static dual-stream fuzzy principle**, not full V56 native temporal network validation.
- Fit two separate **Ridge(alpha=1) signed-additive concept-only regression heads** on exactly the same 39 TRAIN concentration targets (acetone ppm and ethanol ppm). TRAIN-only standardized concept columns, transform intercept/coefficients back to original [0,1] coordinates, verify max error≤1e-9; no raw-sensor bypass, no supervised gradients into source directions.
- Two raw feature comparators use the SAME 39 TRAIN target values: raw16 sensor all-60-bin mean plus last15-bin mean (32 features) `StandardScaler + Ridge(alpha=1)`, and source **linear 2K** absolute+delta (full+late =4K) `StandardScaler + Ridge(alpha=1)`. Include TRAIN-mean trivial baseline and optionally raw HGB fixed 100 iterations max15 leaves (no model tuning). No hyperparameter tuning on VALID/TEST.

## Frozen outputs and rejection criteria

Separate real-trial TEST n8 and VALID n11 for acetone and ethanol; report MAE in the official controller concentrations, RMSE, R² and normalized MAE relative TRAIN target std and constant TRAIN mean; if TEST R² undefined due no target variance, explicitly report undefined. Every score represents **physical trials**, not 7500×16 pseudo-observations. Report all per-trial signal conditions in aggregation only and full source SHA, K, confirmed acquisition periods.

Postfit physical associations: for each previously frozen K axis and each gas, calculate VALID and TEST run-level partial Spearman of full/late fuzzy response versus independently controlled concentration, controlling the *other* gas level; low n may yield undefined. VALID-only assignment of at most two distinct unsigned axes to the two gas reference names and corresponding TEST no rematch is explicitly **label-informed external audit**, not autonomous semantic naming. If K<2, ABSTAIN physical two-source naming. Test a shuffled-reference negative control (one fixed permutation), but do not call a permutation p-value from nTEST8 proof of factor identification.

**Predeclared falsifiers:** original raw ridge may outperform concept-only; K may be less/more than 2 despite two physical sources; real trials may have insufficient independent signal/noise to fit stable fuzzy thresholds; independent reference concentration is a controller setting rather than chemically assayed per-bin truth; n8 TEST cannot support reliable deployment accuracy or formal semantic identification. Retain negative results.

Academic/public CC BY 4.0 dataset; credit creators Ziyatdinov and Fonollosa. No changes to accepted IOI/PIERE manuscripts or master. All work remains a DRAFT research branch.
