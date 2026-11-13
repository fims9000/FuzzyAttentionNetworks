# V67 independent physical regime test — six pure gases, sensor drift, NO source-label concept fit

Registered **2026-11-13 before any V67 model outcomes or source file contents are read**. The architecture family was developed on UCI309 and UCI322, so this is a **new source dataset** but NOT a prospectively registered claim of universal causal identifiability. Preserve all negative results. Dataset is *Gas Sensor Array Drift at Different Concentrations*, UCI **ID270**, not UCI224; UCI224 lacks explicit gas concentrations. Official research-only licensing/attribution applies.

## Falsifiable scientific question

Can a **label-free statistical fuzzy source vocabulary** learned from earlier gas sensor batches retain predictive and **gas-specific physical-class association** in later apparatus batches with substantial real device drift? Or is it simply a lossy and unstable representation compared with original measured sensors? A new source is not an automatically named chemical species; class labels may be used only for separate supervised classification heads and VALID-only *postfit* audit matching.

## Data and immutable evaluation split

UCI270: 13,910 measured exposure instances, **six single-compound pure-gas classes**, 16 physical metal-oxide sensors each summarized by 8 **engineered steady-state/transient response features**, total 128 numeric columns, and measured controller concentration in ppmv. This is *tabular per-exposure sensor-response data*, NOT 128 chronological samples or a 16×8 true timestamped trajectory. Thus V56/V65 temporal recurrence MUST NOT be run on the eight feature indices as if they were time; instead test transfer of V65's **TRAIN-only anonymous source discovery + multiple fuzzy thresholds + exact additive concept-only classifier** using a modality-appropriate static operator. This scope must be explicit in the paper.

The UCI package has ten temporal batches (measurement periods). **TRAIN original batches 1–7**, **VALID batches 8–9**, **untouched final TEST batch10**, independently of class frequency. Do not use TEST to choose rank, class thresholds, scaling, bandwidth, model or VALID matching. All six classes must occur in all three groups or ABSTAIN. Do not randomly split individual rows. Batch-level comparisons are descriptive because VALID has only two batch periods and TEST only one; row-level bootstrap does not establish independent drift environments. Keep a frozen six-class label order: 1 ethanol, 2 ethylene, 3 ammonia, 4 acetaldehyde, 5 acetone, 6 toluene. Concentration ppm is **never a model input**; it is audit-only metadata. Missing UCI raw input features lead to explicit rejection/no imputation, never to synthetic missing =0.

## Unsupervised static source builder — fixed before outcome labels

1. Parse every source exposure's 128 physical engineered response features as 16×8. The first 2 engineered feature indices per sensor represent steady-state and normalized steady-state features, the remaining 6 are transient EMA descriptors **not actual timesteps**. Validate the schema against UCI documentation; if original layout differs, stop and revise schema publicly, never fabricate physical time.
2. Learn all 128 median/25–75% IQR scales on TRAIN batches1–7 alone, then clip standardized values to [-8,+8]. Form two 16-element measurable SENSOR VIEWS: **ABS** = mean of the first two standardized steady-state descriptors per physical sensor, **TRANSIENT** = mean of the remaining six standardized transient descriptors. This is an explicit simple pooling of engineered *features* in two units; label-free, no pretending sensor feature index is a chronology. Concatenate to 32 channels.
3. Fit whitened **FastICA** (random_state=20261967, max_iter=1000, tol1e−4, n_components=K) to TRAIN 32-dimension pooled input. K is smallest PCA component count that reaches **95%** TRAIN explained variance, capped at **12**; report uncapped rank. K is a *statistical information dimension*, never a gas count. FastICA nonconvergence is logged, not hidden.
4. Form the **same three positive-width fuzzy memberships per discovered ICA axis** from TRAIN transformed values at Q25/Q50/Q75 and width max(0.15, IQR/1.35), with no controller setpoints, gas labels or class info used in source fitting. Exactly 3K static, inspectable fuzzy states (no native recurrence because data are tabular). Fix the basis, quantiles and signs before any supervised labels enter the model. The source model never directly consumes test labels.

## Equal-label-budget heads and controls

Train on original TRAIN class labels only, evaluating VALID and TEST without any hyperparameter selection:
- 3K TRAIN-calibrated fuzzy states -> **LogisticRegression(C=1, max_iter=1500, multinomial class probabilities)**, TRAIN-only StandardScaler of state features. Provide exact per-class signed additive logit decomposition **in original fuzzy coordinates**, check <=1e−9, and no raw-to-risk shortcut.
- **Linear ICA-K** source coordinate -> identically configured multinomial logistic, to separate fuzzy compression effects.
- 32 pooled original engineered view channels -> logistic.
- **Raw all 128** engineered descriptors -> identical logistic and ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,random_state=20261967,n_jobs=2). ExtraTrees is a nonlinear strong physical-sensor comparator; score as such.
- Majority-class baseline trained TRAIN-only; report rather than hide.
- Metrics across six classes: accuracy, balanced accuracy, macro/weighted F1, multiclass log loss and macro OvR AUROC when evaluable. Full batch8, batch9 and batch10 breakdown + confusion matrix, class support and train/test prevalence. Do not infer high confidence from 3600 correlated same-batch measurements.

## Semantic / physical-label correspondence audit, held separately from classification

After source basis + fuzzy states frozen, compute each original anonymous ICA source mean for each gas on VALID batch8+9. Construct **VALID-only one-to-one maximum absolute Spearman/OvR AUROC mapping** from anonymous source directions to six observed gas identity labels; since K may be less than six, emit **ABSTAIN** for unmatched classes, not choose K=6 using labels. Apply the fixed alignment without TEST rematching to batch10 and report performance and sign. Also report all K×6 VALID and TEST association matrices, model-class probability quality, intra-gas concentration correlations, and a fixed-seed gas-label permutation negative control. This semantic mapping uses gas labels on VALID and therefore is *supervised posthoc naming*, NOT automatic gas identification. High correlation can reflect gas-specific concentration and sensor batch drift; it is not unique physical factor discovery. If a class is missing in VALID or TEST, ABSTAIN for the relevant comparison.

## Conditional negative controls / limits

- Audit model invariance to unseen TEST label and ppm changes: neither changes unsupervised source manifest, head (trained TRAIN only), or predictions. A hidden/absent feature is not treated as measured zero.
- To diagnose drift, report per-batch feature median shift relative TRAIN and anonymous concept membership shift with no gas outcomes as inputs; but drift association alone is not causal interpretation.
- A 128-feature tabular source does not validate V65 120s recurrent architecture nor auto-granular temporal concept formation.
- One gas is injected per exposure, so there is **no simultaneous multi-gas source separation** or independent intervention in a mixture, and chemical source identity cannot be established simply by a fitted class classifier.
- The three evaluation groups use distinct historical measurement periods in ONE installation; they do not represent three independent labs. The model and its audit may fail drastically in batch10, and such a result must be retained.
- License research-only; no distribution of original records in artifacts. Keep aggregate metrics, source SHA256, source config, gas label audit, code version and complete errors. Never update accepted PIERE/IOI papers.

## Publication gate

A passing TEST with stable source-to-gas mapping on a new temporal batch supports **transfer of anonymous source statistics across sensor drift**, not semantic certainty or causality. If concept-only macro F1 or calibration falls behind full raw ExtraTrees, explicitly report that our interpretable bottleneck is incomplete. Avoid more dataset selection after looking at TEST; future confirmatory study requires truly new physical apparatus and controlled multigas interventions.
