# V61 executed on real independently controlled gas plumes: physical-source concept failure

**Execution:** 2026-11-13, exploratory NEW real physical source family, successful [GitHub Actions run 43475225666](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43475225666), commit `0c8c2a6d6c475e2b62256a5a4594d508a4f41c56`, artifact `conceptfan-v61-real-gas-controlled-interventions`, ID `14061499371`. Complete original immutable results: `V61_REAL_GAS_EVIDENCE/STATUS.json` and `METRICS.csv`. Original protocol `CONCEPTFAN_V61_FROZEN_REAL_GAS_INTERVENTION_PROTOCOL_20261113.md` was committed BEFORE viewing V61 TEST scores.

## Physical truth source

Jordi Fonollosa (2014), [UCI Gas sensor array exposed to turbulent gas mixtures (id 309)](https://archive.ics.uci.edu/dataset/309/gas+sensor+array+exposed+to+turbulent+gas+mixtures), DOI 10.24432/C5JS5P, eight physical chemoresistive sensors, two independently flow-controlled gas sources. Each of 30 ethylene + CO or ethylene + methane flow/mixture configurations is repeated six times (180 recorded experiments total). The separate GC-MS instrument validated **mean concentration levels**, not 100-ms ground-truth concentration trajectories. True experiment phases: 0–60s clean air, 60–240s exposure, 240–300s recovery. The published raw and 100-ms downsampled files coexist; ONLY the 180 downsampled recordings were used, as one physical run each.

Time summaries: 60 non-overlapping five-second bins over 300s, 8 raw sensors, no temperature/humidity/physical label supplied as model features. Stratified 4 TRAIN +1 VALID +1 TEST replicate per gas configuration, seed20261701: **120/30/30 whole physical recordings**. All final classifier metrics are at the *whole-run* level, NEVER at dependent temporal-sample level. No separate testbed/hardware.

Independent gas controls, used only AFTER concept-source fitting and for posthoc audit: Ethylene present vs not (TEST 24/30), CO present vs not (TEST 12/30), methane present vs not (TEST 12/30). All sources/labels derived from experiment filenames and published GC-MS mean levels, **never** passed to unlabeled source rank or native temporal training.

## Automatic source representation: legitimate whole-run unit

V56's original 300-"patient" rule presupposes independent synthetic 16-step episodes. Here there are only 120 independent TRAIN physical runs. Splitting a recording into 3 correlated segments and claiming them to be independent source-discovery patients would be invalid, so V61 used the **existing positive-lag train-only observable spectral rank** on exactly the 120 complete runs (permutation by whole run; lags1/2/3, 60 permutation draws, 20 whole-run bootstraps). The resulting signed spectral source definition and fuzzy thresholds were frozen.

- Detected observable rank **K=1**, whole-run TRAIN bootstrap interval **[1,1]**. Singular values approximately [5.0064, 0.8410, 0.4700, 0.0551, ...] compared against a common familywise null threshold **2.8566**, so only the first direction was selected. This is NOT a proof that the experiment had only one physical gas source.
- After source fitting, three **dependent** non-overlapping temporal crops of each TRAIN recording were used as **360 source-only augmentation segments** to train the existing V56 diagonal recurrent fuzzy concept encoder (8 epochs, neural seed11, denoising/forecast source losses, NO gas labels). They are explicitly NOT independent calibration/evaluation samples.
- Task heads fit on 120 original whole-run TRAIN labels, not augmented segments, and their inputs were only the exact time-averaged signed concept (V55 static and V56 native). Risk logit decomposition remained exact. Same-label raw logistic/PCA-logistic baseline uses masked original 8 sensor means/recency; separately report exposure-aware raw logistic and HistGradientBoosting, which have access to known 60–240 s experimental exposure timing but NO gas label as feature.

## Real physical-run TEST: all frozen comparisons, AUPRC / NLL

The TEST sample is **only 30 runs**, including six negative ethylene runs and 12 positive each for CO and methane. Do not interpret the estimates as external population precision or cross-site performance.

| Controlled physical exposure (TEST prevalence) | Method | AUPRC | AUROC | NLL |
|---|---|---:|---:|---:|
| Ethylene present (80%) | V55 static K=1 | .9613 | .8403 | .4591 |
| Ethylene present (80%) | V56 native K=1 | .9682 | .8611 | .4534 |
| Ethylene present (80%) | Raw logistic | .9928 | .9653 | .1955 |
| Ethylene present (80%) | Raw PCA1 logistic | .9649 | .8472 | .4372 |
| Ethylene present (80%) | Raw exposure-aware logistic | .9954 | .9792 | .1277 |
| Ethylene present (80%) | Raw exposure HGB | **.9983** | .9931 | **.0790** |
| CO present (40%) | V55 static K=1 | .7410 | .7824 | .6267 |
| CO present (40%) | V56 native K=1 | .7296 | .7639 | .6430 |
| CO present (40%) | Raw logistic | .9936 | .9954 | .1621 |
| CO present (40%) | Raw PCA1 logistic | .7107 | .7593 | .5477 |
| CO present (40%) | Raw exposure-aware logistic | **1.0000** | **1.0000** | **.0880** |
| CO present (40%) | Raw exposure HGB | .9936 | .9954 | .1025 |
| Methane present (40%) | V55 static K=1 | .6876 | .7222 | .6370 |
| Methane present (40%) | V56 native K=1 | .6816 | .7130 | .6500 |
| Methane present (40%) | Raw logistic | .9707 | .9815 | .2286 |
| Methane present (40%) | Raw PCA1 logistic | .6913 | .7269 | .5779 |
| Methane present (40%) | Raw exposure-aware logistic | **.9881** | **.9907** | **.1589** |
| Methane present (40%) | Raw exposure HGB | .9480 | .9630 | .2296 |

**Real negative result:** For distinguishing two *separately controlled* gas plumes in a wind tunnel, V56's automatically discovered **one-slot** temporal fuzzy concept is substantially less task-sufficient than raw eight-sensor features. For CO, AUPRC is .7296 versus .9936 for raw logistic; for methane .6816 vs .9707. This cannot be repaired scientifically by only claiming the learned operator is interpretable or exact-additive. A signed exact explanation can exactly decompose an **incomplete** predictor.

For the very common ethylene-present class, AP .968 sounds excellent, but the class prevalence baseline is .800 and stronger raw models reached .993–.998. No superiority claim.

## Held-out semantic audit of physical intervention identity

The experiment has **run-level flow settings / GC-MS-validated mean levels**, not a time-resolved independent gas concentration per sensor sample. Thus we compared each frozen concept's **exposure (60–240s) minus clean-air (0–60s) mean activation** against the controlled gas concentration level across TEST runs. This is a predeclared run-level association audit only. No gas levels, code names or reference data were used in source fitting.

| Gas control across TEST runs | max abs Spearman across V56 discovered concept | max abs raw sensor | shuffled-condition negative control |
|---|---:|---:|---:|
| Ethylene | .6769 | **.8153** | .4079 |
| CO | .3834 | **.8360** | .1458 |
| Methane | .2321 | **.5863** | .1372 |

The single concept largely captures a common exposure effect, not a distinct CO or methane identity. The one-shuffle ethylene negative-control rho .408 is relatively high for 30 runs, so report associations descriptively; no statistical semantic-identity certificate. Sensing turbulence and flow co-variation complicate source separation, and even high coefficients alone would not establish a uniquely interpretable physiological or physical concept.

## Design limits and next falsifiable architecture question

- The controlled independent gas release establishes real **source interventions** absent in V60, but it does not establish independently documented sensor fault events. V57–V58's hardware malfunction recovery remains unvalidated.
- V61's stronger evidence is that a conservative **positive-lag rank K=1** can be much smaller than the number of independently controlled physical inputs under one shared turbulent observation apparatus, and its concept-only bottleneck loses discriminative information. The mathematical rank procedure is not itself a gas-component counter: failure of physical interpretation does not invalidate its narrower observable predictive-rank definition.
- Learning one richer recurrence and fuzzy threshold inside K=1 did NOT solve this. Before pursuing a new paper, a new **train-only, unsupervised and intervention-agnostic** method of creating distinct concept operators is required (e.g. temporal change innovation subspaces, independent-source separation or sensor-specific contrastive operators), followed by *fresh physical-run* evaluation beyond this already inspected TEST. Simply forcing K=2/3 from observed labels would be post-hoc oracle leakage, not autonomous concept discovery.
- Use the disclosed 30-run TEST only for this frozen study; any subsequent model revised after viewing results must be labelled exploratory unless independently replicated. Keep original PIERE/IOI manuscripts unmodified.
