# V60 executed real UCI chemical-sensor evaluation — negative predictive result, partial observational association

**Executed 2026-11-13 on the official UCI dataset. No claims of unique pollutant concept identity.** Frozen before inspecting results: `CONCEPTFAN_V60_FROZEN_REAL_AIRQUALITY_PROTOCOL_20261113.md`. Successful [GitHub Actions run #43473805720](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43473805720), source commit `236d83a0dd44a0163df74449a9ec9d3cac38d09e`, artifact `conceptfan-v60-real-airquality-aggregate` (ID 14061414732), containing `V60_REAL_AGGREGATE_EVIDENCE/STATUS.json`, `METRICS.csv`, source SHA and a one-line log. The official UCI ZIP was downloaded from UCI, not a third-party transformation; its actual SHA is in STATUS.json. Artifact contains no raw sensor timelines or hourly timestamps.

**Source:** S. De Vito, [UCI Air Quality id 360, DOI 10.24432/C59K5F](https://archive.ics.uci.edu/dataset/360/air+quality). Recorded real field responses of 5 metal-oxide PT08 sensors and co-located certified reference concentrations. Sensor drift and cross-sensitivity are documented in dataset notes. This is one physical monitoring station, not a clinical dataset, and not instrument-fault ground truth.

## Exact method: independent analyzer observations excluded from concept formation

- Exactly five input channels: PT08.S1(CO), S2(NMHC), S3(NOx), S4(NO2), S5(O3), with -200 missing values masked and TRAIN-only median/IQR scaling. Certified CO(GT) and NOx(GT), T/RH, date and clock-time are not model input channels.
- **9357** official nonblank hourly records, split into **719** fully contiguous disjoint 13-hour blocks: twelve sensor input hours and the following 13th reference-analysis target hour. No overlapping cases, no imputation, no excluded/non-contiguous blocks. Chronological: TRAIN431 (2004-03-10–2004-10-29), VALID144 (2004-10-30–2005-01-15), TEST144 (2005-01-16–2005-04-03). No temporal TEST outcomes guided the model.
- TRAIN-only outcome thresholds for next-hour certified analyzers: CO>1.7 (CO GT original units), NOx>126.5 (NOx GT original units). They are *data-relative median exceedance labels*, not WHO or regulatory thresholds. Usable TRAIN labels CO323 and NOx324; held-out TEST labels CO139 and NOx140. The source model was formed on ALL 431 unlabeled TRAIN histories, so outcome/reference missingness cannot bias the vocabulary definition.
- The original group-proposal implementation had an unjustified 6-channel minimum inherited from synthetic testbeds. It was generalized to **two** (the actual mathematical minimum to form an association) while preserving 3-fold correlation threshold 0.60, minimum paired cells and all other safeguards. New 5-channel regression and all older measured-group tests executed in CI.
- V56 TRAIN-only `TrainOnlySourceBank` found 1 coherent five-channel correlated group, which did *not* meet the V55 requirement for at least 2 groups; the algorithm therefore routed to `PREDICTIVE_SUBSPACE`. Positive-lag signed observable rank **K=1**, with TRAIN subject bootstrap interval **[1,2]** (15/20 replicates K=1, 5/20 K=2). This is **not** a stable unique number of physical pollutants.
- Native V56 recurrent fuzzy source encoder trained for 8 unlabeled epochs, frozen before either binary outcome task; task head logistic readout from source concept only. Static V55 used exactly the same TRAIN-only source definitions; raw logistic, PCA+logistic and a HistGradientBoosting were separately trained on identical five-channel mask-aware time summaries. Predictor/source SHA remained unchanged when training CO versus NOx heads.

## Numerical results — chronological future TEST, all preregistered methods

AUPRC is prevalence-sensitive: the TEST CO positive fraction was **0.3885**, NOx **0.7929**. AUPRC around 0.87 for NOx represents only about +0.074 above the class-prevalence baseline and must not be described as excellent discrimination without that context.

| Future analyzer target | Model | N TEST | TEST AUPRC | TEST AUROC | TEST NLL | TEST Brier |
|---|---|---:|---:|---:|---:|---:|
| CO high (>1.7) | V55 static concepts | 139 | .6139 | .7272 | .6101 | .2101 |
| CO high (>1.7) | V56 learned temporal concepts | 139 | .6200 | .7379 | .6152 | .2120 |
| CO high (>1.7) | Raw logistic | 139 | **.6618** | **.7974** | .5845 | .1974 |
| CO high (>1.7) | Raw PCA + logistic | 139 | .6450 | .7566 | **.5716** | **.1927** |
| CO high (>1.7) | Raw HGB | 139 | .6144 | .7126 | .6489 | .2248 |
| NOx high (>126.5) | V55 static concepts | 140 | .8656 | .5834 | .7566 | .2801 |
| NOx high (>126.5) | V56 learned temporal concepts | 140 | .8666 | .5828 | .7408 | .2733 |
| NOx high (>126.5) | Raw logistic | 140 | **.9075** | **.7077** | **.5839** | **.2056** |
| NOx high (>126.5) | Raw PCA + logistic | 140 | .8903 | .6459 | .8468 | .3144 |
| NOx high (>126.5) | Raw HGB | 140 | .8955 | .6485 | .8578 | .2887 |

On chronological VALID, raw logistic also beat concept-only methods (CO AUPRC .7986 versus V56 .7141; NOx .9718 versus V56 .9431). Neither data split supports universal predictive superiority. In NOx TEST, raw logistic outperformed V56 by +0.0409 AUPRC and -0.1569 NLL. The heavy prevalence shift (TRAIN 0.50 -> VALID 0.882 -> TEST 0.793) matters for calibration and should be reported.

## Grounded *external semantic* audit (not a causality test)

No CO/NOx analyzer observation was used in source discovery or network training. On the real held-out TEST period, compute each episode's fuzzy concept mean and **same episode's observed certified-analyzer history mean**, excluding missing analyzer hours. This is a held-out, masked, continuous-value association diagnostic, not label-based source learning.

| Real certified analyzer | Single discovered concept Spearman rho | Best raw PT08 sensor abs(rho) | Permuted-analyzer negative control abs(rho) |
|---|---:|---:|---:|
| CO(GT), 143 paired episodes | **-0.8264** | 0.8711 | 0.0425 |
| NOx(GT), 144 paired episodes | **-0.8744** | 0.9403 | 0.0840 |

The associations are real and strong, but the **same one-dimensional concept is strongly correlated with BOTH certified pollutants**, and individual original channels correlate even more strongly. There is no unique identification as a CO source, a NOx source, or separate chemical species. These pollutants co-vary at one urban station; joint source signal does not identify isolated physical causes. The estimated K=1 is also not perfectly stable under TRAIN resampling ([1,2]).

V56 concept dynamics remain task-label-free and exact signed additive, but the real-data comparison suggests **information loss in the one-slot bottleneck**. This is a falsifiable engineering observation, not a proof that increasing K will help or that a one-slot physical source exists.

## Practical scientific next boundary

- First strengthen source interpretability, not model depth. Assess whether a measurable analytic concept can remain semantically stable across independent environments, e.g. weather/sensor drift periods and second physical sensing locations, while keeping concept and semantic reference correspondence frozen.
- A realistic future **source-identity** validation needs independent analyzer channels associated with distinguishable measured processes, source-specific controlled interventions and calibration records. UCI Air Quality provides certified concentration measurements, not labeled sensor fault events, so it cannot validate the V57–V58 fault-repair scenario.
- Preserve the negative predictive result. No post-TEST route threshold adjustment, architecture search, label rebalancing or K inflation may be presented as a new preregistered confirmation on this same TEST.
- Keep IOI, PIERE and clinical manuscripts unchanged unless peer-reviewed experimental framing is revised explicitly. This work is exploratory, a *real-world stress test with external physical measurement reference*, not demonstrated clinical-source discovery.
