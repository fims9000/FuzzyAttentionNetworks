# V70 executed: independent real acetone/ethanol mixture trial challenge

**Date 13 Nov 2026 — successful first new dataset-family execution.** Frozen protocol `CONCEPTFAN_V70_FROZEN_UNSEEN_TWO_GAS_MIXTURE_PROTOCOL_20261113.md` committed BEFORE reading official UCI308 reference responses or evaluating results. This family had not been used in ConceptFAN V53–V69. Source-code grammar and static dual-stream design are derived from older research, so this is a new physical dataset test, not a universal chemical-identifiability theorem.

**[GitHub Actions SUCCESS run #43490623625](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43490623625)**, executed source SHA `a6a50fb8b29fd14f5e013acf66a57c4d677e00a7`, **14 tests passed**. Raw-data source ZIP SHA256 `7b062960dbef1a9e8aefc62bc7dbac09bdadb8cbb40793cf2b8e122da1862f90`. Artifact `conceptfan-v70-unseen-real-two-analyte-mixed-trial-evidence` ID `14068873862`, complete `V70_INDEPENDENT_REAL_TWO_GAS_EVIDENCE/STATUS.json` and `METRICS.csv`, generated script source commit. Original data license UCI308 CC BY4.0.

## Real source, physical units and independent sample identity

Original [UCI308 Gas Sensor Array Under Flow Modulation](https://archive.ics.uci.edu/dataset/308/gas+sensor+array+under+flow+modulation), DOI `10.24432/C5BG7G`: **58 independently prepared five-minute physical experimental trials** with actual acetone/ethanol pure-gas and binary-mixture exposures and air controls. Exactly **16 physical sensor rows per trial, each with 7,500 genuine raw time samples at 25Hz**. Original archive has 928 raw rows but only 58 independent physical trials, plus 58 rows of separately prepared feature metadata — these are *not additional trials*. The original raw data were reduced to 60 measured five-second bins per sensor by median of 125 original samples; no invented sensors, interpolation or sensor/time index mixing.

Each original trial's experimental reference fields `ace_conc`, `eth_conc` encode **0, 0.1, 0.3, or 1 nominal vol.% dilution of analyte in WATER** to prepare the experimental mixture, as stated in Ziyatdinov et al., *Data in Brief*, DOI 10.1016/j.dib.2015.02.016. They do **not** represent measured gas-phase ppm, independent chemical analyzer time traces or exact actual mixture concentrations in ambient air. We regress the nominal water-dilution preparation-dose levels, NOT gas ppm. These physical preparation labels never entered automatic concept definition.

Frozen chronology by whole physical trial:
- TRAIN first three acquisition sessions 19+10+10 = **39** genuine trials;
- VALID day3 morning **11** genuine trials;
- untouched TEST day4 afternoon **8** genuine trials.
No temporal window or sensor from one trial occurs in another partition.

## Source and fuzzy concept mechanism

Each original 39 TRAIN whole-trial physical sensor sequence was normalized by TRAIN-only robust median/IQR (16 actual sensor channels, clipped [-8,8]). For each original TRAIN trial we computed unsupervised late15-bin-minus-early15-bin sensor change and fitted PCA95 cap4 (NO gas labels or reference doses). Algorithm selected **K=2 anonymous signed statistical directions**, explaining **95.8886%** of TRAIN late/early innovation variance (eigenratios .65468 and .30420). Although two separately controlled gas preparation fields also exist, **numerical equality K=2 is NOT evidence of chemical identity**.

For each anonymous axis, define separately absolute physical sensor state and causal first-bin-relative delta; three TRAIN Q25/Q50/Q75 soft memberships each → **12 bounded fuzzy states**, two whole-/late15-time summaries each → 24 inspectable signed concept risk terms. The source basis, thresholds and widths were frozen before supervised TRAIN Ridge(alpha=1) dose heads. Exact additive reconstruction maximum error acetone **1.1×10⁻¹⁶**, ethanol **1.9×10⁻¹⁶**, no original raw sensor bypass into concept-only head. This is a **static dual-stream** ablation of the V65 principle, NOT training or evaluation of the V56/V65 native recurrent network, which would be statistically inappropriate on 39 original trials.

## Actual real independent held-out TEST: regression by *water-dilution vol.% preparation dose*

| Target preparation dose | Exact concept-only fuzzy R² / MAE | Anonymous dual-stream linear R² / MAE | Raw16 measured sensor Ridge R² / MAE |
|---|---|---|---|
| Acetone (TEST n8) | **.5324 / .2085** | .6034 / .1886 | **.8687 / .0951** |
| Ethanol (TEST n8) | **.4763 / .2000** | .5580 / .1882 | **.5946 / .2367** |

**Negative predictive-sufficiency result:** An independent new real mixture trial family supports usable signal in the anonymous fuzzy features, but the raw16-sensor Ridge is better in R² **for both targets**, and the equal-K linear dual-stream source is better than its fuzzy version **for both targets**. Do not claim automatic concepts surpass raw physical predictors. Notably, MAE ranks differ: for ethanol on this small n8, concept-only .2000 is lower than raw Ridge .2367 despite raw having better R² (variance-sensitive). Report both; not universal superiority.

VALID n11 R² acetone fuzzy .794, linear .834, raw Ridge .976; ethanol fuzzy .578, linear .700, raw Ridge .898. Thus the same negative pattern occurred before looking at the final TEST. True trial-level n11/n8 scores have high sampling uncertainty; no useful facility-transfer confidence interval or broad universality claim.

An extra raw HistGradientBoostingRegressor using sklearn defaults with only 39 TRAIN trials was degenerate: its default `min_samples_leaf=20` prevents ordinary tree splitting with 39 samples. Its output equaled the constant TRAIN-mean dose baseline for both targets (TEST R² acetone -.253, ethanol -.094). This is a **noninformative nonlinear comparator**, NOT a robust second raw baseline; retain and explicitly flag the failure. A separate small-sample nonlinear comparator would require a declared posthoc exploratory extension and does not alter this original frozen study.

## Independent physical-source identity: high association still NOT selective

After source fitting, known preparation-dose references were used only in a **VALID-informed postfit identity audit**; they did not enter source construction. VALID11 selected a one-to-one assignment: **acetone→anonymous axis1**, **ethanol→anonymous axis0**. Frozen mapping was applied to the new TEST8 without rematching. TEST partial Spearman (rank-residualized on OTHER analyte preparation dose):
- acetone on assigned axis1: **+.7916**; BUT *ethanol on same axis1* also **+.8213**, which is larger;
- ethanol on assigned axis0: **+.8790**; BUT *acetone on same axis0* **+.8723**, almost equally large.
- VALID cross-reactivity was also substantial: axis0 acetone +.8705 versus ethanol +.8073; axis1 acetone +.8426 versus ethanol +.6880.

Thus **both automatically found statistical directions respond to both independent preparation interventions**, and matching two directions to the two chemicals is **not specific**. Even strong fixed TEST correlations .79/.88 do NOT constitute independent gas-selective factors, unique physical concept identity or a trusted chemical ontology. This is more informative than merely claiming K=number of gas sources. Shuffled-dose single negative control gave weak correlations but n8 is much too small for a source-specific causal certificate.

## Publication-safe conclusion and next criterion

- **Verified:** a fully run-level, label-free source-to-two-stream-fuzzy-to-exact-additive-head pipeline actually executes on new physical gas mixture recordings; no pseudo-replication from 928 sensor rows / 7,500 time samples; K2 selected without chemical labels; strong raw Ridge controls included; fixed postfit mapping applied to unused session.
- **Not verified:** source selectivity, uniquely identified chemical components, calibrated airborne gas concentration, recurrent temporal FAN, independent second apparatus, or predictive superiority over raw models. Small original physical test n8 limits any generalization.
- A scientifically meaningful physical concept must show **specific response to one controlled analyte while the other varies**, low cross-talk, stability across new physical runs and an independent assay/reference. Existing observational source axes do not meet that bar. A future prospective source experiment with independently calibrated gas-phase concentrations is required to certify physical meaning.

Research-only draft; original accepted IOI/PIERE manuscripts and master unchanged.
