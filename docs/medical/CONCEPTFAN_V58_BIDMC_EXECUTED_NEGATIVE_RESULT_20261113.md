# ConceptFAN V58 — genuine BIDMC measurements, patient-isolated, independently audited negative transfer result

**13 November 2026 · EXECUTED REAL-MEASUREMENT EXPLORATORY**. PR [#63](https://github.com/fims9000/FuzzyAttentionNetworks/pull/63), frozen protocol `CONCEPTFAN_V58_BIDMC_REAL_MEASUREMENTS_FROZEN_PROTOCOL_20261113.md`, successful [GitHub Actions run 43477010287](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43477010287), source commit `c67047a967be2b92873a4d2c609a8bca4ff26de1`. Exact official dataset: PhysioNet BIDMC PPG and Respiration v1.0.0, doi:10.13026/C2208R, ODC-BY-1.0. GitHub Actions artifact `conceptfan-v58-real-bidmc-patient-isolated-metrics`, ID `14062623462`, digest `sha256:4b26e486313f806b370988ba9152145836b996e32d1914f19ee974f67d9fe605`. The 53 original Numerics/Fix files were authenticated against the **official PhysioNet SHA256SUMS**, then only aggregate metrics and public BIDMC record ordinal split IDs were uploaded (no raw patient identifiers, raw vital signs or individual risk predictions).

## Real holdout, not a synthetic copy
- **53 real recordings from 46 identifiable patient groups**. Patient-group split: TRAIN **27 groups/1770 overlapping 16-second windows**, VALID1 **6/531**, VALID2 **6/354**, TEST **7/472**. The same patient's recordings and overlapping windows never cross splits.
- Measured source X: PPG monitor PULSE/SpO2 and impedance RESP only, with causal first differences and trailing variability, mask-preserved natural missingness, TRAIN-only normalization. **ECG-derived HR was excluded from inputs, source discovery and normalization**.
- Supervision is a **simultaneous** ECG monitor HR ≥100 bpm threshold, and direct numeric comparison to ECG HR. It is NOT a prospective early warning system. **The ECG reference doubles as the Y label**; this test cannot be called independent corroboration of discovered physiological concepts.
- Positive-window prevalence TRAIN **20.11%**, VALID1 **22.22%**, VALID2 **17.80%**, but TEST only **2.33%**. Such a large prospective patient-split prevalence shift makes TEST AUPRC highly unstable; report patient clustering, do not pool repeated windows as independent patients.

## Registered V54 → V55 source path: **zero accepted candidates**
Under the original V54 `SIGNED_LAG` and `EVEN_ENERGY_LAG` family with exact same stringent 3 TRAIN folds / min |r| .65 / min .30 gain over contemporaneous association, *zero real measured candidates were proposed* from 1770 TRAIN windows. Both natively trained K2 seeds correctly chose **NO_SOURCE** and exhibited **bitwise exact K2 fallback** on all TEST cases; source use fraction **0%**, with no risk bypass or phantom novel concept.

This is substantive negative evidence: the lag-dominance criterion created for the synthetic generator did NOT transfer to slowly varying real monitored physiology. It cannot establish that more general concept discovery is impossible, nor that alternate (properly registered) operators would fail.

### Natural missingness: real TEST, mean of two native K2 seeds (7 patient groups, 472 windows)

| Method | AUPRC ↑ | NLL ↓ |
|---|---:|---:|
| Native FAN K2 | **.0728** | **.5557** |
| Auto FAN V58 | **.0728** | **.5557** (exact fallback) |
| Logistic with original features | **.2379** | **.0939** |
| Logistic with all engineered source features | **.2099** | **.0959** |
| Logistic with only PPG PULSE | **.2297** | **.0811** |

Direct PPG monitor PULSE differs from ECG monitor HR by **2.27 bpm MAE** across 460 available TEST windows. This does not provide clinical accuracy evidence; two algorithms and temporal windows may be correlated. The baseline PPG pulse score itself has AUPRC **.2297** on this heldout split.

Patient-group bootstrap (399 whole-patient draws, descriptive, valid two-class resamples 269–271):
- Seed 11, auto FAN−energy logistic ΔAUPRC = **−.1555**, 95% interval `[−.1935,−.0327]`.
- Seed 23, auto FAN−energy logistic ΔAUPRC = **−.1186**, 95% interval `[−.1374,−.0223]`.
- Auto FAN−K2 exactly **0** across every subject, both seeds; no artificial gain.
Only **7 independent patient groups** in the TEST; overlapping windows are not independent draws, bootstrap intervals are neither multiplicity-adjusted nor stable validation of clinical use.

### Predeclared challenges
- MCAR20: native K2 AUPRC ~.0881, NLL ~.5513; logistic energy AUPRC varies .1467/.1808 by generated mask seed; dropouts are generated independently at the window level, not a realistic device failure model.
- PULSE completely missing: native K2 AUPRC .0384 and NLL .5784; logistic approaches chance with the very low TEST prevalence; direct pulse MAE is unavailable.
- PULSE +25 bpm (input-only): native K2 AUPRC ~.0319, NLL ~.5929; logistic raw/engineered NLL **7.36/7.09** respectively, indicating extreme calibration failure under instrument bias. For direct PULSE-vs-ECG comparison, observed MAE jumps **2.27→23.59 bpm**.
- These stress results are descriptive, not certified drift detectors; no TEST-based adaptation was performed.

## Engineering validation
- 53 authentic files from official checksums downloaded into ephemeral runner cache; private patient IDs used only to isolate groups.
- Focused regression and independent real-case audit passed. The auditor recomputed **8 private TEST prediction arrays** (2 native seeds×4 stress conditions), AUPRC, NLL, patient-macro NLL, signed native decomposition (or exact K2 in absence of source), source coverage and SHA256. Private arrays and individual risk scores were excluded from uploaded artifact.
- Full result manifests and source SHA are retained with GitHub Actions; source download script and patient split make the audit reproducible from official public data.

## Conclusions / next falsifiable architecture
**Current outcome:** real input transfer *fails* to demonstrate automatic new concepts or competitive K2 predictive performance. Stop assuming V54's lag-dominance grammar is general. The correct next capability is an outcome-blind measurement program grammar allowing **same-time and periodic/within-window sensor relationships** with explicit family lineage, subject-group stability and shuffled-pair controls, and to validate it on a new independent cohort, not silently relax thresholds on the TEST just inspected.

A robust result also needs separate clinical assay target and risk Y, or an independent experiment with device metadata / measured physical stimulus; no such semantics are present in BIDMC numerics. A new candidate that is predictive under both VALID cohorts does **not** become physiologically meaningful because of correlation.
