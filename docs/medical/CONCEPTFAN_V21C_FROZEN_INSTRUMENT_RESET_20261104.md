# V21c — frozen paired acquisition / instrument-reset falsification (2026-11-04)

## Why this control is necessary
V21 showed that a predictive sensor artifact can pass a risk-association screen; a common-mode artifact can even pass a separately identified device's measured-reference corroboration. V21b showed that independently TRAIN-aligned risk initialization can make the artifact's signed **predictive contribution** nearly perfectly reproducible across neural runs. Therefore *neither AUPRC, stable signed effects nor independent device agreement* establishes a physical source.

This experiment tests a different kind of evidence: **paired repeated acquisition of the same underlying system with an externally controlled instrument reset**. It requires outside knowledge that the physical state is unchanged between measurements. In actual hospitals such a study would require authorized synchronized repeated measurements and instrument logs; this is only a synthetic falsification, not something ConceptFAN infers from EHR time series.

## Frozen six conditions and a new independent calibration cohort
Reuse the exact V21 TRAIN candidate geometry: 960 TRAIN episodes (seed 405011), first 350 for the unsupervised V17 residual direction, next 300 for independent TRAIN outcome risk screening, next 250 for measured-reference correlation. Their statuses and candidate proposal hyperparameters do not change.

Separately and **independently** generate 240 new paired calibration episodes (seed 405111, repeat/acquisition-reset seed 405112), disjoint from TRAIN, the earlier V21 VALIDATION seed 405012 and sealed TEST seed 405013. The paired episodes have the **same underlying physical state and patient outcome** but two different sensor acquisitions. All 4 model-visible channels are normalized with the original **TRAIN-only** medians/IQR. The paired calibration contains only visible measurements, reference measurements, unique episode IDs and acquired-run metadata; latent generator states and patient outcomes never enter V17/V18/V21c screening or scoring. Synthetic generator internally holds the physical state fixed while resetting/removing instrument noise as specified below.

The exact **six** predeclared conditions:
1. `physical_predictive`: true physical source measured in a candidate channel, predicts outcome; persists across instrument reset.
2. `physical_nontask`: same measurable physical source, outcome independent; persists despite failing risk screen.
3. `artifact_predictive`: predictive device-only colored artifact, disappears when the acquisition device is independently reset.
4. `artifact_nontask`: nonpredictive artifact, also resets.
5. `artifact_common_mode_reference`: colored artifact was shared across two nominally independent devices in V21; a **real controlled acquisition-chain reset** resamples it and exposes the false physical explanation.
6. `artifact_environmental_persistent`: counterexample: shared *external environmental interference* persists despite the reset of both acquisition instruments. The paired repeat can falsely appear to corroborate the instrument artifact — **must report this failure**, rather than claiming a physical identity theorem.

## Measurement-only decision rule, fixed before any result
For each candidate, use last-six-step **observed measurement projection** onto its V17 TRAIN-only direction, before and after the controlled reset. Compute Pearson r across 240 paired calibration cases and two predefined split-half signed r values. All metrics use only paired observed readings. Screen:
- Pearson r ≥ 0.65;
- two-sided paired-index permutation p ≤ 0.01 with 199 permutations;
- 95% bootstrap lower correlation bound > 0.50 with 199 patient-level bootstrap samples;
- both predefined halves Pearson r ≥ 0.45;
- missing a paired acquisition, missing known reset protocol or constant signal => ABSTAIN.

If all pass, report `ACCEPT_ACQUISITION_REPEATABILITY`; otherwise `ABSTAIN_UNSTABLE_UNDER_INSTRUMENT_RESET`. This is **instrument-repeatability evidence only**. Even a passed controlled-reset test cannot prove physiology, since a persistent common-mode artifact can be repeated (the sixth negative case).

Report V17 proposal direction, V18 risk screen status, V21 independent-device corroboration, V21c paired-repeat status, pair correlation, 95% CI, permutation p, split-half correlations, within-acquisition cross-device correlations, and full six-case denominator. The calibration cohort is not used for model fitting or tuning. The generator truth labels enter only the **predeclared family names** / audit, not the test, model or thresholds. No real clinical data, human reviewers or official concept model baselines. No significance claim beyond the specified synthetic calibration samples.

A methodology that improves criterion matching in the first five cases but is fooled by case six is still scientifically useful as a **documented boundary**, not a medically sufficient discovery of true latent physical meaning.

The accepted IEEE PIERE camera-ready is untouched; this study stays in Draft PR #18.
