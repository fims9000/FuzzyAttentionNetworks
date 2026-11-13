# V59: actual PhysioNet 2012 reference-support audit — negative source-link result

Date: 2026-11-13. **Evidence status: EXECUTED on REAL de-identified public measurements (TRAIN only), with an explicit ABSTAIN for source identity**. No mortality model was fitted or evaluated; no physical/clinical concept identity is claimed.

## Provenance and protection of previously sealed data

Official original public dataset: [PhysioNet/CinC Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/) Set A, downloaded intact from official repository inside GitHub Actions. Inputs `set-a.zip` SHA256:
`1433638192fda622e8bb11a4559b8af2dbb1c69f0e1c3b5b606cc12aabd0a4c5`.
`Outcomes-a.txt` SHA256:
`2613ea60ccda29f87571a7d6b09ad130858a8ccd66325bd073365925026883c2`.

[GitHub Actions SUCCESS #43470345811](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43470345811); executed script commit `b4ba8ba719dd4378ab0342dafa6c60c82bcb7085`; artifact `conceptfan-v59-real-physionet-reference-feasibility` ID `14061135530` includes **aggregate-only** `V59_AGGREGATE_EVIDENCE/STATUS.json` and source SHA. Source requires only numpy, pandas, scikit-learn and pytest; five real-source parsing/negative-control tests passed.

Reproduced original frozen Set A split, seed20260717, stratifying outcomes and ICUType to identify patient TRAIN membership: TRAIN2400, VALID600, CAL400, TEST600. The patient-level longitudinal measurements were then read **only from TRAIN**; validation/calibration/test timelines and mortality outcomes were not used for any score, model fitting or source calibration. Raw records, patient IDs and predictions are **not exported**.

The original V59 frozen preinspection protocol specified **three candidate pairs**, first 36h, no imputation, hourly and 4h binning, and availability gate: >=100 TRAIN patients have >=6 co-observed bins and three disjoint TRAIN folds each contain >=250 aligned paired cells. An additional diagnostic of the **already frozen V57 three-fold signed-correlation threshold |r|>=0.65** was added only *after the first V59 availability results were viewed*. It is explicitly post-support diagnostic, not a newly preregistered V59 hypothesis; no correlation threshold was lowered.

## Observed first-36h TRAIN overlap and Pearson correlations

| Pair | Bin width | Patients with >=6 paired bins / 2400 | Aligned paired cells | Pooled Pearson | Three TRAIN-fold Pearsons | Availability pass? | Original V57 match pass? |
|---|---|---:|---:|---:|---|---|---|
| MAP / NIMAP | 1h | 388 | 6640 | +.5600 | +.4858, +.5676, +.6105 | YES | **NO** |
| SysABP / NISysABP | 1h | 393 | 6832 | +.5315 | +.4962, +.5128, +.5704 | YES | **NO** |
| DiasABP / NIDiasABP | 1h | 392 | 6799 | +.4576 | +.4173, +.3794, +.5447 | YES | **NO** |
| MAP / NIMAP | 4h | 125 | 3151 | +.5785 | +.5460, +.5826, +.6009 | YES | **NO** |
| SysABP / NISysABP | 4h | 130 | 3232 | +.5062 | +.5224, +.4947, +.5016 | YES | **NO** |
| DiasABP / NIDiasABP | 4h | 128 | 3208 | +.4921 | +.4931, +.4463, +.5233 | YES | **NO** |

At 1h, the criterion >=6 jointly observed measurements is satisfied by ~16% of TRAIN episodes. At 4h, >=6 of 9 bins is satisfied by ~5% of TRAIN. Note that co-observed in a 4-hour bin means at least one value of each variable **within that four-hour bin**, not simultaneous device sampling. This can boost availability without certifying true calibration correspondence.

## Precise conclusion

**Availability:** All six pair/resolution combinations pass the predeclared numerical overlap floor. The dataset is sufficiently non-empty for *methodological feasibility* analysis and mask audits.

**Source linkage:** ZERO of six combinations passes the pre-existing V57 reference mapping condition **minimum |r| >=0.65 in all three disjoint TRAIN folds**. The maximum fold rho in the most promising candidate (MAP/NIMAP, hourly) is 0.6105, and its worst fold is 0.4858. Relaxing the cutoff after looking at these results would invalidate the existing verification contract. Hence:
`ABSTAIN_NO_PAIR_MEETS_UNCHANGED_V57_CORRELATION_AND_SUPPORT_GATES`.

**Physical provenance:** Even a correlation above cutoff would be insufficient to prove that invasive and non-invasive readings have independent instrument calibration or faults. Different acquisition modalities can share workflow confounding, patient selection, missingness, systemic offsets and biases. Independent source identity must be established from documented device metadata, controlled instruments and/or independently verified semantic references. Correlation is descriptive, not causal.

**Clinical results:** No real-data V58 anchor repair, mortality prediction improvement, concept interpretation, transportability or patient-level explanation result has been established. V57–V58's synthetic benefit is conditional on an extra independent sensor assumption that these records cannot verify.

## Next scientific requirement

A usable real-world reference test needs a documented sensor-pair identity, calibration/cross-device error specification and prospective or retrospective recorded device fault scenarios, preferably in independently collected environments. Keep current V59 status ABSTAIN; do not tune correlation threshold or reuse the sealed TEST cohort to force a positive result. Original accepted PIERE/IOI studies remain unchanged.
