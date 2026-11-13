# V59 pre-data protocol: real ICU independent-measurement anchor feasibility

Registered 2026-11-13 **before V59 public PhysioNet TRAIN support results were read**. This is an aggregate availability audit; it does NOT train, test, or clinically validate V58. Original V58 synthetic outcomes have already been viewed, so this stage is exploratory.

## Actual source and selected pairs

Official public PhysioNet/CinC Challenge 2012 **Set A** (4,000 ICU episodes) and Outcomes-a.txt, official public source https://physionet.org/content/challenge-2012/1.0.0/ . Read-only data use. Do not use Set B/C or any person-level identifiers in artifacts. The pairs are PREDECLARED:

- invasive mean arterial pressure MAP versus non-invasive mean arterial pressure NIMAP;
- invasive systolic SysABP versus non-invasive systolic NISysABP;
- invasive diastolic DiasABP versus non-invasive diastolic NIDiasABP.

They have approximately corresponding physiological measurement targets, but **different names do not prove independent fault domains, independent calibration, or correct instrumentation**. This audit cannot assign a medically validated concept ontology.

## Frozen split and leakage restriction

Use the same project's published `conceptfan_realdata.data.create_split` algorithm and seed **20260717**; stratifies all official 4,000 *metadata-only* RecordID / ICUType / in-hospital-death, then allocates TRAIN 2400 / VALID 600 / CAL 400 / sealed TEST 600. The reference outcome file is used ONLY to reproduce this pre-existing split membership. No outcomes, raw time-series measurements or method selection from VALID/CAL/TEST will be read into the V59 audit. Read official full records only enough to extract ICUType for split; parse **only TRAIN** time-series rows.

Input window: first **36 hours** from ICU admission, aligned to whole-hour bins. For repeated observations within an hour, use median of actually observed values, never impute; -1 and missing values are absent. Independently evaluate 1h aligned co-observation and disjoint 4h bins (9 bins over 36h), with at least one *originally observed* value per bin, no forward filling. Exact train split SHA and source SHA receipts, not raw patient IDs, go to artifact.

## Primary feasibility acceptance checks

For each predeclared pair and temporal resolution report:
- n TRAIN patients with at least one measurement of each sensor (not necessarily simultaneously);
- aligned jointly observed bin count and patients with >=1 aligned pair;
- patients with >=6 aligned paired bins, because V57/V58 per-case anchor verification requires at least 6 co-measured timepoints;
- three disjoint train-fold joint cells, requiring >=250 per fold to adapt V57's source-anchor correlation gate;
- masked observed-only Pearson r across pooled paired values as a *descriptive association*, not sensor identity;
- no subject-level values or predictions retained/published.

The primary gate is PASS only when **>=100 TRAIN patients have >=6 paired bins** AND each train fold has >=250 aligned paired time cells. For 4h bins nine total, this is an intentionally demanding threshold. If no pair passes, **ABSTAIN_CLINICAL_ANCHOR_UNOBSERVABLE**, do not train or assess the V58 reference repair on real patients, and describe what external data would be required. If one pair passes, status **MEASUREMENT_OVERLAP_FEASIBLE_UNVERIFIED_PROVENANCE**: this still does NOT certify clinical robustness or instrument independence.

## Quality and ethics

Only aggregates, official file sha256s, cohort counts and code version are outputs. No patient identifiers, raw patient timelines, per-patient explanations, mortality predictions, fitted risk heads or label associations. No TEST leakage. The output is to be reviewed as real-data **feasibility** only and must never be called prospective clinical performance. Sensor availability may be MNAR; 1h/4h aggregation is a methodological choice and not a source-identification theorem.

The true scientific next gate after an adequate pair is source provenance documentation/calibration, documented independent device faults or controlled sensor interventions, and an explicitly independent concept meaning reference. Without those, no clinical identity claim.
