# V68 EXECUTED: stable statistical source subspaces do NOT certify stable chemical meaning

**Status: successful controlled retrospective negative audit.** V67 had already evaluated UCI270 batch10 before V68's rule was drafted, so the V68 thresholds and outcomes are **post-selection exploratory**, not a prospectively valid independent TEST or a calibrated failure probability. Frozen posthoc protocol: `CONCEPTFAN_V68_FROZEN_POSTHOC_DRIFT_SEMANTIC_GATE_20261113.md`. [GitHub Actions SUCCESS #43486635687](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43486635687), executed SHA `92ec5961aba5faf06b32c113820a3fc05f1a487c`, **13 tests passed**. Original official UCI270 ZIP pin `98fe3a30981a222dd4518fbcc3dddd45d5c0ce9b03ef6dc6fe5cf7a04cfbff5e`; aggregate immutable artifact `conceptfan-v68-posthoc-semantic-stability-abstention` ID **14067183625**, `STATUS.json`, `SEMANTIC_ABSTENTION.csv`.

## What remains physically grounded / unchanged

The original 13,910 UCI270 *single-compound exposure recordings*, 16 sensors × 8 engineered measurement descriptors, ten actual acquisition-period batches; TRAIN original batches1–7 **9546 exposures**, VALID batch8 **294** and batch9 **470**, previously viewed TEST batch10 **3600**. The source extractor is unchanged **TRAIN-only**: robust source median/IQR, 32 summary sensor features, PCA95 **K4**, unsigned ICA source directions fit without any gas type or concentration, 12 TRAIN-calibrated bounded fuzzy states. No gas labels entered the source rank, ICA or fuzzy state calibration. No V65 recurrence is claimed for static 128-feature records, and only one apparatus is represented.

The new V68 project is not a new classifier. It asks whether an autonomous statistical rank and **two sequential physical-reference validation periods** can flag false source/gas identity before a later period. Names necessarily come from previously labeled exposures; this is conditional association auditing, not autonomous molecular discovery.

## Unlabeled TRAIN source reliability: consistently K4, near-identical subspace

The original V67 K4 source was refit seven times, each leaving one of historical TRAIN acquisition batches1–7 out **before refitting training normalization/ICA**. Source rank remained **K4 in every fold**. Score-space canonical subspace overlap squared, invariant to ICA axis sign and permutation:

| Historical training batch withheld | Source rank | Mean squared canonical correlation vs full TRAIN source |
|---|---:|---:|
| 1 | 4 | .99795 |
| 2 | 4 | .99743 |
| 3 | 4 | .99918 |
| 4 | 4 | .99980 |
| 5 | 4 | .99994 |
| 6 | 4 | .99713 |
| 7 | 4 | .99692 |

Thus instability of *rank* alone cannot explain the observed failure to preserve chemical naming. The source subspace is surprisingly stable while its conditional meaning changes across later periods.

## A conservative VALID-only semantic abstention protocol — and its actual failure

For each gas name, form **one-to-one ICA-axis association using batch8 gas labels ONLY**; freeze axis/sign. On independent-in-time batch9 require at least 20 actual exposures per gas in each VALID batch, oriented OvR gas-name AUC>=.80 on **both** periods, difference<=.15, and all seven historical source ranks equal the original K. Otherwise ABSTAIN. *All thresholds were chosen AFTER seeing batch10 in V67*, making this a retrospective method study, not a fresh predeclared future guarantee.

| Physical gas | VALID batch8 AUC | VALID batch9 fixed-axis AUC | previously viewed TEST batch10 fixed-axis AUC | V68 decision from VALID only |
|---|---:|---:|---:|---|
| Ethanol | **1.0000** | **1.0000** | **.6070** | **EARLY VALID** — future semantics FAIL |
| Ethylene | **.9938** | **.9918** | **.6627** | **EARLY VALID** — future semantics FAIL |
| Ammonia | .9219 | 1.0000 | .8566 | EARLY VALID (no physical identity proof) |
| Acetaldehyde | no one-to-one axis | — | — | ABSTAIN |
| Acetone | no one-to-one axis | — | — | ABSTAIN |
| Toluene | .9966 | .8401 | .8907 | ABSTAIN: only 18 VALID8 specimens, AUC drop .1565 above .15 threshold |

The gate marks **3 gas associations as early stable**, but **2 of them fail even a descriptive future criterion AUC≥.75**. This is NOT a statistically estimated failure rate of a calibrated prospective rule: there are only six gases, one apparatus and an already inspected later batch. It directly falsifies any attempt to claim that a stable K, stable source subspace, and two strongly associated historical batches alone are sufficient evidence for permanent chemical source identity.

A shuffled batch10 gas-label control remains near AUC .5, consistent with labels carrying observational information. It does not resolve missing independent chemical reference or source-specific causality.

## Why marginal drift monitoring is insufficient

The mean absolute SHIFT of all bounded source membership state means relative original TRAIN source was approximately **.103 in batch8, .142 in batch9, and .091 in batch10**. Counterintuitively, final batch10 appears *less* shifted than either VALID batch by this global marginal metric, yet ethanol/ethylene source-label AUCs collapse from near1 to .607/.663. Thus an unlabeled monitor of only `P(C)` or of statistical source subspace similarity can pass while the actual conditional association `P(gas identity | C)` changes. This is a concrete field-data example of **semantic / conditional drift not captured by marginal concept drift**. It is a descriptive observational finding, not a causal theorem of sensor physics.

## Implications for ConceptFAN architecture, papers and deployment

- **Mathematical self-consistency (exact additive head) is not semantic identity.** V66 proved a reparameterization nonuniqueness; V67–V68 now show a physically grounded longitudinal mismatch while the source PCA rank and unlabeled geometry are stable.
- A semantic name may be attached **only** to a concept when externally provided, independently verified physical anchors justify it; absent new anchors in the deployment environment, use `UNVERIFIED_OBSERVATIONAL_PROXY` or **abstain from claiming chemical identity**. Even two positive earlier checks do not guarantee future identity.
- No justification for assigning six chemical gas labels to only K4 autonomous statistical axes, nor forcing K=6 after viewing TEST.
- The central scientific novelty is **testable epistemic boundaries**: distinguish (i) output fidelity, (ii) task-sufficient prediction, (iii) conditional proxy stability, and (iv) physically identified cause, with separate required evidence.
- This is an academic research-only dataset with **single-gas** exposures, one laboratory and no independent per-time chemical analyzer trace or simultaneous-source intervention. Not a source of claimed clinical verification, autonomous six-gas chemical ontology or universal transfer guarantees.
- The actual next evidence gate is a future **unseen, separately calibrated physical installation/recording sessions**, with physical source-specific interventions, gas reference labels withheld until the source and semantic calibration procedure are frozen. Do not optimize the current batch10 further.

Keep V67/V68 research PRs DRAFT, original PIERE/IOI accepted papers and master unchanged.
