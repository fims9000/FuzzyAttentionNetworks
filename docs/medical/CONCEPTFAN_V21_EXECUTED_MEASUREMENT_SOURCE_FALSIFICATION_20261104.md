# ConceptFAN V21 — executed physical-vs-instrument source falsification (2026-11-04)

## Evidence and versioned provenance
Frozen protocol: `docs/medical/CONCEPTFAN_V21_FROZEN_MEASUREMENT_PROVENANCE_AND_EXPERT_AUDIT_20261104.md`, committed **before** first V21 experiment execution. [GitHub Actions 41846321592 — SUCCESS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41846321592): all 45 full native ConceptFAN neural fits and the original V16–V18 concept/feedback regressions passed. Data under `docs/medical/evidence_v21/provenance/`: `ALL_45_NATIVE_FEEDBACK_FITS.csv`, `AGGREGATE.csv`, `FIVE_SOURCE_PROVENANCE_PASSPORTS.json`, `STATUS.json`, `COMMIT_SHA.txt`.

Five independent generator conditions, three training seeds 11/23/37, and three arms: native no-review baseline, simulated independent *measured reference* yes/no review (or UNSURE if corroboration fails), and deliberately inverted negative-control review. No real physician participated. References and hidden generator states were never inputs to ConceptFAN; hidden source truth was loaded **only after all nine fits for each family** to evaluate whether the learned native slot followed the physical latent source or instrument artifact.

## 1. Risk relevance ≠ measurement provenance ≠ true physical identity

| Frozen synthetic condition | V17 persistent residual direction | V18 independent TRAIN risk association | V21 separate measured-device corroboration | Postfit ground truth |
|---|---|---|---|---|
| Physical source predicts target | sensor_2 (abs loading ~0.9999) | ACCEPT, signed point-biserial r=+0.360 | ACCEPT, r=+0.997 | Physical source |
| Physical source, no target effect | sensor_2 (~0.9999) | ABSTAIN, r=+0.082 | ACCEPT, r=+0.997 | Physical source, **not** prognostic |
| Sensor-only colored artifact predicts target | sensor_3 (~1.0000) | **ACCEPT**, r=+0.498 | **ABSTAIN**, r=-0.039 | Instrument artifact, not physiological |
| Sensor-only colored artifact, independent target | sensor_3 (~1.0000) | ABSTAIN, r=-0.031 | ABSTAIN, r=-0.039 | Instrument artifact |
| Common-mode artifact contaminates the 'independent' registered reference and predicts target | sensor_3 (~1.0000) | **ACCEPT**, r=+0.528 | **ACCEPT**, r=+0.995 | **Still an instrument artifact** |

The external device was independently registered by ID, but the last negative control deliberately introduced a shared measurement artifact. Cross-device agreement is not enough to establish physical/clinical semantics; it needs independent acquisition assurance and external provenance. All five V21 reference passports correctly include `clinical_or_physical_causal_identity_established=false` and `cross_device_common_mode_artifact_ruled_out=false`. The positive findings are deliberately **measurement corroboration, not physical certification**.

## 2. Heldout neural physical-vs-artifact alignment, not cherry-picked
Mean absolute Spearman rho between the *actual native third concept's raw activation* and generator-known latent paths (post-training audit), all 3 neural seeds:

| Scenario | Native slot vs hidden physical source | Native slot vs hidden instrument artifact |
|---|---:|---:|
| Physical predictive, no feedback | **0.9836** | 0.1119 |
| Physical predictive, measured reference feedback | **0.9848** | 0.1139 |
| Artifact predictive, no feedback | 0.1367 | **0.9906** |
| Artifact predictive, measured reference feedback | 0.1367 | **0.9906** (uncorroborated reference abstained) |
| Common-mode artifact, measured reference feedback | 0.1357 | **0.9904** |

Thus the model *does identify a strong temporal measured direction* in this deliberately simple synthetic measurement geometry. The source may be a stable **instrument artifact** rather than the physical process the research aims to describe. Oracle/source identity labels are never an input to fitting or concept creation. An absolute correlation near one is not sufficient to prove physical meaning in observational data.

## 3. Simulated expert feedback is train-native but not a validated gain

| Synthetic family (mean heldout prediction AUPRC) | No reference feedback | Correct measured-reference yes/no if corroborated | Intentionally inverted reference feedback |
|---|---:|---:|---:|
| Physical source predicts outcome | 0.51018 | **0.51134** | 0.50480 |
| Physical source, independent target | 0.49991 | 0.50019 | 0.49986 |
| Instrument artifact predicts outcome, clean reference | 0.53972 | **0.53972** (reference UNSURE) | 0.53895 |
| Instrument artifact independent outcome, clean reference | 0.49882 | **0.49882** (reference UNSURE) | 0.49940 |
| Common-mode artifact + contaminated reference predicts outcome | 0.53376 | 0.53567 | 0.53121 |

Native reviewer labels `yes/no` supervise the SAME risk-mediating K3 concept raw activation via `native_expert_loss`, not a parallel post-hoc explanation head. But feedback improved physical-predictive AUPRC by only about +0.00116, and physical-source alignment by about +0.00122; in this synthetic setup the original concept already had near-perfect measured-reference AUROC of 1.0. There was very little headroom, so these data do **not** establish that human input substantially improves concept correctness. Intentionally inverted annotations reduced positive-outcome AUPRC, showing that a wrong reviewer can hurt.

The feedback experiment is simulation-driven: 80 labelled TRAIN intervals per yes/no arm, based only on a separately *measured* reference. No physician names, real diagnoses or clinically certified concepts. For an uncorroborated candidate, `unsure` **does not** become a negative clinical label; original V16 loss receives no supervised intervals in that arm.

## 4. Major negative finding: signed patient-specific contribution reproducibility remains unresolved
Across all five conditions and all three review arms, average cross-seed Spearman signed native third-concept contribution ranking is approximately **−0.333**. This reflects orientation/optimization instability; even when *raw concept activation* correlates with the known source nearly perfectly in absolute value, the actual signed risk contribution for individual patients does **not** become stable under independent model initializations. Do not claim that physical alignment or exact additive logit decomposition solves faithful stable clinical explanations.

A valid next V22 study should distinguish slot permutation, source-sign alignment fixed from independent TRAIN-only measurement provenance, and genuine patient-specific decision-effect variability without overwriting the raw signed results. Aligning signs by construction does not prove cross-run semantic identity; compare both raw and aligned values in a predeclared separate cohort.

## 5. Scientific boundaries and decision
- V21 provides controlled **falsification** of the idea that high prediction/measurement correlation implies clinical/physical correctness. Cross-device agreement can be mimicked by a common-mode artifact.
- The V21 generator has extremely simple source mixing with one dominant candidate sensor axis. Strong oracle-alignment numbers cannot be extrapolated to mixed sensors, transfer learning or unknown clinical factors.
- One or two excellent correlations on the same synthetic generator do not constitute a novel physical-law discovery, causal identification theorem, or prospective clinical validity.
- Source vs sensor artifact cannot be inferred solely from time series and outcomes where different physical worlds produce the same observations; external instrument provenance, repeat measurements, independent interventions or validated expert input are needed.
- Native expert feedback currently demonstrates **train-time control**, not clinically verified semantic correctness or meaningful AUPRC gain.
- Existing V20 native-head optimization and TRAIN-only accept/keep-original gate are separate downstream *prediction* improvements; they must not be conflated with this V21 source audit.
- Accepted IEEE PIERE-2026 camera-ready remains unchanged and PR #18 remains Draft.

The scientifically useful V21 accomplishment is therefore an explicit **semantic verification boundary** around predictive concepts, not a claim that a new physical source was proven.
