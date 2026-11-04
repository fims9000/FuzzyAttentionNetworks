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


## 6. V21b executed follow-up: train-only signed-risk INITIALIZATION and patient-effect stability

Predeclared BEFORE run in `CONCEPTFAN_V21B_FROZEN_SIGNED_EFFECT_STABILITY_20261104.md`. [GitHub Actions 41849323592 — SUCCESS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41849323592). Complete `docs/medical/evidence_v21/signed_stability/` contains 30 opportunities: **24 trained full native ConceptFAN K3 networks and six risk-screen abstentions**, 3 initializations per family. All original V21 synthetic generator histories and frozen source/risk/provenance TRAIN cohorts were reused. 240 validation and 240 sealed synthetic test records; test remains sealed. No post-hoc reordering or sign alignment was performed.

Only the K3 native decision **initial raw coefficient** changed (±0.05, direction selected by the disjoint TRAIN-only binary risk screen); the sign is NOT constrained after initialization and the physical source generator/architecture/optimizer remain unchanged.

| Synthetic generator family | Arbitrary original start: AUPRC | TRAIN-screened native signed start: AUPRC | Mean cross-seed Spearman of signed patient-specific effect after signed start |
|---|---:|---:|---:|
| Physical source predicts outcome | 0.510179 | **0.699609** | **0.999841** |
| Physical source independent of outcome | 0.499915 | **ABSTAIN (3/3)** | N/A |
| Instrument-only colored artifact predicts outcome | 0.539715 | **0.760213** | **0.999946** |
| Instrument artifact not predictive | 0.498818 | **ABSTAIN (3/3)** | N/A |
| Common-mode contaminated reference plus predictive artifact | 0.533757 | **0.726691** | **0.999942** |

Without the oriented start, cross-seed Spearman averaged ~−0.333 in all families. The huge rank correlation increase is mostly fixing an arbitrary independent-training **decision-coefficient orientation** on a very simple shared measured direction, NOT proof of learned physical semantic alignment. Raw source matching remained nearly unchanged (physical source abs Spearman ~0.9836→0.9853; instrument artifact ~0.9906→0.9899).

**Critical semantic negative result**: exactly the same method produces very high AUPRC and nearly perfect signed-effect reproducibility for a source that is **known to be an instrument artifact** in the generator. The contaminated second device also passes measurement corroboration. Therefore “intrinsically additive + predictive + cross-seed stable + confirmed by two devices” is STILL NOT logically sufficient to identify a physiological mechanism.

No permanent sign constraint is required for this benefit in the tested setup; the result is consistent with V18b's sign-initialization confound. All five families here were already inspected in V21; interpret the experiment as a controlled diagnostic, not independent confirmatory or clinical generalization evidence. The null risk cases were explicitly recorded as abstentions, not given fabricated zero prediction scores.

## 7. Offline human-ready review instrument (software implemented; NO clinical participants)

`src/conceptfan_discovery/offline_native_review_v21.py` exports a signed and pseudonymous TRAIN-only review packet and a locally openable self-contained HTML view. It asks a plain-language question beside measured historical sensor plots and provides four choices: **Да, Нет, Не уверен, Артефакт измерения**. Answers are JSON and are imported by HMAC-verified pseudonymous TRAIN membership into the EXISTING V16 `PreparedFeedback` object that trains the native risk-mediating concepts. The review page never receives outcomes, original episode IDs, the synthetic oracle or the trained risk predictions. Missing measurement points are not visually interpolated. `unsure` and `artifact` never become false-negative physiological labels.

[GitHub Actions 41848050015 — SUCCESS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41848050015) covers packet signing, tamper refusal, unauthorized episode refusal, duplicate answer handling, schema validation, native gradient receipt and HTML generation. **No patient-level packet or HMAC secret is exported in CI.** This is research engineering, not a physician usability study, authentication of reviewers, GDPR/HIPAA certification, EHR integration or medical validation. Even pseudonymized normalized measurement curves can remain sensitive; reviewer HTML and decisions must be kept in an authorized private environment. Usage: `docs/medical/CONCEPTFAN_V21_LOCAL_REVIEWER_WORKFLOW.md`.

## Updated research conclusion

V21/V21b show that **observable source extraction**, **TRAIN-only risk relevance**, **independent measurement corroboration**, **native training-time expert feedback** and **independently initialized signed decision stability** are separable components of the existing ConceptFAN graph. They do not magically produce real physical concept identity: the common-mode artifact demonstrates an explicit indistinguishability failure. All feedback in experiments was a deterministic simulated measured-reference rule, not real physician annotation.

The next *genuinely missing evidence* requires independently sourced measured-device provenance, a predeclared real reviewer protocol and matched published intrinsic concept baselines on the same real/synthetic labeled task. It cannot be obtained simply by another seed adjustment on these five reused synthetic modes. Retain Draft PR #18, leave accepted PIERE camera-ready unchanged and do not promote any concept to causal/clinical without external evidence.
