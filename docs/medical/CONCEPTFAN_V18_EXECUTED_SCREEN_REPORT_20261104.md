# ConceptFAN V18 — executed 18-case independent predictive screen

## Provenance
GitHub Actions [41826238772 — SUCCESS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41826238772), artifact `conceptfan-v18-all-risk-screen-cases-2` (ID 13318039811), 18 complete records, `ALL_CASES.csv`, `SUMMARY.csv`, `STATUS.json`, `COMMIT_SHA.txt`. Actual checkout SHA `68c58b303b41b54ec1f7c71dd9f98151c14bdbb8`. Downloaded ZIP SHA-256 `464c7dac9a17548d0d266e56e38101583e7b48c35707b7e5a8739497c6a137ba`.

| Generator mode | Candidate proposed | Independent observational risk association accepted | Refusal |
|---|---:|---:|---:|
| Added latent source, positive outcome link | 3/3 | 3/3 (orientation +) | 0/3 |
| Added latent source, negative outcome link | 3/3 | 3/3 (orientation −) | 0/3 |
| Added latent source, **no** outcome link | 3/3 | **0/3** | 3/3 weak risk association |
| Colored sensor-only noise, predictive of label | 3/3 | **3/3 false physical interpretation candidates** | 0/3 |
| Colored sensor-only noise, irrelevant | 3/3 | **0/3** | 3/3 weak association |
| White sensor-only noise, irrelevant | 0/3 | **0/3** | 3/3 no candidate |

All 18 cases executed with 360 disjoint TRAIN-only source-discovery and 260 TRAIN-only labeled risk-screen episodes. Sensor channels fully observed; no oracle latent values used to compute directions, screening labels or risk orientation. `permutation_p_two_sided` on accepted positive/negative scenarios = 0.005 (resolution of 199 permutations). The tests demonstrate the specified selection policy on a simple, intentionally constructed linear-AR generator family; they do **not** establish the stated 1% threshold as a population-level false-discovery guarantee after all selection operations.

## Main scientific observation
V18 can withhold **task-risk interpretation** from novel patterns with no reproducible association, while retaining the existence of an observable trajectory. This is a substantial safety-oriented distinction from automatic promotion of every candidate to a signed clinical risk factor.

Critically, colored *sensor-noise* trajectories predictive of the target pass the screen in all 3/3 scenarios. Their observed process is indistinguishable from the genuine latent source in this experiment. Predictiveness, reproducibility, a stable permutation p-value and even expert approval cannot by themselves prove physical causality. The risk-sign orientation is a conditional **observational association**, not a medical treatment effect.

## What this did not demonstrate
- No new measured physiological concept established or doctor validation.
- No independent real hospital dataset.
- No proof of improved outcome AUPRC, risk-calibration, or semantic quality after signed native retraining.
- No proof that per-patient signed contributions are stable across retrainings. The code forces direction as a constraint when enabled: sign agreement by construction is a **trivial property** and must not be presented as an empirical stability gain.

## Next experiment
Evaluate the same signed-risk policy during the *actual native K2→K3 ConceptFAN training* against matched unrestricted risk readouts. Report AUPRC/AUROC/Brier, magnitude of nonzero contributions, old-path decomposition, and cross-seed rank agreement after controlling for sign. Assess null outcome and inverted-sign controls. Preserve every failed fit and all frozen seeds. Original PIERE camera-ready remains untouched.
