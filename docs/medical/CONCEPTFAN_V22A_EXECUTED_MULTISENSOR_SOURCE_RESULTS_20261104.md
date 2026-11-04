# ConceptFAN V22a — executed mixed-sensor source separation: much better identification, not better risk

**Date:** 2026-11-04. **Status:** exploratory synthetic study on already-designed source families, in Draft PR #19 based on Draft PR #18; accepted PIERE camera-ready unchanged.

Frozen protocol: `docs/medical/CONCEPTFAN_V22_FROZEN_MIXED_SOURCE_ENVIRONMENT_STRESS_20261104.md`. A pre-execution clarification explicitly added an independent TRAIN sensor-subspace stability check for the assumption of fixed sensor mixing; the .99 threshold was informed by local TRAIN simulated geometry and is **not** an externally blinded cutoff. Full executed provenance under `docs/medical/evidence_v22/mixed_sources/`.

[GitHub Actions 41857411144 — SUCCESS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41857411144): 24 registered opportunities (four generator families × 3 NN seeds × 2 native source-basis methods); 18 native models fitted, 6 explicit TRAIN identifiability refusals. Complete CSV/JSON, code hash, tests and refit invariants retained.

## Why this is different from V21
In V21, a hidden physical or instrument-generated factor loaded almost entirely onto one visible sensor, so stable residual detection was easy. V22's two independent colored factors each load nontrivially onto **all five sensors**, with non-orthogonal mixing columns (absolute cosine 0.5484). No one observed sensor can be taken as the corresponding latent source. The second-order source-separation algorithm is **not new mathematics**: pooled covariance whitening + environmental covariance-difference eigendecomposition, whole-patient permutation and bootstrap confidence, environmental mixing subspace-stability check. It sees only TRAIN observed histories and environment IDs, never outcomes or hidden generator factors.

Both arms use exactly the same native K2 V10 ConceptFAN architecture, ten-epoch joint loss, matching seeds, the same TRAIN-only whitened measurements, and exact additive native risk logits. No external post-hoc task classifier. The only difference is the source measurement basis in the native concept layer. Source alignment with generator truth is evaluated only after fitting; Hungarian sign/permutation matching is for **evaluation only**, never model selection, intervention or training.

## Full results including deliberately difficult controls

| V22 generator | Environment BSS identifiability screen | PCA linear source oracle alignment | BSS linear source oracle alignment | Native PCA concept oracle alignment | Native BSS concept oracle alignment | PCA AUPRC | BSS AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|
| Different source variances across environments | ACCEPT (eigengap 1.507; p=.005; bootstrap q10=.99995) | 0.69336 | **0.99922** | 0.68758 | **0.96823** | 0.66426 | 0.66905 |
| Proportional source variance profiles (indeterminate identity) | **ABSTAIN** (eigengap .0269; permutation p=.60) | 0.68591 | — | 0.67653 | — | 0.68443 | — |
| Environment-specific sensor gain drift | **ABSTAIN** (subspace min-cosine .970996 vs required .99) | 0.74342 | — | 0.70718 | — | 0.68890 | — |
| Instrument source variance changes; physical source nearly unchanged | ACCEPT (eigengap .7598; p=.005; bootstrap q10=.99954) | 0.74614 | **0.99896** | 0.71319 | **0.97341** | **0.69574** | **0.52924** |

Three model seeds per arm, all same 300 synthetic validation subjects per family; no p-values of inter-model outcome improvement. Raw source alignment is *absolute Spearman matched to generator-known sources* (two-source Hungarian assignment evaluation). Native alignment is the actual learned native concept raw-activation, not external source score.

**Main positive result:** with genuinely heterogeneous independent source variance across stable environments, observable unsupervised second-order source directions can provide intrinsic concept formation highly aligned with both known generating sources, much more so than plain pooled PCA. This is a recovery of **statistical latent factors under a controlled identifiable generator**, not proof that physical factors are uniquely identifiable in hospital data.

**Major negative result:** excellent recovered source geometry does NOT guarantee task prediction benefits. In artifact-only variance environment, BSS-native AUPRC falls **0.69574 → 0.52924**, and validation Brier worsens **0.23730 → 0.25198**. In distinct variance, the AUPRC change is only +0.00479 and Brier worsens slightly (0.24035 → 0.24115). Do not claim this BSS is a better task model. Unscreened signed concept contributions remain unstable, cross-seed mean signed Spearman roughly −.333 for baseline; environment BSS gives +.335 in the two-source variance family but −.329 in artifact-only family.

**Abstention is substantive:** proportional environmental variance cannot identify source rotation by this covariance method; blind refusal avoids false physical names. Calibration drift violates fixed mixing; the sensor-subspace precondition detects it in this deliberately chosen TRAIN setup. It is not a guarantee of catching arbitrary within-subspace calibration changes, and the near .99 subspace threshold was informed by the simulation geometry.

**Physical-vs-artifact failure remains:** the same algorithm separates the instrument-generated component as strongly as the physical-generated component; external provenance or intervention remains mandatory. Signed prediction, sensor correlation, and recovered source direction can be compelling yet semantically wrong.

## Diagnosis and next experiment
Prior V21b established that small ±.05 initial task-risk weights derived from an independent TRAIN-only outcome association can strongly change actual signed effect reproducibility and task AUPRC. V22a did NOT apply this screen. V22b therefore preregisters 4 families × 3 seeds × two source methods × random vs screened risk native initialization, with 600 source/model-fit TRAIN episodes and **200 completely disjoint TRAIN risk-screen episodes**. Rejected predictive sources are not discarded as invalid physical patterns: they remain measurable native sources with **exact serialized zero risk contribution** (V21d policy code 2), while accepted sources receive a TRAIN-only signed raw-weight initialization. This changes source-risk interpretation, not statistical source formation. Its eventual results must not be combined with V22a as a same-sample controlled model-architecture effect because the core TRAIN fit shrinks from 800 to 600.

No external clinical data, real clinician input, official intrinsic concept baselines or novel blind separation theorem. Code/test/report remain research-only and not merged to mainline.
