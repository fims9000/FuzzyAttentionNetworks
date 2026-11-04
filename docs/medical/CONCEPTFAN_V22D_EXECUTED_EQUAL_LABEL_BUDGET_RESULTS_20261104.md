# ConceptFAN V22d — executed equal 800-label budget native source/task relevance comparison

**Research date:** 2026-11-04. Frozen protocol `CONCEPTFAN_V22D_FROZEN_EQUAL_LABEL_BUDGET_20261104.md` committed before the first V22d outcome evaluation. Execution via `.github/workflows/conceptfan-v22d-equal-label-budget.yml`; full machine-readable evidence in `docs/medical/evidence_v22/equal_label_budget/` (`STATUS.json`, 48-case CSV, 16-arm aggregate CSV, source evidence, independent TRAIN risk screens and original source commit SHA).

All **48 registered cases** accounted for: **36 native trained models, 12 BSS identifiability refusals**, and **36 untouched TEST evaluations** (300 new generator episodes per family, seed **522313**, never used in V22a–c). All TEST rows and negative cases remain intact. The code does not create TEST episodes until all 36 models and all TRAIN source-risk decisions finish. It records native model parameter SHA256 before and after TEST and refuses mutation or incomplete denominators. Generator truth is loaded only for postfit source alignment audit.

## Why the control matters
V22b/V22c screened task relevance using 200 extra independently disjoint TRAIN labels, versus the arbitrary-start baseline which used only 600 labeled neural gradient cases. This could create an unfair label-budget comparison. Here **both arms use all 800 labeled TRAIN episodes**, but allocate them differently while holding the *same* source representation:

- `joint_800_labels`: fit original native V10 jointly for ten epochs on all 800 TRAIN episodes, no independent source-risk gate.
- `screened_600_fit_200_gate`: fit original native V10 jointly for ten epochs on first 600 TRAIN, then use the other 200 independent TRAIN outcomes in a frozen per-source screen to choose starting signed raw coefficients ±0.05 and/or a serialized exact zero task contribution (`signed_risk_policy=2`) for source concepts without corroborated outcome association.

Both methods estimate PCA whitening/BSS source geometry on exactly the **first 600 TRAIN histories**, with no source/normalizer information from the other 200. The same NN architecture, native K2 bottleneck, seed set 11/23/37, V10 auxiliary forecast/denoising loss, batch size 96 and source axes apply at initialization. This is **equal labeled-episode budget, not equal compute/optimizer exposure**: 800 joint training examples appear over ten epochs, while 600 screened examples appear over ten epochs plus a separate 200-label statistical screen. Different optimization trajectories remain a confound.

## Previously uninspected new synthetic TEST 522313: AUPRC

| Generator family | Source basis | Joint 800-label native | Screened 600+200 native | AUPRC delta screened−joint |
|---|---|---:|---:|---:|
| Distinct source variance | PCA | 0.671692 | **0.846775** | +0.175083 |
| Distinct source variance | Environment BSS | 0.690945 | **0.851599** | +0.160654 |
| Proportional variance; sources not separately identified | PCA | 0.706254 | **0.877119** | +0.170865 |
| Sensor gain drift; mixing assumption violated | PCA | 0.696868 | **0.864221** | +0.167352 |
| Instrument-only variance change | PCA | 0.708443 | **0.862555** | +0.154112 |
| Instrument-only variance change | Environment BSS | 0.522398 | **0.866054** | +0.343656 |

BSS-based source separation **ABSTAINS** for the proportional variance and mixing drift families (3 seeds × 2 task arms × 2 families = **12 refusals**). Those missing values must not be interpreted as failed predictive accuracy scores.

## TEST Brier calibration (smaller is better)

| Generator | Basis | Joint 800 | Screened 600+200 |
|---|---|---:|---:|
| Distinct variance | PCA | 0.239695 | **0.235054** |
| Distinct variance | Environment BSS | **0.237793** | 0.241394 (worse) |
| Proportional variance | PCA | 0.238376 | **0.233774** |
| Sensor gain drift | PCA | 0.237399 | **0.233728** |
| Instrument-only variance | PCA | 0.236657 | **0.233185** |
| Instrument-only variance | Environment BSS | 0.252595 | **0.240163** (still worse than screened PCA) |

Thus the screened architecture's advantage in outcome **ranking (AUPRC)** persists under matched total labeled examples and previously unused TEST seed. But **better source separation still does not guarantee better calibrated risk**; BSS-screened has worse Brier than joint-800 BSS in the distinct-variance scenario and than screened PCA in both BSS-admissible scenarios.

## Mechanistic inference and restrictions
The controlled observation supports that allocating a portion of the same labeled TRAIN budget to **explicit native source–task relevance assignment** can improve predictive ranking over feeding the same labels only into the original V10 joint optimizer, at the exact frozen hyperparameters and synthetic generators. The experiment does **not** establish that the gain is caused by a physically meaningful source representation: the task-risk screen may reject genuine non-prognostic physical processes and accept stable sensor artifacts. We retain such non-risk sources as descriptive native concept slots rather than mislabel them nonphysical.

It is NOT a compute/gradient-step equalization, independent hospital trial, real clinician validation, source-causal identification proof or official published concept-bottleneck model baseline. No significance tests from three neural seeds on the same 300 synthetic TEST cases; no further tuning is permitted on already inspected seed 522313.

## Research decision
Keep **separate TRAIN-only source formation + disjoint supervised task-role assignment** as a justified *candidate training scheme* for next external evaluation. It has controlled advantages on synthetic AUPRC but must be compared with task-specific strong predictive baselines and official intrinsic concept models on **the same patient-independent real datasets** before claiming a general AI or medical breakthrough.

Accepted PIERE camera-ready unmodified; Draft PR #19 remains unmerged.
