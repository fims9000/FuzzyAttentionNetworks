# ConceptFAN V18 — consolidated executed findings and architectural decision (2026-11-04)

## Verified reproducibility and immutable histories
This report consolidates three executed, distinct protocol families, all **already versioned in GitHub** under `docs/medical/evidence_v18/`, `docs/medical/evidence_v18b/`, and `docs/medical/evidence_v18c/`. Every suite contains `STATUS.json`, machine-readable full-denominator CSV and `COMMIT_SHA.txt`. All tests and fitted outcomes below are actual GitHub Actions results. The accepted IEEE PIERE paper was not changed. The PR remains Draft.

Evidence files:
- `evidence_v18/screen/ALL_CASES.csv`: 18 source/noise predictive relevance cases
- `evidence_v18/native/ALL_FITS_AND_ABSTENTIONS.csv`: 45 model-arm opportunities (36 fitted models + 9 abstentions)
- `evidence_v18b/ALL_INIT_MATCHED_FITS_AND_REFUSALS.csv`: 27 initialization-control opportunities (18 fits + 9 abstentions)
- `evidence_v18c/BASELINES_ALL.csv`: 12 non-concept prediction baselines on the **same data**
- Original frozen protocol: `CONCEPTFAN_V18_FROZEN_RISK_SCREEN_20261104.md`, `CONCEPTFAN_V18_FROZEN_NATIVE_TRAINING_20261104.md`, `CONCEPTFAN_V18B_FROZEN_INITIALIZATION_CONTROL_20261104.md`, `CONCEPTFAN_V18C_FROZEN_PREDICTIVE_BASELINES_20261104.md`

## A. Source existence, predictive relevance and physical meaning are DIFFERENT
V17 residual-lag proposal was screened against the outcome on independently sampled **TRAIN-only** episodes (360 unsupervised discovery; 260 disjoint risk-screen; three synthetic generator seeds).

| Synthetic scenario | Native candidate proposed | Independent risk association accepted | Physical identity proven |
|---|---:|---:|---|
| New latent dynamic source, positive future risk | 3/3 | 3/3 | No |
| New latent dynamic source, negative future risk | 3/3 | 3/3 | No |
| New latent source, NO association with outcome | 3/3 | 0/3 | No |
| Time-correlated sensor-only noise, irrelevant | 3/3 | 0/3 | No |
| Independent white sensor-only noise | 0/3 | 0/3 | No |
| **Time-correlated sensor-only noise, predictive** | **3/3** | **3/3** | **No — false physiological interpretation** |

Interpretation: an independent risk screen can improve *task-relevance selectivity* but cannot solve physical source-vs-instrument ambiguity. A positive signal is a measured predictive pattern, not a disease, diagnosis or treatment effect. Uniform screening p=0.005 in strong cases is limited by 199 permutations and is not an external population FDR guarantee.

## B. Native ConceptFAN architecture, heldout AUPRC
On a different new synthetic generator seed 20411204: K2 initial native V10 model, 800 TRAIN, 200 heldout validation and 80 sealed test episodes. Third source drives outcome in the positive/negative modes, is unrelated in the null mode. Three model initialization seeds. All expansion and risk effects go through the *actual native additive concept decision*, no post-hoc or raw-input classifier.

| Native model | Positive source → outcome | Negative source → outcome |
|---|---:|---:|
| Continue K2 without adding | 0.504600 | 0.501871 |
| Unrestricted K3, always positive initial readout | 0.587819 | 0.527513 |
| K3 with correct TRAIN-screened signed policy | 0.587819 | 0.604346 |
| K3 additionally grounded to measured source | 0.585954 | 0.600978 |
| K3 with deliberately opposite signed policy | 0.504061 | 0.501787 |

The independently negative outcome mode shows better AUPRC with the correct initial sign than with the arbitrary positive start. Native grounding improves raw-concept/observed-direction alignment slightly (~0.9872 to ~0.9901), but **did not improve AUPRC**. The more complex architecture should NOT be selected by default on these results.

On null-outcome mode, independent risk screening ABSTAINED on all three initializations, while unrestricted unnecessary K3 lowered mean AUPRC from 0.460441 (K2) to 0.435822. A refused risk association does not negate the measured temporal pattern; it prevents an unjustified *risk explanation*.

## C. Critical V18b confound check: signed initialization vs architectural sign constraint
Originally, correct-sign K3 and unrestricted K3 had **different initial effective readout signs** in the negative-source task. V18b matched effective coefficients exactly at initialization and then repeated training with and without hard sign restriction. Full-denominator 27 opportunities, 18 completed fits + 9 null-mode abstentions.

| Same synthetic task | Unrestricted K3, correct signed initialization | K3, identical effective signed initialization + hard sign constraint | Unrestricted K3, deliberately opposite initialization |
|---|---:|---:|---:|
| Positive source | 0.587819 | **0.587819** | 0.524805 |
| Negative source | 0.604346 | **0.604346** | 0.527513 |

This decisively changes the *mechanistic attribution*: **correct, independently screened INITIALIZATION is enough to explain the observed predictive improvement**. The sign constraint did not improve AUPRC, signed magnitude, or patient-specific cross-seed rank correlation under these settings; it is a safety/rule policy, NOT an independently validated mathematical improvement.

The near-perfect per-episode signed effect rank correlation (~0.99999) across seeds occurs *also in unrestricted arms* on this intentionally simple dataset. Therefore do not cite it as a solution to the V16 cross-retraining instability (~-0.33) on the different, more ambiguous synthetic measurement problem. Fixing the sign is not learning a reproducible explanation.

## D. Standard classifiers outperform the native ConceptFAN on the same prediction task
V18c did not tune hyperparameters on heldout labels, never used latent-source oracle, and used exactly the same 800 TRAIN and 200 validation episodes, observed past 24×5 input, all three label modes. Full 12-row CSV in `evidence_v18c/BASELINES_ALL.csv`.

| Predictor (heldout AUPRC) | Positive-source mode | Negative-source mode |
|---|---:|---:|
| Native K3 with proper signed initialization | 0.587819 | 0.604346 |
| **Logistic regression on explicit recent level/trend/variation** | **0.731434** | **0.679452** |
| Logistic regression on full 24×5 past | 0.657644 | 0.655570 |
| HistGradientBoosting on the same recent features | 0.658870 | 0.666854 |
| Constant TRAIN prevalence | 0.515000 | 0.505000 |

Thus the existing ConceptFAN native representation **sacrifices substantial predictive performance** in this benchmark, and a simple classical classifier is not less accurate. Standard linear feature weights can also be auditable, but these comparators do not implement automatic editable concept creation, abstaining risk passports or a native concept graph. A fair next paper must compare both objectives and state these tradeoffs explicitly. This is NOT a comparison against official MAGNETS or a clinical benchmark.

## Scientific conclusion and implementation decision
**Supported architecture capabilities:**
1. Native concepts can receive bounded user feedback during end-to-end training (V16, no post-hoc attribution).
2. A new measurable temporal pattern can be proposed, explicitly approved and inserted into the same native prediction graph with zero immediate decision drift (V17).
3. Independently supported observable predictive associations can orient **initial native risk readouts** and reject new **nonpredictive** risk slots (V18).
4. Negative controls reject task-irrelevant sources, but **cannot** reject a task-predictive instrument artifact without external measurement/cross-view assumptions.

**Not supported:**
- Automatic medically meaningful physiological concept identity.
- General improvement in cross-retraining signed *patient-specific* effects.
- General AUPRC advantage over simpler predictors or official intrinsic concept baselines.
- Causal explanation, clinical suitability, independent hospital replication, or significance with three seeds.
- Real clinician corrections and irregular-sampling admission in V18 (current risk screening deliberately ABSTAINS on missing histories).

**Priority before new architecture versions:** (i) address K3 predictive undercapacity relative to logistic recent temporal features without introducing a hidden input-to-risk bypass; (ii) compare a matched official automatic concept baseline and repeated external family/split tests; (iii) handle ICU missingness and the separate care-observation process; (iv) only then real expert annotations plus independent measurement channel/calibration evidence. Do not conflate this independent research branch with camera-ready PIERE.

All synthetic results are exploratory and confined to the frozen configurations above.
