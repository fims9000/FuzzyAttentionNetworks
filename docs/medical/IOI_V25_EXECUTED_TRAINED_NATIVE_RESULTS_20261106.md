# IOI ConceptFAN V25 — executed trained-native omitted-factor study (2026-11-06)

**Status:** SUCCESS; all preregistered 18 unique native neural fits, 210 detailed evaluation rows, 84 aggregate summary cells, full synthetic oracle falsification. No clinical records and no PIERE revision. GitHub Actions final sealed run: [42112734524](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/42112734524), source commit `baa20fd9b7c758890ba59755f763212bf0bf3044`. Immutable artifact `ioi-v25-trained-complete-2` (ID 13454298146), with STATUS/SUMMARY/TRAINING JSON and raw aggregate logs. Prior run 42112598125 was superseded after fixing measurement masking in the oracle-residual witness; **cite final run only**.

## Independent subjects / model fits

Generator seed 20261106; 1600 synthetic TRAIN, 400 unused calibration, 800 disjoint TEST; 16 time steps, six noisy sensors, two predefined concepts and one candidate. Network seeds 11,23,37. Three ConceptFAN variants fitted on both COMPLETE and OMITTED outcome laws (2 × 3 × 3 = 18 unique fits): native K2, observation-weighted K2, privileged true-K3 concept control. The predictive ARTIFACT family is **exactly the same observable cohort and outcomes as OMITTED**; models/results are deliberately reused and not counted as new independent fits. Extra synthetic concept labels are provided during TRAIN only; the observed artifact's semantic identity is not learned.

## 1. Conditional *oracle* predictive completeness

Existing TRAIN-only logistic-probe diagnostic compares TEST NLL of `Y|C*_1,C*_2` with `Y|C*_1,C*_2,R` where `C*` is true synthetic oracle concept state and `R` is the **available-measurements-only** masked observed third sensor pair. A positive delta is an omitted predictive signal beyond the prescribed true dictionary, not a physical cause.

| Synthetic scenario | Held-out NLL improvement from adding R | 95% paired TEST-subject bootstrap |
|---|---:|---:|
| COMPLETE two-concept dictionary | +0.00065282 | [-0.00027544, +0.00153515] for seed11 |
| OMITTED third outcome factor | +0.09188270 | [+0.06741541, +0.11637905] for seed11 |
| ARTIFACT proxy for unmeasured prognostic source | **exactly same** +0.09188270 | exactly same observational data as OMITTED |

The central delta is the same across the three NN seeds because the oracle pair and candidate do not depend on fitted NN parameters; bootstrapped bounds vary slightly with bootstrap seed. **Do not count these as three independently observed oracle effects**. The measured-sensor witness now honors artificial MCAR20 masks and never reads hidden channel payloads. The complete dictionary does not show a resolved positive gain; the omitted case does. But the predictive artifact and omitted physical factor **cannot be distinguished from this observational evidence**.

## 2. Actual trained ConceptFAN prediction: MODEL underperforms simple baseline

AUPRC on exactly the same 800 TEST subjects under artificial MCAR20 (mean across three NN initialization seeds, single raw logistic baseline):

| Architecture | COMPLETE | OMITTED / ARTIFACT |
|---|---:|---:|
| Raw-input logistic regression, observed history statistics | **0.7601** | **0.8112** |
| Native fuzzy ConceptFAN K2 | 0.6611 | 0.6410 |
| Observation-weighted/gated ConceptFAN K2 | 0.6595 | 0.6308 |
| Native fuzzy ConceptFAN K3, privilege-supervised concept | 0.5577 | 0.7163 |

Native K2 in the omitted world: NLL **0.6704**, Brier **0.2388**; raw-input logistic NLL **0.5212**, Brier **0.1730**. Oracle-guided K3 reaches NLL **0.6611**, still much worse than raw logistic. Native K2 variation is large: per-seed AUPRC **0.469/0.718/0.735**. Thus optimizer/decision-head instability is a major unresolved alternative to explanation-completeness failures. Exact additive decomposition errors are <=2e-5 (logged as 0 in this CPU run); model fit passes, but no advantage is established.

## 3. Measurement-support wrapper is NOT a successful predictive improvement

Observation-aware K2 in OMITTED: TEST MCAR20 coverage **97.125%** and AUPRC **0.6308**. At MCAR40 coverage **9.375%**, overall AUPRC **0.5073** and NLL **0.6885**; at MNAR coverage **8.5%**, AUPRC **0.4886** and NLL **0.6969**. Under blocked 8-step gaps, coverage **0%**: a trivial explanation-abstention strategy is not informative for practice. The wrapper directly removes unavailable concept contributions and changes the risk calculation at test time without having been trained on comparable severe missingness patterns. Its poor behavior supports preferring V24's **non-destructive reporting-only** support audit, while retaining native prediction separately.

## 4. Concept fit, head limitations and interpretation

Synthetic TEST concept trajectory MSE in OMITTED/MCAR20: native K2 ~**0.0073**, observation K2 ~**0.0071**, privileged K3 ~**0.0083**, while prediction AUPRC varies sharply with seed. Low MSE does not guarantee well-optimized output coefficients or proper calibration. The existing fuzzy-head training (14 fixed epochs) may be underoptimized; K3 improved on omission but performed poorly in COMPLETE. Follow-up **V26** is an explicitly exploratory, later-designed control that freezes native K2/K3 representations, compares original fuzzy head, positive affine calibration (cannot improve AUPRC) and a direct additive linear concept head fitted on TRAIN labels. Do **not** treat this as independent prospective TEST evidence.

## Limitations and novelty boundary

1. The successful oracle test measures extra *predictive information*, not physical, semantic or clinical cause identity; observationally equivalent ARTIFACT deliberately falsifies the stronger claim.
2. True synthetic oracle concepts are unavailable in real PhysioNet, and 400 calibration subjects remain unused. No real ICU tests were executed.
3. The raw logistic competitor is not a parameter/computation-matched deep model; it is a transparent strong sanity baseline, and outperforms the proposed neural alternatives at this optimizer budget.
4. Three NN seeds on a single synthetic generator/test cohort are descriptive. TEST subject bootstrap on the same oracle gain does not create new NN or hospital cohorts.
5. This extends completeness/unsupported-explanation research, not PIERE's cross-training signed contribution reproducibility or IITI's concept-latent alignment.
6. No currently defensible claim that the new ConceptFAN architecture **improves** prediction, identifies missing medical concepts, or achieves clinical trustworthiness. Its strongest observed result is a bounded synthetic falsification protocol.

Reproduce from pinned commit:
`PYTHONPATH=src:. python scripts/experiment_trained_conceptfan_omission_v25.py --out-dir V25_NATIVE_EVIDENCE`
