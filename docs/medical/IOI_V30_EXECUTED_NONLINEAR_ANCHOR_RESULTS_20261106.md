# ConceptFAN V30 — executed independent-anchor repair on new nonlinear temporal source simulator

**Date:** 2026-11-06. **Verified:** [GitHub Actions 42120972760](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/42120972760) **SUCCESS**, 23 pytest tests, full actual canonical native ConceptFAN experiment, 3 distinct K2 neural fits, 576 recorded TEST evaluations and 256 aggregate result cells. Exact executed source commit \`f53454d7ac6271bf247dd4322b51f18fde348274\`; [artifact 13457954993](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/42120972760/artifacts/13457954993) includes STATUS, SUMMARY, GATES, MODEL_STATES, full logs and source commit SHA. ZIP SHA256: \`7e4aba076ed62fa64073f6faad20912a80ed943c78203322398aecef81abb5bd\`.

### New generator, not a new external domain

The frozen [pre-execution protocol](./IOI_V30_FROZEN_NONLINEAR_ANCHOR_PROTOCOL_20261106.md) uses a **nonlinear regime-switching driven process with sparse non-Gaussian Laplace shocks**, new seed20261110. This differs materially from the V25–V29 stationary Gaussian AR(1) synthetic family, but is still an artificially designed 3-source simulated law with additive source-risk structure. TRAIN1800, two separated labelled VALID600+600 and independent TEST900 subjects; the same TEST subjects are reused for corruption modes and NN seeds. Original K2 known-concept encoder reads **only sensors 0–3 plus measurement indicators**, not any new source signals. Its concept path has no outcome gradient, and its original risk coefficients are frozen. Synthetic TRAIN provides true concept trajectories — no comparable real ICU ground truth is established.

The new third-source candidate is an explicitly named operator based on actually observed sensor4+5 history; gamma fitted on TRAIN outcome labels, and admission on TWO labelled VALID environments (1200 extra labels, CI/gain >0). All 3 seeds **admitted** a predictive candidate, and all 3 **rejected** a third nonpredictive nuisance. Conditional VALID gains for admitted source were roughly +0.068 to +0.077 log-loss on both environments. No TEST labels, oracle true source states or future readouts used for training or admission.

V29 tests agreement of first two third-source sensors. V30 adds an independently noisy, **assumed externally calibrated** sensor6 measuring that same latent source. A fixed TRAIN-only threshold on disagreement of sensor-pair vs sensor6 is 0.126611 in sigmoid-mean coordinates (99th percentile), and pair integrity thresholds are corr>=0.297928, mean abs gap<=0.642662, at least five jointly measured steps. With agreement, the candidate uses the pair; when pair internally fails or disagrees with anchor, V30 either **rejects** or **substitutes independently measured anchor activation**. The labelled risk head gamma does not change, original two concept activations or K2 risk coefficients do not change, risk remains strictly additive in separately named source contributions.

## Primary results: AUPRC on the same 900 independent TEST subjects, MCAR20

Mean across 3 native K2 neural initializations; raw logistic and oracle ceilings are single models, not counted as three independent fits.

| TEST scenario | Frozen K2 | V28 source | V29 reject | V30 anchor reject | **V30 anchor repair** | Raw 7-sensor logistic |
|---|---:|---:|---:|---:|---:|---:|
| Healthy sensor4/5/6 | 0.7059 | **0.8109** | 0.8078 | 0.8074 | 0.8105 | 0.8186 |
| Only sensor5 sign reversal | 0.7059 | 0.7035 | 0.7059 | 0.7059 | **0.8052** | 0.7826 |
| Only sensor5 +2 offset | 0.7059 | 0.7995 | 0.7059 | 0.7059 | **0.8052** | 0.8186 |
| Both source pair sensors4+5 reversed, anchor6 intact | 0.7059 | 0.5459 | 0.5487 | 0.7048 | **0.8051** | 0.6715 |
| **Only anchor6 reversed**, pair4+5 intact | 0.7059 | **0.8109** | 0.8078 | 0.7064 | **0.5450 (FAIL)** | 0.7308 |
| **All three sensors4/5/6 reversed** | 0.7059 | 0.5459 | 0.5487 | 0.5489 | **0.5453 (FAIL)** | 0.4956 |
| Third factor not predictive | 0.7699 | 0.7699 (gate reject) | 0.7699 | 0.7699 | 0.7699 | 0.7631 |

The oracle TRUE K3 concept logistic ceiling on the predictive-world TEST is AUPRC **0.8189**, NLL 0.5216, regardless of sensor faults (oracle source never corrupted and not a deployable comparator). Even under healthy readings V30 repair is **not** best: raw 7-sensor logistic 0.8186 and the V28 ungated pair 0.8109 both slightly exceed V30 0.8105. No dominant average-performance claim.

The independently calibrated anchor allows a substantial advantage in *declared pair-failure regimes*: for BOTH pair readouts reversed, NLL falls from V29's 0.7740 to V30 repair's **0.5436**, while AUPRC improves 0.5487→0.8051. When ONLY the anchor is reversed, V30 automatically repairs a useful pair toward the WRONG source, making AUPRC drop 0.8078→0.5450 and NLL worsen ~0.5445→0.7782. Both pair+anchor reversed preserves mutual agreement; all schemes fail and forecast cannot infer absolute physical orientation from consistency.

### Coverage and missingness cost (stable source)

| Held-out observation pattern | Frozen K2 AP | V28 pair AP | V29 pair reject AP | V30 anchor repair AP | Fraction of subjects using source with V30 repair | Fraction repaired |
|---|---:|---:|---:|---:|---:|---:|
| MCAR20 | 0.7059 | 0.8109 | 0.8078 | 0.8105 | 100.0% | 3.1% |
| MCAR40 | 0.7044 | 0.7883 | 0.7688 | 0.7798 | 84.3% | 29.1% |
| MNAR | 0.7013 | 0.7848 | 0.7759 | 0.7798 | 61.4% | 15.1% |
| 8-step blocked | 0.6838 | 0.7087 | 0.6977 | 0.7220 | 51.7% | 45.9% |

V30 support/repair rates are FULL 900 TEST subjects, not selectively accepted-only metrics. Under MCAR40 and MNAR the simpler raw-measured logistic baseline still exceeds new source repair in aggregate AP. Source usage and normalization differences do not create new TEST cohorts.

### Research interpretation

**Supported on this new nonlinear controlled generator:** source-only risk contributions can be repaired from a separate observed readout, without ever modifying previously named K2 concepts or their original head. This reduces the destructive false-refusal risk demonstrated in V29, provided that the **extra reference's measurement calibration is externally known and remains trustworthy**.

**Demonstrated failure by experiment, not just conjecture:** a corrupted anchor causes disastrous false repair; simultaneous common-mode corruption fools the consistency check and remains damaging. Thus **statistical concordance and stability DO NOT identify source semantics or establish physical/medical truth**. Repair vs reject must be tied to measurable device provenance, not a universal confidence label.

**Publication and fair comparison limits:** All three third-source readouts share one simulated latent state with independent simulated sensor noise. This is an assumption, not automatic discovered truth. TRAIN labels +1200 extra VALID outcomes and privileged synthetic K2 concept supervision make V30 **not equal-label-/optimizer-budget matched** to published CBM, information bottleneck CBM, causal CBM or residual stochastic CBM (Oct 2026 preprint). Only three NN initializations; TEST AUPRC differences do not have patient/fit-level confidence intervals. The paper must not claim clinical transfer, automatically discovered factors, superiority to published approaches, or a wholly new concept leakage idea. These remain **synthetic falsification results, not a journal-ready novelty proof**.

Unlike accepted PIERE's across-fit attribution reproducibility and IITI-2026's concept-latent alignment, this line studies source admission, conservative forecasting vs evidence repair and the distinction between semantic provenance and mere held-out predictive association.

## Research continuation most likely to improve the journal

1. Provide a **third independent trust anchor with externally certified calibration** or device intervention logs; do not pretend pair agreement can resolve simultaneous common-mode sign flips.
2. A fresh external temporal data family and real protocol with genuinely annotated concept/reference validity; do not open sealed PhysioNet TEST or infer physical concept labels from outcome association.
3. Matched total-label budgets, compute and interventional CBM baselines (Koh 2020, leakage/CTL JMLR2026, IB-CBM ICLR2026, residual stochastic CBM Oct2026 preprint). The correct headline is about *conditions of validity and failure*, not guaranteed superiority.
