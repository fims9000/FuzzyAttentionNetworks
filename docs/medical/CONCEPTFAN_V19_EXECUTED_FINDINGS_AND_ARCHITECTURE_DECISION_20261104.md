# ConceptFAN V19 — executed architecture diagnosis and train-native risk-head correction

**Date:** 2026-11-04. **Scope:** exploratory same-family synthetic benchmark; the accepted PIERE paper remains unchanged. All source scripts, CI runs, 27 + 27 + 36 + 18 + 18 complete scenario denominators and associated per-fit results were versioned under `docs/medical/evidence_v19/`.

## What the code audit established
V10 already constructs signed level, causal recent-mean, first-difference and deviation operators per sensor, and integrates them across learned windows. But its final risk decision reduces each native concept C[t,k] to a **mean of the final six steps** and one coefficient per concept. A strong competing simple logistic predictor used last value, recent level, recent/early difference and variation directly on the historical observed measurements.

It was therefore plausible that lack of shape-aware *readout* caused the predictive gap. V19 implemented an optional, fully native signed eight-operator mixture over concept trajectories (four observable temporal shape functionals plus signed complements), no raw-input bypass. It preserved the original concept trajectories, reconstruction, forecast, exact additive risk logit, physical uncertainty neutrality and V17 expansion behavior. Original V10 mode remains the default.

## V19 / V19b: task readout flexibility is NOT the main bottleneck
V19 from-scratch K3 (27 completed fits) improved AUPRC only very slightly:
- Positive outcome: legacy 0.507381; fixed-shape 0.508038; trained-shape 0.508486.
- Negative outcome: legacy 0.596749; fixed-shape 0.596827; trained-shape 0.596953.
- In the naive from-scratch setup, patient-level signed effect rank correlations remained ~-0.33, not a reproducibility improvement.

V19b matched all K2-pretraining and independent TRAIN-only signed new risk initialization (27 opportunities; 18 fits + 9 null-risk refusals):
- Positive: legacy 0.587819, fixed 0.587598, trained 0.588082.
- Negative: legacy 0.604346, fixed 0.603297, trained 0.604965.
- Both V19b shape arms had identical initial effective logits and native representations before continued training. The learned readout restored some of the initial fixed-shape penalty, but did not give a scientifically meaningful AUPRC advantage.
- Near-perfect signed rank correlation in this simple aligned simulation followed common TRAIN-only source orientation; do not claim general semantic explanation reproducibility.

## V19c: adding arbitrary extra native slots does not fix it
Frozen 3 modes x 3 initializations x 4 arms = 36 fits, all complete. Extra K4/K5 directions came from TRAIN-only SVD of residual measurement signals and are **intentionally unscreened overcapacity controls**, never certified physical discoveries.

| Arm | Positive mean AUPRC | Negative mean AUPRC | Null-mode mean AUPRC |
|---|---:|---:|---:|
| K3 legacy | 0.587819 | 0.604346 | 0.453344 |
| K3 learned shape | 0.588082 | 0.604965 | 0.453371 |
| K5 legacy | 0.589936 | 0.601935 | 0.454366 |
| K5 learned shape | 0.590411 | 0.602284 | 0.454419 |

Capacity-only gain is tiny, reverses sign across outcome modes, and remains far below the raw-feature logistic baseline in positive mode. A generic capacity increase cannot be sold as robust conceptual or predictive improvement.

## V19d: the concept representation still contains predictive information
A diagnostic, TRAIN-only StandardScaler + LogisticRegression was fitted **only to the frozen model's K3 native concept summaries**. This is a diagnostic and NOT a deployable concept head or post-hoc medical explanation. V19d comprises 18 native fits + 18 diagnostic refits (all reported).

| Mode | Native original K3 AUPRC | Diagnostic linear probe on the same frozen native K3 concepts |
|---|---:|---:|
| Positive source | 0.587819 | **0.650960** |
| Negative source | 0.604346 | **0.685645** |
| Outcome-null forced fit | 0.453344 | 0.433570 |

Thus a substantial fraction of the predictive gap originates **downstream of concept formation**, in the current native risk task optimization/regularization/training objective. This is evidence of residual predictive sufficiency, not evidence the concepts represent real biology. Probe train/validation are from the same previously inspected synthetic family, and one cannot distinguish optimizer effects from changed objective weighting on these tests alone.

## V19e: genuine native architecture training improvement
A new opt-in **TRAIN-only second optimization phase** changes ONLY the existing physical decision_raw_weight and decision_bias, using the frozen native physical concepts and the original model's fixed mask-only observation-process contributions. It employs unweighted binary cross-entropy plus preregistered native-effective-weight L2=0.001 and deterministic LBFGS, max 120 iterations; the original trained model is used as initial state.

The *deployed/inference* model remains a single ConceptFAN with **the same exact additive concept-to-logit formula**, not an additional black-box/student classifier. Native concept trajectories, expert-reviewable operator definitions, reconstruction, forecasting, temporal shape readout and care observation head are bit-for-bit unchanged; only native decision parameters are updated. The head-refit helper requires declared TRAIN IDs and rejects attached oracle states. A separate `fit_multiscale(..., native_head_refit=True)` option exposes this phase; the default is off.

V19e GitHub Actions [41834918352](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41834918352) completed 18 native + 18 head-refit training paths. Mean heldout outcomes:

| Native K3 task readout | Positive source | Negative source | Outcome-null |
|---|---:|---:|---:|
| Legacy K3 before refit | 0.587819 | 0.604346 | 0.453344 |
| **Same native K3 after head refit** | **0.650434** | **0.686234** | **0.432619** |
| Learned-shape K3 before refit | 0.588082 | 0.604965 | 0.453371 |
| Learned-shape K3 after head refit | **0.650480** | **0.686081** | **0.432615** |

Head-refit reproduces the *diagnostic concept-only probe performance* to the reported precision, without creating any new inference head or changing concepts. Train native head objective never increased and all 18 exact additive invariants passed. The null-mode ranking AUPRC **worsened**, despite lower/near-flat Brier, warning against unrestricted refit or indiscriminate introduction of risk factors without an independent risk screen.

## Compare to external prediction alternatives
Prior V18c on the **same** TRAIN/validation split:
- Logistic regression on explicit past level/trend/recent variability: AUPRC **0.731434** positive, **0.679452** negative.
- Logistic regression on the full 24×5 history: 0.657644 positive, 0.655570 negative.
- HistGradientBoosting on recent summary: 0.658870 positive, 0.666854 negative.

Therefore V19e has narrowed much of the gap but **still trails simple recent-feature logistic in positive mode**. There is no proven general model superiority. Similar negative-mode AUPRC is only descriptive, not an external clinical result. The logistic comparisons do not test intrinsic concept interpretability or medical validity.

## Scientific attribution, clinical boundaries, and next development choice
- **Supported on this synthetic task:** native concept formation retains predictive signal; the previous joint task/forecast/reconstruction optimization underuses it; a second TRAIN-only refit of the true native decision head is a useful improvement at no added inference complexity.
- **Not isolated:** whether task head underperformance is due primarily to short training time, decision-weight saturation, label class weights, competing auxiliary losses, optimization landscape or architecture. The LBFGS/unweighted refit changes more than one training-design choice; no causal optimizer attribution is justified yet.
- **Not improved much:** V19 shape readout and increasing K3→K5. Keep these experimental options off by default.
- **Not proven at all:** discovery of real clinically/physically meaningful latent factors, robust cross-run signed explanation under ambiguous source/noise, out-of-hospital transfer, irregular ICU missingness, official MAGNETS comparison or statistical significance with only three seeds.
- **Release safety:** V18 demonstrated that a predictive temporally colored *instrument artifact* passes an outcome screen. Prediction relevance is never physical certification. Never label an unseen measured concept as a new disease mechanism on synthetic evidence. In production an unsupported/null risk gate must ABSTAIN; the forced null fit here intentionally demonstrates the harm.
- **Next rigorous work:** isolate native training causes via fixed objective/different optimizer/epoch controls, then improve train-time representation grounding on a genuinely distinct generator and eventually real physician annotations; compare against official intrinsic concept baselines. Do not add V19e results to accepted PIERE camera-ready without approval and new publication review.

## Reproducibility
Code: `src/conceptfan_discovery/temporal_readout_v19.py`, `native_head_refit_v19e.py` plus frozen experiments `experiment_temporal_readout_v19.py`, `experiment_signmatched_temporal_v19b.py`, `experiment_capacity_v19c.py`, `experiment_native_probe_v19d.py` and `experiment_native_head_refit_v19e.py`. Test suites under `tests/discovery/`; each research arm in `docs/medical/evidence_v19/{temporal,signmatched,capacity,probe,native_refit}/` with source checkout SHA, complete CSV and JSON status. PR #16 is deliberately Draft and based on the V18 research branch.
