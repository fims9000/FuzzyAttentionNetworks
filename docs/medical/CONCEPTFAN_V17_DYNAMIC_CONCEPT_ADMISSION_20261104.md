# ConceptFAN V17: new candidate discovery and safe native slot admission

Status: exploratory architecture module, separate from the accepted IEEE PIERE manuscript and from the completed V16 feedback experiment.

## Parent finding that motivates this change

In V16, correct simulated reviews improved the heldout observable-pattern AUROC from 0.8162 (identical bridge-only architecture) to 0.9267 but did not make signed patient-specific logit effects reproducible (approximately -0.33 across independent seed pairs). V17 therefore focuses on a different, missing capability: **proposing new measurable candidate concepts beyond the present fixed slot dictionary**, without pretending this certifies clinical decision influence.

## Mechanism

1. On complete TRAIN histories, remove the existing model's observable sensor subspace, then test whether the **remaining** sensor dynamics contains a statistically stable positive lag direction.
2. Use patient-episode exchange permutations for a finite-sample null (99 permutations minimum) and 24 or more episode bootstraps for direction stability. Candidate count remains one per call. Low-signal and unstable cases **ABSTAIN**, rather than invent a factor.
3. Store a versioned candidate with signed observable sensor loadings, numeric diagnostics, a short ID, and flags `physical_cause_identified=False`. No automatic physiological names.
4. A distinct concept-level approval decision (`approve/reject/unsure`) gates promotion. Rejection and uncertainty never silently become training negatives.
5. On approval, instantiate a new V10 dual-process model with K+1 actual, differentiable concept slot. Copy all old native logits, thresholds, time-scale weights, dynamics, readout parameters, observation and future decoders, and observation-process head. Set the **new slot's direct risk/forecast/reconstruction readouts to zero**, so the current predictions and existing explanations remain unchanged before a new explicit training stage. Subsequent training may assign the new slot predictive weight.

The candidate direction is **not a free latent black box**: it comes from a signed loading of recorded sensors and becomes a native causal-past concept operator after admission. It also does not constitute a proved medical mechanism.

## Scope and failure modes

- This first proposal algorithm **requires full sensor measurement coverage**; irregular clinical masks cause ABSTAIN, since zero fill-in can manufacture lag structure.
- A temporally correlated sensor defect aligned with an unmodeled direction is observationally indistinguishable from a genuine persistent source under this method. The explicit colored-noise experiment is a **false-physical-meaning counterexample**, even if a perfectly stable candidate is proposed.
- Known residual source dynamics with negative autocorrelation, nonlinear interactions and time-changing mixing may abstain; no claim of general source recovery.
- An expert's `approve` decision is a user control for admission, not independent certification of physical semantics, and a real clinician did not participate in these tests.
- A newly added, zero-risk slot does **not yet contribute to prediction** until explicit subsequent training; preserving predictions before consent is intentional.
- Falsification run uses three synthetic seeds (2507/2607/2707), three scenarios per seed (genuine extra dynamic source, white sensor noise only, temporally colored sensor noise only), complete recorded histories, TRAIN-only data. All nine proposal/refusal decisions must be reported. Hidden-source oracle only for evaluation after fitting.

## Scientific acceptance

- All original V10 predictions, additive logit decomposition, reconstructions, future forecasts and observation-process contributions must match to floating precision immediately after approved admission.
- The actual new slot enters the native decision graph after nonzero readout training; it must not be a separate post-hoc explanation.
- Correct additional factors should yield proposals under the planted linear high-SNR generator; white independent noise should usually be rejected. Colored sensor-only shifts should demonstrate a possible false certificate, explicitly reported.
- Future work must test independent measurement views, irregular physiological time series, reviewer reliability, concept duplicates, cross-retraining effect direction and novel concept relevance to outcome. **These are not proven here.**

Code: `src/conceptfan_discovery/candidate_admission_v17.py`, synthetic study: `src/conceptfan_discovery/experiment_candidate_admission_v17.py`; tests in `tests/discovery/test_candidate_admission_v17.py`.
