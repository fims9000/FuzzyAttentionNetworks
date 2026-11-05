# IOI ConceptFAN V25: frozen trained-network omitted-factor falsification (protocol before execution)

Date: 2026-11-06. Branch: `experiment/ioi-completeness-observability-20261105`. This protocol is fixed **before** any V25 training/evaluation results are inspected. No PIERE camera-ready, earlier V10–V23 protocols or previously inspected patient TEST labels may be edited.

## Research claim and falsification criteria

An observed residual correlated with the outcome does **not** demonstrate concept-dictionary incompleteness; it may result from imperfectly estimated existing concepts, task leakage into concept units, model misspecification or a predictive instrument artifact. With synthetic oracle concept states and a fixed held-out diagnostic family, ask whether extra sensor evidence predicts the outcome *conditional on true specified concepts*. A positive test result is merely **conditional predictive incompleteness**, not identification of a physical factor.

### Three prespecified generators, same sensor law across all arms

A 16-step AR(1) latent three-factor process, 6 noisy sensor channels (two fixed channels per source), true trajectories `sigmoid(s_t)`, binary outcome sampled from one fixed sigmoid risk law, and independent TRAIN/CALIBRATION/TEST draws. Original all-six-sensor recordings are fully available; MCAR/MAR/MNAR/block omissions are artificial.

- **COMPLETE**: only first two true concept trajectories cause outcome; third observable factor is a nuisance.
- **OMITTED**: outcome additionally depends on a third hidden dynamic prognostic factor, not in the prescribed K=2 concept dictionary.
- **ARTIFACT**: *exact same observed sensor and label arrays as OMITTED* under a different causal/provenance interpretation: third observed signal is an instrument factor tracking an unmeasured source correlated with prognosis, not a certified physiological concept. An observational classifier cannot distinguish OMITTED and ARTIFACT. Verify bit-exact equality.
- This is an *identifiability negative control*, not a causal SEM inference theorem or a medical simulator.

Split fixed generator seed 20261106: TRAIN=1600, calibration=400, sealed TEST=800 independent episodes. No TEST outcomes used for training, selection, thresholds or normalization. Neural model seeds exactly 11,23,37. All arms share the same 16-step x and outcomes within each generator family; OMITTED/ARTIFACT are the same observational dataset by design. All simulated measurements and latent factors are synthetic; no ICU patients are involved.

### Prespecified trained models (same training episodes, epochs, optimizer, initialization and data order)

Use the **canonical repository** `MultiSetAdditiveTemporalConceptFANModel` with 1 Transformer encoder layer, latent size 24, two heads, FFN 48, dropout 0, 3 sigmoid memberships per concept, `alpha_mode=no_alpha`, attention pooling, BCE + concept trajectory MSE `lambda=5`, AdamW LR=0.002, batch128, 14 fixed epochs, gradient clipping1.0. Concept trajectory labels are privileged *synthetic ground truth provided on TRAIN only* (as in synthetic supervised CBM controls); a real dataset cannot provide this supervision.

(A) native K=2 trained encoder-to-concept-to-additive-logit without observation-aware pooling; input includes masked sensor values and observation bits.
(B) same K=2 native architecture inside `ObservationAwareConceptFAN`; fixed externally supplied K×6 sensor support, reliability-weighted temporal pooling, exact zero contribution for unsupported concept. Prespecified threshold `min_observation_fraction=0.65` (not optimized on TEST).
(C) native K=3 additional-slot *oracle-guided positive control* trained with all three synthetic factor trajectories, including the instrument-artifact variable in the negative control. **Not automatic concept discovery**. This is a ceiling/sufficiency demonstration, not a fair model for real clinical deployment.
(D) external logistic regression on raw available time/channel means, per-channel observation fractions and late observation means, trained on identical labeled TRAIN only; refitted independently per dataset family, no oracle synthetic factor labels.

Include both COMPETE and OMITTED trained fits; ARTIFACT evaluation reuses the exact OMITTED trained weights, predictions and masks and reports their equality, rather than creating fake independent trials. Fixed train mask MCAR20; entire native risk path and all contributions stay concept-only with no encoder-to-risk bypass.

### Held-out TEST conditions, all applied to identical subjects

`complete`, `MCAR10`, `MCAR20`, `MCAR40`, `MAR` (conditional on always-observed channel0), `MNAR` (dependent on the unobserved value itself), `block8` (contiguous 8-step hidden block per sensor). Each corruption mask fixed by TEST generator seed, independent of outcome labels, with recorded original mask kept unchanged; never resample to obtain favorable results. Report cohort size and positives per stress arm.

### Prespecified diagnostics

- Overall and conditional-on-accepted predictions: held-out AUPRC, AUROC, NLL and Brier; ALWAYS include coverage and refusal count. Rejection is not a prediction improvement. Include common-patient matched comparisons between A/B and raw-input baseline. V24 readout is NOT justified as a native clinical certificate.
- Max exact additive-logit decomposition discrepancy for each learned model; concept trajectory MSE versus oracle on synthetic TEST; no unsupported concept should directly contribute.
- Use existing `audit_oracle_concept_coverage`: train-only logistic probes `Y | C*_1,C*_2` vs `Y | C*_1,C*_2,R_measured` where `R_measured` is third sensor-pair aggregate and `C*` is TRUE first two concept means. Separate `Y|Chat` vs `Y|Chat,C*-Chat` for estimated-concept reconstruction failure. Paired subject bootstrap 499 with exact TRAIN/TEST split. A positive residual result in COMPLETE is a falsification of this simple probe's specificity. The second estimate-error test can reflect risk-specific leakage when dictionary omitted; do not present it as pure estimation-error evidence outside COMPLETE.
- Report honest failure cases, abstention and calibration degradation. No selective editing or new retrospective TEST tuning. Three network seeds alone do not establish broad statistical generalization; do not treat repeated within-test masks as independent patients.

### Pass/fail interpretation

- If COMPLETE oracle-residual gain is indistinguishable from zero while OMITTED shows positive gain, a *synthetic conditional predictive signal* is demonstrated.
- If OMITTED and ARTIFACT are observationally indistinguishable, **semantic physical identification fails**; this is a required negative-control result.
- A trained K=3 may use the additional oracle-supervised factor better, but it does not prove automatic discovery.
- K=2 predicted concepts may leak third factor despite supervised alignment; explicitly report concept fidelity and avoid assuming unit names certify concept meaning.
- Any partial predictive advantage of observation-aware wrapper B is restricted to the exact generator and corruption, and must be paired with coverage and calibrated metrics. Failure or worse outcomes remain in the report.

Results should be stored as immutable aggregate machine-readable GitHub Actions artifacts, keyed by commit SHA. No synthetic result is relabeled as PhysioNet or Med-CircuitBench clinical evidence.

## Stop condition

This is *experimental research*, not certified medical assessment, theoretical source identification or a journal-ready novelty proof. The output should be reproducible and falsifiable; final scientific claim remains bounded to conditional predictive completeness under an oracle dictionary.
