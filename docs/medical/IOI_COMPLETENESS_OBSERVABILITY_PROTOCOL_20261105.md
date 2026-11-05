# IOI-2026 -> Elektronnye biblioteki: independent completeness experiment

Research-only prototype. Created 2026-11-05. This document is NOT a claim of a validated journal-ready method.
The accepted PIERE-2026 manuscript and its frozen test protocols must remain unchanged.

## Publication separation

- IITI-2026: concept-latent alignment and FAN concept aggregation.
- PIERE-2026 UID 6530: exact per-model concept contributions versus cross-retraining reproducibility, 30 fits, 435 model pairs on Med-CircuitBench/PhysioNet.
- Planned IOI journal: **which type of evidence is missing from a fixed concept dictionary**, and **when an explanation lacks adequate measurement support**. Do not present previously published cross-run correlations as a new result.

## The methodological distinction

Two failure modes must not be conflated:

1. Concept estimation error: true known concepts are sufficient, but a model estimates them incorrectly.
2. Dictionary omission: even with the known concept values supplied exactly, another separately observable candidate carries predictive information about the outcome.

For synthetic data only, use the oracle concept trajectories C*, model estimates C_hat and a candidate residual witness R. Fit frozen, identically regularized diagnostic models on TRAIN only and evaluate once on a disjoint TEST:

- Diagnostic A: Y from C*.
- Diagnostic B: Y from [C*, R]. The positive TEST log-loss improvement of B over A is **conditional predictive evidence**, not a proof of a new semantic concept. Test for model misspecification and a predictive instrument artifact.
- Diagnostic C: Y from C_hat.
- Diagnostic D: Y from [C_hat, C* - C_hat]. Improvement of D over C measures correction of estimation error, not discovery of an omitted concept.

The source features R must themselves be constructed without TEST outcomes. If available oracle C* is synthesized or unblinded for training or TEST-model selection, invalidate the confirmatory run. On real ICU data, C* is unknown: a residual diagnostic is associative and CANNOT partition these two failure mechanisms without additional assumptions.

The `audit_oracle_concept_coverage` routine implements these four independent diagnostic fits, a disjoint TRAIN/TEST guard and paired bootstrap intervals. Its internal logistic model may be misspecified; robustness against richer baselines and external source provenance must be assessed before drawing a claim about dictionary omission.

## Optional observation-aware ConceptFAN architecture

`ObservationAwareConceptFAN` wraps the pre-existing `MultiSetAdditiveTemporalConceptFANModel`. It requires a fixed concept-to-observed-channel support map defined without TEST labels. The encoder sees zero-filled readings plus binary observation masks; no raw-input-to-risk shortcut is added. Temporal pooling weights are multiplied by available measurement support per concept; no-support columns receive zero weight.

When a concept has insufficient observed evidence, its native signed contribution is structurally zeroed using `concept_mask`, and the output raises `abstain_from_explanation`. If all concepts are unsupported, the native logit collapses to its bias and the result MUST NOT be used as a reliable clinical prediction. The exact native additive logit invariant is asserted. Importantly, this gate verifies only *measurement support*, not *dictionary completeness* or *physical meaning*.

This is a limited engineering hypothesis, not an invention of missing-data attention or a theoretical completeness guarantee. Earlier V10 work also includes explicit observation reliability. The new contribution would have to come from a controlled paired evaluation of **selective explanation coverage** on the PIERE-type fixed concept pathway and from distinguishing estimation error from omitted evidence.

## Frozen next evaluation (not yet run)

1. Keep existing Med-CircuitBench v3 synthetic benchmark and PhysioNet 2012 splits sealed. Do NOT relabel already inspected patient TEST as fresh.
2. Define a genuinely new, preregistered synthetic generator with two outcome mechanisms: known-complete concept dictionary versus one omitted dynamic source. Add a predictive instrument-artifact negative control with matched observed statistics.
3. Fit matched ConceptFAN cores with equal seeds/budgets and the same mask channels: (A) ordinary temporal pooling, (B) observation-weighted pooling plus support/abstention gate. A raw-input baseline and a concept-only logistic should also be evaluated. Include all seeds and failed runs.
4. Corrupt only evaluation inputs with prespecified MCAR 10/20/30%, state-dependent MAR and state-dependent MNAR, plus episode block gaps. Fix the exact corruption generator and sensor support map on TRAIN only.
5. Compare overall and selective AUPRC, AUROC, NLL/Brier, coverage, false-support/false-abstention rates, exact additive logit error and drift in signed contributions; report the accuracy-coverage curve. Only interpret contribution drift on the covered/common-patient subset, without hiding refusals.
6. Report uncertainty using the true independent unit (patient, independent training fit); 435 overlapping model pairs are not 435 independent samples. Calibrate and select thresholds before TEST.
7. On PhysioNet, distinguish heuristic soft proxies, missingness/observation policy and physical concepts. No clinical, causal, or external-site claim without clinical provenance, expert labels and new external cohort.
8. Check comparisons against Yeh et al. 2020 completeness-aware concepts; CT-CBM (EMNLP 2025); DCBM (TPAMI 2025); ICC leakage metric (ICCVW 2025); and the 2026 CBM surveys. Do not describe this architecture as categorically novel absent a literature-matched experiment.

## Preliminary sanity evidence (not scientific validation)

A 4200-episode Gaussian/logistic simulator with an oracle concept dictionary has two controlled outcome conditions. 2800 episodes for fitting diagnostic probes, 1400 untouched for this sanity TEST. Bootstrap n=999, fixed seed=31415.

- Complete dictionary with noisy predicted concepts: oracle-residual gain = -0.000066 (95% paired bootstrap [-0.000171, +0.000044]); restoring true concepts from predicted ones yields 0.1100 lower log-loss.
- An omitted outcome-relevant simulated source: oracle-residual gain = +0.1640 (95% paired bootstrap [0.1410, 0.1887]).
- A predictive instrument artifact with the same observational law produces the same gain as the simulated physical source by construction. Therefore the oracle-residual test **cannot infer physical source identity**, even though it may detect a missing predictive factor.

These controlled numbers are implementation sanity checks on trivial constructed laws, not Med-CircuitBench or PhysioNet benchmarks and not sufficient for publication.

## Verification and boundaries

Local execution: `PYTHONPATH=src pytest -q tests` and `PYTHONPATH=src python scripts/run_oracle_coverage_sanity.py`.
The original repository's full `MultiSetAdditiveTemporalConceptFANModel` integration test is intentionally included; when running the standalone prototype, it skips because the old repository modules are absent. Run it in the real checkout before describing the new wrapper as integrated.

No existing experimental PR, preaccepted camera-ready manuscript, or frozen artifacts may be rewritten. Do not merge into default branch until the matched experiments are complete and reviewed.
