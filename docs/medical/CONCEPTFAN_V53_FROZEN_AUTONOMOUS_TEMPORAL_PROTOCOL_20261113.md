# V53 — autonomously discovered temporal measurement concepts: frozen experiment

Protocol date: 2026-11-13. This is a NEW exploratory extension of the nonmerged automatic-measured-concept branch, not a replacement for published PIERE, IITI, or IOI results. Register this protocol BEFORE examining V53 TEST outcomes.

## Primary unresolved problem

The preceding automatic measured-source pipeline clusters only sensors 4–11 and assumes the first two K2 concepts are defined by humans as sensors (0,1) and (2,3). PD-ConceptFAN v1 used learned thresholds and learned K gates, but K collapsed or became unstable; V3 could recover detectable covariance rank on synthetic systems, not stable individual concept identities. V34 Type-3/Flow did not yield defensible empirical gain.

The new question is whether an *entirely train-derived concept vocabulary* can be used as a strict, exact additive decision bottleneck and expose **temporal changes**, without ever seeing an oracle concept trajectory, a target outcome, or a manual source group during concept construction. Predictive or medical/physical semantics are NOT automatically identified by this.

## Frozen source-construction architecture

1. INPUT all 12 measured sensors, length 16, with separate binary observation mask; no known/grouped channels. TRAIN ONLY, no outcome/true source labels, VALID or TEST data. The existing signed 3-fold complete-clique correlation proposal operates on all available sensors, fixed abs correlation threshold 0.60, >=300 jointly observed cells per fold, maximum 5 groups. It is an established correlation clustering strategy, not a new discovery theorem.
2. For each detected group form a *group-level per-time measured signal*, averaging aligned TRAIN-standardized actually observed readings only. A fully absent group must be neutral (0.5) and contribute exactly zero logit. Source weights and signs are train-only, transparent and fixed.
3. Fit a signed AR(1) coefficient on consecutive observed TRAIN group signals. This yields a one-step prior for each detected source. Absolute one-step residuals are the second *time-localized* concept family. The first family encodes continuous source state; the second encodes deviations from the learned transition. TRAIN-only empirical medians and quantiles set fuzzy activation thresholds and widths. Dynamic event start/end are defined by activation >=0.65 for >=2 consecutive observed steps.
4. Form two groups of concepts per stable source (state, innovation), and optionally omit the innovation family as a controlled ablation. The final model is a strict additive logit: b + SUM beta_k * (mean_t concept_k(t) - 0.5). No raw-input-to-label bypass; each beta is fitted on TRAIN binary labels AFTER the definitions are frozen.
5. Form an optional measurement disagreement safeguard: TRAIN-only upper 99th-percentile cross-sensor disagreement, reject an individual *source* when on-test evidence is visibly inconsistent across its sensors. A common-mode sign flip can remain observationally indistinguishable; this is an explicit negative control, not a promised fix.
6. Concept vocabulary size = 2 * K_stable for the temporal model, or K_stable for state-only. This is the number of *observable group-operator slots*, not the number of genuine physical causes. Do not invent singleton sources. No oracle K may influence selection.

## Registered experiment and baselines

- Two fresh draws of the V30 EXTENDED NONLINEAR Laplace switching generator with seeds 20261221 and 20261237. Each TRAIN 1400, VALID1=300, VALID2=300, TEST 700. Same 12 channels and 16 steps, with five relevant/non-relevant source groups and singleton distractor only for external evaluation.
- Three outcome laws on the same cases: COMPLETE_NUISANCE, PHYSICAL_EXTRA, ARTIFACT_PREDICTIVE. All group discovery and calibration parameters are identical across the three outcome laws and do not depend on Y.
- TRAIN missingness MCAR20. VALID1/VALID2 each have independent MCAR20. TEST regimes: complete, MCAR20, MCAR40, MNAR, BLOCK8; test-only counterexamples: single flipped channel 5, all source channels 4:6 flipped together, all source 4:6 missing, artifact channels 9:10 flipped together. Test masks are generated independently and never used for fitting.
- Compare exactly the same TRAIN labels: **state-only**, **state+innovation**, **state+innovation+disagreement guard**, raw measured history logistic, PCA(K) of raw measured history features + logistic; no oracle given to operational models.
- Group alignment to synthetic ground truth may be performed only after all methods' predictions are frozen. Report group purity, missed groups, spurious sources, observed K and operator count, and per-source temporal event counts. A source correlated with ARTIFACT target is **not** a medical/physical discovery.
- Report full-cohort TEST AUPRC, NLL, Brier, group-level support, per-source ablations, exact signed logit reconstruction and fallback on total absence. Paired TEST-subject bootstraps are DESCRIPTIVE for a pre-inspected model family, not independent model replications; independent draws, not network seeds, are the sampling unit for generator-level generalization.
- No use of VALID/TEST outcome labels for hyperparameter selection, source definition, calibration, threshold selection or risk fit. No patient data.

## Hard falsifiers

- If the full concept bank has no stable groups, it must ABSTAIN instead of manufacturing latent variables.
- If adding dynamics lowers AUPRC/NLL relative to state-only or raw logistic, preserve the negative result; neither classification advantage nor causal source discovery is guaranteed.
- If a single-sensor flip is not detected, retain that negative; if a common-mode flip escapes consistency checking, demonstrate the nonidentifiability boundary explicitly.
- In ARTIFACT_PREDICTIVE, a source may be predictively useful but must remain measurement-association-only, never named medical.
- Run an independent *different observation-family* counterexample with overlapping/mixed sensor loadings when feasible; correlation cliques can fail. This should be reported separately, without changing original thresholds.

## Scientific novelty boundary and related work

This is an engineered synthesis, **not an automatic original invention** of concept bottlenecks, prediction-error temporal segmentation or latent dimension inference. Explicit comparisons needed:
- Forest, Wei, Fink (2026), MAGNETS, DOI 10.1007/s10618-026-01267-y: annotation-free learned temporal concept masks.
- Abeywickrama et al. (2026), ConceptTime, arXiv:2611.12955: predictive surprise, segment boundaries and prototype concepts.
- Existing PD-ConceptFAN V1-V9 and automatic measured-source V30 are internal precursor baselines.

The proposed falsifiable addition is an all-sensor label-free source vocabulary with train-fitted fuzzy state/innovation operators, missingness-aware abstention, exact additive signed prediction, and explicit single-/common-mode observational counterexamples. It does NOT identify a physiological entity, restore natural causes, or justify deployment.
