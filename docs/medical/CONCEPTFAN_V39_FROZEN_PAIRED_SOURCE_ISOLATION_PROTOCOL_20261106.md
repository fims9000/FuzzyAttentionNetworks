# V39 — FROZEN BEFORE ANY NEW TEST: active paired source isolation vs unmonitored source-aligned artifact

Date: 2026-11-06. Parent: audited V38 [PR #35](https://github.com/fims9000/FuzzyAttentionNetworks/pull/35). **Status:** preregistered mathematical falsification and independent synthetic study. No V38 inspected TEST is reused to select parameters, thresholds, models or the new sample cohort.

## Why another experiment?

V38 produced an exact observational equivalence counterexample: identical observations can arise from real latent source fluctuation or from an instrument artifact perfectly aligned with the source-loading direction in all three views and invisible to the negative-control monitor. Thus no observational Transformer, SOBI rank criterion, Type-3 fuzzy grade, Flow, known earlier actuator certificate, or internal sensor agreement alone can disambiguate.

**V39 adds NEW information at inference**, not a more confident classifier: two **paired contemporaneous instrument states**, source path ON and source path deliberately SHIELDED/OFF, with the same nuisance artifact. This is a strong experimentally controlled assumption requiring special hardware, not fully automatic discovery on routine passive data. A clinician cannot safely block arbitrary physiological sources; V39 is NOT a clinical protocol. Its purpose is to test and honestly delimit an *operationally identifiable* source-dependent signal against a previously indistinguishable device artifact.

## Algebra and scientific priority

For each view j in A,B,C with unknown two-by-two mixing H_j and source vector S(t):

- source-coupled measurement: X_on,j(t) = H_j S(t) + H_j[:,0] U(t) + epsilon_on,j(t);
- simultaneous source-isolated reference: X_off,j(t) = H_j[:,0] U(t) + epsilon_off,j(t).

Here U(t) may have identical autocorrelation / temporal shape as the slow physical source, may be arbitrarily strong, may be invisible to independent instrument monitor10, and may shift between TRAIN and TEST. But **the artifact is stable between source-on and source-off measurements from the SAME episode**. Then paired difference

D_j(t) = X_on,j(t) − X_off,j(t) = H_j S(t) + (epsilon_on,j − epsilon_off,j)

cancels even source-aligned nuisance. No H, S, or U is needed to compute D. This is elementary common-mode/reference subtraction, not a new theorem, not a general solution to blind causal representation learning. The independent view noise SD increases by sqrt(2) under subtraction. Existing V37 centered cross-view lag covariance and source power may be fitted to D with masks M_D=M_on∧M_off, with the same fixed TRAIN-only eigengap/cross-energy/agreement thresholds; source still only resolved up to sign/scale/permutation, and is **named only as an operationally source-isolated slow fluctuation mode**. Source response is NOT an event causal effect or true clinical identity.

**Critical negative cases:** reference artifact changes between paired acquisitions, X_off obtains a fraction of the true source because isolation leaks, OFF reference is absent, source-on polarity changes, or both ON/OFF are contaminated by independent artifacts. No algebraic cancellation guarantee in these worlds. If reference unavailable, refuse and return EXACT frozen K2. Positive results require faithful source blocking and matched nuisance, externally verified; neither condition follows from sensor histories alone. The ONLY reason V39 can separate the old indistinguishable worlds is that the new paired OFF observation is different between them.

## Frozen synthetic experiment

Two NEW independent random full-rank mixing devices (seeds 20261317, 20261319) and new synthetic patient samples, never V38 or V37/36 TEST. For each: TRAIN1000 + VALID1=400 + VALID2=400 + TEST600, 16 timestamps, 12 channels; source AR(1) slow rho=.77, fast rho=-.27 with per-episode positive gains. Each view A=(4,5), B=(6,7), C=(8,9) has two unknown mixed sources; noise sd=.19 independent per view and per source-on/off state. Unmonitored source-aligned nuisance U is AR(1) rho=.77 with TRAIN amplitude≈.35. External source path is **simulator-controlled known switch**; no latent factor oracle H/S given to model. Channel10 negative control does NOT observe U by construction, so V38 sentinel cannot solve this world.

Physical target Y depends on true demeaned slow-source energy plus the known K2 sources; same Bernoulli draw per subject across K2-complete and predictive-source worlds. Original native K2 Temporal ConceptFAN trains concept-only on known sensor0..3, no Y gradient into encoder, 10 epochs, 3 model seeds11/23/37 per device; K2 frozen. New source q and signed gamma learned only from masked TRAIN paired measurements/labels and approved separately by two disjoint VALID cohorts using unchanged V37 dual-environment admissibility thresholds (\`min_mean_gain=.003\`, \`min_support_fraction=.25\`, \`min_subject_support=.5\`, 399 bootstrap reps).

**Test method arms** use the same named sample IDs:
1. frozen native K2;
2. V37 centered ON-only AB;
3. V37 centered ON-only ABC with three-view agreement (without source isolation);
4. **V39 paired (ON − OFF) centered AB**;
5. **V39 paired (ON − OFF) centered ABC**;
6. ordinary temporal linear/logistic on observed ON data;
7. ordinary quadratic-feature logistic trained on ON and OFF paired differences, with all 1800 available outcome labels for that law.

The published raw baselines are stronger supervised comparators but not matched in optimizer compute; not faithful reproductions of published interventional CRL or CBM methods. No original H/S in source fitting, only evaluation audit correlation with latent source energy after TEST.

**10 TEST worlds** using the *same* subjects: healthy matched artifact; unseen 4× stronger aligned nuisance shared ON/OFF; still-stronger 8× aligned nuisance shared ON/OFF; false reference where OFF-only nuisance shifts (undetected), reference source-path isolation leakage at 50%, missing reference, one-view A polarity flip in ON only, common offset +2 on ON/OFF, K2-complete outcome law, known VALID2 reference drift. For each world evaluate at least MCAR20, MCAR40, independent block8 and synchronized block8 on ON views, plus missing all OFF and missing view C. Reference OFF view MCAR05 under normal conditions; dynamic missingness additionally applied for reference-missing fault. All source mask decisions must use **actual measured times in both ON and OFF**, and never count imputed values.

**Frozen diagnostic thresholds:** old V37 centered 3-pair lag eigengap ≥ .25, q disagreement ≤ max(.12, TRAIN-calibrated 95th percentile), source cross-energy ≥ .10 raw, each pair observed at least 4 timestamps, independent TRAIN 24-episode bootstrap eigencos 10th percentile ≥ .80. No new threshold or mode selection from outcomes or TEST.

Report: AUPRC/AUROC/NLL/Brier, support/coverage, instrument-reference refusal rate, per-example explanations with evidence ON/OFF provenance, signed gamma and original/expanded model risk, exact K2 fallback when OFF reference missing; 2 separate new devices×3 real native K2 Transformer seeds; saved individual TEST probabilities/checkpoints; at least 18 preregistered paired-subject bootstrap AP contrasts with 399 resamples per device. Statistical intervals are conditional on the two simulated devices and six fixed network fits; not device-population CIs.

## Prediction of failure modes (no retroactive tuning)
- In healthy paired matched instrument nuisance, V39 ON–OFF is a more defensible estimator of true slow source fluctuation energy than ON-only. This does not guarantee higher AUPRC than strong quadratic supervised baselines.
- Under extreme aligned nuisance *identical* in ON and OFF, paired difference may retain near-healthy performance; ON-only may fail. This is the **main falsifiable claim** of reference subtraction in this designed model.
- OFF-only nuisance drift violates paired artifact invariance: V39 can be wrong. Do **not** hide failure or claim self-diagnosis without another independent reference.
- Isolation leakage weakens/removes source; V39 may fail. Without a verified OFF switch source absence is not proven.
- Without OFF reference, V39 must reject and EXACTLY preserve K2, even if ON-only predicts well. This is the safety/coverage cost.
- Both reference and external actuator identity are physically **assumed by the experimental design**. The method is not fully unsupervised in physical semantics.
- A new real-data domain, different noise families and matched compute baselines are required before asserting superiority.

## Literature and novelty boundary
The need for interventions for latent causal representation identification is established by Varici et al. *General Identifiability and Achievability for Causal Representation Learning*, AISTATS 2024, https://proceedings.mlr.press/v238/varici24a.html and *Score-based Causal Representation Learning: Linear and General Transformations*, JMLR 2025 https://jmlr.org/papers/v26/24-0194.html. Grouping alone may suffice under separate conditions: Morioka & Hyvärinen, *Causal Representation Learning Made Identifiable by Grouping of Observational Variables*, ICML 2024 https://proceedings.mlr.press/v235/morioka24a.html. None of these published results is rederived or claimed as ours. This is a **narrow engineering/scientific test** of when external paired reference observation changes our ConceptFAN ability to infer an energy feature and when it cannot.

No public/patient DATA and no master merge; PR remains DRAFT.
