# V57 frozen protocol: independent measurement-anchor falsification of native ConceptFAN concepts

Registered **2026-11-13 before V57 evaluation outcomes**. Exploratory follow-up to V56, not a replacement for accepted PIERE/IOI papers. No clinical data.

## Scientific question and boundary

V56 learns fuzzy temporal concept states from unlabeled measured histories and uses an exact signed additive supervised head. It did not identify physical sources, and a correlated *common-mode inversion of all three sensors* could mislead its measured-source concepts. That cannot be solved from those three sensors alone when the corrupted values are observationally equivalent to a real shift of their shared cause.

We test the **additional, externally supplied assumption** of a truly independent, calibrated reference sensor observing one physical source with independent measurement noise and a different fault domain. A medical diagnosis cannot be inferred merely by fitting correlation. The sensor's independent provenance must come from outside the algorithm. The ground-truth simulator is used only to *generate* that additional noisy measurement and for posthoc auditing, not supplied to the model, evaluator guard, or optimizer.

## Frozen inference architecture

1. Train the **unchanged V56** automatic TRAIN-only source selector, self-supervised 8-epoch causal fuzzy model, and its additive supervised head on the original twelve measured channels, masks and TRAIN outcomes. No initial K2 map or true latent concept trajectories. Neural seed 11, Adam/learning settings frozen by V56.
2. A separately measured **reference channel** is generated at the same times from synthetic source-3 latent dynamics plus independent Gaussian sensor noise sigma=0.27, observation probability 0.80. It is *never* an input to the V56 source discovery, native fuzzy encoder, task head or raw-12 baseline. The generator may access latent truth to synthesize a physical measurement. The algorithm only sees the resulting new noisy channel, its independent mask, and a declared external contract that the reference came from a separately calibrated sensor.
3. Discover an observational association between the reference and **one** existing measured-source group from TRAIN sensor values only: 3 disjoint TRAIN subject folds, >=250 co-observed time cells per fold, stable signed Pearson |r|>=0.65 all folds. Choose the highest worst-fold |r|. No use of TRAIN Y, synthetic source indices, or hidden factors. If no consistent association, return **ANCHOR_UNMATCHED** and do not claim independently validated concept semantics.
4. Fit a TRAIN-only affine alignment of source-group measured level versus reference value, and a **frozen 99th-percentile** threshold of per-episode *median absolute source-vs-reference residual*, only across episodes with >=6 co-observed times. No VALID/TEST outcomes or measurements in threshold fitting.
5. On TEST, check reference-source agreement. If reference is absent/insufficiently observed, flag **NO_REFERENCE** and conservatively set the *matched* concept contribution to zero; if discordance exceeds TRAIN threshold, flag **DISCORDANT** and set only that contribution to zero; leave all other concepts, coefficients, bias and original V56 weights unchanged. The reference must never enter the logit as a new risk predictor. Exact identity after correction: logit=bias+sum_k corrected_signed_contribution_k. Keep baseline and signed correction auditable.
6. On source-search failure or spectral-only data without a stable group, output **UNVERIFIED** with unchanged V56 logits and do not claim physical alignment. Do not fabricate a source identity.

This is a *reference-consistency gate*, not automatic semantic identification or a treatment-causal model. Even an independent measurement can be improperly calibrated, swapped, spoofed, or co-corrupted.

## New data and tests, frozen before outcome results

Two new nonlinear V30-extended synthetic draws: seeds **20261471 and 20261473**, each TRAIN1400, VALID600 **unused**, TEST700, length16 and 12 original sensor channels. Reference generation RNG seed draw+50001, noise sigma0.27, MCAR20 independent mask. Original sensor TRAIN/TEST MCAR20 with seeds draw+101/+102. V56 native model fitted once (seed11) using original 12 channels. Same TRAIN labels per target law: PHYSICAL_EXTRA, COMPLETE_NUISANCE, ARTIFACT_PREDICTIVE; source bank/reference map fitted once per draw without labels.

TEST scenarios share held-out patients:
- STABLE independent reference and original channels.
- SOURCE_COMMON_SIGN_FLIP: channels 4/5/6 inverted, reference remains correct (actual new assumption under test).
- SINGLE_SENSOR_SIGN_FLIP: channel 5 inverted, reference correct.
- ANCHOR_ONLY_SIGN_FLIP: reference inverted while original channels remain correct, to audit false alarms and trust dependence.
- JOINT_ANCHOR_AND_SOURCE_FLIP: original three sensors and reference inverted together — an observational counterexample.
- ANCHOR_MISSING: reference unavailable, conservative source fallback.
- SOURCE_MISSING: original group 4/5/6 all absent, unchanged baseline already neutral; no invented benefit.

Comparator methods on identical TEST patients and outcome labels: original unchanged V56; V57 reference gate over V56; raw-12 logistic (no reference); raw-12+independent-reference logistic (extra-information budget comparator). Fair caveat: reference gate uses external trusted provenance and therefore is NOT a label/data-budget matched autonomous competitor to the pure raw-12 method.

Predeclared metrics: AUPRC, AUROC, mean NLL, Brier; per-episode gate reject rate, paired single-source altered contribution, exactly preserved unaffected K-1 contributions, source-group match to evaluator-only {4,5,6}, no-reference fallback and no-direct-reference-logit input. Descriptive paired patient bootstrap NLL differences are not independent generator/site confirmation. No test tuning of correlation, fault threshold, support or model.

## Explicit falsifiers

- If the reference never maps to a coherent source, no claim of guard capability.
- If false alarms on clean/test-anchor-only shifts cost more outcome NLL than source fault rescue, report the negative.
- If source and anchor are jointly corrupted, guard can be fooled. This is a necessary negative result, not a code bug.
- A correlated recording-process group can remain outcome-predictive. One independent physical reference does not prove all other discovered sources are biological.
- All simulated independent noise is a design assumption, not a real-world validation of provenance. Clinical source meaning requires external calibration records, instrument controls and independently measured reference concepts.
- Same-family seeds 20261471/20261473 cannot establish cross-generator generalization; independent clinical data remain untouched.

## Prior work boundary

This does not invent independent sensing, sensor fault diagnostics, reliability gating, correlation/regression, or fuzzy concept bottlenecks. It tests whether these established controls enforce a *strict concept-level provenance firewall* inside V56 without raw-to-risk bypass, and quantifies exactly what assumptions are required for a common-mode guard.
