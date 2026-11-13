# V55: adaptive, label-free ConceptFAN concept-vocabulary routing

**Frozen design dated 2026-11-13, before V55 TEST and development-VALID outcomes.** This is a new exploratory integration after inspecting prior V53 and V54 experiments. All data synthetic. It does not revise PIERE, IOI, IITI camera-ready results.

## Scientific question

Can a single inference/runtime API automatically decide whether a TRAIN measurement set supports an interpretable **measured sensor-group** concept vocabulary or only an **unnamed observable predictive subspace**, with strict signed additive risk prediction and no manually supplied original K2 concepts? Does it retain meaningful predictions in both sparse source-group and overlapping mixed-measurement regimes?

## Precommitted choice

Only TRAIN observed sensor values and binary availability masks determine the branch. No Y, hidden planted K, concept labels, clinical names, VALID or TEST may affect routing.

1. Fit the existing V54 full-sensor, three TRAIN-fold, 0.60 absolute signed-correlation, clique-confirmed robust source groups. They require >=300 coobserved cells per fold and never include unsupported singletons.
2. Take the **measured source-group route** iff there are at least 2 distinct valid groups AND their union covers >=50% of observed channels. Use V54 median source states for groups with >=3 sensors; existing measured means for groups with only 2 sensors; only observed values enter. Fuzzy thresholds and widths come from TRAIN empirical group distributions.
3. Otherwise take the **observable spectral subspace route** using the existing train-only positive-lag rank algorithm (lags 1,2,3; 60 complete-future-patient permutations; 20 patient bootstraps; 95% familywise spectral cutoff). Signed source axes are not physical identities. If no rank is detectable, produce the exact bias-only prevalence baseline; do not invent concept slots.
4. Once chosen, freeze all source definitions and fit one additive logistic risk head on the same TRAIN binary labels. Every result is b + SUM beta_k*(mean_t c_k - 0.5). No raw input bypass. Expose route, source definitions, observed support, weights, exact decomposition error, and named temporal events of statistical activation.
5. Full masked source inputs are neutral 0.5; hidden sensor payload changes may not affect prediction. An unexpected common-mode sensor flip may still harm predictions; no semantic/causal/source provenance is inferred.

Routing is deterministic from the same TRAIN observed measurements. The 50%/2-group criterion is a frozen engineering capacity threshold, not an identifiability theorem; robust group-vs-spectral preference has NOT been calibrated by external sites.

## Frozen end-to-end experiments

A. **Independent group-sensor family from V30 nonlinear-Laplace generator**, new seeds 20261331, 20261333; per seed TRAIN 1400, unused VALID 600, TEST 700, 16 time steps, 12 channels. Source grouping/branch selection uses only TRAIN MCAR20. For each outcome law PHYSICAL_EXTRA, COMPLETE_NUISANCE, ARTIFACT_PREDICTIVE, fit the same protocol on the identical TRAIN labels and evaluate same TEST cases under MCAR20: stable, single channel-5 flip, common-mode 4:7 flip, and source 4:7 missing. Compare auto-route with prior mean source-state V53, old whole-group guard V53, and conventional raw measured-history logistic. If automatic route disagrees with measured groups on this source family, report it, do not force it.
B. **Different independent variable-factor dynamical family**: fresh seeds 20261373+K, planted K in (3,5,7) used ONLY in post-model evaluator. 960 episodes, 8 overlapping signed channel loadings, 36 input steps; leave original sealed TEST untouched, train on TRAIN and score on development VALID only. Compare auto-route with V53 group-only fallback and measured-history raw logistic; audit detected predictive rank with patient bootstrap. K agreement is not evidence of individual source identity.
C. Null input/fully absent-sensor case tests source ABSTAIN or neutral contributions, finite risk, exact decomposition. Sensor corruption and label permutation should not alter TRAIN-only routing given identical measurements.

Predeclared metrics AUPRC, AUROC, NLL, Brier, selected route and K, source support, direct signed contributions, exact fallback, and route-versus-raw difference. Repeated TEST-subject bootstrap is descriptive within a synthetic family; 2 draws do not prove universal replication; three planted K use a correlated family and are not three independent hospitals.

## Publication and novelty restrictions

There is no new universal spectral rank theorem. Covariance/SVD, deterministic observation concepts, clustering, linear additive concept bottlenecks, and temporal concepts preexist (including our V3, V53 and V54). The new engineering claim is **an integrated, label-independent routing and provenance contract across measurement regimes**. Successful learned risk readout alone is not physical or medical concept identification. No claims of deployment, clinical generalization, or state-of-the-art. MAGNETS (Forest, Wei and Fink 2026) and ConceptTime are related prior systems to compare separately.
