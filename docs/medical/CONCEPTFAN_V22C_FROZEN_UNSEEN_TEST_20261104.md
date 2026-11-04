# V22c — frozen independent synthetic TEST verification of V22b (2026-11-04)

## Rationale and evaluation independence
V22a compared two source bases on four **already inspected** synthetic generator families. V22b used an independently held 200-case TRAIN risk screen to set native risk coefficient starting signs and permanently zero-mask non-prognostic **descriptive** concepts. It achieved large validation AUPRC increases, but validation outcomes were inspected in V22a and V22b. This protocol is frozen **before V22c TEST outcomes are generated or evaluated**.

Use the previously unused original V22 TEST generator seed **522203** and n=300 per structural family (distinct-variance, proportional variance, sensor gain drift, artifact-only variance). This is a **new random test draw within previously inspected data-generation mechanisms**, not genuinely new simulator families or real medical validation.

## Immutable model training and gate policy
- EXACT same V22b 600 core TRAIN episodes from generator seed 522201 for (i) measured PCA/whitening, (ii) environmental covariance unmixing, and (iii) native V10 predictor training.
- EXACT same disjoint final 200 TRAIN episodes for target-risk source screen (permutation=199, bootstrap=199, p<=0.01, |r|>=0.15, stable halves) and same initial signed native raw task coefficients ±0.05 for accepted sources; rejected sources retain **serializable source risk-policy 2** (zero contribution). No V22b hyperparameter or criterion adjustments.
- Same source separability settings (199 environment permutations, 48 bootstrap resamples, fixed-environment subspace cosine threshold 0.99, established covariance eigendifference), same full 4×3×2×2=48 opportunity denominator with 12 identifiability ABSTAINS expected from V22b; all refusals remain present.
- Same model seeds 11,23,37, ten joint V10 training epochs, batch 96, **no head-only additional refit**, no extra classifier/probe, same native forecast/denoise/unknown-neutral loss and concept bottleneck.
- During the **entire fit stage** models receive only V22b original TRAIN and V22b original validation through the unchanged `fit_multiscale` interface; they are **never given V22c TEST measurements, outcome labels, latent factors or even a V22c TEST split**. TEST random episodes are not simulated until **all 36 candidate neural fits (and all screen decisions) have completed across all four families**.
- Then generate the single sealed TEST draw (seed 522203, n=300, every family once) and evaluate every trained model without model or hyperparameter updates. Synthetic true physical/instrument source trajectories are loaded only during this post-fit evaluation; Hungarian matching is audit-only, not training or source selection.

## Frozen comparisons and outputs
For every structural family and eligible source basis, report:
- Arbitrary signed-start native ConceptFAN vs independently TRAIN risk-screened ConceptFAN **TEST AUPRC, AUROC, Brier**, exact additive risk decomposition, physical-vs-artifact two-source absolute alignment; all per seed and aggregate values; contrasts between arms and Brier even when AUPRC improves.
- Explicit ABSTAIN counts/status, source-risk screening notes, no-risk masked slots, and all TEST outcomes evaluated.
- No statistical generalization claim from three neural seeds trained on the same 300 TEST episodes.
- NO optimizer/threshold/model iteration may be made based on the V22c TEST scores. Negative outcomes must remain published. V22c is strictly a frozen *descriptive* external draw diagnostic, not a training set for further tuning.

## Interpretation limits
Sign screening supplies supervised task information from 200 additional independent TRAIN labels and therefore is not a "purely unsupervised conceptual discovery" method. It does not establish source physiology, causality, expert-concept agreement or external hospital generalization. A high TEST AUPRC for an intrinsically mixed artifact direction remains possibly semantically incorrect. Cross-site real-world validation and genuine measured source provenance remain a separate requirement.

The V22a/V22b results, accepted PIERE manuscript and prior Draft PR #18 are not rewritten or overwritten. This is V22 Draft PR #19.
