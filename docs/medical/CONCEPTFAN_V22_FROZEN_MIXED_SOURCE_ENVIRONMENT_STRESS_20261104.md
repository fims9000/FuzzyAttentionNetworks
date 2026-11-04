# ConceptFAN V22 — frozen multisensor mixture/source identifiability stress (2026-11-04)

## Scientific question and relation to V21
V21 used very easy synthetic source geometry: a new persistent concept was almost perfectly aligned with *one observed sensor*. V21b made risk attributions stable even for artifacts; V21c/d showed why stable or repeatable concepts are not automatically physically meaningful. V22 removes the easy geometry and tests whether observed features become meaningful through **train-native learning** rather than a simple near-one-hot sensor identification.

Use a *known existing statistical baseline*: whitening plus covariance-difference eigendecomposition across externally labeled observation environments (a limited second-order blind source separation method). **This algorithm is not claimed to be new.** The interesting research question is whether those source estimates furnish better intrinsic native concept representations and when identification should be explicitly refused.

## Frozen hypotheses, not outcomes
- H1: With a fixed, dense, non-orthogonal sensor mixing matrix and distinguishable environmental variance profiles of statistically independent latent AR factors, second-order environments can identify two source directions up to sign/permutation *under its assumed model*. Test against PCA basis on heldout observable time series and after V10 training.
- H2: If both sources undergo indistinguishable proportional variance changes, the covariance contrast lacks unique eigenvectors, so the algorithm must **ABSTAIN**, despite the predictive task having signal.
- H3: If *sensor calibration gain shifts between environments*, the fixed-mixing assumption is violated; a convincing eigen contrast can still appear. Report algorithm's refusal or false acceptance, **never infer physical identity** from confidence in its own statistical estimate.
- H4: If only the independent instrument artifact changes variance by environment, recovered directions can match an artifact as easily as the physical process. A high source-alignment score is not medical validation.

## Frozen data, splits, and source identifiability conditions
Four generator families: `mixed_distinct_variance`, `mixed_proportional_variance`, `mixed_sensor_gain_drift`, `mixed_artifact_only_variance`. Exactly two independent colored temporal latent factors (physical-generated factor and an instrument-generated factor), both dynamically correlated with AR coefficient 0.72, 24 historic and six future steps. Five observed sensors are **dense, non-orthogonal linear mixtures** of both factors, plus isotropic acquisition noise; all five sensors are a priori unnamed as physical mechanisms.

Two observation environments are **given as metadata** in TRAIN only, balanced across patients; the model never receives them as outcome-side input. Original sensor mixing fixed for first/second/fourth families, while third deliberately applies a device gain change in the second environment. TRAIN 800 episodes (generator seed 522201), validation 300 (522202), sealed synthetic TEST 300 (522203). No latent states accessible in model data objects or fit/selection; synthetic physical/instrument factors may be loaded ONLY after all model fits for an explicit audit. Outcome depends on physical factor in all four families, but outcome labels are **NOT USED** for source proposal. TRAIN-only centering/whitening; no separate reference instruments and no clinician annotations.

A density/entanglement check ensures both latent factors are visible in at least three observed sensors and mixing columns have substantial nonzero cosine overlap. Hold out a separate environment metadata record; never fit normalization on validation/test.

## Frozen covariance unmixing (not a new theorem)
1. Estimate TRAIN-only pooled covariance and top-2 principal subspace, whiten by its two eigenvalues; record inversion back to original sensor coordinates for audit. Whitening defines the **only** model input transformation used by both PCA and contrastive arms. This keeps training signal budget comparable and prevents an extra raw-feature route.
2. In the whitened two-dimensional space, estimate environment-specific covariance matrices A/B, then diagonalize symmetric B−A. Orient eigenvectors by deterministic largest RAW-sensor absolute coefficient. No validation labels or latent factor states enter algorithm.
3. Use a block-per-patient permutation null (199 environment permutations), preserving within-episode temporal dependence, and 48 patient bootstrap replicates; require 10th percentile of *matched two-vector absolute cosine* at least 0.80; require standardized contrast eigengap >=0.20 and a null p <=0.01. If contrast appears only because of a common proportional scale shift or is poorly identified, **ABSTAIN**.
4. The V17 one-direction residual-lag proposal cannot be a sufficient two-source baseline here because the relevant basis already spans the mixed rank-2 subspace. Use fixed PCA axes as the fair no-environment baseline, no outcome labels in either method; report their raw measured time-series source separation and heldout native model results side-by-side.

## Model experiment (all opportunities, do not discard refusals)
4 families × three native seeds 11/23/37 × two directions = **24 registered native opportunities**:
- `pooled_pca_basis`: V10 K2 base concept bank fed the same fixed TRAIN-whitened 5-channel representation, with identity basis for first two whitened channels.
- `environment_contrast_basis`: exactly same native model, input, training algorithm and seed, but K2 basis given by train-only environment contrastive eigenvectors embedded into whitened first two channels. This arm **ABSTAINS** if source identifiability gate fails.

Same V10 10-epoch joint task/forecast/reconstruction loss, batch size 96, feature_exploration=False to preserve basis. All source directions are train-native measurement filters, not explanations of a separate post-hoc classifier. Same inputs and labels, and no extra prediction head. Both native models get same whitened inputs; independent source provenance mapping is retained for interpretable inverse backprojection to original sensors.

Metrics (TRAIN-only proposal; VALIDATION only after all arm decisions): heldout AUPRC/AUROC/Brier, exact native additive decomposition, two-source post-training *absolute* oracle-alignment score with optimal assignment (Hungarian, permutation/sign used **only in evaluation**, never to fit or pick an arm), environment-stratified source identification, pairwise signed patient-level contribution stability; compare with **no neural training** PCA/contrastive linear scores under the same metrics. The registered TEST seed 522203 stays sealed. Report every arm, every ABSTAIN, every high-confidence artifact and negative covariance false acceptance, with unchanged full-denominator manifest.

## Publication and clinical boundary
These are intentionally transparent synthetic data-generating mechanisms. A model that recovers the simulator's *physical*-named source has not discovered a real physiological law; one that recovers the *instrument*-named source has equally strong statistical alignment. Repeated environment signatures can make sources identifiable up to sign/permutation only with the required fixed-mixing and independent component assumptions. Without externally established sensor provenance, the statistical method cannot know physical vs artifact. No real doctor, real EHR cohort, official end-to-end intrinsic concept baseline, causal proof or clinical safety claim.

Based on Draft PR #18. Accepted PIERE camera-ready untouched. V22 is a new Draft PR, not merged into production.
