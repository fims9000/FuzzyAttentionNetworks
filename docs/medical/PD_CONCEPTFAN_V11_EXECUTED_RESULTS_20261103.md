# ConceptFAN V11 — executed research, falsification, and what remains unproved

**Research state, 3 Nov 2026:** exploratory implementation with synthetic
positive/negative identifiability controls and matched real clinical
development ablations. **NOT** a clinical predictor or a publication-ready
causal discovery theorem. Original accepted PIERE and V10 evidence unchanged.

## 1. Executed evidence — pinned sources

- V11 native structural source, 3-model baseline and six new model fits:
  [successful PhysioNet CI 41685277605](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41685277605),
  artifact `conceptfan-v11-predictive-groups-3`.
  All 9 fitted methods were evaluated on **the exact same 300 already
  inspected DEV patients**, after training on the same 2400 official
  PhysioNet Set A TRAIN records. The original 400 calibration and
  600 test records were never inspected in the new analyses.
- [Successful planted 3/5/7 structural falsification 41686176933](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41686176933),
  artifact `conceptfan-v11-observable-factor-stress-8`.
- [Joint model-effect certificate run 41687120995 SUCCESS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41687120995),
  artifact `conceptfan-v11-joint-certificates-2`.
- [First core geometry test run 41683976148 SUCCESS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41683976148).
  Includes degenerate repeated-eigenvalue rotations, exact group-logit
  accounting, train-only bootstrap reproducibility and projector
  gradient tests.

**Important:** the low-rank group Gram implementation introduced a
non-finite backward gradient at zero concept variance in a follow-up
CI run, even though a previous dense-Gram implementation completed
successfully. This failure is reproducible and has been corrected by
`torch.linalg.vector_norm` with a finite zero subgradient and an
explicit all-constant backward regression test. The later full rerun
must pass before treating the optimized kernel as verified. Neither
the failure nor its significance is hidden.

## 2. Distinguishing source structure from clinical semantics

The candidate concept-generating operator uses *training-only*
positive-lag sensor cross-covariance C=[C1 C2 C3], SVD, patient
bootstrap, spectral-gap tests, and sign/rotation-invariant subspace
projector overlaps.

Each source direction can receive ONLY one of:
- empirical source singleton stable;
- empirically stable group, individual axes unresolved;
- abstain: unresolved external rank boundary;
- abstain: unstable predictive subspace.

A source-structure certificate is **not** evidence that changing
such a direction actually alters a physiological mechanism.
The native V11 training replaces V10's arbitrary axis-by-axis SVD
operator weight loss with a group-projection penalty; a further arm
replaces within-model axiswise mask-thinning agreement with a
rotation-invariant temporal Gram loss. The measured-mask policy
remains an independent exact-additive concept source.

## 3. Full clinical V10 / V11 matched ablation

Averages across independent model initialization seeds 11,23,37,
18 epochs, identical train/development input variables, optimization,
36h measurement history and 12h auxiliary observation future.

| DEVELOPMENT metric (NOT independent test) | V10 refit | V11 projector + original view loss | V11 projector + group view loss |
| --- | ---: | ---: | ---: |
| AUPRC original | **0.379405** | 0.368838 | 0.368702 |
| AUPRC additional MCAR | **0.327814** | 0.313210 | 0.313031 |
| AUPRC time/channel MAR | **0.387328** | 0.373247 | 0.376664 |
| AUPRC physical branch alone | **0.372582** | 0.362564 | 0.363530 |
| 6-12h future observed-signal RMSE | 0.806170 | 0.805947 | **0.805923** |
| Randomly withheld observed-signal RMSE | 0.809086 | **0.808088** | 0.808176 |
| Case-specific physiological concept-rank Spearman EXCESS over patient-permutation marginal null | 0.113021 | 0.183561 | **0.189422** |
| Mean learned group definition projector overlap | 0.682838 | 0.777431 | **0.778995** |
| Mean signed group-contribution Spearman over **7** groups | -0.004809 | **0.087708** | 0.087644 |
| Effective number of physiological contributions | 5.2488 | 5.1309 | 5.1689 |
| Max exact logit decomposition error | 1.2e-7 | 1.2e-7 | 8.6e-8 |

**Tradeoff:** improved consistency and slightly lower auxiliary
errors, but AUPRC falls by 0.0106/0.0107; this does **not** meet
the previous -0.01 development noninferiority desideratum. The
signed group-effect score is still very low and is not a verified
biological concept signal.

Exact paired three-seed signflip tests for AUPRC differences V11-V10:
two-sided p=0.5 in both V11 variants. The 27-exhaustive-draw model-only
intervals are [-0.025078, +0.000860] for original view and
[-0.022934, -0.000784] for group view. Even an interval excluding
zero under only three already-inspected initialization clusters is
**not** a population CI; the signflip test has inadequate
resolution for conventional p<0.05.

## 4. Source spectral groups and their functional limits

The **observed train-only predictive computational rank is K=8**.
The bootstrap gap certificate on train assigns:
- individual observable singletons: indices **0,1,2,3,4,7**;
- a two-dimensional *unresolved* group: indices **5,6**.

This proves no unique clinician-named ontology. Under the same
bootstrap, source group indices 5–6 have Q10 projector overlap 0.9143,
yet their individual axes must not be named.

More importantly, source singleton stability is NOT synonymous
with a stable learned risk contribution. Some nominally stable
individual axes have strong *negative signed* same-patient Spearman
between different model seeds (as low as around -0.97).

The separately executed TWO-STAGE clinical explanatory screen uses:
1. train-only structural source-spectral status; and
2. for EVERY independent model pair, positive signed patient-effect
   Spearman >=0.5 AND learned definition-group projector overlap
   >=0.75, with no missing/duplicate model pairs.

Thresholds were introduced for **future confirmatory replication**,
after prior development analysis, so this run is RETROSPECTIVE
descriptive screening, NOT an FWER guarantee.

| Arm | Structurally stable singleton axes | Candidate functionally reproducible individual effects | Candidate group-only | Groups abstained |
| --- | ---: | ---: | ---: | ---: |
| V10 refit | 6 | **1** | 0 | 6 |
| V11 original view | 6 | **1** | 0 | 6 |
| V11 group view | 6 | **1** | 0 | 6 |

The one candidate is **slot 4**, with minimum across-run
signed effect Spearman >0.90 and learned definition overlap >0.89
in both V11 variants. It is a reproducible *computational effect
candidate*, **not** a validated distinct physiological mechanism.

This is the key falsifiable finding:
> Stable observable predictive eigenvectors do not ensure stable
> independently trained patient-specific decision explanations.

Although V11 increases average matching/stability, **it does not
increase the number of individually credible concept explanations**.
Claiming V11 "solves semantic stability" would contradict the evidence.

## 5. Stronger planted synthetic controlled experiment

All controls use 900 generated train patients, 36 steps, 10
observed channels, equal factor stationary variances, measured-cell
coverage 94%, and no hidden latent states provided to the estimator.
Oracle emission projection is evaluated only AFTER fitting.

| True dynamic dimension | Distinct rates: recovered rank / singleton axes | Two equal rates: recovered rank, individual+group | Strict iid-in-time null rank | Weak within-patient shuffled rank |
| --- | --- | --- | ---: | ---: |
| 3 | **3 / 3** | **3, 1 individual + 1 group of 2** | **0** | 2 |
| 5 | **5 / 5** | **5, 3 individual + 1 group of 2** | **0** | 4 |
| 7 | **7 / 7** | **7, 5 individual + 1 group of 2** | **0** | 5 |

When K equals the true planted generator dimension, the
train-only estimated predictive subspace has oracle emission
overlap between **0.999973 and 0.999991** in this deliberately
identifiable linear, high-SNR setting. These near-perfect values
are a PROPERTY OF THE GENERATOR, not a promise of generalization
to nonlinear ICU physiology.

**Crucial negative control:** shuffling time *within the same
patient* does NOT eliminate all patient-specific mean and
temporal correlation induced by short persistent episodes.
Indeed the method erroneously detects ranks 2,4,5 on that
weaker shuffled control. The separately implemented strict
iid-across-time null leaves the cross-sensor mixing intact
but removes all temporal dependence, and yields rank 0 at
all three planted instantaneous factor dimensions.
Do NOT hide this weak-null failure.

## 6. Why this might or might not be a new research contribution

What is implemented and confirmed:
1. explicit empirical train-only spectral gap / projector
   diagnosis of nonunique time-series predictive coordinates;
2. genuinely train-time rotation-invariant projector loss on
   measured signed-concept operator definitions;
3. rotation-invariant group temporal consistency alternative;
4. exact grouped direct-logit accounting and two-stage
   source-plus-functional explanatory abstention;
5. positive linear identifiability and negative degenerate / no-time
   controls, plus a matched real-data Pareto tradeoff.

What is NOT shown:
- uniqueness or identifiability of latent physiological causes;
- significant clinical AUPRC or concept-fidelity improvement;
- new-hospital external validity;
- validity of empirical percentile/gap thresholds as confidence
  intervals or distribution-free certificates;
- model superiority to the 2-layer Transformer:
  the prior matched external benchmark had AUPRC 0.424756
  versus V10 0.379405 on the old DEV cohort;
- method-level novelty PRIORITY over all 2024–2026 work.

Related work to compare explicitly:
- Forest, Wei, Fink, *MAGNETS*, Data Mining and Knowledge Discovery,
  2026. DOI: 10.1007/s10618-026-01267-y (automatic concepts).
- Galliamov et al., *Concepts' Information Bottleneck Models*,
  ICLR 2026 (minimal sufficient concept bottlenecks).
- Nonlinear identifiability and stable subspaces under eigenvalue
  degeneracy have long preceding mathematical literature. We do not
  claim a new SVD theorem.

**Proposed narrower novel method:** uncertainty-aware observable
predictive group formation plus rotation-invariant *native concept
learning* and a separately reported failure of functional
reproducibility, with meaningful abstention instead of arbitrary
per-axis concept identities.

## 7. Concrete next acceptance requirements

**Independent study:** external hospitals/patients unavailable here
are a hard evidence gap, not something to fake. Before a
publication claim, freeze both new consistency loss and screening
thresholds, then:
- replicate across >=10 independent initialization seeds and
  >=3 independent batch-order seeds, with patient/model/site
  hierarchical uncertainty and report per-seed ranges;
- test an externally implemented *automatic concept baseline*
  (e.g. fair adapted MAGNETS, clearly disclose regression-to-
  classification changes), not only V7/V10 internal ablations;
- evaluate nonlinear observational regimes with known latent
  identifiability/ambiguity, including varying true K, and ask
  whether the detector correctly abstains;
- test actual clinical meaning via external measurement/expert
  reference for individual concept interpretations, with
  uncertainty about unavailable patient measurements;
- maintain at least the previously preregistered
  `AUPRC_drop <=0.01` on a new independent cohort,
  while improving **joint** structurally+functionally
  reproducible concept count (currently only 1).

**Honest decision:** retain V11 as a well-specified and partly
falsified research prototype. Do not rewrite accepted PIERE paper
or claim a finished clinical concept ontology.
