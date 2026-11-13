# V54 pre-outcome protocol: robust source ensembles + observable spectral fallback

Date fixed 2026-11-13 after reading V53 results. This is explicitly post-hoc exploration on NEW seed sets. No claim that improvements are prospectively confirmed or that a clinical concept is identified.

## Trigger and design freeze

V53 on fresh nonlinear-12-channel synthetic draws 20261221/20261237 recovered all 5 measured correlation-clique groups without manually supplied K2, but temporal surprise offered no material predictive improvement, its all-or-nothing consistency guard harmed single-sensor fault responses, and it generated K=0 on the unrelated variable-factor mixed-loading family (true planted latent K=3,5,7; truth evaluation-only).

Hypothesis A: In an automatically identified group of 3+ independently noisy sensors, a **robust per-time median**, with fixed train-only normalization and frozen fuzzy bounds, retains the healthy majority during a single-sensor sign flip better than the V53 whole-source NO_LINK gate. In 2-sensor groups do not guess which sensor is damaged. Common-mode changes may still be undetectable.

Hypothesis B: When strict source cliques fail because measurements are overlapping mixtures, infer **train-only positive-lag predictive rank** and use its signed sensor basis as *unnamed observable subspace concepts*. This is existing system identification / PD-ConceptFAN V3 methodology. A resulting basis is a coordinate convention, NOT a unique latent physical source. If rank is 0 or has no observation support, abstain rather than invent a physical concept.

## Frozen algorithms

1. V53 group proposals remain exactly the same, including all-sensor unsupervised source construction; Y and oracle factors may enter only the downstream risk learner and post-fit audit.
2. For each source group with >=3 channels: calculate a robust signed-standardized timepoint median from actually observed channels; all missing => neutral 0.5; state threshold and width are recalibrated from TRAIN-only robust signals. The primary V54 risk head uses state concepts (V53 surprise failed primary exploratory screen). For groups of size 2 use the unchanged V53 observed-value signed mean. Enforce exact additive risk logit and no direct feature bypass.
3. For spectral fallback use existing `fit_lagged_predictive_rank` from repository on observed TRAIN only (lags 1/2/3, 60 patient permutations, 20 patient bootstraps, familywise 95% largest-singular-value threshold). TRAIN-only means/stds and canonicalized left spectral loadings define signed *dense observable operator concepts*; these are explicitly distinguished from sparse measurement groups.
4. The predictive spectral state is a normalized observed signed projection with support-reweighted fuzzy sigmoid, using TRAIN-only median and IQR width. Rank and number of concepts are data-derived; the classifier reads only those concepts, with separate signed additive logistic weights, no latent-output shortcut.
5. All unfitted/no-source models return bias-only prevalence prediction. Do not call a stable K a confirmed number of biological causes. Do not claim that spectral components uniquely equal true factor axes.

## Frozen evaluation

- New nonlinear extended V30 seeds **20261301, 20261303**, TRAIN1400 VALID600 TEST700 (VALID untouched by architecture choice), 12 channels. Evaluate original V53 state, V53 whole-source guard, V54 robust grouped-state, raw logistic at MCAR20, single channel5 flip, all channels4:6 flip, fully absent source4:6.
- New independent mixed-loading variable-factor generator seeds **20261310 + K** with K=3,5,7 as oracle-only external metadata; n=960, 8 channels, TRAIN/DEV-VALID only, no sealed TEST. Evaluate V53 strict groups, V54 spectral-only and raw logistic. Do not change hyperparameters per K.
- Primary metrics AUPRC, NLL, Brier, exact logit decomposition, observed source K and effective spectral rank, support/fallback. Bootstraps when reported must resample patients within each test cohort, never call model/seed paired rows independent hospitals.
- All real patient data and previous journal manuscripts remain untouched; no raw personal identifiers.

## Failure boundaries

- Majority method may underperform if 3 sensors have correlated noise, two are simultaneously wrong, or group co-observation is too poor.
- Spectral rank may return K=0 or wrong K under nonstationarity, informative missingness, weak excitation, high noise; return low confidence instead of magical semantic discovery.
- All physical source names remain UNKNOWN without independent source-specific assay or interventions.
- Compare to already published MAGNETS (2026) and ConceptTime (2026); do not announce first automatic temporal concept generation.

Reproducing V53 on its already viewed TEST is not fresh confirmation. Only V54 new draws may be scored, and improvements observed there remain exploratory because the design was informed by V53.
