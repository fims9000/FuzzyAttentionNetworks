# ConceptFAN V18 — independent predictive relevance gate for new concepts

## Scientific motivation
V16 showed that native concept content can be more reliably grounded by reviewer examples while the signed decision contribution remains unstable across independent restarts. V17 showed a new native concept can be proposed/admitted and then learn risk prediction; however temporally colored **sensor-only noise** produced 3/3 candidate proposals indistinguishable from genuine latent dynamic factors. A new-source proposal **must not automatically become a medical risk explanation**.

## Frozen V18 design
**Separate the evidence sources in TRAIN.** V17 proposes a sensor-direction from an unsupervised discovery cohort. On independent TRAIN episodes, V18 tests whether the *same observable signed direction* is prognostically associated with a binary future outcome. The risk orientation is read only from the independent risk-screen outcomes, not from validation/test or a synthetic latent oracle.

The screen accepts an *observational predictive association* only when all frozen conditions hold: two-sided label-permutation p<=0.01 (199 permutations), absolute point-biserial r>=0.15, percentile bootstrap 95% confidence interval excluding zero (199 resamples), both disjoint halves show the same effect sign with absolute r>=0.035. Otherwise **ABSTAIN**. Neither this test nor the statistical gate claims causal/medical meaning, and repeated split/selection may affect nominal p-value interpretation.

## Native architecture integration
For accepted candidates, a signed policy is attached to the *actual V10 ConceptFAN decision weight* (signed-risk buffer on native physical concept layer), leaving original slots unrestricted. It is not an explainer head or bypass. An approved new slot starts with coefficient zero to preserve old predictions; a distinct explicit training step seeds a small trainable coefficient so the sign-constrained absolute-value transform has a nonzero gradient. The sign will then be stable **by construction**: sign consistency by itself must NEVER be scored as an independently demonstrated explanation-quality improvement. Nontrivial signed contribution magnitudes, heldout predictive performance, and agreement with evidence must be assessed separately.

## Source versus sensor falsification, all cases
Three simulator seeds (2507,2607,2707), 6 cases each = **18 cases**.
1. A new persistent source predicts outcome positively.
2. Same source predicts negatively.
3. Same persistent source but outcome independent (candidate can be mathematically real but not task-relevant).
4. **Temporally colored sensor noise** predicts outcome (may pass risk screen but is not physical biology).
5. Temporally colored noise irrelevant to outcome (new candidate but risk screen should often abstain).
6. White sensor-only noise irrelevant to outcome (normally proposal abstains).

Each case: 360 TRAIN-only unsupervised discovery episodes + disjoint 260 TRAIN-only supervised screening episodes, 17 observed timesteps, 5 channels, prior K2 sensor-space basis, complete recorded observations. V17 source null 99 permutations, 24 bootstraps; V18 risk screen 199 permutations and 199 bootstrap resamples. Fixed criteria, no failed-case deletion or scenario-specific tuning. All 18 rows and aggregate to CSV/JSON/Actions artifact.

## Limits and falsifiable expectations
The gate should improve *task relevance selectivity*, not source identification. It may reject true predictive factors of small magnitude (conservative), or admit spurious signals correlated with care decisions. Colored noise genuinely predictive of the target remains observationally indistinguishable from a physiological source. These data are synthetic and deliberately favorable with full measurement coverage; irregular real ICU inputs currently trigger ABSTAIN. No claims about discovering new diseases or medical causality. No publication-ready clinical evidence.

**Additional architecture experiment required before paper claims:** after demonstrating the screening algorithm, evaluate native V10 retraining with sign-screened newly admitted slots versus identical unconstrained new slots, plus outcome-null and negative-sign controls. Report all seeds, AUPRC/calibration, nontrivial contributions, and raw concept vs screened sensor-pattern alignment. Constrained sign agreement alone is a tautology, not a scientific finding.

Accepted PIERE article remains unchanged.
