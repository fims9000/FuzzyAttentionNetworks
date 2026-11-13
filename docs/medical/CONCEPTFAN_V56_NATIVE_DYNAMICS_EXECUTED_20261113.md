# ConceptFAN V56: executed native trainable temporal concepts (13 November 2026)

**Evidence status:** executed exploratory synthetic experiment, not a clinically validated discovery. Original pre-outcome protocol: `CONCEPTFAN_V56_FROZEN_NATIVE_DYNAMICS_PROTOCOL_20261113.md`. Full [GitHub Actions successful run 43440020212](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43440020212), commit `55bbbe386144aeeb9de109ba795eac6f6a4cea1b`, evidence artifact `conceptfan-v56-native-temporal` (ID `14044313242`). Seventeen V56/V55/V54 tests passed; smoke and full experiments passed. Machine-readable files `V56_EVIDENCE/STATUS.json`, `GROUP_RESULTS.csv` and `MIXED_RESULTS.csv` are definitive.

## Architecture actually trained

1. V55 unsupervised TRAIN source routing formed statistical observable concepts without any outcome labels or oracle factor trajectories. Two new nonlinear draws selected `MEASURED_GROUPS` (K=5); independent mixed-loading examples selected `PREDICTIVE_SUBSPACE` (K=3/5).
2. **Native differentiable concept formation:** diagonal causal recurrent filter with per-source learnable input gain, recurrence, threshold and positive fuzzy width. Self-supervised source loss (TRAIN only) combines 20% randomized observed-cell withholding, source-level denoising, and next-step source prediction. Original source operators and K are not trained against the task labels.
3. Concept encoder was frozen before fitting a separate TRAIN-only logistic task head. The task receives only time-averaged fuzzy concepts, and the signed output logit is exactly `bias + sum_k beta_k*(mean_t c_k-0.5)`.
4. At a fully unavailable source, its concept is exactly neutral 0.5 and adds zero to task logit. Prefix causality, hidden-value invariance, source-parameter immutability under label permutation, absence of supervised gradients into concept training and exact output decomposition were covered by tests.
5. This is a **trainable concept-state model**, not a native Transformer or general causal/clinical source identifier. K and source coordinate axes still rely on the V55 measurement-oriented selector. No real patient records or original published PIERE/IOI checkpoints were changed.

## Unlabeled optimization sanity check

Across 4 distinct group-family neural fits (2 source draws x seeds 11,23), eight epochs each, the TRAIN-only objective decreased from around 0.343–0.346 (epoch 1) to 0.265–0.267 (epoch 8). Measured-source denoising MSE fell from about 0.217–0.220 to 0.171–0.173; next-step source forecast MSE from ~0.496–0.499 to ~0.379–0.382.

These are **TRAIN** losses, not independent evidence that learned concepts generalize or have acquired physical meaning.

## Fresh nonlinear synthetic tests, PHYSICAL_EXTRA target

Two previously uninspected nonlinear draws, each TRAIN=1400 / VALID unused=600 / TEST=700, K=5. Native seeds 11/23 operate on the SAME TEST episodes; they are not independent cohorts.

| Draw / TEST intervention | V55 static AUPRC / NLL | V56 seed11 AUPRC / NLL | V56 seed23 AUPRC / NLL | Raw logistic AUPRC / NLL |
|---|---|---|---|---|
| 20261393 / MCAR20 stable | .82963 / .51669 | .83222 / .51596 | .83226 / .51574 | .82191 / .51112 |
| 20261393 / single source-sensor flip | .82643 / .53041 | .82928 / .52721 | .82944 / .52711 | .71223 / .64458 |
| 20261393 / common-mode three-sensor flip | .57505 / .76844 | .57700 / .76634 | .57676 / .76629 | .57260 / .90947 |
| 20261397 / MCAR20 stable | .77908 / .52869 | .77998 / .52670 | .78038 / .52659 | .76924 / .52268 |
| 20261397 / single source-sensor flip | .76934 / .54278 | .77281 / .53813 | .77358 / .53785 | .63507 / .65030 |
| 20261397 / common-mode three-sensor flip | .51303 / .77097 | .51990 / .76839 | .51961 / .76870 | .50497 / .90367 |

**Interpretation:** modest and consistent small improvement over static V55 in these two *same-family* synthetic draws; not enough to assert a statistically established advantage or state-of-the-art. Clean NLL can remain lower for the raw logistic baseline, despite its weaker AUPRC. V56 does not repair unobserved common-mode sensor inversion.

## Independent mixed-loadings family — DEVELOPMENT validation only

Frozen original TEST remains untouched. Native source seed11, train-only rank source:

| Planted K (post-model oracle only) | Detected observable K | V55 static AUPRC / NLL | V56 native AUPRC / NLL | Raw logistic AUPRC / NLL |
|---:|---:|---:|---:|---:|
| 3 | 3 | .42339 / .68156 | .41930 / .68213 | .40519 / .71629 |
| 5 | 5 | .47894 / .68964 | .47188 / .68951 | .43842 / .70954 |

**Negative result:** extra native recurrence and self-supervised concept dynamics did **not** improve AUPRC over the simple static spectral concept head in this other synthetic family. Correct detectable K does not identify individual physical latent factors.

## Scientific readiness and next real validity gate

The original missing capability is now implemented as a working **unlabeled trainable temporal concept mechanism**. But there is no verified unique number of clinical concepts, no identification of medical mechanisms, no across-hospital testing, and no established superior predictive performance. Proper further work requires independent measurement anchors/physical references, a source-shifted evaluation not sharing the same V30 simulator family, explicit near-identifiability limits, direct comparison against CBM/MAGNETS/ConceptTime, and evidence that native concept dynamics are informative beyond simpler measured-source operators. A manuscript should describe this as exploratory research rather than a completed clinical concept discovery solution.
