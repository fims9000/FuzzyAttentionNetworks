# V22b — frozen disjoint TRAIN risk-screened native source admission, 2026-11-04

## Motivating negative result
V22a [Actions 41857411144] separated two densely mixed synthetic sources exceptionally well in source-identifiable families (oracle validation matched absolute Spearman 0.999 vs 0.693–0.746 for PCA), but **did not reliably improve prediction**. In the artifact-only-environment-variance family, native ConceptFAN AUPRC fell from **0.69574 PCA** to **0.52924 source-unmixed**. V21b had independently shown that sign-random native risk coefficient initialization changes actual patient-specific signed risk contributions and outcome AUPRC enormously.

V22b isolates whether a **train-only source-risk relevance screen followed by risk-oriented initialization and a structural no-risk mask** improves actual native prediction and stability without changing discovered source axes or creating a post-hoc model. Source formation and task-risk interpretation are distinct.

## Frozen data and denominators
Reuse all four *already inspected* V22a structural synthetic generator families, five observed sensors, TRAIN generator seed 522201 with 800 episodes, validation seed 522202 with 300 episodes, and still sealed TEST seed 522203.

To make risk-screen labels genuinely episode-disjoint from model fitting and source proposal:
- First **600 TRAIN** episodes (same deterministic episode IDs) fit the unsupervised whitened source representation and native ConceptFAN model.
- Remaining **200 independent TRAIN-screen** episodes are used **only** for candidate-to-target association: no gradients, no normalization fitting, no covariance proposal from those 200 episodes.
- VALIDATION remains descriptive; no arm or hyperparameter chosen based on its labels; TEST remains sealed. Both screen and model cohort must have both target classes.

Both method families use the *same* CORE-TRAIN 600 whitened observed measurements, same V10 native K2 architecture, signed time operators, process-only care branch, 10 epochs, batch size 96, and model seeds 11,23,37:
1. `pooled_pca_basis / arbitrary_native_start`: original native coefficient initialization.
2. `pooled_pca_basis / independently_screened_risk`: fixed source axes but TRAIN-screened concept-risk sign.
3. `environment_contrast_basis / arbitrary_native_start`: V22 covariance diagonalized source axes if identifiable on CORE TRAIN; else **ABSTAIN**.
4. `environment_contrast_basis / independently_screened_risk`: same source axes with TRAIN-only native sign and descriptive-only risk masking; else ABSTAIN.

4 generator families × 3 native seeds × 4 arms = **48 registered opportunities**. Expect 36 actual native training runs and 12 identifiability ABSTAINS if the existing two positive and two negative source gates persist under the smaller CORE TRAIN. All deviations must be recorded, never corrected after looking at validation.

## Fixed independent risk screen and action
For each proposed native K2 direction, screen its last-6-step observed whitened measurement projection using only 200 independent TRAIN SCREEN episodes. Use two-sided patient permutation p<=.01 (199 draws), patient bootstrap 95% CI excluding zero (199 resamples), |point-biserial r|>=.15 and stable direction in both predeclared halves with |r|>=.035. The risk screen sees no latent oracle states, no validation labels, no instrument identity or physical provenance; it establishes only predictive association on that TRAIN cohort.

- Accepted risk association: initialize ONLY the existing original native decision_raw_weight[k] to ±0.05 according to TRAIN-screened r. The sign is **unrestricted during subsequent V10 training**; no custom classifier or post-hoc student.
- Rejected/uncertain risk association: set serialized native `signed_risk_policy[k]=2` (V21d), enforcing an **exact zero prediction contribution** for that measurable but risk-uncertain source, while still allowing its real temporal concept, reconstruction and forecast representation to train. Never treat a missing risk association as proof the pattern is nonphysical.
- Control arms preserve all original native parameters and training settings except no source-risk screen initialization/mask. Other than this opt-in change, the underlying model code/graph is unchanged.

Evaluate heldout outcome AUPRC/AUROC/Brier, concept-to-oracle physical/instrument alignment (oracle loaded **after all training**), signed patient-level cross-seed contribution ranks without post-hoc sign alignment, retained count of descriptive-only zero-risk native slots, exact additive logit and equality of same original features between arms at initialization.

## Interpretation and validity limits
An increase in AUPRC may be due to class association priors, risk omission of spurious features or optimization dynamics; all three move together in the screened arm and are **not isolated** in this first V22b test. Do NOT attribute gains solely to concept semantic correctness or to source identification. A colored instrument source can be statistically stable but non-prognostic; a shared artifact may be outcome-associated and pass all statistical screens. No causal/clinical naming allowed, no existing PIERE accepted manuscript modification and no official intrinsic concept baseline claim.

V22a already inspected this generator family and its validation cohorts. V22b is an exploratory diagnostic using disjoint TRAIN labels, **not prospective generalization evidence**. If source-isolation improves but risk still declines, retain that negative conclusion.
