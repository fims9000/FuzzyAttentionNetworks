# ConceptFAN V19 — predeclared native temporal readout experiment

**Scientific question:** On the synthetic V18 prediction task, why does native ConceptFAN trail a simple logistic model using recency, endpoint, historical change and observed variation? This is an exploratory architecture ablation, not a new clinical validation.

## Code-based defect hypothesis
V10 concept generation already contains signed sensor level, recent mean, first difference and deviation operators across multiple scales. But the *task decision* reduces each native concept trajectory to a **mean of the last six time steps**. The observed recent-feature logistic baseline uses recent mean, last level, early-to-late difference, and recent variability. The V10 reduction can lose some within-episode shape even if its concept generator represented the information.

A competing explanation is limited concept capacity (K3), saturation/regularization or optimization. This experiment tests only the readout pooling hypothesis and must not pretend to resolve the others.

## V19 architecture
Use only the already learned **native temporal concept trajectories** C[t,k] (same concept generation, same prediction/forecast/denoising path). Replace the task summary with a signed simplex mixture over four inspectable operators:
1. mean of last six concept states;
2. final concept state;
3. normalized difference of last-six and first-six concept means;
4. normalized difference of recent and early mean absolute step change.

For every operator include the signed complement gate[k]-value. The resulting native summary lies within [0,gate] and equals 0.5*gate when no sensor measurements exist. Task logit remains exactly bias + sum_k beta[k]*(summary[k]-0.5*gate[k]) plus the separate mask-only care-observation concepts. There is **no raw-to-logit bypass, separate post-hoc model or oracle latent variable**. The model's concept-generation, forecasting and reconstruction modules continue to operate on the original C[t,k]. Expert correction to C remains training-native.

Each concept's readout has eight explicit signed mixture weights; parameterization is softmax (no hard one-hot zero-Jacobian at initialization). Initialize all mixture logits with +2.5 for the original recent-average operator and -1.5 for the seven other choices. This is close to legacy, but **not exactly identical** — hence a frozen-shape arm is necessary. The new summary may become decision-specific; do not assert it changes sensor physics.

## Frozen comparison
Dataset: exactly the 2026-11-04 V18 native generator (`make_data`, generator seed 20411204), **previously inspected validation**, complete synthetic 24 historical × 5 channels, 6 future, 800 TRAIN / 200 validation / 80 sealed test. Three label modes: added source raises future binary risk, lowers future risk, or has no association (null).

All variants share a TRAIN-only candidate basis from first 350 TRAIN episodes via V17 unsupervised residual-lag proposal from K2 prior and then K3. No latent factor oracle, validation outcomes, other clinical cohorts or test IDs used for basis fitting. Three initialization seeds 11,23,37; V10 two-head architecture, adaptive operator gradient bridge enabled equally; fixed hyperparameters: training epochs 12, batch size 64, same loss weights, no consistency regularizer. 3 modes × 3 arms × 3 seeds = **27 completed model fits** unless a real failure is recorded, with failed models counted explicitly.

Arms:
- `legacy`: exact current V10 decision summary, last-six mean; adaptive feature bridge enabled.
- `fixed_shape`: new eight-operator summary with frozen initial mixture parameters, isolating initialization change.
- `learned_shape`: **identical initial mixture as fixed_shape**, but trained shape operator mixture; isolates learned temporal use.

Evaluation: outcome AUPRC, AUROC, Brier on 200 already-viewed synthetic validation episodes; exact additive decomposition; mean absolute contribution of newly proposed slot; observable candidate-alignment of native raw concept; cross-seed signed patient-contribution Spearman (three dependent pairs, no false p-value); learned temporal operator weights and their entropy. No p-values or new holdout claim with three initializations. Report all modes, all seeds and all arms.

External benchmark: V18c on **exact same data** yielded simple logistic recent-feature AUPRC ~0.7314 and ~0.6795 in positive/negative modes. Comparator includes only predictive metrics and is not a trained/equally sized intrinsic concept model; therefore don't claim clinical explanation superiority. Tests must keep V16–V18 existing behavior unchanged.

## Decision rule and confounds
A positive result requires better heldout AUPRC for **learned_shape vs both legacy and fixed_shape**, non-negligible learned shape weights beyond last-six mean, no break of exact additive native concept decomposition, and no reversal of observable grounding. A small improvement may still leave a large predictive gap relative to logistic regression. If no meaningful gain, reject the hypothesis that **readout pooling alone** explains the accuracy gap; investigate capacity and optimization next.

No real physicians, physical-source causal identity, irregular sampling, independent new hospital or true semantic concept discovery established. Accepted IEEE PIERE article unchanged.
