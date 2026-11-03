# ConceptFAN V12: observable-anchored signed effects (EXPLORATORY)

## Scope and original finding

The original accepted IEEE PIERE paper concerns the non-reproducibility of
local concept explanations after independent refits. **It is unchanged.**

The first complete pre-registered 4,000-record PhysioNet Set B evaluation
(run [41710356212](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41710356212))
compared frozen V10/V11 architectures. Mean AUPRC was **0.389276 for V10**
and **0.387814 for V11 group**, with signed functional/definition stability
passing only **one** singleton per method. The descriptive bootstrap interval
for V11 minus V10 was **[-0.019020,+0.015577]**. These are NOT proof of
noninferiority, superiority, or independent-hospital transfer.

This V12 branch is a **new architectural research hypothesis**, not a
retrospective reinterpretation of the accepted PIERE experiment.
The first Set B result is used solely to MOTIVATE this new hypothesis,
not to optimize V12 or select its hyperparameters. The Set B cohort is
already spent as a prospective holdout and is not accessible to V12.

## Mathematical construction

Use exactly the same train-only positive-lag observable cross-covariance
C = [C_1 C_2 C_3] and SVD/gap/bootstrapped group certificate from V11.
The computational concept capacity K is inherited from original
**train-only** predictive rank; K is not five merely because five named
diseases are convenient. Clinical latent factor count is NOT identified.

For each **structurally identifiable singleton** source mode u_k only,
choose a deterministic sign: let d*=argmax_d |u_{dk}| (first index breaks
ties), then require u_{d*k} > 0. For unresolved grouped singular values
NO individual orientation is defined, so a group of two may rotate and
must be explained as a group.

For observed input x_{itd} and observation mask m_{itd}, define the
last-six-history observed sensor average

    h_{id} = sum_{t=T-5}^{T} m_{itd}x_{itd} /
             max(1,sum_{t=T-5}^{T} m_{itd})

and per-concept observed support

    r_{ik} = sum_d |u_{dk}| 1[any_{t=T-5}^{T}m_{itd}=1] /
             sum_d |u_{dk}|.

The explicitly measured, bounded *TRAIN supervision proxy* is

    a_{ik} = (r_{ik}/2) tanh(sum_d u_{dk} h_{id}).

This is **not a clinician-labeled biomarker**, just an inspectable
historical data operation. With no observed history r=0 and a=0.
The native model is still trained through the original V10/V11 fuzzy,
multiscale concept head: raw values -> observed operators -> learned
thresholds/widths -> trajectories -> native exact signed logit.
The anchor targets only the **centered native concept summary** and
has no direct model-to-logit bypass.

For each structurally resolved singleton, the training set class
difference delta_k = E[a_k|y=1] - E[a_k|y=0] is tested using
**96 stratified patient bootstrap resamples** and **199 two-sided
patient-label permutations**. Both a 90% signed bootstrap interval
not crossing zero AND a Bonferroni-adjusted p threshold AND absolute
delta>=0.02 are required before a task-effect sign is permitted.
Otherwise risk sign = 0 (ABSTAIN). No DEV, test, oracle or Set B outcome
may participate in this selection. As this is an observational label
association, its sign must NOT be presented as physiologically causal.

The V12 training objective:

    L_V12 = L_V11_rotation_and_prediction
            + 0.75 L_observable_anchoring
            + 1.50 L_two_views_direct_signed_logit_contribution
            + 0.20 L_train_label_supported_beta_sign

The one-step ablation **V12 anchor-only** includes just the first added
term. The **V12 anchor + signed effect** arm includes all three.

- L_observable_anchoring is reliability-weighted smooth L1 on BOTH
  randomly thinned representations of the SAME training patient.
- L_two_views_direct_signed_logit_contribution is a reliability-weighted
  smooth L1 distance between the native exact direct logit contributions
  phi_k on those two views, for certified singletons only.
- L_train_label_supported_beta_sign is a hinge-squared loss
  max(0,0.08 - s_k*beta_k)^2 only where train-only association provided
  a nonzero s_k. Inferred label association influences TRAIN loss and is
  fully disclosed; **a forced sign does not count as independent proof
  of signed concept identifiability**.

For empirically unresolved groups, V11's rotation-invariant
subspace alignment and Gram view-consistency remain intact.
Do not treat a stable 2D projector as two separately discovered entities.

## Matched study and independent architecture

Preregistered config:
`configs/medical/pd_v12_trainonly_signed_effects_20261103.yaml`.

The new exploratory experiment fits every method from scratch on the
same 2,400 original Set A TRAIN records with the 36-hour history and
12-hour forecasting objective, using 3 initialization seeds (11,23,37).
The **300 Set A DEV** cases have been repeatedly inspected by previous
experiments; all current confidence intervals are therefore DESCRIPTIVE.
The separate 400 calibration and original 600 test cases are sealed.

| Method | Interpretation |
| --- | --- |
| V11 projector + group consistency | Frozen original architecture, exact refit comparability |
| V12 observable anchor only | Controls impact of orientation/observable supervision |
| V12 anchor + signed-effect training | Tests additional functional contribution regularization |
| MAGNETS-style binary adaptation | Independently implemented temporal mask-and-aggregate learned concepts (NOT authors' original code) |
| HistGradientBoosting | Independent prediction-only point reference, no concept claims |

The binary adaptation is based on Forest, Wei, and Fink,
**When, how long and how much? Interpretable neural networks for time
series regression by learning to mask and aggregate**,
*Data Mining and Knowledge Discovery* 40, article 103 (2026),
https://doi.org/10.1007/s10618-026-01267-y.
**Their actual method is for regression**, with different mask handling
and loss. Our classifier shares the high-level mask -> time aggregation
-> concept bottleneck -> additive decision idea, but is NOT a drop-in
faithful replication or an official MAGNETS baseline.

MAGNETS-style concept labels are unordered: Hungarian matching uses
**TRAIN-only** signed-effect shapes, and never DEV or Set B. Only
the *permutation* is permitted; negative signed effects remain NEGATIVE,
and weak train-match correlations below 0.2 ABSTAIN.

Primary reporting: AUPRC, AUROC, Brier, all per seed and original/MCAR/MAR.
For V11/V12, source-group signed contribution Spearman, definition
projector overlap, 3 independent model pairs, and a joint strict
every-pair pass requirement Spearman>=0.5 plus projector overlap>=0.75.
The same fixed thresholds used in the exploratory V11 analysis are
carried forward. For MAGNETS-style, only TRAIN-matched contribution
stability, not native sensor-axis identity, is measured.

## Explicit falsification criteria

- If only the risk coefficient sign becomes stable but patient-specific
  signed effect rankings or reproducible singleton counts do not improve,
  V12 fails its target despite applying a train-time sign constraint.
- If V12 degrades prediction materially, report the tradeoff rather
  than changing loss weights on the repeatedly observed DEV.
- If observed sign can arise entirely from measurement masks or random
  label permutations, do not interpret it as biological signal.
- If a low-variance or constant concept appears stable, it must ABSTAIN.
- Passing synthetic axis recovery and train-only risk association does
  not prove that the learned concepts are clinically meaningful.
- The original Set B is never to be used again as a fresh comparison
  or to pick V12 weights, concept count, risk margins, or architecture.

**Essential next independent evidence:** withheld synthetic generator
regimes with true latent factors and known interventions, followed by
a genuinely different patient/site dataset with the same outcome
definition and permissible data access. A pass on historical Set A DEV
alone cannot substantiate clinical deployment or novel causal factors.
