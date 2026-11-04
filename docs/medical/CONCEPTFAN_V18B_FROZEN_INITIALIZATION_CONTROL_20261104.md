# V18b matched signed-initialization falsification — preregistered control

## Motivation
Initial V18 native experiment on the new synthetic generator found improved AUPRC with correct negative TRAIN-screened risk orientation. However the original unrestricted K3 comparator starts every new coefficient with **positive** raw +0.05. When independent evidence is negative, a sign-constrained new slot begins with effective negative coefficient, which is a different **initial condition**. Before attributing a gain to the sign constraint itself, compare models with *identical initial effective native signed contributions*.

## Frozen 27-opportunity design
Same independent synthetic generator and outcome modes as V18 native ablation: positive association, negative association and risk null; K2 model initially trained for six epochs then expanded, 6 continuation epochs, seeds 11/23/37, 800 TRAIN + 200 untouched validation + 80 sealed test. Candidate direction from first 350 train episodes; risk sign from separate 300 TRAIN examples. Keep all three arms per mode and seed, including abstentions.

1. **unrestricted_correct_signed_start:** K3 native model with unrestricted readout initialized raw +0.05 for independently positive sign, -0.05 for independently negative sign.
2. **restricted_correct_signed_start:** K3 with train-screened native sign policy, raw +0.05, producing the **same initial effective signed logit coefficient** as arm 1 (including for negative sign). Explicitly assert their logits are identical at initialization over 32 disjoint validation episodes *without using labels*.
3. **unrestricted_opposite_signed_start:** K3 native model with unrestricted readout deliberately started on the opposite sign, a negative optimization control.

Under risk-screen ABSTAIN (null association) all three signed-orientation arms **ABSTAIN** and are recorded; none should be silently assigned zero AUPRC.

All fits use same model, V10 objective, optimizer, patient ordering and artificial missingness. The only difference between arms 1 and 2 is the sign-policy reparameterization after an *identical initial effective signed logit*. The V10 native bottleneck is unchanged; no patient-level artifacts or oracle latent data.

## Evaluation
All 27 opportunities with complete counts; heldout AUPRC/AUROC/Brier, actual nonzero native signed-effect magnitude, exact logit decomposition, native concept-to-measured-proxy correlation and cross-seed patient-specific signed effect Spearman on the common untouched validation cohort. No statistical significance claim with three seeds.

If arm 1 and arm 2 are equal, the earlier gain was due to correct initial orientation rather than independent algorithmic advantages of sign constraints. The physical semantics of a candidate are still not identifiable from these data. The prior V18 18-case risk screen and 45-case native experiment remain immutable independent execution records.

Implementation: `src/conceptfan_discovery/experiment_sign_initialization_v18b.py`. Accepted PIERE camera-ready unchanged.
