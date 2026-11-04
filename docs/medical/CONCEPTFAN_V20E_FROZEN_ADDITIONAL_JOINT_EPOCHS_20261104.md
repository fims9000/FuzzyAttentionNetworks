# V20e — frozen additional joint training vs head-only training budget (2026-11-04)

## Remaining mechanism hypothesis
V20a found that LBFGS and AdamW with the same frozen concept summaries reach nearly identical native head AUPRC, regardless of near-unity class weights. V20b and the genuinely unused TEST seed V20d showed this remains useful on distinct synthetic processes and conservative TRAIN-gated selection. However **additional head-only optimization** and **additional optimization budget** are still confounded: maybe ordinary ConceptFAN would achieve the same gains if jointly trained longer.

This experiment compares the original native K3 model after 6 K2 pretraining + 6 K3 training against more joint **end-to-end** epochs, versus optimizing **only the existing risk coefficients** on frozen K3 concepts. No changes to observation process, native concept graph, forecasting/reconstruction, data cohort, labels or published PIERE article.

## Frozen exact arms
Return to the previously inspected **simple V18 synthetic family** (not fresh generalization): generator seed 20411204, TRAIN 800, VALIDATION 200, sealed TEST 80. Outcome modes positive, negative, independent null. Model seeds 11,23,37 (all nine original model pairs).

From **exactly the same** 6-epoch K2 then 6-epoch K3 model with independently TRAIN-screened new measured source direction and initial raw ±0.05; in null outcome force a zero new coefficient strictly for negative control:

- `baseline_joint_6plus6`: no further updates.
- `extra_joint_6`: an additional six epochs of the **same original V10 combined task+forecast+masked-reconstruction+regularization loss**; new AdamW optimizer (restart) and deterministic new shuffle/mask-seed offset.
- `extra_joint_12`: 12 additional epochs, implemented as two separately initialized six-epoch original V10 objective blocks with different deterministic ordering/masking offsets.
- `frozen_head_adamw300`: no new concept-generator/forecast/care updates, only 300 full-batch AdamW steps to actual original decision coefficients and bias with TRAIN-only unweighted BCE and effective-beta L2=0.001, exactly the verified V20a arm.

Three modes × three seeds × four arms = **36 complete model evaluations** (not 36 independent initializations). There are 9 pretrained backbones; all arms must copy that backbone before updates. All arms use only TRAIN fitting. Validation is for descriptive heldout evaluation and no seed/epoch selection. Total optimization compute/exposures are NOT exactly matched: full-batch head steps and stochastic joint minibatch steps have different cost and objectives; do not attribute any result solely to optimization algorithm.

## Audit and metrics
AUPRC/AUROC/Brier and TRAIN binary logloss; exact native additive decomposition; average abs learned signed slot contributions; per-patient native concept summary drift from the fixed baseline on the same validation histories; process/forecast parameters may update ONLY in the extra_joint arms, while they must remain bit-for-bit identical in frozen_head arm. No raw-to-risk bypass, no post-hoc classifier, no hidden factor oracle.

**Interpretation guard:** if extra_joint_12 closes the gap, the previous improvements were largely explainable by training budget rather than independent risk-phase architecture. If extra_joint_12 remains substantially behind native head-only, separate head training is more effective per these unbalanced budgets, perhaps because continuously changing concept definitions interfere with task head convergence, but this is not a rigorous gradient-causality proof. Always report null worsening.

Follow-up beyond this experiment must concern semantic grounding and real human review, not another non-identifiable synthetic source theorem. Accepted PIERE camera-ready untouched.
