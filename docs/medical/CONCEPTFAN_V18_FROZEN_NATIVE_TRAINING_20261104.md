# V18: native retraining after independently screened concept discovery (frozen before execution)

This study follows the successful 18-case **observational** risk screen. It tests whether a sign associated with a TRAIN-only measured concept can be attached to the **same native V10 risk coefficient** during actual model fitting, and whether additional native measurement grounding stabilizes its patient-specific contribution. It does not test causal physiology.

## New dataset / sealed partitions
Generator seed 20411204, distinct from prior V16 and V17 datasets; 1,080 fully synthetic episodes (800 TRAIN, 200 heldout validation, 80 sealed test), 24 observed time steps and six future steps, five sensors and three AR hidden factors. The third factor projects to sensor_2 and, for the positive and negative modes, controls a stochastic future binary outcome. In the null mode the outcome is independent of that factor. Hidden source trajectories are *not* passed to the model, data object or risk screening.

Source proposal is computed on first 350 TRAIN episodes without labels, using preexisting K2 observable directions. Independent predictive risk screening uses 300 other TRAIN episodes (no overlap with source discovery), 199 label permutations, 199 bootstraps, 95% CI, two-split sign check and abs correlation threshold 0.15. All 800 TRAIN episodes may subsequently be used for supervised model fitting; the 200 validation episodes are never used to select model parameters. All three mode-specific screens are executed before model comparison.

## Matched architecture arms
Three model initialization seeds (11,23,37), five nominal arms, three independently generated **outcome modes** (positive, negative, null); thus 45 arm/seed/mode *opportunities*, including abstentions.

Each arm begins from the same 6-epoch trained V10 K2 model (adaptive feature gradients true). Then:
1. **continued_k2:** continue K2 for six matched epochs, unchanged bottleneck.
2. **unrestricted_k3:** new native K3 admitted without risk gate, its new head seeded +0.05 and continued six epochs.
3. **screened_k3:** only if the risk screen is ACCEPT; new native K3 gets the independent TRAIN signed predictive orientation and continues six epochs.
4. **screened_grounded_k3:** same sign screen plus bounded measurement-only grounding loss with coefficient 1.0, acting on the actual raw native concept activation; no post-hoc student head or oracle state.
5. **intentionally_wrong_sign_k3:** sign explicitly inverted relative to independently screened outcome association. This is an **adversarial scientific ablation**, not a deployable workflow.

A rejected risk screen records ABSTAIN for arms 3–5, **not** a zero-valued model outcome. In all cases, original K2 and K3 unrestricted models are trained as comparators. K3 admission starts with zero readout to preserve old predictions, then the explicit training start seeds ± oriented coefficient magnitude 0.05. Training continues on the same examples, optimizer, scheduler, masking RNG, batch order, loss terms and epochs across arms; within-seed comparisons are paired. No early stopping/hyperparameter tuning on validation. The sign is **imposed by construction**, not independently learned stable.

## Evaluation and falsification
Heldout outcome AUPRC/AUROC/Brier. Maximum exact-additive logit decomposition error. Mean absolute patient-specific new slot logit contribution (guard against trivial zero/sign collapse). Spearman of native raw concept activation versus the separately defined observable sensor-direction score on validation. Cross-seed **patient-level rank correlation** of new slot signed logit contributions across 3 dependent seed pairs. Report all 45 opportunities, all training failures and all screening refusals, not only successes. No p-values with three model seeds. The outcome-null mode may refuse all signed policies, which is an acceptable negative result.

## Scientific limits
This is an intentionally favorable synthetic generation family with complete measurement coverage, known initial K2 undercapacity, and three seeds. It does not establish general gains over official MAGNETS, CEM or other concept models, clinical utility or physiological source identity. A predictive colored sensor artifact remains able to pass the V18 gate (proven in the first study). Sign agreement of oriented risk coefficients is trivially enforced; only patient-specific *magnitude/rank alignment* can provide nontrivial evidence of reproducibility. Grounding to a measured signal is not a causal intervention.

## Reproducibility
Executable script: `src/conceptfan_discovery/experiment_native_risk_v18.py`; test: `tests/discovery/test_risk_screen_v18.py`; workflow stores CSV/JSON and Git commit SHA. Original accepted PIERE camera-ready remains unchanged.
