# V18c — fixed predictive baselines (predeclared prior to first CI run)

**Question:** do V18 positive-control gains provide competitive predictive performance or merely demonstrate a working native concept expansion? Compare basic non-concept classifiers on **exactly the same** training and validation episodes and the same past observational inputs.

## Freeze
Source `src/conceptfan_discovery/experiment_external_baselines_v18c.py` calls `make_data` from the already frozen V18 native study without changing data generation. Generator seed 20411204, three outcome modes (positive/negative/null), 800 TRAIN and 200 heldout validation episodes, 80 untouched test episodes, 24 past timesteps, five observed channels. Report **all 12 fitted baseline/mode combinations**.

Four fixed unsupervised/standard predictive controls:
- DummyClassifier constant TRAIN outcome prevalence (no discrimination).
- L2 logistic regression C=1 on bounded observable recency/level/early-late change and recent variance features, using train-only feature standardization.
- L2 logistic regression C=1 on all 24 × 5 historical raw sensor values, using train-only standardization.
- HistGradientBoostingClassifier with 150 iterations, at most 15 leaves, minimum 20 train samples per leaf, learning rate 0.05, fixed seed 3, same low-dimensional observable history as logistic-recent.

All controls use **past observations only**, no latent factor oracle, no concept reviewer labels and no future trajectories as features. No hyperparameter tuning or threshold selection using heldout outcomes. Metrics: AUPRC, AUROC, Brier, log-loss and prevalence. No code/data preparation changes to ConceptFAN results. Comparison is exploratory because this validation was already inspected in V18.

## Limitations
These are **predictive competitors**, not matched intrinsic or post-hoc explainability baselines. Any predictive superiority of simple logistic models does NOT refute the value of a faithful, auditable native concept bottleneck. Conversely ConceptFAN is NOT entitled to claim competitive prediction accuracy without such comparisons. Official MAGNETS and other automatic-concept models are not implemented here; literature-level comparisons must not be fabricated. No real hospital, physical validation, or robust multi-generator generalization has been established.

The expected next architectural target, depending on results, is joint AUPRC/interpretability on external datasets, not another short synthetic architecture variant. Accepted PIERE paper remains untouched.
