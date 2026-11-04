# V20c — frozen independent-TRAIN acceptance gate for optional native head refit (2026-11-04)

## Problem
V20b showed that directly refitting the **native** ConceptFAN risk weights improved heldout synthetic TEST AUPRC on three distinct dynamical families, but degraded AUPRC when target is independent of all measurements. The V18 risk screen demonstrated that a predictive instrument artifact can still pass a task-relevance gate, hence safety requires keeping *predictive evidence* distinct from *physical semantic validity*. The head refit should not be enabled on every task, especially null-like tasks.

## New capability
An opt-in **TRAIN-only, model-level acceptance check** for the refitted native risk head. It compares the original model and a candidate refitted *same-model* risk head on an independent **fit-disjoint** labeled gate cohort and returns ACCEPT or ABSTAIN/KEEP_ORIGINAL. If refused, the original model is retained. This is a **prediction improvement gate**, not a guarantee of physical concept meaning.

## Frozen generator and partitions
Use V20b's distinct four *structural* simulator families (nonlinear lag, regime switch, state-informative missingness, independent-label null). Unlike V20b, generate four **separate independent cohorts**:
- TRAIN core, generator seed 305011, 800 episodes; all normalization and model fitting use ONLY TRAIN core observed entries.
- TRAIN holdout gate, independent generator seed 305014, 300 episodes; no model fitting, head fitting, normalization or selection uses these observations/labels except the final one-time acceptance rule.
- Synthetic TEST, independent generator seed 305013, 300 episodes. It is **never** used for fitting/acceptance/threshold tuning. In one frozen run it is evaluated after gate decision, for unbiased descriptive comparison of the predefined policy.
- The previous V20b validation seed 305012 is NOT used in this study.

Model: V10 K3 with 3 predeclared observed sensor axes (not automatically discovered physiological identities), 12 original joint epochs, adaptive operator gradient enabled, seeds 11/23/37. A duplicate is refitted on core TRAIN only using fixed V20 unweighted LBFGS 120 iterations/L2=0.001. No new inference head, raw input bypass or oracle factors. Original model and refit undergo **the same native concept graph**; only decision coefficient/bias differ.

Acceptance rule predeclared:
1. Independent gate outcomes contain both classes.
2. Refitted risk probabilities have a **positive** point-biserial correlation with gate labels, one-sided permutation p ≤ 0.01 with 499 permutations.
3. Mean per-episode gate log-loss improvement (baseline − refit) ≥ 0.005.
4. Percentile 90% confidence interval of that mean improvement from 499 patient-level bootstrap resamples has a **strictly positive lower bound**.

Otherwise **KEEP_ORIGINAL** and do not apply the potentially overfit refit. These numerical thresholds are engineering hypotheses, not population-wide certified family-wise error bounds; the one-sided association permutation test does not establish medical causality. Same gate cohort reused across 3 model initialization seeds, so tests are statistically dependent and not independent population replications. No tuning thresholds after gate/test results.

3 native model initialization seeds × 4 families = 12 original models and 12 refitted candidates, 12 acceptance decisions, and 36 test evaluations for ORIGINAL/CANDIDATE/SELECTED; plus 4 raw-input predictive baselines from V20b (previously fixed). Always report all 12 gate outcomes, accepted and rejected, including test performance of rejected offline candidates for transparent negative-control analysis, not actual deployment.

## Critical success/failure
- Desired: accept refit on some predictive families and reject all or most label-null cases; preserve external TEST AUPRC/Brier compared with unguarded refit.
- A positive result does not prove improvements everywhere, since conservative gate can reject useful refits and a permissive gate can accept spurious luck.
- Never present a signed, predictive sensor pattern as a newly identified physiological mechanism. That requires independent instrument/provenance evidence and real expert validation.

V18–V20 accepted IEEE PIERE camera-ready remains unchanged. Study stays exploratory despite independent generated synthetic TEST.
