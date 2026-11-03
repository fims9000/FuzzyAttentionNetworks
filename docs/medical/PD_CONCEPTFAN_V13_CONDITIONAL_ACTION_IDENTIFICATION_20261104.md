# ConceptFAN V13: conditional identifiability under randomized actions

**This is a separate research prototype, NOT a revision of the accepted IEEE PIERE article, and NOT a clinically validated model.** Original PIERE/V10–V12 findings (in particular, the negative cross-run reproducibility results) remain unchanged. The previously used PhysioNet Set B holdout is not accessed in any V13 script.

## Narrow mathematical result

Assume a dynamical model
`Z[t+1]=f(Z[t])+G U[t]+eps[t+1]`, `X[t]=B Z[t]+eta[t]`, with (i) individually targeted, independent and externally verified randomized actions `U[t]`; (ii) `G=diag(g_k)`, all `g_k>0`; (iii) full rank `Cov(U[t])` and sensor mixing `B`; (iv) independence of U from the current state and future noises, and (v) complete, time-aligned sensor measurement. Then

```
R = Cov(X[t+1],U[t]) Cov(U[t])^-1 = B G.
```

Proof: expand the first covariance, drop terms independent of randomized action, multiply by the inverse intervention covariance. **Consequently the oriented sensor-response column for each separately named action is identifiable in the population up to an unknown positive scalar g_k.** This is NOT a general nonlinear latent-factor recovery theorem and does not identify the absolute latent scale. If actions are mixed or dependent, the guarantee fails. The finite-sample estimator requires a full rank response, bootstrapped sign orientation and an action-exogeneity placebo check; passing these empirical checks does NOT establish randomization without external protocol documentation.

Why the requirement matters: observational `X=BZ` is unchanged under any simultaneous rotation `B'=BQ`, `Z'=ZQ` (Q orthogonal), and an appropriately transformed outcome coefficient preserves logits. Without an external coordinate system, individual latent concept directions remain ambiguous.

## Native training path and scientific controls

The estimated columns `R[:,k]` are normalized and the left pseudoinverse becomes a **fixed, interpretable sensor-to-concept operator**. A tiny trainable native bottleneck learns concept thresholds, widths and signed task coefficients, with `logit=b+sum_k beta_k c_k(x)`. No bypass, no post-hoc surrogate, no factor labels in the optimizer.

Strict rejections: missing or misaligned sensors; missing proof of independently targeted randomized actions; rank-deficient action covariance; collinear response directions; significant pre-action placebo (confounding); unstable action orientation across training-episode bootstraps. These are refusals to claim identification, not retrospective feature filtering.

## Executed in-memory local experiment (two distinct frozen seeds)

Each independently seeded generator has 1300 independent episodes, 1000 TRAIN, 300 never fitted validation, 12 observed steps plus 2 future risk steps, D=K+3 sensors, K=3/5/7 known latent factors, and three training random states 11,23,37. For each model, sample 800 distinct TRAIN episodes without replacement; no validation case is used for identifying sources, normalizing, or fitting the classifier. Latent trajectories and the planted sensor mixing are opened only **after** training for oracle audit.

| Generator | Latent K | Mean AUPRC | Mean signed true-factor Spearman | Joint stable signed individual contributions |
|---|---:|---:|---:|---:|
| Exploratory 20261104 | 3 | .7145 | .9984 | 3 / 3 |
| Exploratory 20261104 | 5 | .6725 | .9979 | 5 / 5 |
| Exploratory 20261104 | 7 | .6921 | .9977 | 7 / 7 |
| Fresh 20261204 | 3 | .6197 | .9982 | 3 / 3 |
| Fresh 20261204 | 5 | .7364 | .9980 | 5 / 5 |
| Fresh 20261204 | 7 | .6349 | .9974 | 7 / 7 |

Criteria for a **nontrivial repeatable signed contribution**: for all 3 independent fit pairs, action-labelled sensor-signature cosine >0.9, same-case signed effect Spearman >0.5, both fitted task coefficients absolute >0.08, and same fitted task coefficient sign. These criteria are relevant only under the fixed action dictionary and are not evidence for uncontrolled semantic factor recovery.

All 9 intentionally invalid intervention designs (confounded, collinear, no action across K=3/5/7) were rejected in each generator stage. 11 targeted tests passed locally. Observational PCA offered a diagnostic with lower **optimistic evaluation-only Hungarian** true-factor matching, but this is not an official V11/V12/MAGNETS comparison and cannot substantiate superiority over these prior models.

The first exploratory attempt used bootstrap episode selection with replacement *before* applying an independence-based permutation placebo, and correctly refused to certify some genuine randomized actions. The protocol was amended to select 800 **distinct** training episodes, before either successful run or the fresh generator's evaluation. This is an important constraint on the unit of statistical resampling.

## Boundaries on the novelty claim

- Concept count is known from the number of externally documented, separately randomized action targets, **not autonomously inferred from observational ICU series**.
- Positive action effects and independent actions are supplied by the simulator. True disease mechanisms, treatment causal effects on risk and arbitrary nonlinear factor identities are not recovered.
- Very high source correlations follow from strong experimental supervision through the action design. They are not comparable to observational V12's difficult identification problem without explicitly stating this changed task.
- Synthetic results cannot justify use of this method on observational PhysioNet, including existing Set A/B.
- Three model seeds and one generator family are insufficient for universal statistical inference.

Source: `src/conceptfan_discovery/instrumented_v13.py`, `intervention_synthetic_v13.py`, `experiment_v13.py`. Tests: `tests/discovery/test_instrumented_v13.py`. Workflow: `.github/workflows/conceptfan-v13-synthetic-conditional-identification.yml`.

Reproduce:

```bash
python -m pytest tests/discovery/test_instrumented_v13.py -q
python -m conceptfan_discovery.experiment_v13 --out-dir ./v13_exploratory --generator-seed-base 20261104
python -m conceptfan_discovery.experiment_v13 --out-dir ./v13_new_seed --generator-seed-base 20261204
```

**Do not merge these numbers into the original accepted PIERE manuscript as if they came from the same problem.** The next important scientific hurdle is identification under missing/partially-targeted interventions, multi-environment changes and independent intervention-generating mechanisms, with honest abstention whenever an assumption fails.
