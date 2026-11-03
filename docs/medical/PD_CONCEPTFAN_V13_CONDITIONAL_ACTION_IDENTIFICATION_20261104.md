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


## Priority-related work and *non-novel* identification equation

**Important scientific correction:** the covariance identity `R=Cov(X[t+1],U[t])Cov(U[t])^-1` under randomized action assignment is a straightforward linear moment / instrumental-variable-style argument. It is **not a new theorem** in causal representation learning. We must not market the synthetic recovery as the first solution to causal concept identification.

Relevant established theory:
- Varıcı, Acartürk, Shanmugam, Tajer, *General Identifiability and Achievability for Causal Representation Learning*, AISTATS 2024, https://proceedings.mlr.press/v238/varici24a.html . Covers much more general latent and observation models and uncoupled intervention settings.
- Varıcı et al., *Score-based Causal Representation Learning: Linear and General Transformations*, JMLR 2025, https://jmlr.org/papers/v26/24-0194.html . Establishes identifiability under suitable hard interventions and partial results for soft interventions.
- Bing, Ninad, Wahl, Runge, *Identifying Linearly-Mixed Causal Representations from Multi-Node Interventions*, CLeaR 2024, https://proceedings.mlr.press/v236/bing24a . Addresses multi-node interventions, beyond our known diagonal action-target assumption.
- Morioka, Hyvärinen, *Causal Representation Learning Made Identifiable by Grouping of Observational Variables*, ICML 2024, https://proceedings.mlr.press/v235/morioka24a.html . An alternative route using grouped observational variables rather than targeted interventions.
- Yao et al., *Unifying Causal Representation Learning with the Invariance Principle*, ICLR 2025, https://proceedings.iclr.cc/paper_files/paper/2025/hash/85381f4549b5ddf1d48e2e287d7d3d15-Abstract-Conference.html . Distinguishes representation invariance from genuinely causal assumptions.

**Defensible V13 contribution at this stage:** a controlled integration of externally identified action-response coordinates into an exact train-time additive temporal concept bottleneck, together with honest per-concept abstention, randomized-action and confounding negative controls, independently refitted signed-logit audit, and source-oracle evaluation. Novelty relative to existing CRL and concept-bottleneck techniques is **not established** without a matched implementation and performance comparison. The strength of the identification assumption is substantial; it does not solve the user's original stronger objective of automatically identifying concepts from unlabeled observational ICU time series.

Next pre-registered comparator should include an ordinary logistic regression on the **same identified action-response coordinates**, and potentially an established CRL/interventional representation method adapted with equal prior information. All comparison weights and baselines must be frozen before generating a third independent test seed; the two existing generators cannot become fresh validation again.


## Third independent synthetic generator: ordinary logistic comparator (20270104)

**CI verification:** [GitHub Actions 41733269434](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41733269434), SUCCESS, downloadable aggregate artifact `conceptfan-v13-third-seed-matched-logistic-1`.

After acknowledging that the intervention cross-covariance identification formula is a classical conditional moment identity, a simple `sklearn.linear_model.LogisticRegression` baseline was specified. On generator seed **20270104**, both the V13 native sigmoid/tanh concept bottleneck and ordinary logistic regression use the **same estimated TRAIN-only randomized-action response matrix/pseudoinverse, same 800-of-1000 training episodes, same action labels and the same 300 untouched validation episodes**. No opportunity to improve source identification is given to one method but not the other; both have exact additive signed contributions. Logistic parameters C=1, lbfgs, 1000 max iterations, without tuning on those validation outcomes. The native head uses its previous 85-epoch recipe and no new hyperparameter search.

| True factors | V13 native mean AUPRC (3 fits) | Ordinary logistic mean AUPRC (3 fits) | Native strictly stable signed effects | Logistic strictly stable signed effects |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 0.6490 | 0.6385 | 3 / 3 | 3 / 3 |
| 5 | 0.6177 | **0.6327** | 5 / 5 | 5 / 5 |
| 7 | 0.5990 | **0.6276** | 7 / 7 | 7 / 7 |

The ordinary logistic model also had higher AUROC and lower Brier score for K=5/7. **This falsifies a general predictive-accuracy or attribution-stability advantage of the V13 nonlinear concept head over a simple, source-matched linear method.** The successful source recovery is principally explained by known, independently targeted action assignments. The extra nonlinear concept layer does not consistently improve the outcome, and the simple logistic baseline attains the same source-aligned reproducible signed-effect count.

Statistical qualification: only three fits of one additional synthetic generator; reported means and comparisons are descriptive. This is a **frozen third seed**, not patient validation and not an established population-level performance ordering. An official modern causal representation learning baseline with matching data access has not been reproduced.

## Blind spot: falsely declared action identity

A new adversarial test constructs randomized **multi-target** interventions while presenting false metadata that each action targets only its same-index latent source. The covariance response matrix is still correctly estimated, its rank and placebo checks pass, and the resulting action response columns are stable; however their purported one-to-one latent factor names are wrong (same-index true-source similarities below 0.1). This shows an unavoidable distinction: **the data can identify action responses, but cannot authenticate the scientific meaning of an unverified action-to-latent-target map**. External experiment design and metadata are necessary for the stronger concept identity claim. The estimator must not be described as automatically recognizing false treatment target labels.

Overall V13 is a valid **conditional, synthetic, action-signature proof of concept**, not yet a demonstrated new generalizable neural architecture nor a solution to unlabeled observational concept discovery. The comparison favors publishing the negative-vs-positive conditions and identifiability limits, rather than claiming universal superiority.
