# V14 — Multi-environment covariance contrasts without a known action map

**Scope:** purely synthetic, conditional, noninterventional source discovery. The accepted PIERE manuscript, V10–V13 findings and previously consumed PhysioNet Set B are untouched.

## Identifiability and mathematical assumptions

Let X_e = B Z_e + eta_e, where e is one of three observed environment IDs, B is an unknown D×K full-column-rank mixing matrix, the latent factors have diagonal Cov(Z_e)=diag(v_e), and measurement-noise covariance Ω is invariant across environments. Define:

    D1=Cov(X|e=1)-Cov(X|e=0)=B diag(v1-v0) Bᵀ
    D2=Cov(X|e=2)-Cov(X|e=0)=B diag(v2-v0) Bᵀ.

If each v1-v0 coordinate is strictly positive and ratios (v2-v0)_j/(v1-v0)_j are all distinct, whitening D1 and eigendecomposing the whitened D2 identifies sensor mixing directions up to **permutation, sign, and scale**. No intervention target labels or true K enter fitting. This is a *classical* second-order blind-source-separation argument, NOT a novel mathematical theorem, general nonlinear ICA guarantee, biological causal identification or clinical disease concept discovery.

Prior art: Belouchrani et al., IEEE TSP 1997, https://doi.org/10.1109/78.554307 ; Pan et al., 2022 review, https://pmc.ncbi.nlm.nih.gov/articles/PMC9540980/ ; Khemakhem et al., AISTATS 2020, https://proceedings.mlr.press/v108/khemakhem20a.html .

## Implemented research pipeline

The user supplies only complete historical sensor time series and three environment labels (0/1/2). The train-only estimator selects the number of statistically resolved covariance-contrast directions relative to a 99-permutation null, with patient-level bootstrap (32 in the full study) to test direction stability. Ordering uses distinct environment variance ratios; orientation uses a deterministic largest-sensor-loading sign, **not a clinical label**. Ambiguous ratios, unstable bootstrap directions and missing measurements cause ABSTAIN.

The native ConceptFAN-style head has learnable concept thresholds/widths and a direct additive signed logit with no encoder bypass. Comparators are logistic regression on exactly the **same identified source inputs**, PCA+logistic with the same train-selected K, and an unstructured raw-sensor logistic model. Source-oracle Hungarian matching is performed only for evaluation after all models are trained; absolute Spearman is optimistically aligned and is not proof of identifiable clinical meaning.

## Frozen data protocol and independently seeded evidence

Each regime has K_true=3/5/7, three environments, 4,000 independent episodes per environment, 12 historical time steps and four unseen future steps determining binary risk. Each environment has 3,200 original TRAIN and 800 evaluation episodes; each independently initialized fit uses 2,800 of the TRAIN episodes. The generator uses independent Gaussian latent AR(0.75) states, distinct environment-modulated variances, nonorthogonal linear sensor mixing and common correlated sensor noise. Pilot generator seed 20261104 was used for developing the forecast signal; a different generator seed 20270104 was then used without further hyperparameter changes.

**Fresh generator 20270104, three training fits each; descriptive means:**

| True K (oracle only) | Certified fits | V14 optimistic abs factor rho | PCA optimistic abs factor rho (same inferred K) | V14 native AUPRC | Same-factor logistic AUPRC | Raw logistic AUPRC |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 3/3 | 0.9912 | 0.7672 | 0.6556 | 0.6565 | 0.6573 |
| 5 | 3/3 | 0.9696 | 0.7080 | 0.6745 | 0.6823 | 0.6843 |
| 7 | **2/3** | 0.9549 on accepted fits only | 0.6979 on accepted fits only | 0.6961 on accepted fits only | 0.6989 | 0.6995 |

At K=3 and K=5, all three fit pairs pass strict same-sensor-direction cosine>0.9, same-episode signed contribution Spearman>0.5, |risk coefficient|>0.08 and risk-sign agreement, giving **3/3 and 5/5** individually reproducible mathematical effects. At K=7, one fit abstains because bootstrap directions/variance gaps are unresolved, so NO all-three-runs certification is permitted. Mean rho 0.9549 and AUPRC at K=7 exclude abstained runs and cannot be presented as general performance.

There is **no demonstrated predictive advantage**: same-factor logistic and even raw-sensor logistic often have higher AUPRC than the native neural head. The positive observation is conditional source-recovery under deliberately favorable assumptions, not improved classification.

## Negative controls and constructive impossibility

The estimator abstains on identical environment covariances, mean-only shifts (because covariances are centered), and complete-sensor violations. With equal variance-ratio profiles the individual axes can rotate; an estimator may abstain or undercount K. Neither result proves all latent factors have been found.

Changing measurement noise violates the theorem's assumptions; a covariance-only estimator cannot establish their validity. In particular, the following two mechanisms have the SAME observed covariance for every environment:

    A: source covariance D0+Δ_e, sensor noise covariance Ω
    B: source covariance D0, sensor noise covariance Ω+BΔ_eBᵀ.

The adversarial test constructs randomized Gaussian sensor noise changing along source-mixing directions while the true latent source variances remain exactly constant. The finite-sample V14 procedure **falsely certifies 3 statistically stable source directions** with bootstrap cosine q10>0.9, but they do not represent identified original latent states. This is an explicit *failure control*, not a success. No additional bootstrap of the same covariances can solve this non-identifiability.

## Novelty and boundary of claims

Second-order source separation is not new. Conditional integration into a native additive temporal concept head, rigorous train-only source count screening, bootstrap abstention, matched predictive baselines, independently seeded oracle audit and observable-noise impossibility demonstration are prototype engineering contributions, **not demonstrated theoretical novelty**. The entire experiment is synthetic; there is no evidence of identifiability of clinical concepts from the observational PhysioNet cohort, no source names, no unknown nonlinear mixing recovery, no independent hospital validation, and no result on partial observations.

Code: src/conceptfan_discovery/environment_contrast_v14.py, environment_synthetic_v14.py, experiment_v14.py. Tests: tests/discovery/test_environment_v14.py, test_environment_adversary_v14.py. GitHub Actions: .github/workflows/conceptfan-v14-three-environment-synthetic.yml.

Next objective: distinguish sensor-environment shifts from genuine source shifts using *additional independently measured constraints or valid instruments*, and evaluate identifiability when observations are missing. Do not reinterpret the existing Set B or old PIERE results as confirmation of V14.