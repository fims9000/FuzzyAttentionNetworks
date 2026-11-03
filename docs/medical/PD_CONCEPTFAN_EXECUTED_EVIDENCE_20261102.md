# PD-ConceptFAN: executed evidence and falsification audit, 2026-11-02

**Research status:** experimental code and separately validated analyses. Nothing
here modifies the accepted/archived PIERE paper or its previously used test.

## Frozen sources, verified code and runs

- Original 5-concept paper and trained models: `1dda55e287ccb613065e26d1ab58d5e8efe61e51`.
- New research branch: `experiment/predictive-dynamics-concept-discovery-v1`, draft [PR #4](https://github.com/fims9000/FuzzyAttentionNetworks/pull/4).
- [Med-CircuitBench 24-model v1 train/validation run PASS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41491449889); 3 predefined task-only/task+future/full objective ablations, 4 initialization x 2 data-order seeds each.
- [Frozen 24-checkpoint concept cardinality-selection evaluation PASS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41492954407), selects K on one half of validation and audits on the other.
- [Variable planted dimension 3/5/7 stress PASS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41493823726); 27 trained networks, hidden states accessible to evaluator ONLY.
- [Official raw PhysioNet Set A 12-model validation PASS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41493279876), no manual five-concept proxy targets, sealed calibration/test, aggregate clinical reports only. The official ZIP SHA256 is
  `1433638192fda622e8bb11a4559b8af2dbb1c69f0e1c3b5b606cc12aabd0a4c5`;
  Outcomes-a.txt SHA256
  `2613ea60ccda29f87571a7d6b09ad130858a8ccd66325bd073365925026883c2`.

## Exact numerical observations

### Original synthetic Med-CircuitBench protocol

| Result | task-only | task+future | full |
|---|---:|---:|---:|
| Full-model validation AUPRC | 0.8210 | 0.8189 | 0.8253 |
| Cross-run Spearman **excess over patient-permuted marginal null** | 0.2890 | 0.2815 | 0.1996 |
| Active slots after learned hard-concrete gate | 10 | 10 | 10 |

The original L0 relaxation **failed** to select compact K. The full objective
also did not improve the actual patient-specific cross-fit stability over the
simpler task-only ablation.

When native *trained* slots are pruned on inner validation and scored on
separate outer validation:
- full method: (K=2\ldots3), mean (2.375); audit AUPRC before
  0.7472, after 0.7559; matched hidden oracle concepts only **47.5%**
  coverage, coverage-adjusted average absolute correlation only 0.3115.
- task-only: K mean 2.75; audit AUPRC 0.7478 before, 0.7668 after;
  55% oracle state coverage.
- task+future: K mean 2.75; audit AUPRC 0.7467 before, 0.7667 after;
  55% oracle state coverage.

This is an operational finding of a small *task-sufficient concept subset*,
**not proof of the original five planted mechanistic variables**. A best
matched-only correlation conceals missing factors and cannot substitute
coverage-adjusted recovery.

### Independent synthetic 3/5/7 hidden-factor generator

For **every** (K_{\mathrm{truth}}=3,5,7), and every ablation,
the fixed procedure chose (K_{\mathrm{selected}}=1).

| True latent count | All methods' chosen count | Max oracle coverage possible |
|---|---:|---:|
| 3 | 1 | 1/3 |
| 5 | 1 | 1/5 |
| 7 | 1 | 1/7 |

The full objective's coverage-adjusted absolute latent correspondence was
only 0.0504, 0.0380 and 0.0470, respectively. It **failed** the claim that
selected predictive slot count recovers true latent dynamical dimensionality.

This does **not** establish that the learner is broken on every possible
process. The synthetic generator's binary outcome has a limited signal,
and observed future dimensions are noisy, so a small predictive subset may
be adequate. It establishes that **the current training and selection
protocol does not make factor recovery identifiable**, even when planted
latent truth exists for external evaluation.

### Actual clinical PhysioNet Set A

On official 4,000 ICU records, the original stratified split is preserved:
2,400 train / 600 validation / 400 calibration sealed / 600 test sealed.
The new discovery experiment uses **only 36h historical data**, with the
following 12h observed measurements as forecast targets. It never imports
the five clinical proxy concept definitions.

| Outer validation result | task-only | task+future | full |
|---|---:|---:|---:|
| Trained models | 4 | 4 | 4 |
| Mean selected K | 1 | 1 | 1 |
| AUPRC before cardinality selection | 0.2275 | 0.2332 | 0.2247 |
| AUPRC after selection | 0.2011 | 0.1898 | 0.2002 |
| Future observed-measurement RMSE after selection | 0.8400 | 0.8325 | 0.8325 |
| Fraction of model pairs permitting >=2-concept rank agreement | 0% | 0% | 0% |

Validation-set mortality prevalence over all 600 cases is 83/600 =
0.13833. The outer audit half has its own prevalence and cannot be
assumed identical. Clinically **no reproducible multiconcept finding was
demonstrated**; four independent runs choose quite different operator
combinations (examples: bilirubin/FiO2 vs arterial pressure vs lactate).

Exactly additive computational attribution remains correct, but is **not**
semantic correctness or a clinical interpretation certificate. Report the
current result as negative.

## Why the v2 method is a justified next experiment, not a corrected success

**Mathematical ambiguity:** Mortality label alone admits a scalar sufficient
statistic (P(Y=1\mid X)). Even mortality plus next-hour prediction may
tolerate a compressed, noisy, one-dimensional representation.

So v2 adds a third, concept-mediated **masked observation reconstruction**
constraint, with 20% of actually observed input measurements deliberately
hidden from the encoder and used only as targets for its reconstruction
decoder. No direct reconstruction shortcut bypasses concept trajectories.

The full training loss acquires a separately pre-declared term
(\lambda_m \sum M^{\mathrm{hidden}}_{tj}
\operatorname{Huber}(D(C_{tj}),X_{tj})).
Native K selection preserves **outcome quality, future prediction, and
masked reconstruction** separately, using one half of validation. The
other half is untouched until reporting.

Ablations:
- `task_only`: outcome BCE only;
- `full_without_denoising`: outcome, future, dynamics, sparsity,
  redundancy, nonconstant constraints without reconstruction;
- `full`: above plus masked observed-feature reconstruction and
  an observed-input masking augmentation.

v2 studies K3,5,7 systems with **fresh seeds and a separately frozen YAML**.
Its success is not guaranteed. If it still collapses to one factor, the
next methodology must introduce stronger identifiability assumptions
(e.g., multiple observed environments/interventions) rather than
misleadingly force K to equal the planted dimension.

## What is proven vs not proven

**Proven by code contracts:** no manual concept targets, no direct hidden
classifier bypass, exact additive logit, causal-in-time measurable operator
dictionary, explicit slot definition, and train-only preprocessing.

**Observed in experiments:** valid software execution, some predictive
performance, small operational predictive subsets; no robust true-state
recovery and no clinical semantic validation.

**Not claimed:** uniquely identifiable physiological factors,
transportability, counterfactual treatment effects, clinical readiness,
multi-method significance, optimization/global minimum of MDL, genuinely
independent cross-validation of new hyperparameters.

The basic term "causal operator" means *does not inspect future time
points*, not *causal effect*. The original paper is still important:
functional attribution exactness does not imply semantic completeness.
This new negative experiment is consistent with it.

## Reproduction

```bash
PYTHONPATH=src:. python -m pytest tests/discovery -q
PYTHONPATH=src:. python scripts/medical/run_predictive_dynamics_discovery.py \
 --config configs/medical/predictive_dynamics_discovery_v1.yaml \
 --out-dir artifacts/medical/pd_conceptfan_v1_001
```

The K3/5/7 benchmark:
```bash
PYTHONPATH=src:. python -m conceptfan_discovery.variable_factor_experiment \
 --config configs/medical/pd_discovery_factor_count_stress.yaml \
 --out-dir artifacts/medical/pd_factor_count_v1_001
```

The clinical experiment requires **official raw PhysioNet 2012 Set A and
Outcomes-a.txt** (not the archived metrics-only ZIP):
```bash
PYTHONPATH=src:. python -m conceptfan_discovery.clinical_experiment \
 --config configs/medical/predictive_dynamics_physionet_v1.yaml \
 --out-dir artifacts/medical/pd_clinical_v1_001
```

All experiments are recorded as exploratory and must not be promoted to
the original frozen paper or to any medical deployment.
