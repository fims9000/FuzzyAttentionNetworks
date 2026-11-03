# V13 — first frozen unseen-seed intervention falsification (2026-11-04)

**Evidence:** first successful independent-generator-seed [GitHub Actions 41728978959](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41728978959), artifact `V13-first-frozen-unseen-20270213-PYYAML-environment-fixed`, code commit `77434d08f317d794000086dff1cc911d34aa483e`. Generator seed **20270213** was frozen in `configs/medical/pd_v13_intervention_identifiability_synthetic_20261104.yaml` before evaluation (SHA256 `92a741d5215f024c4fc7faad67885a88ea19cf3391a91bbd4b96fab9cdb7abbf`). Original code SHA256 `1048f36254a832432fcf0b97aa3368d0d09c9b5b5bb269fe6c8d418b3dcb7dd5` and experiment code SHA256 `4f070cbb409b74edf0efb3b4da0d0f8ce73799452b0ad6466d2dd63c2f8df653`.

**Reproducibility:** [GitHub Actions 41728259002](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41728259002) passed **208 discovery tests**, one pre-existing warning. The first attempted scoring workflow [41728817034](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41728817034) aborted **before any generator code** because it imported `yaml` during source verification, before `pip install -e .`; no scoring, model training, or inspecting generated outcomes occurred. The corrected workflow only reordered dependency installation ahead of the unchanged SHA256/seed guard; algorithm, data, losses and results were not edited. The fully successful frozen-first-scoring run is **41728978959**. Prior local debugging used seed 20261104; no new-seed outcome was used to modify model selection or training weights.

## 1. Observed experimental outcomes

The synthetic generator uses (x=Az+\epsilon) with **randomized, externally declared single-latent-target interventions**, no downstream propagation, MCAR observations, fixed linear mixing and an unchanged emission mechanism across training/validation. With these strong assumptions, direct mean differences identify **sensor-level intervention response directions up to scale**, not general clinical causes. True (A) and latent (z) enter only the evaluation step. Each accepted run trains three independently initialized native concept models (11,23,37) plus no-observable-probe ablation and compares a training-only raw logistic predictive baseline.

| True K | V13 response direction oracle target-order cosine | Optimistic PCA axis cosine, control TRAIN only | V13 native AUPRC (3 seeds) | V13 no-observable-probe AUPRC | Raw logistic AUPRC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | **0.99589** | 0.85248 | 0.79853 | 0.80383 | **0.83369** |
| 5 | **0.99623** | 0.77325 | 0.81047 | 0.81048 | **0.83972** |
| 7 | **0.99512** | 0.69460 | 0.86166 | 0.86116 | **0.86206** |

These cosines compare **intervention response directions against known simulator source loadings**, not learned clinical meanings. The estimator receives **one environment label per externally targeted latent factor**, so K is constrained by intervention design; this is **not** unsupervised cardinality discovery. A high cosine is expected from the stated linear mean-shift formula and provides **implementation verification of those assumptions** rather than evidence of a new identifiability theorem or transfer to arbitrary unseen problems.

All three exclusive-target scenarios correctly pass the registered conditional gate. For true K=3,5,7 respectively:
- Minimum bootstrap q10 sign-preserving response-direction cosine: **0.993, 0.986, 0.983**.
- Smallest singular value / q90 bootstrap spectral perturbation: **4.88, 3.16, 2.31** (threshold >2).
- Observed contrast condition numbers: **1.14, 1.18, 1.21** (threshold <=30).
- Minimum signed same-patient contribution Spearman over ALL 3 cross-seed pairs and slots: **0.99765, 0.99897, 0.99769**.

Because model initialization shares the **same TRAIN-only measured environment response matrix** `D`, cross-seed signed agreement partly reflects anchoring *by construction*. It does **not** independently prove meaningful latent discovery or causal risk effects.

## 2. Negative controls: necessary provenance, not merely numerical full rank

| True K | Condition | Empirical numerical rank | Numerically resolved after bootstrap? | Conditional individual concept certificate | True target-order cosine |
| ---: | --- | ---: | --- | --- | ---: |
| 3 | Exclusive verified | 3 | Yes | **CERTIFIED (conditional)** | 0.99589 |
| 3 | Multi-target | 3 | Yes | **ABSTAIN — unverified exclusivity** | 0.45398 |
| 3 | Duplicate-target | 3 | No | **ABSTAIN** | 0.68968 |
| 3 | Sham | 3 | No | **ABSTAIN** | 0.44933 |
| 5 | Exclusive verified | 5 | Yes | **CERTIFIED (conditional)** | 0.99623 |
| 5 | Multi-target | 5 | Yes | **ABSTAIN — unverified exclusivity** | 0.51561 |
| 5 | Duplicate-target | 5 | No | **ABSTAIN** | 0.80845 |
| 5 | Sham | 5 | No | **ABSTAIN** | 0.21130 |
| 7 | Exclusive verified | 7 | Yes | **CERTIFIED (conditional)** | 0.99512 |
| 7 | Multi-target | 7 | Yes | **ABSTAIN — unverified exclusivity** | 0.20361 |
| 7 | Duplicate-target | 7 | No | **ABSTAIN** | 0.85668 |
| 7 | Sham | 7 | No | **ABSTAIN** | 0.27821 |

**Crucial limitation:** even duplicate-target and sham experiments produced a *numerically full-rank estimated matrix* because finite-sample noise makes singular values nonzero. The noise-adjusted singular-value bound/conditioning/angle gates correctly rejected these regimes, rather than naïvely equating matrix rank to recovered true factor count.

**Strong falsifier:** for all K=3,5,7, the mixed-target generator produced full-rank, well-conditioned, bootstrap-stable contrasts, yet **if the source metadata falsely claimed exclusive targeting**, the numeric + provenance gate would **CERTIFY all three cases incorrectly**. True target-order cosine was only 0.454/0.516/0.204. That is a constructive non-identifiability counterexample: `D=A B` with unknown invertible `B` cannot identify the columns of A. This test is positive evidence about **limitations**, not about recovering mixed-target clinical factors.

Thus V13 does **not** infer intervention exclusivity from observations. Every individually reported concept requires an independently audited experimental targeting mechanism. Without that, even apparently stable numeric axes must be named *environment-response components*, not causes.

## 3. What the native observable-response loss actually contributes

The model's **train-time** concept bottleneck produces native `c_k`, patient-specific signed contributions `phi_k=beta_k c_k`, and an exact additive target logit `b+sum_k phi_k` with no encoder-to-risk bypass. It includes a train-measured sensor-response probe regularizer, with an otherwise matched `probe_weight=0` ablation.

For K=3/5/7, evaluation-only off-target response fractions on new synthetic patient perturbations were:

| True K | Off-target fraction with probe | Without probe | Absolute change |
| ---: | ---: | ---: | ---: |
| 3 | **0.11678** | 0.15173 | −0.03495 |
| 5 | **0.19008** | 0.24774 | −0.05766 |
| 7 | **0.28071** | 0.34001 | −0.05930 |

The probe reduced off-target activation in these **three studied synthetic settings**, but it did **not** improve predictive AUPRC consistently (K3: 0.79853 vs 0.80383 without; K5: 0.81047 vs 0.81048; K7: 0.86166 vs 0.86116). Off-target activation remains substantial, especially K=7. The sensor perturbation is an **evaluation-only simulator operation**, not a factual patient-level do-intervention. No p-values, site-level uncertainty, clinical causal semantics or predictive superiority are claimed.

## 4. Interpretation and decision

- **Supported:** In this particular linear, independent, exclusive-target randomized synthetic system, the known-environment mean-shift method recovers observable sensor response directions and the native concept model is trainable with exact additive output. Conservative failure decisions reject unverified/mixed, duplicated and sham regimes at the **claim** level.
- **Unsupported:** Discovery of true medical factors from retrospective time series, identification under arbitrary nonlinear mixing or downstream propagation, mathematical novelty of the contrast identity, high-confidence population generalization from one simulator seed/family, superiority to the raw logistic baseline, or efficacy of intervention-source labeling absent experimental provenance.
- **Why this is scientifically different from V10/V11/V12:** those models attempted to discover concepts from predictive/observational structure and repeatedly suffered unidentifiable individual signs. V13 introduces an **independently documented experiment design** to break rotational ambiguity, making the conditions for individual concept identification **explicit and falsifiable** rather than quietly pinning an SVD sign. It does **not** solve observational concept discovery without such extra information.
- **Next experimental hypothesis (must be NEW generator/regime, not re-scoring this data):** controlled propagation from a randomized target into other latent nodes (total response vs direct response), observational missing-not-at-random mechanisms, partial intervention coverage, and independent alternative source generation. The first unmodified V13 results remain immutable and must not be optimized against their new-seed outcomes.

## 5. Evidence and replication

- [First successful frozen-seed V13 (CI 41728978959)](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41728978959).
- [SHA-verified V13 source import, 208 tests (CI 41728259002)](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41728259002).
- Current [V13 draft PR #9](https://github.com/fims9000/FuzzyAttentionNetworks/pull/9).
- Original full method and assumptions: `docs/medical/PD_CONCEPTFAN_V13_CONDITIONAL_INTERVENTION_PROTOCOL_20261104.md`.
- Aggregate files: `V13_INTERVENTION_EXCLUSIVITY_STRESS.csv`, `TRAIN_ONLY_INTERVENTION_CERTIFICATES.json`, `STATUS.json`, plus execution logs and source/config SHA256 in artifact `V13-first-frozen-unseen-20270213-PYYAML-environment-fixed`.
- **No patient records, patient labels or clinical datasets** included in these artifacts.
- Literature for prior intervention-identifiability methods: Squires et al., *Linear Causal Disentanglement via Interventions* (ICML 2023); Bing et al., *Identifying Linearly-Mixed Causal Representations from Multi-Node Interventions* (CLeaR 2024). The V13 linear contrast identity is not claimed as novel relative to them.
