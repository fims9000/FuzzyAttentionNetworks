# ConceptFAN-Anchor v1 — semantic concept identification

**Status:** new, isolated **exploratory** study. The previously frozen
Med-CircuitBench/PhysioNet runs, checkpoints, PIERE camera-ready results, and
test split are immutable. This is not a published performance improvement.

## Motivation

The legacy ConceptFAN independently learns the temporal concept attention,
per-concept fuzzy centers/widths and decision coefficients. Distinct
parameterizations can have similar final predictions but disagree about which
concept generated the logit, especially with correlated clinical concepts.
Increasing a consistency weight between two coupled training replicas has not
resolved independent-run identifiability on PhysioNet.

**Do not confuse:** (1) predictive significance across 10 initialization x 3
data-order seeds with (2) stability of concept identities and contributions
across separately trained models. Neither improvement is guaranteed.

## Mathematical definition

Let (x_{1:T}) be observed historical features. A shared encoder predicts
named concept trajectories (hat c_{t,k}=\sigma(g_k(f_\theta(x_{1:T})_t))).
The training target (c^*_{t,k}) is independently specified by synthetic
ground-truth physiological states, not learned from labels after the fact.

Define **fixed** recency weights from the training protocol:

[
w_t=\frac{\exp[-\gamma(T-t)]}{\sum_{u=1}^T\exp[-\gamma(T-u)]},
\qquad\bar c_k=\sum_t w_t\hat c_{t,k}.
]

Concept center (mu_k) is the **train-only median** of the corresponding
true concept summary; scale (s_k=max(Q_{.9,k}-Q_{.1,k},0.05)).
Learn a shared (\beta_k,b) by L2 logistic regression **on training
ground-truth summaries only** and then freeze them for *every* independently
initialized neural encoder:

[
u_k(x)=\sigma\left(\frac{\bar c_k(x)-\mu_k}{s_k}\right)-\frac12,
\quad
v_k(x)=\beta_k^*u_k(x),
\quad
\ell(x)=b^*+\sum_{k=1}^5v_k(x).
]

Thus (ell(x)) has an **exact additive concept decomposition** by
construction, with no residual hidden-logit route. The meaning of (v_k) is
anchored by a shared semantic scale and a shared decision coefficient, though
this does **not** in itself make concept predictions accurate, guarantee
equal allocations, or identify real-world causal effects.

For (hat C) and observed concept labels (C^*), train with:

[
\mathcal L=\mathrm{BCE}(\ell(x),y)+\lambda_c\mathrm{Huber}
(\hat C,C^*)+\lambda_t\Vert\Delta\hat C-\Delta C^*\Vert_2^2
+\lambda_e\mathrm{Huber}(v(\hat C),v(C^*))
+\lambda_r\Vert\mathrm{offdiag}(\mathrm{Cov}(\hat C-C^*,C^*))\Vert_F^2.
]

Temporal differences are scored only when adjacent concept labels are
observed. The cross-concept term penalizes *prediction residuals* that align
with other targets; it does **not** forcibly decorrelate true physiological
concepts. This is a hypothesis and may be counterproductive.

## Controlled ablation

`legacy`: original trainable fuzzy head and temporal attention.

`anchor_only`: same neural encoder/projector, frozen semantic head and
temporal weights, original BCE + supervised concept loss. Quantifies the
effect of constraining the explanation basis.

`anchor_regularized`: anchored head plus pre-declared temporal fidelity,
oracle-evidence and residual cross-axis regularizers. Quantifies extra gain
beyond simply freezing the head.

All three use same training episodes, input scaling fitted on **train**,
initial seeds, order seed, number of epochs and optimizer. The benchmark has
an independent synthetic training/validation/test split. **The test
identifiers are sealed; no test feature, concept value or outcome is passed
to fitting or evaluation.** Only validation results are published here.

## Metrics (must jointly report)

- Concept MAE, concept temporal-derivative MAE.
- Validation AUPRC/AUROC (prediction quality), fixed number of epochs.
- Independent-run mean signed Spearman, absolute top-1, sign agreement,
  cosine, and per-run effective number of nontrivial concept contributions.
- Maximum additive reconstruction error and one-concept intervention
  isolation.
- Pairwise independent-initialization statistics with correct uncertainty
  only in **later** larger pre-registered experiments. With three
  independent initializations there is no confirmatory claim, no p-value,
  and no statistical guarantee.
- Reject shortcut "success" where effective number of significant concepts
  collapses to one or concept MAE degrades materially.

## Run on Linux

```bash
PYTHONPATH=src:. python scripts/medical/run_semantic_anchor_validation.py \
  --config configs/medical/semantic_anchor_validation_v1.yaml \
  --out-dir artifacts/medical/semantic_anchor_validation_v1_01
```

**Do not change the frozen YAML based on the test outcomes.** Any new
hypothesis must be versioned with a new config and a new complete result.

The GitHub Actions workflow
`.github/workflows/conceptfan-semantic-anchors.yml` runs the same fixed
config on CPU, with regression tests; it saves only aggregate validation
scores and a provenance manifest.

## Known limitations and publication gate

The synthetic dataset has **fully observed, exact latent concepts**. The
clinical PhysioNet proxies are incomplete and noisy, and the current v1
anchor fitting intentionally **rejects** missing concept ground truth.
A new missingness-aware clinical protocol must be designed and
train/validation checked before any clinical conclusions. Otherwise
imputation can silently encode outcome information.

Structural stabilization by freezing the head is not equivalent to
independently demonstrating generalizable model stability; the
explanations must remain nontrivial and faithful to the concept targets.

The original 30x5-arm PhysioNet comparison requires a new validation-only
experiment, followed by a *single pre-specified test* after the method is
frozen. Do not reuse the already inspected historical test results to tune
this architecture or report an unbiased p-value.

Related conceptual frameworks:
- Koh et al. (ICML 2020), Concept Bottleneck Models
  https://research.google/pubs/concept-bottleneck-models/
- Raman et al. (ICML 2024), Understanding Inter-Concept Relationships
  https://proceedings.mlr.press/v235/raman24a.html
- Park et al. (NeurIPS 2025), Impact of Noisy Concept Annotations
  https://proceedings.nips.cc/paper_files/paper/2025/hash/084de6d7423c3b7af111e091799e4074-Abstract-Conference.html
- Galliamov et al. (ICLR 2026), Concepts' Information Bottleneck Models
  https://proceedings.iclr.cc/paper_files/paper/2026/hash/8555cf308efef02adacb0f9e09bd5f10-Abstract-Conference.html

## Stronger independent-data replication / trivial-consistency control

Because pilot results suggested near-perfect rank consistency, the second,
frozen, newly seeded synthetic experiment also measures whether this is
merely a *global ranking of concepts* independent of the patient.

For each pair of trained models (a,b), with explanation vectors
(v^{(a)}(x_i),v^{(b)}(x_i)), define:

[
S_\mathrm{paired}=\frac1n\sum_i\rho(v^{(a)}(x_i),v^{(b)}(x_i)),
\qquad
S_\mathrm{null}=\frac1{n^2}\sum_{i,j}\rho(v^{(a)}(x_i),v^{(b)}(x_j)).
]

The *excess episode alignment* is
(Delta S=S_\mathrm{paired}-S_\mathrm{null}).
For Spearman, the implementation computes the exact marginal expectation
efficiently from centered, L2-normalized rank vectors. Analogous
independent-episode expectations are calculated for absolute Top-1 and
sign agreement by retaining each model's empirical frequency histogram.

A model that always ranks the same concept highest for every patient will
have Top-1 agreement (1.0) **but zero excess episode alignment**. The latter
is the better diagnostic for case-specific interpretation. We report both,
not just the favorable raw number.

The second experiment uses a new generator seed `20261103`, held-out
validation examples distinct from the initial pilot seed, **10 training
initializations x 3 data-order seeds x 3 methods**. This makes 435 overlapping
model pairs per method: the bootstrap resamples seed *factors*, not the pairs.

This replication is exploratory because the model choice followed the
previous development pilot. Reporting a small p-value as if the method
had been fully pre-registered before observing any earlier data would be
misleading.

The official historical published test remains unmodified and is NOT
part of this experimental pipeline.
