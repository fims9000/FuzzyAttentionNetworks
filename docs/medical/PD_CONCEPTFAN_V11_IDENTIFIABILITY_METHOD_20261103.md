# ConceptFAN V11 — predictive subspace identifiability and honest concept-group abstention

**Experimental research proposal, not a validated clinical mechanism.**
This branch does not alter accepted PIERE findings or the original V10
development evidence. The purpose is to make CONCEPT discovery and the
limit of its reproducibility part of the trained architecture, not
to decorate a high-AUPRC classifier with post-hoc explanations.

## 1. Why this is a distinct testable method, not just a new penalty

V10 had 1/4/12/24-hour learned signed measurement operators, EXACT
concept-logit additivity, and a separate observation-policy head.
It obtained development AUPRC 0.3794 against a matched Transformer
0.4248, but the independent-run *patient-specific physiological
concept contribution* Spearman excess was only 0.1037. Increasing
ordinary cross-mask per-slot consistency did not fix the issue.

V10's spectral initialization uses left singular vectors of an
observable positive-lag cross-covariance. **SVD axes are not unique
when singular values are tied or insufficiently separated.**
Axis-wise MSE anchoring to such singular vectors forces one
arbitrary rotation and can create *apparently* standardized concepts.

The methodological question is therefore:

> Can a train-only, uncertainty-aware predictive-subspace certificate
> separate reproducible individual observable directions from
> reproducible groups and unidentifiable subspaces, while a native
> concept network is trained through the *projector* rather than the
> arbitrary within-subspace coordinate system?

## 2. Data, spectral operator and what counts as a concept

For observed past sensor vector X[t], mask M[t], define the
train-only masked centered lag operator

```text
C_lag = sum_{patient,t} (X[t] - observed_train_mean) ⊗
                            (X[t+lag] - observed_train_mean)
                     / N_jointly_observed
C = concatenate(C_1, C_2, C_3)
C = U diag(sigma) V^T
```

Only actual observed cells contribute to numerator, and
the denominator is the per-channel-pair count, not the number
of patients with possibly missing observations. The existing
family-wise permutation threshold defines computational rank K,
possibly capped at Kmax for available compute. This is an
estimate of **observable predictive rank**, not unique biological
state dimensionality.

For each computational mode j, a patient bootstrap gives spectral
perturbations. A boundary between modes j and j+1 is *operationally
resolved* only when both hold:

```math
\sigma_j - \sigma_{j+1}
 > Q_{0.90}(|\sigma_j^{(b)}-\sigma_j|
           +|\sigma_{j+1}^{(b)}-\sigma_{j+1}|)

Q_{0.10}\left[
\frac{1}{j}\operatorname{tr}
\left(P_{U_{1:j}}P_{U^{(b)}_{1:j}}\right)
\right] \geq 0.8.
```

These thresholds are declared before evaluation and CAN return no
resolved boundaries. The partitions between resolved boundaries
form predictive groups G. For group g of dimension q:

```math
P_g = U_g U_g^{\top},\quad
S_g(U,V)=\frac{1}{q}\operatorname{tr}(P_{U_g}P_{V_g}).
```

The projector S_g is unchanged if either group's basis is
independently rotated by an orthogonal matrix Q. This is a
finite-sample geometry identity, not a guarantee of factor
identifiability.

**Certificate outcomes, all reported**:
- `INDIVIDUAL_OBSERVABLE_AXIS_STABLE`: singleton with stable
  internal bootstrap subspace and both separating boundaries;
- `GROUP_SUBSPACE_STABLE_AXES_UNRESOLVED`: reproducible predictive
  subspace with unresolved inner axes, so **no individual concept
  identity claim inside it**;
- `ABSTAIN_UNRESOLVED_RANK_BOUNDARY`: even the top-K boundary is
  not stably separated;
- `ABSTAIN_UNSTABLE_SUBSPACE`: empirical group itself not
  reproducible at the declared threshold.

Even the first outcome means only *observable predictive axis stable
under train resampling*. It does NOT mean a real anatomical diagnosis
or clinically manipulable causal process.

## 3. Native training: rotation-invariant identifiable projections

The V11 concept generator retains V10's signed transparent
level/difference/deviation dictionary, trainable scale mixture,
bounded thresholds, UNKNOWN-neutral missing measurements,
and original mask-only observation-process head. Its direct
prediction remains:

```math
\ell(X) = b + \sum_k v_k^{phys}(X) + \sum_j v_j^{mask}(M),
```

with per-patient machine-precision additive decomposition.

For group g, form A_g from the learned **signed sensor
operator weights**, combining explicitly named temporal operators
into their sensor-direction component. Then replace V10's
axis-specific loss \|W-W_SVD\|² by

```math
L_{group} = \frac1{|G|}\sum_g\frac1{|g|}
  \left\|A_g(A_g^\top A_g+\epsilon I)^{-1}A_g^\top -
                      U_g U_g^\top\right\|_F^2.
```

This supports any within-group orthogonal rotation and penalizes
rank collapse. Coefficient `0.035` and ridge `1e-4` are frozen
research hyperparameters, not optimized using 300 dev patients.
The prior scale-entropy and concept nonconstant/redundancy terms
remain, as do 12h future signal prediction and self-supervised
masked-input reconstruction: all run THROUGH the native concept
bottleneck, never a post-hoc explanation model.

A second V11 training arm replaces per-axis cross-mask consistency
with a **group temporal Gram**:

```math
L_{view} = \frac1{|G|}\sum_g
 \left\|
 \frac{\widetilde C^{(1)}_g\widetilde C^{(1)\top}_g}
      {\|\widetilde C^{(1)}_g\widetilde C^{(1)\top}_g\|_F +\epsilon}
 -
 \frac{\widetilde C^{(2)}_g\widetilde C^{(2)\top}_g}
      {\|\widetilde C^{(2)}_g\widetilde C^{(2)\top}_g\|_F +\epsilon}
 \right\|^2,
```

where weighted centering uses joint recorded visibility. This
is invariant to rotation within g, and there is a separate
variance/nonconstant loss to discourage trivial zero patterns.

For source patient p, group contribution is **exactly**

```math
v_g(p)=\sum_{k\in g}v_k(p).
```

The grouped terms reconstruct the original physiological logit.
A passport shows group members, bootstrap stability and formulas.
When an individual axis is not certified, the UI/report must not
name that axis a stable physiological concept. Inspectable formulas
remain available as *model-internal parameterizations*, not truth.

## 4. What is mathematically guaranteed vs merely hypothesized

**Identity 1 — rotation invariance.** For any orthonormal U_g and
orthogonal R, (U_g R)(U_g R)^T = U_g U_g^T. Hence the training
projector target does not move when an SVD numerical routine
chooses another within-group coordinate basis.

**Identity 2 — exact grouped attribution.** Given an additive
model logit b + sum_k v_k + sum_j v_j^{mask}, partitioning
the physiological slot set into disjoint groups preserves the
logit exactly: sum_g sum_{k∈g} v_k = sum_k v_k.

**What is NOT a theorem.** Bootstrap overlap/gap criteria do NOT
guarantee true hidden-factor identification in arbitrary nonlinear
time series. They provide a conservative, auditable diagnostic
relative to this operator, sampling process and finite cohort.
Groups may still be physically meaningless or task-irrelevant.

## 5. Registered ablations and baselines

**Clinical development:**
- `v10_consistency_refit`: original native V10 with SVD-axis anchor
  and per-axis cross-mask consistency;
- `v11_projector_axis_view`: identical architecture, same
  per-axis cross-mask objective; only SVD axis anchor is replaced
  by the rotation-invariant group projector;
- `v11_projector_group_view`: also use rotation-invariant group
  temporal consistency.

All 3×3 runs use the **same** official original Set A
TRAIN 2400 patients, same previously inspected 300 development
patients, same recorded/visible masks, 36h input/12h future,
train-only sensor selection and train-only rank,
18 epochs, optimizer, 11/23/37 initializations.
The old 400 calibration + 600 test patients remain sealed.
No report can call this independent clinical validation.

**Structural controls:** a second independent linear-AR generator
with planted 3/5/7 dynamic factors, each with
(a) distinct transition rates, (b) two repeated rates/rotating
subspace, (c) independently time-permuted episode trajectories.
Evaluate empirical operator rank, bootstrap gap groups and
whole-space overlap with the oracle emission span **after** fit;
never use planted K or loading matrix as the estimator input.

**Critical comparison dimensions:**
- AUPRC original, MCAR and time-channel MAR; AUROC and
  forecast/denoising RMSE;
- individual contribution Spearman *excess* over patient-permutation
  null (still worth reporting, but cannot be called identified
  when there is group rotation ambiguity);
- group-level same-patient attribution agreement and learned
  projector overlap, **along with group size/status**;
- K estimated, number individually certified, group-only and
  abstained directions, exact logit error and runtime;
- negative outcomes retained, e.g. if no stable group or
  if 8 nominal slots collapse into a single task risk.

**External literature:** MAGNETS discovers unannotated concepts
during training; ICLR 2026 Concepts' Information Bottleneck Models
regularize minimal task-sufficient representations. Neither makes
"concept discovery without labels" itself a first-ever claim.
The specific proposed *empirical uncertainty-aware group certificate
plus native projector loss with abstention* must be evaluated
against these and additional subspace-identification methods before
any novelty or priority statement.

References:
- Forest, Wei & Fink, 2026, DOI: 10.1007/s10618-026-01267-y
- Galliamov et al., ICLR 2026, Concepts' Information Bottleneck Models,
  https://proceedings.iclr.cc/paper_files/paper/2026/hash/8555cf308efef02adacb0f9e09bd5f10-Abstract-Conference.html
- Wedin / Davis-Kahan-style subspace perturbation principles are
  used for **intuition**, not claimed as a derived new theorem.

## 6. Reproduce and inspect

```bash
PYTHONPATH=src:. python -m pytest \
  tests/discovery/test_identifiability_v11.py \
  tests/discovery/test_group_learning_v11.py -q

PYTHONPATH=src:. python -m conceptfan_discovery.identifiability_stress_v11 \
  --config configs/medical/pd_v11_identifiability_stress.yaml \
  --out-dir artifacts/medical/v11_structural_stress
```

Clinical run:
```bash
PYTHONPATH=src:. python -m conceptfan_discovery.clinical_identifiability_v11 \
  --config configs/medical/pd_clinical_v11_identifiable_groups.yaml \
  --reference-v6-config configs/medical/pd_clinical_unknown_neutral_v6.yaml \
  --reference-v7-config configs/medical/pd_clinical_dual_process_v7.yaml \
  --reference-v10-config configs/medical/pd_clinical_v10_multiscale.yaml \
  --frozen-v10-receipt path/to/verified_V10_STATUS.json \
  --raw-zip path/to/set-a.zip \
  --outcomes path/to/Outcomes-a.txt \
  --out-dir artifacts/medical/v11_clinical_frozen
```

GitHub Actions workflow scripts fetch the pinned previous V10
report and official PhysioNet data automatically, and publish only
aggregate results. The historical test is NOT used.

## 7. Next evidence before a paper may claim a new method

- Run an independent real dataset/hospital with a prior, frozen
  feature correspondence and its observation schedule.
- 10 or more init seeds crossed with several batch-order seeds;
  account for model and patient/center uncertainty.
- Complete same-data, same-target comparison with a genuine
  unannotated concept baseline MAGNETS, not just Transformer/GRU-D.
- Show actual positive cases of stable *individual* physiological
  concepts, if any, and negative cases of grouped/abstained concepts.
  Do not promote a single group containing every slot into
  evidence of rich interpretable concept discovery.
- Report predictive performance + recovery/identifiability metrics,
  not only a smaller AUPRC interval.
