# V13: conditionally identifiable observable concepts from randomized interventions

**Experimental branch, not a revision to accepted PIERE**. This work is a conditional extension to V10/V11/V12; it does not prove that arbitrary physiological factors can be inferred from retrospective ICU data. The frozen V11 Set B holdout has already been used and MUST NOT be reused to tune V13.

## Motivation from the negative V12 evidence

On newly generated 3/5/7-factor synthetic tasks, V12 did not raise the number of reproducible individual signed effects relative to V11, and on the repeatedly inspected Set A DEV its limited improvement in signed-effect stability came with lower predictive AUPRC. This is an identifiability problem, not automatically fixable by another agreement penalty.

## Mathematical identification boundary

Assume a **linear, time-stationary measurement model** `x = A z + eps` with an unknown full-column-rank `A` and environment-invariant, zero-mean measurement noise `eps`. Experimental environment `e=0` is randomized control. In environment `j` a **known distinct single latent factor j** is shifted by nonzero `delta_j`, with no effects on other latent factors (including downstream causal propagation) during the measurement window. No confounding of intervention assignment with latent baseline and no environment-dependent measurement mechanism is assumed. Then

    Delta_j = E[x | randomized e=j] - E[x | e=0] = delta_j A[:,j].
    D = [Delta_1, ..., Delta_J] = A[:,1:J] diag(delta_1, ..., delta_J).

Therefore, **under these assumptions**, the observable source-loading directions are identified up to scale and the externally known intervention ordering. This is straightforward linear algebra, **not a novel theorem**. If `D` is full column rank, `W = D^+` is a response-coordinate demixer, and `W D = I`. `J`, the number of independently targeted and externally verified intervention labels, sets the identifiable response capacity. It is NOT the total latent dimension unless these interventions cover every latent factor; do not claim that K was inferred unsupervised from the classification target.

Intervention mean contrasts only identify *environment response* directions without the single-target guarantee. For unknown/multi-target interventions, `D = A B`, and `A R`, `R^{-1}B` produce exactly the same `D` for any invertible `R`. Even a stable full-rank `D` **cannot prove single-factor identities**. Stable observational temporal dynamics likewise can identify subspaces but not individual latent names. These are important impossibility controls, not bugs.

Papers already studying stronger and more general intervention identifiability include Squires et al., *Linear Causal Disentanglement via Interventions*, ICML 2023, https://proceedings.mlr.press/v202/squires23a.html, and Bing et al., *Identifying Linearly-Mixed Causal Representations from Multi-Node Interventions*, CLeaR 2024, https://proceedings.mlr.press/v236/bing24a.html. V13 claims **implementation of an assumption-aware concept training and abstention protocol**, not priority over these papers.

## Native model and training objective

The estimator receives only TRAIN observations, masks and randomized environment labels; it computes per-channel observed six-hour histories, randomized condition means and bootstrap response contrasts. A certificate demands all intervention contrasts are stable (`q10 cosine >=0.8`), well-conditioned (`cond(D) <=30`) and resolved above bootstrap spectral noise (`sigma_min(D) > 2 q90(||D_boot-D||_2)`). It **also requires external provenance** that intervention labels correspond to independently targeted and distinct latent factors; this assumption is not statistically verified by those tests.

The model itself is concept-native, not post-hoc. Its trainable sensor definitions start at `W0 = D^+` and continue to receive outcome gradients. For the measured six-hour sensor mean `h`, concept `k` is

    c_k(x,m) = r_k(m) * tanh((w_k^T h - threshold_k) / softplus(width_k)),
    phi_k(x,m) = beta_k c_k(x,m),
    logit P(y=1|x,m) = bias + sum_k phi_k(x,m).

Coverage `r_k` is the fraction of the concept's signed sensor mass with any actual measurement in the window. With no observed evidence, `r=0` and all concept contributions vanish. There is no encoder-to-risk bypass. The native parameters `W`, thresholds, widths, beta and bias are trained together on TRAIN outcome labels. The loss includes BCE plus the train-only instrumental-response identity and definition drift:

    L = BCE + 2 ||W D - I||_F^2 / K^2 + 0.2 ||W-W0||_F^2 / (K D)
        + 0.8 L_observable_probe.

The **observable-probe regularizer** adds one train-estimated `D_j` to *currently measured past sensor cells only*, computes the concept changes inside the same native model, suppresses non-target concept responses and requires small positive target response. This is a mathematical *sensor-level training perturbation*, **NOT a factual individual-patient causal intervention**. Report the ablation with coefficient `0` to prevent crediting the entire gain to the regularizer.

## Frozen falsification design and metrics

The only final synthetic study uses untouched generated seed **20270213**, frozen prior to its first CI scoring. Earlier local pilot and debugging used **20261104**, explicitly excluded from the frozen study. `K_true in {3,5,7}` is used by the simulator and evaluator only, never passed into the estimator as latent truth. The number of intervention environment labels equals K in the positive synthetic generator by design, a **strong assumption**.

For each true K, 180 independent TRAIN episodes per environment (control and J distinct interventions), 36 observed hours, MCAR 95% sensor coverage, 64 TRAIN-patient bootstraps, and 260 entirely new observational validation episodes are generated. The latent AR mechanism, sensor emission columns and intervention targets are withheld from estimation/training, and used **only in evaluation**. Train models from three initialization seeds (11,23,37), with a no-probe native ablation and a raw logistic predictive control. No clinical PhysioNet data or Set A/B outcomes are accessed.

Four falsification regimes:

- `exclusive`: single-target interventions with distinct targets, additive mixing and no inter-factor propagation; identification is *conditional*.
- `multitarget`: invertible dense orthogonal mixes of intervention targets. Empirical contrast geometry can be stable but single-factor identity is unidentifiable without provenance. An intentionally **false provenance assertion** is used as a required counterexample.
- `duplicated`: two intervention environments perturb the same factor; effective response rank is deficient, and the model must abstain.
- `sham`: interventions have zero latent effect; any apparent response is finite-sample noise and must abstain.

Metrics include train-only condition number, bootstrap subspace resolution, evaluation-only true-loading cosine by intervention label, **optimistically** Hungarian-matched control-only PCA cosine, new-patient AUPRC and AUROC, minimum signed concept-effect Spearman over all 3 model pairs, learned definition drift, observable perturbation off-target concept activation (full model vs no-probe), and the deliberately false-provenance counterexample. Absence of patient-level files in output is required.

**Falsifiers:** the false provenance assertion passing a numerical check is *not* a success; it shows a reliance on external experimental assumptions. If rank/sham inputs receive individual certified concepts, the method fails. A high oracle cosine cannot be extrapolated to nonlinear or unknown-target interventions. Because the new scenario is generated from our own simulator, even an impressive performance result demonstrates only a limited conditional proof-of-concept and cannot be described as clinical or frontier scientific novelty.

## What independent evidence would actually advance this research

1. A truly different intervention simulator with coupled mechanisms and controlled downstream propagation, to quantify when identifying total-response directions is insufficient for latent factor identification.
2. Time-varying/nonlinear measurement transformations and non-random missingness, with guaranteed coverage and negative controls.
3. An independent public dataset with credible assignment/exposure provenance and target-specific assumptions (none is supplied by original PhysioNet 2012 Set A/B), followed by supervised scientific review of whether intervention-response concepts map to meaningful domain variables.
4. Direct comparison to a properly reproduced published intervention-identifiability method, not a renamed local variant. The V12 MAGNETS-style baseline is for a different weakly-supervised task and cannot establish V13's novelty.

## How to reproduce

Run from the root of the research repository after installing its Python dependencies:

    PYTHONPATH=src:. python -m pytest tests/discovery/test_intervention_v13.py -q
    PYTHONPATH=src:. python -m conceptfan_discovery.intervention_stress_v13 \
        --config configs/medical/pd_v13_intervention_identifiability_synthetic_20261104.yaml \
        --out-dir /tmp/V13_FIRST_FROZEN_SYNTHETIC

Only the first scoring run on the new seed counts as the frozen synthetic validation. Do not modify its loss/thresholds or select seeds based on that output.
