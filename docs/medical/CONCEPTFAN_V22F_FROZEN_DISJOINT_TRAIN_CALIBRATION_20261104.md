# ConceptFAN V22f — FROZEN disjoint TRAIN probability calibration, 2026-11-04

## Motivation, against a concrete external predictive baseline
Executed V22e on an untouched synthetic TEST seed 522323 compared the directly trained ConceptFAN risk logit against actual off-the-shelf logistic regression (30 predeclared raw sensor time-series summary features) and HGB on the SAME 800 TRAIN labels. Native AUPRC was frequently close to logistic regression, but native probability calibration was **much worse** (native Brier ~0.235–0.242; logistic Brier ~0.159–0.176). This is a serious, falsifiable failure of the risk probability, despite strong ranking and native concept decomposition.

The V22f hypothesis is limited: a standard **two-parameter positive affine transformation of the original native logit** can improve Brier/logloss on a new TEST draw without changing ranking, changing concept formation, bypassing the native risk bottleneck or claiming causal meaning. This is **ordinary Platt-style model calibration, not a new scientific concept method or novelty claim**.

## Frozen 800-TRAIN episode budget and three disjoint roles
All four pre-existing V22 source families and their original 800 TRAIN cases (generator seed 522201). Allocate disjoint outcome labels:
- **450 earliest TRAIN episodes**: train the native K2 ConceptFAN task+forecast+denoising model using original fixed V10 training (10 epochs, batch size 96, NN seeds 11/23/37).
- **150 next TRAIN episodes**: fit only the two frozen scalar calibration parameters (positive logit slope and intercept). Their outcomes are NEVER used in native fitting or native task-risk screening.
- **200 last TRAIN episodes**: frozen V22b independent source–outcome risk screen, with patient-level permutation=199, bootstrap=199, p<=.01, stable split halves. Accepted sources set initial raw native risk weight ±0.05; uncertain sources remain native descriptive-only with exact zero risk policy=2.

*Unsupervised* source formation is shared: the PCA whitening and two-environment covariance BSS proposal may see measured histories from the first 600 TRAIN episodes **without the 150 calibration outcomes**. This means calibration histories appear in **unsupervised** source formation but their labels are disjoint from native model and source-risk tasks. Declare this transductive unsupervised use; it is not a fully new-patient heldout calibration feature split.

If BSS source identification ABSTAINS in proportional-variance or sensor-gain-drift families, all related BSS calibration arms must retain ABSTAIN; never substitute PCA silently.

## Fixed calibration transformation and fitting
After the 450-episode native model is frozen, collect its *unaltered native logits* on the disjoint 150 TRAIN calibration episodes. Fit two parameters by deterministic scipy L-BFGS-B on
`mean(logaddexp(0, a + s*z) - y*(a+s*z)) + 0.002*(log(s)^2+a^2)`,
where `z` is the existing complete native risk logit and `s>0`. Predeclared bounds: `s ∈ [0.05,50]`, `a ∈ [-6,6]`. Initialization `s=1, a=0`, max 200 optimizer iterations. No calibration optimizer step is permitted to change ANY native parameter or source identity. The scalar calibration is the same for all native physical concept slots and observation-policy concepts; it is not a new raw observation model or an added concept discovery mechanism.

Mathematically, with the existing native exact additive decision
`z = b + Σ_i physical_i + Σ_j observation_process_j`,
the calibrated decision remains EXACTLY
`z_cal = (a+s*b) + Σ_i s*physical_i + Σ_j s*observation_process_j`.
The component presence, native masks, physical-vs-instrument distinction and concept semantics remain unchanged. A patient-specific rank is unchanged by positive `s`; AUPRC/AUROC should match the uncalibrated version up to numerical rounding. Do NOT claim a conceptual breakthrough for improved probability calibration.

## Frozen TEST and denominators
- Four structural synthetic families × 3 NN seeds × PCA/BSS × **uncalibrated vs calibrated** = **48 registered native scoring opportunities** (36 scored / 12 BSS ABSTAINS expected; **18 actual native models** fitted once; 18 independent 2-param scalar calibrators fitted).
- Four external **full-800-label logistic regression** controls on the fixed 30 observed summary features, one deterministic fit each. These are reference predictive benchmarks, not intrinsic concept baselines.
- **New never-inspected synthetic TEST seed 522333**, exactly 300 examples per family; no TEST generation until all 18 native fits, 18 calibration fits, four external logistic fits and all outcome-screen decisions have completed. No test-driven tuning, model selection or fallback. Store pre/post TEST state SHA256 for ALL native models and calibrator parameters. Expected TEST evaluations = 36 native scoring conditions + 4 external = **40**.
- Record TEST AUPRC, AUROC, Brier, logloss, source oracle alignment as postfit audit only; explicit native additive-logit error before AND after calibration, invariance of AUPRC/AUROC between paired calibrated and raw models, all 12 ABSTAIN records, and individual fitted scalar scale/intercept. No p-values from three seeds on same 300 TEST patients.

## Interpretation / limits
The new seed is an untouched stochastic draw of already studied simulator families: not real hospital generalization. A two-parameter calibration is standard ML, not concept transparency evidence. It cannot correct a concept that follows an instrument artifact, only the mapping from existing native score to probability. It may fail on heldout data; do not alter bounds/regularization after observing 522333. Independent calibration labels and risk screen are disjoint from model gradients, but the external logistic uses all 800 TRAIN labels, while the native uses 450+150+200; 450-vs-800 neural gradient budget is not compute matched. If AUPRC changes materially after positive affine transform, that indicates a bug or unexpected ties and must block scientific reporting. If Brier still lags behind logistic, report the negative result explicitly.

No human clinicians, clinical/physical causal discovery, official CBM/MAGNETS benchmark, prospective validation or accepted PIERE camera-ready modification. Research-only Draft PR #19.
