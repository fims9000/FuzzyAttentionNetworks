# ConceptFAN V22c — executed sign/mask factorial and untouched-seed synthetic TEST

**Research date:** 2026-11-04. Draft PR #19; accepted IEEE PIERE manuscript unchanged. Two separate frozen protocols were written before the respective executions:
- `CONCEPTFAN_V22C_FROZEN_RISK_SIGN_VS_MASK_FACTORIAL_20261104.md`: factorial signed risk start versus structural no-risk masking on already inspected VALIDATION.
- `CONCEPTFAN_V22C_FROZEN_UNSEEN_TEST_20261104.md`: identical V22b model/gates and 48 arm registrations evaluated on genuinely unused TEST generator seed **522203**, only after all fits complete.

**Executed CI and immutable artifacts:**
- Factorial [Actions 41861657332](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41861657332), 48 registered, **24 TRAINED / 24 ABSTAIN**; model fits, tests and artifact upload succeeded. Its last action to **push** the artifact files failed for a legitimate **non-fast-forward concurrent branch update**, not because the experiment or tests failed. Artifact `conceptfan-v22c-all-48-risk-factorial-5`, ID **13340381457**, GitHub ZIP SHA256 **b8553de1dee6956a1324a11cf2bff6cb8b37bcaa328b707ba416e86f9044ef61**. Do not misrepresent this Actions job as fully green until backfill/CI wiring is repaired.
- Independent new-seed TEST [Actions 41861657354 — SUCCESS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41861657354), **36 fully pretrained native models, 12 identical structural ABSTAINS, 36 frozen TEST evaluations**. Artifact `conceptfan-v22c-fresh-test-522203-all-48-2`, ID **13339299661**, ZIP SHA256 **f34cc1f867932a146b7ed784258baf187347df2d2146020da369b6a24d9c6ac1**. The TRAIN and V22b IDENTIFIABILITY gate source seed 522201, 600 model-fit episodes + 200 disjoint independent risk-screen cases, 10 epochs and initial signed coefficient/mask policy were not changed. Runtime asserts that all 36 models have been fitted before TEST seed 522203 is instantiated, and before/after parameter SHA256 values are identical for every scoring.

## 1. Independent TEST 522203 confirms predictive improvements *within known synthetic simulator families*

All AUPRC values are mean over exactly three NN initializations, evaluated on the **same 300 newly drawn synthetic TEST episodes per family**, not three independent patient cohorts.

| Simulated family | PCA arbitrary | PCA TRAIN-screened risk | Environment BSS arbitrary | Environment BSS TRAIN-screened risk |
|---|---:|---:|---:|---:|
| Distinct source variance / identifiable | 0.639228 | **0.800839** | 0.641383 | **0.803675** |
| Proportional variance / non-identifiable source rotation | 0.665260 | **0.810566** | ABSTAIN | ABSTAIN |
| Sensor calibration gain drift / mixing change | 0.675217 | **0.831890** | ABSTAIN | ABSTAIN |
| Artifact-only variance change / identifiable source statistics | 0.687998 | **0.825511** | 0.521025 | **0.834679** |

The large improvements observed on the previously inspected VALIDATION V22b substantially persist on a **new random generator seed**. But a well-screened BSS basis does **not** overwhelmingly improve AUPRC compared with similarly screened PCA: identifiable variance ~+0.00284, artifact-only variance ~+0.00917.

**Concept source alignment on new TEST:** distinct variance PCA 0.664 vs BSS 0.969; artifact-only PCA 0.710 vs BSS 0.972 (post-fit absolute Spearman with generator-known physical/instrument hidden sources after matching sign and permutation *only for audit*). The separation model clearly identifies original generating factors better in these controlled conditions. Yet this also identifies **instrumental sources**, not necessarily physiology.

**Important calibration tradeoff:** distinct variance TEST Brier PCA screened 0.237181 vs BSS screened 0.242831 (worse BSS), artifact-only PCA screened 0.235910 vs BSS screened 0.241601 (worse BSS). Better source recovery, sometimes modestly better AUPRC, **does not imply better calibrated risk prediction**.

## 2. Controlled factorial: what caused V22b's prediction improvements?

This factorial uses V22b's **already inspected VALIDATION seed 522202**, so it is an exploratory training mechanism analysis rather than a fresh prospective confirmation. Every source method arm uses identical original V10 native K2 model, source vectors, TRAIN cohort and labels; ONLY the initial signed risk weight and/or permanent descriptive-only mask vary. The environment BSS source basis was usable only in two families, giving **24 trained models / 24 predeclared method refusals**, all with results retained.

| Environment BSS native source family | Random original | Signed start **only** | Descriptive no-risk mask **only** | **Both** |
|---|---:|---:|---:|---:|
| Distinct physical/instrument source variance | 0.650962 | 0.680766 | **0.801203** | **0.801905** |
| Instrument-only variance change | 0.527081 | 0.771123 | 0.510206 (worse) | **0.838231** |

**Mechanism interpretation:**
- When both recovered sources have distinguishable variance changes, excluding the source that fails a disjoint independent TRAIN outcome-association test is almost all of the performance gain; merely setting another initial task sign helps much less.
- When the only identifiable environmental variation occurs in a *measurement artifact*, both interventions are important. Exclusion **alone makes performance slightly worse**, while risk sign orientation alone helps substantially and combined orientation + exclusion does best on the inspected VALIDATION.
- These mechanisms are NOT universal source semantical tests; they use **extra supervised TRAIN labels** (200 different cases), and the artifact factor can still be a stable source. A clinician could still find it physically meaningless.
- The original mean signed rank used both slots including zero-constant descriptive-only sources and could misleadingly appear to be ~0.5. The factorial corrects this by reporting **active-only** patient-specific signed effect rank Spearman: distinct variance masked arms **0.99976/0.99990**, artifact variance combined **0.99997**, artifact variance mask-only **−0.33326**. Constant masked sources are excluded, not treated as zero stability.
- **Brier remains problematic**: for distinct variance, combining sign and mask yields mean Brier **0.242963**, slightly **worse** than original 0.242021 despite high AUPRC. Better ranking alone does not justify a clinical decision system.

## 3. Research conclusion and honest limitations

We have stronger evidence for a **two-level concept research architecture**: (i) measurement-driven TRAIN source formation, with explicit **ABSTAIN** where directions are non-identifiable; (ii) a separate disjoint TRAIN supervised source–task relevance assignment, which sets a native signed start or a structural zero task contribution without bypassing the concept bottleneck.

This is useful experimental engineering, not demonstrated novelty as a theorem: environmental covariance-based BSS and CBM intervention/ablation already exist in literature. The combination requires official method baselines, real world/source ground-truth provenance, adequately powered cross-population evaluation and blinded human review before scientific novelty and clinical trustworthiness can be claimed.

Critically, simulator factors known as “physical” vs “instrument” have statistical signatures that can be observationally indistinguishable; the model does not acquire a causal physical label from covariance, high TEST source alignment or task relevance. Native additive logits are necessary for a mediated prediction graph but insufficient to validate concept semantic meaning.

A next useful control would compare against original two-stage CBM and official automatic time-series concept baselines on identically processed patient-independent real data, and explicitly quantify calibration (Brier, logloss, ECE) under dataset and instrument shifts. Do not tune the now-inspected TEST seed 522203 to retrofit a positive result.
