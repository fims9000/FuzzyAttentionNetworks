# ConceptFAN V9 — executable common-core discovery, careful limits

**Executed 2026-11-03. Status: exploratory source-native model, not independent clinical concept discovery.**

## Why this experiment was necessary

The previous V8 train-only unsupervised care-process rank estimate was **not reproducible**. In the 1800 TRAIN patients after an independent mask-only sensor pilot, source K=9, while independent TRAIN halves yielded K=8/K=7. Under simulated measurement-policy shifts the source estimates changed again to K6/K7. Even fixed learned mask-operator definitions could not establish the correct number or meaning of clinical latent causes.

Successful V8 prediction did not exceed the preceding V7: the V8 source-only AUPRC was 0.285949 vs V7 0.298550 on the previously reused development validation. It is incorrect to call V8 superior or to equate its automatic K9 with nine real clinician decisions.

Sources: [V8 independent rank negative control](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41512934565), [V8 trained-model comparison](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41512252657).

## New mathematical hypothesis — identify only the shared observed mask-operator subspace

Let the six mask-only, train-fitted, disjoint patient half/policy bases be U_s. Using `P_s = U_s U_s^T` avoids arbitrary changes of basis within each source eigenspace. Compute `P_mean = mean_s P_s`. A direction qualifies for the *common core* only if its P_mean eigenvalue is at least 0.95 and the direction's projected squared length is at least 0.90 in **every** source. The number of qualifying leading directions is K_common.

Identifiable operator-source axes are a separate problem: to give each slot an inspectable and reproducible reference, project a fixed named measurement-mask operator dictionary into that common subspace, then choose canonical coordinates by deterministic pivoted Gram–Schmidt. Floating-point ties are broken by predefined coordinate indices. This fixes a **computational coordinate convention**, not an ontology or causal uniqueness theorem.

No outcome labels, synthetic ground-truth concepts or medical proxy names enter source K estimation. The same historical patients appear across additional simulated sampling regimes, so six source bases are NOT six hospitals or independent patient samples.

## Completed V9 observed subspace result

[GitHub Actions PASS 41609582143](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41609582143), pinned artifact `stable-masked-care-core-v9-2`.

| Mask source group | Detectable raw K |
| --- | ---: |
| Original, half 1 / half 2 | 8 / 7 |
| Additional MCAR thinning, half 1 / half 2 | 6 / 7 |
| Hour/channel-specific MAR thinning, half 1 / half 2 | 6 / 7 |

**Common core K=4** at primary projector threshold 0.95, min per-source squared support 0.90. Threshold sensitivity 0.85/0.90/0.95/0.98 all returned 4 under fixed second gate. Named canonical operator pivots were recency of measurement of Weight, RespRate, HR and NIMAP. Actual directions are **linear combinations** of all supported signed mask operators, not those four single variables or disease labels.

This is a detectable **lower-dimensional common mask-observation structure**, not the true number of independent medical causes, workflows, or physiological states. The current study cannot determine whether K4 would persist at another hospital or after a truly independent clinical policy intervention.

## Actual training-time concept bottleneck and negative predictive result

[GitHub Actions PASS 41610735337](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41610735337), artifact `trained-stable-mask-care-v9-1` (six fitted dual-family models, 3 seeds per method). Reuses the V8 train/validation split and measured-value concept budget; the V9 care K4 and signed operator definitions come from the separate frozen TRAIN rank artifact.

Both concept families participate in task training, not posthoc:
- Measured-value / physical-evidence concepts are unknown-neutral when no measurement is present; their earlier physiological K budget is not externally validated.
- Four signed *care-process* concept operators have fixed train-derived source directions, learn fuzzy thresholds and direct task weights. The same source is used under two predeclared head-training alternatives.
- Final logit remains exactly the sum of physical and sampling-process signed contributions plus bias; there is no bypass classifier.

| V9 variant | Mean development AUPRC | Std across 3 inits | Model-pair case-specific care-process Spearman excess | Care-process definition cosine |
| --- | ---: | ---: | ---: | ---: |
| Frozen shared source, free task beta | 0.283122 | 0.055203 | 0.522467 | 1.0 |
| Same source, task beta prior | 0.214601 | 0.028524 | 0.638077 | 1.0 |

For context, V7 with three manually budgeted care-process slots gave AUPRC 0.298550, and V8 with nine source-ranked care slots gave AUPRC 0.285949 under original development validation. Thus neither K4 predictive architecture exceeded V7 in mean observed AUPRC. Shared source definitions are identical by construction; case-specific signed contributions are not equivalent to them. Task-prior alignment especially degraded AUPRC in this experiment.

Only three original training initializations and the repeatedly inspected 300-patient development validation were used. These means are not clinical significance claims. A paired exact source-model bootstrap against pinned original V7 and V8 evidence is separately versioned and must be interpreted as model-only uncertainty, not a substitute for patient/site replication.

## What is justified and what is not

**Justified:** an algorithm that no longer arbitrarily takes one source's K9, identifies a reproducible K4 *observational common core* under predeclared source policies, and trains its transparent non-posthoc contributions directly inside ConceptFAN.

**Not justified:** four clinically interpretable or causal care-process concepts, four unique physiological factors, clinical transportability, prediction superiority, significant outcome gains, or contamination-free causality from sampling behavior.

The original frozen PIERE paper's conclusion that exact functional decomposition does not guarantee semantic reproducibility remains unchanged. The corrected camera-ready PIERE draft does **not** silently replace its negative original benchmark with V9's post-selected new protocol.


## Pinned, dependent-model-corrected V7/V8/V9 numerical comparison

[GitHub Actions PASS 41611555259](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41611555259), artifact `v7-v8-v9-true-k-comparison-2`. The workflow downloaded the exact frozen original V7, V8 and V9 reports from their successful runs, verified identical initialization seeds 11/23/37 and all three observation scenarios, and enumerated **all 3^3=27 model-index resamples** for each paired method difference. No neural retraining or patient row export was needed.

| Prespecified architecture | Original validation AUPRC | MCAR-thinned AUPRC | Time/channel MAR AUPRC |
| --- | ---: | ---: | ---: |
| V7, 3 fixed care-process slots | 0.298550 | 0.263071 | 0.286363 |
| V8, rank-estimated 9 source-frozen care slots | 0.285949 | 0.242719 | 0.260491 |
| V9, 4 common-core source-frozen care slots | 0.283122 | 0.239737 | 0.258150 |
| V9, 4 common-core slots + task-head prior | 0.214601 | 0.172977 | 0.186542 |

**V9 source-only minus V7 source:**

- Original development validation AUPRC **-0.015429**, descriptive 95% **model-only** bootstrap interval **[-0.033773,+0.008451]**.
- Simulated MCAR AUPRC **-0.023333** [-0.041550,-0.002044].
- Simulated time/channel MAR AUPRC **-0.028213** [-0.046348,-0.007381].

**V9 source-only minus V8 source-only:**

- Original AUPRC **-0.002828** [-0.006910,+0.001744].
- MCAR **-0.002982** [-0.014932,+0.012437].
- Time/channel MAR **-0.002340** [-0.009533,+0.004730].

The ability to recover a smaller, independently supported source-dictionary subspace is therefore **not evidence of increased predictive accuracy**. In this experiment a shared source-K4 did not outperform either earlier architecture. Adding a logistic task-head prior to the K4 dictionary degraded original AUPRC by approximately 0.069 compared with its source-only V9 counterpart and by 0.084 compared with V7.

These intervals are conditional on one historical development validation and three model initialization seeds; they do not include new patients, source hospitals, centers, an independent discovery sample or adaptive method selection. The unchanged originally accepted PIERE experiment must not be retroactively replaced by these new, post-hoc-developed methods.
