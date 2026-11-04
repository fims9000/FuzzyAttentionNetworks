# ConceptFAN V22b — executed independent TRAIN risk policy closes the mixed-source prediction deficit

**2026-11-04, exploratory synthetic V22.** Protocol fixed in `docs/medical/CONCEPTFAN_V22B_FROZEN_DISJOINT_TRAIN_RISK_20261104.md` before V22b execution. Complete results versioned under `docs/medical/evidence_v22/disjoint_source_risk/`, including every one of 48 source/method/seed/arm opportunities, all 36 native fits, 12 source-identifiability refusals, individual risk screens, commit SHA and CSV/JSON.

[GitHub Actions 41859971976 — SUCCESS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41859971976) completed the complete 48-case study; [GitHub Actions 41859101641 — SUCCESS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41859101641) independently completed the same initial study but archived to a GitHub workflow artifact. The later trusted same-repository CI run safely committed complete scientific evidence to PR #19 without modifying the task or its predeclared thresholds.

## Experimental design and source-vs-risk distinction
All models use **the same 600 core-TRAIN patients**, TRAIN-only measurement whitening, fixed two-slot native V10 concept bank, 10-epoch joint risk/forecast/reconstruction training, and three original neural seeds 11/23/37. Both PCA and covariance environment unmixing are *standard statistical bases*, not a newly invented physical identification theorem. Each method's two measurement directions are trained inside the **actual native ConceptFAN decision path**, never as a post-hoc predictor.

A completely disjoint set of **200 TRAIN patient labels** independently screens each concept's *observed* last-six-step score for binary-outcome association (199 patient permutations, 199 patient bootstraps, split-half sign, confidence interval and minimum correlation criteria). Accepted risk concepts get only a small **±0.05 initialization of the existing native decision coefficient**, not a post-hoc classifier or permanent sign. Rejected or uncertain risk concepts **remain real native measurable concepts** for reconstruction/forecast, but native V21d `signed_risk_policy=2` enforces an **exact zero decision contribution**, including under later risk-head refits. A negative outcome screen is not evidence the measured concept is physically wrong.

Independent TRUE synthetic physical/instrument source trajectories are consulted ONLY after every neural model fitting in each family for measurement correctness audit. TEST seed 522203 stays sealed.

## Heldout validation results, all predeclared arms

| Mixed observation generator | Source basis | Original native AUPRC | TRAIN-screened native risk AUPRC | Original signed effect rank across seeds | Screened signed-effect rank across seeds |
|---|---|---:|---:|---:|---:|
| Distinct source variance profiles | PCA | 0.64343 | **0.80021** | −0.33322 | +0.99992 |
| Distinct source variance profiles | Environment unmixing | 0.65096 | **0.80191** | +0.33484 | +0.49995* |
| Proportional, unidentifiable variance profiles | PCA | 0.65636 | **0.83655** | −0.33323 | +0.99994 |
| Proportional, unidentifiable | Environment unmixing | **ABSTAIN** | **ABSTAIN** | — | — |
| Gain-shift / sensor calibration drift | PCA | 0.66091 | **0.83952** | −0.33325 | +0.99995 |
| Gain-shift / sensor calibration drift | Environment unmixing | **ABSTAIN** | **ABSTAIN** | — | — |
| Only the instrument-source variance shifts | PCA | 0.66916 | **0.83841** | −0.33324 | +0.99994 |
| Only instrument variance shifts | Environment unmixing | 0.52708 | **0.83823** | −0.32902 | +0.49999* |

*The signed stability metric is averaged across **both** concept slots; in the screened environment-unmixing arms, one source is structurally excluded from risk and its signed contribution is a constant zero. An undefined constant-sequence Spearman contributes zero in the original aggregation, so mean ≈0.5 is **not evidence that the active source itself is unstable**. Recompute active-only crossrun stability with explicitly saved signed trajectories before any scientific claim about stable patient-level explanations.

**Core result:** independent TRAIN source-risk initialization/eligibility greatly improved actual native prediction with fixed concepts, including the previous severe source-unmixed negative case (0.52708→0.83823). In the distinct variance family, source-separated vs PCA screened models achieved practically equal AUPRC (0.80191 vs 0.80021) while having very different semantic recovery of generated factor trajectories.

## Actual concept alignment to known synthetic sources
No oracle factors enter concept learning or screens. After all model fits, native raw-activation-to-source matched absolute Spearman:

| Mixed family | PCA native concept alignment, screened | Environment-unmixed native concept alignment, screened |
|---|---:|---:|
| Distinct variance | 0.68134 | **0.96723** |
| Proportional variance | 0.67958 | ABSTAIN |
| Calibration/gain drift | 0.70474 | ABSTAIN |
| Artifact-only variance | 0.70601 | **0.97307** |

Thus task prediction quality is **not** a surrogate for semantic correctness. PCA predictions can be as accurate while not separating the two latent factors. The recovered model-native concept can be highly aligned with an **instrument artifact**, not biology, because source identity is unspecified in the observations.

On the identifiable families, the independent TRAIN task screen selected **one signed outcome-relevant source** and gave the other a **persistently zero** task coefficient. On original PCA mixtures, **both mixed axes** passed the risk screen, so both received native task priors. This is evidence of a useful *measurement/source-vs-decision separation*, not a proof of physiological source discovery.

## Mechanistic interpretation, uncertainty and release boundary
The V22b screened arm changes **both** risk-weight initialization and whether nonassociated concepts can participate in prediction; it does not isolate the independent causal contribution of those two modifications. The two arms also differ in how risk-irrelevant sources are treated; attributing all AUPRC improvement to *sign initialization alone* would be incorrect. A future two-factor ablation (initial sign only, risk mask only, both, original) is needed to isolate causes.

The synthetic TRAIN source cohort shrinks from **800 in V22a** to **600 in V22b** because 200 independent training records are now reserved for screening. Therefore do not directly subtract V22b AUPRC from V22a to infer an architecture or label-efficiency effect. Valid comparisons are the *within-V22b* matched arms. All three model seeds share the same validation patients; there is no proper confidence interval across independent clinical cohorts.

**Established:** in this synthetic generator, meaningful statistical factor unmixing improves actual native concept/source alignment, and separate TRAIN-only task relevance controls substantially improve task prediction without an inference bypass. Some environments are not source-identifiable and trigger ABSTAIN. These properties are different from and complementary to V20 separate native-head optimization.

**Not established:** new physical/medical causal factors, a general identifiability theorem, official published intrinsic-concept baseline superiority, independent external-hospital generalization, a genuine human clinician feedback study, true instrument artifact provenance or a label-efficient generalization guarantee. An outcome-associated instrument can still be a false physical explanation. Accepted IEEE PIERE camera-ready remains untouched, PR stays Draft.

**Next controlled step:** isolate risk weight sign initialization vs structural mask as separate factors *without retuning V22b validation* and then evaluate a preregistered unseen mixing geometry/test family. Until that, do not promote these exploratory findings as a final scientific discovery.
