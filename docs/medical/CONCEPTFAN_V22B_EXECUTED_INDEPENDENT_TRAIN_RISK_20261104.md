# ConceptFAN V22b — executed independent TRAIN risk screen on statistically separated native concepts

**Research date:** 2026-11-04. **Execution:** [GitHub Actions 41859101641 — SUCCESS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41859101641). Full-denominator artifact `conceptfan-v22b-all-48-source-risk-2`, artifact ID **13339196831**, SHA256 ZIP digest `85609d9ab67d2f51d26393ae4ba0652bc1323b2e302a9979ebbbfbc7ccfc3e1c`. The original, untouched artifact contains `STATUS.json`, `ALL_16_MODEL_ARM_SUMMARIES.csv`, `ALL_48_TRAIN_SIGNED_SOURCE_OPPORTUNITIES.csv`, `TRAIN_ONLY_RISK_PLAN_PASSports.json`, `SOURCE_IDENTIFIABILITY_CORE_TRAIN_PASSports.json` and `COMMIT_SHA.txt`.

**Frozen protocol:** `docs/medical/CONCEPTFAN_V22B_FROZEN_DISJOINT_TRAIN_RISK_20261104.md` was written before inspecting V22b validation outcomes. All 48 registered opportunities accounted for: **36 full native model fits, 12 method-identifiability refusals**, four synthetic families × three neural seeds × two source bases × random or independently screened native risk initialization.

## Actual model results (all validation AUPRC means over 3 NN seeds)

| Family | PCA arbitrary | PCA independent TRAIN source-risk screen | Environment BSS arbitrary | Environment BSS independent TRAIN source-risk screen |
|---|---:|---:|---:|---:|
| Distinct source variance | 0.643434 | **0.800206** | 0.650961 | **0.801905** |
| Proportional variance, nonidentifiable | 0.656362 | **0.836553** | **ABSTAIN** | **ABSTAIN** |
| Device gain shift, invalid fixed mixing | 0.660908 | **0.839516** | **ABSTAIN** | **ABSTAIN** |
| Instrument-source variance changes | 0.669156 | **0.838412** | 0.527079 | **0.838229** |

**Important:** V22a used 800 TRAIN observations while V22b uses **600 CORE TRAIN** and **200 disjoint TRAIN screening** episodes. These are *different effective training designs*; do not compare V22a and V22b AUPRC as if paired same-budget neural fits. The above table compares V22b arms with shared 600-case neural fitting budget.

For the source-identifiable physical/instrument mixture, environment-BSS native concept absolute oracle alignment is **~0.967**, PCA ~0.682, both with and without risk-screened initialization. Risk screening thus improved the *existing native decision optimization* rather than the raw statistical source identification. In artifact-only variance condition, environment BSS alone performed badly (0.527079) but TRAIN-screened native source-risk initialization raised AUPRC to 0.838229, virtually identical to PCA 0.838412. Here independently screened BSS designated **one of its two conceptual sources descriptive-only** with exact serialized zero risk policy and independently initialized the other source as outcome-associated.

## Actual TRAIN-only source-risk screens
- Distinct-variance BSS: source 0 initialized at +0.05 (TRAIN-associated), source 1 given **permanent native no-risk descriptive mask**. PCA: both mixed components initialized −0.05, neither blocked.
- Instrument-only variance BSS: source 0 descriptive-only/no risk; source 1 initialized +0.05. PCA: both mixed axes independently considered outcome-associated and initialized −0.05.
- The two nonidentifiable BSS families remained **ABSTAIN** for BSS arms, not covertly replaced with PCA; PCA control still trained.

### Native signed contribution reproducibility
Signed patient-level rank Spearman across three seeds is ~−0.333 for arbitrary-sign PCA but ~+0.9999 for independently signed-start PCA in all four families. For BSS after risk screening, mean signed effect rank Spearman is ~+0.50 in identifiable conditions, **not near +1 for every source** because one BSS source is descriptive-only, giving zero/constant contributions and undefined per-source correlations excluded from averaging. Do not treat a +0.50 aggregate as unstable physical factors without inspecting the defined nonzero source separately. In artifact-only BSS before screening, signed rank stability remains negative (~−0.329).

### Predictive calibration
Measured Brier scores still show task calibration trade-offs: distinct-variance PCA improves from 0.242134→0.237858, whereas BSS improves AUPRC from 0.650961→0.801905 but Brier **worsens** from 0.242021→0.242955. In instrument-only variance PCA improves from 0.239765→0.235325 while BSS improves from 0.253371→0.241412; although BSS now has high AUPRC, it has worse Brier than PCA on the same validation cohort. **Do not claim a universally better risk predictor or calibration.**

## Epistemic and causal boundaries

1. V22b *does not show physical source identity*; a TRAIN outcome-association screen can be passed by stable artifacts. The term "physical" refers to the **generating mechanism's known simulated factor**, not a diagnosis.
2. Separately trained concepts inside a risk bottleneck are not a post-hoc explanation, but exact additivity does not prove semantic correctness. High source alignment is from synthetic evaluation-only latent oracle.
3. Screening n=200 episode-level TEST-like held TRAIN split is independent of neural fitting and unsupervised source proposal, but the original **validation families have already been inspected across V22a**. V22b validation is exploratory, **not new blind confirmation**.
4. Improvements combine independently initialized task-risk coefficients, a structural no-risk mask and extra outcome-side screened labels. The current experiment does **not isolate all three mechanisms individually** or provide compute/exposure-matched baselines.
5. Two nonidentifiable families correctly abstain from BSS; source separation mathematically requires sufficiently different environment covariance signatures and stable source mixing. A shared persistent instrument artifact can still be consistent with every statistical screen.
6. No real clinicians, private EHR data, external prospective validation, or official CBM/MAGNETS baselines have been evaluated. Accepted PIERE camera-ready unchanged; PR remains Draft.

## Predeclared follow-on
V22c uses a genuinely uninspected synthetic TEST seed **522203** with identical V22b TRAIN and gate data, exact same screening criteria and all 48 arms. TEST outcomes must be scored only after all model fits/screens are complete, without retraining or threshold adaptation. It can strengthen confirmation **within known simulator families**, but it remains synthetic and is not a prospective external clinical result.
