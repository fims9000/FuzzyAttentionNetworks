# ConceptFAN V11 — first completed independent-record Set B audit (frozen report)

**Original successful, immutable experiment:** [GitHub Actions run 41710356212](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41710356212), `conceptfan-v11-ONE-SHOT-set-b-A17-verified-PY31215-1`. All interpretation below is based **only** on this first successful complete Set B evaluation. Original accepted PIERE paper and prior V10/V11 training evidence remain unchanged.

**Pre-access provenance.** All 17 frozen Set A refits match their prior per-seed DEV AUPRC **exactly** (maximum absolute difference 0). The source adapter filters out all >=36-hour observations before feature aggregation and uses Set A TRAIN medians, IQR, selected 12 channels and frozen input masks. No new Set B record/outcome was read until all refits passed. Prior aborted runs 41694189228, 41696990574, and 41708168502 failed the Set A Transformer refit check **before Set B download**, under a different Python patch environment. A-only forensics 41706946832 / 41709363597 showed perfect 17/17 refit agreement on Python 3.12.15 and PYTHONHASHSEED=0.

**Holdout:** 4,000 Set B ICU stays, **568 in-hospital deaths = 14.2%**. No overlapping Set A/Set B RecordID, but this **does not prove distinct persons** across admissions: they come from the SAME historic MIMIC-II source, NOT different hospitals. Only 0–35h of observations used. Set A original calibration and test patients remain sealed.

## Complete external-record results

All neural results are **means of 3 independent initializations** (11,23,37), not per-patient ensembles. Logistic and HistGradientBoosting each have one deterministic fit. No model was tuned on Set B.

| Model | Set B original AUPRC | AUROC | MCAR AUPRC | time/channel MAR AUPRC |
| --- | ---: | ---: | ---: | ---: |
| V10 consistency | 0.389276 | 0.775888 | 0.386617 | 0.383457 |
| V11 projector, axis view | 0.387674 | 0.771956 | 0.385928 | 0.381233 |
| V11 projector, group view | 0.387814 | 0.772081 | 0.385796 | 0.381120 |
| Logistic history | 0.371178 | 0.779174 | 0.346144 | 0.334853 |
| HistGradientBoosting | **0.425044** | **0.797340** | **0.399579** | **0.424499** |
| GRU-D style (local adaptation) | 0.355993 | 0.760184 | 0.335371 | 0.336473 |
| Transformer time series (local adaptation) | 0.357712 | 0.768102 | 0.341260 | 0.339250 |

Unlike the **previous repeatedly inspected DEV** results (V10 AUPRC 0.379405, V11 group 0.368722, Transformer 0.424756; HGB DEV 0.313649), the independent-record Set B study shows:
- V11 group versus V10: **-0.001462** AUPRC, descriptive paired-patient-and-training-seed percentile bootstrap **[-0.019020, 0.015577]**. This interval spans zero; no superiority and no demonstrated noninferiority at 0.01 margin can be claimed from this interval.
- V11 group versus Transformer: **+0.030102**, descriptive bootstrap **[-0.004431, 0.065834]**; not statistically established as superior. This opposite point ranking relative to old DEV deserves a distribution-shift audit *without tuning on Set B*.
- V10 versus Transformer: **+0.031564**, descriptive bootstrap **[-0.001849, 0.064678]**; likewise inconclusive.
- V11 group versus V10 MCAR: **-0.000821** [-0.019719, 0.017097]; MAR: **-0.002338** [-0.021641, 0.016154]. Neither supports superiority.
- HGB had the highest **point-estimate** original AUPRC (0.425044). A direct HGB-vs-V11 paired bootstrap was NOT preregistered and is NOT claimed here. HGB differs in model family and uses one deterministic fit. No p-values for HGB-vs-neural models should be retroactively constructed from this single cohort.

These intervals resample 4,000 same-source patient records and 3 model seeds, not sites/hospitals. Metrics were obtained on the same Set B patients for all models. No additional evaluation variants were selected from Set B outcomes.

## Key explanatory/identifiability result

Train-only spectral certificate originally assigned 8 computational observable axes: **6 source-stable singleton directions (0,1,2,3,4,7)** and one unresolved 2D group (5,6). The Set B SIGNED same-patient effect audit compares *all three* independent model pairs against the fixed thresholds Spearman >=0.5 AND learned definition projector overlap >=0.75.

| Model | Candidate reproducible singleton effects | Candidate reproducible multi-axis groups | Abstained groups |
| --- | ---: | ---: | ---: |
| V10 | **1** | 0 | 6 |
| V11 projector + axis | **1** | 0 | 6 |
| V11 projector + group | **1** | 0 | 6 |

The **same computational slot 4** survives both stages in all three methods. On Set B for V11 group, across the 3 model pairs, the **minimum signed contribution Spearman is 0.90037**, and minimum definition-projector overlap is **0.89210**. For example, V11 group source singleton 2 has minimum signed Spearman **-0.97328**, even though the source spectral direction was structurally split. Group 5–6 has minimum signed Spearman -0.00247 and minimum learned definition overlap 0.37287, so it does not pass the functional gate. Group-level geometry and exact logit reconstruction alone are **not sufficient**.

One candidate is a reproducible **internal mathematical risk contribution**, NOT a validated physiological biomarker or clinically causal mechanism. No new stable candidate emerges from V11 relative to V10.

## Scientific interpretation and remaining limitations

1. The contribution of V11 is a **train-time observable-predictive-subspace grouping mechanism with exact additive explanation and honest abstention**; its speculative novelty remains subject to comparison with independent automatic concept discovery methods.
2. **The empirical stability improvement previously observed on Set A DEV did NOT increase the count of functionally reproducible individual concepts on the independent Set B cohort.**
3. The initially expected >0.01 AUPRC degradation of V11 on old DEV shrank to ~0.00146 on Set B; this is a new observation, not evidence of equivalence or noninferiority, since the uncertainty interval does not establish the chosen margin.
4. Cross-run signed effects of most concepts remain highly unstable and frequently change sign. Any claim that V11 solves semantic concept discovery would be contradicted by these data.
5. Evidence is restricted to the historical same-source Set B ICU stays, three neural model initializations, no external hospital, no clinical ontology reference or nonlinear latent oracle. A genuine outside automated-concept baseline such as a documented adaptation of MAGNETS and additional independent cohort/sites are still required before an expansive novelty claim.
6. The successful first complete Set B scoring is **run 41710356212**. A second workflow 41716047196 was inadvertently triggered **after the successful run was discovered** during ongoing automation. It is **not** included as independent replication or used for tuning/model selection. Any duplicate output must be marked secondary technical replay and must NOT be represented as another independent cohort or new validation.

**Result:** proceed with writing a method/negative-result research extension only as a separate experimental V11 study, and preserve the accepted PIERE manuscript unchanged. Do not modify/fine-tune models on this same Set B for future purported holdout claims.

## Immutable source evidence

- [Successful Set B experiment 41710356212](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41710356212).
- Artifact `conceptfan-v11-ONE-SHOT-set-b-A17-verified-PY31215-1` (aggregates, run-by-run metrics, exact functional audit, SHA256 source manifest, no patient records).
- Original Set A 17/17 A-only verification [41709363597](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41709363597).
- Frozen protocol `configs/medical/pd_v11_setb_one_shot_locked_20261103.yaml`.
