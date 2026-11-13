# V71 executed: supervised conditional preparation-dose anchors can improve useful proxies, NOT autonomous physical identity

**Posthoc exploratory result on the already viewed V70 UCI308 TEST.** V71 mechanism and fixed residualization/fuzzy thresholds protocol were committed after inspecting V70 test outcomes but before V71 run: `CONCEPTFAN_V71_FROZEN_SUPERVISED_CONDITIONAL_ANCHORS_20261113.md`. This is **NOT a new independent confirmatory TEST**, even if its test scores are numerically higher.

[Successful GitHub Actions run #43491577773](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43491577773); **13 tests passed**. Artifact `conceptfan-v71-supervised-reference-dose-anchoring-posthoc`, ID **14070480612**, contains full `V71_POSTHOC_ANCHORED_EVIDENCE/STATUS.json`, 24 model×target×cohort rows `METRICS.csv`, source SHA and aggregate log. Exact original official UCI308 ZIP SHA256 `7b062960dbef1a9e8aefc62bc7dbac09bdadb8cbb40793cf2b8e122da1862f90`, 58 real original separate trials, same TRAIN39/VALID11/**already inspected TEST8**. All quantities are nominal **water-dilution vol.%** preparation doses of acetone or ethanol, not independent gas-phase ppm analyzer results.

## Why supervision is deliberately explicit here

Previous V70 **UNLABELED** concept formation generated K2 from TRAIN late/early sensor changes, but frozen axes reacted to both chemical dosing sources with high partial correlations. New V71 instead starts from same original TRAIN39 physical sensor raw32 whole+late summaries and takes the real TRAIN water-dilution labels `y_acetone`, `y_ethanol` to define TWO **supervised reference-dose proxy directions**:

1. For each analyte, calculate its TRAIN OLS residual after regressing its nominal preparation level on the OTHER nominal analyte level; this is a statistical residual, not an independent physical intervention result.
2. Ridge(alpha1) on TRAIN-only standardized 32 measured sensor summaries predicts each residual. The resulting 32D direction is explicitly named **by the external TRAIN label used to fit it**, not by an autonomous statistical source-discovery model.
3. Each of the two TRAIN-supervised reference proxy directions gets TRAIN-only Q25/Q50/Q75 fuzzy memberships (6 states) and a fully signed additive concept-only Ridge(alpha1) head fit on the **same 39 TRAIN target records**. Supervision is used twice (source formation and head fit), although the **number of original labeled TRAIN trials does not change**. This is a target-supervised architectural control, not an equal-supervision-budget win over unlabeled V70.
4. Comparators: V71 supervised linear K2 proxies, original frozen V70 anonymous dualstream 12-state fuzzy, raw16 Ridge on same original32 measured summary channels, constant TRAIN mean; source-negative control permutes the full TRAIN39 pair of dose labels **only during source fitting**, while fitting the downstream head on the correct original TRAIN labels. No V70 source labels are secretly inserted.

## Real previously viewed TEST8, R² / MAE in nominal water-dilution vol.%

| Target | Original V70 autonomous fuzzy | V71 supervised proxy fuzzy | V71 supervised proxy linear | Raw original sensor Ridge | Permuted-source reference control |
|---|---|---|---|---|---|
| Acetone | .5324 / .2085 | **.8298 / .1076** | **.8697 / .0940** | .8687 / .0951 | **−.9030 / .3574** |
| Ethanol | .4763 / .2000 | **.6117 / .1878** | .6106 / .2330 | .5946 / .2367 | **.6083 / .2160** |

VALID11, prior to V71 reused TEST assessment: acetone V70 fuzzy .7945, anchored fuzzy .9091, anchored linear .9766, raw Ridge .9762, permuted control −.0095; ethanol V70 fuzzy .5776, anchored fuzzy .8869, anchored linear .9001, raw Ridge .8976, permuted control .1378.

**Bounded engineering observation:** A source built using TRAIN physical-dose reference labels retains far more of the targeted signal than blind PCA fuzzy sources on these recordings. For acetone, the permuted-label negative control collapses, supporting the importance of correct reference alignment **in this specific fixed experiment**. However for ethanol, the same permuted-source control unexpectedly attains R² .608 on the tiny TEST8, almost equal to the correctly supervised .612 (despite poor VALID .138). This is a **crucial limitation**: one split and one permutation cannot prove the observed ethanol gain specifically depends on correct source labels, nor estimate causal importance. Retain rather than suppress.

The original raw Ridge is essentially tied with V71 supervised linear for acetone and just below the supervised fuzzy for ethanol. **Do not claim a robust win over raw sensors** from n8 or after TEST-guided architecture design.

## Source specificity versus target association

Freeze the two anchored source proxy axes before VALID11 and reused TEST8. For each axis, partial Spearman of the score with the **matched** preparation-dose label controlling the other, and with the **other** preparation dose controlling the matched one:

| Axes named from TRAIN labels | VALID matched / cross | reused TEST matched / cross | TEST abs-specificity margin |
|---|---|---|---:|
| Acetone dose proxy | .968 / .435 | **.941 / .372** | **+.569** |
| Ethanol dose proxy | .899 / **.817** | **.946 / .790** | **+.156** |

Acetone's statistical dose proxy became much more selective for the TRAIN-designated analyte under this model. The ethanol proxy **still has strong cross-response to acetone** (conditional partial rho .790), so even a supervised label-defined axis is not proven to represent a uniquely isolated chemical source. High partial rho with nominal water-dilution preparation levels is not physical gas-molecule identity evidence. The n8 TEST permits fragile correlations, not a reliable source-specificity error guarantee.

## Exactness, correctness, limitations

The fitted fuzzy heads have exact signed additive coefficient reconstruction with no raw-to-head bypass; unit tests check TRAIN-only reference labeling, manipulated source-label dependence, shape, freeze and source integrity. However exact logit decomposition (V66 theorem) does not imply semantic identity. V71 **deliberately uses known TRAIN labels in source formation**, meaning it is not automatic concept discovery. Its chemical names come from supervised reference fields, and those fields specify doses **in a liquid dilution preparation**, not an independently calibrated gas phase analyzer response.

Scientific decision: retain both V70 (strictly unsupervised, source entanglement) and V71 (explicitly anchored proxy with strong acetone selectivity, unresolved ethanol cross-talk) as complementary **negative and constructive evidence of the need for external grounding**. A prospective independent validation with newly acquired gas mixtures and calibrated gas-phase concentrations is required to substantiate real physical source identification. No accepted IOI/PIERE or master changes; research PR DRAFT.
