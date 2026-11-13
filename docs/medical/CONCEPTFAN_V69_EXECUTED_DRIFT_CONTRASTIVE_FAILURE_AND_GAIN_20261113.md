# V69 executed: label-free within-batch versus between-batch source directions

**Status:** successful real UCI270 *post-V67* exploratory architecture test. Source dataset batch10 TEST had already been inspected by V67/V68. Protocol `CONCEPTFAN_V69_FROZEN_DRIFT_CONTRASTIVE_SOURCE_20261113.md` committed before V69 results, but the architecture was selected after viewing that same heldout period; **this is not new independent validation**.

[GitHub Actions SUCCESS #43488780811](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43488780811), executed commit `00d38a36b48b9d3a0ee56dc43f0370849bb1163e`, **19 tests passed**, saved artifact ID `14068002896`, ZIP `conceptfan-v69-drift-contrastive-real-exploratory`: full `STATUS.json`, 32-row METRICS.csv, script hash/commit. Verified official research-use UCI270 ZIP SHA256 `98fe3a30981a222dd4518fbcc3dddd45d5c0ce9b03ef6dc6fe5cf7a04cfbff5e`. Same original 9,546 TRAIN exposures batches1–7, VALID 764 batches8+9 and **already seen TEST 3,600 batch10**, six pure gases (not gas mixtures), one instrument.

## What actually changed

V67 learns statistical source directions through global PCA95/FastICA over unlabeled TRAIN 32 pooled physical sensor descriptors. V69 retains the exact same TRAIN-only normalization and 32 descriptors but chooses directions by **within-acquisition-batch variation relative to total historical variation**:

`W=sum_b sum_{i in b}(z_i-mu_b)(z_i-mu_b)^T/N`, `B=sum_b n_b(mu_b-mu)(mu_b-mu)^T/N`, `T=W+B+0.1trace(W)/32 I`. Take leading K eigenvectors of `T^(-1/2) W T^(-1/2)`, unwhiten, normalize and canonicalize sign. Select **K=4** from exactly the same PCA95 TRAIN criterion as V67, not gas labels. Three TRAIN Q25/Q50/Q75 bounded memberships per statistical axis produce 12 fuzzy states. No gas label, gas ppm, VALID row or TEST metadata is provided to source fitting. The exact signed-additive six-class logistic head only uses original TRAIN labels and reconstructs probabilities numerically identically; original V67 fuzzy source was recomputed with **identical source SHA256** for paired comparison.

## Actual previously viewed TEST batch10 results (posthoc)

| Method | Accuracy | Macro-F1 | Multiclass log loss |
|---|---:|---:|---:|
| V67 original global ICA 12 fuzzy states | .4833 | **.4451** | 2.9307 |
| **V69 contrastive 12 fuzzy states** | **.5683** | **.5188** | 2.4471 |
| V67 linear ICA K4 | .5567 | .5090 | 1.9665 |
| V69 linear contrastive K4 | **.5778** | **.5556** | **1.7473** |
| Original 32 pooled physical sensor descriptors logistic | .6172 | .6017 | 2.3260 |
| Original 128 measured descriptors logistic | **.6844** | **.6873** | 2.2181 |
| Original 128 ExtraTrees | .6444 | .6442 | **.9803** |
| Majority TRAIN class | .1667 | .0476 | 30.0364 |

This is a **posthoc engineering gain** in a previously viewed TEST, not a new sealed result or new gas-specific semantics. The V69 fuzzy gain from original V67 is +.0737 Macro-F1; however **raw128 logistic is still ahead by .1685 Macro-F1**, and even the simple V69 linear K4 source representation does better than V69 fuzzy (Macro-F1 .556 vs .519). On earlier VALID batches8–9, V69 fuzzy Macro-F1 .6072 vs V67 .5589, but raw pooled32 logistic .8713 was better. More bounded memberships and exact signed attribution do not ensure task sufficiency. Raw logistic/pooled retests on fresh CI have small numeric differences from V67 runs (original data/source SHA identical) and must not be claimed bitwise reproducible across unpinned sklearn environments; primary source gain is compared within the same successful V69 run where original V67 fuzzy scores reproduce exactly.

## Drift contrast and source stability

The originally **unlabeled TRAIN between-batch/within-batch score variance ratio** decreased from **.1157366 (original V67 ICA)** to **.0134272 (V69 contrastive)**. This confirms a statistical effect of the changed source objective, NOT proof it removed exclusively physical sensor drift. Gas prevalence differs by acquisition batch (some historical periods contain no toluene), so a batch-contrastive source objective can discard genuine chemical class information correlated with batch.

Leave-one-original-TRAIN-batch-out re-fits: source rank stayed K4 in all seven folds, and source subspace canonical correlations squared averaged .97435–.99976 depending on withheld batch. The geometry is consistent but not perfectly invariant and **source stability is not chemical semantic identification**.

## Failure of gas identity still persists

Fix one-to-one anonymous source→gas label assignments on VALID batches8+9, then evaluate same orientation without rematching on already viewed TEST batch10:

| Physically labeled gas | Fixed V69 source TEST OvR AUC | Warning |
|---|---:|---|
| Ethanol | .6342 | Unstable gas specificity |
| Ethylene | **.5213** | Essentially uninformative / almost random |
| Ammonia | .8739 | Association only |
| Acetaldehyde | ABSTAIN | No K4 source assigned |
| Acetone | .8932 | Association only |
| Toluene | ABSTAIN | No K4 source assigned |

Thus even when the **between-batch source drift measure shrinks by ~8.6×**, the label-conditional gas identity of ethylene deteriorates. This is a direct demonstration that `P(source)` invariance and cross-batch covariance design do not guarantee stable `P(gas|source)`. No unique chemical source factors were proven; this is one testbed with pure-gas, not independent source mixture interventions.

## Scientific decision

**Keep V69 as a bounded exploratory negative+partial-positive result**, not a novel verified chemical concept generator. It improves the representational objective and some heldout task scores but does not beat raw features, establish identifiable physical sources, or transfer V65 recurrent dynamics. The original code and tests remain in this research branch, accepted IOI/PIERE manuscripts and master unchanged.

The highest-value next independent real experiment is UCI308 **Gas Sensor Array Under Flow Modulation**, which has 58 physically independent acetone/ethanol gas mixture trials across five acquisition sessions with actual per-trial gas concentrations. Register strict **entire-trial** independent split prior to seeing the labels; do not treat 16 sensors×7500 time samples as millions of independent episodes. With only eight final trial-level TEST samples, any estimates will be low-power descriptive, not physically unique identification proof.
