# ConceptFAN V57 — executed external-assay numeric concept with matching native source head
**13 November 2026 · EXECUTED EXPLORATORY; no clinical or causal certification.** Draft [PR #60](https://github.com/fims9000/FuzzyAttentionNetworks/pull/60). Protocol was frozen in `CONCEPTFAN_V57_ASSAY_CALIBRATED_VALUE_FROZEN_PROTOCOL_20261113.md` before TEST. Successful [GitHub Actions run 43474307678](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43474307678), source commit `e3680c23ff79959c5443722b2a02f3a8762b617c`. Artifact `conceptfan-v57-assay-scaled-risk-head` ID `14062152166`, digest `sha256:f6d6dc77fbbdf1a3bdf1d533bdf209eca8cff4111ec5b3901c3193f8f3e1e9cf`.

## Actual architecture and verification
V54 proposes a source channel/lag/nonlinear operator from TRAIN X only. V55 learns constrained fuzzy mixture and signed source effect from TRAIN Y. V56 validates the selected source against a separately simulated, independently noisy external target-energy measurement on **190 purchased TRAIN + 100 VALID1 + 100 VALID2** episodes and requires predictive dual-VALID acceptance. **V57 changes the *numeric q***: TRAIN-fitted independent external assay readout `E_hat=max(0,slope*q_V55+intercept)` followed by `q_assay=sigmoid((E_hat−TRAIN_assay_median)/TRAIN_assay_IQR)`. When sensors are missing q_assay=.5.

**Critical contract:** the final signed source coefficient is freshly fitted on TRAIN Y using **q_assay**, *not transplanted* from a V55 head fitted on q_V55. On the useful source, V55 native gamma was about **4.99**, whereas V57 refitted gamma was **3.30/3.33** across two seeds. Independent dual VALID predictive gates accepted the new q in both seeds: VALID NLL improvements approximately .077/.081 with positive 99% one-sided resampled lower bounds (descriptive, same VALID cohorts used for earlier selection). The frozen K2 encoder and weights are never refit using TEST.

The new numeric q enters the real native PyTorch ConceptFAN signed concept bottleneck, where the per-episode risk decomposition is verified. If reference identity/device, candidate contract, risk admission, source support or cohort drift fails, the head returns **bitwise exact original K2**. Reference ID is not a cryptographic certificate. The TEST has **no external assay** or hidden generator source.

### Fresh held-out TEST, MCAR20; AUPRC / NLL, mean of two native seeds

| Mode | K2 | V55 unrestricted | V56 assay gate, old q | V57 calibrated q + new gamma | Energy-engineered logistic | Same assay-derived q + engineered logistic |
|---|---|---|---|---|---|---|
| Useful new source | .6332 / .6721 | .7667 / .6114 | .7667 / .6114 | **.7637 / .6000** | .8219 / .5076 | **.8222 / .5065** |
| Useless delayed sources | .7136 / .6571 | .7136 / .6571 | .7136 / .6571 | .7136 / .6571 | .7038 / .6012 | .7038 / .6012 |
| Predictive device artifact | .6407 / .6685 | .7792 / .6123 | .6407 / .6685 | .6407 / .6685 | .8335 / .4917 | .8335 / .4917 |
| Permuted assay, same X/Y as useful | .6332 / .6721 | .7667 / .6114 | .6332 / .6721 | .6332 / .6721 | .8219 / .5076 | .8219 / .5076 |

**V57 calibrated new-concept learning works**, and the numeric rescaling/refit lowers healthy TEST NLL from .6114 to .6000 in this particular synthetic world. But AUPRC changes from .7667 to **.7637**, not an improvement. V57−V56 AUPRC mean about −.0030, descriptive 95% paired interval **[−.01176,+.00617]**; no confirmed superiority. Against engineered logistic with the *same assay-derived q*, V57 is worse in both AUPRC and NLL (AUPRC interval **[−.08799,−.03204]**). Never claim model superiority.

### Negative controls and source scope
- The outcome-irrelevant case refuses all new concepts, preserving K2 exactly.
- The predictive recording artifact is accepted by V55 but **rejected** by V56/V57 because it lacks corroboration with the independently measured target; this sacrifices AUPRC .7792→.6407 in exchange for a conditional measurement-identity refusal.
- Permuting the external reference **without changing X or Y** causes V56/V57 to refuse while V55 remains identical.
- Even-energy q is invariant to inversion of both selected sensor signs; this is **algebraic parity** not a real physical validity test.
- A TEST-only 1.5× gain shift triggers cohort-level source revocation; V57 returns K2 (.6332 AUPRC) even though unrestricted V55 retains .7081. This is a measurable calibration/scope tradeoff. Forced missingness also returns exact K2.

### Reproducibility and limits
A new previously unexamined generator seed **20265713**, 950 TRAIN / 400 VALID1 / 400 VALID2 / 650 TEST, 4 modes ×2 native seeds ×6 TEST stresses = **8 trained K2 and 48 individual TEST prediction arrays**, all hashes, AUPRC/NLL, source contributions, concept scale and exact fallbacks independently rechecked. Pairwise subject bootstrap uses the same 650 TEST subjects across seeds and 399 resamples; all bands descriptive and unadjusted for multiple comparisons.

Limitations: one synthetic generative family despite new random test cohort; same VALID subjects used in V56 source admission and V57 risk-head admission (not a fully independent second confirmation); source reference IDs untrusted strings; no real physical assay or clinical subject labels; no independent device-family transfer; q learned from X may be unidentifiable under gauge transformations; no causal semantics or physical source identity guaranteed. An independently measured E constrains the value scale only under declared assumptions. Do not merge to master or retrofit already submitted papers based on these results.

**Remaining empirical blocker:** real multilocation sensor measurements with an independently trusted reference to the same physical property, explicit device provenance, diverse nuisance shifts and prospective validation before a physically meaningful FAN concept label can be defended.
