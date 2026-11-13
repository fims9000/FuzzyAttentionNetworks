# ConceptFAN V55 — constrained joint new-concept calibration, executed and independently audited

**13 November 2026 · Status: EXECUTED EXPLORATORY.** Draft [PR #57](https://github.com/fims9000/FuzzyAttentionNetworks/pull/57). Frozen prospective design: [V55 protocol](./CONCEPTFAN_V55_CONSTRAINED_JOINT_FROZEN_PROTOCOL_20261113.md). Successful [CI 43470366025](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43470366025), exact source `2fd894432b4e6bfdaaaf2ebdbb2781a7087a8666`; artifact `conceptfan-v55-constrained-joint-research` ID `14059844164` (SHA256 `3c6cbca91965ab5bedbe3f07471d86a148ac472300c18defec73be26e7d126e3`).

## Implemented architecture
- The V54 TRAIN-only program discovers which **measured channels, lag and explicit signed/energy operator family** form a candidate concept, without Y. 
- A constrained native PyTorch fuzzy operator **jointly fits the two nonnegative mixture weights, bounded temperature, and signed gamma** from TRAIN Y. Frozen V54 q regularizes source identity, so task learning cannot arbitrarily relabel sensors or invent a hidden bypass. Mixture weights sum to one; temperature constrained [.5,2], gamma [-10,10].
- The new statistical concept actually contributes to the `TemporalConceptFANModel` native signed concept-only decision path, with machine-checked exact K2 fallback for rejected/missing candidates. Independent disjoint VALID1/VALID2 cohorts admit at most one new concept.
- All 30 TEST prediction arrays from 3 modes ×2 native K2 random initializations ×5 preregistered TEST stresses were recomputed independently: labels, hashes, AUPRC/NLL, signed source contribution, exact refusal and subject-paired posthoc bootstrap. CI unit and regression tests passed.

## New cohort (not previously evaluated V54 TEST), MCAR20, mean of two seeds

| Mode, metric | K2 | V53 signed-only | V54 frozen operator | **V55 joint definition** | engineered energy logistic |
|---|---:|---:|---:|---:|---:|
| Useful nonlinear source AUPRC | .63325 | .63325 | .79682 | **.80237** | **.80754** |
| Useful nonlinear source NLL | .67407 | .67407 | .64145 | **.61117** | **.53138** |
| Irrelevant delayed sources AUPRC | .67105 | .67105 | .67105 | **.67105** | **.70618** |
| Artifact source AUPRC | .64516 | .64516 | **.77780** | .76811 | **.81738** |
| Artifact source NLL | .66448 | .66448 | .63385 | **.60127** | **.48107** |

V55 chooses the `EVEN_ENERGY_LAG` source channels 4/5 for the useful synthetic source, 8/9 for a **predictive recording artifact**, and **refuses** all new sources on the irrelevant mode (both seeds, exact K2). On the useful source, mixture source weights were .485 and .486, temperature 1.596 and 1.553 and gamma 4.861 and 4.843, respectively; thus the concept definition did change under supervised TRAIN. Train objective decreased by ~.016 per model.

### Descriptive posthoc paired patient intervals, 399 bootstrap resamples
- Useful nonlinear source V55−K2 ΔAUPRC = **+.16912**, 95% `[+.11830,+.21611]`.
- Useful V55−V54 ΔAUPRC = **+.00555**, 95% `[-.01245,+.02222]`: **no statistically supported superiority** of joint calibration over V54 on AUPRC.
- Useful V55−engineered logistic ΔAUPRC = **−.00517**, 95% `[-.03822,+.02552]`: **no demonstrated superiority** over this stronger baseline, and NLL is still substantially worse.
- Artifact V55−K2 ΔAUPRC = **+.12295**, 95% `[+.06257,+.17810]`.
- Artifact V55−V54 ΔAUPRC = **−.00969**, 95% `[-.03256,+.00760]`; V55 improves NLL but not AUPRC.
- Artifact V55−engineered logistic ΔAUPRC = **−.04927**, 95% `[-.07568,−.02168]`: the strong equal-label baseline is clearly better on this synthetic case.
- All intervals are conditional, unadjusted for multiple comparisons. Two seeds share the same 650 TEST people; they are **not** 1300 independent cases.

### Stress tests
- Under inversion of **both** source/readout signs, the trained EVEN operator has algebraically unchanged q and V55 AUPRC/NLL (useful .80237/.61117). This is an operator symmetry, **not** evidence that the physical apparatus is safe under arbitrary corruption.
- Under 1.5× unknown gain on the paired channels, V55 AUPRC **falls to .75741** on useful nonlinear source vs .78021 frozen V54, and **.72738** on predictive artifact vs .75680 V54. More expressive jointly fitted source can become *less* robust in ranking. NLL changes differently, highlighting calibration/ranking tradeoff. No post-TEST threshold changes are allowed.

## Scientific conclusions
**Verified:** V53's lagged signed measured concepts can be extended to outcome-blind nonlinear even-energy discovery (V54), and the discovered programs can now be constrained-jointly calibrated with risk labels inside the true native FAN concept-only head (V55). Exact refusal, observation masks, dual VALID admission, source provenance, and individual TEST checks work.

**Not supported:** generic concept or physical-source discovery; causal identification; clinical validity; superiority over engineered logistic regression; universal robustness to device shifts. A **predictive recording artifact is still admitted**; V50 observational nonidentifiability is not removed by nonlinear training. The tested dataset family remains synthetic despite fresh V55 cohorts.

**Recommendation:** preserve V54 as simpler frozen-operator option and V55 as constrained trainable option, compare by decision calibration and stability rather than claiming V55 universally better. Do not merge this exploratory PR into master or rewrite the published IOI article until a genuinely independent physical measurement reference and a new held-out device-family study exist.
