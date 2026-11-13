# V62 executed — cross-view response concepts on real gas interventions

Date: 2026-11-13. **STATUS: EXECUTED exploratory reanalysis, NOT independent confirmation.** V62 was designed after V61's exact same 30-run TEST had been inspected. V62 hypothesis and hyperparameters were committed before V62 numeric outputs, but the V61 study motivated the method and the TEST sample was no longer sealed. Do not claim a separate confirmatory test.

Successful [GitHub Actions run 43476431679](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43476431679), executed commit `099fbdbae7fc979d6720dcd3dc7a24070bb71f34`, artifact ID `14062373563` name `conceptfan-v62-crossview-post-v61-exploratory`. Machine-readable `V62_EXPLORATORY_EVIDENCE/STATUS.json`, `METRICS.csv`, source SHA and log; scientific method frozen in `CONCEPTFAN_V62_FROZEN_CROSSVIEW_CONTRAST_PROTOCOL_20261113.md`.

## Precise new concept-generator mechanism

The physically controlled [UCI Gas Sensor Array Exposed to Turbulent Gas Mixtures (id309)](https://archive.ics.uci.edu/dataset/309/gas+sensor+array+exposed+to+turbulent+gas+mixtures) has 180 whole-run trials, 30 physically independent ethylene/CO or ethylene/methane flow configurations each tested six times, 8 physical chemoresistive sensors, 60 five-second bins per run. Exactly the V61 original run split 120 TRAIN/30 VALID/30 **already inspected TEST** was reused; no gas label or flow level enters source fitting.

Instead of positive-lag average predictive covariance, V62 forms two unlabeled per-run sensor views:
`A=mean(sensor at 60–150s)-mean(sensor at 0–30s)`, `B=mean(sensor at 150–240s)-mean(sensor at 30–60s)`. Computes centered TRAIN `Cov(A,B)`, then signed left-singular sensor directions. The number of directions K is selected by **all-run** 95th percentile max-singular-value null of 100 permutations of whole TRAIN run identities in view B, with 20 independent-run bootstraps. Labels and gas count never influence K.

The fitted basis is a statistical observable response subspace; it is not a chemical species identification. Native V56 frozen temporal fuzzy encoder is trained on 3 temporally disjoint crops per original TRAIN run (360 *dependent* unlabeled augmentation crops, not 360 independent source runs); task-specific logistic heads see the SAME 120 real gas run labels as all comparators and only the concept means. True raw features cannot bypass the concept-only signed additive logit. The original positive-lag V61 source bank and its native model were refit exactly under the old protocol for paired comparison; raw logistic and exposure-timing-aware logistic retained.

## TRAIN rank evidence (without gas supervision)

| Statistic | Old positive-lag V61 | New cross-view V62 |
|---|---:|---:|
| Automatically chosen K | 1 | **2** |
| TRAIN run-bootstrap interval | [1,1] | **[1,3]** |
| Leading signal singular values | ~5.0064; ~0.8410 | 1.35795; 0.29284; 0.23421 |
| 95% familywise max-null singular threshold | 2.8566 | 0.27105 |

V62's second direction clears its fixed null threshold on the full independent TRAIN runs, but K is not reproducibly exactly 2 under whole-run resampling: 5/20 bootstraps returned K1, 11/20 K2, 4/20 K3. Thus no stable physical source count has been established, and source volatility may contribute additional axes.

## Real controlled gas VALID / already-seen TEST classification

Each VALID and TEST contains 30 **whole runs** with gas presence prevalence ethylene .8, CO .4, methane .4. The following comparison does not provide an unobserved independent replication.

| Gas label | Method | VALID AUPRC / NLL | reused TEST AUPRC / NLL |
|---|---|---|---|
| Ethylene | V61 native K1 | .9286 / .4702 | .9682 / .4534 |
| Ethylene | **V62 native K2** | .9354 / .4680 | **.9732 / .4491** |
| Ethylene | Raw logistic | .9894 / .2298 | **.9928 / .1955** |
| CO | V61 native K1 | .6840 / .6397 | **.7296 / .6430** |
| CO | **V62 native K2** | .6477 / .6147 | .6114 / .6321 |
| CO | Raw logistic | **.9936 / .1328** | **.9936 / .1621** |
| Methane | V61 native K1 | .5523 / .6600 | .6816 / .6500 |
| Methane | **V62 native K2** | **.9297 / .5225** | **.8241 / .5586** |
| Methane | Raw logistic | **1.0000 / .1094** | **.9707 / .2286** |

**Interpretation:** automatically increasing the statistical source vocabulary from K1 to K2 improves the model's sensitivity to methane significantly in this real physical record family, but CO *AUPRC decreases*, and raw models still strongly dominate. CO NLL slightly improves even as ranking falls; use both measures. V62 static K2 was slightly better than V62 native K2 on CO TEST (AUPRC .6381 vs .6114), so native temporal recurrence itself does not uniformly help.

## Physical-source *association* audit

On previously seen TEST, per-run exposure-minus-baseline fuzzy concept activation was compared posthoc with independently controlled mean gas concentrations (GC-MS validated) using full K×3 Spearman. The maximum absolute association across concept slots:

| Gas flow | V61 K1 | V62 K2 | best original sensor |
|---|---:|---:|---:|
| Ethylene | .6769 | .6875 | **.8153** |
| CO | .3834 | .3678 | **.8360** |
| Methane | .2321 | **.5827** | .5863 |

V62 gains an axis associated with methane control, but not a clean CO-specific concept. Even methane max abs correlation is comparable to one original sensor; no proof of source-specific sensitivity, independent component identity or cause. The shuffled-condition control yielded abs correlations up to .369 for methane in only 30 independent TEST runs, another reason to avoid declaring semantic identification.

## Readiness and next valid research gate

**Verified:** an explicit train-only and label-free real multi-view estimator can support K2 measured-response slots on this dataset and help one gas. Underlying unit tests check zero-source abstention, independent train-run permutations, hidden measured-value invariance and truly observed baseline bins. The full CI and numeric comparisons passed.

**Not verified:** stable two-source physical identification, CO-vs-methane separation, independent prospective predictive advantage, consistency across instruments or clinical interpretation. A further K choice tuned on the same already viewed gas TEST would be additional model selection, not an independent statistical trial. Real confirmation needs *new* physical runs or a different wind-tunnel/chemical-sensor source dataset and preregistered concept-to-source alignment before seeing results. An instrument-level paired reference and recorded hardware faults are still needed to verify V57–V58 claims.

The source bank itself is grounded in existing multi-view cross covariance and permutation testing, **not** an original source-identification theorem. Preserve failures and keep all connected PRs DRAFT. Accepted PIERE/IOI/clinical submissions are untouched.
