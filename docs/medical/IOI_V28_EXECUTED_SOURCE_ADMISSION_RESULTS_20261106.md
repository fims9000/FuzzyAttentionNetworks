# ConceptFAN V28 — executed frozen-source admission under device changes

**Executed 2026-11-06.** Final corrected GitHub Actions run [42118864192](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/42118864192), source code commit adf3fef99adfdd9b8e09556df797c52cfc4b8050, artifact ID 13456279533 ("conceptfan-ioi-v28-isolated-source-2"). A prior successful run 42118580375 reported zero mean sensor support due to a *summary field name mismatch only*; the final corrected run must be cited for coverage numbers. Final: **9 tests passed, 3 unique full K2 neural fits**, 375 detailed TEST evaluations, 175 aggregate cells and 15 recorded VALID gate decisions (including observationally duplicated stable instrument interpretation). No clinical data or PIERE edits.

The pre-run [protocol](./IOI_V28_FROZEN_ISOLATED_SOURCE_PROTOCOL_20261106.md) fixed a new random synthetic draw seed20261108 from the SAME V25–V27 generator family. TRAIN1800 episodes for K2 staged concept training and risk-source fitting, VALID1200 independently separated into 600/600 labelled episodes for gate, frozen TEST900. Three NN seeds 11/23/37, original K2 concept path shared across task-worlds and verified unchanged by state SHA.

## What is preserved and what is NEW

Train only the first TWO known synthetic concept predictors (TRAIN oracle concept trajectories allowed), then freeze them. Freeze the linear K2 risk head as well. A third SOURCE is **not a free latent bypass**: it is an explicitly named, fixed operator on the measured channels 4+5, evaluated only with observed historical values and a support flag. Its additive coefficient gamma is fitted on TRAIN labels; the final risk is exactly

    logit = old_bias + old_w1 * q1 + old_w2 * q2
            + I[approved AND measured] * gamma * (q_candidate - 0.5).

Any rejected or unsupported patient gets the EXACT original K2 logit. This protects the already defined concept values AND their decision coefficients; the new source has its own visibly separate signed contribution. Unlike the V27 staged method, the candidate is manually supplied from known sensor channels: **no automatic concept invention or semantic certification**.

## Validation gate outcomes

A source is globally admitted only when two disjoint labelled VALID environments *both* show mean paired TEST-independent (VALID only) NLL gain >=0.002 and bootstrap 95% lower bound >0. At TRAIN fitting, prior K2 and candidate source coefficients use the same 1800 outcome labels; the gate additionally consults **1200 VALID labels**, so total supervision is not matched to a simple K2 risk baseline. This is an extra safety-validation resource, not an equal-label experiment.

| World | Gate seed 11/23/37 | VALID-env1 mean NLL gain (approx.) | VALID-env2 mean NLL gain (approx.) |
|---|---|---:|---:|
| Stable extra predictive source | admitted / admitted / admitted | +0.043 to +0.047 | +0.043 to +0.046 |
| Third source not predictive (nuisance) | rejected / rejected / rejected | approx. 0 | approx. 0 |
| Instrument channel-pair inverted in VALID-env2 | rejected / rejected / rejected | +0.043 to +0.047 | **-0.066** |
| Instrument channel-pair inverted only in unknown TEST | **admitted / admitted / admitted** | +0.043 to +0.047 | +0.043 to +0.046 |
| Stable instrument world with identical observed distribution | same admitted decisions as stable source | identical | identical |

The last two rows demonstrate the identifiability boundary: known validation shifts may be rejected but **unseen shifts cannot be inferred**, and identical observational data cannot identify physical semantics.

## Independent 900 TEST subjects, MCAR20 missingness

All metric entries below are mean across 3 NN initializations for neural predictors, single fit for external raw logistic. The 900 TEST subjects are SHARED across fits. Lower NLL and higher AUPRC are better.

| Scenario | Frozen K2 AUPRC / NLL | Forced isolated candidate AUPRC / NLL | 2-env qualified isolated candidate AUPRC / NLL | Standard refitted K3 CBM AUPRC / NLL | Raw-measured logistic AUPRC / NLL |
|---|---|---|---|---|---|
| Stable extra signal | .6719 / .6222 | .7249 / .5810 | **.7249 / .5810** | .7590 / .5560 | .7563 / .5599 |
| Nuisance third source | .7135 / .6030 | .7139 / .6030 | **.7135 / .6030 (reject)** | .7142 / .6032 | .7092 / .6084 |
| Device shift already present in second VALID env | .6696 / .6243 | .6021 / .6863 | **.6696 / .6243 (reject)** | .5160 / .8480 | .5085 / .8780 |
| Device shift occurs ONLY on unknown TEST | **.6696 / .6243** | .6021 / .6863 | .6021 / .6863 (**FAIL**) | .5160 / .8480 | .5085 / .8780 |

At MCAR20 the candidate measurements have full per-episode eligibility by the predeclared >=0.50 sensor-support threshold, so rejected world SOURCE USAGE is exactly 0 and admitted world source usage is 100%; observed fraction of raw sensor cells is not the same as eligibility, and this is not a statement that all sensors were present. The corrected aggregate source *support-eligibility* fraction is 100% in all MCAR20 neural rows.

**Honest negative results:**
- Even in the stable world, the isolated source AUPRC .7249 remains below ordinary K3 refit (.7590) and raw logistic (.7563). Freezing the original risk coefficients constrains potential ranking gains.
- Strong additional candidate under observed device shift would harm TEST considerably; gate catches it only because the shifted environment was included in VALID.
- Under unobserved device shift, the validation gate incorrectly admits it and hurts AUPRC by .0675. A gate with only historically known environmental labels is not a physical or causal certificate.
- The original K2 encoder *also* sees sensors 4/5 at input, despite stage-one concept supervision. Therefore a complete absence of any third-factor information in K2 is NOT proven; only its parameter preservation by this extension is enforced.
- 3 seeds, one generator family, 900 TEST cases, extra VALID labels. Avoid cross-seed significance claims or medical transfer.

## Direct continuation

A fixed check of **cross-sensor pair consistency** using only jointly observed historical measurements may detect *single-sensor* sign/bias faults without using TEST outcomes. But simultaneous common-mode flips of both channels should defeat a pair-only sensor agreement check. Freeze and test both positive and negative cases on another draw; do not retroactively optimize thresholds on V28 TEST. Compare the best constraint with source refit, true oracle and existing published residual CBMs; no claim of new method yet.

Literature and publication overlap boundaries: [IOI_V28_RELATED_WORK_AND_NOVELTY_BOUNDARY_20261106.md](./IOI_V28_RELATED_WORK_AND_NOVELTY_BOUNDARY_20261106.md).
