# ConceptFAN V29 — executed unlabeled two-sensor consistency guard and failure modes

**Executed 2026-11-06.** GitHub Actions [42119670144](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/42119670144) SUCCESS, source commit 741b2cbf5763830ec84c33219f61553680243bf7, immutable artifact [13458008037](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/42119670144/artifacts/13458008037) (STATUS/SUMMARY/GATES.json, full log and smoke). **12 pytest tests passed**, 3 unique native K2 model fits, 288 detailed evaluations, 144 aggregate cells. Five actual TEST scenario outcomes plus one observationally duplicate physical-source identity, not extra independent fits.

The V29 [protocol](./IOI_V29_FROZEN_SENSOR_PROVENANCE_PROTOCOL_20261106.md) was committed BEFORE generating V29 TEST data. Generator seed20261109: new 1800 synthetic TRAIN, 1200 VALID (two separate 600-patient environments), 900 independent TEST episodes, same 16-step three-factor AR generator family as previous V25–V28. Stage-one known K2 concepts learned using synthetic TRAIN oracle targets only; original concept neural weights and K2 risk coefficients frozen. The third source is the named sensor4+sensor5 readout defined in V28, trained separately on TRAIN risk labels and accepted via 1200 additional VALID labels. All three neural seeds (11/23/37) admitted predictive source; all three rejected irrelevant nuisance.

## Additional source integrity without outcome labels

The V29 monitor compares TWO measured sensor readouts assumed by the generator to measure the same source with independent sensor noise. Only jointly observed historical time points count. Thresholds fixed on TRAIN without outcome labels or TEST values:

- at least **5 jointly measured time steps**;
- signed pairwise Pearson correlation >= **0.357386** (1% TRAIN quantile with nonnegative floor);
- mean absolute pair difference <= **0.580097** (99% TRAIN quantile);
- combined with V28's per-patient source support and two-environment global risk gate; otherwise **exact K2 frozen risk**.

TRAIN fraction passing sensor check **97.78%**; TEST stable at MCAR20 **97.44%**, thus a real healthy-data refusal cost. For monitoring, 'source disagreement' is NOT synonymous with 'source is physically wrong', nor proof about the patient. Strongly correlated/common-mode sensor artifacts may defeat this test.

## Main held-out TEST evidence

AUPRC on fixed 900 TEST episodes at MCAR20, arithmetic mean over three distinct native model initialization seeds:

| Scenario | Frozen K2 | V28 candidate with risk gate | V29 candidate with pair monitor | V29 source usage |
|---|---:|---:|---:|---:|
| Both sensors stable | .7288 | **.7961** | .7939 | 97.44% |
| Channel 5 sign inverted only on TEST | .7276 | .7257 | **.7276** | 0.00% |
| Channel 5 receives +2 offset only on TEST | .7287 | **.7918** | .7287 (**worse than V28**) | 0.00% |
| BOTH channels sign inverted only on TEST | **.7284** | .6225 (**harm**) | .6256 (**still harm**) | 97.44% |
| Additional source unrelated to Y | .7745 | .7745 (global rejection) | .7745 | 0.00% |

**This negative finding must stay visible:** Pair consistency can detect and block *individual* readout changes, but censoring the offset sensor also discards useful signal, so overall accuracy can DROP relative to V28. The simultaneous inversion keeps the two readouts mutually consistent and almost entirely slips through, substantially harming prognosis ranking. Thus V29 is an outcome-blind integrity/audit component, not a robust universal safety guarantee or a validated medical source identity certificate.

## Missingness cost, normal source

| TEST mechanism | Frozen K2 AUPRC | V28 candidate AUPRC | V29 monitored candidate AUPRC | V29 candidate used |
|---|---:|---:|---:|---:|
| MCAR20 | .7288 | **.7961** | .7939 | 97.4% |
| MCAR40 | .7278 | **.7801** | .7684 | 69.3% |
| MNAR | .7174 | .7688 | **.7691** | 56.9% |
| 8-step contiguous missing blocks | .7203 | **.7376** | .7269 | 17.4% |

Under long missing blocks the readout pair has too little simultaneous history, and monitor drops most extra-source contributions. Do not compare selected-only performance without reporting the covered subjects. All methods use the **same underlying TEST subjects** across modes; repeated masks and NN seeds are not independent cohorts.

## Scientific interpretation

1. Original K2 concept trajectory formation does not receive target gradients, and all V29 source changes are carried through an *independent named source*, preserving the existing concept coordinates and K2 risk coefficients. This resolves V27's specific hidden-information pathway by construction **under known synthetic K2 annotations**; does not prove high-level concept semantics in any real dataset.
2. Additional prediction usefulness on known environments can coexist with a harmful unseen physical sensing failure. The V28 two-environment risk gate is an observable association check, not a physical identity certificate.
3. A paired-sensor monitor can detect single-channel inconsistency under an explicit same-source sensor assumption, but cannot determine which channel is wrong without outside references. The +2 bias case demonstrates both a fault indication **and an unnecessary rejection of useful predictive evidence**. A common-mode sign flip preserves paired agreement: both V28/V29 fail.
4. Neither V28 nor V29 has been compared fairly to published information-bottleneck CBM, residual stochastic CBM, or dynamic concept discovery methods under matched label counts, FLOPs, new simulator families, external medical data or causal interventions. Literature overlap note: [IOI_V28_RELATED_WORK_AND_NOVELTY_BOUNDARY_20261106.md](./IOI_V28_RELATED_WORK_AND_NOVELTY_BOUNDARY_20261106.md). These methods are **experimental safeguards**, not established novel results or clinical guidance.

## Concrete research next step

Do not keep adding hard refusal thresholds as a substitute for real robustness. Under a KNOWN single-sensor bias, try **independently validated measurement repair** or a third independent corroborating channel, compare repair/reject against prior V29 on a *new generator law*, and count false repairs. Under pairwise simultaneous inversion, no two-readout agreement test can establish absolute orientation: external sensor calibration reference, intervention or third distinct reliable source is required. Preserve this failure as a motivation for a genuine methodological contribution, not an excuse to claim broad robustness.
