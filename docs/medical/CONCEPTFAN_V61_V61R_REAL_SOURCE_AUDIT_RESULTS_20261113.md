# ConceptFAN V61 and V61R: what the real conditional-source experiments established

**Research note, 13 Nov 2026. Original V61 experiment was preregistered; V61R is a separately disclosed posthoc numerical stability repair.** Both use previously inspected PhysioNet BIDMC and VitalDB cohorts; they cannot prove generality, medical efficacy or physical source identity.

## Original V61: genuine 60s-future tasks, non-PULSE new concepts

Full [V61 protocol](https://github.com/fims9000/FuzzyAttentionNetworks/blob/experiment/conceptfan-v61-conditional-novelty-20261113/docs/medical/CONCEPTFAN_V61_CONDITIONAL_SENSOR_NOVELTY_FROZEN_PROTOCOL_20261113.md). Independent CI audit: [BIDMC run 43488423056](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43488423056), [BIDMC→VitalDB run 43488669890](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43488669890). Exact aggregate-only GitHub Actions artifacts are independently auditable from source files and protocol.

### Causal chronology and leakage
- Both targets start **60 seconds after** input history ends; four future monitor seconds define the binary target.
- **HR task:** ECG-derived monitor HR ≥100 bpm, input observed PULSE/RESP/SpO2.
- **RESP task:** monitor RESP ≥20 breaths/min, input PULSE/SpO2 only. **RESP is completely excluded** from model input, TRAIN normalization, and source compiler (bitwise unit test when its excluded values are arbitrarily modified).
- Sources use 16s causal current/past masked values; 8s stride. Patient-disjoint BIDMC 27 TRAIN /6 VALID1/6 VALID2/7 TEST groups; VitalDB 40 external people selected by V59 availability only. Temporal windows overlap within a patient, so effective sample size is patient count, not window count.
- An honest and strong **PULSE-only native concept head** is first fitted. Conditional candidate programs cannot read PULSE, and their residual numerical values come from TRAIN-only, group-weighted Ridge prediction from PULSE program summaries and mask bits. At most six new signed source contributions are fitted on TRAIN Y. Admission requires dual whole-patient VALID improvements (99% lower confidence limit >0 and patient-macro NLL gain ≥.005), plus each environment outperforming 49 within-patient PULSE-quartile source permutations by ≥.003. Otherwise exact original PULSE-only risk is returned.
- This only tests *incremental predictive association not linearly explained by pulse*, not physical source separation, a causal mediation test, or an independent device-verified assay.

### Independent heldout natural-missingness TEST (first original V61 external runner, seed 11)

| Task & site | Real persons | Overlapping windows | Positive rate | Pulse-only native AUPRC / NLL | Proposed conditional AUPRC / NLL | **Deployed** conditional AUPRC / NLL |
|---|---:|---:|---:|---|---|---|
| HR+60, BIDMC | 7 | 408 | 2.7% | **.2116 / .0928** | .1860 / .0973 | **.2116 / .0928** |
| HR+60, VitalDB | 40 | 2010 | 14.1% | **.7474 / .2651** | .7528 / .2704 | **.7474 / .2651** |
| RESP+60, BIDMC | 7 | 408 | 16.7% | **.1766 / .4826** | .1371 / .6751 | **.1766 / .4826** |
| RESP+60, VitalDB | 40 | 1905 | 7.2% | **.0755 / .3560** | .1025 / .2737 | **.0755 / .3560** |

The RESP+60 proposed model has a tempting **exploratory VitalDB-only** improvement, but the same frozen proposal **worsens** the local BIDMC TEST; neither the 99% lower bound on the two VALID patient cohorts nor the 49-shuffle conditional-null gate passed. There is **zero accepted independent non-PULSE source** in four V61 fit/gate trials (two tasks, ostensibly two seeds).

**One additional crucial integrity problem:** original V61 seeds **11 and 23 are identical trained models** (V60 torch parameters initialized deterministically, despite setting different seeds). This is not independent replication. Further, two otherwise matching GitHub CI workflows showed TRAIN Ridge changes of only ~1e−14 but selected different gamma-ranked top-six source *identities*, changing *ungated* TEST ranking noticeably (e.g. HR+60 BIDMC proposed AUPRC .216 vs .186). The model's deployed exact PULSE-only predictions and refusal decision remained stable. No scientific claim should rely on original V61 *ungated* candidate ranking.

## V61R: separately labeled, posthoc engineering repair

[Explicit V61R amendment](./CONCEPTFAN_V61R_POSTHOC_NUMERICAL_STABILITY_AMENDMENT_20261113.md) and [draft PR #75](https://github.com/fims9000/FuzzyAttentionNetworks/pull/75). Source definitions now use a deterministic TRAIN-outcome-blind selection of up to 6 group-diverse residual programs before gamma training; supervised gamma can no longer change program identities. The same TRAIN Ridge coefficients and numeric scales are rounded to 12 decimals and used in train/test. The native PyTorch head permits *genuinely different* seeded initial alpha/offset/gamma values with unchanged default behavior for older V60/V61 callers; outcome-blind dictionaries retained in the fixed risk head and no top-six magnitude reranking after Y. CPU deterministic single-thread mode. Strict original VALID +49 null gate unchanged.

[BIDMC V61R run 43489865559](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43489865559): **SUCCESS** on unit and independent per-person case audit; 6 HR novel source expressions and 4 RESP non-PULSE source expressions fixed identically across two genuinely distinct model seeds. Conditional gamma differs between seeds; both tasks still **refused** the new sensor signal on independent BIDMC VALID+null gates.

V61R remains a **posthoc sensitivity/reproducibility analysis**, not a confirmatory repeat of preregistered V61. Any gains after changing model selection code are exploratory and should not be compared as preregistered results. The remaining strict engineering gate is replaying **two completely independent training Python processes** on identical patient groups and source data, verifying exactly matching canonical source identity/hash and max per-episode logit drift ≤1e−4; no reliance on merely printing "success" or on seed-based pseudo-replication. If the replay fails, report the failure.

## Scientific interpretation

The original architectural goal has been decomposed into distinct testable claims:
1. **Typed statistical concepts from measured signals:** demonstrated by V60, 20–21 real TRAIN candidates.
2. **A differentiable signed, inspectable risk bottleneck:** implemented and case-audited in V60/V61; no raw-to-risk shortcut. A "PyTorch-native concept head" is not the original V47 Transformer encoder; state this distinction.
3. **Signal beyond a dominant sensor PULSE:** not demonstrated by original V61; deployment refused new sources.
4. **Stable source identities:** brittle in V61 despite correct split/gating; V61R addresses this as an engineering failure, not by model-performance cherry-picking.
5. **Newly identified physical physiology / trustable causal concepts / medical utility / superiority over PULSE-only:** NOT established. The monitor-derived outcome itself can be closely related to a source monitor, and no independent physical metrology is available.

Research next step after replay: a genuinely new domain and untouched subject-disjoint cohort with distinct accelerometer+gyroscope sensors and externally protocol-labeled task (e.g. original UCI HAR, DOI 10.24432/C54S4K), or an independent physical assay benchmark. Pre-register exact source channels, task and patient/participant partitions BEFORE any new TEST outcomes; preserve strong accelerometer-only and all-sensor baselines and do not promote source candidates using unseen-test feedback.

**No master merge and no retroactive article claim from the above exploratory medical benchmark.**
