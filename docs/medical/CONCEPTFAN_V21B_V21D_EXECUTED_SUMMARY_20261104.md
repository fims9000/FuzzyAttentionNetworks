# ConceptFAN V21b–d — native signed-effect reproducibility, instrument-reset falsification, and bounded concept admission

**Research date:** 2026-11-04. **Status:** research-only Draft PR #18. This report extends the V21 baseline `CONCEPTFAN_V21_EXECUTED_MEASUREMENT_SOURCE_FALSIFICATION_20261104.md`; its originally executed 45 fits and evidence remain unchanged.

## Main research problem
ConceptFAN's purpose is to make the **actual native predictive latent concepts** understandable, reliable and controllable, not to add a plausible post-hoc explanation. V19/V20 improved prediction by separately optimizing the **existing native head**, without introducing a model bypass, but did not show that automatically formed concepts had physical/clinical identities. The V21 baseline showed predictive instrument artifacts can pass both source-persistence and risk association; a shared reference device can be contaminated by the same artifact. Native raw-source alignment was high, but signed patient-level risk contributions across seeds were unstable (~−0.333 rank Spearman).

V21b–d separately investigate reproducibility, controlled acquisition repeatability and safety of candidate insertion. Nothing here establishes physiological causality; all data are synthetic and all model-reviewer labels are simulated.

## 1. V21b: a fixed TRAIN-only initial risk orientation stabilizes the NATIVE decision, including incorrect concepts

**Frozen protocol:** `CONCEPTFAN_V21B_FROZEN_SIGNED_EFFECT_STABILITY_20261104.md`. [Actions 41850217581 — SUCCESS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41850217581). 30 preregistered opportunities, 24 trained native K3 models and six independently screened ABSTAIN cases. Records under `docs/medical/evidence_v21/signed_stability/`.

The signed-start arm differs **only** in a small `physical.decision_raw_weight[2] = sign(TRAIN-screened outcome association) × 0.05` prior to the same 10-epoch joint training. No permanent coefficient sign restriction and **no oracle** physical/instrument latent state enters source search, training or orientation. No post-hoc reorientation of patient contributions was applied.

| V21 scenario | Arbitrary start mean AUPRC | TRAIN-screened start mean AUPRC | Signed patient-effect mean rank Spearman arbitrary → screened |
|---|---:|---:|---|
| Physical predictive | 0.51018 | **0.69961** | −0.33328 → **+0.99984** |
| Predictive colored sensor artifact | 0.53972 | **0.76021** | −0.33311 → **+0.99995** |
| Predictive common-mode artifact contaminating two devices | 0.53376 | **0.72669** | −0.33318 → **+0.99994** |
| Physical nonpredictive | 0.49991 | ABSTAIN (3/3 seeds) | Unchanged original only |
| Artifact nonpredictive | 0.49882 | ABSTAIN (3/3 seeds) | Unchanged original only |

The result is a strong, concrete **initialization/optimization control** on this already-inspected simulator: naive arbitrary signed starts made the risk contribution reverse between seeds despite high correlation of raw concept evidence with the underlying measured source. But signed stability is also gained by **known artifacts**, so this correction cannot be marketed as discovery of true physiology. All three NN initializations share the same VAL episodes; no population-level significance claims.

## 2. V21c: a paired controlled acquisition reset exposes some instrument artifacts — and fails on persistent common-mode interference

**Frozen protocol:** `CONCEPTFAN_V21C_FROZEN_INSTRUMENT_RESET_20261104.md`. [Actions 41851699683 — SUCCESS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41851699683). Six declared conditions, 240 new paired calibration episodes each, 199 permutation and 199 bootstrap draws, original TRAIN-only normalizer and source/risk/device-reference evidence. Calibration sample seed 405111 and separate acquisition reset seed 405112, distinct from V21 source/risk and validation/test episodes. Evidence in `docs/medical/evidence_v21/instrument_reset/`.

| Paired simulator condition | Risk relevance | Cross-device original corroboration | Pearson r between first and repeated source measurements | V21c paired result |
|---|---|---|---:|---|
| Physical predictive | ACCEPT | ACCEPT | **0.99805** | ACCEPT |
| Physical nonpredictive | ABSTAIN | ACCEPT | **0.99805** | ACCEPT (descriptive source) |
| Device-only artifact predictive | ACCEPT | ABSTAIN | 0.02185 | ABSTAIN |
| Device-only artifact nonpredictive | ABSTAIN | ABSTAIN | 0.02185 | ABSTAIN |
| Shared-device artifact, independently resettable | ACCEPT | **ACCEPT** | 0.02185 | **ABSTAIN** |
| Shared environmental artifact persisting through device reset | ACCEPT | **ACCEPT** | **0.99839** | **False ACCEPT** |

This test assumes **a genuine externally controlled paired acquisition while the underlying physiological system stays unchanged**. The simulator has explicit knowledge of that condition; ConceptFAN itself does not infer it. It is an instrument-provenance experiment, not an intrinsic neural concept discovery theorem. A persistent environmental artifact still fools it, and this is an intentional negative result.

## 3. V21d: bounded scientific evidence governs ACTUAL concept insertion, with permanent descriptive-only risk veto

**Frozen protocol:** `CONCEPTFAN_V21D_FROZEN_CONCEPT_EVIDENCE_POLICY_20261104.md`. [Actions 41853611039 — SUCCESS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41853611039). All six preregistered cases in `docs/medical/evidence_v21/bounded_admission/`.

`concept_evidence_policy_v21d.py` admits sources only as **provisional measurable concepts** after independent reference + paired-acquisition evidence. Outcome-risk relevance controls whether the **native concept is allowed to influence prediction**; it is NOT required to call a reliably measured factor a descriptive concept. Explicit independent instrument-artifact provenance is a hard BLOCK, overriding all prediction and source-correlation statistics. Ordinary per-episode YES/NO/UNSURE/ARTIFACT feedback must not automatically become a candidate-level physiological certificate.

| Case | V21d native admission | Can update native risk after admission? |
|---|---|---|
| Physical predictive | ALLOW provisional measured risk source | Yes, after explicit training |
| Physical nonpredictive | ALLOW provisional descriptive-only source | **No; structurally masked to zero** |
| Device artifact predictive | ABSTAIN insufficient independent reference | No |
| Device artifact nonpredictive | ABSTAIN insufficient independent reference | No |
| Resettable common-mode artifact | ABSTAIN nonrepeatable acquisition | No |
| Persistent environmental artifact | **ALLOW provisional measured risk source — false physical interpretation** | Yes, purely observational; **never physically certified** |

Exactly **3 admitted / 3 refused**, one **descriptive-only native slot** with new serialized risk policy `signed_risk_policy == 2`. This mask is applied inside `MultiScaleUnknownNeutralConceptFAN.decision_weights()` and understood by both V19e LBFGS and V20 AdamW native-head refits via `_native_betas`; it is not a promise that can silently be undone by future training. All admitted source expansions have **zero initial effect** on old patient logits, forecasts and reconstruction. The current original graph and concept operators remain intact; no raw-to-logit bypass or extra post-hoc predictive head.

The persistent environmental artifact remains a deliberately preserved **false positive** at the provisional-measurement level. A separate simulated `instrument_artifact` source veto blocks it completely; **this was not provided by a real doctor**, and the software does not authenticate its validity. Absolutely no `PHYSICALLY_VALIDATED` or `CLINICALLY_VALIDATED` classification exists.

## 4. Separately implemented local four-choice reviewer workflow

`offline_native_review_v21.py` and `CONCEPTFAN_V21_LOCAL_REVIEWER_WORKFLOW.md` provide an offline Russian form with short measured curves and four answers: **Да / Нет / Не уверен / Артефакт измерения**. It signs TRAIN-only pseudonymous case packets with HMAC and imports authorized responses as existing **V16 native concept feedback**, not an independent classifier or task label. Unsure/artifact are **never interpreted as negative concept absence**. Tests verified HMAC/replay/partition validation and native operator gradient. No actual physician annotation study or local EHR integration was performed. Measurement curves may still contain sensitive information despite pseudonyms and must never be uploaded to public CI or GitHub.

## 5. What the evidence justifies / what it does not

**Demonstrated on these synthetic families:** V21b initial signed-risk orientation has a large impact on actual native patient-level ranking stability and AUPRC; V21c controlled repeat measurements can falsify certain artifacts that fool the independent-device test; V21d can keep independently observed but risk-irrelevant concepts in the **actual native concept graph** with a persistent zero risk contribution, and veto flagged artifacts. All source/oracle tests preserve separation: no hidden physical state used for training or patient risk, no REAL physician claimed.

**Not demonstrated:** automatic recovery of unknown physical or biological laws, causal source identifiability in observational medicine, robustness against persistent common-mode acquisition failures, real cross-hospital transfer, multi-source strongly mixed latent factor identification, official concept-bottleneck algorithm comparison, or statistically independent confirmation of claims on actual expert data. High raw-source alignment or near-perfect signed-effect Spearman **do not** prove semantic medical completeness.

**Next meaningful V22 research:** independently labeled, strongly non-diagonal multisensor mixtures; measurement source-vs-device calibration metadata, genuinely blinded clinician yes/no/artifact reviews, and a fixed independent prospective evaluation against published intrinsic CBM baselines. Without real provenance labels or external interventions, no amount of synthetic neural-head tuning can certify physiology. Accepted PIERE camera-ready unchanged; PR #18 remains Draft.
