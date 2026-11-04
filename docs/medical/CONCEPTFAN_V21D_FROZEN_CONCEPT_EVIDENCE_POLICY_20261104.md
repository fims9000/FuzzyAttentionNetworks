# V21d — frozen concept evidence policy and exact native admission safety audit

**Written before policy execution, 2026-11-04.**

## Motivation and explicit bounded claim
V21 showed task-predictive instrument artifacts, including a shared two-device artifact, that would be called meaningful concepts by naive risk relevance + cross-device correlation. V21b showed the artifact's signed native explanation can become nearly perfectly reproducible after correct TRAIN risk sign initialization. V21c showed controlled independent instrument resets separate resettable artifacts, but **persistent environmental interference** passes every statistical measurement screen even when the simulator explicitly knows that the source is an artifact.

Therefore an automatic source-discovery engine must **not** equate any set of these statistical tests with clinical/physical meaning. This is an engineering policy to prevent silent promotion of statistical features to medical concepts, not an artifact-identification theorem.

## Frozen evidence dimensions
A single V17 candidate ID, independent TRAIN V18 outcome-risk evidence, V21 independent-device measurement evidence, and V21c controlled-repeat measurement evidence. Each must match the same candidate ID and use disjoint independent cohorts as enforced by its own source tool. Add a separate optional candidate-level human *source provenance* judgement:
- `unknown`: no reviewer assertion, not negative by default;
- `measured_pattern`: reviewer states an observable pattern is present, **NOT** clinical mechanism validation;
- `instrument_artifact`: external evidence/reviewer identifies the candidate source as instrument contamination. Hard veto regardless of all correlation/forecast metrics.

Per-case YES/NO/UNSURE from V16/V21 HTML remains *event-level pattern presence* feedback and is **not** converted into an unconditional candidate-level or clinical certification judgement.

Frozen decisions on six V21c scenarios (original V21 candidate/risk/reference screens preserved):
1. physical predictive: **ALLOW_EXPERIMENTAL_MEASURED_RISK_CANDIDATE**;
2. physical nonpredictive: **ALLOW_EXPERIMENTAL_MEASURED_DESCRIPTIVE_CANDIDATE** with no prediction coefficient promotion;
3. resettable predictive artifact: **ABSTAIN_NO_INDEPENDENT_REFERENCE** (or non-repeatable);
4. resettable irrelevant artifact: abstain;
5. resettable shared-reference artifact: **ABSTAIN_NONREPEATABLE_ACQUISITION**;
6. persistent external shared artifact: **ALLOW_EXPERIMENTAL_MEASURED_RISK_CANDIDATE**, *known false semantic positive* — retained explicitly. If a reviewer independently supplies `instrument_artifact` for this same candidate, policy must BLOCK despite statistical pass.

The first two ALLOW decisions are **exploratory measured** concepts only; no state `PHYSICAL_VALIDATED` exists. If risk screening fails, new source's native risk weight must remain exactly zero. A candidate can be inserted into the original V10 concept graph only with **zero decision / reconstruction / future coefficients** on the new slot initially; all previous patient logits, forecasts and reconstruction must be bitwise or tolerance equal before subsequent permitted native training. The policy may never insert a raw-input predictor bypass. Human artifact veto must not call admission, edit data, or change original model.

## Limits and tests
Test identity matching, missing evidence, out-of-order or unregistered review judgements, veto, source/no-risk distinction, exact native no-drift, audit provenance and explicitly the **persistent artifact false positive**. Report all six scenarios, both approved and refused, rather than optimizing thresholds. External human annotation is optional and is **not** claimed to be obtained or authenticated; this test uses only synthetic provenance labels for its veto counterfactual.

No automatic physical certification, clinician approval, causal identification, hospital deployment or publication superiority. Original PIERE paper remains unchanged. PR #18 stays Draft.
