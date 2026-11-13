# ConceptFAN V57–V58: independent reference, native concept repair and falsifiers

**Execution date:** 13 Nov 2026. **Status:** completed *exploratory, synthetic-only*. Both experimental protocols were committed before **their own** new TEST outcomes. V58 was designed after inspecting V57 TEST, so these are dependent research iterations, not two independent confirmation studies. Previously accepted PIERE/IOI documents and master are unchanged.

## Reproducible source and runs

- V57 frozen design: `docs/medical/CONCEPTFAN_V57_FROZEN_INDEPENDENT_ANCHOR_PROTOCOL_20261113.md`. [GitHub Actions SUCCESS #43466945040](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43466945040); executed source commit `f64ea610e08723ae926ebdbee19e60cfd00c1963`; complete artifact `conceptfan-v57-independent-reference` ID 14059107176 (JSON/CSV for all 168 method/condition metrics).
- V58 frozen design: `docs/medical/CONCEPTFAN_V58_FROZEN_COHERENT_REPAIR_PROTOCOL_20261113.md`. [GitHub Actions SUCCESS #43468160798](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43468160798); executed source commit `088c5d088498de735e5536dd2d8d90efb0ce5b0c`; complete artifact `conceptfan-v58-coherence-reference-repair` ID 14058118856 (JSON/CSV for 240 method/condition metrics). **22 V58/V57/V56 regression tests passed**, smoke and full new-data trials PASS.
- New non-reused random-draw seeds: V57 20261471 / 20261473; V58 20261491 / 20261493; all are different draws of the **same** V30 extended nonlinear synthetic generator. Each full draw TRAIN=1400, VALID=600 unused, TEST=700, 12 original measurement channels, 16 history steps. The same TEST subjects are re-evaluated across conditions, not independent cohorts.

## Exact scientific contract

1. V56 original 12-channel auto source router, native diagonal recurrent fuzzy concepts and concept-only additive risk coefficients are unchanged; zero oracle/human target supervision during concept formation. Original model never receives the independent reference as a direct feature.
2. The **simulator** constructs a new separate measured reference by reading its latent physical source and adding independently generated Gaussian noise sigma=0.27 and independent 20% missingness. The **fitted model and independent source matcher see only this noisy observable, NEVER the simulator's latent trajectory**. Independent, correctly calibrated provenance is therefore an *extra externally supplied physical assumption*, not autonomously inferred semantics.
3. TRAIN-only three-fold stable measured correlation matches that extra reference to one pre-existing unnamed group. V57 selected measured sensors [4,5,6] in both draws (minimum fold |r| 0.95895 and 0.96174). Reference-source affine alignment and patient-level median residual threshold use TRAIN *observations*, not outcomes or TEST. V57: discordant or unavailable reference => source contribution neutral zero, all other sources unchanged.
4. V58 additionally fits an internal sensor-consistency cutoff at the TRAIN 99th percentile (0.24407 and 0.25315 across draws); reference mapping stays train-only. On discordant **internally coherent** triplet it replaces source-level inputs by time-aligned trusted-reference measurements, passes the replacement through the *unchanged frozen native fuzzy encoder*, and keeps all supervised beta, bias and non-target concept contributions. On incoherent sensor triplet it retains V56 robust median rather than deleting the whole concept; on unavailable reference it returns the original model with explicit **UNVERIFIED** status; on fully missing source and available anchor it can recover source input from the independent sensor. Output still equals bias plus exact signed concept contributions, error <=1e-12.
5. Neither V57 nor V58 can infer physiological meaning of any concept, verify calibration of the external reference automatically, or distinguish all shared-fault patterns. A predictive measurement-artifact concept [9,10] remains semantically ambiguous; its outcome association is not a diagnosis.

## V57 verified outcomes and negative observations

PHYSICAL_EXTRA outcome (AUPRC):

| New V57 draw | V56 common-mode flip | V57 reference zeroing | Fault flag rate |
|---|---:|---:|---:|
| 20261471 | .5050 | .7187 | 99.71% |
| 20261473 | .5745 | .7521 | 99.57% |

V57 common-mode correction improved NLL by +0.17821 and +0.15968 per case, respectively (positive = lower NLL), while correct-source interpretation still requires synthetic reference provenance. Negative controls:
- SINGLE_SENSOR flip: V57 NLL **worse** than V56 by .01555 / .01237, due to unnecessary source deletion when V54 robust median already tolerates one faulty sensor.
- ANCHOR_ONLY flip: V57 NLL **worse** by .07815 / .06847; unrelated reference corruption falsely disables a healthy concept.
- ANCHOR_MISSING: V57 NLL **worse** by .07842 / .06905.
- JOINT source/reference sign flip: detected at 0.71% / 0.43%, essentially indistinguishable from normal residuals. No claim of general protection.

## V58 new-draw result: better when reference genuinely independent

PHYSICAL_EXTRA synthetic outcome AUPRC; original sensor data and reference masks are fixed for all methods:

| Draw / fault | V56 no reference | V57 zero-group guard | V58 conditional source repair | Raw logistic 12 channels | Raw logistic + reference |
|---|---:|---:|---:|---:|---:|
| 20261491 / intact MCAR20 | .8283 | .8295 | .8289 | .8233 | .8247 |
| 20261493 / intact MCAR20 | .7774 | .7821 | .7765 | .7588 | .7584 |
| 20261491 / all 3 source sensors inverted | .5509 | .7432 | **.8128** | .5543 | .6009 |
| 20261493 / all 3 source sensors inverted | .5006 | .6896 | **.7664** | .5016 | .5070 |
| 20261491 / 1 sensor inverted | .8212 | .7885 | **.8212** | .8201 | .8248 |
| 20261493 / 1 sensor inverted | .7738 | .7491 | **.7738** | .7571 | .7488 |
| 20261491 / whole source missing | .7437 | .7437 | **.8240** | .7424 | .7675 |
| 20261493 / whole source missing | .6914 | .6914 | **.7724** | .6814 | .6764 |

When all three original sensors inverted but reference stayed correct, V58 chose reference reconstruction for 98.6% / 99.3% of subjects; NLL gain over unchanged V56 = **+.25464** [descriptive subject-bootstrap 95% CI .21469–.29430] / **+.22778** [.18292–.27699]. On intact observations only ~0.7% were repaired and performance changed marginally. On single-sensor inversion V58 usually retained the majority-median signal, with almost no repair. On 100% missing original group, independent reference replacement restored much of the lost information. This is **external additional information**, not reconstruction from nothing.

## Explicit failure cases — DO NOT OMIT from a paper

| TEST control | V58, draw 20261491 | V58, draw 20261493 | Consequence |
|---|---:|---:|---|
| Two of three source sensors inverted | AUPRC .5943 (V57 .7436) | .5337 (V57 .6884) | Internal disagreement alone cannot tell 1 from 2 corrupted channels; skipping repair preserves wrong median |
| External reference inverted, original group healthy | .5654 (V56 .8283) | .5064 (V56 .7774) | A trusted but faulty reference can inject invalid source state; severe risk |
| Reference and all three original sensors inverted together | .5507 (V56 .5509) | .5009 (V56 .5006) | Shared fault remains undetectable |
| Reference unavailable | exactly original V56 .8283 | exactly original V56 .7774 | Honest UNVERIFIED, not a claim of verification |

NLL on clean draw 20261491 is .5137 for V58 versus .5043 for raw-12 logistic and .5031 for raw-plus-reference logistic: raw predictors can still be better calibrated even if V58 has greater AUPRC in some conditions.

The ARTIFACT_PREDICTIVE target law stays observationally ambiguous. V58 sometimes changes prediction under a reference fault; it does not prove that the source group tied to the target is physiological rather than an instrument recording process.

## Scientific verdict and next actual boundary

**Implemented and reproduced under explicit assumptions:** label-free measured-group association with an independent sensor; protected source-specific additive logit; reconstruction *inside* a native differentiable fuzzy temporal concept encoder; fault-specific ablations with hard negative controls. These are robust engineering controls assembled from established statistical methods, not a unique new concept-discovery theorem.

**Not established:** automatic medical concept identity, independence of real devices, attribution stability across hospitals, universal predictive superiority, true latent-factor completeness, protection from 2/3 compromised source channels or corrupt anchors, cross-generator proof, real patient outcomes. More synthetic anchors could make a benchmark easier but would not satisfy the scientific missing requirement.

**Real next gate:** one dataset with independently documented measurement provenance/calibration, patient-disjoint records, actual paired sensor overlap, recorded instrument faults or verified measurement interventions, and an **independent semantic reference** for at least one concept. Without these sources it is not defensible to claim that automatically generated factors correspond to genuine physical/clinical mechanisms. Keep the research PRs DRAFT and original PIERE/IOI manuscript untouched.
