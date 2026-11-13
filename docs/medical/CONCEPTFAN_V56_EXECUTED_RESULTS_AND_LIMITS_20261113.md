# ConceptFAN V56 — executed independent-reference gate and cohort scope, 13 November 2026

**Status:** EXECUTED EXPLORATORY on new synthetic cohort; clinical/physical validation **not** established. [Draft PR #59](https://github.com/fims9000/FuzzyAttentionNetworks/pull/59). Frozen registered protocol: [V56](./CONCEPTFAN_V56_EXTERNAL_REFERENCE_FROZEN_PROTOCOL_20261113.md). Successful full [GitHub Actions run 43472886014](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43472886014), source commit `ba9f88ecaf0e82daf8025d1a894c5c4969e32915`. Artifact `conceptfan-v56-independent-assay-and-revocation` ID `14062076969`, digest `sha256:cd3e0d80a4cfe1a9f5a29575397ac5381272feac49319d98a15c8a2478c44bd4`.

## What is implemented, not merely proposed
Original native `TemporalConceptFANModel` K2 and constrained-joint V55 candidate q/gamma stay unchanged. A second admission gate requires a separately simulated noisy independent source-energy reference (NOT a copy of the measured input pair): buy only **190 TRAIN plus 100+100 disjoint VALID** assays, fit q→reference on purchased TRAIN, require two separate VALID groups to satisfy positive slope, Pearson correlation >= .55 and >=10% MAE improvement over TRAIN-constant. Candidate also independently must pass dual-VALID predictive-risk evidence. No TEST assay ever provided to model. An unlabeled TRAIN-frozen q-mean source scope monitor can revoke a cohort, while missing or mismatched declared device/reference and missing sensors give bitwise exact K2. This is conditional **assay-correlated statistical proxy**, not physical causality or cryptographic device attestation.

The executed comparison used four registered source/reference modes, 950 TRAIN / 400 VALID1 / 400 VALID2 / 650 shared TEST episodes, two native initializations (11,23) × six stresses = **48 individual TEST arrays**, all independently recomputed in `verify_v56_audit.json`. Two seeds share the *same* 650 TEST individuals in each mode.

### Held-out MCAR20: mean AUPRC over two native seeds

| Mode | Frozen K2 | Original V55 (predictive only) | V56 (predictive + assay) | Engineered-energy logistic |
|---|---:|---:|---:|---:|
| Useful target energy | **.6186** | **.8178** | **.8178** | **.8477** |
| Outcome-irrelevant delayed sources | .7266 | .7266 | .7266 | .6813 |
| Predictive **recording artifact** | .6278 | .7716 | **.6278** (refused) | .8072 |
| **Permuted independent assay**; identical X,Y | .6186 | .8178 | **.6186** (refused) | .8477 |

In the useful mode, both V55 and V56 NLL are **.6133** versus engineered logistic **.5073**; V56 retains 84.6%/88.2% episode-level source support in the two seeds at MCAR20. Both independent VALID assay correlations for selected source were approx **.696/.733**. In artifact mode, the unrestricted V55 chose 8/9 and V56 did not admit it; in useless mode neither admitted a predictive new source. The broken-assay control used *identical X and Y and identical V55 outputs*, with only external reference pairing shuffled; V56 correctly refused.

Paired posthoc patient-bootstrap 95% **descriptive** bands for MCAR20:
- Useful V56-K2 ΔAUPRC ≈+.1992, interval **[+.1507,+.2462]**.
- Useful V56−engineered logistic interval **[−.0604,−.0042]** — cannot claim predictive superiority.
- Artifact V56−V55 interval **[−.1981,−.0980]** — explicit **price of measurement-semantic refusal**, not a hidden success on prediction.
- Permuted-assay V56−V55 interval **[−.2461,−.1493]** — reference mismatch causes a measured refusal.

### Stresses
- Under TEST-only sign inversion of both selected even-energy sensors, the measured source is unchanged algebraically in this generator.
- At unseen 1.5× sensor gain, the frozen cohort shift check **revoked** the optional source for both useful-source seeds; V56 returned K2 (AUPRC .6186) while unrestricted V55 had AUPRC .7664. Revocation preserves the declared scope, not predictive performance.
- Under complete selected-source missingness, source is unavailable and exact native K2 is returned. At MCAR40, supported source episodes fall to approximately 37%, but the cohort monitor still admitted the source for supported patients.

## Limits and next architecture
- The 'external assay' is simulated from latent source energy with **independent additive noise**; no real-world metrology or physiological ground truth is provided. String IDs may be spoofed. Observationally equivalent worlds and artifact–assay confounding remain unresolved.
- A mean-q cohort monitor cannot detect transformations preserving the q distribution, and requires a *cohort*, not isolated online patients.
- V56's external gate certifies predictive **corroboration only**; its q remains an arbitrary statistical activation, not an energy measurement. The next version must calibrate q numerically from the assay and **retrain the native signed gamma on that exact new q**, not substitute source scaling into V55 risk weights. Also compare against an engineered model given the same assay-derived q, on an entirely fresh TEST cohort.
- All heldout intervals descriptive, conditional on generator/hyperparameters and not corrected for multiplicity. The assay budget adds 390 additional training/validation measurements. No generalization, medical or SOTA claim.

**Do not merge into master or rewrite previously submitted papers based on these synthetic results.**
