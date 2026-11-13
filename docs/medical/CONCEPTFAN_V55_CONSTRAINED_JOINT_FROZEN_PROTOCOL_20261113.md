# ConceptFAN V55: constrained joint new-concept definition plus risk, frozen protocol

**Pre-TEST design, 13 November 2026**, branch `experiment/conceptfan-v55-joint-operator-20261113`.

## Goal
V53 created only signed delayed statistical concepts and trained a scalar risk coefficient. V54 added nonlinear signed/energy operator library but froze the definition before training the risk coefficient. V55 jointly optimizes a **small, measurable, nonnegative-weighted activation** and its **signed predictive coefficient** from TRAIN Y, while preserving the newly discovered sensor/lag/operator definition as a constrained passport, a strict K2-only decision path on rejection, and independent dual-VALID admission.

## Identifiability boundaries
- Inferred sensor lag and EVEN_ENERGY / SIGNED_LAG family remain **statistical observation operators**, not physical or causal identities. A device artifact can remain predictive and be admitted. No external attested reference exists on this generator.
- No free latent vector or raw-to-risk bypass is introduced. A new concept is the last-six-step observed support-shrunk mean of a sigmoid of a convex mixture of two **registered and normalized masked sensor operators**.
- The only jointly trained parameters are two nonnegative mixture weights summing to one (softmax), bounded temperature in [.5, 2], and scalar signed gamma clipped to [-10,10]. No trainable choice of sensor, lag, operator family, hidden variables or new concept cardinality after proposal.
- The reference V54 q and native K2 are frozen. TRAIN loss is BCE of `frozen_K2_logits + gamma*(joint_q-.5)` plus 0.10×mean-square(q_joint−q_V54), 0.01×squared weight deviations, 0.01×squared log-temperature, and 0.0005×gamma². Exactly 80 full-batch Adam steps lr=.03 for each independent candidate; no TEST influence.
- V55 uses the real native signed concept head at inference after validating numerical agreement with the constrained torch operator. Unavailable or rejected source must return **bitwise exact** K2.
- The same two disjoint VALID400 cohorts from a *new random cohort generated with seed 20265505* admit at most one source by the unmodified V28 dual-environment 99% bootstrap policy. No source selection or refitting on TEST. Report all cases including abstentions and negative results.

## Predeclared experiment
- Source discovery via frozen V54 `SIGNED_LAG`/`EVEN_ENERGY_LAG` operator library, TRAIN950 with 20% missing; no Y in proposal.
- Independent (from V54 outcome rows) synthetic seed 20265505, 16 time steps, 12 channels, three label modes `nonlinear_energy`, `irrelevant_delays`, `predictive_recording_artifact`, native random seeds 11 and 23; VALID1=400, VALID2=400, TEST=650; shared TEST subjects across seeds in each mode.
- Compare K2, frozen V54 coefficient-only, constrained joint V55, signed-only V53, raw logistic and *energy-engineered* logistic with equal outcome-label budgets.
- TEST stresses: complete, MCAR20, MCAR40, post-admission sensor sign inversion, and 1.5× gain change. Sign flipping should preserve even operator values algebraically but physical invariance is **not** inferred. New cohort is still one synthetic generator family.
- Save 30 TEST per-patient prediction arrays, chosen source passports, TRAIN optimization trace, two-VALID admission receipts. Independent audit recomputes AUPRC/NLL, signed contribution, exact fallback and posthoc paired-subject bootstrap. No publication claim before CI reproduces outcomes.

## Stop criteria and negative reporting
- If V55 loses to V54 or energy-engineered logistic on new TEST, publish this as an architecture tradeoff. Do not alter fixed training objective, optimizer, admission thresholds or random seed based on those TEST outcomes.
- No clinical physical source validity or certified causal concepts. A concept whose statistical usefulness passes VALID still must not be named physiological without an external independent measurement source.
- New concept is only constrained-jointly calibrated *within a small finite operator family*, not universal program synthesis or arbitrary concept induction.
