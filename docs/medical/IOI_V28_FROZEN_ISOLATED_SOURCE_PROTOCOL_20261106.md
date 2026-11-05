# IOI ConceptFAN V28 — frozen isolated-source admission with device-shift falsification

Protocol frozen **before examining V28 data, training or TEST evidence**, 2026-11-06. Git base: experiment/ioi-v27-task-leakage-guard-20261106. This is an exploratory synthetic follow-up, not independent scientific validation: same AR latent generator family as V25–V27, new random episodes and a new intervention on candidate sensor provenance.

## Hypothesis, not an established novelty claim

Joint outcome gradients can put missing task evidence inside previously named concepts (shown by V27). An alternative is to **freeze a previously defined concept path** and add a separately attributed candidate source, only if it demonstrates incremental out-of-sample predictive value across **two known validation environments**. A failure must explicitly give ABSTAIN, not silently relabel a concept. This is a constrained-source/CBM-extension baseline, not guaranteed physical source identification.

The admission gate observes no TEST outcome, no oracle factor in its risk calculation, no external medical expert endorsements, and no future measurements. Prior physical K2 concept decoder and its source checksum are fixed; a candidate uses exclusively the measured time-history of sensors 4 and 5, a predeclared transparent operator, not a learned arbitrary raw-to-risk bypass. Risk logit must remain exactly b + sum_k w_k q_k over **separately named** concepts.

## Fixed synthetic design

- Generator frozen_world(seed=20261108), 16 historical time steps, 6 measured sensors, three independent AR factors. 1800 TRAIN, 1200 GATE-VALIDATION, 900 TEST; GATE-VALIDATION split into nonoverlapping 600+600 environments. 3 independent neural initializations: 11, 23, 37. Only K2 true synthetic concept trajectories are provided on TRAIN to supervise *stage-one concept formation* (unavailable in ordinary clinical work). No true third factor is used for candidate formation, admission, or risk fitting.
- TRAIN corruption: MCAR20, seed 20264101. GATE validation: MCAR20, seeds 20264102 and 20264103. TEST: complete, MCAR20, MCAR40, MNAR and contiguous block8 using the exact V25 mask implementations, seeds 20264900 plus ordinal in its original STRESS tuple.
- Stage 1: native K2 canonical MultiSetAdditiveTemporalConceptFANModel, only concept-trajectory MSE (coefficient 5), AdamW LR .002, 14 epochs, batch128; same schedule and seeds as V27 STAGED, frozen after TRAIN. Model is trained **once per NN seed and shared across all world interpretations**, so no outcome gradient can alter K2.
- Stage 2: same TRAIN labels (1800) fit (A) linear additive frozen K2 baseline and (B) linear additive K2 + explicit measured candidate. Both use exact StandardScaler+LogisticRegression(C=1.0). The third concept has fixed signed measured operator c_new = 0.5 + fraction_available * (sigmoid(mean of *available* sensor 4–5 readings) - 0.5). Its provenance is a supplied sensing formula, **not** a new verified physical/medical diagnosis or automatic discovery.
- Candidate is used per patient only if observed sensor-pair coverage is at least 0.50; else fallback to K2 baseline. No source modifies any of the original K2 outputs. Attribution consists of two existing contributions or two plus a distinctly named third contribution. No latent/encoder/raw bypass.
- Gate: fit heads on TRAIN only; compare held-out per-patient NLL (K2 baseline minus K2+candidate) on each of 600 GATE environments, with paired 399-bootstrap 95% lower bound > 0 and observed mean improvement >=0.002 **in both environments**. All thresholds fixed now, never selected with TEST labels. Reject if either validation gate fails. A rejected candidate is recorded, not silently inserted.

## Environments and negative controls

1. **stable_physical**: Y depends on true third factor and its two measurement channels remain stable in TRAIN, both GATE environments and TEST. Candidate may be admitted if positive held-out evidence in both known environments.
2. **nuisance**: same sensors and independent third factor, but Y depends only on first two known concepts. Candidate should normally be rejected; false admissions are measured.
3. **observed_instrument_break**: TRAIN and GATE-env1 sensor pair tracks third predictive factor, GATE-env2 and TEST sensor pair flips sign (explicit device mapping shift) while outcomes remain from original Y. Gate should detect unstable predictive association and reject; if it does not, report failure.
4. **unseen_instrument_break**: TRAIN and both GATE environments are observationally identical to stable_physical, but TEST sensor pair flips sign. An observational gate **cannot know about this future unobserved shift**; a false candidate admission and subsequent TEST degradation is a required negative control if observed.
5. **stable_instrument_interpretation**: every observed feature and label is bit-for-bit identical to stable_physical (only interpretation differs); **reuse all results**. No observational method may treat this as separately identifiable physiological semantics, and no extra independent model fit is counted.

Device flips are controlled synthetic interventions revealing a new observation environment, not causal medical experiments. In controlled real deployment such shifts require external device provenance.

## Evaluation and restraints

- Report admission per neural seed, per-validation-environment NLL improvement and 95% bootstrap, source support rate, immutable K2 parameter digest, effect of adding source on original K2 states, structural additive decomposition error.
- On each TEST missingness scheme and each scenario report AUPRC, AUROC, Brier, NLL, source coverage/mean fallback, baseline K2, forced K3 measured-source, gate-limited source, external raw-history logistic and oracle-true-K2/K3 risk upper controls (privileged evaluation).
- Maintain exact TRAIN/GATE/TEST disjoint identities; gate uses ONLY TRAIN model fits and VALID outcome labels, never TEST outcomes. Different stress masks on the same 900 TEST episodes do not constitute independent cohorts.
- **Important fairness limitation:** candidate admission uses 1200 additional labelled GATE patients; direct baseline does not use these labels to improve its risk fit. They are nonetheless available to all arms for auditing. CPU/FLOPs and label gradients for candidate selection are not matched. Do not claim equal total labels or superiority over published CBM architectures.
- Test hypotheses: stable source may help, nuisance and observed shift may trigger correct refusal, unseen shift may deceive gate, source isolation prevents third-factor information from silently contaminating the original named K2 path. All failures are reportable.
- A rejected source does not prove the initial dictionary is complete. An accepted source is only predictive under the **specified validated environments**, not physically causal or clinically identified.
- V25–V27 and accepted IEEE PIERE remain unchanged. No PhysioNet/MIMIC or clinical labelled TEST.
