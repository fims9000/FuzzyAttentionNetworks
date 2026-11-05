# IOI ConceptFAN V30 — frozen nonlinear regime-switching generator + independent measurement anchor

Frozen 2026-11-06 **before V30 generator results or any held-out TEST measurements/outcomes are inspected**. Branch: experiment/ioi-v30-trusted-anchor-repair-20261106, base V29. This is **one additional simulation family**, not an externally validated biomedical source or a novel statistical identification theorem. PIERE camera-ready and sealed real-patient cohorts are not touched.

## Scientific question

V27 showed outcome gradients can contaminate named concepts with omitted predictive information. V28 isolated the additional source but a new unseen sensor-pair inversion was admitted. V29 corroborated a pair by its own internal agreement but couldn't detect a shared common-mode reversal and sometimes rejected a biased but still useful readout. V30 asks: with an **explicit external, independently calibrated measurement of the same source**, can a frozen-concept predictor detect and sometimes repair those failures without touching named concepts, while exposing the consequences of the reference being wrong?

## A genuinely different simulator family

Do not use previous Gaussian AR(1) generator. Generate a **three-source nonlinear, sparse-shock, switching-regime dynamical model** over 16 observations. For each source the state follows a driven nonlinear tanh dynamic with per-sequence persistent regimes, sparse Laplace shocks and time-dependent switching impulses; all noise random streams and states are independent across three physical/source coordinates. Three true concept trajectories are sigmoid(1.2*state). The **binary outcome is sampled from an additive** 3-concept *mean-state* logit, so the additive risk readout is a fair model family, not intrinsically misspecified. COMPLETE/NUISANCE labels depend only on first two true sources, PREDICTIVE includes the third; observationally identical instrumental interpretation must not be counted as additional independent data.

Generate exactly 7 sensors (two noisy readouts each for source1/source2/source3 + **one separate externally calibrated anchor sensor6** for third source). Three source readings 4,5,6 have independent sensor-specific errors **by controlled construction**. This is an assumption requiring real device provenance, not derivable from correlations.

Frozen seed 20261110: TRAIN=1800, two labelled VALID environments=600+600, sealed TEST=900 **independent episodes**. Fixed source mask generator: TRAIN MCAR20 seed20266101, VALID MCAR20 seeds20266102/20266103, TEST MCAR20, MCAR40, MNAR and block8 seeds20266900+ordinal in ("complete","MCAR20","MCAR40","MNAR","block8"). TEST scenario manipulations are fixed BEFORE missingness mask generation, with same patient IDs and outcomes; mask mechanism never uses outcomes. Three neural seeds: 11,23,37.

## Architecture, source mapping and training

- Actual canonical native K2 ConceptFAN with identical V27 model dimensions except input_dim=8 for **only four input channels 0:4 and their availability bits**. The K2 concept encoder **cannot even access** source channel4/5 or independent anchor6 (enforced by input interface tests). Synthetic oracle trajectories for K2 supervise its 14 epoch label-free stage on TRAIN only; stage1 model is shared across predictive/nuisance worlds and all fault conditions, checksum frozen. Risk K2 head is a standard TRAIN-only logistic on the learned two concept summaries, frozen after fitting.
- Predeclared candidate measured source 4+5: V28 operator 0.5+fraction_available*(sigmoid(mean of observed historical pair)-0.5), fitted source gamma on labelled TRAIN only and *independently* admitted by V28 paired 399-bootstrap NLL gate on BOTH labelled VALID600+600 environments (mean NLL improvement >=.002, CI lower>0, per-subject support>=.5). Source q3 is separately named. If gate rejects, all variants revert exactly to K2 logit.
- V29 pair-only monitor trained on **TRAIN observed values without labels** (>=5 paired points, 1%-corr and 99%-gap thresholds), with no adaptation from VALID or TEST.
- New independent anchor readout sensor6 uses **only its observed past values**. Compute anchor activation 0.5 + measured_fraction*(sigmoid(anchor observed mean)-0.5). Train an *outcome-blind* anchor-vs-pair discrepancy threshold on TRAIN measured history: 99th percentile of abs(pair sigmoid mean minus anchor sigmoid mean), only when both have support>=.5. Do not choose threshold by VALID or TEST risk outcomes.
- Two additional test-time policies, both after same labelled risk gate: (A) **anchor_reject**: use pair candidate only if pair internally credible AND discrepancy to anchor <= TRAIN threshold; else use exact K2 frozen logit. (B) **anchor_repair**: when pair is not internally credible OR pair-to-anchor disagreement exceeds threshold, substitute the **anchor-derived** third-source activation and retain the separately fitted gamma if anchor coverage>=.5; otherwise fall back to K2. When pair is credible and anchor agrees, retain pair activation. Store explicit choice reason ("pair"/"repaired_from_anchor"/"abstain"). Anchor repair assumes anchor is **independently calibrated/trustworthy**; it is falsified by an anchor-only corruption. It is NOT a valid general repair from observational agreement alone.

Test methods: frozen_K2, V28_pair_candidate, V29_pair_reject, V30_anchor_reject, V30_anchor_repair, measured-raw-history logistic using all available seven sensors (TRAIN only), oracle true-K2 and oracle true-K3 logistic ceilings (TEST true concepts only for audit). Evaluate full-cohort AUPRC, AUROC, NLL, Brier, source usage, false rejection/false repair rates, native exact additive logit, source coefficients and K2 source/state hashes.

## Frozen scenario interventions, all on same 900 TEST episodes

1. stable: all source sensors calibrated, genuine extra prognostic third factor;
2. single_pair_sign_flip: **only sensor5** reversed at TEST (other two sensors 4,6 stable);
3. single_pair_bias: only sensor5 +2 offset at TEST;
4. both_pair_sign_flip: sensors4+5 jointly reversed at TEST, independently calibrated anchor6 stable;
5. anchor_only_sign_flip: ONLY anchor6 reversed at TEST, pair4+5 clean — **required harmful-repair negative control**;
6. all_three_sign_flip: pair4+5 AND independent anchor6 simultaneously reversed at TEST — **required failure of any pair/anchor agreement-only checker**;
7. nuisance: all sensors stable but source3 not outcome-predictive; global source gate should reject;
8. stable_instrument_interpretation: identical observations/labels/models to stable, **reuse data**; source identity remains observationally nonidentifiable under the original labelling semantics.

Pair/anchor consistency cannot prove clinical physiological identity. Shared common-mode orientation changes require **external calibration provenance / intervening physical reference**, not merely a third perfectly correlated readout.

## Criteria and honest limitations

- A credible distinct-generator positive result would improve TEST risk under stable and single/pair fault, with explicit normal false refusals and anchor-only failure. **No outcome may be dropped**; include all 900 cases even if candidate not used.
- The anchor may have more missingness/noise than the pair and the repaired q3 may have different calibration. Do not silently retrain gamma on TEST or rescale anchor against labels after inspection. Repairs may hurt even if the anchor itself is truthful.
- Source gate uses 1200 additional VALID labels; risk head candidate gamma uses TRAIN labels, K2 concept supervision uses synthetic true concepts unavailable clinically. No equal-total-label or compute-matched superiority to published CBM/IB-CBM/Res-SCBM.
- Train/VALID/TEST independent episodes but all missingness modes and sensor failures reuse the same TEST subjects. Three NN fits are correlated by generator; do not claim independent hospital cohorts or causal discovery.
- This adds a **new nonlinear generator family** and a strong held-out failure control but remains synthetic, with no external clinical ground truth and no novel source-discovery theorem. Stop if any claimed empirical result fails checks; preserve negative runs.
