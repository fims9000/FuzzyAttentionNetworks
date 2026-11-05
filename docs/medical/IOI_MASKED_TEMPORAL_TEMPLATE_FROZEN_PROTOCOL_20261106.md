# Frozen research protocol — masked low-rank temporal source evidence (2026-11-06)

**Precommit before any new evaluation:** parent [V31 executed pulse-order comparison](../medical/IOI_V31_EXECUTED_TEMPORAL_SOURCE_RESULTS_20261106.md); new branch \`experiment/ioi-mask-aware-source-template-20261106\`. Research question is NEW: when long gaps prevent both half-window means, can a source's TRAIN-learned temporal template estimate one separately named risk feature from the actually measured sections, with a conservative uncertainty/support check, without changing known K2 concepts or using outcome labels in temporal-template learning?

This is a **conventional rank-1 masked-PCA / weighted least-squares source-template estimator**, not a novel theorem or new physical source identification algorithm. Any gain is conditional on TRAIN/TEST signal-shape stationarity. Source-template accuracy and semantic meaning remain distinct.

## Prespecified generator/data

Reuse V31's **non-autoregressive two-phase pulse-order synthetic family** on a fresh, previously unseen draw: **seed=20261118**, TRAIN1500, disjoint VALID1 500/VALID2 500, TEST800, 16 steps, 12 channels. Three native K2 neural initializations **11,23,37**, 14 AdamW epochs. Concept encoder only channels0:3 + masks, trained on measured known-pair proxy trajectories from TRAIN, no oracle latent targets or Y gradients. Original K2 risk b,w1,w2 frozen after TRAIN fitting. True source amplitudes and physical-vs-instrument artifact status are GENERATOR provenance for posthoc audits only.

TRAIN MCAR20 mask seed20270301; VALID1 MCAR20 seed20270302, VALID2 MCAR20 seed20270303; TEST 5 original \`complete/MCAR20/MCAR40/MNAR/block8\` modes at seeds20270900+registered ordinal, **same 800 TEST subjects** and outcomes across all masks. True data generations and all risk decisions independent across TRAIN/VALID/TEST indices. None of the methods may inspect TEST Y, source identity metadata or latent source amplitudes for training, candidate grouping or thresholds.

## TRAIN-only source-group identification

Reuse unchanged PR26 correlation-complete-clique proposer on unknown sensors4:11, minimum three disjoint TRAIN folds, |ρ|>=.60 on jointly measured histories, min300 co-observed cells/fold, at most5 groups. No labels or true hidden sensor groups. True known group mapping sensors[0,1] and[2,3] supplied externally, not discovered.

For EACH proposed group, compare exactly:
1. \`static_mean\`: PR26 mean whole history;
2. \`half_contrast\`: V31 half-early-minus-late (requires coverage in BOTH halves);
3. **\`masked_rank1_template\`:** source temporal template fitted unsupervised on TRAIN observed masked sensor means, standardized and signed using TRAIN source group measurements ONLY. First fit rank1 SVD to zero-filled, per-time co-observed group averages; then 5 alternating weighted rank-one least squares iterations on ACTUALLY OBSERVED source-time cells. Unit RMS (sum φ² = T) and deterministic sign orientation (first four template time positions have positive mean), independent of Y and oracle amplitude. Store template φ(t), TRAIN residual variance and frozen SHA. For TEST episode: project ONLY observed group time positions with WLS onto φ; use shrinkage based on measured Fisher information \`sum_t observed_t φ²(t)\` and TRAIN-estimated noise/signal variance; nonobserved time points never contribute to numerator or denom. Candidate bounded value \`q=0.5+reliability*(sigmoid(a_hat)-.5)\`. Report approximate uncertainty, effective information, and source availability. Require effective information relative to full 16-step template >=.50; otherwise return q=.5/support=0 and exact K2 fallback.
- Candidate template forms and group sensor signs MUST NOT depend on target labels, TEST, reference latent amplitudes, VALID2 regimes, or any declared outcome world.
- All three variants use identical source group, TRAIN1500 target-label risk gamma optimizer, and same two disjoint VALID500+500 bootstrap gate. Max 15 candidates across 5 source groups, no exact FWER guarantee from the 99% lower percentile.
- All source evidence contributes only via one new named additive signed risk term \`gamma*(qnew−.5)\`; exact original K2 contributions and parameter SHA unchanged.

## Shift/semantic controls

Six prespecified scenarios:
A. Healthy physical source; VALID2 benign source+unknown-channel gain 1.10, bias +.05; TEST healthy.
B. PHYSICAL source pair 4/5/6 sign-reversed in VALID2+TEST; gate should reject or report if accepts.
C. Recording artifact group9/10 sign-reversed in VALID2+TEST; gate should reject or report if accepts. Predictive recording process is NOT a medical concept.
D. PHYSICAL source sign reversed ONLY on TEST, benign VALID2; expected adverse if accepted.
E. COMPLETE dictionary, extra pulse sources irrelevant to Y; false admissions measured.
F. **Unseen pulse SHAPE drift only on TEST:** physical pulse changepoint shifted from step8 to step3 (original amplitude and outcomes unchanged) and sensor channels4/5/6 remapped to this changed underlying waveform while original hidden source reference remains the TRAIN task. Not a sensor-sign inversion, but a different temporal pattern. A learned fixed template may be biased; record any negative results. Noise terms are preserved where possible: \`x' = x + a*(new_template − old_template)\` uses hidden amplitude ONLY in the synthetic TEST corruption generator to define ground truth, NOT in predictor/evaluator feature formation. Do not treat this as an independently sampled TEST cohort.

## Baselines and uncertainty

On all cases/missingness modes, full heldout TEST800 AUPRC, AUROC, NLL, Brier, fraction source permitted, TEST raw sensor coverage and WLS posterior SE (both counts). Compare:
- Native K2 frozen risk;
- auto-static with same 2-VALID source gate;
- auto-halfcontrast with same source gate;
- auto-masked-rank1-template with SAME source selection and gate;
- 3-statistic automatic bank (at most one source) to expose selection budget;
- Standard ordinary additive CBM head refit on all candidate statistics with TRAIN1500; and CBM with all 2500 TRAIN+VALID labels;
- Strong raw temporal logistic using early, late, full means/masks TRAIN1500; and same raw temporal logistic fit to all 2500 TRAIN+VALID labels (VALID2 has case-specific transformed readings);
- Oracle true K2 and true predictive source amplitude, synthetic-only privileged performance references.

Prespecify source gate: candidate gamma TRAIN labels only, VALID1+VALID2 each 500 episodes, mean paired NLL gain≥.003 each, empirical paired bootstrap lower 1% quantile>0 with 399 bootstrap draws, source eligible fraction≥.75 in each VALID; test usage requires support≥.50. Apply same to all candidate types, no TEST label or choice. Report source selection/failed gates for every group and method, 399 paired TEST subject bootstrap 95% CI for TEMPLATE vs HALF and vs STATIC in healthy and shape-drift / unseen-source-inversion scenarios. CONDITIONED on three fixed NN initializations and the same TEST, not generalization across generators/hospitals.

**No label/compute fairness overclaim:** all methods are offered 2500 TOTAL labelled episodes, but our source gate uses 1000 VALID outcomes to select, whereas CBM/raw models fit on them; compute/optimizer budget not aligned. The temporal template is trained from *unlabelled* TRAIN measurements which all baselines also have, but not necessarily use.

## Required falsification

- If template shows no advantage on block8/missingness, report it. Never tune acceptance/uncertainty thresholds on TEST to improve coverage.
- If template is worse than simple halfcontrast on clean MCAR20, preserve it; expected tradeoff.
- If a source shape changes after VALID, template projection may be biased and lead to harmful risk; preserve this failure.
- Any correlation group aligned with a recording artifact remains an instrument group even if target-predictive.
- If all-group CBM/raw models still win at 2500 labels, report it.
- The masked rank-one template is known classical statistical machinery. Do NOT claim new source discovery, causal semantics, original CIBM/Res-SCBM replication, clinical transfer, or independent human concept annotations.

No changes to original accepted PIERE manuscript, IITI work or earlier GitHub frozen evidence. Archive full code SHA, masks, proposal templates, original/negative trials, candidate decisions and TEST probability receipts.
