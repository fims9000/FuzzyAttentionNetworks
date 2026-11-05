# IOI ConceptFAN V29 — frozen two-sensor consistency guard vs common-mode failure

**Frozen before V29 TEST/fit:** 2026-11-06. Base: experiment/ioi-v28-isolated-candidate-admission-20261106. V28's accepted-source risks/negative controls were ALREADY inspected; V29 uses a NEW random sample **seed 20261109** of the same familiar three-factor AR generator, not new independent underlying dynamics or clinical validation. The intervention types and thresholds below must not be tuned after TEST.

## Question

V28 can reject a known sensor mapping shift using labelled validation environments, but admits a harmful new source when the mapping changes only at an unknown TEST environment. Can **two-channel measurement consistency** (same underlying synthetic factor, independent noise) prevent some future sensor failures without using TEST outcome labels? What failures remain fundamentally invisible to a pair-only monitor?

## Source, sensor model and causal limits

Two measured readouts x4 and x5 are noisy copies of the same latent AR source. Their independent instrument noise is assumed BY THE SYNTHETIC GENERATOR, not externally established for any actual medical device. The source was previously admitted from two unchanged labelled VALID environments under V28's exact gate; unchanged original K2 concept states and K2 risk coefficients are frozen. The source coefficient gamma is TRAIN-only and has a separate signed additive contribution. No raw/latent path to risk outside named concepts.

**Unlabelled sensor evidence guard**: for every episode compute the Pearson correlation between x4 and x5 and their mean absolute paired difference over *jointly observed* historical timesteps. Require at least 5 such timesteps. Choose lower 1st-percentile TRAIN correlation threshold (with lower bound 0), and upper 99th-percentile TRAIN mean absolute difference threshold (no TRAIN outcome labels, no VALID or TEST tuning). After the V28 global qualified risk gate and normal source-availability check, use candidate only if all these per-episode checks pass; otherwise fall back exactly to K2 frozen risk. One sensor failing the agreement test is a measurement integrity reason to withhold a source **not** a statement of clinical semantic meaning.

## Fixed experiment

- generator seed 20261109, n TRAIN 1800, two independent labelled VALID environments each 600 (1200 total), TEST900; exactly 16 history steps, 6 sensors, known original K2 concept oracle used on TRAIN **only** for staged representation training. Three NN seeds 11,23,37. AdamW .002, 14 epochs, 128 batches, concept-only MSE with no Y gradient to K2. The **same** trained original K2 source module is reused for all reported worlds and label laws.
- TRAIN MCAR20 seed20265101; VALID MCAR20 seeds20265102 and20265103. TEST modes MCAR20/MCAR40/MNAR/block8 using existing mask generator with seeds20265900 + V25 STRESS index; same independent TEST patients across masks, not independent cohorts. In MNAR, missingness is recalculated conditional on the actual shifted readings.
- Compare four source/outcome situations evaluated on disjoint TEST:
  1. STABLE: true third prognostic factor, both sensor channels calibrated.
  2. FLIP_ONE: at TEST only, sign of channel5 reverses; cross-sensor correlation should flag unsupported source.
  3. BIAS_ONE: at TEST only, channel5 gains +2 constant offset; correlation alone cannot detect additive bias, but paired absolute discrepancy should.
  4. FLIP_BOTH: at TEST only, both channels reverse sign, preserving their pairwise correlation and difference distributions when source symmetric; pair-only integrity gate **cannot infer absolute orientation**, and may still be harmful.
  5. NUISANCE: third source is measured but unrelated to Y; labelled V28 gate should already reject.
  6. STABLE_INSTRUMENT_INTERPRETATION: identical observations and outcomes to STABLE; reuse as a semantic indistinguishability negative control, not new fit or test sample.
- For every case report: TRAIN-only sensor coherence threshold and usable fraction, original K2 source hash, V28 global gate admission from two VALID environments, fraction of TEST episodes passing source reliability vs pair agreement, full cohort AUPRC/AUROC/NLL/Brier for frozen K2, V28 source gate, combined V29 pair consistency gate, raw input logistic and true synthetic oracle K2/K3 reference. Exact additive logit/unchanged frozen baseline check.
- Hold all sources to the same 1800 labelled TRAIN cases. The risk qualification gate uses 1200 additional labelled VALID patients; sensor monitor uses only TRAIN unlabeled measurement differences/correlations, not TEST outcomes. Risk-head training optimizers and candidate evidence are **not compute matched to published IB-CBM/Res-SCBM**.
- Positive case should not degrade normal predictive quality excessively; monitor must report NORMAL false refusals and any loss. Require the FLIP_BOTH negative failure to be visible, not censored, even if pair consistency passes.

## Hypothesis / failure criteria

Pairwise redundancy may detect certain individual-channel corruptions without outcome labels, but it **does not certify true physical source identity, clinical safety or robustness to correlated instrument failures**. A simultaneous sensor transformation that preserves joint agreement can deceive the monitor. If a single-channel shift slips through or normal data are frequently rejected, report the failure; never alter thresholds on TEST to hide it.

Reference and priority: existing Koh 2020, Havasi 2022, Parisini JMLR 2026 leakage, Galliamov ICLR 2026 information bottleneck and Marcou 2026 residual CBM preprint make generic "new leakage detection" claims unjustified. Treat V29 as measured-source provenance stress testing, not a published architecture. PIERE 2026 camera-ready and PhysioNet sealed TEST untouched.
