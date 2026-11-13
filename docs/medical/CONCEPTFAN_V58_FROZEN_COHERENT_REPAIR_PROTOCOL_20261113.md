# ConceptFAN V58 — frozen coherent reference-based concept repair

Date: 13 November 2026. Design registered **after inspecting V57 TEST outcomes**, but **before** V58 new synthetic TEST results. Exploratory, not confirmatory, no real patient data and no clinical or source-causal identity proof.

## Observation motivating the experiment

V57 successfully used an independently measured physical anchor to flag ~99.6–99.7% of synthetic complete source-triplet inversions, improving AUPRC after joint source malfunction. But simply ZEROING the entire suspect concept has two demonstrated costs: it harms the prediction under a single faulty channel even though the existing 3-channel V54 median is robust, and it harms healthy predictions when the reference itself fails or is absent. A jointly corrupted reference and source remain observationally plausible. These are V57 negative results and must not be erased.

## Hypothesis and frozen mechanism

Instead of deleting the concept in all disagreement cases, distinguish two *observationally different* disagreement patterns and use the independent sensor to reconstruct an ordinary V56 fuzzy source state, **without retraining task coefficients**:

1. All measured-group source definitions, unlabeled V56 neural concept encoder and independent V57 reference mapping are fit from TRAIN measured data only; risk head uses TRAIN outcomes only after concept fitting. Do not expose reference to the original 12-channel auto-group selector.
2. TRAIN-only source *internal coherence* is scored by the existing V53 sensor disagreement operator on >=2 jointly measured channels, source episode score = mean across eligible times of signed standardized sensor disagreement. Freeze 99th TRAIN subject quantile of score for the reference-matched group. This is independent of Y. Needs >=100 TRAIN reference group subjects with >=3 concurrent pair-observed times.
3. On a TEST episode, when reference and measured group DISAGREE (V57 frozen threshold) but **the group is internally coherent** (coherence score <= its TRAIN-only 99% bound and >=3 co-observed timepoints), label a common-mode group fault *candidate*. **Do not claim it is definitely a source fault:** a broken external anchor gives the same pattern. On this candidate, replace only the matched group's measured *time-local source input* by the TRAIN-fitted affine reference projection at available reference timestamps. Feed repaired source history through the ORIGINAL FROZEN native recurrent fuzzy encoder. Do NOT change any other group input, source model parameters, or task-head weights. The anchor enters via an explicitly named source repair, not a new direct risk feature.
4. If the group is internally **incoherent**, trust the original robust V56 group median, with no repair. This targets the V57 single-sensor negative result, because a three-sensor robust median handles one gross faulty channel.
5. If the entire matched group is unavailable and the independent anchor has >=6 observed timepoints, recover its source state from the independently measured anchor; this is an explicit extra-sensor **reconstruction**, not spontaneous inference from absent information.
6. If reference is missing/insufficiently measured, keep the original V56 prediction and emit **UNVERIFIED_NO_REFERENCE**; do not magically certify the measurement or automatically delete useful signal.
7. If no reproducible TRAIN source-anchor map, remain unverified and return unchanged V56. If reference/source are corrupted together, the method may miss failure; if reference alone is wrong, the method may inject an invalid measurement. Record both failures.

Risk always has the exact mathematically transparent form `logit=bias+sum_k beta_k*(mean_t repaired_or_original_c_k-.5)`. No direct raw features or reference go to the risk head. Source calibration and repair selection NEVER inspect TEST outcome or oracle factors.

## Frozen evaluation

Use TWO NEW original V30-extended nonlinear generator draws, seeds **20261491 and 20261493**, each TRAIN1400 / VALID600 unused / TEST700; 12 original measured channels x16 time steps. Additional separately noisy source-3 reference sigma0.27, MCAR20 independent sensor mask; original sensors TRAIN and TEST MCAR20. Independent calibration and source weights computed as V57, network V56 neural seed11 epochs8.

Synthetic outcome laws: PHYSICAL_EXTRA, COMPLETE_NUISANCE, ARTIFACT_PREDICTIVE. Frozen TEST scenarios:
- STABLE reference and original;
- SINGLE_SENSOR_SIGN_FLIP, only original channel 5;
- TWO_SENSOR_SIGN_FLIP, original channels 4 and 5; new *negative control* for median majority;
- SOURCE_COMMON_SIGN_FLIP, original channels 4,5,6;
- ANCHOR_ONLY_SIGN_FLIP;
- JOINT_ANCHOR_AND_SOURCE_FLIP;
- ANCHOR_MISSING;
- SOURCE_MISSING, original channels 4,5,6 unavailable, independent reference observed.

Compare same patient labels, masks, TRAIN label budgets:
- original V56 native fuzzy model with no reference;
- V57 drop-group reference gate;
- V58 coherence-aware reference **repair**;
- raw logistic on original 12 channels;
- raw logistic on original 12 + independent reference (extra information control).

AUPRC, AUROC, NLL, Brier, repair rates/reasons, TRAIN coherence threshold, exact additive reconstruction, changed-vs-unchanged per-group contributions, pairwise descriptive TEST bootstrap NLL intervals. Critically report anchor-only and jointly corrupted failure even if repaired model wins on common-mode faults. Do not select thresholds or variants from V58 TEST outputs.

## Interpretation / falsification

The reference sensor has externally declared independence but **the simulator's assumption is not clinical truth**. Even perfect recovery is limited to a source with known independent sensor provenance. Our method cannot identify unnamed physiological concepts or prove the number of real physical factors, nor resolve a device artifact admitted for its predictive association. A common-mode source fault can remain observationally indistinguishable from a real physiological event if all reference instruments share the fault. Real anchor fault behavior needs independent device calibration and actual fault observations.

The method combines pre-existing consistency diagnostics and sensor-fusion ideas with a strictly concept-mediated native fuzzy model. No first-invention claim for those techniques. Compare externally against established CBM, MAGNETS, ConceptTime when pursuing publication.
