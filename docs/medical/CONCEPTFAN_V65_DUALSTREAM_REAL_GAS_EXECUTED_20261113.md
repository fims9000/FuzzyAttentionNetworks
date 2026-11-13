# V65 EXECUTED — two-stream absolute/change multi-threshold ConceptFAN on real dynamic gas mixtures

**2026-11-13; exploratory posthoc architecture improvement, NOT independent generalization validation.** Original development sequence: V63 new unseen dataset -> V64 information loss diagnostic -> V65 dual-stream architecture, all on the same already inspected UCI322 TEST. Frozen architecture/metrics in `CONCEPTFAN_V65_FROZEN_DUAL_STREAM_MULTIFUZZY_20261113.md` before seeing V65 results. The TRAIN-only StandardScaler fairness correction to the concept-only Ridge head was committed BEFORE V65 TEST scores.

## Reproduction and provenance

[Successful GitHub Actions run 43481505487](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43481505487), executed source commit `8499c79aeb6917c3e18f60f7543498c6ecae800b`, artifact `conceptfan-v65-real-dualstream-fuzzy-architecture` ID **14064479814**; full `V65_DUALSTREAM_EVIDENCE/STATUS.json`, `METRICS.csv` (64 rows), SHA and log. **22 tests PASSED**, complete source/frozen-head execution PASS. This is the final standardized-head run; earlier CI runs on older/unscaled code do NOT constitute the primary evidence.

Data: same **official UCI322 dynamic gas mixtures**, 16 measured physical sensor outputs, two commanded gas channels as downstream references only, 2 full continuous recording sessions, first60% TRAIN/middle20% VALID/last20% TEST with 120s bilateral embargo, 120s nonoverlapping correlated windows (**TRAIN414, VALID134, TEST136, 68 TEST per gas pair**). Original ZIP SHA256 `8b6323b801363e11343ba1a5a7718e76666b715e753b3a40dbdfd1838d953fe7`. Setpoints are controller command traces, NOT independent chemical analyzer readings; windows are not independent device trials.

## What the updated architecture actually implements

- Same original **K=3 anonymous** TRAIN-only unlabeled ICA innovation directions as V63, not a hand-specified count of two gas sources.
- Each axis has **absolute sensor level and causal change from observed initial baseline** branches: 6 time-local source streams. Each has TRAIN-calibrated Q25/Q50/Q75 smooth sigmoid fuzzy memberships -> **18 fuzzy states**, emphatically NOT 18 distinct chemical sources.
- Use both **whole-history and last30s** aggregation -> **36 inspectable signed concept contributions** for the Ridge task head. Task head is fit on TRAIN controller setpoints only, with TRAIN-only per-feature standardization, then transformed back to the original fuzzy coordinate system for exactly additive signed attribution. Original raw measurement channels never bypass concepts into the final ridge head.
- Two variants: 18× static fuzzy states and trainable **V56 native diagonal recurrent fuzzy dynamics** (8 unlabeled self-supervised epochs, neural seed11, no source gas labels). Native self-supervised total TRAIN loss decreased **.06321 -> .03937**, which is TRAIN optimization evidence, NOT a physical meaning certificate.
- Compare same supervised-label budget raw 16-sensor Ridge/HGB and fixed earlier V64 ICA absolute/delta linear and V63 single-fuzzy K3 controls. Heldout TEST already inspected in prior V63/V64 studies, so these are *exploratory engineering* comparisons.

## Entire heldout TEST results, all preregistered relevant baselines

R² greater is better; negative indicates worse than predicting heldout sample mean.

| Physically commanded target | Original V63 ICA fuzzy K3 | Original V64 ICA absolute-linear K3 | V65 static dual stream Q25/Q50/Q75 | V65 native dual stream | Raw sensor Ridge |
|---|---:|---:|---:|---:|---:|
| CO (ethylene+CO) | -.0134 | +.6016 | **+.5874** | +.5353 | **+.6792** |
| Ethylene (CO session) | +.0398 | +.5319 | **+.5831** | +.5667 | **+.6964** |
| Methane (ethylene+methane) | -.4825 | +.4063 | +.4652 | **+.4774** | **+.5348** |
| Ethylene (methane session) | +.0125 | +.5650 | **+.5957** | +.5900 | **+.6498** |

Raw sensor Ridge is better than any version of ConceptFAN for **all four** gas references, including both static and neural dynamic V65. The V65 revision nevertheless restores task information lost under the old delta-only fuzzification. V65 static outperforms native on TEST for three of four tasks, so recurrent native dynamics has NOT established a robust advantage over static fuzzy states. CO is a special caution: **V64 ICA absolute-linear** +.602 is better than V65 fuzzy static +.587 and native +.535; a larger nonlinear concept bank is NOT always an improvement.

MAE ppm, same TEST:
- CO: raw Ridge **47.77** vs static **60.10** vs native **66.34**.
- Ethylene+CO: raw **2.387** vs static **2.643** vs native **2.734**.
- Methane: raw **20.86** vs static **25.06** vs native **26.01**.
- Ethylene+methane: raw **2.124** vs static **2.261** vs native **2.371**.

## VALID-only prospective warning rule — do not hide deficits

V65 defines an unchanged and fixed **VALID-only** flag of source explanation incompleteness when the equal-label raw Ridge R² exceeds native V65 R² by **greater than 0.05**. This is a performance sufficiency warning, NOT semantic or physical identity proof. Threshold is not tuned on TEST, no raw-score shortcut can alter the fuzzy predictor.

| Source target | VALID raw Ridge R² | VALID native V65 R² | Flag |
|---|---:|---:|---|
| CO | .6429 | .5311 | **Incomplete** |
| Ethylene+CO | .5561 | .3331 | **Incomplete** |
| Methane | .6827 | .4948 | **Incomplete** |
| Ethylene+methane | .8022 | .7586 | No warning (gap .0436) |

That last non-warning does not certify real source completeness; it simply did not cross the prespecified *predictive* threshold. On TEST the raw Ridge still exceeds V65 for that target by about 0.060 R², demonstrating the fallibility of a small VALID-only rule.

## What is and is not scientifically established

**Executable positive engineering result:** A train-only unsupervised ICA source bank with a **separate absolute state and causal difference** channel, multiple calibrated fuzzy thresholds, and late-time summaries can recover substantial predictive information previously lost. It can produce exact signed, additive, inspection-friendly explanations without using measured physical gas setpoints to learn source definitions. Tests cover causal prefixes, withheld measurements, initial-base missingness, source integrity, coefficient identity and no task gradients into source learning.

**Not established:** independently identifiable ethylene/CO/methane concepts, source count, reliable cross-device/source robustness, superiority to raw models, source fault recovery from independent hardware, or clinical validity. Increasing the number of membership functions does not create independently meaningful source variables. A physical/semantic factorization requires independent source-specific interventions/controls and a truly new external testbed.

**Next scientific boundary:** explicitly disambiguate functional fidelity from semantic identification, specify how to attach separately calibrated physical anchors to discovered axes, and evaluate on **new untouched physical recording sessions**. Any new threshold/architecture change based on this same UCI322 TEST is additional model selection, not independent publication-grade confirmation. Keep the PR DRAFT; accepted PIERE/IOI manuscripts and master remain unchanged.
