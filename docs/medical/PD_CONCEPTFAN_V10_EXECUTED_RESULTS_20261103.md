# ConceptFAN V10 — executed clinical research evidence, 2026-11-03

**Status:** completed exploratory V10 study; cannot be used as an
independently verified clinical paper result. Archived accepted PIERE
paper unchanged.

## Reproducibility

[GitHub Actions run 41623853277](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41623853277)
**PASS**: tested 13 focused V10/V7 regression contracts,
downloaded official PhysioNet/CinC 2012 Set A,
train-only measured-channel selection, refitted all three methods
under same 18-epoch optimizer, source seeds 11/23/37,
same V7 rank-capped K=8 physiological slots and exactly three
separate care-process concepts.

**Patients:** original Set A 2,400 TRAIN; 600 overall development
validation of which 300 patient IDs from a deterministic prior
selection are evaluated here. No original calibration/test outcome or
features are used. The 300 cases have already been inspected
repeatedly during V6/V7/V8/V9 development; this is **not a new
confirmatory validation set**.

All analysis files originate from pinned artifact
`conceptfan-v10-matched-clinical-4` attached to successful run.
Individual patient predictions, IDs, waveforms and masks were not
uploaded as release artifacts.

## Matched aggregate results

| Development metric (mean 3 seeds) | V7 refit | V10 multi-scale | V10 + cross-view |
|---|---:|---:|---:|
| Original AUPRC | **0.298550** | **0.375207** | **0.379405** |
| MCAR-thinned AUPRC | 0.263071 | 0.318919 | 0.327838 |
| Time/channel MAR-thinned AUPRC | 0.286363 | 0.383375 | 0.387322 |
| Physical-branch-only AUPRC | 0.283102 | 0.367822 | 0.372582 |
| Care-process-only AUPRC | 0.202106 | 0.187180 | 0.188977 |
| Future observed-feature RMSE | 0.806244 | 0.806053 | 0.806170 |
| Equal random-masked feature reconstruction RMSE | 0.818596 | 0.808392 | 0.809086 |
| Physiological contribution effective-number entropy | 5.5903 | 5.2502 | 5.2420 |
| Physiological case-specific signed Spearman excess | **0.268614** | 0.117369 | 0.103670 |
| Physiological definition alignment, scale included | **0.902762** | 0.824141 | 0.801630 |
| Care-process case-specific Spearman excess | 0.008513 | 0.233191 | 0.119500 |
| Care-process absolute direct logit share | 0.307135 | 0.187799 | 0.180706 |
| Max exact-additivity error | 8.94e-8 | 1.19e-7 | 1.19e-7 |

These are *observed* differences. Repeating the three V7
initialization seeds provides no independent patient, hospital or
hyperparameter-selection replication.

**Physiology improved discrimination**: AUPRC increases in original,
MCAR and time/channel MAR scenarios and the physiological-only
branch also improves. Average reconstruction error is lower,
future-observation RMSE changes only marginally.

**Physiological explanation agreement deteriorated**: the primary
negative outcome is a decline from 0.2686 to 0.1174 case-specific
Spearman excess after proper operator+time-scale slot matching. Adding
the within-network mask consistency penalty reduced it further to
0.1037, even though AUPRC rose by another ~0.0042.

Thus it is **wrong** to describe V10 as a solution to explanation
reproducibility. It demonstrates a tradeoff: more predictive
multi-resolution physiological concepts can assign importance
differently under independent optimization.

### Exact finite-seed sensitivity

With **only 3 initialization clusters**, there are exactly 8 signed
permutations and 27 paired seed bootstrap combinations. The smallest
two-sided exact sign-flip p is **0.25**. It is impossible to establish
conventional significance at alpha 0.05 using this seed-only exact test.

Paired V10-minus-V7 AUPRC differences on the original development
subset are:

- V10 multi-scale: +0.076656; individual seed differences
  [0.03838, 0.12334, 0.06824]. Descriptive 3-seed resample percentile
  interval **[0.04485, 0.11141]**.
- V10 multi-scale + cross-view consistency: +0.080854;
  [0.03860, 0.12369, 0.08027], percentile interval
  **[0.04763, 0.11428]**.

These intervals **do not** include new patients, centers, independent
data-order effects, statistical method-selection uncertainty or
multiple-model tuning on this prior development dataset. Even though
the intervals exclude zero, they are NOT confirmatory confidence
intervals for patient-population generalization. Do not claim
statistically significant superiority.

## What exactly changes inside V10

Each physiological concept k combines the same train-only, native
signed measured-sensor operator dictionary over four *strictly
historical* windows 1,4,12,24 hours via a learned simplex
pi[k,h] = softmax(scale_logits[k,:]). Time-scale coefficients were
trained jointly with outcomes, future observations and masked
reconstruction, not obtained post-hoc.

Representative learned scale weights from a single successful V10
initialization illustrate the interpretability of the parameters
(not all three runs, and not a true clinical identity). For slot 0:
`{1h:0.474, 4h:0.199, 12h:0.158, 24h:0.169}`.
For slot 6: `{1h:0.671, 4h:0.108, 12h:0.109, 24h:0.113}`.
Names/meaning follow measured physiological operators, not manually
assigned disease diagnoses.

The observation-policy head consumes only the ORIGINAL recorded mask,
and artificially withheld training/validation observations never
change that source. At total absent support, the physiological
representation is UNKNOWN-neutral, not negative. All direct logit
terms remain exact.

## Why not publish now

Neither improved AUPRC nor lower feature RMSE proves that discovered
physiological concepts correspond to clinical causal processes.
The observed decline in case-specific explanation reproducibility is
substantial and must not be removed from the report. There is no
independent hospital, no new validation cohort, no physical concept
ground truth, and only three independent model initializations.

Next **falsifiable** research gate:
1. On a **new validation population**, preregister the V10
   multi-scale architecture, do not use prior development data to
   choose its scale weight, consistency weight or K.
2. Compare with matched V7, shared V9 mask source and same-cohort
   nonconcept Transformer; include an unmodified physiological
   concept reference/grounding task.
3. At least 10 independent model initializations x 3 data-order
   repetitions; estimate uncertainty by resampling originating
   model/patient/center levels rather than 435 overlapping model pairs.
4. Evaluate known-dimension synthetic dynamics without concept
   supervision, source-perturbation and genuinely external observation
   policy shifts, plus computational cost and three meaningful failures.
5. Gate acceptance on **both** AUPRC and physiological
   patient-specific concept reproducibility. If no Pareto advance,
   keep V10 as a documented tradeoff rather than claiming the
   research objective solved.

No accepted PIERE results or camera-ready manuscript are modified.
