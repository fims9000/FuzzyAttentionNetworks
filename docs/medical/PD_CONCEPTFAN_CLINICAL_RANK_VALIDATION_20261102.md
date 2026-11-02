# Clinical train-only predictive rank: final matched-sensor audit, 2026-11-02

**Executed:** [GitHub Actions 41503446183 — success](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41503446183),
`clinical-predictive-rank-9` artifact,
`published_clinical_rank/clinical_predictive_rank.json`.
Tests for temporal observation support, mathematical rank recovery,
synthetic white-noise controls and unobservable-factor counterexamples
are separate. The raw 4,000-case PhysioNet Set A was checked by its
source digest. The statistical rank audit used **only the 2,400 frozen
TRAIN cases**; the 600 validation, 400 calibration, and 600 test
patient labels and measurements were not used.

## Clinical rank is not stable across measurement granularity and channels

For 1h/4h/6h temporal binning, two transparent feasibility analyses were
executed.

- `Full cohort / standard coverage`: find each temporal resolution's
  jointly observed channels with sufficient **aligned** positive-lag
  patient-time pairs, then estimate rank.
- `Matched-channel analysis`: select a single fixed channel subset
  with support both among **true within-patient** positive-lag pairs
  AND expected pairs under **independent patient permutations**
  in *each* half of the TRAIN population. On EXACTLY this subset,
  recompute the full TRAIN rank, then independently estimate rank
  for both disjoint train halves.

The source method still has a limitation: some realized shuffled
permutations can have fewer pairs than their EXPECTATION, so a
half-sample rank may honestly remain NOT_IDENTIFIABLE despite passing
the expected-overlap filter. It is not replaced with zero covariance.

| Time aggregation | Original full TRAIN, source channels | Original full TRAIN rank | Common half-supported channels | Full TRAIN rank on common channels | Half 1 rank | Half 2 rank |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 hour | 28 | 13 | 23 | 11 | NOT_IDENTIFIABLE | NOT_IDENTIFIABLE |
| 4 hours | 34 | 3 | 28 | 14 | 9 | 8 |
| 6 hours | 34 | 3 | 28 | 14 | 9 | NOT_IDENTIFIABLE |

At 4-hour aggregation the *eight* matched directions between the
two independently estimated half-sample predictive subspaces have
mean squared principal-direction cosine **0.8224**. This is
partially aligned subspace structure, but dimensions 9 and 8
are not equal, and full-cohort dimension 14 is not recovered
by either half. Subspace overlap is not semantic validation of
physiological concepts.

At a fixed 4-hour aggregation, changing the transparent
observation-support subset from 34 to 28 channels changed the
detected rank **3 → 14**. This is not a harmless numerical detail:
the rank estimator and its permutation-derived spectral noise floor
are sensitive to channel count, missingness and source availability.
The same pattern holds at 6 hours.

## Consequences

1. There is **NO verified unique clinical number of dynamic concepts**
   in these ICU records. A full-cohort K of 3 or 14 would be
   cherry-picked if reported without the alternative availability
   protocol.
2. There is **NO evidenced clinical meaning** of spectral directions
   (nephrological, inflammatory, etc.). Their coefficients on
   measured channels can be displayed, but are not diagnostic labels
   and not proof of causes or clinical intervention effects.
3. The remaining real-world obstacle is not tuning regularization to
   increase accuracy or force K to equal the original manually
   specified five groups. It is defining a more defensible
   missingness-aware generative measurement process, controlled
   temporal environments, and an **independent reference for
   concept identity**.
4. The spectral 6/6 K match on well-observed synthetic generators
   remains a valid *conditional* finding but does not transfer to
   PhysioNet without verifying the required assumptions.
5. Because the common channel set was chosen using only the
   measurement-availability masks **of both halves**, the half-sample
   comparison is not a fully independent feature-selection and
   inference protocol. It is an exploratory feasibility
   analysis; p-values/confidence sets are not corrected for all
   possible channel subsets.

## Reproducibility

```bash
PYTHONPATH=src:. python -m pytest tests/discovery/test_clinical_observability.py -q
PYTHONPATH=src:. python -m conceptfan_discovery.clinical_predictive_rank \
  --config configs/medical/pd_clinical_predictive_rank_v3.yaml \
  --raw-zip data/physionet2012/raw/set-a.zip \
  --outcomes data/physionet2012/raw/Outcomes-a.txt \
  --out artifacts/medical/clinical_rank_exploratory_v3.json
```

No patient-by-patient identifiers, risk predictions or private
checkpoints appear in the publicly uploaded audit. The existing
frozen PIERE publication remains unchanged.
