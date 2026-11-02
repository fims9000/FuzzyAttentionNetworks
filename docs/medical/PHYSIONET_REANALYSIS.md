# PhysioNet 2012: frozen-evidence corrections (2 Nov 2026)

This change is an **analysis-only extension**, not a replacement for the frozen
`conceptfan-posthoc-attribution-final` release. Previously published reports,
checkpoints, split files, and acceptance statuses are intentionally unchanged.

## 1. Cross-retraining stability: dependency-correct uncertainty

The legacy `stability.py::_hierarchical_ci` resampled 435 overlapping pairs as
though they were independent. The new implementation resamples **trained
models** with replacement, weights each distinct original pair by the product
of its model multiplicities, excludes self-comparisons, and resamples the 600
held-out patients once per bootstrap replicate. Both levels retain the
dependencies induced by shared models and shared evaluation cases.

The original mean/median and model-pair count remain descriptive statistics.
The **new confidence limits must be recalculated** from frozen per-episode
artifacts. No CI numbers have been copied forward from the old table.

From the project root:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/medical/physionet2012/recompute_stability_ci.py \
  --artifacts-dir artifacts/physionet2012_last_run \
  --out artifacts/physionet2012_last_run/reanalysis/model_level_stability_ci.csv \
  --repetitions 2000
```

Required inputs:
- `artifacts/physionet2012_last_run/TABLES/episode_pairwise_stability.parquet`
- `artifacts/physionet2012_last_run/TABLES/stability_pair_summary.parquet`

The script deliberately fails instead of reconstructing phantom observations
from an aggregated CSV. The released article tables are not overwritten.

## 2. Residual signal conditional on the original proxy concepts

The existing leakage audit predicts mortality **from residual summaries**.
That proves association, not predictive information *in addition to true
proxy-concept trajectories*. The extension compares two probes on identical
frozen patients:

- proxy-concept time-series summaries only;
- the exact same summaries **plus** residual summaries.

Both probes use the same six feature families per concept (including
observation/missingness fraction); standardization is fitted on train only;
L2 strength is selected on validation only; calibration uses the calibration
split only; test is used once for paired Delta-AUPRC and Delta-NLL.

Run on an **existing** ConceptFAN-NoAlpha run (no model retraining):

```bash
PYTHONPATH=src:. .venv/bin/python scripts/medical/physionet2012/run_conditional_residual_audit.py \
  --artifacts-dir artifacts/physionet2012_last_run \
  --run-dir artifacts/physionet2012_last_run/runs/ConceptFAN-NoAlpha/run_01_init_11_order_1001 \
  --out artifacts/physionet2012_last_run/reanalysis/conditional_residual_run_01.json
```

This is a **one-model exploratory analysis**. Because each of 30 neural
checkpoints shares the same patients, running the same script over multiple
models does not create independent patient samples; any pooled uncertainty
requires a separate model-level analysis. Positive Delta-AUPRC by itself
does not prove causal information leakage or a causal encoder bypass.

## 3. Attribution report: interpret intervals correctly

In the frozen `posthoc_article_table.csv`,
`ci95_low` / `ci95_high` were created by taking the 2.5% and 97.5%
**quantiles of the 435 pair-specific metric values**. Those two columns are
not a 95% confidence interval for the *mean*. Treat them as a central 95%
pair-value interval until the paper table is updated separately. The
post-hoc `posthoc_method_comparison.csv` is different: it contains
model-level bootstrap intervals for *differences* between methods.

Do not silently rename/relabel the original frozen report. When writing
the expanded manuscript, distinguish pair-value distribution intervals
from bootstrap uncertainty about a reported average.

## 4. Predictive significance tests

`paired_model_comparisons` uses `calibration_method='none'`. The
newly generated table labels these comparisons as
`reporting_state='raw'`, `calibration_method='none'`.
Do **not** ascribe its Brier/NLL Holm-adjusted p-values to the calibrated
metrics listed elsewhere in the article.

## 5. Limits and QA

Unit tests for the bootstrap and conditional probes use synthetic arrays and
do not require patient data. The reanalysis scripts require the historical
private/local frozen artifacts. Until the per-episode artifacts are supplied and those scripts are executed,
**no corrected two-level confidence limits or conditional residual gains are claimed**.
A separate fixed-cohort model-only interval can be calculated from the public
pairwise CSV (Section 6).

Repository source and scientific manuscript remain separate deliverables.
The manuscript must also describe the different fuzzy-evidence
parameterizations in synthetic (multi-membership) and PhysioNet (single
monotone-sigmoid) protocols. Full clinical validation is not claimed.

## 6. Verified model-only reanalysis from checked-in post-hoc CSV

The post-hoc pairwise CSV is public and contains all 435 model pairs for
ConceptFAN, Integrated Gradients and GradientSHAP for each representation.
The source table was independently parsed and its 0.00346858 / 0.78033745 /
0.84568889 primary signed Spearman averages were confirmed. A 3,000-replicate
model-resampling pilot (held-out 600 patients **fixed**) produced approximate
95% ranges: ConceptFAN [-0.0499, 0.0820], IG [0.7676, 0.7942], and
GradientSHAP [0.8382, 0.8539]. These are not patient-bootstrap intervals
and should not replace the two-level clinical uncertainty report.

Reproduce the comparable Python analysis from the checked-in CSV:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/medical/physionet2012/recompute_posthoc_model_only_ci.py \
  --out artifacts/physionet2012_last_run/reanalysis/posthoc_model_only_ci.csv \
  --repetitions 3000
```

The point estimates agree with the registered post-hoc article table; the CI
bounds are Monte Carlo approximations and will vary slightly with the PRNG.

## 7. Empirical agreement nulls with observed frequency preservation

For five concepts, 0.20 (Top-1) and 0.50 (sign agreement) are only simplified
idealized chance levels. To avoid inappropriate baselines the new audit
reports both model-specific marginal-frequency expectations and an empirical
null obtained by permuting patient correspondence independently within every
frozen model. The procedure preserves each trained model's empirical Top-1
frequencies and sign frequencies by concept, while disrupting cross-model
episode matching. This tests episode alignment under a controlled permutation,
not the entire notion of explanation identifiability.

```bash
PYTHONPATH=src:. .venv/bin/python scripts/medical/physionet2012/run_empirical_nulls.py \
  --runs-root artifacts/physionet2012_last_run/runs \
  --out artifacts/physionet2012_last_run/reanalysis/empirical_nulls_fan.json \
  --permutations 1000
```

The audit fails if frozen `contributions_test.parquet` files are missing.
Do not substitute pairwise-average summaries, from which frequency-preserving
nulls cannot be reconstructed.

## 8. Full 30-model conditional residual comparison

The single-run script above is a local smoke audit, not the confirmatory
multi-model result. The supplementary frozen-grid pipeline now runs all 30
model-specific residual probes and compares their predictions to an identical
proxy-only baseline; it uses a paired two-way bootstrap with shared patient
resampling across all sampled models.

```bash
PYTHONPATH=src:. .venv/bin/python scripts/medical/physionet2012/run_conditional_residual_grid.py \
  --artifacts-dir artifacts/physionet2012_last_run \
  --out-dir artifacts/physionet2012_last_run/reanalysis/conditional_grid_30 \
  --bootstrap 2000
```

Outputs are `per_run_deltas.csv` and `summary.json`. They remain exploratory
post-hoc analyses, not preregistered proof of causal leakage. The runner
requires existing 30 model-level frozen `concepts_{split}.parquet` exports.

## 9. Archive availability and limitations

The public PhysioNet release asset `PHYSIONET2012_CONCEPTFAN_LAST_RUN_e60d9a3.zip`
contains `artifacts/TABLES/episode_pairwise_stability.parquet` and
`artifacts/TABLES/stability_pair_summary.parquet`, so the two-level interval
reanalysis can be run after extracting that public archive. It deliberately
excludes patient raw ZIPs, checkpoint `.pt` files and `.npz` prepared data.
It also does not include per-run `concepts_{split}.parquet` or
`contributions_test.parquet` files, which are required for the empirical null
and conditional residual analyses. Do **not** claim that extracting the
public release alone provides these missing private/local outputs.

The confirmed archive digest from the release metadata is
`17faed9c89d553c3dd971dca92ee078dfb76543ec820e7c674ce128ea240b6f7`.
Verify the downloaded archive against its matching GitHub `.sha256` asset
before running the read-only reanalysis. Never commit raw patient data or
intermediate per-patient predictions.

## 10. Single-command reproducible reanalysis on Linux

Use the repository virtual environment and run the read-only suite:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/medical/physionet2012/run_reanalysis_suite.py \
  --artifacts-dir artifacts/physionet2012_last_run \
  --out-dir artifacts/physionet2012_reanalysis_20261102 \
  --tests
```

The suite first runs the public model-only analysis, then all other analyses
*only when their required frozen input files exist*. Its machine-readable
`STATUS.json` marks each step PASS, NOT_AVAILABLE, or FAIL, along with missing
paths, execution logs and SHA256 of output files. It never alters the original
publication tables, changes checkpoints, or overwrites an existing audit folder.
A PARTIAL_OR_FAILED status is expected when private local source tensors are
not mounted; **it is not scientific validation**.
