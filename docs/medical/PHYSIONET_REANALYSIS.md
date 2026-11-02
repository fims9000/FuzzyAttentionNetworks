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
- `artifacts/physionet2012_last_run/tables/episode_pairwise_stability.parquet`
- `artifacts/physionet2012_last_run/tables/stability_pair_summary.parquet`

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
private/local frozen artifacts. Until those are supplied and the scripts are
executed, **no corrected confidence limits or residual gains are claimed**.

Repository source and scientific manuscript remain separate deliverables.
The manuscript must also describe the different fuzzy-evidence
parameterizations in synthetic (multi-membership) and PhysioNet (single
monotone-sigmoid) protocols. Full clinical validation is not claimed.
