# Executed PhysioNet evidence reanalysis — 2026-11-02

## Provenance

- Frozen code/research tag: `conceptfan-posthoc-attribution-final` -> `1dda55e287ccb613065e26d1ab58d5e8efe61e51`.
- Published release: [physionet2012-last-run-a3a8cf8](https://github.com/fims9000/FuzzyAttentionNetworks/releases/tag/physionet2012-last-run-a3a8cf8).
- Verified release ZIP SHA-256: `17faed9c89d553c3dd971dca92ee078dfb76543ec820e7c674ce128ea240b6f7`.
- Executed GitHub Actions: [frozen-evidence reanalysis #4](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41483344327), commit `fe0d48443c590e03910db953afe763efdbd6dbcd`, successful.
- New aggregate-only evidence (no raw patient data): `reports/physionet2012/reanalysis_20261102/two_way_stability_ci.csv` and `model_only_posthoc_ci.csv`.
- Archived CI outputs included STATUS.json, full 63-row two-level stability table and 18-row post-hoc model-only interval table.

## Corrected signed contribution Spearman uncertainty

The complete frozen episode-level table (three concept-model arms, 30 fitted
models per arm, 435 model pairs, 600 common held-out patients, three vector
representations, seven agreement metrics) was re-aggregated with 2,000
bootstrap repetitions using the **trained model** as the resampling unit and
one common patient resample for all selected pairs in each repetition.

| Frozen arm | Mean signed Spearman | Corrected 95% model+patient bootstrap CI |
| --- | ---: | ---: |
| ConceptFAN-NoAlpha | 0.00346858 | [-0.04959653, 0.08313994] |
| ConceptFAN-StabilityReg | 0.00037471 | [-0.05039394, 0.07198948] |
| PureNoFuzzy | 0.08876207 | [-0.00475029, 0.21593338] |

*Endpoints are rounded to eight decimal places; the machine-readable CSV above preserves full precision.* Means exactly reproduce
the original aggregate means, as they should. The previously published CIs
that resampled overlapping **pairs** as independent should not be cited as
model-level inference.

Interpretation: the corrected intervals for the mean signed Spearman each
include zero. The primary negative result about cross-fit reproducibility of
internal ConceptFAN contributions is unchanged.

### Additional verified checks

- Public post-hoc pair summaries were independently recomputed as a separate
  **model-only, fixed-patient** sensitivity analysis. IG and GradientSHAP
  remain much more stable on five grouped proxy dimensions than the
  architecture's internal contributions, but they attribute a different
  computational object.
- The repository's entire PhysioNet test module passed: **26 passed** in
  Actions #4. The first 25 passed before the repair was applied; the
  corrected integration test catches the old broken `stability.py` code path.
- No model training was performed. No frozen reports or original checkpoints
  were rewritten.

## Unexecuted audits and explicit limits

The public frozen ZIP intentionally omits per-run
`contributions_test.parquet`, `concepts_{split}.parquet` and the prepared
patient `.npz`. Thus the frequency-preserving Top-1/sign permutation
controls and the **conditional** proxy-versus-proxy+residual experiment
remain implemented and unit-tested but **have no new numerical results**.
Original residual-only AUPRC numbers cannot establish incremental
information beyond the observed proxies, nor a causal bypass.

The bootstrap describes variation among this set of trained models and shared
patients; it is not external-hospital validation. Ten reused initialization
seeds crossed with three order seeds may induce dependence across the
thirty fitted models; the current resampling treats those thirty fits as
the independent model-level units and does not separately estimate a
two-factor random-effects model. This is a limit on population-level
inference, not a reason to invent alternative uncertainty bounds.

Only commit a fully evaluated conditional/permutation outcome in a **new,
versioned** supplementary analysis with source manifests and verification.
Negative results must be retained.

## Reproduction

The workflow `.github/workflows/conceptfan-reanalysis.yml` downloads the
published release via `gh release download`, verifies its SHA-256,
extracts the `artifacts/TABLES` tree, runs the two-level script and test
suite, and uploads only aggregate outputs. See
`docs/medical/PHYSIONET_REANALYSIS.md` for local commands and full
program contracts.
