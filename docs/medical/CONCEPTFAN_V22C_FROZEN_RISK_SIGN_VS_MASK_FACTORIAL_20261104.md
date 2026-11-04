# V22c — frozen independent TRAIN risk sign vs descriptive-slot exclusion factorial (2026-11-04)

## Research question (pre-execution)
V22b used a **joint** change to native risk training: TRAIN-screened initial sign ±0.05 on outcome-relevant concept slots and **structural zero risk contribution** on outcome-uncertain slots. The risk AUPRC gain was large (0.52708→0.83823 on a source-unmixed artifact variance simulator) but this first intervention cannot identify whether sign or exclusion drives the improvement. In addition, the V22b aggregate signed cross-seed Spearman was calculated over both slots even where a descriptive-only slot has constant zero contribution, creating an artificial ≈0.5 aggregate. V22c corrects these two scientific gaps.

## Frozen factorial design
All four existing V22 families (already inspected; no claim of new independent generalization), same source-producing TRAIN seed 522201. Source and model fit only **first 600 TRAIN episodes**, independent source–outcome label screen only remaining **200 TRAIN episodes**. Same TRAIN-only whitened covariance estimator, external environment labels, fixed original two-slot native V10 ConceptFAN training task/forecast/reconstruction loss, 10 epochs, batch size 96, three neural seeds 11/23/37. VALIDATION seed 522202 only for descriptive metrics; TEST seed 522203 sealed. The same K2 covariance BSS basis is obtained **only** where the V22 unsupervised source-identifiability policy ACCEPTS; otherwise the study records four explicit statistical ABSTAINS per neural seed.

Per accepted source family, exactly **four** arms differing ONLY in native task parameter initialization:
- `random_original`: original pretrained random risk coefficients, no TRAIN sign, no no-risk gate;
- `signed_start_only`: if independent TRAIN screen ACCEPTS a source, set that source's raw decision weight to ±0.05, leave other coefficient unrestricted;
- `descriptive_mask_only`: mark TRAIN-screen-unrelated concepts with persistent native risk policy 2 (exact zero) but keep accepted source coefficient randomly initialized;
- `signed_start_and_mask`: both actions.

4 structural generators × 3 neural seeds × four arms = **48 opportunities**, of which 24 are expected trained (two source-identifiable families) and 24 independently justified structural/source refusals (two nonidentifiable/drift families). Report actual counts, never alter status to pass CI.

## Metrics and controlled acceptance
Record validation AUPRC/AUROC/Brier, actual native additive-logit error, learned individual signed decision coefficients and masks, mean signed rank across seeds:
- **all-unmasked mode:** slots not structurally prohibited (may include a weakly predictive/irrelevant slot);
- **active-only mode:** excludes risk-policy=2 slots from pairwise patient-level Spearman calculation; constant contributions yield undefined/NULL, never numeric zero.
- Raw physical/instrument oracle source matching only loaded after all fits and for evaluation, never an origin label used in task training; evidence may be stronger for a generator-designated artifact.

Inspect per-seed results as well as mean; positive mean differences are not independent cohort p-values (all seeds share validation patients). The conditions differ in native decision weight prior, not source basis or extra model capacity; no post-hoc classifier or black-box bypass.

**Expected outcomes are not predetermined:** sign-only, mask-only, both or neither may win. If sign-only explains most gain, do not portray risk masking as a prediction innovation; if mask-only matters, preserve that as independent contribution. If both are needed, state that. This is an exploratory mechanism ablation, not a clinical correctness test.

## Validity and publication boundary
Known mixed-source simulator, no official intrinsic concept model baseline, no real doctor feedback, no physical/causal source identity or clinical safety claim; accepted IEEE PIERE paper untouched. Preserve every negative result. Draft PR #19 stays Draft.
