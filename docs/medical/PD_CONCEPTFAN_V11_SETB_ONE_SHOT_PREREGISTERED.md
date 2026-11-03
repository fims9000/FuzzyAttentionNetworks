# ONE-SHOT PhysioNet Set B validation — FROZEN BEFORE EXECUTION

**Status:** Protocol registered before *any programmatic Set B model evaluation*.
Data are publicly archived as of 2026 but were held out during the
2012 competition. We checked availability/header/source metadata,
**not cohort metrics or trained-model predictions** before freezing.
This is an independent *patient* cohort, **NOT an external hospital**:
Set A and Set B both derive from the historical MIMIC-II source.

**Exact research question:** Do the V11 development tradeoffs — poorer
AUPRC yet better concept-pair consistency — reproduce on 4000 different
patients, and do the signed effects of the previously identified
single-axis candidates generalize to an unseen cohort?

## Non-negotiable pre-registered protocol

**Source split**:
- Set A: 4000 original patient records, train 2400, validation 600,
  calibration 400 sealed, original test 600 sealed.
- Only original 2400 Set A train cases may affect train-only spectral
  rank, threshold/candidate construction, feature selection,
  normalization, model parameters or cross-seed stability regularizers.
- Old repeatedly observed 300-case Set A DEV is consulted **ONLY**
  for exact deterministic refit verification against archived
  runs 41689286080 (V11) and 41626717843 (nonconcept baselines).
  Required absolute AUPRC drift <=0.002 for every method/seed,
  or **ABORT without even downloading Set B**.
- After ALL model-check receipts pass, the same process downloads
  official public Set B 4000 patients and `Outcomes-b.txt` for
  one-shot scoring. No Set B patients, masks or labels enter a
  normalizer, feature pilot, computational rank or training loss.
- Set A/Set B RecordIDs must be disjoint, and Set B filenames must
  agree with embedded IDs and outcome rows. Model receives ONLY
  recorded observed 0..35h time series; 36..48h is filtered BEFORE
  aggregation. The outcome file is never a feature source.
- No per-patient predictions, labels, case IDs, concept trajectories
  or model checkpoints are uploaded to GitHub Actions artifacts.
  Source file SHA256 and SHA256 sorted IDs are reported; the IDs
  themselves are excluded.

**Methods** (same trained-from-scratch CPU code, no hyperparameter
changes or holdout early stopping):
1. V10 consistency refit;
2. V11 projector + axis-view loss;
3. V11 projector + group-view loss;
4. Logistic Regression on observed 36h patient summaries;
5. sklearn HistGradientBoosting (NOT XGBoost);
6. GRU-D-style local recurrent decay implementation (NOT official
   original pretrained weights);
7. 2-layer/4-head/64-d time-series Transformer, no pretrained weights.

Three independent initializations (11,23,37) for each neural method;
Logistic and HGB each receive one deterministic fit with seed 11.
Existing Set A 2400 train records only, 18 epochs, batch 128,
AdamW 0.001, weight decay 1e-4, 20% artificial observed-value
masking during training, original clinical mask as a separate
observational concept source.

**Prespecified evaluation conditions**:
- original observed 36h time series (PRIMARY);
- additional 35% MCAR thinning of recorded observations;
- artificial time/channel MAR thinning. Both secondary views keep
  original clinician sampling mask separately known to ALL models.
- model-level mean AUPRC primary; AUROC and raw class-weighted
  Brier auxiliary; train and CPU inference seconds.
- AUPRC differences: V11 group vs V10, V11 group vs Transformer,
  V10 vs Transformer, V11 axis vs V10, across all three views.
  Paired stratified patient AND independent trained seed
  percentile bootstrap, 800 draws with fixed seed 20261103;
  no cross-hospital claims or paper-promising p-values.

**Pre-registered concept-fidelity evaluation**:
- The native source groups are built on Set A TRAIN only.
- On new Set B patients, each of the three model pairs (11/23,11/37,
  23/37) must agree with *signed* Spearman >=0.5 for direct
  physiological **logit** contribution and >=0.75 on the
  learned definition projector overlap. Both thresholds were
  previously introduced on Set A DEV; they are not reselected
  using Set B.
- An unstable source group MUST ABSTAIN. A group of two stable
  but rotated axes remains a GROUP, not two uniquely identified
  clinical factors. Individual computational effect candidates
  cannot be named real physiological mechanisms without separate
  clinical validation.
- Failure to increase the count of reproducible singletons,
  a negative V11 AUPRC result, a near-zero strict signed correlation,
  or a CI crossing zero are all *valid recorded negative outcomes*.

**Decision rule**:
- If V11's mean AUPRC (original) falls more than 0.01 below V10
  AND there is no increase in empirically reproducible candidate
  effect count, interpret as **negative result**.
- Even if V11 meets this rule, a contribution to modeling requires
  external concept-method evaluation, several training orders
  and an independent hospital/site; one MIMIC-II source cannot
  substantiate general clinical transfer.

**One-shot guarantee**:
The independent-cohort scoring workflow triggers **ONLY** when its own
workflow file is first added under
`.github/workflows/conceptfan-v11-setb-ONE-SHOT.yml`.
Its triggering commit follows passing all fake-record and full concept
preflight tests. Later changes to source code or docs cannot trigger
another Set B score automatically. Do not rerun/repair/follow-up
against Set B to tune a method based on observed labels.

**Reproduce model-preflight WITHOUT Set B**:
```bash
PYTHONPATH=src:. python -m pytest \
  tests/discovery/test_holdout_setb_v11.py \
  tests/discovery/test_effect_certification_v11.py \
  tests/discovery/test_passports_v11.py \
  tests/discovery/test_group_learning_v11.py -q
```

**Data availability provenance**:
- PhysioNet/CinC Challenge 2012 v1.0.0, released materials:
  https://physionet.org/content/challenge-2012/1.0.0/
- Current Set B outcome file:
  https://physionet.org/files/challenge-2012/1.0.0/Outcomes-b.txt
- It was withheld from 2012 contestants, but the file is now
  publicly accessible. The legacy challenge page's past-tense
  restrictions should NOT be misread as an absence of public data
  today.

**Important caveat**: This pre-registration is a research protocol
recorded in the same authors' repository, not a third-party trusted
registration service. Treat uncertainty intervals and any
borderline effects accordingly.


## Forensic amendment BEFORE ever accessing Set B (2026-11-03)

[First attempted CI run 41694189228](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41694189228) **aborted while still in Set A TRAIN-only refit verification**. All three V10, all six V11 and all three GRU-D models reproduced their prior DEV AUPRC; Logistic deviated 0.000907 (within the preregistered 0.002 tolerance), HGB matched. The Transformer seed 11 yielded 0.440585 instead of 0.460356 and the script raised `REFIT DRIFT before Set B access`. Its Set B download call occurs strictly after the guard, so **no Set B feature or outcome was downloaded, opened or scored** in that attempt.

A separate [A-ONLY two-environment forensic test, CI 41695791418](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41695791418) recovered the cause: the original external baseline workflow explicitly fixed `PYTHONHASHSEED=0` while the attempted one-shot workflow omitted it. Python/torch/scikit-learn versions, the data adapter, model code and 12 observed input channels are identical. Under `PYTHONHASHSEED=0` two independent seed-11 Transformer refits both reproduce AUPRC **0.46035551611473996** exactly. Under `PYTHONHASHSEED=random`, both refits produced **0.4522096274019566**, and the first aborted run with an unpinned environment was 0.4405848. These are SOURCE-COHORT metrics only, not Set B outcomes.

The protocol is now strengthened to **reject execution unless `PYTHONHASHSEED=0`**, keeping all previous fixed model weights, optimization rules, seeds and held-out evaluation criteria unchanged. The corrected B workflow will receive a distinct one-shot filename and link back to the aborted pre-B run, not silently overwrite historical evidence. A failed refit guard would still stop before downloading B.

This amendment fixes reproducibility before the first actual Set B access and **must not** be described as optimization on external outcome labels. The difference in random-hash environments is itself an implementation reproducibility finding, not proof that the Transformer is fundamentally stochastic under a fully pinned environment.

## Additional A-only abort and execution-order forensic amendment (2026-11-03)

The [fixed-hash second attempt 41696990574](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41696990574) also ABORTED before Set B access: despite the correct environment variable, the Transformer refit at seed 11 still returned original-DEV AUPRC 0.440585 instead of the frozen 0.460356. **Thus hash seeding is necessary but not sufficient in the joint refit context.** In particular, the stronger claim in the first amendment that the omitted hash seed was the sole cause must be withdrawn. All other checked methods' original-DEV AUPRC continued to match or fall within the fixed tolerance.

Inspecting the historical external baseline pipeline reveals an additional uncontrolled procedural difference. The original benchmark trained V10 (3 seeds), then Logistic/HGB, then GRU-D (3 seeds) and Transformer (3 seeds). The one-shot script had trained V10, then **six additional V11 networks**, then Logistic/HGB, GRU-D, Transformer, while assuming every random effect was insulated by a local `manual_seed`. The altered sequence was NOT part of original external comparison, and may change process/global state affecting a stochastic Transformer.

We have therefore implemented an A-only diagnostic `FIT_ORDER=(V10,Logistic,HGB,GRU-D,Transformer,V11-axis,V11-group)`, matching the historical original external training order exactly and leaving model weights, hyperparameters, seed values and input features unchanged. The **17-model Set A-only regression** runs through an explicit `--a-only-preflight` flag and exits BEFORE downloading B, with a new separate CI receipt. The project will authorize a new B evaluation ONLY if every original previous-development result reproduces within the already-locked AUPRC tolerance under that execution order.

**Scientific integrity:** neither the first nor second aborted job downloaded, scored or reported Set B records, so no new-cohort outcome information has influenced this remediation. Both failed job logs are retained; no changes may be made in response to Set B results after access. The issue is deterministic re-training reproducibility, not a claim about new data.

## A-only forensic SUCCESS, final code freeze before Set B (2026-11-03)

[All-17-model matched REFIT on Set A, CI 41706946832](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41706946832) **PASSED**. This used the original historical external baseline ordering and `PYTHONHASHSEED=0`; every A-only DEV AUPRC for V10 3 fits, V11 axis 3, V11 group 3, Logistic 1, HGB 1, GRU-D 3 and Transformer 3 matched the prior historical receipt **exactly**: maximum reported AUPRC difference **0.0**. The terminal STATUS explicitly reports `A_ONLY_REFIT_PRECHECK_NO_NEW_COHORT_ACCESS`, `total_frozen_models_verified=17`, `new_set_b_zip_and_outcomes_never_downloaded=true`. The artifact `conceptfan-seta-noB-all17-refit-2` contains the exact old-versus-new per-model comparison and original source code SHA.

The exact frozen `src/conceptfan_discovery/holdout_setb_v11.py` blob SHA at the successful Set A-only run is `d084fe9fc3b0a50d4581b71cdbd7cd7ab70aa79b`, verified unchanged before the new-cohort workflow creation. This freeze is more restrictive than simply comparing summary averages.

The previous second failed Set B workflow, despite having correct PYTHONHASHSEED=0, is documented above and remains visible. The source issue was the **combined per-process execution sequence**, not an intrinsic lack of Transformer reproducibility in its historical original execution context. No Set B labels have been read in any previous attempt; no hyperparameter or data-preprocessing modification was driven by external outcomes.

The first actual Set B analysis must now use the same A-only-verified `FIT_ORDER` and environment settings, reject any refit drift, and publish only the predefined cohort aggregate outputs. Any B result, positive or negative, must be retained without optimizing on that holdout.
