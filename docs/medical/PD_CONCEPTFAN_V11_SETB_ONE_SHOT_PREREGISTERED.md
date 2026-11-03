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
