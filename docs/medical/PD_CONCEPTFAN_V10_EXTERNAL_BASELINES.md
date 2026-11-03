# ConceptFAN **V10** vs real external prediction baselines

**Scope**: ONLY V10 is considered as the user's concept model. V7, V8
and V9 are **not** comparators in this experiment. All other rows are
independent established prediction architectures, implemented locally
and trained from scratch.

## Mandatory matched protocol

| Decision | Frozen specification |
|---|---|
| Source | Official public PhysioNet/Computing in Cardiology Challenge 2012 Set A |
| Label | In-hospital mortality (binary), no concept targets |
| Available history | **36 hours**, strictly no later signal in the input |
| Training | The original 2,400 patient IDs, full original outcome labels |
| Comparison cohort | The **same 300 repeatedly viewed development** cases as previous V10 |
| Sealed data | Original 400 calibration + 600 test patients |
| Observed sensors | Exact same train-only sensor pilot and observable rank / subset as V10 |
| Numerical preprocessing | V10 train-only robust median and IQR, no dev/test fitting |
| Artificial training dropout | 20% of available measured entries; original recorded mask separately accessible |
| Stress tests | Original, additional MCAR thinning, time/sensor-conditional MAR thinning |
| Clinical scope | Exploratory single-center-data source; no proof of real clinical concept identity |
| Output privacy | Only aggregate metrics and training times; never publish raw patient records, probabilities or IDs |

**Why provide the original recorded mask under thinning?** The V10
architecture sees the raw observed-measurement process separately from
the artificially hidden physiological entries. To avoid an unfair
advantage, every external classifier also receives the original
recorded source mask **plus** the visible measurement mask in MCAR/MAR
simulations. The source mask is historical and was actually observed
within 36 hours. It is not a hidden future outcome. Artificial thinning
is not an independent hospital.

## Externally established architecture families

1. **Logistic Regression on explicit 36h patient summaries**:
   measured-only mean, spread, min/max, first/last and trend plus source
   observation fractions and missingness. Scaled inputs; balanced
   training weights. Not a version of ConceptFAN.
2. **Histogram Gradient Boosting (sklearn)**: identical patient
   summaries to logistic, 180 trees max, <=15 leaves and balanced
   training weights. It is gradient boosting, **not XGBoost**, and
   should be named accordingly in articles.
3. **GRU-D-style** (Che et al., 2018): train-only channel means,
   last observation carry, time-since-observation, learned input and
   recurrent hidden-state exponential decay; an explicit input mask
   and original observation schedule. Locally trained 64-dimensional
   GRUCell, not an exact reproduction of the paper's released
   implementation or published score.
4. **Temporal Transformer** (nonconcept): standard 2-layer/4-head
   attention architecture, 64-dimensional hidden states. Its input
   includes the visible measured time series, both observation masks
   and time-since-last observation. Not a pretrained foundation model.
5. **ConceptFAN V10 with cross-view consistency**: exact frozen method
   from the prior PhysioNet V10 comparison: native 1h/4h/12h/24h
   physiological concept mixtures, separate mask-process concepts,
   physiological self-supervised denoising and future observed-feature
   prediction. The only concept architecture in the new study.

Three independent initializations (11/23/37) are trained for V10,
GRU-D-style and Transformer. Logistic/HGB are deterministic one-model
baseline fits: do not duplicate their result and call it multiple
independent model trainings.

For all neural methods: 18 epochs, common train-batch order,
batch size 128, AdamW 0.001, weight decay 0.0001,
train-only inverse-prevalence positive class weighting,
same available-signal dropout mechanism, no dev early stopping.
Logistic/HGB use one additional dropout-augmented training view per
TRAIN patient; duplicates **never count as additional patients**.

**Important**: V10 has extra self-supervised future/denoising targets
while external classifiers optimize the binary outcome only; this is
a legitimate comparison of published architecture *families* but not
a same-loss ablation. The measured future targets were available
only during V10 training, never used as its inference input. The
external results are not direct reproductions of previously published
48h/33-sensor PhysioNet GRU-D results because the 36h V10 protocol and
train-only sensor subset differ.

## Interpretation of resulting numbers

For each approach and additional observation-thinning scenario:

- mean across actually distinct model runs: AUPRC, AUROC, raw Brier;
- train seconds per model, measured CPU inference time per 300 cases;
- number of trainable parameters for neural models;
- stratified paired patient bootstrap with an independently resampled
  array of observed independent initialization seeds (800 draws)
  for each V10-minus-external AUPRC difference;
- run-level CSV and aggregate JSON for verification.

These **descriptive percentile intervals are not statistical proof**
of new test-population performance: model choice and hyperparameters
were chosen after many evaluations on the same 300 development cases,
and the clinical source remains one 2012 dataset.

Brier score uses the *raw* output of positively class-weighted models,
without learning a calibration mapping on independent data. It
diagnoses score calibration only provisionally; AUPRC is the primary
ranking metric and the clinic would need a separately calibrated
out-of-sample risk model.

The model-comparison objective differs from the concept-explanation
objective: a Transformer achieving a higher AUPRC would defeat V10's
claim of superior **prediction**, but cannot prove its nonconcept
features are more faithful, explainable or stable. Conversely, V10's
exact concept-logit decomposition does not prove the clinical meaning
of the discovered physiological slots.

## Run

```bash
PYTHONPATH=src:. python -m pytest tests/discovery/test_external_baselines_v10.py -q

PYTHONPATH=src:. python -m conceptfan_discovery.clinical_v10_external_benchmark \
  --config configs/medical/pd_v10_vs_external_20261103.yaml \
  --reference-v6-config configs/medical/pd_clinical_unknown_neutral_v6.yaml \
  --reference-v7-config configs/medical/pd_clinical_dual_process_v7.yaml \
  --reference-v10-config configs/medical/pd_clinical_v10_multiscale.yaml \
  --frozen-v10-receipt artifacts/pinned_V10/STATUS.json \
  --raw-zip data/physionet2012/raw/set-a.zip \
  --outcomes data/physionet2012/raw/Outcomes-a.txt \
  --out-dir artifacts/medical/v10_vs_external_only
```

Workflow: `.github/workflows/conceptfan-v10-external.yml`.
It automatically retrieves the verified frozen V10 development receipt
from run 41623853277 and executes all methods using the original
PhysioNet source.
