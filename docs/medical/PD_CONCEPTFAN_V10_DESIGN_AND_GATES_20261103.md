# ConceptFAN V10 — patient physiology across multiple causal temporal scales

**Status:** exploratory, separate from frozen PIERE / ConceptFAN original findings.
The accepted manuscript and old V7/V8/V9 evidence are not rewritten.

## Scientific question

Can a native, inspectable multi-timescale physiological concept improve
task quality, self-supervised representation sufficiency, and case-specific
explanation reproducibility, without the observation-policy shortcuts or
frozen-head artifacts previously observed?

The previous V7 physiological feature bank computes level, recent mean4,
first difference and deviation from recent mean4. V7 already learned a
sparse signed operator mixture and 8 train-rank-capped physiological slots,
but the temporal aggregation scale was fixed by a manually chosen formula.
V9 discovered a shared 4-D **mask-process** subspace, not four causal or
physiological phenomena, and had lower mean AUPRC than V7 in the previous
development validation. Thus simply adopting V9's source head into V10
is not scientifically justified.

V10 retains V7's exact separation between measured physiology and the
original recorded observation schedule. It changes ONLY physiological
concept formation.

## Intrinsic concept definition (not post-hoc)

Let `M[t,j]` be the original recorded mask and `x[t,j]` the
train-normalized value of sensor j. The network sees only past or present
measurements. For each named scale h in {1,4,12,24}, it computes an
observed-only causal mean. At any time when no observation in that
historical window exists, the scale receives a zero reliability flag.

The same predeclared **signed operator dictionary** (level, recent 4h
mean, first difference, deviation from recent 4h mean, plus signs) is
applied to each historical-scale view. Operator definitions remain
inspectable measured quantities, not anonymous hidden embeddings.

For physiological concept k in time step t and scale h:

```
r[t,k,h] = weighted observation reliability in the h-history view
z[t,k,h] = (sparse_signed_operator_sum[t,k,h]) / max(r[t,k,h], 0.2)
u[t,k,h] = 0.5 + r[t,k,h] * (
              sigmoid((z[t,k,h] - threshold[k]) / width[k]) - 0.5)
pi[k,h]  = softmax(scale_logits[k,:])[h]
C[t,k]   = gate[k] * sum_h pi[k,h] * u[t,k,h]
```

The training-dependent `scale_logits` are native parameters participating
in the downstream training outcome, observation reconstruction and forecast
losses. No one constructs concepts after training.

**Missingness contract:** if a channel has no historical observed
support, reliability is zero and that concept component equals 0.5
(UNKNOWN), NOT 0 (negative physiology). Original clinician observation
patterns are passed separately into V7's mask-only head. Artificial
training omissions never masquerade as recorded clinical measurement
decisions.

The task remains exactly additive:

```
logit(y) = b + sum_k v_physiology[k] + sum_j v_care_mask[j].
```

No unconstrained bypass classifier reads raw x or mask. The recorded
measurement process concepts are the V7 native head, trained on outcome
labels during the same optimization.

V10 additionally offers an optional native cross-view consistency loss,
only weighted where both independently thinned views retain sufficient
observed support; this is not a mechanism-identity theorem.
A small scale-entropy penalty (0.002) encourages more readable support
but does not impose a preferred hour or concept count.

## Controlled comparison

All methods are **refit** in one executable workflow (no comparison
between accidentally incompatible checkpoints):

| Method | Physiological concepts | Observation head | Consistency |
| --- | --- | --- | --- |
| `v7_retrained_baseline` | original V7 | original V7 | none |
| `v10_multiscale` | trainable 1,4,12,24h support | original V7 | none |
| `v10_multiscale_consistency` | same as above | original V7 | overlap-weighted |

Frozen original official Set A split, 2400 TRAIN, same 600 development
validation of which **300 previously inspected cases** are scored,
same deterministic train-only sensor pilot, spectral rank and capped
physiological K. Identical 18 epochs, AdamW learning rate, batch size,
initialization seeds (11/23/37), order seed (1001), training mask/dropout
and original disease-label/forecast/denoising objective in each arm.
Artificial observation-policy stresses: original, MCAR and
hour/channel-conditional MAR nested thinning (not new hospitals).

Evaluation reports:

- AUPRC full, physiology-only and sampling-process-only, plus AUROC;
- future observed-measurement RMSE and randomly masked observed-channel
  reconstruction RMSE using the SAME fixed evaluation mask;
- original-to-thinned prediction and native concept-contribution shifts;
- mean effective physiological concept count (to detect collapse);
- exact logit decomposition error;
- independent-seed signed case-specific Spearman **excess over
  permutation-of-patients null**, together with definition cosine;
- across-run concept alignment using BOTH signed operators and the
  temporal-scale mixture, not raw concept IDs or outcome labels;
- every concept's learned scale profile and exact measured-operator formula
  in a model-native passport without patient-level examples.

Three training seeds and overlapping model pairs are **descriptive only**.
No two-sided significance claim, causal effect claim, clinical concept
identity claim, claim of total physiological cause cardinality, or
cross-hospital transfer claim is allowed.

## Run

```bash
PYTHONPATH=src:. python -m pytest \
  tests/discovery/test_multiscale_physiology_v10.py \
  tests/discovery/test_observation_process_concepts_v7.py -q

PYTHONPATH=src:. python -m conceptfan_discovery.clinical_multiscale_v10 \
  --config configs/medical/pd_clinical_v10_multiscale.yaml \
  --reference-v6-config configs/medical/pd_clinical_unknown_neutral_v6.yaml \
  --reference-v7-config configs/medical/pd_clinical_dual_process_v7.yaml \
  --raw-zip data/physionet2012/raw/set-a.zip \
  --outcomes data/physionet2012/raw/Outcomes-a.txt \
  --out-dir artifacts/medical/v10_matched_development
```

GitHub Actions: `.github/workflows/conceptfan-v10-matched.yml`.

## Decision criteria

**Do not promote V10 solely because its AUPRC is higher in this already
viewed development cohort.** To merit a later independent replication, it
should:

1. retain AUPRC within a *proposed*, not retrospectively claimed,
   non-inferiority margin of -0.01 vs the *same-run V7 baseline*;
2. not worsen future and denoising error substantially;
3. improve case-specific physiological concept agreement above the
   patient-independent chance expectation without explanation collapse;
4. preserve the UNKNOWN/no-observation contract and exact decomposition;
5. demonstrate improvement on an independently sourced cohort
   (preferably new hospitals/measurements), not the repeatedly inspected
   historic validation.

If losses conflict, explicitly report the Pareto tradeoff; do not hide
negative arms. An increase in mask-process weight may indicate a
shortcut rather than physiological discovery.

## Next research gates

(a) Multi-hospital external validation with pre-registered outcome and
sensor alignment; (b) 10 init x 3 order seeds + patient/center-aware
uncertainty; (c) independent, variable-dimensional synthetic dynamics
where true latent states are accessible **only after training**; (d) fair
same-cohort plain Transformer / MAGNETS benchmark and missingness
controls. Present V10 as a possible *method*, not as an already proven
new clinical ontology.
