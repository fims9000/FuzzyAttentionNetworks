# PD-ConceptFAN V7: separate measured-evidence versus care-sampling concepts

**Executed:** 2026-11-02.
[Original V7 dual-head CI PASS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41508831155),
artifact `physionet-separate-care-process-concepts-v7-1`.
Source is official PhysioNet 2012 Set A. Models fit only historical 2400
TRAIN records; the repeatedly used validation set is a **development**
benchmark, not an unseen confirmation. Calibration and test splits
remain unused in this exploratory protocol. Clinical concept dimension
identification remains explicitly **ABSTAIN_CLINICAL_K_UNSTABLE**.

## Rationale

The V6 model correctly treats completely unmeasured physiological
values as UNKNOWN, not a risk factor. However the **clinical measurement
schedule** can itself carry outcome information. Doctors measure some
patients more often. Reintroducing measurement frequency into the
*same physiological concept* would conflate an observation or treatment
policy with the actual patient state.

V7 instead creates two separately inspectable, jointly trained concept
families with a single **exactly additive training-time task logit**:

```text
C_phys[k] = gate[k] * (0.5 + reliability[k] *
                      (sigmoid((signed_value_operator[k]-tau[k])/width[k]) - 0.5))

O_process[j] = sigmoid((signed_measurement_mask_operator[j]
                        - tau_process[j])/width_process[j])

logit(Y) = bias + sum_k beta[k]*(C_phys[k]-0.5*gate[k])
                + sum_j gamma[j]*(O_process[j]-0.5)
```

The physical family consumes 12 measured observed channels, has
a train-rank-derived computational budget K=8, and participates in
future measurement prediction, masked denoising, **and** the task logit.

The observation-policy family consumes **only the observed/not-observed
mask**; the source operators are observation fraction per sensor,
recent observation fraction, time since last measurement and
measurement transition rate. Its three learned concepts have separate
fuzzy thresholds, widths, measured-mask operator definitions and
signed task coefficients. This family enters only the task logit.

The direct mathematical model intervention on either family can be
computed exactly, leaving the other family's contribution unchanged.
This does not identify a physiological or treatment causal effect.

**Critical control:** an artificially withheld training/validation
sensor measurement is NOT an actual clinician observation decision.
During denoising dropout the physiological family sees the artificial
mask, but the care-process family receives the ORIGINAL recorded
clinical mask. This makes the two source vocabularies explicit.
Nevertheless, physiological concepts remain dependent on availability
via their reliability, so this separation is **not complete causal
statistical disentanglement** or unbiased correction of MNAR.

## Original successful study: 3 initializations, real ICU validation

Same original source and deterministic train/validation/sensor-selection
rule as the prior V6 study, so the comparison is exploratory but
matched by case and seed.

| Source | Mean outer validation AUPRC |
| --- | ---: |
| V6 old uncorrected physiological concepts | 0.279848 |
| V6 neutral physiological concepts | 0.286336 |
| V7 physiological family alone (remove care logit) | 0.283102 |
| V7 care-process concepts alone (remove physiological logit) | 0.202106 |
| **V7 full dual-family logit** | **0.298550** |

Under additional **simulated MCAR** thinning, V7 full AUPRC is
0.263071, versus V6 neutral 0.238780. Under time/channel-specific
MAR thinning V7 AUPRC is 0.286363, versus V6 neutral 0.282560.
No claims of external site generalization are possible.

For the original validation observations, observation policy effects
accounted for 0.307135 of the *absolute sum* of direct task-logit
concept contributions on average. This value is NOT a causal
fraction of patient outcome and not a Shapley allocation of the
joint outcome; it is an accounting share **within the model**.

Under additional artificial observation thinning the process concept
contributions remained EXACTLY unchanged (the same original clinical
schedule is supplied); physiological contributions changed by
approximately 0.158351 (MCAR) or 0.159383 (time/channel MAR)
in absolute task-logit units per patient.

Machine-precision additive error was below 1e-7 in all scenarios.

## Real source-linked, trained concept interpretation examples

The first-init process concept 0 has a positive direct logit
coefficient +0.5387 and a mask-only operator definition emphasizing:
- negative observation fraction of RespRate: coefficient 0.2831;
- negative recent observation fraction of RespRate: coefficient 0.2709;
- positive time since last RespRate measurement: coefficient 0.2064.

Process concept 1 has signed logit coefficient -0.3267 and
mask-operator mixture including recent NIDiasABP, DiasABP and
urine measurement frequency.

One process concept has a near-zero signed decision coefficient,
around -0.0009, and must NOT be described as a material contributor
merely because it is an active learned slot.

The physiological (measured-evidence) concept 0 has direct signed beta
+0.8665, uses signed operators including GCS recent 4-hour mean
(negative), MAP level (positive), NISysABP level (positive),
and NIMAP level (positive). Exact definition is printed in the
artifact; do not assign a diagnosis name without external validation.

All example coefficients are learned in training. The concept
passports deliberately omit individual patient IDs and explanations.

## Evidence versus proof

**Shown:** fully operational no-posthoc explicit dual-head model;
reproducible direct signed influence;
mask-only process operators do not inspect measurement values;
no measured values means zero *physiological* direct effect;
sample-frequency concepts can help an observational mortality
predictor on reused validation.

**Not shown:** independent transport to another hospital,
true clinical semantic labels, clinical treatment effects,
lack of latent confounding by measurement policy, robust
medical concept K or statistically significant AUPRC gains
under patient/center/generalization uncertainty.

**Important tradeoff:** a care-process head can learn an
undesirable, site-specific shortcut. The pure physiological head
must be evaluated separately from full, and a real multi-hospital
or observation-policy shift test is required before deployment.
No acceptance criteria should be changed after seeing the outcome.

## Reproduction

```bash
PYTHONPATH=src:. python -m pytest \
  tests/discovery/test_observation_process_concepts_v7.py \
  tests/discovery/test_unknown_neutral_v6.py -q

PYTHONPATH=src:. python -m conceptfan_discovery.clinical_dual_process_experiment_v7 \
  --config configs/medical/pd_clinical_dual_process_v7.yaml \
  --reference-v6-config configs/medical/pd_clinical_unknown_neutral_v6.yaml \
  --raw-zip data/physionet2012/raw/set-a.zip \
  --outcomes data/physionet2012/raw/Outcomes-a.txt \
  --out-dir artifacts/medical/dual_family_v7_exploratory
```

The accepted original PIERE analysis and its historical model
checkpoints remain unchanged. The V7 work is in an unmerged
research draft PR.

## Cardinality and identifiability disclaimer

**Neither the eight measured-evidence slots nor the three care-process
slots are a confirmed number of genuine physiological/healthcare mechanisms.**
The eight physiological slots came from train-only spectral rank,
*explicitly capped at eight*, despite the failed previous independent
clinical dimension-invariance gate. The three care-process slots are
a **manual architectural capacity hyperparameter**, not data-derived
latent-source cardinality. Each slot's signed operator, training-time
threshold, width, direct coefficient and per-case contribution are
automatically fit, but the number of sampling families is NOT selected
without the researcher.

Consequently V7 solves an interpretability **source-accounting problem**
and improves exploratory mortality prediction; it does not yet solve
the original broader problem of discovering the unique ontologically
correct number and identities of all clinical concepts. The next
scientifically defensible experiment requires a patient-disjoint
train-only procedure for care-process slot count and genuinely
independent observation environments/clinical truth. Tuning the
existing historical validation to favor three slots would be
selection leakage, not a solution.
