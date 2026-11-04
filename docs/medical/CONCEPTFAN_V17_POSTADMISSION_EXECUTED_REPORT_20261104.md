# ConceptFAN V17 — executed actual native retraining after a discovered concept is admitted

Source run: [GitHub Actions 41824532541 SUCCESS](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41824532541); 9 fitted models; archived artifact `conceptfan-v17-postadmission-native-2` (ID 13318761602). The zip includes all 9 individual rows, aggregate, STATUS.json, and CI SHA.
CI checkout SHA: `9b2b958f0b4549fac3bcc7c724c98c41f31bb2db`.
Downloaded archive SHA-256: `0ccad8e86063da9c28ae822aaf3e503802107ca76ccf26f2c75e9d9042a8931c`.

**Frozen synthetic positive control**: generator seed 20311104; 600 independent TRAIN, 200 independent VALIDATION, 40 sealed test episodes; 5 observed channels and 3 planted dynamic factors. An initially unmodeled third factor drives the future-label distribution. K2 contains only the first two known sensor directions. The candidate is found from TRAIN observed residuals, not source-oracle labels; a programmatic approval admits the resulting new native slot, initially with exactly zero decision, forecast and decoder coefficients. Risk coefficient is learned only upon continued native training. No clinician participated.

| Model arm | Mean heldout AUPRC (3 seeds) | Average absolute new native logit effect |
|---|---:|---:|
| Continue trained K2 capacity | 0.555271 | N/A |
| **Grow K2→K3; native retraining** | **0.608307** | **0.027086** |
| K3 trained from scratch for total same 12 epochs | 0.556935 | 0.091153 |

All individually seeded results must be reported (see per-seed CSV in the repo). There is large and meaningful variance across initializations: for seed 11, K3-from-scratch achieved **0.73793**, outperforming growth **0.53103**; for seeds 23/37, growth outperformed scratch. Therefore this is not proof of generalized optimization superiority.

**Observed positive-control findings:** the added slot remains in the exact additive native graph, has learned a nonzero logit influence, and improves heldout AUPRC over the matched continued-K2 arm in all three tested initialization seeds; mean delta +0.05304. In this deliberately constructed setting, the capacity growth can capture task-relevant information that K2 cannot access through its existing physical concept basis. It is not a new semantic source-discovery theorem and does not prove general benefits across datasets.

**Primary limitations**: the model was deliberately restricted to two sensor sources while label depends on a third, so this is an intentionally favorable control. The full native model has an observation-process head but no direct raw sensor bypass. Same epochs does not mean exactly identical optimizer trajectory: growth restarts its optimizer, whereas scratch trains from an independently initialized larger model. Three seeds, one synthetic generator, no actual doctor, no medical validity, no formal significance tests, and no source/noise identification. The earlier falsification proves colored sensor errors can create exactly the same candidate IDs as genuine factors.

**Next scientific requirement**: make admission contingent on separately established *task relevance and measurement provenance*, then test many independent data-generating families, observation masks, external expert verification, matched official automatic concept baselines, and reproducibility of signed effects. Do not add these results to accepted PIERE camera-ready.
