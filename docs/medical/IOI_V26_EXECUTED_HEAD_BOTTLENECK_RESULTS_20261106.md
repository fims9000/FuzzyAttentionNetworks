# ConceptFAN V26 — executed concept-head ablation (2026-11-06)

**Status:** SUCCESS, GitHub Actions [42113929392](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/42113929392), source commit `9cc7ec2526fbd3c349f3c6cc2c9b14cf8d58cc09`. [Artifact 13454264642](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/42113929392/artifacts/13454264642) preserves all 189 evaluations and 81 aggregated method/stress cells. **Post-V25 exploratory**: reuses the very same synthetic TEST; not independent prospective replication.

Re-fit 12 distinct canonical temporal ConceptFAN models (2 observable outcome laws × K2/K3 × neural seeds 11,23,37) using EXACT V25 trained-concept schedule. ARTIFACT reuses OMITTED models and reports, not independent fits. On each frozen learned-concept representation, fit a direct additive linear `b+Σbeta_k q_k` risk head using TRAIN labels. As controls: original fuzzy risk head, positive Platt calibration of fuzzy logits (cannot change rankings), logistic on TRUE oracle concept summaries, and logistic on measured raw history.

## Heldout TEST with MCAR20, mean over three NN fits

| Outcome family | Model | Original native fuzzy head AUPRC/NLL | Positive Platt AUPRC/NLL | Frozen concept-only linear refit AUPRC/NLL | True concept logistic AUPRC/NLL |
|---|---|---|---|---|---|
| COMPLETE | K2 | 0.6611 / 0.6727 | 0.6611 / 0.6472 | **0.7650 / 0.5785** | 0.7680 / 0.5786 |
| COMPLETE | K3 privileged targets | 0.5577 / 0.6883 | 0.5577 / 0.6742 | **0.7617 / 0.5795** | 0.7676 / 0.5776 |
| OMITTED/ARTIFACT | K2 | 0.6410 / 0.6704 | 0.6410 / 0.6374 | **0.7379 / 0.5893** | 0.7141 / 0.6120 |
| OMITTED/ARTIFACT | K3 privileged targets | 0.7163 / 0.6611 | 0.7163 / 0.6088 | **0.8066 / 0.5237** | 0.8127 / 0.5180 |

Raw-sensor logistic: COMPLETE **0.7601 / 0.5829** and OMITTED/ARTIFACT **0.8112 / 0.5212**. Source-reuse ARTIFACT gives *exact same observed outcomes*, never semantic physical identification.

**Mechanistic finding:** for this fixed 14-epoch recipe the fuzzy decision head is a substantial bottleneck; applying positive logit calibration improves NLL/Brier only, never AUPRC, whereas replacing ONLY the risk readout with a linear additive concept head improves ranking. The experiment does not establish that the fuzzy mechanism is inherently bad under adequate task-head optimization or a different training budget. The new head uses extra postfit optimization on the same TRAIN labels, so the comparison is **not compute/optimizer-matched**. A standard linear concept head is a known CBM baseline, not architectural novelty.

**Warning suggesting conceptual task leakage:** in omitted-factor K2, a linear head on the **learned** two concept summaries (AUPRC 0.7379) even beats the two **true** oracle concept summaries (0.7141). This alone does not prove leakage (finite samples, parameterization and partial learning confound), so we preregistered separate V27 fresh-seed testing of `Y|C*2` versus `Y|C*2,Chat2` with distinct joint versus staged concept-path training. Do not present V26 as automatic source discovery.

Exact additive head and Platt rank-preservation checked on every condition and all seeds. No PIERE accepted paper or PhysioNet clinical patient TEST changed.
