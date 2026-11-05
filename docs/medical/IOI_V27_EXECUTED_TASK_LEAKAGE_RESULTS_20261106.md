# ConceptFAN V27 — executed fresh-draw task leakage experiment (2026-11-06)

**Verified:** GitHub Actions [42114705006](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/42114705006) **SUCCESS**. Source commit `73a43d5906f1420bbcd29f8e7635528c1fbd17d7`; [artifact 13455403570](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/42114705006/artifacts/13455403570) has full STATUS / SUMMARY / LEAKAGE JSON, smoke and full logs. The [frozen V27 protocol](./IOI_V27_FROZEN_CONCEPT_LEAKAGE_PROTOCOL_20261106.md) was committed before training, result examination or TEST generation.

## What was independently redrawn

A NEW set of 1600 TRAIN / 400 UNUSED CAL / 800 TEST independent synthetic episodes was generated using seed **20261107**, from the **same AR-factor simulator family already inspected in V25/V26**. This is a new random draw but not a new hospital, mechanism family or external validation. There are 16 timesteps, 6 recorded sensors and true K=2 named concept trajectories. Outcome laws: COMPLETE (only two concepts predictive), OMITTED (a third hidden source also predicts outcome), and observationally **identical** ARTIFACT interpretation of the omitted case (same sensor values, outcomes and fitted models). All source labels are synthetic, not validated medical diagnoses.

Three neural initialization seeds 11,23,37. Two canonical K2 native training arms: JOINT (BCE +5 MSE on concept targets) versus STAGED (5 MSE ONLY, **no outcome-label gradients into concept formation**), same initialization, 14 epochs, LR, 1600 episodes and SGD batch order, followed by identical TRAIN-only additive linear decision head refit in both arms. Every fitted model's final logit is exactly bias plus 2 signed concept contributions. The label-free stage trained on identical x has **exactly identical parameter SHA** across COMPLETE/OMITTED for each seed, confirming outcome independence of stage one.

## Primary fresh TEST/MCAR20 performance

| Outcome law | Measured raw logistic | Oracle true K2 logistic | Oracle true K3 logistic | JOINT (2 concept model + linear head) | STAGED (2 concept model + linear head) |
|---|---:|---:|---:|---:|---:|
| COMPLETE | 0.7506 | 0.7489 | 0.7483 | **0.7371** | **0.7362** |
| OMITTED/ARTIFACT | 0.8041 | 0.7362 | 0.8136 | **0.7457** | **0.7279** |

Values are **AUPRC** on 800 TEST records; network methods are arithmetic means of three fits on the SAME records. NLL (COMPLETE) JOINT 0.5904 versus STAGED 0.5911. NLL (OMITTED) JOINT **0.5936** versus STAGED **0.6093**. Difference is a real ranking/decision tradeoff at this generator and training budget, not evidence of a superior medical model.

Concept trajectory reconstruction MSE (MCAR20):
- COMPLETE: JOINT 0.006883, STAGED 0.005987;
- OMITTED: JOINT 0.007213, STAGED **0.005987**.

With an omitted outcome-relevant factor, the outcome-gradient-sensitive concept formation improves risk AUPRC by **+0.01786** relative to staged training but increases concept reconstruction MSE by about **0.00123**.

## Conditional task leakage: exact positive and null controls

Fit logistic probes on TRAIN only; evaluate once on fresh TEST for incremental NLL improvement of `Y | true C*_1,C*_2, learned Chat1,Chat2` versus `Y | true C*_1,C*_2`. A positive holdout delta means learned concept coordinates retain **additional predictive information that is not present in the prescribed two true concept summaries**. It is *not* a proof of a causal latent source or physical identity.

| World | Concept training | TEST gain in NLL, seed11 | seed23 | seed37 |
|---|---|---:|---:|---:|
| COMPLETE | JOINT | −0.00435 | −0.00590 | −0.00314 |
| COMPLETE | STAGED | −0.00492 | −0.00459 | −0.00326 |
| OMITTED | JOINT | **+0.01356** | **+0.03904** | **+0.04340** |
| OMITTED | STAGED | −0.00133 | +0.00059 | −0.00107 |

For all three JOINT/OMITTED fits paired TEST-subject bootstrap 95% intervals EXCLUDE ZERO on the positive side: seed11 [0.00562,0.02168], seed23 [0.02008,0.05950], seed37 [0.02389,0.06114]. All three STAGED/OMITTED intervals include zero: seed11 [−0.00305,0.00045], seed23 [−0.00102,0.00231], seed37 [−0.00310,0.00085]. The TEST patients are shared across fit seeds; these are *not* 3 independent cohorts or proof of zero leaked information in STAGED.

Independent oracle-audit corroboration (not provided to the model for fitting): largest absolute Pearson correlation between learned concept residual (after TRAIN-based regression on true C*2) and the true omitted C*3 on TEST. JOINT/OMITTED values **0.288/0.692/0.772** versus STAGED/OMITTED **0.090/0.047/0.052**. In COMPLETE, both arms have low values ~0.05–0.09. This is suggestive of hidden prognostic-source encoding during joint training, even when the concept trajectory MSE is low.

## Scientific conclusion and limits

**Empirically supported, for this preregistered generator:** Direct task gradients into named concept units improve ranking partly by allowing them to carry additional information outside the intended true K2 concept definition. Splitting concept training from task training greatly reduces the measured conditional predictive excess for these cases, but loses some target AUPRC when the concept dictionary is incomplete. Thus concept faithfulness/meaning and predictive sufficiency are distinct acceptance criteria, not interchangeable numbers.

**Not supported:** Automatically discovered or identified physical causes, semantic completeness on real ICU data, guaranteed absence of leakage at STAGED, clinical usefulness, or superiority over published CBM methods. STAGED Concept Bottleneck Models are existing methodology; the contribution at this point is controlled falsification and a potential distinct journal problem, not architectural invention. Extra factors and predictive instruments remain observationally indistinguishable. Only 3 NN fits and one pre-inspected simulator family; no new external mechanisms, no patient data. Same number of neural gradient steps and identical second-stage solver, but different label gradient exposure and FLOPs because that is the intervention.

Unlike PIERE-2026 (cross-retraining signed contribution reproducibility) and IITI-2026 (concept-latent alignment), this tests **task-induced contamination of concept meaning and conditional dictionary completeness**; do not recycle earlier figures/tables as new.

Reproduce from pinned branch with:
`PYTHONPATH=src:. python scripts/experiment_concept_leakage_guard_v27.py --out-dir V27_NEW_DRAW_EVIDENCE`
