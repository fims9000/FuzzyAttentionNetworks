# ConceptFAN V16: executed native-training expert proxy feedback ablation

Executed: 2026-11-04. GitHub Actions [SUCCESS, 12 complete native V10 fits](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41822741792), artifact `conceptfan-v16-native-feedback-full-denominator-15` (artifact ID 13318465006). All source metrics copied from `AGGREGATE.csv` in the immutable CI artifact, not filled in from expectations.

**CI checkout HEAD**: `e178ecb0a5832d2e04d68296e0c3bf702b383fa8` (GitHub PR synthetic merge checkout); original research branch identified by PR #13.
**Artifact ZIP SHA-256 after download**: `cbd56f1dbd116283a02788e84a9d44657fdbd8911e71da0dbfc106f51edb1002`.
**STATUS.json SHA-256**: `eedafc0e5a6ff1f77d9f5f49bf9c685b634bb2e2afcc86bb0795011c524125ea`.

## Full-denominator aggregate (mean of 3 independently initialized fits per arm)

| Arm | Outcome AUPRC | Observable sensor-rule proxy AUROC | External synthetic latent-factor-0 absolute Spearman | Same-slot signed effect cross-run Spearman |
|:--|--:|--:|--:|--:|
| Original V10 | 0.510677 | 0.810956 | 0.378852 | -0.332414 |
| 4%-bridge only, no reviews | 0.510938 | 0.816239 | 0.387970 | -0.332326 |
| 4%-bridge + correct proxy reviews | 0.511326 | **0.926651** | 0.549046 | -0.333230 |
| 4%-bridge + all labels inverted | 0.505062 | **0.575369** | 0.061750 | -0.333748 |

12/12 model fits completed. Exactly 112 synthetic review intervals in correct/inverted arms, 0 in baseline/bridge arms. Training uses synthetic *observed-sensor* thresholds as reviewer surrogates, not clinician judgements. Validation episodes and oracle factor trajectories are never used for parameter updates or review selection; synthetic oracle factor 0 appears in evaluation only. Group effects were evaluated on the same validation episodes across seed pairs; there are only 3 independent model seeds, **not 3 statistically independent pair replicates**.

## Scientific interpretation

1. The extra 4%-softmax operator support alone changes observed-pattern AUROC by only ~0.0053. **Correct proxy feedback gives +0.1104 AUROC above the matching bridge-only control.** This confirms the *programmatic ability* of native expert-label supervision to alter the learned observable content of one V10 slot.
2. AUPRC of the actual prediction task remains essentially flat: correct proxy 0.51133 versus bridge-only 0.51094 (delta +0.00039), far too small for a superiority claim.
3. Full inversion of those same labels severely worsens measured pattern grounding. A consistently wrong reviewer is NOT rejected by the current workflow. Disagreement/unknown/artifact quarantining is a software safeguard, not a defense against systematic expert error.
4. **Primary limitation:** same-patient signed logit effect of that now more grounded slot is *negatively* correlated across independent model trainings (~-0.33), almost unchanged from baseline. Improved concept identity did NOT induce reproducible clinical decision contribution; the target hypothesis is therefore only partly supported.
5. The one observable proxy pattern is defined by an explicit threshold, not automatically discovered, and cannot establish physical or clinical meaning. True latent factors remain nonidentified without external evidence. Claim only synthetically observed pattern grounding.
6. Only one generator, one train-validation split, 3 initializations, and 112 simulated reviewer marks. No p-values or population confidence intervals, no independent hospital or actual doctor evaluation.

## Next scientific bottleneck

Do not add more source-covariance layers merely to inflate the architecture. Separate (a) verifiable **concept content**, (b) train-only independently supported **risk effect**, and (c) actual influence in the additive native prediction graph. Only present a concept as a risk explanation when all three have evidence; otherwise display the measured pattern as a grounded *observation* but abstain from giving it a medical causal risk name. Any future signed-risk orientation must be independently tested, not fixed by construction. A new concept addition mechanism should be tested only after this distinction is preserved.

## Exact provenance

Repository code: `src/conceptfan_discovery/experiment_expert_feedback_v16.py`; CI run link above. Machine-readable per-seed table `ALL_MODEL_FITS.csv`, aggregate `AGGREGATE.csv`, `STATUS.json`, and CI checkout SHA are retained in workflow artifact, with CSV aggregate also stored at `docs/medical/CONCEPTFAN_V16_EXECUTED_AGGREGATE_20261104.csv`. No patient-level private data exported.
