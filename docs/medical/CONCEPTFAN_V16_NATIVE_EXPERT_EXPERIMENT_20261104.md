# ConceptFAN V16 — native expert-feedback ablation, frozen before its first GitHub Actions result

## What changes in the architecture
Based on V10, a reviewer marks a given *existing* concept slot in an observed history interval yes/no/unsure/artifact, without entering equations. Feedback regularization acts on the same native differentiable raw concept activation used by the additive prediction bottleneck, the reconstruction decoder, and forecasting. It is NOT a post-hoc explainer. The clinician observation-process head stays separate. The default V10 forward, output, and training modes are unchanged when expert_feedback=None.

## Explicit assumptions and limitations
- This first experiment uses a deterministic, observed-sensor **proxy for an expert** (high vs low recent value in sensor_1) on fully synthetic variable-factor trajectories. No real clinicians, medical notes, expert-validated diagnoses, clinical patient records or real physiological semantics.
- The reviewer labels an **existing** native concept slot chosen using the TRAIN-only sensor loading. This is NOT new automatic concept invention, nor semantic factor identification.
- The synthetic generator has K=3 hidden factors; the native computational capacity is **prespecified at 4**, not secretly copied from generator K. A TRAIN-only measurement SVD defines the basis, and TRAIN-only observed sensor_1 selects a slot.
- The artificially selected reviewer question is about a measurement-level pattern. It is not an oracle label for a latent physiological source; ground-truth latent trajectories are never sent to the optimizer or review procedure.
- Model predictions remain entirely through a native, exactly additive concept bottleneck. No separate student or post-hoc concept predictor.
- Wrong expert labels remain an intentionally adversarial comparator: bounded gradient is not robust statistical estimation.
- We use a distinct synthetic generator seed. No PhysioNet Set B validation, previously consumed hospital holdout, or original PIERE result is modified.

## Frozen exploratory protocol
Source: src/conceptfan_discovery/experiment_expert_feedback_v16.py.
- Generator seed 20281104, split seed 20300419, N=960 generated, 8 sensors, 36 history / 6 future, train/validation and a separate sealed test split.
- Same V10 physiological and observation-process architecture, train-derived PCA/SVD basis K=4, scales 1/4/12/24, training 12 epochs at fixed learning rate, batch size 128, 3 initialization seeds 11/23/37, and cross-view consistency enabled for all variants. The baseline uses original sparsemax; the three exploration arms share an explicit 4% softmax bridge.
- **Four arms**: exact V10 baseline; 4%-softmax gradient-bridge only (no labels); bridge plus up to 112 balanced *correct* simulated answers; bridge plus the same answers deliberately inverted. The bridge-only arm isolates the architectural change from supervision; added BEFORE any successful V16 nine-fit CI, in response to an observed unit-test sparsemax zero-Jacobian failure. Maximum 56 positives + 56 negatives; the availability condition is >=4 out of 6 observed history steps and abs(normalized average)>0.12. All labels are computed on TRAIN input measurements only.
- The expert regularization strength is set to 1.0 before looking at results; 4% smooth support permits concept-feature gradients at exact sparsemax one-hot initialization and is NOT claimed as novel mathematics.
- Twelve model runs including all failures/rejections. Evaluation on a disjoint, untouched synthetic validation set uses original outcome AUPRC/AUROC/Brier, a predeclared sensor-pattern AUROC/AUPRC, evaluation-only latent factor 0 Spearman, additive logit audit, and raw signed effect cross-seed consistency (three dependent pairs, no pseudo-independent CI).
- Report all arms and all three seeds, no cherry-picking. Aggregates only in public Actions artifact; no individual review, patient, or raw trajectory outputs.

## Scientific decision criteria
1. **Architecture validity**: expert feedback backpropagates to the native concept generator; ablation with feedback disabled exactly follows V10; no hidden classifier bypass; unavailable measurement never becomes a negative physiological label; train-only review IDs enforced.
2. **Grounding effectiveness (hypothesis H1, exploratory)**: positive feedback increases heldout agreement with the predeclared observed sensor pattern compared with no reviews. Report failures transparently.
3. **Adversarial response (hypothesis H2, exploratory)**: inverted feedback may lower pattern agreement. Show whether label disagreement, unreliability and contradiction are quarantined; do not claim robust protection against a consistently wrong reviewer.
4. **Predictive tradeoff (hypothesis H3, exploratory)**: compare AUPRC, calibration and same-slot signed-effect stability; no effect is significant with only three seeds or this single synthetic dataset family.
5. **Future, not claimed here**: genuinely new slot discovery, external expert annotations, uncertainty-aware new concept admission, replicated factors with non-AR/nonlinear dynamics, real multi-site clinical records, independent physical calibration.

## Publication boundaries
This is an architecture-training experiment, not an IEEE PIERE camera-ready amendment. It should not be merged into the accepted paper or summarized as a demonstrated medical concept discovery method. V13–V15 noise-identifiability counterexamples remain valid and should motivate future falsification controls.
