# ConceptFAN V19d — frozen diagnostic: concept representation sufficiency versus risk-readout optimization

## Hypothesis after V19a–c

The original V10 model computes signed sensor level, smoothed values, local change and deviation, yet its exact additive task readout pools K3 native concepts into three scalar summaries. V19a learned a shape-aware decision summary but improved AUPRC by <=~0.0011. V19b with independently SCREENED risk sign showed the same; V19c adding two unsupported sensor-derived slots K3→K5 improved one outcome by ~0.002 and degraded the other. All of these remain far below the already run logistic regression with explicit temporal measurement features.

**New diagnostic question:** Once the native ConceptFAN has been fitted, do its K3 task concept summaries contain enough information for a simple, *independently refit* logistic decision head to match the predictive baseline? If the probe improves heldout AUPRC substantially, the bottleneck may preserve useful information but the neural task objective/readout optimization is inadequate. If the probe also underperforms, some critical task signal has been lost in the native concept trajectories/operators.

## Frozen protocol (before looking at new probe results)

Same *already inspected* synthetic V18 dataset family, no test-set unsealing: generator 20411204, 800 TRAIN / 200 validation / 80 sealed test; three labels modes positive, negative and outcome-null. Three neural seeds 11/23/37. The V18 train-only unsupervised K2→K3 candidate and independent TRAIN outcome risk screen are unchanged.

For every mode and seed, pretrain common original K2 native model six epochs. Admit K3 from only TRAIN-observed residual measurements. In positive/negative modes start the new native decision coefficient with independently screened signed raw ±0.05; in outcome-null mode explicitly force a zero new risk coefficient for a **nondeployable, overfitting negative control** even though the proper risk screen ABSTAINS. Continue for six matched epochs with two native readout variants:
1. `legacy`: original K3 recent-mean task concept summaries;
2. `learned_shape`: V19 K3 learned signed native shape summaries.

Then **freeze each fitted neural model**, set eval mode, and extract ONLY its own physical concept summary vector (K=3) on fully observed TRAIN and validation past histories. Fit a separate StandardScaler + ordinary LogisticRegression(C=1,lbfgs) to the TRAIN summaries and TRAIN binary labels. This probe has **no access to raw sensor histories, future observations, masks, latent hidden factors or test IDs**; a model with identical concept summaries but different task-readout training can be compared. The probe is **diagnostic, post-training** and not a deployable alternative ConceptFAN architecture; its fit is allowed additional supervised optimization and does not preserve the original neural readout's fixed sign constraints.

Evaluate native vs probe AUPRC/AUROC/Brier on the same heldout validation examples. Report all 3 outcome modes x 3 neural seeds x 2 arms = 18 native fits + 18 train-only logistic probes, no hidden failed runs; report mean train/validation losses and gap to prior V18c raw-feature LogisticRegression. No tuning on previously inspected validation, no significance with only three neural seeds.

## Interpretation guards

If probe > native, it does not imply conceptual meaning, causal correctness, or clinical suitability. A logistic probe is a possible way to **diagnose** underfitting of a linear native readout, not an independently learned concept model. The data already had repeated validation inspections, so this is descriptive only, not out-of-sample claims beyond the frozen generator.

If probe still trails direct input-feature logistic, the next architectural direction is improving native concept **formation and grounding**, rather than adding more task-side readout complexity. Verify label leakage boundaries, patient mask neutrality and direct causal-past input constraints in all versions. Original accepted PIERE article unchanged.
