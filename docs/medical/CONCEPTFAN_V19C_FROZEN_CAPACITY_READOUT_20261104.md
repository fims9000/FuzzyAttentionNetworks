# V19c — train-only concept capacity test, frozen before execution

**Motivation:** V19 and V19b demonstrated that allowing the final decision to inspect learned native concept-trend/last/variability statistics raises AUPRC by only ~0.0003–0.001. That cannot explain the difference to simple logistic historical summaries. The remaining hypotheses include an overly narrow concept bank, saturating nonlinear concepts, feature-operator training and optimization rather than temporal *readout* alone.

## Question
Does increasing genuine **native concept capacity** from K3 to K5 (where K5 equals 5 recorded channels, not a claimed physical factor count) improve prediction, while retaining source traceability and exact additive native logit? No raw-input bypass or extra black-box head may be used. The two additional slots must not be falsely named biological concepts.

## Frozen 36-fit capacity/readout factorial ablation
Use the same already-inspected V18 synthetic task: generator seed 20411204, positive/negative/null outcome mode, 800 TRAIN, 200 heldout validation, 80 sealed test; three neural initialization seeds 11/23/37. Base model is first fit for six epochs on K2 sensor directions, exactly as V18b/V19b.

A third source is proposed **on TRAIN measurements only** using V17 residual lag. Its outcome sign is assessed on a disjoint TRAIN label subset exactly as V18b. For positive/negative outcome modes, the true TRAIN-screened risk orientation initializes the new unrestricted K3 native decision coefficient at ±0.05 raw. For outcome-null mode, this sign gate ABSTAINS. To study overfitting, **a separate explicitly unsafe forced-fit null control** is run with K3 and K5 readouts starting at 0; these are never treated as scientifically accepted risk concepts.

For K5, two additional observable sensor directions are computed from **TRAIN-only** SVD of residualized measurements orthogonal to the K3 measurement basis. They are inserted with zero logit, forecast and reconstruction coefficients through the same V17 native expansion path. Their statistical evidence is *not* screened; these K4 and K5 slots are overcapacity controls, **not automatic meaningful factor discoveries**. Always check exact equality of model logits and forecasts between K3 and K5 before subsequent training.

Four arms for each mode/seed, all continued for six matched epochs:
- K3 legacy final-six mean (same reference as signmatched V19b).
- K3 learned native temporal-shape mixture.
- K5 legacy final-six mean with 2 extra unscreened observable slots.
- K5 learned temporal-shape mixture with the same additional slots.

36 completed native model fits; all 36 outcomes, including negative/null, retained. Same optimizer, loss weighting, feature-gradient bridge, train examples, batch order, seeds and masking across arms. No validation-based hyperparameter selection. Original model from K2 pretraining reused per seed across the four arms.

Measure heldout AUPRC/AUROC/Brier, exact native additive decomposition, direct signed contribution of the third measured concept, mean absolute total new K4/K5 effect, raw concept-to-measured source alignment for the third concept, and cross-seed patient-level signed rank correlation. For learned temporal readout, also report nonrecent shape mixing.

## Falsifiability and limits
If K5 does not improve beyond K3, simply adding extra concept slots does not close the predictive gap (at least on this generator). If K5 helps, determine whether the gain is true heldout prediction rather than extra unscreened noise concepts; pay attention to null-label behavior and stability. K5 is not a clinical release and would violate the existing semantic-admission gate in live use.

All examples are completely observed linear synthetic histories. The validation cohort has been seen many times in prior experiments. This is an **exploratory diagnostic only**, not a new confirmatory holdout. Real medical expert corrections, novel physical identities and official MAGNETS comparisons are outside this experiment; those claims must not be invented. Accepted IEEE PIERE paper unchanged.
