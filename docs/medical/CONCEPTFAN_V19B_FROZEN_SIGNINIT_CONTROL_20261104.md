# ConceptFAN V19b — predeclared sign-initialization-controlled temporal readout experiment

## Why an extra control is necessary

V19 completed 27 full-from-scratch K3 fits using old mean-last-six, fixed signed temporal-shape mixture, and learned temporal-shape mixture. The learned arm improved heldout AUPRC by only ~0.0011/0.0002 in positive/negative synthetic outcome modes, while per-patient signed contributions remained negatively correlated across retrainings (~-0.33). Those runs used *arbitrary random initial risk coefficients*.

Previous V18b established, on a related but differently initialized K2→K3 training protocol, that correct independent TRAIN-only **risk sign initialization**, not sign constraint, can substantially improve AUPRC and cross-seed contribution agreement. Therefore it would be an error to attribute all V19 failures to temporal pooling without holding this optimizer confound constant.

## Frozen V19b design

The original V18 synthetic generator is used AGAIN (20411204), with **already inspected** 200-patient validation, 800 TRAIN, 80 sealed test, three future-outcome modes (positive association/negative association/null), three model seeds (11,23,37). This is strictly exploratory, not a new confirmatory dataset or a hospital validation.

For each outcome mode, use the same first 350 TRAIN histories for unsupervised K2→K3 source proposal, and the next disjoint 300 TRAIN examples for a binary outcome screen and sign selection (same fixed V18 test: label permutation, bootstrap, two-split agreement). Null mode should trigger ABSTAIN. Model fitting may use all 800 TRAIN examples after the screen.

For each initialization seed, **fit the same legacy K2 V10** for 6 epochs, admit a new K3 physical concept with its source basis, and set the same *unrestricted* native decision weight to the correct TRAIN-screened initial raw value ±0.05 before continuation. No hard sign constraint. Then copy all pretraining parameters into three readout arms:

1. `legacy_signinit`: original V10 task readout, last-six mean.
2. `fixed_shape_signinit`: eight signed temporal summary features with initial readout logits frozen.
3. `learned_shape_signinit`: exactly the same initial shape features/weights, but trainable.

Check that arms 2 and 3 have exactly identical logits before retraining on the same input histories (no validation labels used for this identity audit). Continue **all** accepted models six epochs, same loss, optimizer, train order, mask RNG and hyperparameters. There are three modes × three arms × three seeds = 27 possible entries. Correct/noisy labels never feed shape mixture directly. In null mode all risk-oriented arms ABSTAIN and are counted rather than assigned zero AUPRC.

## Metrics and decision rule

Report all per-seed AUPRC, AUROC, Brier; exact native additive decomposition, mean abs direct contribution from new concept, raw native concept versus observed signed direction, signed patient-effect cross-seed rank Spearman, and learned shape weights. All cases and abstentions in CSV/JSON and immutable GitHub Actions / branch. No p-values for 3 seeds.

If learned and fixed shape converge to nearly identical scores, task-side readout flexibility is not providing substantial additional predictive information, at least on this dataset; investigate the limited number/quality of concept channels or optimization. Even a large gain would not prove physical meaning or general accuracy advantage to independent logistic feature baselines.

Original IEEE PIERE camera-ready unchanged.
