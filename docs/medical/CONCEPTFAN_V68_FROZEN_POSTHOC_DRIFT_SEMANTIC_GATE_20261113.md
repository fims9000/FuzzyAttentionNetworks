# V68 frozen *post-V67 exploratory* drift/semantic identity abstention audit

**Authored 2026-11-13 AFTER V67 UCI270 batch10 TEST results were seen; no new sealed TEST exists.** This is a scientifically honest retrospective diagnostic of source-identification reliability and potential abstention, not independent prospectively validated recall/specificity or a new physical semantics certificate. Do not tune the rule on batch10.

## Problem

V67 generated four unlabeled statistical gas-response axes from UCI270 batches1–7, then used **both** earlier batches8+9 to match those axes to six physical gas names, yielding high VALID AUROC but ethanol/ethylene associations collapsed on later batch10. Need assess whether (i) anonymous source subspace remains stable under leaving out past batches, and (ii) matching a gas name based on ONE earlier drift period can be verified against a SECOND independently recorded period *before* seeing the much later batch10. A physical source identity still requires independent chemical reference and causal/controlled exposure evidence; this rule is ONLY a descriptive applicability gate.

## Locked unchanged base experiment

Official original UCI270 zip SHA256 `98fe3a30981a222dd4518fbcc3dddd45d5c0ce9b03ef6dc6fe5cf7a04cfbff5e`. Same static source extractor `FrozenSixGasSourceBank`, fixed PCA95 K cap12, FastICA seed20261967, no reference gas labels in source fitting, unchanged 9,546 batch1–7 TRAIN records. New procedure NEVER changes the ICA K4 model, fuzzy memberships, or V67 gas classifier; no gas labels in source rank calculations.

## Predeclared *retrospective* controls using TRAIN and VALID only

**Control A — original TRAIN-only leave-one-batch-out source stability:** Fit the identical frozen source algorithm seven more times, each time omitting ONE of batches1–7, refitting normalization and ICA on remaining batches only. Report each K, convergence, and geometric overlap of transformed anonymous source subspaces on the SAME unlabeled TRAIN probe. Compare centered score matrices Q0,Qj via QR/SVD; squared canonical correlations are invariant to axis permutation/sign. No gas label can influence it. If ranks vary, record; do not force K4. Do not mistake subspace stability for chemical identity.

**Control B — two chronological physical-label audits:** Freeze the original TRAIN1–7 source first. Use only **batch8** gas labels to match K anonymous axes to six gas labels via existing Hungarian max absolute OvR AUC separation. Hold fixed each axis and AUC sign; evaluate unselected **batch9** gas labels before testing batch10. A gas mapping can be called `EARLY_VALID_ASSOCIATION` only if:
- gas has >=**20 independent exposure records in EACH** batch8 and batch9;
- the oriented OvR AUC >=**0.80** in batch8 and batch9;
- absolute between-batch oriented AUC difference <=**0.15**;
- source rank K is identically the same in **all seven** TRAIN leave-one-batch-out source fits (strict global source stability prerequisite).
Otherwise abstain with explicit support, signal, or rank failure reason. No postfit source K increase, no score mapping selected based on batch9 or batch10.

**Evaluation:** For every frozen batch8-class mapping, whether admitted or abstained, report original oriented batch8, batch9 and batch10 AUROCs (and a fixed-seed batch10 shuffled-label negative control), but batch10 is already observed from V67 and cannot be used for model selection nor claimed as a sealed prospective validation. Report full gas-class counts in each period and which conditions are actually met. For `EARLY_VALID_ASSOCIATION` classes also report how many class memberships remain separable in batch10 at descriptive threshold .75 — but this threshold and all V68 gates are post-V67 research hypotheses, NOT a newly preregistered external validation.

**Control C — unlabeled marginal drift:** On original TRAIN-fitted fuzzy memberships, compute mean absolute batch8/batch9/batch10 state shift relative TRAIN source means; compare against semantic AUC failures. A low marginal state shift can coexist with conditional class correspondence failure, so global shift monitoring alone cannot certify semantic validity.

## Success and honest failure conditions

An early gate that abstains on all classes is a valid conservative result, but **not** proof of reliable calibrated clinical/source semantics. A gate that admits a class with batch10 AUC near .5 is a documented **false sense of semantic stability**; report it. A fully stable source subspace does not mean axes are physically unique, and validation labels still supply their chemical names. UCI270 is one apparatus with six *pure-gas* exposures, not independent multi-gas mixtures.

Keep accepted IOI/PIERE articles and master unchanged. Retain original V67 negative evidence and all V68 retrospective limitations.
