# V20d — frozen independent *new test seed* evaluation of the V20c gate

**Predeclared before observing V20d metrics.** Critical provenance correction: V20c used synthetic TEST seed **305013** for each family, which was **already evaluated in V20b**. Thus despite TRAIN-only fit-disjoint gate data, V20c's TEST metrics do NOT constitute fresh heldout confirmation relative to previous research. Retain all V20c artifacts exactly and do not retroactively claim that V20c had independent new TEST outcomes.

## Frozen correction
Keep all V20c constants **identical**: same four structural simulator families, core TRAIN seed 305011 with 800 episodes, independently generated train-only acceptance GATE seed 305014 with 300 episodes, original K3 native V10 physical sensor slots, 12 joint epochs, all 3 model seeds 11/23/37, the same unweighted LBFGS native risk refit, **unchanged** one-sided permutation p<=.01 (499 permutations), mean gate logloss gain>=.005, 90% bootstrap CI gain lower>0 (499 resamples), no new classifier/path, exactly 12 fixed accept/refuse decisions and 36 model TEST arm evaluations. Same TRAIN-only medians/IQR, untouched operator/forecast/mask-process representations and original coefficients preserved on gate refusal.

Change exactly one factor: **synthetic TEST seed = 305115** (a previously unevaluated seed), n=300 per family. This is not a new *structural* data-generating family, only a genuinely unused TEST draw from the already inspected structural families. V20d code must make the new seed explicit, record it in the STATUS receipt, and mechanically confirm that no fitting/threshold selection uses V20d TEST outcomes.

Predeclared comparisons: Original vs Candidate Refit vs Selected-by-Gate TEST AUPRC/AUROC/Brier/logloss in each of four families; all 12 gate decisions and 36 TEST rows, including rejects. No criteria may be tuned after TEST results are seen. The gate's clinical validity and physical-source identity remain unproved. No significance claims from three dependent NN seeds.

This corrective protocol addresses a provenance flaw in the evidence trail. The old V20c results must not be overwritten or discarded; users must be able to audit the exact old vs new TEST seeds and distinguish them.

Accepted PIERE paper unchanged; PR remains Draft.
