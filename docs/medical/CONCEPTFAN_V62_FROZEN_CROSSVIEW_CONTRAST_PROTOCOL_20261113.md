# V62 protocol: unsupervised cross-view response concepts (post-V61 exploratory)

Registered 2026-11-13 after V61 TEST results were inspected. **No V62 results had been computed when this protocol was committed.** This is a hypothesis formed from a prior negative experiment and tested on the SAME UCI gas recordings: results can show engineering feasibility but cannot be called independent confirmation or a new sealed TEST.

## Scientific issue

V61 physically manipulated two gas plumes independently but detected predictive lag rank K=1, conflating observable temporal predictability with physical-source count. Its one-dimensional concept had lower AUPRC than raw sensor logistic for CO and methane. The next method must generate additional concepts with NO gas identity labels, rather than setting K=2 because we already know the experimental sources.

## Frozen train-only candidate construction

- Use **exact same 180 genuine UCI 309 downsampled real experiments and V61 run-level split** (TRAIN120, VALID30, previously inspected TEST30); no windows from a run are treated as independent subjects for K fitting or evaluation.
- Only unlabeled TRAIN eight-sensor values, no gas identities, flow levels, concentrations, filenames or outcome labels. TRAIN sensor normalization remains exactly V61 median/IQR.
- The public physical protocol defines a 0–60 s clean-air baseline, 60–240 s exposure, 240–300 s recovery. This *timing* is known a priori and is available to all comparative methods, not the identity of gas or individual measured concentration.
- Form **two independent temporal views** of the same episode's response, with independent baseline windows: view A = average sensor response during 60–150 s minus average during 0–30 s; view B = average during 150–240 s minus average during 30–60 s. All values are from one TRAIN physical run. No target gas label enters.
- Compute centered TRAIN run-level 8×8 cross-view covariance C = A^T B / (N-1), and its SVD. Determine K by counting singular values greater than a fixed **familywise 95%** null maximum singular value formed by independently permuting the order of WHOLE TRAIN runs in view B (100 null permutations), and report a 20 whole-run bootstrap K interval. No manual K, label-dependent rank or TEST threshold.
- Loadings = canonicalized left-singular sensor directions for the selected predictive cross-view components, never given chemical names. Calibrate per-axis fuzzy location/scale on all TRAIN actually observed projected values (median and IQR/1.35, min width=.15). Use the fixed V54 masked observable projection semantics. No source can be identified as a pollutant from an axis alone.
- Native V56 causal fuzzy temporal encoder is trained only on three non-overlapping 20-step crops per original TRAIN run as **dependent unlabeled augmentation**. Network seed11, 8 epochs, 20% observed-cell dropout, denoising+forecast. Freeze before separate supervised head fitting. 120 independent run labels per task, never 360 artificially independent patient labels.

## Frozen evaluation and baselines

- Three V61 real physical exposure tasks ethylene_present, co_present, methane_present; same original TRAIN labels and run splits. Evaluate both VALID and (already viewed in prior V61) TEST, always explicitly calling these **exploratory post-selection** due to reused records.
- Compare V61 original positive-lag K1 V56 native and V55 static, V62 cross-view static and V62 cross-view native, raw sensor logistic/PCA, and exposure-aware logistic. Label and primary physical measurement budgets equal. Hyperparameters and K chosen from unsupervised TRAIN before labels.
- Report run-level AUPRC, AUROC, NLL, Brier; source rank and full shuffle null, TRAIN whole-run bootstrap support; concept versus physical gas level full K×3 Spearman (exposure-minus-baseline) and permuted-condition negative controls. Do NOT name individual statistical axes by gas except as posthoc measurement associations, never by task-head coefficients.
- Two specific controls: (1) permutation of TRAIN whole-run view-B pairings should destroy the cross-view source covariance; (2) replacing withheld/missing sensor payloads must not affect concept values. Spectral K0 must yield bias-only head and neutral concepts. These checks prevent hidden raw bypass and arbitrary source inflation.
- Compare raw observational source operators against two physically manipulated plumes without pretending K must equal 2: a nuisance effect, concentration coupling and sensor chemistry may create K>2 or K1. Preserve any negative result even if K rises and prediction worsens.

## Delineation

This is a *cross-view predictive covariance / canonical subspace* method assembled from established multiview statistics and concept bottleneck architecture, not a newly proven source-separation or causal-ICA theorem. TRAIN stability, permutation threshold and response contrast cannot establish unique chemical source semantics or instrumentation independence. This is one controlled lab, 30 TEST runs and already-viewed results. A genuine confirmatory source-specific effect requires fresh physical runs/dataset and preregistration before seeing its outcome.

No modification to accepted PIERE, IOI or clinical manuscripts.
