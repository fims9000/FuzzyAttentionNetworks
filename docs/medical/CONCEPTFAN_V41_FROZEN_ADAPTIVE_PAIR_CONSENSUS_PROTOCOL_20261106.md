# ConceptFAN V41 — FROZEN pre-TEST protocol: adaptive pairwise sensor evidence under missingness

**6 November 2026.** Parent [V40 executed PR #37](https://github.com/fims9000/FuzzyAttentionNetworks/pull/37); **DO NOT view or retune V41 TEST before this protocol and implementation are committed.** Research-only. Same controlled ON/OFF source-isolation hardware assumption as V39/V40, NOT unsupervised real-world biological or causal concept certification.

## Problem and main non-predictive validity goal

V40's measured paired-source variant with 40% missingness has only around 7% source use with AB alone and 0.2% when requiring all three pairs AB/AC/BC. Increasing source use by blindly taking any pair may falsely assign a physical factor: each independently fitted BSS pair differs in sampling noise/scale, and correlated device artifacts can deceive all available views. We test **whether TRAIN-only alignment and agreement certification enable use of whichever independent measured pair is still observed**, without filling values or relabelling a statistical predictor as a physically identified concept.

## Fixed method V41: TRAIN-only gauge alignment + per-episode pair selection

For each independent view pair \(p\in\{AB,AC,BC\}\), retain the V37 measured-source energy \(e_p\) (same fixed lag moments, eigengap, bootstrap, strict 4 actually joint ON&OFF timestamps and 0.10 energy threshold); normalize to \(q_p\in[0,1]\) using V37 TRAIN-only energy median/scale. Since three independently learned blind decompositions may have arbitrary relative scales, learn a *positive, monotone affine logit map* to the canonical AB **using TRAIN only**, among subject episodes where both pairs are supported:
\[
t_p=\logit(\operatorname{clip}(q_p,10^{-5},1-10^{-5})),\quad
t'_p=a_pt_p+b_p,\quad \widetilde q_p=\sigma(t'_p).
\]
Canonical AB uses \(a_{AB}=1,b_{AB}=0\). For AC/BC, fit \(a_p=\operatorname{cov}(t_p,t_{AB})/(\operatorname{var}(t_p)+10^{-6})\), \(b_p=\bar t_{AB}-a_p\bar t_p\); **reject a pair** if TRAIN overlap <60, Pearson correlation <.20, \(a_p\notin[.25,4]\), or nonfinite estimates. Do not decide calibration by outcome Y or TEST response. Fit is an observational scale *association*, not an independently proved physical gauge: correlations among artifacts can create false associations. No re-use of simulator's H or true source S in fitting.

Calibrate three pairwise disagreement thresholds \(\tau_{pq}=\max(.12,\min(.38,Q_{.95}(|\tilde q_p-\tilde q_q|)))\) on TRAIN-supported calibrated overlapping episodes, with at least 60 joint source-supported episodes per pair. If any threshold cannot be estimated, mark that **comparison invalid**, and refuse any episode needing it. No TEST-calibrated threshold.

At inference select **all** calibrated pairs with >=4 genuinely measured joint timepoints and positive measured cross-view energy. If 2+ pairs supported, **ALL pairwise disagreement checks must pass**. If disagreement or missing TRAIN certificate -> exact source refusal (q=.5/support=0). If exactly one trustworthy pair is supported, allow an *observationally single-pair inferred concept* only under documented independently wired view partitions/source ON–OFF calibration, noting there is NO per-episode multi-view consistency guarantee. Prefer pair with most jointly measured timesteps, deterministic AB<AC<BC ties. Do not select the pair with biggest q or biggest predicted Y. Then q is that selected pair's aligned measurement; explain which pair was used, actual number of observed times, agreement status (single-pair = uncorroborated, 2+ = corroborated), and exact additive contribution to the **frozen original K2**.

Two preregistered ABlation arms:
- \`DIFF_AB\` — original AB-only fixed source V40;
- \`DIFF_ABC\` — original all-three strict V40;
- \`DIFF_ADAPTIVE\` — V41 TRAIN-certified trusted pairs plus multi-pair agreement;
- \`DIFF_ANY_UNCHECKED\` — unsafe ablation: choose most observed V37 source pair without gauge alignment/agreement. **Do not call it an identifiable concept.**
- Original K2 fallback; matched predictive baselines \`RAW_DIFF_HGB_TRAIN1000\`, \`RAW_DIFF_HGB_ALL1800\`, \`RAW_DIFF_LOGIT_ALL1800\` from frozen V40 hyperparameters, receiving exactly the same original ON0:3 + measured paired difference4:9 tensor and mask. Same 1000 fitting outcome labels / 1800 upper-budget. ALL methods get same K2 first4 test mask.

Per method, signed gamma uses exactly TRAIN1000 Y and source features, original K2 encoder trained 10 epochs on known measured concept proxy labels (no Y gradients). All methods require original two independent VALID1+VALID2 (400+400) signed risk gate \`min_mean_gain=.003\`, \`min_support_fraction=.25\`, \`min_subject_support=.50\`, 399 bootstraps, and preserve exact original K2 logits when refused. Distinguish TEST *measurement support* before global gate from actual *risk source use* after VALID; report both to avoid claiming the algorithm fixed missingness when VALID disallows the source. Original model K2 weights never retrained after source admission.

## Four NEW devices / no re-used TEST

Linear AR mixing seeds \`20261507,20261509\`, nonlinear switching/heavy-tailed sensor+nonlinear transfer seeds \`20261521,20261523\`; never used as earlier devices. These extend the V40 **generator family** with fresh draws — NOT external validation on a new physical process. TRAIN1000, disjoint VALID400+400 and untouched TEST600 per device; original 16 time steps, 12 instrument channels, ON/OFF two physically different measurements with common source-aligned U invisible to monitor10; known switch source-isolates OFF, requiring this EXTERNAL truth.

TRAIN / VALID masks MCAR20; fresh TEST conditions with SAME K2 first4 mask across all scenarios:
- \`complete\`, \`MCAR20\`, \`MCAR40\`, \`MCAR50\`, \`synchronized_block8\`, \`missing_reference\`, \`missing_view_A\`, \`missing_view_B\`, \`missing_view_C\`, \`random_missing_one_view\`.
- The random missing view is per episode, chosen from A/B/C by fixed device/stress seed. Goal: verify pair selection BC/AC/AB respectively with no arbitrary q choice.

**Registered fault/outcome worlds**: healthy energy; strong matched common source-aligned instrument noise×8 on both ON/OFF; OFF-only drift×4; physical source leaking 50% into OFF; TEST reference missing; known source K2 complete; VALID2-known OFF drift; unknown sign inversion in ON view A (physically dubious OFF pairing, should reveal failures where remaining views are insufficient to detect). For each, evaluate all TEST masks. No models fitted to any TEST label.

**Registered outcomes** AUPRC/AUROC/NLL/Brier, measured support, admitted use, pair-choice count, number of single-pair/2+-pair episodes, disagreements, fallback counts, fraction of erroneous A concept selection in TEST A inversion, TEST energy association **AUDIT ONLY** (not available to source discovery).

**Primary claim falsifiers and control comparisons (all paired patient bootstrap AP 399 resamples, same TEST episodes and three fixed NN seeds):**
- healthy MCAR40: ADAPTIVE−AB, ADAPTIVE−ABC, ADAPTIVE−HGB1000;
- healthy MCAR50: ADAPTIVE−AB;
- healthy random_missing_one_view: ADAPTIVE−AB;
- strong matched artifact×8 MCAR40: ADAPTIVE−AB, ADAPTIVE−HGB1000;
- OFF-drift MCAR40: ADAPTIVE−AB (expect source may be unsafe);
- TEST A polarity inversion MCAR40: ADAPTIVE−AB (not guaranteed safe);
- complete K2 MCAR40: ADAPTIVE−K2 (global gate should reject);
- healthy MCAR20: ADAPTIVE−AB (cost of added checking);
- healthy MCAR40: ADAPTIVE−ANY_UNCHECKED (cost of enforcing consensus).
**At least 12 predeclared contrasts/device**; no post hoc threshold optimization.

## Strict honesty and provenance

- A single supported pair after TRAIN agreement with others does NOT prove correct orientation on current TEST, and all three pairs can share common-mode errors (V38 impossibility). Track **single-pair uncorroborated admissions** separately.
- V37 has a stringent >=4 jointly observed points/episode; changing this threshold based on V40 or V41 TEST is forbidden. At MCAR50 coverage can still be poor even with 3 pairs. No diffusion/Flow/Type3 sampling counts as a measurement.
- If a TRAIN pair fit or calibration fails, reject the affected method or pair and report the failure; do not lower thresholds to obtain favorable results. V41 must not receive true H/S.
- More pair support does not necessarily mean more AUPRC: no claim that automatic source physical name is proven. Defend novelty only if both honest measurement support and semantic integrity improve *conditional on explicit external physical assumptions*.
- No independent external clinical cohorts, no faithful published authors' CBM implementations and no matched training FLOPs; report these gaps.
- PR draft, base V40; do not merge master. Keep original V40 files and audited receipts unchanged.

**Completion evidence:** exact code implementing mask-aware pair choice, TRAIN gauge alignment, rejection and reason codes, proof-style unit tests for missing views/conflicting pairs/hidden NaNs/batch-one behavior, full CI 4 devices × 3 native K2 fits, TEST-level probabilities/receipts and original K2 snapshots, paired CIs, independently recomputed metrics, Russian research report including failures.
