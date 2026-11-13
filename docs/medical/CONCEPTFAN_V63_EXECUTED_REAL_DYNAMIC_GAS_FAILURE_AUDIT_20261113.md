# V63 EXECUTED: previously unused real continuously varying gas mixtures — automatic concept identification FAILS

**Final green run:** [GitHub Actions #43479481509](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43479481509), commit `cc25aa4599b0f02cf6c54597179fc4491e9aae08`, **22 focused + legacy regression tests PASS**. Evidence artifact `conceptfan-v63-unseen-real-dynamic-gas-ica`, ID **14065005980**; exact machine-readable `V63_REAL_DYNAMIC_GAS_EVIDENCE/STATUS.json`, `METRICS.csv`, source SHA. There were two failed engineering attempts due only to output bookkeeping; the penultimate attempt actually wrote the complete evidence but had a stdout indexing error after writing. Those failures were fixed without touching the model selection, model seeds, reference target or held-out split.

## Source and prespecified protocol

Official UCI [Gas sensor array under dynamic gas mixtures, ID322](https://archive.ics.uci.edu/dataset/322/gas+sensor+array+under+dynamic+gas+mixtures), DOI 10.24432/C5WP4C, exact original archive SHA256 `8b6323b801363e11343ba1a5a7718e76666b715e753b3a40dbdfd1838d953fe7`. This new physical recording family had **not** been used in preceding V60/61/62 experiments. Protocol was fixed in `CONCEPTFAN_V63_FROZEN_DYNAMIC_GAS_ICA_PROTOCOL_20261113.md` BEFORE UCI322 data values were read.

Two real continuous gas-stream recordings, **NOT** thousands of independent experimental sessions: ethylene + CO and ethylene + methane. Original record count per physical session: CO 4,208,261 sensor samples / 42,087 valid 1-second medians, methane 4,178,504 sensor samples / 41,790 medians. Eight million samples are correlated, and setpoint commands are not external GC-MS per-second assays.

Original columns: time; independently commanded secondary-gas level; commanded ethylene level; 16 actual sensor outputs. Only the last 16 are passed to all source/auto-concept/unsupervised algorithms. The two controller setpoints are used AFTER frozen concept formation as TRAIN concentration-regression outcomes and VALID/TEST physical-reference audit values. Input sensor median/IQR learned TRAIN-only. Strict chronological two-sided **120 s embargo**, nonoverlapping **120 s episode** windows per recording (CO 208/67/68 and methane 206/67/68): TRAIN **414**, VALID **134**, TEST **136** correlated windows. No train/TEST sensor overlap or label leakage; physical independent-session count stays **2**.

## Actual source construction

- PCA rank was selected from standardized unlabeled TRAIN innovations (median last30 seconds − median first30 seconds); smallest number reaching 95% TRAIN variance, cap K8.
- Automatically selected **K=3** for both PCA controls and ICA. Explained TRAIN innovation variance by K3 **95.243%**; block bootstrap based on full 30-minute temporal blocks yielded K3 in 16/20 and K4 in 4/20. Thus even K3 is not fully stable and emphatically does NOT mean three distinct physical molecules.
- FastICA fit on the same TRAIN innovations with seed20261913, whiten=unit-variance, fixed 1000-iteration/1e-4 convergence rule; **converged**. No gas labels in PCA/ICA or thresholds.
- Both frozen projected source banks pass into separately trained, task-label-free V56 causal recurrent fuzzy encoders for eight epochs; ordinary PCA and ICA static sigmoid concepts are separate controls. Train labels enter only Ridge(alpha1) from **concept mean** to separate reference concentration targets; exact signed output sum can be reconstructed, no raw-sensor bypass. Model controls use raw sensor Ridge and HistGradientBoosting with identical train-label budgets.

## Real TEST concentration inference, R² (higher is better; negative means worse than predicting held-out mean)

Each individual physical recording contributed 68 correlated test windows; no independent device held out.

| Physically commanded real gas | PCA static | PCA native | ICA static | ICA native | raw 16-sensor Ridge | raw HGB |
|---|---:|---:|---:|---:|---:|---:|
| CO (ethyl+CO session) | -.089 | -.090 | -.042 | **-.054** | **+.679** | +.522 |
| Ethylene (ethyl+CO session) | +.029 | +.028 | +.035 | **+.033** | **+.696** | +.486 |
| Methane (ethyl+methane session) | -.433 | -.435 | -.424 | **-.429** | **+.535** | +.413 |
| Ethylene (ethyl+methane session) | +.007 | +.005 | +.009 | **+.008** | **+.650** | +.595 |

The negative V63 result is not a rounding artifact: actual raw Ridge reproduces commanded gas levels meaningfully in the same TEST windows, whereas concept-only variants retain very little task-predictive variation. On chronological VALID, raw Ridge also wins all four tasks: CO .643 versus ICA native -.110; ethylene with CO .556 vs -.257; methane .683 vs +.087; ethylene with methane .802 vs -.036. The native recurrence has no demonstrated advantage over simple static ICA concepts. This is not a case where merely adding more concept slots (K3 vs V62 K2) resolves the physical separation problem.

## Held-out physical association of independently commanded gas setpoints

All K3 axes were inspected. Spearman correlations were *rank-residualized with respect to the other gas setpoint* (conditional descriptive associations), on TEST windows. The source basis was fixed before any gas reference. The strongest abs partial correlations were:

| TEST controlled gas | Best ICA native concept | Best individual raw gas sensor |
|---|---:|---:|
| CO | .304 | **.785** |
| Ethylene within CO | .327 | **.834** |
| Methane | .405 | **.682** |
| Ethylene within methane | .364 | **.691** |

Even a VALID-selected one-to-one assignment of anonymous axes to gas names fails to establish chemically clean physical factors. It is **label-informed posthoc audit naming**, not self-organized chemical discovery. TEST gas-setpoint permutation controls yield smaller associations, but temporal autocorrelation and only two recording sessions make this far from a statistical causal-identification proof.

## Why this matters and what is NOT shown

The general observed failure cannot be attributed uniquely to ICA itself. The **V63 representation first subtracts each episode's initial sensor reading**, projects on a 3-D innovation subspace, applies bounded fuzzy sigmoid states, and averages full 120-second concept trajectories. Any of these may discard absolute concentration, short-term response and gas-specific information. Claiming the original physical sources are unidentifiable from this data is not warranted merely because this particular encoder fails; the raw Ridge baseline contradicts any broader claim that gas setpoints are inherently unobservable.

The conservative next test is an explicit **stage-by-stage information-loss ablation** using the same sensor normalization, same K3 source basis, same original TRAIN targets, and VALID/TEST. This will be labeled **post-V63 exploratory**, since the V63 TEST has already been viewed. It is preregistered in V64 and must isolate absolute baseline subtraction vs axis compression vs bounded fuzzy nonlinearity vs episode averaging. Do not change the old V63 numeric evidence or claim new independent confirmation.

**Scientific conclusion:** We have now demonstrated, on two distinct previously unused real experimental data families (UCI309 and UCI322), that current ConceptFAN automatic *observable statistical sources* may lose critical, physically measured information compared with simple raw-sensor predictors. The labels are actual controlled gas setpoints but not a measured independent chemical assay at each timepoint. A genuine physical meaning certificate, multi-installation replication, or clinical interpretation has NOT been established. Keep research PRs DRAFT. Original PIERE/IOI publications untouched.
