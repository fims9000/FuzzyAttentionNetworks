# V65 preregistration: dual-stream absolute + change concepts with multithreshold fuzzy states

**Registered on 2026-11-13 after V63 and V64 TEST scores were already viewed, before V65 results. This is strictly a post-selection architecture experiment, NOT independent test confirmation.**

## Motivation and scientifically bounded hypothesis

V63's unlabeled ICA/PCA K3 physical response source bank subtracts each episode's initial sensor level; V64 on the **same UCI322 real, temporally held-out data** showed substantial information loss from removing absolute sensor levels, followed by further loss in one bounded fuzzy membership and full-window averaging. V65 tests a concrete fix rather than adding a Transformer or arbitrary manually named gas source: *retain both absolute and innovation channels for each automatically discovered anonymous source direction; use several inspectable fuzzy reference thresholds and retain a recent-time state summary.*

## Locked data and source-learning setup

Original [UCI Gas sensor array under dynamic gas mixtures, ID322](https://archive.ics.uci.edu/dataset/322/gas+sensor+array+under+dynamic+gas+mixtures), zip SHA256 `8b6323b801363e11343ba1a5a7718e76666b715e753b3a40dbdfd1838d953fe7`, two full continuous sessions; independent controlled ethylene+CO and ethylene+methane setpoint signals withheld until AFTER source fitting. Use EXACT V63 original first 60% TRAIN, middle 20% VALID, last 20% TEST per file with 120-second two-sided embargo, 120-second nonoverlapping windows, same TRAIN-only 16-channel median/IQR and source parser. Same TRAIN414/VALID134/TEST136. Treat windows as correlated, physical session count=2.

For source directions use the identical **TRAIN-only V63 FastICA PCA95 innovation K** (seed20261913, no label, max8), expected prior TRAIN K3. No manual K matching two gases, no target setpoint in source selection and no TEST-driven rank, thresholds or encoder parameter selection.

## Dual-stream unlabeled source representation

For each of K anonymous ICA axes `w_k` and measured channel vector `x_t`, construct separately:
- **ABS** source `u_{t,k}=w_k^T x_t`, retaining level / approximate stationary chemical response;
- **DELTA** source `d_{t,k}=w_k^T (x_t-x_0)`, detecting causal changes relative to observed episode start.

Both use only measured sensor values available at or before t; no external reference controller setpoints or future samples. The difference between the streams is meaningful. From TRAIN projected source values alone, choose fixed fuzzy thresholds at quantiles **25%, 50%, 75%** per stream/axis and positive width `max(0.15, TRAIN-IQR/1.35)`. Thus *three fuzzy states per stream per statistical axis*. With K3 this is 2×3×3=18 fuzzy membership states, but only THREE discovered independent statistical axes, **not eighteen different physical gases**.

Two representations, identical information budget and model:
1. **STATIC-2STREAM**: apply original inspectable soft membership `0.5 + support*(sigmoid((v-tau)/width)-0.5)` to each state, fixed TRAIN-only thresholds.
2. **NATIVE-2STREAM**: original V56 diagonal causal recurrent per-source `NativeCausalFuzzyDynamics`, trained for exactly **8 epochs**, neural seed11, self-supervised source denoise/next-step forecast with 20% observed-cell dropout on original TRAIN sensor trajectories, no gas target labels or concept oracle. Freeze all source and dynamic weights before any gas-label training.

For either representation compute **whole-episode and most-recent-30-seconds mean** membership for each state (2×18=36 exposed concept summaries). This is a signed-additive **concept-only** Ridge(alpha=1) regression head fit on identical TRAIN setpoint labels. **Fairness refinement committed before V65 TEST outcomes were read:** use the same TRAIN-only per-column StandardScaler as the raw Ridge baseline, then transform its fitted coefficients and intercept back into original bounded fuzzy coordinates so `prediction=bias+sum_j beta_j*(fuzzy_summary_j-0.5)` exactly (numerical error ≤1e−9), no raw-to-risk bypass. No VALID/TEST distribution statistics enter that standardization. No per-gas hand-selected axis, human concept name, task gradient into source discovery or familywise cherry-picking.

## Precommitted comparators and outcomes

- Same target tasks and chrono splits as V63: CO and ethylene in CO session, methane and ethylene in methane session, concentrations in ppm, contemporaneous last30s setpoints; not forecasting future gas or independent chemical analyzer measurement.
- Compare STATIC-2STREAM, NATIVE-2STREAM, same TRAIN label budget raw 16-sensor Ridge, raw HGB (same V63 max_iter100/max_leaves15), V64 **ICA linear-absolute 2K summary Ridge** (frozen loadings), V64 baseline-differenced linear 2K Ridge, and original V63 static K3 fuzzy whole-episode. Do not omit models that beat the new representation. Metrics R², MAE ppm, NMAE to TRAIN target std, Spearman across all 4 gases × VALID/TEST.
- A **TRAIN-only** semantic audit reports the 18 state thresholds/widths and K axis permutations, but no physically named concepts. VALID and TEST partial Spearman of full K×gas state summaries separately may show association, not unique chemical factor identity. Labels never enter source formation.
- An explicit **incompleteness indicator** is computed on VALID only when raw Ridge R² > NATIVE-2STREAM R² + **0.05**. It is reported unchanged on TEST and does NOT add a raw model shortcut, cancel risk predictions or claim known chemical meaning. This detects when a concept-only explanation may be exact but the concept set fails predictive sufficiency.
- Since V63/V64 TEST has been viewed, any numeric gain here is **exploratory within two original recording sessions**. A real confirmatory gain would require genuinely new physical sessions/hardware or another previously unused dataset with controlled independent sources.

## Explicit failure conditions

1. A large 36-feature summary may recover raw predictive performance yet still not produce physically identifiable gases, especially if the same ICA axis predicts both gas setpoints.
2. If NATIVE underperforms STATIC, recurrence is not established as beneficial.
3. If RAW still outperforms both, an inspectable concept bottleneck remains insufficient, not a failed computational execution.
4. ICA chooses source axes using variance/independence under strong unverified mixing assumptions; 18 fuzzy states are mathematical membership functions, not 18 latent causal concepts.
5. Both physical sessions were used to choose the architecture family through previous V63/V64 result inspection, so any V65 result cannot be treated as a newly unobserved TEST claim. No patient/clinical generalization.

No modifications to accepted IOI/PIERE articles or repository master; research-only draft PR.
