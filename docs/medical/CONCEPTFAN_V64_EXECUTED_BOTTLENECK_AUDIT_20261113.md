# V64 EXECUTED: why real gas concentration information disappears in ConceptFAN

**Scientific status:** successful exploratory **post-V63 ablation**, not new independent TEST confirmation. Frozen diagnostic plan `CONCEPTFAN_V64_FROZEN_BOTTLENECK_ABLATION_20261113.md` registered AFTER V63 TEST results had been viewed, before V64 numeric outputs. Complete [GitHub Actions SUCCESS #43480257048](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43480257048), executed SHA `b506ca418c62883701520c90703b1ac018ab5450`, artifact `conceptfan-v64-real-dynamic-bottleneck-ablation` ID **14063559534** (complete STATUS.json and 88-row METRICS.csv). **11 targeted regression tests passed**.

## Experimental contract

Same original physical dataset [UCI ID 322](https://archive.ics.uci.edu/dataset/322/gas+sensor+array+under+dynamic+gas+mixtures), source ZIP SHA `8b6323b801363e11343ba1a5a7718e76666b715e753b3a40dbdfd1838d953fe7`, 16 sensor inputs, 2 independently recorded 12-hour continuous sessions, strict bilateral 120s chronology embargo, TRAIN414/VALID134/TEST136 correlated 120-second windows, no labels in source rank/ICA/fuzzy calibration. Original V63 ICA/PCA 95% TRAIN-only innovation rank **K=3**, same signed projections and TRAIN sensor normalizer. Source rank never changed to match gas type. All supervised heads are Ridge(alpha=1) on identical TRAIN gas setpoints; no future or TEST values enter TRAIN transformation.

This study diagnoses five stage-specific information losses:
1. Retain absolute raw sensor levels (reference baseline).
2. Subtract window initial levels (`x(t)-x(0)`), without any ICA or fuzzy compression.
3. Compress into K3 original unlabeled ICA/PCA source axes (absolute vs initial-differenced).
4. Transform projected values into bounded, TRAIN-calibrated fuzzy membership states.
5. Average full 120-second sequence vs only last30-second current-state summary.

Controls also include an unrelated PCA-K projection on absolute raw inputs to isolate whether **K3 itself** is inadequate. No variants or K were selected based on TEST. Every one of 11 stages per physical target was reported.

## Exact held-out TEST, continuous concentration-regression R²

Each physical recording contributes only 68 temporally correlated TEST windows, not 68 independent apparatus experiments.

| Representation | CO | Ethylene+CO | Methane | Ethylene+methane |
|---|---:|---:|---:|---:|
| Raw absolute sensor means (32 features) | **+.6792** | **+.6964** | **+.5348** | **+.6498** |
| Raw first-sample-differenced means (32) | +.2547 | +.3384 | -.0210 | +.4276 |
| PCA K3 of absolute 32 raw features | +.2302 | +.3441 | +.2433 | +.3803 |
| TRAIN ICA K3, **absolute linear** projected means (6) | **+.6016** | **+.5319** | **+.4063** | **+.5650** |
| TRAIN ICA K3, *baseline-differenced linear* means (6) | +.2047 | +.3307 | -.1104 | +.3701 |
| ICA K3, bounded sigmoid fuzzy **whole-window mean** (3) | -.0134 | +.0398 | **-.4825** | +.0125 |
| ICA K3, bounded fuzzy **last30-second mean** (3) | +.0938 | +.1261 | -.2813 | +.1241 |

PCA projections of baseline-differenced linear and fuzzy signals tell the same qualitative story: for methane, PCA projected absolute +.4055, projected differenced -.1102, bounded fuzzy whole-window -.4670, late fuzzy -.3035. No ICA-specific artifact is needed to explain this failure.

## Causal failure path; not proof of physical non-identifiability

**Failure A, information destruction:** V63's source map used `delta X(t)=X(t)-X(0)`. For a gas concentration that is elevated but approximately constant during a 120-second episode, absolute component of concentration affects both initial and later sensor signal, and subtraction can cancel it. This representation is more suited to detecting *changes* than estimating an *absolute state*. Raw level-preserving ICA with the SAME K3 loadings restores much of the predictive information (.602 CO and .406 methane), which makes the baseline-subtraction loss directly measurable.

**Failure B, scalar bottleneck saturation / aggregation:** After source projection, V63 further applies one sigmoid-like fuzzy membership per ICA axis and averages all 120 timesteps. The difference from linear K3 to fuzzy K3 is large (CO +.205 -> -.013; methane -.110 -> -.483). This ablation changes both feature linearity and dimensionality (2K source summaries vs K fuzzy summaries); it therefore **cannot uniquely attribute the loss to the sigmoid alone**. Late-only fuzzy pooling partially restores accuracy (CO +.094; methane -.281), supporting but not proving that temporal pooling suppresses useful current-state information. A finer independent factorization of fuzzy nonlinearity vs summary dimensions remains required.

**Not solely rank K:** ICA K3 *absolute* features achieve .602/.406, so selecting K3 alone cannot explain the catastrophic original fuzzy results. Conversely, they still fall behind raw Ridge .679/.535, so source compression does lose some relevant information.

No increase of K or new Transformer was required to identify these failures. The architecture now needs:
- Separate **absolute physical sensor level** and **causal change/innovation** channels, not a single baseline-differenced source bank.
- Multiple calibrated fuzzy threshold memberships per statistical axis **and** comparable matched summary capacity to distinguish bounded nonlinearity from reduced head dimensionality.
- A late-window/prefix-safe state summary alongside whole-window mean, with explicit evidence of information preserved.
- A quantitative held-out **source-completeness** check against stronger raw-sensor training predictors; exact signed additive concept attribution is not a guarantee of semantic or predictive sufficiency.

## Mathematical qualification

If raw observations satisfy `X(t)=A S(t)+b+noise`, then `X(t)-X(0)=A[S(t)-S(0)]+noise(t)-noise(0)`: a stationary component of `S` is removed by construction. More generally the unsupervised factorization `X=A S` is observationally nonunique: for any invertible `R`, `X=(A R^{-1})(R S)`. ICA can restrict this ambiguity under stronger independent non-Gaussian source assumptions; this experiment DOES NOT show those assumptions hold for independently controlled gas mixtures. Therefore statistical anonymous axes, no matter how predictive or mathematically additive, do not intrinsically carry unique physical gas identities.

**Limitations:** Same already-viewed TEST, only two physical recording sessions, controlled gas setpoints not an independent chemical-analyzer time series, no instrument fault labels, no cross-hardware or clinical replication. Results are diagnosis of this particular ConceptFAN implementation, not causal/clinical certification. Research PR remains DRAFT; accepted IOI/PIERE manuscripts unchanged.
