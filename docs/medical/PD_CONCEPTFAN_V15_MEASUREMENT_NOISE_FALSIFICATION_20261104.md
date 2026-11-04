# V15: source versus measurement-process changes under explicit temporal or two-view assumptions

**Status:** separate synthetic research branch based on V14. No change to accepted PIERE manuscript or previous PhysioNet Set A/B research results.

## Problem and scope

The V14 covariance-only counterexample is genuine and cannot be repaired by further bootstrap or an additional generic regularizer. For every environment e, changing source covariance by Δ_e is observationally identical at covariance order to changing sensor noise covariance by B Δ_e Bᵀ. It is impossible to identify which mechanism occurred from these covariances without new assumptions or measurements.

V15 implements two **conditional** alternatives, not a novel identifiability theorem.

**Route A — lag-1 covariance contrasts.** Let X[e,t]=B Z[e,t]+η[e,t] and assume common full-rank B, zero source/noise cross-covariance, source lag covariance Cov(Z[e,t+1],Z[e,t])=diag(ρ v_e) with common positive ρ, and sensor error η temporally **white**, though its instantaneous covariance Ω_e may change between environments. Then

    L_e = Cov(X[e,t+1], X[e,t]) = B diag(ρ v_e) Bᵀ.

Differencing L_1−L_0 and L_2−L_0 and jointly diagonalizing with distinct variance-change ratios recovers B directions up to sign, scale and permutation. Measurement noise shifts that are *white in time* cancel from L. This is a classical SOBI/second-order-source-separation idea. The program never claims to statistically verify sensor error whiteness from X alone. External documentation is mandatory and otherwise it abstains.

**Route B — synchronized two independent sensor views.** If x_A=A Z+η_A and x_B=C Z+η_B, with zero source/noise correlations and independent errors across views **within each environment**, then

    M_e=Cov(x_A,x_B |e)=A diag(v_e) Cᵀ

regardless of autocorrelation and environment variation of noise *within* either view. Cross-view covariance contrasts may identify source mixing columns under suitable rank and distinct variance-shift ratios. This is also a classical multi-view blind-source-separation result, not a new theorem. A real independent sensor channel, not copying a sensor input array and calling it two views, is required.

Both methods enforce minimum environment episode counts, permutation screening, rank/contrast-gap checks and episode-bootstrap stability. Both produce only conditionally identified, unnamed mathematical sensor directions. False or undocumented assumptions cause ABSTAIN by contract where they can be detected, but data-based tests cannot authenticate the metadata.

## Preliminary frozen synthetic tests

V15 test fixture has three unlabeled environments, autoregressive Gaussian latent sources, K=3/5/7, time series of length 12, a shared sensor mixing matrix, and fully observed time points. No physiological measurements or patient records are read. We deliberately compare (i) real latent variance shifts, (ii) environment-dependent temporally white measurement-noise shifts, (iii) temporally colored noise aligned with source mixing, (iv) unchanged source and sensor processes. Two-view simulations additionally compare independent colored sensor errors and **shared** colored sensor errors that violate cross-view independence.

**Crucial constructive negative results:**
- Under time-white sensor-noise shifts with unchanged true latent factor variance, the new lag contrast generally ABSTAINS; unlike V14's ordinary covariance contrast, it does not automatically mistake changes in instantaneous sensor noise for source dynamics.
- Under AR(1) colored sensor noise sharing the source mixing and temporal dynamics, lagged covariance can still falsely certify *stable* directions while true source variance is unchanged.
- Under two independent sensor views, changes in colored noise within the views should not affect cross-view covariances; under shared cross-view colored noise, the cross-view method can falsely certify directions even when the source itself does not change.
- Population equalities establish impossibility even with unlimited samples: a source lag covariance shift B(ρΔ)Bᵀ is identical to a colored-sensor lag shift B(ρΔ)Bᵀ; likewise a source-induced cross-view covariance shift AΔCᵀ is identical to a shared-noise covariance shift AΔCᵀ.

**Limits:** no fully automatic physical-source identification without an externally justified noise structure. At K=7, rejection under limited sample sizes is a desired conservative outcome, not proof the source count is lower. A method that accepts a case where its stated assumptions are false is not a valid certificate of scientific factor meaning. Learning native neural signed risk heads is not attempted in V15 because V14 showed no consistent advantage over simple linear heads; the bottleneck here is measurement-process identifiability.

## Existing scientific context

Methods based on lagged covariance / joint diagonalization have a long history (e.g. SOBI and other second-order BSS, Belouchrani et al., IEEE TSP 1997, https://doi.org/10.1109/78.554307 ). Multi-view identification under independent measurement noise likewise does not originate here. Scientific novelty remains **unestablished** without matched prior-art experiments and independent source-measurement provenance.

## Code, verification and provenance

- `src/conceptfan_discovery/lag_contrast_v15.py`, `lag_synthetic_v15.py`
- `src/conceptfan_discovery/crossview_v15.py`, `crossview_synthetic_v15.py`
- `tests/discovery/test_noise_separation_v15.py`
- `.github/workflows/conceptfan-v15-sensor-noise-falsification.yml`

The accepted PIERE negative cross-run reproducibility conclusion remains unchanged. V15 is a deliberately limited *research falsification prototype* and should not be merged into clinical training or characterized as a validated general concept-discovery method.


## Executed first new-seed audit, 72 conditions, **source recovery evaluated on fitting episodes**

[CI success 41815626891](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41815626891), artifact `V15-frozen-twoseed-full-denominator-1`. Two generators 20261105 / 20261205 × K_true=3,5,7. Exactly 6 trials for each estimator × noise/scenario combination; accepted/rejected attempts all retained. These first new-seed experiments are **exploratory in-sample oracle diagnostics**, because source directions and source-oracle matches were evaluated on the same episodes. This was not hidden validation and must not be described as such. A separately frozen, distinct-case generator was subsequently created.

| Estimator | Genuinely changed source variance | White noise only | AR(1) colored noise only | No changes |
|---|---:|---:|---:|---:|
| V14 same-time contrast | 2/6 | 1/6 false positive | 2/6 false positive | 0/6 |
| V15 lagged contrast | 4/6 | **0/6 false positive** | **4/6 false positives** | 0/6 |

Paired-view V15 additionally certified **2/6 genuine source shifts**, **0/6 independent colored sensor-only shifts**, **4/6 shared colored sensor-only shifts falsely**, and 0/6 no-shift controls. At K_true=7, the lag method accepted a true-source shift in only 1/2 trials while the paired-view method accepted none in the six positive K5/K7 trials. This sensitivity deficit is substantive; it cannot be concealed by reporting high correlations on successful cases only.

**A new protocol was then frozen before using its new generator**: `src/conceptfan_discovery/experiment_v15_holdout.py`, generator seed **20280105**, 700 separate TRAIN episodes in each environment, 250 (lag) or 300 (paired-view) **disjoint evaluation episodes** in each environment, 36 total positive/negative estimator runs. All estimator parameters and refusal thresholds remain unchanged, and all 36 acceptance outcomes belong in the denominator. See `.github/workflows/conceptfan-v15-HELDOUT-third-seed-ONCE.yml`. The third generator is not a substitute for out-of-family or clinical validation.
