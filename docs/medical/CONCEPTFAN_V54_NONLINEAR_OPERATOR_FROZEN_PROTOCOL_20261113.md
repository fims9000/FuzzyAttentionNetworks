# ConceptFAN V54: prospective nonlinear operator-family discovery protocol

**Frozen before V54 TEST outcomes, 13 November 2026.** Research branch `experiment/conceptfan-v54-operator-library-20261113`. This is not a claim of clinical or causal concept identification.

## Scientific question
Can an observed-data proposal discover a *sign-invariant nonlinear energy-like concept* where V53's signed lagged source is insufficient, feed it into the native exact concept-only FAN head, and preserve refusals, missingness and independent TEST isolation? A baseline with the same energy features is mandatory.

## Frozen mechanism
- V54 proposes from a **finite, explicit** operator library: `SIGNED_LAG` and `EVEN_ENERGY_LAG` (squared observed sensor values), over ordered unknown sensor pairs, lags 1–3, three disjoint TRAIN subject folds. All pairs must exceed absolute signed transformed-correlation .65 in every fold, beat the maximum instantaneous transformed-correlation by .30, and keep correlation sign. At most three nonoverlapping pairs. No Y, generator hidden source or semantic name at proposal time.
- Normalize each selected operator from TRAIN observed values only. At t use only source at t−lag and readout at t with both available. Aggregate last six steps, shrink missing episodes toward neutral 0.5.
- V53 native K2 TemporalConceptFAN is trained on observed K2 proxies with TRAIN Y; freeze K2. Fit exactly one source coefficient per proposed concept on TRAIN Y only, then require *two* disjoint VALID400 cohorts passing existing V28 conservative signed NLL gain, bootstrap and support criterion. Admit at most one source, yielding K2 or K3. No predictor bypass from raw X to risk. No source => exact K2.
- Three independent label modes on a **new synthetic** 16-step, 12-sensor generator: (A) predictive delayed sign-randomized energy source; (B) outcome-irrelevant delayed sources; (C) predictive *recording artifact* energy source. Unknown pairs: 4/5 even-lag, 6/7 signed-lag distractor, 8/9 even-lag artifact, noise 10/11. Generator does not provide physical labels to model.
- TRAIN950 / VALID1 400 / VALID2 400 / TEST650; native initializations 11 and 23. TEST stresses fixed before results: complete, MCAR20, MCAR40, sign flip on candidate paired channels, and unknown gain change on paired channels. Sign flips should leave *even operator* exactly invariant; gain shift need not.
- Compare native frozen K2, signed-only V53 proposal K3, full V54 operator-library K3, raw original-features logistic, and same-label-budget energy-engineered logistic. The latter should be strong: V54 is not allowed to claim prediction superiority unless it beats this baseline.
- Primary descriptive outcomes: TEST AUPRC, NLL, concept acceptance/refusal, per-case exact K2 fallback, TEST-only sign flip and gain outcomes. Save all TEST prediction arrays for independent post-hoc audit. Two seeds share the same TEST people; no false independent-people interpretation. Reproduce by source hashes and TEST arrays.

## Explicit falsification / limitations
- `EVEN_ENERGY_LAG` is a measured nonlinear feature, not a semantic physical concept. A predictive device artifact may pass the identical gate and must be named **statistical measurement association**, not physiology.
- Sign parity is only an algebraic invariance to flipping **both measured channel signs** under a squared operator. It does not imply immunity to gain drift, background change, clinical shift, or ability to detect physical causal direction.
- TEST-only gain shift can worsen NLL despite VALID acceptance; preserve failure and do not retroactively tune thresholds.
- Operator family library remains narrow and not continuously learned end-to-end. It does not make new concept semantics identifiable, even if AUPRC improves.
- If full V54 fails to outperform signed-only V53 on the nonlinear world, report negative result and preserve seeds/protocol. Do not edit prior IOI article to include unverified results.
