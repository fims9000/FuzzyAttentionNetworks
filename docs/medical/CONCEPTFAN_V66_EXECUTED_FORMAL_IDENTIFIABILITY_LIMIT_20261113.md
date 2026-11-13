# V66 EXECUTED: exact signed fuzzy explanations are algebraically non-identifiable without extra assumptions

**Status:** mathematical proof plus deterministic synthetic counterexample. This does NOT establish nonidentifiability under every physical-source model, nor provide independently new sensor observations. Execution [GitHub Actions SUCCESS #43482671241](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43482671241), source commit `33baa68bd6fa92a1c936e51e10f7523ad64a8ef2`, **17 unit tests PASS**, immutable artifact `conceptfan-v66-formal-explanation-identifiability` (ID 14064774234) `V66_FORMAL_EVIDENCE/STATUS.json`. The nine sample tests use **purely synthetic membership arrays**, not patient or gas recordings.

## Formal proposition

Let (C\in [0,1]^{N\times K}) with K>=2 be a matrix of fuzzy membership states, beta a signed additive head and b an intercept:
`f(C)=b+(C - 0.5*ones(N,K)) @ beta`.

For any alpha in (0,1) define the nonnegative *doubly stochastic and invertible* mixing matrix
`A=(1-alpha)*I + alpha*ones(K,K)/K`.
Its eigenvalues are 1 (multiplicity1) and 1-alpha (multiplicity K-1). Then `C_new=C@A` lies in [0,1] because it is a convex combination of fuzzy states, and `beta_new=A^{-1}@beta`. Since `A@1=1`, the exact head identity holds:
`b+(C_new-0.5) @ beta_new = b+(C-0.5)@beta`
for every possible input C, not just observed train/test points. Therefore output and every calibrated risk probability are exactly equal, while the individual concept states and signed contribution terms may change.

**Scope of theorem:** Bounding fuzzy memberships, exact signed additive reconstruction and prediction identity are **insufficient constraints** for unique *semantic* concept axes. It is NOT a statement that every ICA model is unidentifiable: stronger justified independent-source assumptions, physical interventions, externally calibrated anchors or model class restrictions may forbid cross-axis mixing. In particular the row-stochastic remixed states are not necessarily each a single independent source sigmoid, so the theorem is about the often-claimed explanatory algebra, not an arbitrary constrained generative system.

## Executable numerical verification (synthetic only)

Nine parameter combinations, each 1000 synthetic observations, K=3/18/36 and alpha=.05/.35/.75; no optimization, task labels or external reference data. All nine retained [0,1] bounds, had max absolute logit error <=**8.9×10^{-15}**, and risk probabilities identical to floating precision. At more substantial alpha=.75 the *identity of the largest absolute signed concept contribution* changed despite identical predictions:

| Fuzzy head dimension | Changed top absolute attribution axis (synthetic observations) | Max absolute error in logits |
|---|---:|---:|
| K=3 | **26.8%** | 2.44e-15 |
| K=18 | **28.4%** | 3.55e-15 |
| K=36 | **30.0%** | 8.88e-15 |

The rates are attributes of THIS deterministic synthetic distribution and chosen alpha, **not** clinical patient-level instability statistics or estimates on V65 UCI322 gas data. With alpha=.35 the changed-top rates were 15.1%, 8.3%, and 9.9% for K3/K18/K36 respectively. The difference in fuzzy states and individual contributions is nonzero despite identical exact logits.

## Research implication for IOI / ConceptFAN

V56–V65 now provide real evidence of **functional explanation fidelity** (exact reconstructible risk logits) and sometimes predictive usefulness, but not identity of physiological or gas-chemical concepts. Multiple equally bounded fuzzy coordinate systems can give identical predictive decisions with different per-concept signed attributions unless extra physical/structural assumptions are enforced and independently verified.

To claim semantic completeness or physical identity, specify the identifiability assumptions **a priori**, use independent source-specific interventions or a documented calibrated anchor, and show stable concept-to-reference correspondence in an unused external test setting. A VALID-based matching of learned axes to setpoint names remains **label-informed posthoc naming**, never autonomous concept discovery.

This is a **negative mathematical audit** usable as a distinct theory section or limitation in IOI and other methodology papers. It is not an automatically generated real physiological concept system, nor a source of new empirical predictive gains. Keep research PR DRAFT and accepted publication files unchanged.
