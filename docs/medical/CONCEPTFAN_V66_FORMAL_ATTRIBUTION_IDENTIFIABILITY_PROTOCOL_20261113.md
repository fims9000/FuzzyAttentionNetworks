# V66 formal identifiability barrier: exact additive explanation is not semantic identification

Date 2026-11-13. A **purely mathematical falsification** applicable to V53–V65 concept-only signed additive heads. It does not need a new physical dataset and is **not** independent empirical validation of any previously trained ConceptFAN.

## Claim to prove and interpretation

Take any K >= 2 bounded concept-membership coordinates C in [0,1]^(N,K), signed additive coefficients beta in R^K, intercept b. All predictions are f_i=b+sum_j beta_j(C_ij-0.5).

For any mixing strength 0 < a < 1, define A=(1-a)I_K+a 11^T/K. Every entry nonnegative and both row and column sums equal one, so C' = C A still lies in [0,1]^(N,K). The matrix is invertible, with eigenvalues 1 (once) and 1-a (K-1 times). Set beta' = A^{-1} beta and b'=b. Because A 1=1 and 1^T A=1^T, for every observation f_i'=b'+sum_j beta'_j(C'_ij-.5)=f_i exactly (within floating-point precision). Therefore K, boundedness, exact signed sum and identical predictions do not force unique coordinates or semantic attribution.

A continuum of distinct C' exists for any a in (0,1) and K>=2, unless the original concepts are all identical or otherwise degenerate. The decomposition is therefore algebraically non-identifiable under these specified constraints. The proof does NOT say no stronger model is identifiable: enforcing independence, mechanistic source-specific interventions, anchor calibration, an identifiable functional family, or known mixing geometry may exclude these transformations. It does NOT establish physical non-identifiability in all possible models.

## Executable deterministic verification

- Generate deterministic synthetic [N=1000, K=3], [N=1000,K=18], [N=1000,K=36] bounded memberships and diverse signed coefficients, seed20261966; real V65 concept head has 36 visible feature coordinates, but these synthetic points must NOT be described as real UCI322 observations.
- Test a=0.05, 0.35 and 0.75. Preserve bounds, matrix spectrum, condition number, exact risk logit/probabilities across sample, and total attribution sum.
- Measure whether individual signed attributions/top-absolute-contribution axis change, while forecast is unchanged. No guarantee that every random sample changes rank; report exact observed count, no cherry-picking.
- Produce explicit JSON per-K/strength diagnostics with max|delta logit|, bound violation count, changed top axis fraction, changed individual attribution norm, and formula proof note.
- Test degenerate K1 shows trivial identity (no counterexample), and reject a outside (0,1).
- Do not call C' a second physical sensor calibration or claim independent semantic discovery. Coefficients beta' are a reparameterization of the same already-fitted head; no new task training or outcome information required.

## Publication consequence

V65's exact contribution decomposition and V56's fuzzy membership bounds are **functional/explanatory** features; they do not alone identify physical gas sources. Semantic validation requires an independently defined reference (separately calibrated analyzers or controlled source-specific interventions) and cross-context invariance tests, with an explicitly stated identifiability assumption. An interpretability paper may quote this mathematical negative result without implying successful automatic medical concept discovery.

This is a standalone research test; no IOI/PIERE original accepted manuscripts or main branch altered.
