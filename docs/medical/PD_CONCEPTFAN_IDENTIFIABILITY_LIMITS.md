# Identifiability limits of label-free dynamic concepts

**A mathematical falsification gate for the PD-ConceptFAN research program.**

A numerical algorithm cannot discover the *total true number of hidden
physical processes* from a time series unless all those processes leave
sufficient distinguishable traces in observed data. This is an
information-theoretic obstruction, not simply a limitation of training
losses, L0/MDL regularizers, or current neural architecture.

## Proposition 1 — no universal inference of hidden K from only (X,Y)

Let (Z_t\) be any latent Markov process, and let the measured time
series and outcome be generated as

```text
Z[t+1] = F(Z[t], noise[t])
X[t]   = H(Z[t], sensor_noise[t])
Y      = G(Z[1:T], label_noise)
```

Consider a second system with an additional statistically independent
hidden Markov process (U_t\):

```text
(Z[t+1], U[t+1]) = (F(Z[t], noise[t]), Q(U[t], independent_noise[t]))
X[t]            = H(Z[t], sensor_noise[t])
Y               = G(Z[1:T], label_noise)
```

The first physical system has latent dimension (K\); the second
has latent dimension (K+\dim(U)\). For every observed history (X\)
and every outcome (Y\), **the entire joint distribution**
(P(X_{1:T},Y)\) is identical in the two systems, because (U\)
does not affect measured channels or the outcome.

**Conclusion:** no procedure receiving only ((X_{1:T},Y)\) can
distinguish the two systems, even with infinitely many samples and
unlimited computation. Consequently no fully unsupervised method
can universally prove that the number of latent states is the
number of real physical processes. This is a formal impossibility
result for universal exact K recovery, not a statement that
discovery always fails.

## Proposition 2 — rotation/gauge ambiguity even when K is known

For linear observation (X_t=BZ_t+\epsilon_t\) with invertible
(R\in\mathbb R^{K\times K}\), the transformation

```text
Z_prime[t] = R @ Z[t]
B_prime    = B @ inverse(R)
A_prime    = R @ A @ inverse(R)
```

produces **the same observed distribution** as the original model.
A predictive covariance rank can be identifiable while the individual
latent axes are *not*. Thus consistent identification of (K\)
is logically weaker than consistent identification of each
physiological concept (c_k\).

This ambiguity is reduced, not magically solved, by restrictions
such as sparse, signed, measurable sensor combinations, temporal
independence, non-Gaussian innovations, intervention environments,
known anchor channels, ordered dynamics, or controlled mixtures.
For each restriction its truth must be checked independently.

## A statement PD-ConceptFAN can legitimately make

Under a stated data-generating class with (a) sufficiently observable
latent modes, (b) temporal excitation, (c) appropriately modeled
measurement noise and missingness, and (d) stable observational
environments, the rank of the positive-lag predictive observation
operator can estimate the **detectable predictive subspace
dimension** (K_{\rm obs}\). It cannot estimate factors whose effects
are absent from X; it cannot establish clinical names.

The program's revised engineering acceptance gates are:

1. **Dimension gate:** recover planted K across independent
   observable generators *and* report a counterexample on an
   unobservable generator where correct output is only
   (K_{\rm obs}<K_{\rm true}\). Never tune to Ktrue.
2. **Basis gate:** definitions must be sparse measured operators
   with train-only source coefficients, explicit signed direction,
   thresholds, measurement coverage and time intervals.
3. **Task gate:** prediction, future-forecast and denoising must be
   correct within a tolerable performance frontier; no hidden
   latent-to-target shortcut.
4. **Influence gate:** report exact additive contributions, their
   signed training-task weights, and change in model logit under
   transparent per-concept direct interventions. Do not claim
   patient treatment causality.
5. **Stability gate:** compare independently trained
   concept definitions and *patient-specific* signed effects,
   subtracting a chance null that preserves model marginals.
   Model bootstrap must sample trained models, not pairs.
6. **Real-data gate:** claim an observable K only if temporal
   channel overlap is adequate under both genuine and permuted
   observations, and train-subset estimates are stable across
   sampling scales. If any gate fails, status is *not identifiable*
   rather than a clinically named K.
7. **Clinical semantics gate:** without an external medical
   measurement, expert ontology, multiple observational environments
   with defensible invariance assumptions or interventional
   evidence, physiological correctness is **unverified**.

A strong paper can state both the positive mathematical
constraints **and** the exact limits of what autonomous concept
discovery can know. A network that confidently assigns "renal failure"
to an unvalidated latent axis is not interpretability.
