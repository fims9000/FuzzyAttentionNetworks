# V56 frozen protocol: genuinely trainable temporal fuzzy concepts, no outcome gradients in concept formation

Protocol registered 2026-11-13 **after** reading V53–V55 new synthetic results, before any V56 TEST outcomes. V53–V55 are two-stage statistical source-definition models. V56 tests a NATIVE differentiable **self-supervised time-local concept layer**, not a new Transformer/Flow/Type-3 novelty claim and not medical factor identification.

## Scientific objective

Can an unlabeled TRAIN-only masked dynamics and reconstruction loss improve concept *state estimation* beyond the V55 static fuzzy source bank, while freezing observable definitions and retaining strict training-time concept-only risk prediction?

## Frozen mechanism

1. Use V55 source routing on UNLABELED TRAIN x/m only: if >=2 signed-correlation cliques cover >=50% of sensors, keep V54 robust median measurements; otherwise existing positive-lag predictive subspace; train rank 0 => ABSTAIN/bias-only. No initial K2, no oracle concepts, no task labels for source definitions.
2. Form source level z_k(t) and source availability s_k(t) from the fixed TRAIN-only measured/spectral operators. These are the ONLY inputs to the native concept network. No raw-to-logit bypass, no post-hoc concept extractor.
3. For each k, a **learned scalar causal recurrent state** h_k(t) uses only current observed source level and h_k(t-1), with diagonal recurrence. Learnable input scale, recurrent decay, bias, fuzzy threshold tau_k and positive width sigma_k. At each time, output **measured-support gated** fuzzy concept c_k(t)=0.5+s_k(t)*(sigmoid((h_k(t)-tau_k)/sigma_k)-0.5); thus a missing source is exactly UNKNOWN-neutral even if recurrence predicts h.
4. Self-supervised concept fitting uses **ONLY unlabeled TRAIN histories** and randomized withholding of 20% actually observed sensor cells, seed fixed per model. The network predicts (a) the original measured source level at t from its partial measurement history (denoising), and (b) next-step observed source level at t+1 from h(t) (causal forecasting). Use weighted squared loss and a small fixed regularizer to prevent hidden source-axis relabeling. TARGET outcomes never enter loss or gradients for this stage, regardless of scenario.
5. Freeze the entire fuzzy concept encoder after unsupervised training. Fit a logistic head to TRAIN outcome labels only through the mean concept activations. Logit always bias+SUM(beta_k*(mean_t c_k-0.5)); direct signed contribution checks every TEST case. This separates source definitions, concept state fitting, and task relevance; do not claim fully JOINT end-to-end clinical discovery.

## Frozen evaluation

- NEW nonlinear extended V30 seeds **20261393, 20261397**, TRAIN1400/VALID600 unused/TEST700, 12 sensors and 16 steps. K groups discovered unlabeled. Native neural seeds **11 and 23**, 8 epochs, 160 batch size, Adam lr=.015, gradient clip1; same TRAIN measured histories. Compare against V55 static fuzzy concepts and raw measured-history logistic, on identical labels and TEST patients, three outcome laws COMPLETE_NUISANCE/PHYSICAL_EXTRA/ARTIFACT_PREDICTIVE and TEST MCAR20 intact, single flipped source sensor 5, and common-mode flipped source 4:7.
- Different fresh variable-factor mixed-source generator with planted K in {3,5} and seeds **20261423+K**, train and DEV-VALID only, sealed TEST unused. Single neural seed 11, same 8 epochs. Compare against V55 spectral static and raw logistic; do not tune to oracle K.
- Metrics AUPRC, NLL, Brier, observed route and K, self-supervised denoising and forecast TRAIN losses, exact additive logit error and zero-contribution behavior when a source is completely unavailable. Across neural seeds on SAME TEST are not independent patient cohorts.
- No network, head, source-group rule or hyperparameter selected using TEST performance. V56 is post-hoc relative to V55 and uses a familiar generator family; its evidence remains exploratory.

## Stop/falsification rules

- If native temporal dynamics fail to improve over V55, publish the negative result. Additional capacity and self-supervised objectives are not a guarantee of more interpretable predictions.
- If a learned fuzzy slot carries a recording artifact, semantic/medical identity is still UNKNOWN; self-supervised dynamics cannot resolve observational equivalence without provenance anchors.
- If all measured support is absent, c must equal 0.5 and the logit must equal learned bias even if recurrent h is nonzero.
- The model is a **differentiable, diagonally recurrent temporal fuzzy concept** with fixed sources, not a general purpose self-discovered physical ontology, causal model, or first annotation-free time-series concept network. Earlier MAGNETS (2026) and ConceptTime (2026) are relevant prior work; compare before publication.
