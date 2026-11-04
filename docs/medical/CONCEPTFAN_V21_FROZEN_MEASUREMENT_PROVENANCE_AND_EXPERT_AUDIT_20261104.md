# ConceptFAN V21 — frozen measurement provenance, native feedback, and source/artifact falsification (2026-11-04)

## Scientific hypothesis (written before any V21 execution)
Earlier ConceptFAN V16 trained expert yes/no labels directly on **native** concept trajectories but used measurement-level simulated reviewer labels, not clinicians. V17 discovered a persistent residual sensor direction, and V18 showed that a **task-predictive instrument artifact** could pass the same independent outcome risk screen as a genuine latent source. V19–V20 improved the native concept-to-risk training process but did not solve this *semantic identifiability* problem.

**V21 hypothesis:** independent measurement provenance and a second device channel can sometimes corroborate a concept, and soft TRAIN-only expert examples can affect the actual decision-mediating concept representation; however a common-mode artifact can spoof even independent-device correlations. The correct system must separate:
1. A reproducible temporal measurement direction;
2. An outcome-predictive association;
3. An independent measurement corroboration;
4. An expert-supported measured pattern;
5. A **physically or clinically validated causal concept** — NOT demonstrated by these tests.

This study is designed to retain explicit negative results, not to certify discovery.

## Frozen independent synthetic generator families
Five predeclared families with 4 model-visible historical sensor channels (24 past, 6 future), an **additional simulated measured reference device** held outside ConceptFAN, and separately generated latent truth accessible **only by a post-training audit function**. Generator seeds TRAIN 405011 (960 episodes), VALIDATION 405012 (240 episodes), sealed TEST 405013 (240 episodes). No latent oracle, external reference or test outcome is passed as an input to ConceptFAN, its supervised prediction loss, candidate creation or source risk screen. Train-only robust normalizers; all measurement histories complete for V17/V18 compatibility.

- `physical_predictive`: measured residual phys2 source drives visible sensor_2 and an independent reference; predicted outcome depends on that latent source.
- `physical_nontask`: same residual physical source and independent measured reference, but target is statistically independent.
- `artifact_predictive`: persistent **instrument-only** artifact drives sensor_3 and correlates with the outcome in the generator. Independent reference measures a different latent process, not the artifact.
- `artifact_nontask`: artifact persistent in the visible sensor, labels independent of it.
- `artifact_common_mode_reference`: artifact drives sensor_3 **and leaks to the separately identified reference device**, while target is correlated with the artifact. This is a deliberately deceptive provenance counterexample: cross-device agreement can be caused by common-mode contamination. It must NOT be silently counted as a physically valid concept.

The first two sensor coordinate axes are preselected existing measured directions for an initial K2 model (not real physical diagnoses). A **third direction is proposed using V17 residual temporal structure from first 350 TRAIN episodes only**. A binary outcome association is screened using V18 on next 300 **disjoint TRAIN episodes only**. A cross-device measurement-replication test uses the final 250 **TRAIN episodes disjoint from both**: two-sided label-pair permutation (199 draws), 95% bootstrap correlation bounds (199 draws), |r|≥.35 and the lower bootstrap **absolute** correlation bound >.20, plus same sign in both predeclared halves. The reference device is independently identified by a provenance registry; the test certifies only **MEASUREMENT_CORROBORATED**, never a causal source. If the reference channel is absent, the gate ABSTAINS.

The retained 60 TRAIN episodes are not allocated to a gate but may be used in the actual final neural fit. Every cohort ID set is checked disjoint for source/risk/provenance screening. Candidate construction cannot read the reference measurements or latent states. The reference is NOT a feature of the neural decision graph.

## Expert-feedback experiment (simulated measurement reviewer, no doctor)
On TRAIN episodes inside the discovery cohort, automatically generate up to 40 strong-positive and 40 strong-negative review intervals from the independent *measured reference* last-six mean (≥0.25 or ≤−0.25 standardized, not from oracle physical state). Reviews are prepared using existing V16 `ExpertReview` / `prepare_feedback` / `native_expert_loss` and supervised on the **same native K3 concept slot**. Exactly the same neural architecture, loss, optimizer, train epochs and seed used across arms.

Three predeclared arms × five families × three seeds 11/23/37 = **45 native full-model fits**:
1. `no_feedback`: native V10 K3, no feedback supervision, adaptive feature gradients enabled.
2. `measured_reference_feedback`: if measurement corroborated, use correctly oriented simulated yes/no reference examples; if not corroborated, produce only **artifact/unsure** reviews with zero native supervision (do NOT treat artifact as a physiological negative).
3. `intentionally_inverted_feedback`: deliberately reverse valid positive/negative measured-reference reviews irrespective of corroboration (a negative control for harmful expert labels; not a real clinician or an approved practice).

All candidate directions and feedback episode selections are frozen and generated from TRAIN histories/reference only before fitting. V21 feedback weight 1.0, 10 epochs, V10 standard TRAIN-only objective, optional native head refit **off** for initial concept correctness isolation. Model risk sees TRAIN binary targets, but feedback labels do not depend on target or hidden truth. No patient-level expert records or raw truth trajectories exported.

## Separated evaluations and epistemic boundary
On VALIDATION, measure mean direct outcome AUPRC/AUROC/Brier; simulated reviewer-reference concept AUROC and AUPRC on strongly measured reference examples; *after fitting* compute Spearman correlation of native slot activation to generator-known **physical source** and **instrument artifact** separately; native additive-logit error; and per-seed concept effect stability. Oracle diagnostics NEVER influence basis selection, risk screening, feedback generation, loss, candidate naming or hyperparameters. Crucially compare matching of physical versus instrumental latent processes rather than calling any predictive slot physiology.

Expected falsification: isolated predictive artifact may pass V17/V18 risk screen but must **fail independent clean-reference corroboration**; common-mode contaminated artifact may pass both. Whether these happen is measured, not assumed. A physical source unrelated to risk may pass measurement corroboration but fail outcome-risk screen — a legitimate measured concept is not the same thing as a prognostic concept.

This remains synthetic with simulation-based reviewers; success does not justify clinician accuracy, causal discovery, independent-hospital generalization or a claim that the model can identify a new disease mechanism. Accepted IEEE PIERE camera-ready stays unchanged. Code and evidence in a separate Draft PR, not merged into V20 before review.
