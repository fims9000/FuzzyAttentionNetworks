# ConceptFAN V21b — frozen signed-risk explanation reproducibility control

## Motivation before execution (2026-11-04)
V21 successfully demonstrated observable physical-source vs instrument-artifact falsification, but the patient-specific **signed** contribution rank of the third native concept had a cross-seed mean Spearman ≈−0.333 in every family, despite source/raw concept absolute Spearman ≈0.98–0.99. This is not stable **faithful risk explanation**.

A leading confound is that every V21 neural seed started with arbitrarily signed native task coefficients. V18b previously showed that an independently TRAIN-screened sign is a better optimization initial condition than arbitrary positive sign on simple data. We must test whether this also explains V21 signed-effect reproducibility. **Do not** impose a permanently fixed sign or claim clinical identity from signed task association.

## Frozen V21b experiment
Exactly the same already-inspected five V21 synthetic generator families and split seeds; 960 TRAIN / 240 VALIDATION, 240 sealed TEST. Same 4 model-visible measured sensors and external independent reference (never a model input), same first 350 TRAIN sensor histories for V17 residual candidate, disjoint next 300 TRAIN labels for V18 candidate–risk association, same K3 physical concept bank, 10 joint training epochs, feature-gradient bridge, optimizer/loss/masks/seeds 11/23/37. No latent source truth to the optimizer or reviewer.

Arms:
1. `arbitrary_native_start`: exact existing V21 unreviewed model trained with native random initialized coefficients.
2. `train_screened_signed_start`: **only if** V18 independent TRAIN risk association ACCEPTS, set the *unrestricted native raw decision parameter for slot 2* to ±0.05 according to the TRAIN evidence **BEFORE** starting the identical joint training (no permanent sign constraint, no new model/head). If V18 ABSTAINS, this arm must also ABSTAIN (record it, not zero AUPRC).

All 5 families × 3 seeds × 2 arms = 30 registered opportunities, **24 fitted native models and 6 explicit abstentions** expected if the V21 TRAIN risk screen accepts its three predictive families and abstains its two null families. Report all statuses as actually observed. Same initial model parameters, differing only by the predeclared native parameter initialization for the accepted arm.

Evaluation after training: VALIDATION outcome AUPRC/AUROC/Brier; native third-slot **signed decision coefficient**, average absolute direct contribution, exact additive logit decomposition, cross-seed rank Spearman of patient-specific signed contributions **without any post-hoc orientation alignment**; native raw activation to both known true physical and artifact sources (ground truth loaded only after fitting as evaluation audit); source-vs-artifact provenance screen status. In particular, if signed Spearman improves for `artifact_predictive` or `artifact_common_mode_reference`, that demonstrates stable *prediction* but an **incorrect physical explanation**.

**Interpretation:** if independently oriented starts change patient signed-rank reproducibility while raw source alignment remains similar, it is an optimization/init-sign effect; it does NOT solve physical concept identity. If an artifact receives reproducibly signed risk, this is an important **negative** semantic result to retain. The uncontrolled arm should not be framed as certified clinician-ready risk.

No p-values from 3 seeds. VAL already inspected in V21, TEST stays sealed, no real physician annotations, no clinical intervention, no direct official intrinsic concept baseline. Existing V21 source/audit results remain immutable. Accepted PIERE paper untouched.
