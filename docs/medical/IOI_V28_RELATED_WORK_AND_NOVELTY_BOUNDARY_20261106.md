# IOI 2026 expanded journal: related work and novelty audit (2026-11-06)

**Scope:** Scientific audit of research lines V25–V28. This is a source list and assessment, NOT a claim that these publications have already been experimentally reproduced in the repository. References below are public primary publication sources or clearly labeled preprints, inspected by 2026-11-06.

## Established work that makes generic claims non-novel

1. **Koh et al., Concept Bottleneck Models, ICML 2020.** Classic sequential/joint concept bottleneck training and interventions. A two-stage concept model and a linear head are NOT novel by themselves. https://proceedings.mlr.press/v119/koh20a.html
2. **Yeh et al., On Completeness-aware Concept-Based Explanations in Deep Neural Networks, NeurIPS 2020.** Direct definition/measurement of concept completeness; completeness itself is NOT a newly invented construct. https://research.google/pubs/on-completeness-aware-concept-based-explanations-in-deep-neural-networks/
3. **Havasi, Parbhoo and Doshi-Velez, Addressing Leakage in Concept Bottleneck Models, NeurIPS 2022.** Known problem of implicit target leakage into concept probabilities, and incomplete dictionary versus inadequate concept predictor. V27 cannot claim to have first discovered task leakage. https://proceedings.neurips.cc/paper_files/paper/2022/hash/944ecf65a46feb578a43abfd5cddd960-Abstract.html
4. **Parisini et al., Leakage and Interpretability in Concept-Based Models, JMLR 2026, 27(174):1–40.** Formal concepts-task leakage (CTL) and interconcept leakage (ICL). V27's frozen TRAIN-fitted conditional NLL is a *task-specific synthetic proxy*, not a first leakage metric or an information-theoretic identity proof. Baseline CTL/ICL is needed. https://www.jmlr.org/beta/papers/v27/25-1121.html
5. **Galliamov et al., Concepts' Information Bottleneck Models, ICLR 2026.** Information-bottleneck regularization for reducing CBM leakage. V27 staged supervision should be compared to a properly implemented capacity-/budget-matched IB variant, not only joint raw CBM. https://proceedings.iclr.cc/paper_files/paper/2026/hash/8555cf308efef02adacb0f9e09bd5f10-Abstract-Conference.html
6. **Marcou et al., Residual Stochastic Concept Bottleneck Models, arXiv preprint submitted 2026-10-16 (not yet asserted peer-reviewed).** Explicitly addresses concept incompleteness by jointly modeling known concepts and residual variables, preserving intervention propagation through Gaussian conditioning. Very directly overlaps a broad proposed V28 claim of "adding residual source without hidden bypass"; needs matched baseline and intervention-composition comparison. https://arxiv.org/abs/2610.23421
7. **Bhan et al., Towards Achieving Concept Completeness for Textual Concept Bottleneck Models, Findings of EMNLP 2025.** Iterative addition of important and identifiable concepts; dynamic concept augmentation not novel in itself. https://aclanthology.org/2025.findings-emnlp.106/
8. **Tapli et al., Rethinking Concept Bottleneck Models: From Pitfalls to Solutions, CVPR 2026.** Discusses concept relevance assessment and unintended bypass of concept bottlenecks. Original additive logit is an enforceable structural constraint but not enough for interpretability. https://openaccess.thecvf.com/content/CVPR2026/html/Tapli_Rethinking_Concept_Bottleneck_Models_From_Pitfalls_to_Solutions_CVPR_2026_paper.html

## What V25–V28 do, and what remains open

- V25: oracle-conditioned predictive diagnostic distinguishes a complete known synthetic dictionary from a planted omitted predictive factor. A device-artifact world with exactly matching observed distribution exposes non-identifiability. Not a clinical source certificate.
- V26: a postfit linear concept head strongly improves ranking over a 14-epoch trained fuzzy head on a fixed generator, at unequal optimizer budgets. Known CBM method, not novel.
- V27: outcome gradients can encode extra third-factor information in nominal K2 concept summaries; staged oracle-supervised training prevents that measured contamination on one generator at some AUPRC cost. Known **type** of leakage; own contribution is explicit controlled temporal experiment under this generator.
- V28: protect K2 concept coordinates *and risk coefficients*, add a separate transparent measured source, demand independent validation in two known environments, and ABSTAIN on failure. This is a useful engineering control and falsifiable generalization test, not guaranteed semantic identification. New unseen sensor shifts may still defeat it.

## Nonoverlap with user's other publications

- IITI 2026 #7579 covers concept–latent alignment and ConceptFAN architecture. Cite as own prior art, no duplicate architecture novelty claim.
- Accepted PIERE 2026 UID6530 tests 30-fold cross-training **signed contribution reproducibility**. Do not recycle its numerical figures, train/test tables, or identity/non-identifiability claims as V25–V28 novelty.
- Expanded IOI journal could instead focus on a narrower question: conditions under which **measured concept completeness tests**, **concept identity safeguards**, and **source admission** can / cannot support an interpretable explanation with incomplete time-series observations.

## Required next tests before a rigorous novel method claim

1. At least one *fundamentally different simulator family* (e.g. nonlinear multiplicative source interactions, delay/phase changes with equal time means) with independently frozen generator/seed. Different seeds in the SAME AR simulator are not external generalization.
2. Compare under **matched TRAIN labels, VALID labels, optimizer steps/FLOPs** against staged CBM, joint CBM, IB-regularized CBM, residual stochastic CBM, explicit non-concept raw baseline and no-gate K3 refit. Full external baselines require faithful published implementations, not loosely named local approximations.
3. Exact concept intervention contract and admission-source intervention: edit original q1 while holding q2 and qnew fixed; ensure signed logit difference is exactly predicted and no raw bypass. Test causal-source *identity* only with external interventions or true oracle, never by predictive F1 alone.
4. Independent real dataset only when ethical/data provenance restrictions and previous PhysioNet sealed cohorts are respected. A real patient label does not reveal a true source identity.
5. Quantify **false admissions of instrument proxies** under unseen shifts, rejected useful sources, and coverage conditional on missingness. Confirm that a two-environment gate can fail on a third environment; its failure is an explicit scope boundary.

**Recommendation:** Do not call the present code "novel physically grounded concept discovery" or "complete concept dictionary." Keep PIERE accepted; use V28 as a measured-source audit baseline. Reserve a stronger journal novelty claim for a validated new admission mechanism with true external-source provenance and a reproducible advantage over the cited current methods.
