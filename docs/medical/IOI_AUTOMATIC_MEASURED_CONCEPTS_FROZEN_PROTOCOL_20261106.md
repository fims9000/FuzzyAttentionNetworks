# IOI: frozen automatic *measured* concept formation and independent source admission

Date frozen **2026-11-06 before experiment code or any TEST outputs were inspected**. Branch: \`experiment/ioi-automatic-measured-concepts-20261106\`, parent: executed V30 (PR #25). Prior PIERE/IOI/IITI documents, accepted articles and earlier evidence frozen; no real patient TEST cohorts.

## Problem and novelty boundary

Prior V17 already proposes TRAIN-only temporally structured residual directions but rejects all incomplete measurement histories; V22 already uses environment-variance BSS and risk screening. V30 manually hardcodes physical-source sensor4/5 and independent anchor6 and still trains original K2 known concepts from privileged TRUE simulator factor trajectories. A generic "automatic concepts", "concept completeness", "staged CBM", "risk gating" or "independent source discovery" first-invention claim would be false.

Goal: implement an END-TO-END prototype that:
1. Constructs two predefined **measured**, not oracle-supervised, K2 concept estimates from their known observable sensor pairs; outcome labels must have **zero gradient path** to these K2 concept values.
2. In *unassigned* channels, forms candidate SOURCE GROUPS from **TRAIN-only unlabeled, partially observed multichannel trajectories**, without a hardcoded new sensor grouping or the hidden simulator states.
3. Screens predictive relevance on TRAIN labels, then admits AT MOST ONE newly named source if **both independent VALID outcome cohorts** support gain, with a multiple-candidate penalty and explicit ABSTAIN otherwise.
4. Preserves native K2 model weights, K2 decision weights, bias and concept estimates; new source enters only as explicit additive signed contribution. Exposes group, sensor loadings, TRAIN measurement stats, masked value support, confidence and provable prediction drift for accepted vs rejected source.
5. Contrasts false source interpretations and future shifts without falsely treating observational prediction as physical/medical concept identification.

This is a **measurement-source association method**, not semantic concept discovery, causal inference or a human-verifiable clinical label. Its candidate grouping resembles established correlation clustering and is a **baseline engineering integration**, not an original source-separation theorem. V17/V22 already offer different source proposal methods; do not label them invented in this study.

## PREDECLARED synthetic source universe and splits

Use V30's NONLINEAR switching/shock generator, but augment the remaining unassigned channels with stable, independently simulated dynamic distractor sources:
- Known sensors 0–1 (concept A) and 2–3 (concept B), only these four may enter the frozen neural concept path.
- Unassigned channels 4–6: third predictive latent source measured with 3 independent-noise readouts, exactly as V30. The code proposing source groups must **not** know that triplet.
- Unknown channels 7–8: additional NONPREDICTIVE measurement source with independent nonlinear temporal dynamics.
- Unknown channels 9–10: measurement/recording-artifact source, ALSO independent and temporally correlated. It can be associated with outcome under an ARTIFACT-only target law; do not label such a discovered candidate physiological.
- Unknown channel 11: uncorrelated single-channel distractor without reliable multisensor validation.

Total 12 sensors, 16 time steps. Generator seed **20261113**; independent patient splits TRAIN1700, GATE VALID1200 separated into **two distinct 600-episode environments**, sealed TEST900. Three native neural initialization seeds **11,23,37**. Independent MCAR20 TRAIN corruption seed20267201, MCAR20 VALID seeds20267202 and 20267203, frozen TEST modes [complete, MCAR20, MCAR40, MNAR, block8] seeds20267900 + ordinal; all TEST interventions use SAME patients, not independent cohorts. Two target laws complete/no new predictive source and extra PHYSICAL source; third ARTIFACT law where extra predictive information tracks the nonphysiological recording process (no causal identity inference from statistics). Use identical X across outcome laws; no TEST labels used for training, candidate selection, thresholds, source admission or model configuration.

Also TEST-only negative controls: sign flip of ALL source channels 4/5/6, missingness of all 4/5/6 channels, and (in artifact world) common-mode sensor9/10 sign flip. These shifts are absent from TRAIN and both VALID environments; observational validity beyond them cannot be guaranteed.

## Frozen measured-known-concept training

Use canonical temporal ConceptFAN, 1 Transformer encoder layer/latent24/2 heads/FFN48/3 sigmoid membership functions/additive risk, dropout0, attention pooling; **input_dim=8 exactly** (sensor0–3 with observation bits). Native stage1 TRAIN ONLY, AdamW LR .002, batch128, 14 epochs, seeds11/23/37, fixed batch-order RNG 82000+seed.

**No oracle true concept trajectories supplied as targets.** For each known concept and TRAIN time step, use only the available two-sensor masked mean, with bounded sigmoid nonlinearity as observed measurement proxy. A proxy is defined only if at least one of its two sensors is observed. Minimize weighted concept-trajectory MSE over supported steps; ignore unsupported steps in the loss, no imputed payloads in targets. No outcome labels enter training stage1. Freeze the resulting native K2 path. Evaluation may report oracle K2 trajectory MSE as SYNTHETIC-ONLY audit, never used in training or gating.

Stage2 fit a **standard TRAIN-only logistic additive K2 risk head** on frozen native concept summaries (Y labels only here); freeze K2 risk coefficients for candidate admission. No raw/unassigned channel bypass into this known K2 path.

## Frozen TRAIN measurement-only candidate proposal

Unknown sensor indices are ALL channels not explicitly assigned to K2, i.e. [4,5,6,7,8,9,10,11], not any preassigned source clusters. For every unknown channel pair, compute a Pearson correlation on co-observed historical time points, using three fixed DISJOINT subject splits of TRAIN (seed 20267204, independent of outcomes). Require |rho| >= **0.60** and CONSISTENT correlation sign in all three splits; min jointly observed points >= **300** per fold. Form connected components of the resulting signed agreement graph; accept a component of size>=2 only if it is a complete clique with sign consistency to its lowest-numbered sensor; ambiguous non-cliques ABSTAIN, singleton channels are not promoted to concepts. Cap at **five** proposed candidates selected by largest minimum absolute within-group fold correlation (UNSUPERVISED only); if more, unused candidates are explicitly logged.
- Fit mean and standard deviation for each group readout only on TRAIN observed values; orientation signs from TRAIN measured correlation, never outcome or oracle source sign.
- Candidate per subject: q=0.5+observed_fraction * (sigmoid(mean of standardized/sign-aligned *observed historical readings*)-0.5); q=.5 if entirely unsupported. All grouping/normalization parameters are frozen at TEST, hashes recorded. No future/hidden sensor payloads.
- For no stable group, return zero candidates with explicit refusal. Group proposal's fitted physical semantics are always unknown.

Existing V22 covariance BSS uses a different known rank-two/environment schema and cannot be treated as a directly comparable 12-channel masked-source method without a faithful adapter; report this as a limitation rather than inventing a "V22 baseline" label.

## Independently validated *predictive* admission (not semantics)

For each <=5 candidate from TRAIN-only grouping, separately fit one coefficient gamma on TRAIN outcome labels, holding K2 head exactly frozen, minimizing logistic loss + .002 gamma² using original V28 optimizer. At VALID, evaluate additive candidate with per-subject observed support>=**0.50**, otherwise exact K2 fallback. On each disjoint 600-case VALID environment compute full-cohort NLL improvement K2 minus candidate, and bootstrap lower one-sided **99%** percentile from **399** resamples with whole-subject sampling (seed20267210+candidate ordinal+environment ordinal). Candidate passes only if BOTH environments have improvement>=**.003**, lower percentile>0, support fraction>=**.75**. The stricter per-candidate 99% bound helps control multiple-candidate screening (at most 5) but is NOT a formal universally valid finite-sample FWER theorem.
If more than one passes, choose the candidate maximizing the **minimum** of the two VALID mean gains, stable tie broken by frozen proposal ID. No TEST outcome is seen until the winner is frozen. If none passes, emit explicit source ABSTAIN with original K2 risk for every patient.

Output all proposed, rejected, gate-supported candidates with TRAIN correlation quality, sensor orientation, all VALID gains and confidence intervals, source usage/rejection on each TEST missingness mechanism; do not hide candidates chosen under artifact target law.

## Precommitted baselines on *exactly* same data

- Original frozen known K2 only.
- Our proposed automatic measured-source gate, separate named additional contribution.
- TRAIN-only **manual V30 pair** from channels4/5 with the SAME VALID gate and TOTAL labels (not available as a realistic automatic detector, reported as privileged sensor-map baseline).
- TRAIN-only 1D **PCA of all unknown channel episode means** followed by the same single-source risk-gain VALID screen and additive head, a simple conventional unsupervised baseline.
- Classical **multi-feature logistic risk** on frozen K2 plus ALL unsupervised candidate values with TRAIN-only refit, to expose effects of simply giving more head capacity (does not preserve original K2 risk weights).
- Raw 12-channel measured input logistic regression fitted TRAIN only (strong non-concept baseline).
- Oracle TRUE K2/K3 logistic score ceiling on synthetic TEST for diagnosis only, not deployable and no oracle TRAIN guidance to our candidate proposal. Oracle reference deliberately privileged, labeled clearly.

Evaluate full-cohort TEST AUPRC/AUROC/NLL/Brier and exact additive logit/decomposition. Record both observed absence of source and complete TEST cohort results; detector precision vs hidden source GROUPS only with oracle generator metadata, after all TEST measurements are evaluated.

## Required falsifiers

1. In COMPLETE target law, extra source should normally be rejected; if one is admitted, report false discovery, do not censor.
2. In PHYSICAL extra-source law, the algorithm should discover unassigned group sensors4–6 without oracle guidance; train-only evidence alone cannot certify its physical meaning. If not detected/admitted, state failure.
3. In ARTIFACT predictive target law, the gate may admit sensor9/10 because it is outcome-associated; **this must be reported as semantic ambiguity/false source interpretation**, not celebrated as physiologic discovery.
4. In source-flip/hidden TEST shift, predictive validation gate may still admit an invalid source; preserve the negative. A test-only missing group should fall back exactly to frozen K2.
5. Risk may underperform raw logistic/PCA/fully refitted CBM on some cases; no claim of overall superiority or clinical utility.
6. Training and risk admission use no true hidden concept trajectories (unlike V30) but still assume **externally provided identification of the TWO initial known sensor groups**. Completely unsupervised known K2 formation is NOT solved.
7. Conditional on known-source masks and a fixed generator; new latent-source grouping is based on temporal co-measurement correlation, NOT an independent validated causal source-identification result.

Immutable executed artifact keyed by commit SHA, all seeds and failures retained. Do not merge into user-provided PIERE camera-ready or present as validated new publication until genuine published CBM comparators, matched label/compute and another external generator family are executed.
