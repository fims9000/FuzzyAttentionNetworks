# ConceptFAN V17: executed native concept candidate source-versus-noise controls

**GitHub Actions success**: [run 41823908038](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41823908038); artifact `conceptfan-v17-candidate-all-controls-2` (ID 13318132015), all nine scenario records, three summary rows, STATUS.json and COMMIT_SHA.txt.
**CI checkout SHA**: `1c64672edca28fd54a3e91b6565a62fe58a4b9ac`.
**Downloaded artifact ZIP SHA-256**: `5d9ee3ee45ee258175172be98ba0b642e6421298c74ebb9c25fcfc1f65272426`.
22 tests passed (one warning), including explicit expected unchanged outputs before retraining and proof that an admitted slot enters the model's native decision path when activated.

| Scenario | Candidate proposed | ABSTAIN | Physical cause certified |
|---|---:|---:|---|
| Genuine new persistent hidden source | **3/3** | 0/3 | **No** |
| Independent white sensor noise only | 0/3 | **3/3** | No |
| Temporally correlated colored sensor noise only | **3/3 false physical hypotheses** | 0/3 | **No** |

All conditions use complete high-SNR linear synthetic time series with identical source/noise observational mixing, three simulator seeds, 99 null permutations, 24 bootstrap resamples, TRAIN-only residuals. The minimum attainable permutation p in this setup is 0.01, not a general population guarantee.

**Crucial counterexample:** within each same simulator seed, the observed process under genuine new source versus sensor-only AR noise is literally the same distribution, with identical candidate IDs under the test implementation:
- 2507: observable-residual-e64805169b44784d
- 2607: observable-residual-4dee5d173fc10759
- 2707: observable-residual-6d40eb80cf964c61

A high score or bootstrap consistency can therefore produce a useful *observable temporal concept* but cannot establish whether it is physiology or instrument drift. This test reproduces the V14–V15 non-identifiability concern instead of hiding it.

**Architectural contribution proved only in software:** on explicit concept-level `approve`, constructing a new K+1 native ConceptFAN slot preserves the trained preexisting logits, forecasts, reconstructions and clinician-process head at numerical tolerance. The new slot's risk/decoder readouts are initially zero. Tests show activating its decision coefficient changes the native exact-additive logit. Neither empirical post-admission retraining, outcome AUPRC improvement, nor medical semantics is established.

**Main limitations:** this prototype refuses missing/irregular observations; tests do not cover multiple environmental regimes, changes in mixing, nonlinear systems, or realistic ICU sampling; approval is a synthetic programmatic action, not actual physician review. Claim a proposed **measured dynamic pattern**, not a new physical factor.

See `src/conceptfan_discovery/candidate_admission_v17.py` and `src/conceptfan_discovery/experiment_candidate_admission_v17.py`. No accepted PIERE source changed.
