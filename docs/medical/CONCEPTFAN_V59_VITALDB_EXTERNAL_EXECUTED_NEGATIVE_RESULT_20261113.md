# ConceptFAN V59 — frozen BIDMC → VitalDB external-hospital transfer, executed negative result

**13 Nov 2026 · EXECUTED EXPLORATORY**. Draft [PR #65](https://github.com/fims9000/FuzzyAttentionNetworks/pull/65). Frozen protocol `CONCEPTFAN_V59_EXTERNAL_VITALDB_FROZEN_PROTOCOL_20261113.md`, amended only to repair deprecated per-track CSV transport **before accessing any external numeric TEST**. GitHub [CI run 43479595589](https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/43479595589) SUCCESS, source commit `2ada709858c7b6dcdb1e739fc2491874b22a34cd`; artifact `conceptfan-v59-vitaldb-external-hospital-transfer` ID `14062894937`, SHA256 `ead7fc685df5883272acb0e09c37c4f0dd9b7f11c6f7898ee0890e5649520d0f`. Independent auditor passed on all 8 private case arrays (2 native seeds×4 stresses); private per-person scores, patient IDs and timeseries were NOT uploaded.

## Real-data acquisition and source freeze
- BIDMC original: PhysioNet DOI 10.13026/C2208R, 53 checksummed records, original TRAIN 27 genuine patients / 1770 overlapping windows, VALID1 6/531, VALID2 6/354; labels contemporaneous ECG monitor HR≥100, no ECG inputs. V58 frozen source bank **zero V54 admissible operators**, correctly remained zero; V59 no new concept fitted/accepted on external source, auto FAN **exact K2 predictions on all external episodes**.
- VitalDB Seoul National University Hospital OR: 80 *pre-selected* distinct subjects based on public /cases+/trks track presence, seed20265913, one case per subject, 76/80 passed earliest predeclared >=70% four-track coverage; first **40** accepted used, 2313 16s overlapping TEST windows. No source label/prediction-based inclusion. The legacy per-TID CSV API returned nonnumeric pages (80/80 failures), so transport was repaired to official versioned packed `VitalFile` with the same original patient candidates, exact source track names, last observed ≤3sec, 480sec windows and >=70% rule. Case/track hashes and aggregate selection receipt retained.
- External source X: Solar8000 PLETH_HR (PPG pulse monitor), RR_CO2 (capnography; **different measurement modality** from BIDMC impedance RESP), PLETH_SPO2; Solar8000 HR **target only**. Monitored contemporaneous HR≥100, **not clinical prognosis**. HR is itself an outcome/reference, not an independent second physical assay.
- External TEST positive prevalence **13.705%** vs BIDMC TEST **2.331%**; cross-site AUPRC directly cannot isolate architecture from prevalence or site modality differences.

## Frozen BIDMC-trained models on VitalDB TEST, natural missingness (means of native seeds 11/23)

| Method | AUPRC↑ | NLL↓ | Patient macro NLL↓ |
|---|---:|---:|---:|
| Native FAN K2 | **0.5615** | **0.5451** | **0.5446** |
| Auto FAN V59 (NO_SOURCE) | **0.5615** | **0.5451** | **0.5446** |
| BIDMC-trained raw logistic | **0.9586** | **0.1760** | **0.1875** |
| BIDMC-trained energy-engineered logistic | **0.9574** | **0.1817** | **0.1937** |
| BIDMC-trained pulse-only logistic | **0.9602** | **0.1325** | **0.1407** |

Direct PPG PULSE against Solar8000 HR had **2.150 bpm** MAE on **2293** external windows with available PULSE. Direct pulse score AUPRC .9604. Strong ranking is unsurprising: contemporaneous monitor pulse-derived input approximates contemporaneous monitor HR label. **Not a forecast, and not independent causal measurement of a concept.**

### Predeclared stress, mean AUPRC / NLL (FAN vs engineered logistic)
- Extra MCAR20: FAN **.5253/.5466**, engineered logistic **.9563/.2823**.
- Complete PULSE missing: FAN **.1290/.5576**, engineered logistic **.1393/1.2721**; pulse-only logistic .1371/.7173.
- Input PULSE+25 bpm bias: FAN **.4559/.5593**, engineered logistic **.9542/2.8667**; pulse-only logistic .9604/1.9428. Direct PULSE/HR MAE grew **2.15→25.30 bpm**. Logistic retains ranking under monotone bias but becomes dangerously miscalibrated; FAN is less extreme in NLL but much worse in ranking. No model robust enough to claim medical deployment.

## Honest limits and next scientific mechanism
V53–V57 concept discovery was built around a fixed lag-over-instantaneous-correlation generator; real TRAIN BIDMC has **zero valid new concepts** under the frozen grammar. V59 demonstrated that this abstention persists on a genuinely external hospital and K2 native FAN is not competitive with simple monitor-score baselines for this task. This is rigorous **negative transfer**; no source identification, clinical predictive validity, physical corroboration or superiority claim.

Any next family of concept operators must be frozen and tested separately without tuning on these inspected BIDMC/VitalDB holdouts. In particular, same-time/recurrent window co-activity with patient-group stable null controls is a legitimate new hypothesis but needs a fresh independent external cohort and, for physical semantic claims, a *separate independently attested measurement target not also used as Y*. Do not silently treat correlated monitors as physical truth. Do not merge the exploratory PR to master; retain the negative result and source provenance.
