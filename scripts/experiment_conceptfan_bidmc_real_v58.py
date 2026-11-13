#!/usr/bin/env python3
"""V58 real BIDMC transfer of *actual native* ConceptFAN and discovered operators.

PhysioNet 53 patient records -> patient-disjoint TRAIN/VALID1/VALID2/TEST.
ECG-derived HR is a simultaneous binary target *and* a scoring reference,
not an independently attested assay. All X uses only PPG monitor PULSE,
impedance RESP and SpO2 numerics with causal measured feature operators.

Do not claim physiological concept identities, future prognosis, or external
physical-source corroboration from this benchmark. All negative discoveries,
admission refusals and failed baselines are preserved.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score,log_loss

from conceptfan_discovery.bidmc_realworld_v58 import (
    load_bidmc_realworld_v58,causal_monitor_features,
    apply_train_normalization,THRESHOLD_BPM,
)
from conceptfan_discovery.operator_library_concepts_v54 import (
    discover_operator_library_v54,
)
from scripts.experiment_conceptfan_joint_operator_v55 import (
    _fit_joint_candidates,native_joint_prediction,
)
from scripts.experiment_conceptfan_operator_library_v54 import (
    _baseline,_apply_baseline,native_k2,native_logits,
    run_joint_epochs_v47,
)

SEEDS=(11,23)
STRESS=("natural","MCAR20","pulse_missing","pulse_plus_25_bpm")
PROTOCOL="CONCEPTFAN_V58_REAL_BIDMC_PATIENT_DISJOINT_TRANSFER"


def _test_variant(heldout,norm,scenario,seed):
    if scenario=="natural":
        return heldout["x"].copy(),heldout["mask"].copy(),heldout["pulse"].copy()
    rng=np.random.default_rng(seed)
    input_measured=np.where(heldout["mask"][:,:,:3],
                            heldout["raw"][:,:,:3],np.nan).astype(float)
    if scenario=="MCAR20":
        keep=rng.random(input_measured.shape)>=.20
        input_measured=np.where(keep,input_measured,np.nan)
    elif scenario=="pulse_missing":
        input_measured[:,:,0]=np.nan
    elif scenario=="pulse_plus_25_bpm":
        input_measured[:,:,0]=np.where(np.isfinite(input_measured[:,:,0]),
                                      input_measured[:,:,0]+25.,np.nan)
    else:raise ValueError("Unregistered real-world stress")
    values=np.zeros_like(heldout["raw"],dtype=np.float64)
    available=np.zeros_like(heldout["mask"],dtype=bool)
    for idx in range(len(input_measured)):
        vv,mm=causal_monitor_features(input_measured[idx])
        values[idx]=vv
        available[idx]=mm
    z=apply_train_normalization(values,available,norm)
    last_pulse=input_measured[:,-4:,0]
    counts=np.isfinite(last_pulse).sum(axis=1)
    pulse=np.where(counts>=3,
                   np.nansum(last_pulse,axis=1)/np.maximum(1,counts),
                   np.nan)
    return z,available,pulse


def _scores(y,logit):
    y=np.asarray(y,np.uint8)
    z=np.asarray(logit,float)
    if y.ndim!=1 or z.shape!=y.shape or not np.isfinite(z).all():
        raise ValueError("Invalid held-out real scores")
    if not np.isin(y,[0,1]).all():
        raise ValueError("Invalid binary ECG-HR target")
    prob=expit(z)
    return {
        "auprc":None if np.unique(y).size<2 else
                 float(average_precision_score(y,prob)),
        "nll":float(log_loss(y,prob,labels=[0,1])),
        "target_rate":float(y.mean()),
        "mean_predicted_probability":float(prob.mean()),
        "n_windows":len(y),
        "both_classes_evaluable":bool(np.unique(y).size==2),
    }


def _pulse_score(y,pulse,ecg):
    mask=np.isfinite(pulse)
    if mask.sum()<25:
        return {"n_available":int(mask.sum()),
                "ECG_HR_vs_PPG_PULSE_MAE_bpm":None,
                "pulse_threshold_AUPRC":None}
    both=np.unique(y[mask]).size==2
    return {
        "n_available":int(mask.sum()),
        "ECG_HR_vs_PPG_PULSE_MAE_bpm":
            float(np.mean(np.abs(pulse[mask]-ecg[mask]))),
        "pulse_threshold_AUPRC": (
            float(average_precision_score(y[mask],pulse[mask]))
            if both else None),
    }


def _train_pulse_logistic(train):
    pulse=np.asarray(train["pulse"],float)
    alive=np.isfinite(pulse)
    if alive.sum()<100:
        raise RuntimeError("Insufficient observed PPG monitor PULSE")
    center=float(pulse[alive].mean())
    std=float(max(pulse[alive].std(),.1))
    x=np.column_stack((np.where(alive,(pulse-center)/std,0.),
                       alive.astype(float)))
    model=LogisticRegression(max_iter=1000,random_state=31,C=1.)
    model.fit(x,train["y"])
    return model,center,std


def _pulse_logit(pulse,fitted):
    model,center,std=fitted
    observed=np.isfinite(pulse)
    values=np.column_stack((np.where(observed,(pulse-center)/std,0.),
                            observed.astype(float)))
    return model.decision_function(values)


def _bootstrap_by_real_patient(y,logit_a,logit_b,patient,*,n_draws=399,seed=20265855):
    """Resample patient *groups*, never overlapping 16-second windows."""
    people=sorted(set(patient.tolist()))
    if len(people)<4:return None
    position={g:np.flatnonzero(patient==g) for g in people}
    rng=np.random.default_rng(seed)
    diffs=[]
    for _ in range(n_draws):
        chosen=rng.integers(0,len(people),size=len(people))
        ids=np.concatenate([position[people[i]] for i in chosen])
        if np.unique(y[ids]).size!=2:continue
        va=average_precision_score(y[ids],expit(logit_a[ids]))
        vb=average_precision_score(y[ids],expit(logit_b[ids]))
        diffs.append(float(va-vb))
    if len(diffs)<100:return {
        "status":"not_estimable_too_few_two_class_patient_bootstrap_resamples",
        "valid_draws":len(diffs),"total_draws":n_draws}
    return {
        "difference_observed":(
            None if np.unique(y).size<2 else float(
                average_precision_score(y,expit(logit_a))-
                average_precision_score(y,expit(logit_b)))),
        "interval95_descriptive":list(map(float,np.quantile(diffs,[.025,.975]))),
        "unique_patient_groups":len(people),
        "valid_draws":len(diffs),
        "resample_unit":"whole held-out patient groups; overlapping windows stay together",
    }


def _fit_risk_gate(model,bank,data):
    if not bank.candidates:
        return None,[],[],"NO_MEASURED_OPERATOR_PROPOSED"
    train,v1,v2=(data[k] for k in ("train","valid1","valid2"))
    if any(np.unique(data[k]["y"]).size<2 for k in ("train","valid1","valid2")):
        return None,[],[],"ONE_CLASS_IN_VALID_CANNOT_ACCEPT_SOURCE"
    n=sum(len(data[k]["y"]) for k in ("train","valid1","valid2"))
    # V55 fit consumes no TEST arrays or risk labels.
    objects={
        "train":train,"valid1":v1,"valid2":v2,
    }
    import numpy as np
    xx=np.concatenate([train["x"],v1["x"],v2["x"]])
    yy=np.concatenate([train["y"],v1["y"],v2["y"]])
    mm=np.concatenate([train["mask"],v1["mask"],v2["mask"]])
    sl0=np.arange(len(train["y"]))
    sl1=np.arange(len(train["y"]),len(train["y"])+len(v1["y"]))
    sl2=np.arange(len(train["y"])+len(v1["y"]),n)
    candidate,allfits,gates=_fit_joint_candidates(
        bank,model,xx,yy,sl0,sl1,sl2,mm[sl0],mm[sl1],mm[sl2])
    state=("ADMITTED_TRAIN_AND_DUAL_VALID" if candidate is not None
           else "REJECTED_ALL_DUAL_VALID_GATES")
    return candidate,allfits,gates,state


def _aggregate_patient_nll(y,logit,patient):
    """Unlike pooled window NLL, this weights each TEST patient once."""
    vals=[]
    for p in set(patient.tolist()):
        choose=patient==p
        vals.append(float(log_loss(
            y[choose],expit(logit[choose]),labels=[0,1])))
    return float(np.mean(vals)),len(vals)


def execute(root:Path,out:Path,seeds=SEEDS):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    data,norm,split_manifest=load_bidmc_realworld_v58(root)
    (out/"group_split_manifest.json").write_text(
        json.dumps(split_manifest,ensure_ascii=False,indent=2)+"\n")
    tr,v1,v2,te=(data[k] for k in ("train","valid1","valid2","test"))
    if np.unique(tr["y"]).size<2:
        raise RuntimeError("One-class TRAIN on frozen real-patient split; do not retune.")
    bank=discover_operator_library_v54(
        tr["x"],tr["mask"],known_channels=(0,1,2,3))
    raw_and_energy=_baseline(tr["x"],tr["mask"],tr["y"])
    pulse_only=_train_pulse_logistic(tr)
    cases=[];receipts=[];public_predictions=[]
    for seed in seeds:
        model=native_k2(seed)
        train_receipt=run_joint_epochs_v47(
            model,tr["x"][:,:,:4],tr["mask"][:,:,:4],
            np.full(len(tr["y"]),.5,np.float32),
            np.zeros(len(tr["y"]),bool),tr["y"],
            seed=seed,consistency_weight=0.,epochs=5)
        winner,allfits,gates,disposition=_fit_risk_gate(model,bank,data)
        receipts.append({
            "native_seed":seed,
            "known_k2_train_receipt":train_receipt,
            "discovery_candidates_from_train_only":bank.manifest(),
            "candidate_status":disposition,
            "all_fit_sources_train_only":allfits,
            "valid_gate_receipts":gates,
            "source_admitted":None if winner is None else winner.passport(),
            "physical_source_identity_proven":False,
        })
        for stress in STRESS:
            x,mask,pulse=_test_variant(te,norm,stress,seed=20265888+seed)
            base=native_logits(model,x,mask)
            after,used,q,support=native_joint_prediction(
                model,base,winner,x,mask)
            z_raw=_apply_baseline(raw_and_energy[0],x,mask)
            z_eng=_apply_baseline(
                raw_and_energy[1],x,mask,energy=True)
            z_pulse=_pulse_logit(pulse,pulse_only)
            if winner is None and not np.array_equal(base,after):
                raise AssertionError("No new real measured source but changed frozen native K2")
            if not np.array_equal(base[~used],after[~used]):
                raise AssertionError("Real data missing-source exact K2 failed")
            outcomes={
                "K2":base,
                "V58_automatically_selected":after,
                "raw_logistic":z_raw,
                "energy_logistic":z_eng,
                "pulse_only_logistic":z_pulse,
            }
            score={name:_scores(te["y"],logits)
                   for name,logits in outcomes.items()}
            for name,logits in outcomes.items():
                macro,n_patients=_aggregate_patient_nll(
                    te["y"],logits,te["patient"])
                score[name]["patient_macro_NLL"]=macro
                score[name]["patient_groups"]=n_patients
            # transient local predictions for independent auditor; excluded
            # from all GitHub artifact uploads (patient-level monitor values).
            artifact=out/"private_case_arrays"
            artifact.mkdir(parents=True,exist_ok=True)
            key=f"seed{seed}_{stress}"
            path=artifact/f"{key}.npz"
            np.savez_compressed(path,y=te["y"],base=base,selected=after,
                raw=z_raw,engineered=z_eng,pulse_only=z_pulse,
                source_used=used,source_q=q,source_support=support,
                # Repeated patient IDs replaced by anonymous integer codes.
                patient_group=np.unique(te["patient"],return_inverse=True)[1])
            public_predictions.append({
                "case":key,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_risk_decomposition_checked":True,
                "private_individual_predictions_not_uploaded":True})
            cases.append({
                "seed":seed,"stress":stress,
                "source_status":disposition,
                "candidate_proposals":len(bank.candidates),
                "source_usage_fraction":float(used.mean()),
                "mean_available_input_fraction":float(mask.mean()),
                "real_patient_groups_in_test":len(set(te["patient"])),
                "direct_monitor":_pulse_score(te["y"],pulse,te["hr"]),
                "test_metrics":score,
                "paired_patient_bootstrap_vs_K2":
                    _bootstrap_by_real_patient(
                        te["y"],after,base,te["patient"],
                        seed=20265855+seed),
                "paired_patient_bootstrap_vs_energy_logistic":
                    _bootstrap_by_real_patient(
                        te["y"],after,z_eng,te["patient"],
                        seed=20265877+seed),
                "clinical_diagnostic_or_future_forecast":False,
            })
            print("V58",seed,stress,disposition,
                  "K2",score["K2"]["auprc"],
                  "selected",score["V58_automatically_selected"]["auprc"],
                  flush=True)
    report={
        "protocol":PROTOCOL,
        "status":"EXECUTED_REAL_MEASUREMENT_EXPLORATORY",
        "data_origin":"PhysioNet BIDMC PPG and Respiration v1.0.0, ODC-By",
        "dataset_DOI":"10.13026/C2208R",
        "record_count":53,
        "split_manifest":"group_split_manifest.json",
        "frozen_seed_split":20265813,
        "native_model_seeds":list(seeds),
        "n_models":len(receipts),
        "n_test_conditions":len(cases),
        "n_operator_candidates":len(bank.candidates),
        "real_target":"same-time ECG monitor HR >=100 bpm, not prognosis",
        "HR_is_both_training_label_and_test_reference":True,
        "no_independent_assay_concept_certificate":True,
        "no_HR_in_input_features_or_scaler":True,
        "no_medical_or_causal_interpretation":True,
        "window_leakage_prevented_by_patient_group":True,
        "patient_bootstrap_resamples_whole_groups":True,
        "individual_patient_values_exported":False,
        "case_receipts":receipts,
        "cases":cases,
        "private_array_hash_manifest":public_predictions,
    }
    (out/"report_v58.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    with (out/"aggregate_cases_v58.csv").open(
        "w",newline="",encoding="utf8") as f:
        writer=csv.writer(f)
        writer.writerow(["seed","stress","source_status",
            "n_candidates","usage_fraction",
            "k2_auprc","v58_auprc","engineered_auprc",
            "k2_nll","v58_nll","engineered_nll"])
        for case in cases:
            s=case["test_metrics"]
            writer.writerow([
                case["seed"],case["stress"],case["source_status"],
                case["candidate_proposals"],case["source_usage_fraction"],
                s["K2"]["auprc"],s["V58_automatically_selected"]["auprc"],
                s["energy_logistic"]["auprc"],
                s["K2"]["nll"],s["V58_automatically_selected"]["nll"],
                s["energy_logistic"]["nll"],
            ])
    print(json.dumps({"status":report["status"],
        "candidates":len(bank.candidates),
        "test_windows":len(te["y"]),
        "patients":len(set(te["patient"])),
        "cases":len(cases)},indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS))
    a=p.parse_args()
    execute(a.data,a.out,tuple(a.seeds))
