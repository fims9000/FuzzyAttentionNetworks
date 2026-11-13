#!/usr/bin/env python3
"""V57 new-cohort assay-scaled concept with a *retrained* native risk head.

Never reuse a statistical V55 gamma with a different measurement-scaled q.
V56 source certificate and V57 source calibration both use purchased TRAIN and
two disjoint VALID assays; the only final TEST input is measured X and masks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from conceptfan_discovery.operator_library_concepts_v54 import (
    discover_operator_library_v54,
)
from conceptfan_discovery.assay_calibrated_concept_v57 import (
    fit_calibrated_concept_and_native_head_v57,
    predict_calibrated_source_scope_v57,
    map_to_external_measurement_v57,
)
from scripts.experiment_conceptfan_assay_scope_v56 import (
    generate,_fit_candidates_and_assay,_native_selected_with_reference,
    DEVICE_ID,REFERENCE_ID,STRESS,MODES,
)
from scripts.experiment_conceptfan_joint_operator_v55 import (
    native_joint_prediction,
)
from scripts.experiment_conceptfan_operator_library_v54 import (
    TRAIN_N,VALID_N,TEST_N,
    native_k2,native_logits,run_joint_epochs_v47,
    _baseline,_apply_baseline,energy_tabular,masked,metrics,
)

PROTOCOL="CONCEPTFAN_V57_SAME_SCALE_SOURCE_TRAIN_INFER_ASSAY"
DATA_SEED=20265713
SEEDS=(11,23)


def _fit_assay_augmented_logistic(x,m,y,source,contract):
    if source is None or contract is None:
        measured=np.full(len(x),.5)
    else:
        measured=map_to_external_measurement_v57(
            source,contract.source_contract,x,m)["q_assay"]
    features=np.column_stack([energy_tabular(x,m),measured])
    scaler=StandardScaler()
    model=LogisticRegression(max_iter=1000,C=1.,random_state=31)
    model.fit(scaler.fit_transform(features),y)
    return model,scaler


def _predict_assay_augmented_logistic(baseline,x,m,source,contract):
    model,scaler=baseline
    q=(np.full(len(x),.5) if source is None or contract is None
       else map_to_external_measurement_v57(
           source,contract.source_contract,x,m)["q_assay"])
    return model.decision_function(
        scaler.transform(np.column_stack([energy_tabular(x,m),q])))


def _native_calibrated(
    model,base,source,certificate,contract,x,m,
    *,device_id=DEVICE_ID,reference_id=REFERENCE_ID
):
    if source is None or contract is None:
        return (base.copy(),np.zeros(len(x),bool),
                np.full(len(x),.5),np.zeros(len(x)),
                {"status":"ABSTAIN_NO_ASSAY_RECALIBRATED_SOURCE",
                 "approved":False,"external_test_assay_used":False})
    scope=predict_calibrated_source_scope_v57(
        source,certificate,contract,x,m,
        device_id=device_id,reference_id=reference_id)
    q=scope["q_assay"]
    used=scope["source_used"]
    gamma=float(contract.risk_gamma_train_only)
    with torch.no_grad():model.gamma.fill_(gamma)
    z=native_logits(model,x,m,q.astype(np.float32),used)
    expected=base+gamma*np.where(used,q-.5,0.)
    if not np.allclose(z,expected,atol=2e-6,rtol=0):
        raise AssertionError("V57 calibrated q not on same native risk-head scale")
    if not np.array_equal(z[~used],base[~used]):
        raise AssertionError("Refused measured source broke frozen K2")
    return z,used,q,scope["source_support"],scope


def one_run(world,mode,seed,out):
    x,y=world["x"],world["y"][mode]
    tr,v1,v2,te=(world[k] for k in ("train","valid1","valid2","test"))
    m0,m1,m2=(world["masks"][k] for k in ("train","valid1","valid2"))
    bank=discover_operator_library_v54(x[tr],m0,known_channels=(0,1,2,3))
    model=native_k2(seed)
    known=run_joint_epochs_v47(
        model,x[tr,:,:4],m0[:,:,:4],np.full(len(tr),.5,np.float32),
        np.zeros(len(tr),bool),y[tr],seed=seed,
        consistency_weight=0.,epochs=5)
    selected=_fit_candidates_and_assay(world,bank,model,mode)
    v55=selected["unrestricted_v55"]
    v56=selected["selected_v56"]
    certificate=selected["certificate_v56"]
    contract=None
    if v56 is not None:
        ztrain=native_logits(model,x[tr],m0)
        zv1=native_logits(model,x[v1],m1)
        zv2=native_logits(model,x[v2],m2)
        contract=fit_calibrated_concept_and_native_head_v57(
            v56,certificate,
            world["purchased_assay"]["train"]["readings"],
            x[tr],m0,y[tr],ztrain,
            x[v1],m1,y[v1],zv1,
            x[v2],m2,y[v2],zv2,
            device_id=DEVICE_ID,reference_id=REFERENCE_ID)
    v57=(v56 if contract is not None and
         contract.risk_gate.approved else None)
    native_ref=(contract if v57 is not None else None)
    raw,energy=_baseline(x[tr],m0,y[tr])
    energy_with_assay=_fit_assay_augmented_logistic(
        x[tr],m0,y[tr],v57,native_ref)
    # Same signal is stressed for all arms. V57 when admitted; otherwise
    # V56 candidate, unrestricted V55 candidate, then 4/5 as fallback.
    stressed=v57 if v57 is not None else (v56 if v56 is not None else v55)
    stress_channels=((stressed.candidate.source_channel,stressed.candidate.readout_channel)
                     if stressed is not None else (4,5))
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for stress in STRESS:
        xt=x[te].copy()
        m=masked(xt,
                 "MCAR20" if stress in ("source_sign_flip","source_gain_1p5",
                                       "source_missing") else stress,
                 seed=20265755+seed)
        if stress in ("source_sign_flip","source_gain_1p5"):
            multiplier=-1. if stress=="source_sign_flip" else 1.5
            for j in stress_channels:xt[:,:,j]*=multiplier
        if stress=="source_missing":
            for j in stress_channels:m[:,:,j]=False
        original=native_logits(model,xt,m)
        z55,u55,q55,s55=native_joint_prediction(
            model,original,v55,xt,m)
        z56,u56,q56,s56,scope56=_native_selected_with_reference(
            model,original,v56,certificate,xt,m)
        z57,u57,q57,s57,scope57=_native_calibrated(
            model,original,v57,certificate,native_ref,xt,m)
        raw_z=_apply_baseline(raw,xt,m)
        energy_z=_apply_baseline(energy,xt,m,energy=True)
        matched_z=_predict_assay_augmented_logistic(
            energy_with_assay,xt,m,v57,native_ref)
        np.savez_compressed(
            out/f"{mode}_seed{seed}_{stress}.npz",
            y=y[te],k2_logits=original,
            v55_logits=z55,v56_logits=z56,v57_logits=z57,
            raw_logits=raw_z,energy_logits=energy_z,
            assay_energy_logistic_logits=matched_z,
            used_v55=u55,used_v56=u56,used_v57=u57,
            q_v55=q55,q_v56=q56,q_v57=q57,
            support_v55=s55,support_v56=s56,support_v57=s57)
        rows.append({
            "mode":mode,"seed":seed,"stress":stress,
            "source_v55":None if v55 is None else v55.candidate.candidate_id,
            "source_v56":None if v56 is None else v56.candidate.candidate_id,
            "source_v57":None if v57 is None else v57.candidate.candidate_id,
            "status_v56":scope56["status"],"status_v57":scope57["status"],
            "v56_usage":float(u56.mean()),"v57_usage":float(u57.mean()),
            "K2":metrics(y[te],original),
            "V55_unrestricted":metrics(y[te],z55),
            "V56_assay_gated":metrics(y[te],z56),
            "V57_retrained_calibrated":metrics(y[te],z57),
            "raw_logistic":metrics(y[te],raw_z),
            "energy_logistic":metrics(y[te],energy_z),
            "assay_energy_logistic":metrics(y[te],matched_z),
            "same_calibrated_q_used_in_training_and_inference":True,
            "no_test_assay_accessed":True,
        })
    receipt={
        "mode":mode,"seed":seed,"known_model":known,
        "discovery":bank.manifest(),
        "v55_unrestricted":None if v55 is None else v55.passport(),
        "v56_assay_selected":None if v56 is None else v56.passport(),
        "v56_certificate":None if certificate is None else certificate.passport(),
        "v56_risk_gates":selected["all_risk_gates"],
        "v56_assay_gates":selected["all_assay_gates"],
        "v57_calibration_contract":None if contract is None else contract.passport(),
        "v57_admitted":v57 is not None,
        "stress_channels":list(stress_channels),
        "reference_budget":{"TRAIN":190,"VALID1":100,"VALID2":100,"TEST":0},
        "physical_identity_verified":False,
    }
    return rows,receipt


def execute(out:Path,seeds=SEEDS):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    world=generate(seed=DATA_SEED)
    rows=[];receipts=[]
    for mode in MODES:
        for seed in seeds:
            case,rec=one_run(world,mode,int(seed),out/"predictions")
            rows.extend(case);receipts.append(rec)
            print("V57",mode,seed,"new_calibrated_admitted",
                  rec["v57_admitted"],flush=True)
    methods=("K2","V55_unrestricted","V56_assay_gated",
             "V57_retrained_calibrated","raw_logistic",
             "energy_logistic","assay_energy_logistic")
    csvpath=out/"cases_v57.csv"
    with csvpath.open("w",newline="",encoding="utf8") as fp:
        w=csv.writer(fp)
        w.writerow(["mode","seed","stress","source_v57",
                    "v57_status","v57_usage",
                    *[s+"_AUPRC" for s in methods],
                    *[s+"_NLL" for s in methods]])
        for r in rows:
            w.writerow([r["mode"],r["seed"],r["stress"],r["source_v57"],
                r["status_v57"],r["v57_usage"],
                *[r[s]["auprc"] for s in methods],
                *[r[s]["nll"] for s in methods]])
    report={
        "protocol":PROTOCOL,"status":"EXECUTED_EXPLORATORY",
        "fresh_generator_seed":DATA_SEED,"native_seeds":list(seeds),
        "n_train":TRAIN_N,"n_valid_each":VALID_N,"n_test":TEST_N,
        "n_models":len(receipts),"n_test_rows":len(rows),
        "assay_budget":{"TRAIN":190,"VALID1":100,"VALID2":100,"TEST":0},
        "gamma_retrained_on_calibrated_train_q":True,
        "reuses_statistical_v55_gamma_on_assay_q":False,
        "train_only_source_calibration":True,"two_valid_risk_gates":True,
        "native_concept_only_signed_risk":True,
        "physical_or_causal_semantics_proven":False,
        "no_test_assay_used_for_inference":True,
        "rows":rows,"receipts":receipts,
        "prediction_files":[{"path":str(p.relative_to(out)),
              "sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
              for p in sorted((out/"predictions").glob("*.npz"))],
        "cases_csv_sha256":hashlib.sha256(csvpath.read_bytes()).hexdigest(),
        "limits":"Single synthetic generator family; TRAIN/VALID reference not real-world authenticated; repeated VALID selection and re-admission exploratory, not confirmatory significance."
    }
    (out/"report_v57.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"status":report["status"],
                      "rows":len(rows),"models":len(receipts)}))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS))
    a=p.parse_args()
    execute(a.out,tuple(a.seeds))
