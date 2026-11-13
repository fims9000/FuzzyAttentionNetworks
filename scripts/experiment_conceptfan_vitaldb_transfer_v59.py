#!/usr/bin/env python3
"""V59 unretuned native FAN transfer: BIDMC TRAIN+VALID -> VitalDB TEST only.

Training, operator proposal, normalization and source gates are entirely
BIDMC-derived. VitalDB values enter *only* external-site TEST inference.
No external Y, HR track, case metadata or diagnostic marker enters training.
No patient-level risk or raw clinical time series is uploaded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from conceptfan_discovery.bidmc_realworld_v58 import (
    load_bidmc_realworld_v58,causal_monitor_features,
    apply_train_normalization,TIME_STEPS,WINDOW_STRIDE,THRESHOLD_BPM,
)
from conceptfan_discovery.operator_library_concepts_v54 import (
    discover_operator_library_v54,
)
from scripts.experiment_conceptfan_bidmc_real_v58 import (
    _scores,_pulse_score,_test_variant,_bootstrap_by_real_patient,
    _fit_risk_gate,_train_pulse_logistic,_pulse_logit,_aggregate_patient_nll,
    STRESS,SEEDS,
)
from scripts.experiment_conceptfan_joint_operator_v55 import (
    native_joint_prediction,
)
from scripts.experiment_conceptfan_operator_library_v54 import (
    native_k2,native_logits,run_joint_epochs_v47,_baseline,_apply_baseline,
)

PROTOCOL="CONCEPTFAN_V59_EXTERNAL_HOSPITAL_BIDMC_TO_VITALDB_FROZEN"
EXTERNAL_MAX=40
EXTERNAL_MIN=30


def load_external_vitaldb(source:Path,norm):
    receipt=json.loads((source/"vitaldb_download_receipt.json").read_text())
    cases=receipt["selected_public_case_ordinals"]
    if (len(cases)<EXTERNAL_MIN or len(cases)>EXTERNAL_MAX or
        len(set(cases))!=len(cases) or not receipt["clinical_HR_not_an_input"]):
        raise RuntimeError("Unexpected external fixed enrollment")
    rows=[]
    for case in cases:
        case_path=source/f"vitaldb_case_{case}.npz"
        with np.load(case_path,allow_pickle=False) as d:
            signals=d["signals"].astype(float)
        if signals.shape!=(480,4):
            raise RuntimeError("VitalDB external monitor case is not 480x4")
        xmeas=signals[:,:3]
        h=signals[:,3]
        feature,mask=causal_monitor_features(xmeas)
        for start in range(0,len(h)-TIME_STEPS+1,WINDOW_STRIDE):
            end=start+TIME_STEPS
            current=h[end-4:end]
            good=np.isfinite(current)&(current>=25)&(current<=250)
            if good.sum()<3:continue
            hr=float(current[good].mean())
            pulse=xmeas[end-4:end,0]
            pg=np.isfinite(pulse)
            p=float(pulse[pg].mean()) if pg.sum()>=3 else float("nan")
            rows.append({
                "source_case_ordinal":int(case),
                "raw":feature[start:end].copy(),
                "mask":mask[start:end].copy(),
                "y":int(hr>=THRESHOLD_BPM),"hr":hr,"pulse":p,
            })
    if len(rows)<800:
        raise RuntimeError("Not enough external group-isolated evaluated windows")
    raw=np.asarray([r["raw"] for r in rows],dtype=np.float64)
    mask=np.asarray([r["mask"] for r in rows],dtype=bool)
    test={
        "raw":raw,"mask":mask,
        "x":apply_train_normalization(raw,mask,norm),
        "y":np.asarray([r["y"] for r in rows],dtype=np.uint8),
        "hr":np.asarray([r["hr"] for r in rows],float),
        "pulse":np.asarray([r["pulse"] for r in rows],float),
        # Unique VitalDB patient subject per case assured by download script.
        # Public case ordinals stand in as anonymous group keys for bootstrap.
        "patient":np.asarray([r["source_case_ordinal"] for r in rows]),
    }
    if np.unique(test["patient"]).size!=len(cases):
        raise AssertionError("External patient exclusion logic failed")
    return test,receipt


def execute(bidmc:Path,vitaldb:Path,out:Path,seeds=SEEDS):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    source,norm,manifest=load_bidmc_realworld_v58(bidmc)
    test,external_receipt=load_external_vitaldb(vitaldb,norm)
    train=source["train"]
    if np.unique(train["y"]).size!=2:
        raise RuntimeError("Frozen BIDMC training outcome lost one class")
    bank=discover_operator_library_v54(
        train["x"],train["mask"],known_channels=(0,1,2,3))
    # Frozen V58 experiment discovered ZERO. Changing the source bank
    # secretly between independent hospital tests invalidates this transfer.
    if len(bank.candidates)!=0:
        raise AssertionError("BIDMC-V58 candidate set drift: would constitute new model")
    trained_logistics=_baseline(
        train["x"],train["mask"],train["y"])
    pulse_logistic=_train_pulse_logistic(train)
    results=[];receipts=[];private=[]
    for seed in seeds:
        model=native_k2(seed)
        known=run_joint_epochs_v47(
            model,train["x"][:,:,:4],train["mask"][:,:,:4],
            np.full(len(train["y"]),.5,np.float32),
            np.zeros(len(train["y"]),bool),train["y"],seed=seed,
            consistency_weight=0.,epochs=5)
        winner,all_fits,gates,state=_fit_risk_gate(model,bank,source)
        if winner is not None or state!="NO_MEASURED_OPERATOR_PROPOSED":
            raise AssertionError("Original BIDMC source admission altered")
        receipts.append({
            "native_seed":seed,"BIDMC_K2_TRAIN":known,
            "BIDMC_operator_bank":bank.manifest(),
            "BIDMC_source_gate_state":state,
            "source_admitted":None,
            "train_site":"BIDMC ICU Boston, USA",
            "external_site":"VitalDB operating rooms Seoul, South Korea",
            "trained_on_any_external_VitalDB_values":False,
        })
        for stress in STRESS:
            x,m,pulse=_test_variant(test,norm,stress,seed=20265988+seed)
            base=native_logits(model,x,m)
            new,used,q,support=native_joint_prediction(
                model,base,winner,x,m)
            if used.any() or not np.array_equal(base,new):
                raise AssertionError("New concept appeared in frozen transfer")
            z_raw=_apply_baseline(trained_logistics[0],x,m)
            z_energy=_apply_baseline(
                trained_logistics[1],x,m,energy=True)
            z_pulse=_pulse_logit(pulse,pulse_logistic)
            scores={
                "K2":base,
                "V59_frozen_source_FAN":new,
                "raw_logistic":z_raw,
                "energy_logistic":z_energy,
                "pulse_only_logistic":z_pulse,
            }
            measurements={}
            for name,logits in scores.items():
                group_macro,_=_aggregate_patient_nll(
                    test["y"],logits,test["patient"])
                measurements[name]={
                    **_scores(test["y"],logits),
                    "patient_macro_NLL":group_macro,
                }
            folder=out/"private_case_arrays"
            folder.mkdir(parents=True,exist_ok=True)
            key=f"seed{seed}_{stress}"
            path=folder/f"{key}.npz"
            np.savez_compressed(path,
                y=test["y"],case_group=test["patient"],
                base=base,selected=new,raw=z_raw,engineered=z_energy,
                pulse_only=z_pulse,used=used,q=q,support=support)
            private.append({
                "case":key,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "private_person_predictions_not_in_artifact":True})
            results.append({
                "seed":seed,"stress":stress,
                "source_status":state,"new_source_count":0,
                "new_source_usage_fraction":float(used.mean()),
                "external_test_patient_groups":len(set(test["patient"])),
                "heldout_class_positive_rate":float(test["y"].mean()),
                "direct_PULSE_vs_HR":_pulse_score(
                    test["y"],pulse,test["hr"]),
                "method_metrics":measurements,
                "subject_cluster_bootstrap_new_vs_energy":
                    _bootstrap_by_real_patient(
                        test["y"],new,z_energy,test["patient"],
                        seed=20265920+seed),
                "external_medical_utility_proven":False,
            })
            print("V59 EXTERNAL",seed,stress,
                  "TEST prevalence",round(test["y"].mean(),4),
                  "K2 AUPRC",measurements["K2"]["auprc"],
                  "energy",measurements["energy_logistic"]["auprc"],
                  flush=True)
    report={
        "protocol":PROTOCOL,
        "status":"EXECUTED_INDEPENDENT_HOSPITAL_EXPLORATORY",
        "training_source":"BIDMC, PhysioNet DOI 10.13026/C2208R",
        "test_source":"VitalDB Seoul open dataset, CC BY 4.0",
        "source_sites_disjoint":True,
        "source_predeclared_VitalDB_selection_seed":20265913,
        "frozen_BIDMC_train_seed":20265813,
        "n_external_test_independent_cases":len(
            external_receipt["selected_public_case_ordinals"]),
        "n_external_test_windows":len(test["y"]),
        "n_native_models":len(receipts),"n_test_runs":len(results),
        "test_HR_is_supervised_target_and_external_monitor_reference":True,
        "external_HR_absent_from_model_inputs":True,
        "no_BIDMC_source_new_concepts":True,
        "no_new_sources_accepted_using_test_site":True,
        "no_V56_external_source_certification":True,
        "input_RESP_mode_differs_impedance_vs_capnography":True,
        "patient_ID_values_exported":False,
        "raw_external_patient_risks_exported":False,
        "BIDMC_split_counts":{
            k:{z:val[z] for z in ("patient_group_count","windows",
                                  "positive_prevalence")}
            for k,val in manifest["splits"].items()},
        "external_public_case_ordinals":
            external_receipt["selected_public_case_ordinals"],
        "cases":results,"frozen_train_receipts":receipts,
        "private_prediction_hashes":private,
        "limitations":"Two genuinely different hospitals; source monitor numerics may not be algorithmically independent; RESP measurement modality differs; no new operator selected at TRAIN; site-wide HR threshold prevalence can change; TEST labels not used to tune."
    }
    (out/"report_v59.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    with (out/"aggregate_cases_v59.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.writer(f)
        writer.writerow(["seed","stress","n_external_cases","test_windows",
            "positive_rate","K2_AUPRC","K2_NLL","energy_logistic_AUPRC",
            "energy_logistic_NLL","pulse_only_AUPRC","pulse_only_NLL"])
        for item in results:
            m=item["method_metrics"]
            writer.writerow([
                item["seed"],item["stress"],item["external_test_patient_groups"],
                m["K2"]["n_windows"],item["heldout_class_positive_rate"],
                m["K2"]["auprc"],m["K2"]["nll"],
                m["energy_logistic"]["auprc"],m["energy_logistic"]["nll"],
                m["pulse_only_logistic"]["auprc"],
                m["pulse_only_logistic"]["nll"],
            ])
    print(json.dumps({"status":report["status"],
                      "external_subject_groups":report["n_external_test_independent_cases"],
                      "windows":len(test["y"]),"runs":len(results)},indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--bidmc",type=Path,required=True)
    p.add_argument("--vitaldb",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS))
    args=p.parse_args()
    execute(args.bidmc,args.vitaldb,args.out,tuple(args.seeds))
