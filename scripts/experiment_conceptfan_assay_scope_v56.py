#!/usr/bin/env python3
"""V56: new synthetic cohort with genuinely independent *purchased* external assay.

Original ConceptFAN V55 is trained unchanged. Two separate gates are required:
prediction usefulness (TRAIN Y + dual VALID Y) and external measured-target
corroboration (purchased TRAIN + dual VALID assay). Then a frozen TRAIN-only
source-mean monitor may revoke the source using unlabeled TEST sensor data.

No TEST assay, hidden simulator factors or TEST Y are passed to any gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from conceptfan_discovery.operator_library_concepts_v54 import (
    MeasuredOperatorConceptV54,discover_operator_library_v54,
)
from conceptfan_discovery.constrained_joint_operator_v55 import (
    JointMeasuredConceptFitV55,
)
from conceptfan_discovery.independent_assay_scope_v56 import (
    fit_independent_assay_gate_v56,assess_observed_source_scope_v56,
)
from scripts.experiment_conceptfan_joint_operator_v55 import (
    _fit_joint_candidates,native_joint_prediction,
)
from scripts.experiment_conceptfan_operator_library_v54 import (
    TRAIN_N,VALID_N,TEST_N,STRESS as V55_STRESS,MODES as V55_MODES,
    _baseline,_apply_baseline,masked,metrics,
    propose_train_and_gate,native_added_logits,
    native_k2,native_logits,run_joint_epochs_v47,
)

PROTOCOL="CONCEPTFAN_V56_INDEPENDENT_MEASUREMENT_COHORT_REVOKE"
SEED=20265621
SEEDS=(11,23)
MODES=("nonlinear_energy","irrelevant_delays",
       "predictive_recording_artifact","reference_permuted")
STRESS=("complete","MCAR20","MCAR40",
        "source_sign_flip","source_gain_1p5","source_missing")
DEVICE_ID="SYNTHETIC_DEVICE_V56"
REFERENCE_ID="SYNTHETIC_INDEPENDENT_ENERGY_ASSAY_SOURCE_4"
BUDGET_TRAIN=190
BUDGET_VALID=100


def generate(seed=SEED):
    """Return observed X,Y and ONLY purchased independent TRAIN/VALID assay.

    Latent source trajectories and the unpurchased TEST assay do not escape
    this generator. External reference has its own independent noise draw,
    not an imputed copy of the source/readout device sensors.
    """
    rng=np.random.default_rng(seed)
    n=TRAIN_N+2*VALID_N+TEST_N
    t=16
    x=np.empty((n,t,12),dtype=np.float32)
    a,b=rng.normal(size=(2,n))
    for channel,signal in ((0,a),(1,a),(2,b),(3,b)):
        x[:,:,channel]=signal[:,None]+.20*rng.normal(size=(n,t))
    energy={}
    for i,j,kind in ((4,5,"even"),(6,7,"signed"),(8,9,"even")):
        amp=np.exp(.20*rng.normal(size=(n,1)))
        hidden=rng.normal(size=(n,t))*amp
        x[:,:,i]=hidden+.035*rng.normal(size=(n,t))
        x[:,:2,j]=rng.normal(size=(n,2))
        if kind=="even":
            sign=rng.choice(np.array([-1.,1.]),size=(n,t-2))
            x[:,2:,j]=sign*hidden[:,:-2]+.035*rng.normal(size=(n,t-2))
        else:
            x[:,2:,j]=hidden[:,:-2]+.035*rng.normal(size=(n,t-2))
        energy[i]=np.mean(hidden[:,8:14]**2,axis=1)
    x[:,:,10:]=rng.normal(size=(n,t,2))
    # A new independent measurement instrument observes target E_4
    # imperfectly. Its reference is not a synthetic generator oracle passed
    # to the predictor. Only purchased readings in TRAIN/VALID are exposed.
    independent_assay=np.maximum(0.,energy[4]+.18*rng.normal(size=n))
    known=.95*a-.75*b-.25
    risk={}
    for mode in MODES:
        if mode in ("nonlinear_energy","reference_permuted"):
            extra=np.clip(energy[4]-1.,-1.2,2.4)
        elif mode=="predictive_recording_artifact":
            extra=np.clip(energy[8]-1.,-1.2,2.4)
        else:
            extra=np.zeros(n)
        risk[mode]=expit(known+3.*extra)
    # The broken reference has identical observed targets and X to useful
    # case: only reference identity pairing is corrupted.
    y={mode:(rng.random(n)<risk[mode]).astype(np.uint8)
       for mode in MODES if mode!="reference_permuted"}
    y["reference_permuted"]=y["nonlinear_energy"].copy()
    tr=np.arange(TRAIN_N)
    v1=np.arange(TRAIN_N,TRAIN_N+VALID_N)
    v2=np.arange(TRAIN_N+VALID_N,TRAIN_N+2*VALID_N)
    te=np.arange(n-TEST_N,n)
    partitions={"train":tr,"valid1":v1,"valid2":v2,"test":te}
    masks={"train":masked(x[tr],"MCAR20",seed=seed+2),
           "valid1":masked(x[v1],"MCAR20",seed=seed+3),
           "valid2":masked(x[v2],"MCAR20",seed=seed+4)}
    r=np.random.default_rng(seed+100)
    purchased={}
    for label,budget in (("train",BUDGET_TRAIN),
                         ("valid1",BUDGET_VALID),("valid2",BUDGET_VALID)):
        indices=np.sort(r.choice(len(partitions[label]),budget,replace=False))
        purchased[label]={"indices":indices,
                          "readings":independent_assay[partitions[label]][indices]}
    # Only independent assays explicitly purchased before TEST. There is
    # deliberately no array of the unpurchased source energy in return.
    return {"x":x,"y":y,**partitions,"masks":masks,
            "purchased_assay":purchased,
            "test_assay_available_to_inference":False,
            "generator_latent_sources_returned":False}


def _fit_from_passport(payload):
    data=dict(payload)
    data["candidate"]=MeasuredOperatorConceptV54(**data["candidate"])
    return JointMeasuredConceptFitV55(**data)


def _fit_candidates_and_assay(world,bank,model,mode):
    x,y=world["x"],world["y"][mode]
    tr,v1,v2=(world[k] for k in ("train","valid1","valid2"))
    m0,m1,m2=(world["masks"][k] for k in ("train","valid1","valid2"))
    risk_selected,all_fit_passports,risk_receipts=_fit_joint_candidates(
        bank,model,x,y,tr,v1,v2,m0,m1,m2)
    fits={p["candidate"]["candidate_id"]:_fit_from_passport(p)
          for p in all_fit_passports}
    gate_by_id={g["candidate_id"]:g for g in risk_receipts}
    budget=world["purchased_assay"]
    assays={}
    control=(mode=="reference_permuted")
    for name in ("train","valid1","valid2"):
        values=budget[name]["readings"].copy()
        if control:
            values=np.random.default_rng(SEED+{
                "train":401,"valid1":402,"valid2":403}[name]).permutation(values)
        assays[name]=(budget[name]["indices"],values)
    evidence=[]
    for candidate in bank.candidates:
        fit=fits[candidate.candidate_id]
        cert=fit_independent_assay_gate_v56(
            fit,x[tr],m0,*assays["train"],
            x[v1],m1,*assays["valid1"],
            x[v2],m2,*assays["valid2"],
            device_id=DEVICE_ID,reference_id=REFERENCE_ID,
            seed=SEED+124)
        evidence.append(cert)
    approved=[(fits[e.candidate_id],e,gate_by_id[e.candidate_id])
              for e in evidence
              if e.dual_VALID_corroborated and
                 gate_by_id[e.candidate_id]["approved"]]
    approved.sort(key=lambda item:(
        -item[2]["min_two_env_gain"],item[0].candidate.candidate_id))
    if approved:
        selected,reference,selected_risk=approved[0]
    else:
        selected=reference=selected_risk=None
    return {
        "unrestricted_v55":risk_selected,
        "selected_v56":selected,
        "certificate_v56":reference,
        "all_risk_gates":risk_receipts,
        "all_assay_gates":[e.passport() for e in evidence],
        "all_joint_fits":all_fit_passports,
        "selected_risk_gate":selected_risk,
        "assay_budgets":{"train":BUDGET_TRAIN,
                         "valid1":BUDGET_VALID,"valid2":BUDGET_VALID},
        "assay_selection_with_no_test_assay":True,
    }


def _native_selected_with_reference(
    model,baseline,fit,certificate,x,mask,
    *,device_id=DEVICE_ID,reference_id=REFERENCE_ID,
):
    if fit is None:
        return (baseline.copy(),np.zeros(len(x),bool),
                np.full(len(x),.5,np.float64),np.zeros(len(x)),
                {"status":"ABSTAIN_NO_DUAL_GATED_SOURCE",
                 "approved":False,"physical_or_causal_identity_proven":False,
                 "external_assay_read_at_test":False})
    assessment=assess_observed_source_scope_v56(
        fit,certificate,x,mask,device_id=device_id,
        reference_id=reference_id)
    q=assessment["source_q"];sup=assessment["source_support"]
    used=assessment["source_used"]
    with torch.no_grad():
        model.gamma.fill_(float(fit.gamma))
    z=native_logits(model,x,mask,q.astype(np.float32),used)
    expected=baseline+fit.gamma*np.where(used,q-.5,0.)
    if not np.allclose(z,expected,atol=2e-6,rtol=0):
        raise AssertionError("V56 reference-gated source bypassed native FAN")
    if not np.array_equal(z[~used],baseline[~used]):
        raise AssertionError("V56 exact original K2 fallback failed")
    return z,used,q,sup,assessment


def one_run(world,mode,seed,destination):
    x,y=world["x"],world["y"][mode]
    tr,v1,v2,te=(world[k] for k in ("train","valid1","valid2","test"))
    m0,m1,m2=(world["masks"][k] for k in ("train","valid1","valid2"))
    bank=discover_operator_library_v54(x[tr],m0,known_channels=(0,1,2,3))
    if len(bank.candidates)<2:
        raise RuntimeError("Nontrivial measured operator candidates absent")
    model=native_k2(seed)
    known_training=run_joint_epochs_v47(
        model,x[tr,:,:4],m0[:,:,:4],
        np.full(len(tr),.5,np.float32),
        np.zeros(len(tr),bool),y[tr],seed=seed,consistency_weight=0.,epochs=5)
    fits=_fit_candidates_and_assay(world,bank,model,mode)
    frozen_v54,gamma_v54,v54_receipts=propose_train_and_gate(
        bank,"operator",model,x,y,m0,m1,m2,tr,v1,v2)
    raw,engineered=_baseline(x[tr],m0,y[tr])
    native_fit=fits["unrestricted_v55"]
    selected=fits["selected_v56"]
    certificate=fits["certificate_v56"]
    anchor=selected if selected is not None else native_fit
    stressed_channels=(anchor.candidate.source_channel,anchor.candidate.readout_channel
                       ) if anchor is not None else (4,5)
    rows=[]
    destination.mkdir(parents=True,exist_ok=True)
    for scenario in STRESS:
        xt=x[te].copy()
        mask=masked(
            xt,"MCAR20" if scenario in (
                "source_sign_flip","source_gain_1p5","source_missing") else scenario,
            seed=20265650+seed)
        if scenario in ("source_sign_flip","source_gain_1p5"):
            multiplier=-1. if scenario=="source_sign_flip" else 1.5
            for channel in stressed_channels:xt[:,:,channel]*=multiplier
        if scenario=="source_missing":
            for channel in stressed_channels:mask[:,:,channel]=False
        base=native_logits(model,xt,mask)
        v54_z,v54_used,_,_=native_added_logits(
            model,base,frozen_v54,gamma_v54,xt,mask,"operator")
        v55_z,v55_used,v55_q,v55_sup=native_joint_prediction(
            model,base,native_fit,xt,mask)
        v56_z,v56_used,v56_q,v56_sup,scope=_native_selected_with_reference(
            model,base,selected,certificate,xt,mask)
        raw_z=_apply_baseline(raw,xt,mask)
        energy_z=_apply_baseline(engineered,xt,mask,energy=True)
        np.savez_compressed(
            destination/f"{mode}_seed{seed}_{scenario}.npz",
            y=y[te],k2_logits=base,v54_logits=v54_z,
            v55_logits=v55_z,v56_logits=v56_z,
            raw_logits=raw_z,energy_logits=energy_z,
            used_v55=v55_used,used_v56=v56_used,
            q_v55=v55_q,q_v56=v56_q,
            support_v55=v55_sup,support_v56=v56_sup,
        )
        rows.append({
            "mode":mode,"seed":seed,"stress":scenario,
            "unrestricted_source":None if native_fit is None
                                  else native_fit.candidate.candidate_id,
            "assay_selected_source":None if selected is None
                                    else selected.candidate.candidate_id,
            "v56_status":scope["status"],
            "v55_usage":float(v55_used.mean()),"v56_usage":float(v56_used.mean()),
            "K2":metrics(y[te],base),
            "V54_frozen":metrics(y[te],v54_z),
            "V55_unrestricted":metrics(y[te],v55_z),
            "V56_corroborated":metrics(y[te],v56_z),
            "raw_logistic":metrics(y[te],raw_z),
            "energy_logistic":metrics(y[te],energy_z),
            "source_signed_path_verified":True,
            "no_test_assay_or_physical_label_inference":True,
        })
    receipt={
        "mode":mode,"seed":seed,
        "known_training":known_training,
        "source_proposals":bank.manifest(),
        "unrestricted_v55":None if native_fit is None
                           else native_fit.passport(),
        "selected_v56":None if selected is None else selected.passport(),
        "selected_reference":None if certificate is None else certificate.passport(),
        "all_risk_gates":fits["all_risk_gates"],
        "all_assay_gates":fits["all_assay_gates"],
        "assay_budget":fits["assay_budgets"],
        "assay_bought_only_train_valid":True,
        "selected_reference_id":REFERENCE_ID if certificate else None,
        "v54_selected":None if frozen_v54 is None else frozen_v54.passport(),
        "v54_train_gamma":float(gamma_v54),
        "v54_predictive_gates":v54_receipts,
        "source_stress_channels":list(stressed_channels),
    }
    return rows,receipt


def execute(out:Path,seeds=SEEDS):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    world=generate()
    rows=[];receipts=[]
    for mode in MODES:
        for seed in seeds:
            rr,rec=one_run(world,mode,int(seed),out/"predictions")
            rows.extend(rr);receipts.append(rec)
            print("V56",mode,seed,"predictive",
                  None if rec["unrestricted_v55"] is None else
                  rec["unrestricted_v55"]["candidate"]["source_channel"],
                  "assay",
                  None if rec["selected_v56"] is None else
                  rec["selected_v56"]["candidate"]["source_channel"],
                  flush=True)
    cols=("K2","V54_frozen","V55_unrestricted",
          "V56_corroborated","raw_logistic","energy_logistic")
    csvpath=out/"cases_v56.csv"
    with csvpath.open("w",newline="",encoding="utf8") as fp:
        w=csv.writer(fp)
        w.writerow(["mode","seed","stress","status","v55_usage","v56_usage",
                    *[c+"_AUPRC" for c in cols],
                    *[c+"_NLL" for c in cols]])
        for item in rows:
            w.writerow([item["mode"],item["seed"],item["stress"],
                item["v56_status"],item["v55_usage"],item["v56_usage"],
                *[item[c]["auprc"] for c in cols],
                *[item[c]["nll"] for c in cols]])
    report={
        "protocol":PROTOCOL,"status":"EXECUTED_EXPLORATORY",
        "new_generator_seed":SEED,"n_train":TRAIN_N,
        "n_valid_each":VALID_N,"n_test":TEST_N,
        "native_seeds":list(seeds),"n_native_models":len(receipts),
        "n_test_rows":len(rows),"assay_purchased_train":BUDGET_TRAIN,
        "assay_purchased_valid_each":BUDGET_VALID,
        "test_assay_purchased":0,"test_assay_accessed_for_inference":False,
        "hidden_generator_latents_accessed_during_training":False,
        "physical_or_causal_source_identity_proven":False,
        "source_predictive_gate_valid_only":True,
        "external_assay_gate_train_valid_only":True,
        "train_only_q_cohort_monitor":True,
        "exact_frozen_native_K2_on_refusal":True,
        "rows":rows,"receipts":receipts,
        "prediction_files":[{"path":str(p.relative_to(out)),
            "sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted((out/"predictions").glob("*.npz"))],
        "cases_csv_sha256":hashlib.sha256(csvpath.read_bytes()).hexdigest(),
        "limits":"Synthetic external reference with independent noise; measured-target corroboration only; paired IDs not cryptographically authenticated; no causal/clinical truth; cohort drift monitor cannot detect invariant shifts."
    }
    (out/"report_v56.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"status":report["status"],"native_models":len(receipts),
                      "test_rows":len(rows),
                      "checksum":report["cases_csv_sha256"]}))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS))
    a=p.parse_args()
    execute(a.out,tuple(a.seeds))
