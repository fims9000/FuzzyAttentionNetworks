#!/usr/bin/env python3
"""V54 preregistered nonlinear measured-concept native FAN experiment.

TRAIN-only operator discovery; TRAIN Y risk coefficients; separate VALID1/2
admission; TEST evaluated once, never used in tuning. Synthetic evidence only.
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
from sklearn.metrics import average_precision_score, log_loss
from sklearn.preprocessing import StandardScaler

from conceptfan_discovery.lagged_operator_concepts_v53 import (
    discover_lagged_concepts_v53, infer_lagged_concept_v53,
)
from conceptfan_discovery.operator_library_concepts_v54 import (
    discover_operator_library_v54, infer_operator_concept_v54,
    fingerprint_v54, OP_EVEN,
)
from conceptfan_discovery.anchored_joint_bottleneck_v47 import (
    run_joint_epochs_v47,
)
from conceptfan_discovery.auto_measured_concept_admission import (
    evaluate_two_independent_environments,select_at_most_one_candidate,
)
from conceptfan_discovery.isolated_source_admission_v28 import (
    train_source_coefficient,
)
from scripts.experiment_conceptfan_lagged_native_v53 import (
    native_k2,native_logits,tabular,metrics,masked,
)

PROTOCOL="CONCEPTFAN_V54_NONLINEAR_OPERATOR_LIBRARY_NATIVE"
MODES=("nonlinear_energy","irrelevant_delays","predictive_recording_artifact")
STRESS=("complete","MCAR20","MCAR40","source_sign_flip","source_gain_1p5")
SEEDS=(11,23)
TRAIN_N,VALID_N,TEST_N=950,400,650


def generate(seed=20265412):
    rng=np.random.default_rng(seed)
    n=TRAIN_N+VALID_N*2+TEST_N
    t=16
    x=np.empty((n,t,12),dtype=np.float32)
    known_a=rng.normal(size=n)
    known_b=rng.normal(size=n)
    for j,hidden in ((0,known_a),(1,known_a),(2,known_b),(3,known_b)):
        x[:,:,j]=hidden[:,None]+.20*rng.normal(size=(n,t))

    energy_by_pair={}
    for i,j,kind in ((4,5,"even"),(6,7,"signed"),(8,9,"even")):
        episode_scale=np.exp(.20*rng.normal(size=(n,1)))
        source=rng.normal(size=(n,t))*episode_scale
        x[:,:,i]=source+.035*rng.normal(size=(n,t))
        x[:,:2,j]=rng.normal(size=(n,2))
        if kind=="even":
            # Sign-randomized readout: zero signed correlation, high
            # lagged **energy** correlation; a nontrivial operator family.
            signs=rng.choice(np.array([-1.,1.]),size=(n,t-2))
            x[:,2:,j]=source[:,:-2]*signs+.035*rng.normal(size=(n,t-2))
        else:
            x[:,2:,j]=source[:,:-2]+.035*rng.normal(size=(n,t-2))
        energy_by_pair[i]=np.mean(source[:,8:14]**2,axis=1)
    x[:,:,10:]=rng.normal(size=(n,t,2))
    baseline=.95*known_a-.75*known_b-.25
    y={}
    for mode in MODES:
        if mode=="nonlinear_energy":
            extra=np.clip(energy_by_pair[4]-1.,-1.2,2.4)
        elif mode=="predictive_recording_artifact":
            extra=np.clip(energy_by_pair[8]-1.,-1.2,2.4)
        else:
            extra=np.zeros(n)
        y[mode]=(rng.random(n)<expit(baseline+3.0*extra)).astype(np.uint8)
    train=np.arange(TRAIN_N)
    valid1=np.arange(TRAIN_N,TRAIN_N+VALID_N)
    valid2=np.arange(TRAIN_N+VALID_N,TRAIN_N+2*VALID_N)
    test=np.arange(n-TEST_N,n)
    return {"x":x,"y":y,"train":train,"valid1":valid1,
            "valid2":valid2,"test":test,
            "masks":{
                "train":masked(x[train],"MCAR20",seed=seed+2),
                "valid1":masked(x[valid1],"MCAR20",seed=seed+3),
                "valid2":masked(x[valid2],"MCAR20",seed=seed+4),
            },"generator_latents_never_passed_to_training":True}


def energy_tabular(x,mask):
    """Strong matched-label engineered baseline, no hidden-source oracle."""
    raw=tabular(x,mask)
    feats=[]
    for j in range(4,x.shape[-1]):
        observed=mask[:,-8:,j]
        values=np.where(observed,x[:,-8:,j],0.)
        cnt=observed.sum(axis=1)
        energy=(values**2).sum(axis=1)/np.maximum(cnt,1)
        feats.extend((energy,observed.mean(axis=1)))
    return np.column_stack((raw,*feats))


def _baseline(xtrain,mtrain,ytrain):
    def fit(features):
        scaler=StandardScaler()
        a=scaler.fit_transform(features)
        model=LogisticRegression(max_iter=1000,C=1.,random_state=31)
        model.fit(a,ytrain)
        return model,scaler
    simple=fit(tabular(xtrain,mtrain))
    energy=fit(energy_tabular(xtrain,mtrain))
    return simple,energy


def _apply_baseline(trained,x,m,*,energy=False):
    model,scale=trained
    feats=energy_tabular(x,m) if energy else tabular(x,m)
    return model.decision_function(scale.transform(feats))


def propose_train_and_gate(discovery,kind,model,x,y,tm,m1,m2,tr,v1,v2):
    base={
        "train":native_logits(model,x[tr],tm),
        "v1":native_logits(model,x[v1],m1),
        "v2":native_logits(model,x[v2],m2)}
    evaluator=(infer_operator_concept_v54 if kind=="operator"
               else infer_lagged_concept_v53)
    receipts=[];sources={}
    for candidate in discovery.candidates:
        observed_train=evaluator(candidate,x[tr],tm)
        observed_v1=evaluator(candidate,x[v1],m1)
        observed_v2=evaluator(candidate,x[v2],m2)
        gamma=train_source_coefficient(base["train"],
                                        observed_train["summary"],y[tr])
        gate=evaluate_two_independent_environments(
            candidate.candidate_id,gamma,
            ("VALID1",y[v1],base["v1"],observed_v1["summary"],observed_v1["support"]),
            ("VALID2",y[v2],base["v2"],observed_v2["summary"],observed_v2["support"]),
            train_label_count=len(tr),seed=20265414)
        receipts.append(gate)
        sources[candidate.candidate_id]=candidate
    winner=select_at_most_one_candidate(tuple(receipts))
    gamma=next((g.gamma_train_only for g in receipts if g.candidate_id==winner),0.)
    return sources.get(winner),float(gamma),[g.as_dict() for g in receipts]


def added_logits(base,source,gamma,x,mask,kind):
    if source is None:
        return base.copy(),np.zeros(len(base),bool),np.full(len(base),.5),np.zeros(len(base))
    infer=infer_operator_concept_v54 if kind=="operator" else infer_lagged_concept_v53
    result=infer(source,x,mask)
    q=result["summary"]
    support=result["support"]
    used=support>=.50
    z=base+np.where(used,gamma*(q-.5),0.)
    if not np.array_equal(z[~used],base[~used]):
        raise AssertionError("Exact K2 fallback required")
    return z,used,q,support


def native_added_logits(model,base,source,gamma,x,mask,kind):
    """Use the actual PyTorch native signed concept bottleneck for new source.

    The scalar-numpy formulation is only an independent equality check, not
    the reported prediction path. No learned raw-input-to-risk bypass.
    """
    expected,used,q,support=added_logits(base,source,gamma,x,mask,kind)
    with torch.no_grad():
        model.gamma.fill_(float(gamma))
    native=native_logits(model,x,mask,q.astype(np.float32),used)
    if not np.allclose(native,expected,atol=2e-6,rtol=0):
        raise AssertionError("Native source contribution disagrees with signed audit")
    if not np.array_equal(native[~used],base[~used]):
        raise AssertionError("Native model failed exact K2 abstention")
    return native,used,q,support


def score(y,z):
    return metrics(y,z)


def one_run(world,mode,seed,save_dir):
    x=world["x"];y=world["y"][mode]
    tr,v1,v2,te=(world[name] for name in ("train","valid1","valid2","test"))
    tm,m1,m2=(world["masks"][name] for name in ("train","valid1","valid2"))
    operator=discover_operator_library_v54(x[tr],tm,known_channels=(0,1,2,3))
    signed=discover_lagged_concepts_v53(x[tr],tm,known_channels=(0,1,2,3))
    if not operator.candidates:raise RuntimeError("No nonlinear candidate: explicit negative study")
    model=native_k2(seed)
    training=run_joint_epochs_v47(model,x[tr,:,:4],tm[:,:,:4],
        np.full(len(tr),.5,np.float32),np.zeros(len(tr),bool),y[tr],
        seed=seed,consistency_weight=0.,epochs=5)
    chosen_v54,gamma_v54,gates_v54=propose_train_and_gate(
        operator,"operator",model,x,y,tm,m1,m2,tr,v1,v2)
    chosen_v53,gamma_v53,gates_v53=propose_train_and_gate(
        signed,"signed",model,x,y,tm,m1,m2,tr,v1,v2)
    raw,engineered=_baseline(x[tr],tm,y[tr])
    rows=[];save_dir.mkdir(parents=True,exist_ok=True)
    for stress in STRESS:
        xt=x[te].copy()
        mask=masked(xt,"MCAR20" if stress in ("source_sign_flip","source_gain_1p5")
                    else stress,seed=20265450+seed)
        if chosen_v54 is not None and stress in ("source_sign_flip","source_gain_1p5"):
            channels=(chosen_v54.source_channel,chosen_v54.readout_channel)
            for j in channels:
                xt[:,:,j]*=(-1. if stress=="source_sign_flip" else 1.5)
        base=native_logits(model,xt,mask)
        z54,used54,q54,supp54=native_added_logits(model,base,chosen_v54,gamma_v54,
                                                 xt,mask,"operator")
        z53,used53,q53,supp53=native_added_logits(model,base,chosen_v53,gamma_v53,
                                                 xt,mask,"signed")
        raw_z=_apply_baseline(raw,xt,mask)
        energy_z=_apply_baseline(engineered,xt,mask,energy=True)
        if chosen_v54 is None and not np.array_equal(base,z54):
            raise AssertionError("Refused concept changed risk")
        np.savez_compressed(save_dir/f"{mode}_seed{seed}_{stress}.npz",
            y=y[te],k2_logits=base,v53_logits=z53,v54_logits=z54,
            raw_logits=raw_z,energy_logits=energy_z,
            used_v53=used53,used_v54=used54,q_v53=q53,q_v54=q54,
            support_v53=supp53,support_v54=supp54)
        rows.append({"mode":mode,"seed":seed,"stress":stress,
            "v54_candidate":None if chosen_v54 is None else chosen_v54.candidate_id,
            "v54_family":None if chosen_v54 is None else chosen_v54.family,
            "v54_channels":None if chosen_v54 is None else
                            [chosen_v54.source_channel,chosen_v54.readout_channel],
            "v53_candidate":None if chosen_v53 is None else chosen_v53.candidate_id,
            "admitted_fraction":float(used54.mean()),
            "K2":score(y[te],base),"V53_signed":score(y[te],z53),
            "V54_library":score(y[te],z54),
            "raw_logistic":score(y[te],raw_z),
            "energy_logistic":score(y[te],energy_z),
            "exact_unavailable_fallback":True,
            "physical_concept_identity_validated":False})
    receipt={"mode":mode,"seed":seed,"known_training":training,
             "operator_proposal":operator.manifest(),
             "operator_fingerprint":fingerprint_v54(operator),
             "signed_proposal":signed.manifest(),
             "v54_gates":gates_v54,"v53_gates":gates_v53,
             "v54_selected":None if chosen_v54 is None else chosen_v54.passport(),
             "gamma_v54_train_only":gamma_v54,
             "v53_selected":None if chosen_v53 is None else chosen_v53.passport(),
             "gamma_v53_train_only":gamma_v53}
    return rows,receipt


def execute(out:Path,seeds=SEEDS):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    world=generate()
    rows=[];receipts=[]
    for mode in MODES:
        for seed in seeds:
            results,info=one_run(world,mode,int(seed),out/"predictions")
            rows.extend(results);receipts.append(info)
            print("V54",mode,seed,"selected",None if info["v54_selected"] is None else
                  (info["v54_selected"]["family"],info["v54_selected"]["source_channel"]),
                  flush=True)
    csvpath=out/"cases_v54.csv"
    with csvpath.open("w",newline="",encoding="utf8") as fp:
        w=csv.writer(fp)
        w.writerow(["mode","seed","stress","v54_family","usage",
                    "K2_AUPRC","V53_AUPRC","V54_AUPRC","raw_AUPRC","energy_AUPRC",
                    "K2_NLL","V53_NLL","V54_NLL","raw_NLL","energy_NLL"])
        for r in rows:
            w.writerow([r["mode"],r["seed"],r["stress"],r["v54_family"],
                        r["admitted_fraction"],
                        *[r[k]["auprc"] for k in ("K2","V53_signed","V54_library",
                                                "raw_logistic","energy_logistic")],
                        *[r[k]["nll"] for k in ("K2","V53_signed","V54_library",
                                               "raw_logistic","energy_logistic")]])
    report={"protocol":PROTOCOL,"status":"EXECUTED_EXPLORATORY",
            "independent_new_synthetic_generator":True,
            "train_subjects":TRAIN_N,"valid_subjects_each":VALID_N,
            "test_subjects":TEST_N,"seeds":list(seeds),
            "n_native_models":len(receipts),"n_test_rows":len(rows),
            "all_proposals_y_blind":True,
            "all_risk_admission_train_and_valid_only":True,
            "no_direct_risk_bypass":True,
            "no_physical_or_causal_identification":True,
            "sign_invariance_only_when_even_family_selected":True,
            "rows":rows,"receipts":receipts,
            "prediction_files":[{"path":str(p.relative_to(out)),
                "sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sorted((out/"predictions").glob("*.npz"))],
            "csv_sha256":hashlib.sha256(csvpath.read_bytes()).hexdigest(),
            "limitations":"Single synthetic family; TEST-only gain drift uncalibrated; artifact may pass; matched engineered logistic remains strong."}
    (out/"report_v54.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"status":report["status"],"models":len(receipts),
                      "test_rows":len(rows),"csv_sha256":report["csv_sha256"]}))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS))
    a=p.parse_args()
    execute(a.out,tuple(a.seeds))

if __name__=="__main__":main()
