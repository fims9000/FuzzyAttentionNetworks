#!/usr/bin/env python3
"""V55 independent synthetic TEST of TRAIN-joint constrained measured concepts.

V54 and V55 use exactly the same newly generated cohort, K2 predictions,
Y label budgets, two VALID cohorts and external competitor features.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from conceptfan_discovery.operator_library_concepts_v54 import (
    discover_operator_library_v54,infer_operator_concept_v54,
)
from conceptfan_discovery.lagged_operator_concepts_v53 import (
    discover_lagged_concepts_v53,
)
from conceptfan_discovery.constrained_joint_operator_v55 import (
    fit_joint_operator_v55,infer_joint_operator_v55,
)
from conceptfan_discovery.auto_measured_concept_admission import (
    evaluate_two_independent_environments,select_at_most_one_candidate,
)
from scripts.experiment_conceptfan_operator_library_v54 import (
    generate,energy_tabular,_baseline,_apply_baseline,
    propose_train_and_gate,native_added_logits,
    TRAIN_N,VALID_N,TEST_N,STRESS,MODES,
    native_k2,native_logits,run_joint_epochs_v47,masked,metrics,
)

PROTOCOL="CONCEPTFAN_V55_CONSTRAINED_JOINT_NEW_MEASURED_OPERATOR_NATIVE"
NEW_DATA_SEED=20265505
SEEDS=(11,23)


def _fit_joint_candidates(bank,model,x,y,tr,v1,v2,tm,m1,m2):
    baselines={name:native_logits(model,x[ids],mask)
               for name,ids,mask in (("train",tr,tm),("valid1",v1,m1),
                                      ("valid2",v2,m2))}
    fitted=[];gates=[]
    for candidate in bank.candidates:
        fit=fit_joint_operator_v55(candidate,x[tr],tm,
                                   baselines["train"],y[tr])
        qa=infer_joint_operator_v55(fit,x[v1],m1)
        qb=infer_joint_operator_v55(fit,x[v2],m2)
        gate=evaluate_two_independent_environments(
            candidate.candidate_id,fit.gamma,
            ("VALID1",y[v1],baselines["valid1"],qa["summary"],qa["support"]),
            ("VALID2",y[v2],baselines["valid2"],qb["summary"],qb["support"]),
            train_label_count=len(tr),seed=20265515)
        fitted.append(fit)
        gates.append(gate)
    admitted=select_at_most_one_candidate(tuple(gates))
    fit=next((f for f in fitted if f.candidate.candidate_id==admitted),None)
    return fit,[f.passport() for f in fitted],[g.as_dict() for g in gates]


def native_joint_prediction(model,base,fit,x,mask):
    """The prediction genuinely traverses the frozen native PyTorch K2+q head."""
    if fit is None:
        return base.copy(),np.zeros(len(x),bool),np.full(len(x),.5),np.zeros(len(x))
    observed=infer_joint_operator_v55(fit,x,mask)
    q=observed["summary"]
    support=observed["support"]
    used=support>=.50
    with torch.no_grad():model.gamma.fill_(float(fit.gamma))
    logits=native_logits(model,x,mask,q.astype(np.float32),used)
    expected=base+np.where(used,fit.gamma*(q-.5),0.)
    if not np.allclose(logits,expected,rtol=0,atol=2e-6):
        raise AssertionError("Joint operator missed native concept-only risk")
    if not np.array_equal(logits[~used],base[~used]):
        raise AssertionError("Joint source rejection must exactly equal K2")
    return logits,used,q,support


def one_run(world,mode,seed,out):
    x=world["x"]
    y=world["y"][mode]
    tr,v1,v2,te=(world[k] for k in ("train","valid1","valid2","test"))
    tm,m1,m2=(world["masks"][k] for k in ("train","valid1","valid2"))
    bank=discover_operator_library_v54(x[tr],tm,known_channels=(0,1,2,3))
    signed=discover_lagged_concepts_v53(x[tr],tm,known_channels=(0,1,2,3))
    if not bank.candidates:
        raise RuntimeError("No discovered measured candidates, negative result")
    model=native_k2(seed)
    training=run_joint_epochs_v47(
        model,x[tr,:,:4],tm[:,:,:4],np.full(len(tr),.5,np.float32),
        np.zeros(len(tr),bool),y[tr],seed=seed,
        consistency_weight=0.,epochs=5)
    v54,gamma54,gate54=propose_train_and_gate(
        bank,"operator",model,x,y,tm,m1,m2,tr,v1,v2)
    v53,gamma53,gate53=propose_train_and_gate(
        signed,"signed",model,x,y,tm,m1,m2,tr,v1,v2)
    v55,allfits,gate55=_fit_joint_candidates(
        bank,model,x,y,tr,v1,v2,tm,m1,m2)
    raw,engineered=_baseline(x[tr],tm,y[tr])
    rows=[]
    out.mkdir(parents=True,exist_ok=True)
    for stress in STRESS:
        xt=x[te].copy()
        mask=masked(xt,"MCAR20" if stress in ("source_sign_flip","source_gain_1p5")
                    else stress,seed=20265550+seed)
        stressed=v55.candidate if v55 is not None else v54
        if stressed is not None and stress in ("source_sign_flip","source_gain_1p5"):
            for j in (stressed.source_channel,stressed.readout_channel):
                xt[:,:,j]*=(-1. if stress=="source_sign_flip" else 1.5)
        base=native_logits(model,xt,mask)
        v53_z,v53_used,_,_=native_added_logits(
            model,base,v53,gamma53,xt,mask,"signed")
        v54_z,v54_used,v54_q,v54_support=native_added_logits(
            model,base,v54,gamma54,xt,mask,"operator")
        v55_z,v55_used,v55_q,v55_support=native_joint_prediction(
            model,base,v55,xt,mask)
        raw_z=_apply_baseline(raw,xt,mask)
        eng_z=_apply_baseline(engineered,xt,mask,energy=True)
        np.savez_compressed(out/f"{mode}_seed{seed}_{stress}.npz",
            y=y[te],k2_logits=base,v53_logits=v53_z,
            v54_logits=v54_z,v55_logits=v55_z,
            raw_logits=raw_z,energy_logits=eng_z,
            used_v54=v54_used,used_v55=v55_used,
            q_v54=v54_q,q_v55=v55_q,
            support_v54=v54_support,support_v55=v55_support)
        rows.append({"mode":mode,"seed":seed,"stress":stress,
                     "source_v54":None if v54 is None else v54.candidate_id,
                     "source_v55":None if v55 is None else v55.candidate.candidate_id,
                     "source_family_v55":None if v55 is None else v55.candidate.family,
                     "v55_usage":float(v55_used.mean()),
                     "K2":metrics(y[te],base),
                     "V53_signed":metrics(y[te],v53_z),
                     "V54_frozen":metrics(y[te],v54_z),
                     "V55_joint":metrics(y[te],v55_z),
                     "raw_logistic":metrics(y[te],raw_z),
                     "energy_logistic":metrics(y[te],eng_z),
                     "native_signed_path_checked":True,
                     "source_identity_not_physically_certified":True})
    receipt={
        "mode":mode,"seed":seed,"native_known_k2_training":training,
        "source_discovery":bank.manifest(),
        "v54_accepted":None if v54 is None else v54.passport(),
        "v54_train_gamma":float(gamma54),
        "v54_gates":gate54,
        "v53_accepted":None if v53 is None else v53.passport(),
        "v53_train_gamma":float(gamma53),
        "v53_gates":gate53,
        "v55_all_fits_train_only":allfits,
        "v55_accepted":None if v55 is None else v55.passport(),
        "v55_gates":gate55,
    }
    return rows,receipt


def execute(out:Path,seeds=SEEDS):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    world=generate(seed=NEW_DATA_SEED)
    rows=[];receipts=[]
    for mode in MODES:
        for seed in seeds:
            rr,receipt=one_run(world,mode,int(seed),out/"predictions")
            rows.extend(rr);receipts.append(receipt)
            print("V55",mode,seed,
                  "chosen",None if receipt["v55_accepted"] is None
                  else receipt["v55_accepted"]["candidate"]["family"],
                  flush=True)
    csv_path=out/"cases_v55.csv"
    with csv_path.open("w",newline="",encoding="utf8") as fp:
        writer=csv.writer(fp)
        cols=("K2","V53_signed","V54_frozen","V55_joint",
              "raw_logistic","energy_logistic")
        writer.writerow(["mode","seed","stress","v55_source_family",
              "v55_usage",*[k+"_AUPRC" for k in cols],
              *[k+"_NLL" for k in cols]])
        for row in rows:
            writer.writerow([row["mode"],row["seed"],row["stress"],
                row["source_family_v55"],row["v55_usage"],
                *[row[k]["auprc"] for k in cols],
                *[row[k]["nll"] for k in cols]])
    report={
        "protocol":PROTOCOL,"status":"EXECUTED_EXPLORATORY",
        "new_generator_random_seed":NEW_DATA_SEED,
        "n_train":TRAIN_N,"n_valid_each":VALID_N,"n_test":TEST_N,
        "n_native_models":len(receipts),"n_registered_test_rows":len(rows),
        "seeds":list(seeds),"all_proposals_y_blind":True,
        "jointly_trainable_only_mixture_temperature_gamma":True,
        "no_raw_to_risk_shortcut":True,
        "exact_original_K2_fallback":True,
        "independent_dual_valid_risk_admission":True,
        "no_independent_physical_source_semantics":True,
        "rows":rows,"receipts":receipts,
        "prediction_files":[{"path":str(p.relative_to(out)),
                "sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sorted((out/"predictions").glob("*.npz"))],
        "cases_csv_sha256":hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "limits":"New random cohort, same synthetic structural family; artifact can be admitted, not externally certified, no causal or clinical conclusions."}
    (out/"report_v55.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"status":report["status"],"native_models":len(receipts),
                      "test_rows":len(rows)}))
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS))
    a=p.parse_args()
    execute(a.out,tuple(a.seeds))

if __name__=="__main__":
    main()
