#!/usr/bin/env python3
"""V53: discovered lag concepts enter trained native ConceptFAN signed risk.

Synthetic evaluation only. Group proposal is OUTCOME-BLIND TRAIN-only; the
risk coefficient is TRAIN-only; admission is two disjoint VALID cohorts;
the TEST is evaluated last and can fail. No physical/medical identity claim.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,log_loss
from conceptfan_discovery.lagged_operator_concepts_v53 import (
    discover_lagged_concepts_v53,infer_lagged_concept_v53,discovery_fingerprint_v53)
from conceptfan_discovery.anchored_joint_bottleneck_v47 import (
    AnchoredJointConceptBottleneckV47,run_joint_epochs_v47,predict_joint_v47)
from conceptfan_discovery.auto_measured_concept_admission import (
    evaluate_two_independent_environments,select_at_most_one_candidate)
from conceptfan_discovery.isolated_source_admission_v28 import train_source_coefficient
from fan.concept.temporal import TemporalConceptFANModel

PROTOCOL="CONCEPTFAN_V53_MASKED_LAGGED_OPERATOR_NATIVE_K2_TO_K3"
MODES=("physical_extra","irrelevant_source","predictive_artifact")
STRESS=("complete","MCAR20","MCAR40","source_sign_flip")
TRAIN_N,VALID_N,TEST_N=950,400,650
TEST_SEEDS=(11,23)


def masked(x,scenario,*,seed):
    if scenario=="complete":return np.ones_like(x,dtype=bool)
    if scenario=="MCAR20":drop=.2
    elif scenario=="MCAR40":drop=.4
    else:raise ValueError("Unregistered missingness scenario")
    return np.random.default_rng(seed).random(x.shape)>=drop


def generate(seed=20261113):
    rng=np.random.default_rng(seed)
    n=TRAIN_N+2*VALID_N+TEST_N;t=16
    x=np.empty((n,t,12),dtype=np.float32)
    known_a,known_b=rng.normal(size=(2,n))
    for col,latent in ((0,known_a),(1,known_a),(2,known_b),(3,known_b)):
        x[:,:,col]=latent[:,None]+.2*rng.normal(size=(n,t))
    unobserved=[rng.normal(size=(n,t)) for _ in range(3)]
    for (i,j),z in zip(((4,5),(6,7),(8,9)),unobserved):
        x[:,:,i]=z+.12*rng.normal(size=(n,t))
        x[:,:2,j]=rng.normal(size=(n,2))
        x[:,2:,j]=z[:,:-2]+.12*rng.normal(size=(n,t-2))
    x[:,:,10]=rng.normal(size=(n,t))
    x[:,:,11]=rng.normal(size=(n,t))
    known_logit=.95*known_a-.75*known_b-.25
    y={}
    for mode in MODES:
        if mode=="physical_extra":source=unobserved[0]
        elif mode=="predictive_artifact":source=unobserved[2]
        else:source=None
        extra=np.zeros(n) if source is None else source[:,8:14].mean(axis=1)
        y[mode]=(rng.random(n)<expit(known_logit+3.1*extra)).astype(np.int64)
    train=np.arange(TRAIN_N)
    valid1=np.arange(TRAIN_N,TRAIN_N+VALID_N)
    valid2=np.arange(TRAIN_N+VALID_N,TRAIN_N+2*VALID_N)
    test=np.arange(n-TEST_N,n)
    return dict(x=x,y=y,train=train,valid1=valid1,valid2=valid2,test=test,
        masks={"train":masked(x[train],"MCAR20",seed=seed+2),
               "valid1":masked(x[valid1],"MCAR20",seed=seed+3),
               "valid2":masked(x[valid2],"MCAR20",seed=seed+4)},
        only_generator_knows_physical_labels=True)


def native_k2(seed):
    torch.manual_seed(seed)
    encoder=TemporalConceptFANModel(
        input_dim=8,sequence_length=16,latent_dim=16,n_concepts=2,
        encoder_layers=1,encoder_heads=2,encoder_ffn=48,
        dropout=0.,temporal_mode="attention")
    return AnchoredJointConceptBottleneckV47(encoder,(.6,-.6),-.2,0.)


def native_logits(model,x,mask,q=None,support=None):
    if q is None:q=np.full(len(x),.5,np.float32)
    if support is None:support=np.zeros(len(x),bool)
    return predict_joint_v47(model,x[:,:,:4],mask[:,:,:4],q,support)["logits"]


def tabular(x,m):
    observed=np.where(m,x,0.)
    cnt=np.maximum(1,m.sum(axis=1))
    tail=np.maximum(1,m[:,-6:,:].sum(axis=1))
    return np.concatenate((observed.sum(axis=1)/cnt,
             observed[:,-6:,:].sum(axis=1)/tail,m.mean(axis=1)),axis=1)


def metrics(y,z):
    return {"auprc":float(average_precision_score(y,expit(z))),
            "nll":float(log_loss(y,expit(z),labels=[0,1])),
            "positive_rate":float(np.mean(y))}


def one_run(world,mode,seed):
    x,y=world["x"],world["y"][mode]
    tr,v1,v2,te=(world[k] for k in ("train","valid1","valid2","test"))
    tm,m1,m2=(world["masks"][k] for k in ("train","valid1","valid2"))
    manifest=discover_lagged_concepts_v53(x[tr],tm,
                                             known_channels=(0,1,2,3))
    if not manifest.candidates:
        raise RuntimeError("No measured source candidates; report as explicit negative")
    model=native_k2(seed)
    training=run_joint_epochs_v47(
        model,x[tr,:,:4],tm[:,:,:4],
        np.full(len(tr),.5,np.float32),np.zeros(len(tr),dtype=bool),
        y[tr],seed=seed,consistency_weight=0.,epochs=5)
    original={
        "train":native_logits(model,x[tr],tm),
        "v1":native_logits(model,x[v1],m1),
        "v2":native_logits(model,x[v2],m2)}
    gates=[]
    for c in manifest.candidates:
        qtr=infer_lagged_concept_v53(c,x[tr],tm)
        q1=infer_lagged_concept_v53(c,x[v1],m1)
        q2=infer_lagged_concept_v53(c,x[v2],m2)
        gamma=train_source_coefficient(original["train"],qtr["summary"],y[tr])
        gates.append(evaluate_two_independent_environments(
            c.candidate_id,gamma,
            ("VALID1",y[v1],original["v1"],q1["summary"],q1["support"]),
            ("VALID2",y[v2],original["v2"],q2["summary"],q2["support"]),
            train_label_count=len(tr),seed=20261115))
    chosen=select_at_most_one_candidate(tuple(gates))
    source=next((c for c in manifest.candidates if c.candidate_id==chosen),None)
    gamma=next((g.gamma_train_only for g in gates if g.candidate_id==chosen),0.)
    with torch.no_grad():model.gamma.fill_(float(gamma))
    raw=LogisticRegression(max_iter=400,C=1.,random_state=seed)
    raw.fit(tabular(x[tr],tm),y[tr])
    rows=[]
    for stress in STRESS:
        xt=x[te].copy()
        m=masked(xt,"MCAR20" if stress=="source_sign_flip" else stress,
                 seed=20264540+seed)
        if stress=="source_sign_flip" and source is not None:
            for j in (source.source_channel,source.readout_channel):
                xt[:,:,j]*=-1.
        baseline=native_logits(model,xt,m)
        if source is None:
            proposed=baseline.copy()
            used=np.zeros(len(te),bool)
        else:
            q=infer_lagged_concept_v53(source,xt,m)
            used=q["support"]>=.5
            proposed=native_logits(model,xt,m,q["summary"].astype(np.float32),used)
            if not np.array_equal(proposed[~used],baseline[~used]):
                raise AssertionError("Without source, exact native K2 fallback required")
            expected=baseline+float(gamma)*np.where(used,q["summary"]-.5,0.)
            if not np.allclose(proposed,expected,rtol=0,atol=2e-6):
                raise AssertionError("Discovered concept was not exact signed native decision")
        raw_z=raw.decision_function(tabular(xt,m))
        rows.append({
            "mode":mode,"seed":seed,"stress":stress,
            "chosen_candidate":chosen,
            "channels":None if source is None else
                       [source.source_channel,source.readout_channel],
            "lag":None if source is None else source.lag_steps,
            "used_fraction":float(used.mean()),
            "K2":metrics(y[te],baseline),
            "K3_auto":metrics(y[te],proposed),
            "raw_logistic":metrics(y[te],raw_z),
            "exact_fallback_checked":True,
            "physical_source_identified":False})
    return rows,{"mode":mode,"seed":seed,"known_train_receipt":training,
        "unsupervised_definition":manifest.manifest(),
        "train_only_definition_hash":discovery_fingerprint_v53(manifest),
        "all_two_VALID_gates":[g.as_dict() for g in gates],
        "selected":chosen,"source_gamma_train_only":float(gamma)}


def execute(out:Path,seeds=TEST_SEEDS):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    world=generate()
    rows=[];receipts=[]
    for mode in MODES:
        for seed in seeds:
            result,receipt=one_run(world,mode,int(seed))
            rows.extend(result);receipts.append(receipt)
            print("V53",mode,seed,"selected",receipt["selected"],flush=True)
    csvpath=out/"cases_v53.csv"
    with csvpath.open("w",newline="",encoding="utf8") as f:
        writer=csv.writer(f)
        writer.writerow(["mode","seed","stress","chosen","lag","usage",
                         "K2_AUPRC","K3_AUPRC","raw_AUPRC",
                         "K2_NLL","K3_NLL","raw_NLL"])
        for r in rows:
            writer.writerow([r["mode"],r["seed"],r["stress"],
              r["chosen_candidate"],r["lag"],r["used_fraction"],
              r["K2"]["auprc"],r["K3_auto"]["auprc"],r["raw_logistic"]["auprc"],
              r["K2"]["nll"],r["K3_auto"]["nll"],r["raw_logistic"]["nll"]])
    report={
      "protocol":PROTOCOL,"status":"EXECUTED_EXPLORATORY",
      "n_train":TRAIN_N,"n_valid_each":VALID_N,"n_test":TEST_N,
      "seeds":list(seeds),"n_native_trained_models":len(receipts),
      "frozen_source_proposal_without_Y":True,
      "external_semantic_assay_provided":False,
      "no_physical_or_causal_identification_claim":True,
      "new_concept_admission_requires_two_independent_VALID_subjects":True,
      "original_K2_exact_fallback":True,
      "risk_has_no_raw_shortcut":True,
      "results":rows,"receipts":receipts,
      "cases_csv_sha256":hashlib.sha256(csvpath.read_bytes()).hexdigest(),
      "limits":"Synthetic single generator; test-only sign inversion may fail; no clinical/physical identification or SOTA claim."}
    (out/"report_v53.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"n_results":len(rows),
                      "n_models":len(receipts),"sha256":report["cases_csv_sha256"]}))
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seeds",nargs="+",type=int,default=list(TEST_SEEDS))
    args=p.parse_args()
    execute(args.out,tuple(args.seeds))

if __name__=="__main__":main()
