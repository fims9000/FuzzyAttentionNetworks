#!/usr/bin/env python3
"""V55 frozen new-data test of the adaptive unsupervised concept-source router."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conceptfan_discovery.autonomous_adaptive_conceptfan_v55 import AdaptiveConceptFAN
from conceptfan_discovery.autonomous_temporal_bank_v53 import (
    AutonomousTemporalConceptFAN,fit_temporal_vocabulary
)
from conceptfan_discovery.auto_measured_concept_benchmark import (
    extended_nonlinear_world, masked_histories
)
from conceptfan_discovery.variable_factor_data import generate_variable_factor_data
from scripts.experiment_autonomous_temporal_v53 import (
    raw_features, metric, matched_nll_bootstrap
)

DRAWS=(20261331,20261333)
OUTCOME_LAWS=("complete_nuisance","physical_extra","artifact_predictive")
CONDITIONS=("stable","single_sensor_flip","common_sensor_flip","source_missing")
GROUP_METHODS=("v55_adaptive","v53_group_mean","v53_whole_guard",
               "raw_logistic","pca_auto_k")
MIXED_METHODS=("v55_adaptive","v53_group_mean","raw_logistic","pca_auto_k")


def perturb(x,mask,condition):
    a=np.asarray(x,dtype=float).copy()
    m=np.asarray(mask,dtype=bool).copy()
    if condition=="stable":
        pass
    elif condition=="single_sensor_flip":
        a[:,:,5]*=-1.
    elif condition=="common_sensor_flip":
        a[:,:,4:7]*=-1.
    elif condition=="source_missing":
        m[:,:,4:7]=False
    else:
        raise ValueError(condition)
    return a,m


def raw_baselines(x,m,y,*,n_components):
    features=raw_features(x,m)
    raw=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1200))
    raw.fit(features,y)
    pca=make_pipeline(StandardScaler(),
                      PCA(n_components=max(1,min(n_components,features.shape[-1]))),
                      LogisticRegression(C=1.,max_iter=1200))
    pca.fit(features,y)
    return raw,pca


def _readout_scores(models,raw,pca,x,m):
    outputs={name:method.predict(x,m) for name,method in models.items()}
    probs={name:z["probability"] for name,z in outputs.items()}
    features=raw_features(x,m)
    probs["raw_logistic"]=raw.predict_proba(features)[:,1]
    probs["pca_auto_k"]=pca.predict_proba(features)[:,1]
    return outputs,probs


def evaluate_group_draw(seed,*,smoke=False):
    train_n,valid_n,test_n=(360,160,160) if smoke else (1400,600,700)
    data=extended_nonlinear_world(seed=seed,train_n=train_n,
                                  valid_n=valid_n,test_n=test_n)
    x=data["x"]
    tr,te=data["train"],data["test"]
    v1,v2=data["valid_env1"],data["valid_env2"]
    if set(tr)&set(te) or set(v1)&set(te) or set(v2)&set(te):
        raise AssertionError("Overlapping cohorts")
    mtr=masked_histories(x[tr],"MCAR20",seed=seed+101)
    mte=masked_histories(x[te],"MCAR20",seed=seed+102)
    old_vocab=fit_temporal_vocabulary(x[tr],mtr,seed=seed+131)
    rows=[];boot=[];models_meta={};references={}
    for world in OUTCOME_LAWS:
        y=data["outcomes"][world]
        auto=AdaptiveConceptFAN(
            seed=seed+131,permutations=25 if smoke else 60,
            bootstraps=5 if smoke else 20
        ).fit(x[tr],mtr,y[tr])
        mean=AutonomousTemporalConceptFAN(
            old_vocab,include_surprise=False).fit_risk_head(x[tr],mtr,y[tr])
        guard=AutonomousTemporalConceptFAN(
            old_vocab,include_surprise=False,guard=True)
        guard.bias=mean.bias
        guard.beta=mean.beta.copy()
        guard.n_risk_train_labels=mean.n_risk_train_labels
        raw,pca=raw_baselines(x[tr],mtr,y[tr],n_components=auto.selected_k)
        models={
            "v55_adaptive":auto,"v53_group_mean":mean,
            "v53_whole_guard":guard
        }
        models_meta[world]=auto.public_passport()
        references[world]={"train_outcome_labels":len(tr),
                           "test_outcome_labels_never_used_for_fit":True,
                           "route":auto.route,"detected_k":auto.selected_k}
        for condition in CONDITIONS:
            test_x,test_m=perturb(x[te],mte,condition)
            outputs,prob=_readout_scores(models,raw,pca,test_x,test_m)
            for method,p in prob.items():
                z=outputs.get(method)
                rows.append({
                    "seed":seed,"world":world,"condition":condition,
                    "method":method,"N_TEST":len(te),
                    "route":auto.route,"detected_K":auto.selected_k,
                    "source_family":"EXTENDED_V30_NONLINEAR_NEW_DRAW",
                    "source_type":(
                        z.get("source_semantics","MEASURED_GROUP_PROXY_ONLY")
                        if z is not None else "DIRECT_RAW_BASELINE"),
                    "mean_source_support":(
                        float(z["support"].mean()) if z is not None and z["support"].size
                        else None),
                    **metric(y[te],p)
                })
            if world=="physical_extra":
                boot.append({
                    "condition":condition,
                    "v55_vs_raw":matched_nll_bootstrap(
                        y[te],prob["v55_adaptive"],prob["raw_logistic"],
                        seed=seed+341+CONDITIONS.index(condition)),
                    "v55_vs_mean":matched_nll_bootstrap(
                        y[te],prob["v55_adaptive"],prob["v53_group_mean"],
                        seed=seed+381+CONDITIONS.index(condition)),
                })
    return {
        "generator_seed":seed,"TRAIN":len(tr),"VALID_UNUSED":len(v1)+len(v2),
        "TEST":len(te),"group_ground_truth_for_external_audit_only":{
            "physical_extra":[4,5,6],
            "nuisance":[7,8],"recording_artifact":[9,10]},
        "distinct_test_generator_but_same_family":True,
        "no_true_concepts_or_outcome_labels_used_for_source_discovery":True,
        "no_manual_known_K2_sensor_groups":True,
        "model_passports":models_meta,"routing_receipts":references,
        "paired_test_subject_bootstrap_descriptive_only":boot,
    },rows


def evaluate_mixed(*,smoke=False):
    result=[];receipts=[]
    for k in (3,5,7):
        seed=20261373+k
        data=generate_variable_factor_data(
            n_true_factors=k,seed=seed,split_seed=seed+97,
            n_samples=960,n_channels=8,input_steps=36,horizon=6)
        tr,ev=data.train,data.validation
        auto=AdaptiveConceptFAN(
            seed=seed+19,permutations=25 if smoke else 60,
            bootstraps=5 if smoke else 20).fit(tr.x,tr.mask,tr.y)
        previous=AutonomousTemporalConceptFAN(
            fit_temporal_vocabulary(tr.x,tr.mask,seed=seed+19),
            include_surprise=False).fit_risk_head(tr.x,tr.mask,tr.y)
        raw,pca=raw_baselines(tr.x,tr.mask,tr.y,n_components=auto.selected_k)
        outputs,prob=_readout_scores({
            "v55_adaptive":auto,"v53_group_mean":previous
        },raw,pca,ev.x,ev.mask)
        receipts.append({
            "true_K_synthetic_oracle_evaluation_only":k,
            "observed_source_route":auto.route,
            "estimated_K":auto.selected_k,
            "n_train":len(tr.x),"n_development_valid":len(ev.x),
            "sealed_test_untouched":True,
            "passport":auto.public_passport(),
        })
        for method,p in prob.items():
            result.append({
                "dataset":"MIXED_LOADING_VARIABLE_K_DEVELOPMENT",
                "seed":seed,"oracle_K_eval_only":k,
                "method":method,"n_development_valid":len(ev.x),
                "route":auto.route,"detected_K":auto.selected_k,
                "not_physical_semantic_identification":True,
                **metric(ev.y,p)
            })
    return result,receipts


def run(out:Path,*,smoke=False):
    if out.exists():
        raise FileExistsError("Existing evidence is immutable")
    rows=[];group_receipts=[]
    for seed in (DRAWS[:1] if smoke else DRAWS):
        receipt,case_rows=evaluate_group_draw(seed,smoke=smoke)
        group_receipts.append(receipt)
        rows.extend(case_rows)
        print(json.dumps({
            "event":"V55_MEASURED_FAMILY","draw":seed,
            "routes":{world:x["route"] for world,x in receipt["routing_receipts"].items()},
            "primary":[r for r in case_rows if r["world"]=="physical_extra"
                      and r["condition"]=="single_sensor_flip"]},flush=True)
    mixed,mixed_receipts=evaluate_mixed(smoke=smoke)
    evidence={
        "status":"SMOKE_ONLY" if smoke else "V55_EXPLORATORY_COMPLETED_IF_ALL_CASES_PASS",
        "frozen_protocol":"docs/medical/CONCEPTFAN_V55_FROZEN_ADAPTIVE_VOCABULARY_PROTOCOL_20261113.md",
        "route_uses_only_unlabeled_train_measurements":True,
        "concept_label_supervision":False,
        "known_initial_concepts_supplied":False,
        "physical_or_clinical_semantics_certified":False,
        "risk_output_exact_signed_concept_bottleneck":True,
        "new_group_draws":group_receipts,
        "mixed_loading_development_receipts":mixed_receipts,
        "new_group_test_results":rows,
        "mixed_loading_development_results":mixed,
        "no_person_level_clinical_data":True,
        "no_superiority_promised":True,
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"STATUS.json").write_text(json.dumps(
        evidence,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf8")
    for name,cases in (("GROUP_RESULTS.csv",rows),("MIXED_RESULTS.csv",mixed)):
        with (out/name).open("w",newline="",encoding="utf8") as f:
            w=csv.DictWriter(f,fieldnames=list(cases[0]))
            w.writeheader();w.writerows(cases)
    print(json.dumps({
        "event":"V55_END","status":evidence["status"],
        "new_draws":len(group_receipts),
        "group_rows":len(rows),"mixed_rows":len(mixed),
        "selected_routes_mixed":[x["observed_source_route"] for x in mixed_receipts]
    },flush=True))
    return evidence


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",required=True,type=Path)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    run(a.out_dir,smoke=a.smoke)

if __name__=="__main__":
    main()
