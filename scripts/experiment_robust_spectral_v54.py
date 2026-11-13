#!/usr/bin/env python3
"""V54 new-data falsification for autonomous ConceptFAN robust and spectral layers."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conceptfan_discovery.autonomous_temporal_bank_v53 import (
    AutonomousTemporalConceptFAN,fit_temporal_vocabulary,
)
from conceptfan_discovery.robust_observable_concepts_v54 import (
    RobustGroupConceptFAN,fit_robust_vocabulary,
    SpectralObservableConceptFAN,fit_observable_spectral_definitions,
)
from conceptfan_discovery.auto_measured_concept_benchmark import (
    extended_nonlinear_world,masked_histories,
)
from conceptfan_discovery.variable_factor_data import generate_variable_factor_data
from scripts.experiment_autonomous_temporal_v53 import (
    raw_features,metric,matched_nll_bootstrap
)

DRAWS=(20261301,20261303)
WORLD=("complete_nuisance","physical_extra","artifact_predictive")
SCENARIOS=("stable","single_sensor_flip","common_sensor_flip","source_missing")
METHODS=("v53_state","v53_whole_source_guard",
         "v54_median_source","raw_logistic")

def perturb(x,m,scenario):
    x=np.asarray(x,dtype=float).copy()
    m=np.asarray(m,dtype=bool).copy()
    if scenario=="single_sensor_flip":
        x[:,:,5]*=-1
    elif scenario=="common_sensor_flip":
        x[:,:,4:7]*=-1
    elif scenario=="source_missing":
        m[:,:,4:7]=False
    elif scenario!="stable":
        raise ValueError(scenario)
    return x,m

def run_group(draw,*,smoke):
    ntrain,nvalid,ntest=(360,160,160) if smoke else (1400,600,700)
    data=extended_nonlinear_world(seed=draw,train_n=ntrain,valid_n=nvalid,test_n=ntest)
    x=data["x"]
    train=data["train"];test=data["test"]
    tm=masked_histories(x[train],"MCAR20",seed=draw+101)
    test_mask=masked_histories(x[test],"MCAR20",seed=draw+102)
    standard_vocab=fit_temporal_vocabulary(x[train],tm,seed=draw+131)
    robust_vocab=fit_robust_vocabulary(x[train],tm,seed=draw+131)
    if [z.sensor_indices for z in standard_vocab.groups]!=[
       z.sensor_indices for z in robust_vocab.groups]:
        raise AssertionError("Different source proposal due to pooling arm")
    result=[];boots=[]
    for world in WORLD:
        y=data["outcomes"][world]
        state=AutonomousTemporalConceptFAN(standard_vocab,include_surprise=False)
        state.fit_risk_head(x[train],tm,y[train])
        old_gate=AutonomousTemporalConceptFAN(
            standard_vocab,include_surprise=False,guard=True)
        old_gate.bias=state.bias;old_gate.beta=state.beta.copy()
        old_gate.n_risk_train_labels=state.n_risk_train_labels
        robust=RobustGroupConceptFAN(robust_vocab).fit_risk_head(
            x[train],tm,y[train])
        raw=make_pipeline(StandardScaler(),
                          LogisticRegression(C=1.,max_iter=1200))
        raw.fit(raw_features(x[train],tm),y[train])
        for scenario in SCENARIOS:
            tx,mm=perturb(x[test],test_mask,scenario)
            source_pred={
                "v53_state":state.predict(tx,mm),
                "v53_whole_source_guard":old_gate.predict(tx,mm),
                "v54_median_source":robust.predict(tx,mm),
            }
            prob={method:p["probability"] for method,p in source_pred.items()}
            prob["raw_logistic"]=raw.predict_proba(raw_features(tx,mm))[:,1]
            for method,p in prob.items():
                test_metric=metric(y[test],p)
                row={
                    "draw":draw,"world":world,"scenario":scenario,
                    "method":method,"N_TEST":len(test),
                    "detected_group_K":len(standard_vocab.groups),
                    "source_type":"MEASURED_ASSOCIATION_ONLY",
                    **test_metric
                }
                if method in source_pred:
                    z=source_pred[method]
                    err=z.get("logit_additive_error",0.)
                    if err>1e-12:raise AssertionError("Source head not additive")
                    row["source_rejected_fraction"]=float(z["coherence_rejected"].mean())
                    row["source_mean_availability"]=float(z["support"].mean())
                else:
                    row["source_rejected_fraction"]=None
                    row["source_mean_availability"]=None
                result.append(row)
            if world=="physical_extra":
                boots.append({
                    "draw":draw,"world":world,"scenario":scenario,
                    "median_vs_state":matched_nll_bootstrap(
                        y[test],prob["v54_median_source"],prob["v53_state"],
                        seed=draw+301+SCENARIOS.index(scenario)),
                    "median_vs_whole_guard":matched_nll_bootstrap(
                        y[test],prob["v54_median_source"],prob["v53_whole_source_guard"],
                        seed=draw+341+SCENARIOS.index(scenario)),
                })
    return {
        "draw":draw,
        "N_TRAIN":len(train),"N_VALID_UNTOUCHED":nvalid,"N_TEST":len(test),
        "estimated_K":len(standard_vocab.groups),
        "measured_groups":[list(z.sensor_indices) for z in standard_vocab.groups],
        "different_source_grouping_methods_in_comparison":False,
        "risk_head_uses_only_fitted_concepts":True,
        "oracle_labels_used_for_group_discovery":False,
        "paired_descriptive_bootstrap":boots,
    },result

def run_mixed_family(*,smoke):
    rows=[]
    for k in (3,5,7):
        seed=20261310+k
        data=generate_variable_factor_data(
            n_true_factors=k,seed=seed,split_seed=seed+61,
            n_samples=960,n_channels=8,input_steps=36,horizon=6)
        tr,ev=data.train,data.validation
        known=fit_temporal_vocabulary(tr.x,tr.mask,seed=seed+19)
        spectral_def=fit_observable_spectral_definitions(
            tr.x,tr.mask,seed=seed+31,
            permutations=25 if smoke else 60,
            bootstraps=5 if smoke else 20)
        spectral=SpectralObservableConceptFAN(spectral_def).fit_risk_head(
            tr.x,tr.mask,tr.y)
        measured=AutonomousTemporalConceptFAN(
            known,include_surprise=False).fit_risk_head(tr.x,tr.mask,tr.y)
        raw=make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.,max_iter=1200))
        raw.fit(raw_features(tr.x,tr.mask),tr.y)
        predicted={
            "v53_clique_state":measured.predict(ev.x,ev.mask)["probability"],
            "v54_spectral":spectral.predict(ev.x,ev.mask)["probability"],
            "raw_logistic":raw.predict_proba(raw_features(ev.x,ev.mask))[:,1],
        }
        for name,p in predicted.items():
            rows.append({
                "oracle_true_K_evaluation_only":k,
                "model":name,
                "n_train":len(tr.x),"n_dev_validation":len(ev.x),
                "detected_measurement_groups":len(known.groups),
                "detected_spectral_predictive_rank":spectral_def.rank,
                "rank_bootstrap_interval":list(
                    spectral_def.train_rank_receipt["patient_bootstrap_interval"]),
                "rank_familywise_threshold":float(
                    spectral_def.train_rank_receipt["familywise_noise_max_threshold"]),
                "status":"DEVELOPMENT_VALID_ONLY_NONSEMANTIC_SUBSPACE",
                **metric(ev.y,p)
            })
    return rows

def run(out:Path,smoke=False):
    if out.exists():raise FileExistsError("Evidence is immutable")
    group=[];rows=[]
    for d in (DRAWS[:1] if smoke else DRAWS):
        receipt,cases=run_group(d,smoke=smoke)
        group.append(receipt);rows.extend(cases)
        print(json.dumps({
            "event":"V54_GROUP_DRAW","seed":d,
            "estimated_K":receipt["estimated_K"],
            "primary":[z for z in cases if z["world"]=="physical_extra"
                       and z["scenario"]=="single_sensor_flip"]},flush=True)
    mixed=run_mixed_family(smoke=smoke)
    final={
        "status":"SMOKE_ONLY" if smoke else "V54_EXPLORATORY_EXECUTED_NEW_DRAWS",
        "pre_outcome_protocol":"docs/medical/CONCEPTFAN_V54_FROZEN_ROBUST_SPECTRAL_PROTOCOL_20261113.md",
        "nonlinear_draws":group,"nonlinear_results":rows,
        "independent_mixed_loading_results":mixed,
        "K_physically_identified":False,
        "all_source_definitions_train_only":True,
        "known_manually_supplied_K2_concepts":False,
        "raw_predictive_advantage_not_guaranteed":True,
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"STATUS.json").write_text(json.dumps(final,indent=2,allow_nan=False)+"\n")
    with (out/"GROUP_RESULTS.csv").open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    with (out/"MIXED_LOADING_RESULTS.csv").open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(mixed[0]))
        w.writeheader();w.writerows(mixed)
    print(json.dumps({"event":"V54_DONE","status":final["status"],
                      "n_group_rows":len(rows),"n_mixed_rows":len(mixed)},flush=True))
    return final

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args();run(a.out_dir,smoke=a.smoke)

if __name__=="__main__":main()
