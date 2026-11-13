#!/usr/bin/env python3
"""V53 full-label-free concept vocabulary (not physically labeled concepts).

Every measured source, fuzzy threshold, and temporal innovation operator is
fitted from unlabeled TRAIN historical observations. Risk heads use TRAIN Y
and only concept summaries. No oracle, manual K2 map or TEST selection.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conceptfan_discovery.autonomous_temporal_bank_v53 import (
    AutonomousTemporalConceptFAN, event_intervals, fit_temporal_vocabulary,
)
from conceptfan_discovery.auto_measured_concept_benchmark import (
    extended_nonlinear_world, masked_histories, raw_measured_features,
)
from conceptfan_discovery.variable_factor_data import generate_variable_factor_data

DRAWS=(20261221,20261237)
TRAIN_N=1400
VALID_N=600
TEST_N=700
TREATMENTS=("stable","single_sensor_flip","common_sensor_flip",
            "source_missing","artifact_flip")
MODES=("complete","MCAR20","MCAR40","MNAR","block8")
METHODS=("state","state_plus_surprise","state_plus_surprise_guarded",
         "raw_logistic","pca_auto_k")
TRUTH_GROUPS=((0,1),(2,3),(4,5,6),(7,8),(9,10)) # EVALUATION ONLY

def raw_features(x,m):
    x=np.asarray(x,dtype=np.float64)
    m=np.asarray(m,dtype=bool)
    if x.ndim!=3 or x.shape!=m.shape:raise ValueError("raw mask mismatch")
    visible=np.where(m,x,0.)
    whole=visible.sum(axis=1)/np.maximum(1,m.sum(axis=1))
    last=visible[:,-6:].sum(axis=1)/np.maximum(1,m[:,-6:].sum(axis=1))
    return np.column_stack((whole,last,m.mean(axis=1)))

def metric(y,p):
    y=np.asarray(y).reshape(-1)
    p=np.clip(np.asarray(p,dtype=float).reshape(-1),1e-8,1-1e-8)
    if len(y)!=len(p):raise ValueError("mismatched probabilities")
    return {
        "auprc":float(average_precision_score(y,p)),
        "auroc":float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
        "nll":float((-y*np.log(p)-(1-y)*np.log1p(-p)).mean()),
        "brier":float(np.mean((p-y)**2)),
    }

def matched_nll_bootstrap(y,p1,p2,*,seed:int,repeats:int=499):
    """Paired same-subject TEST, descriptive only within synthetic family."""
    y=np.asarray(y).reshape(-1)
    a=np.clip(np.asarray(p1),1e-8,1-1e-8)
    b=np.clip(np.asarray(p2),1e-8,1-1e-8)
    losses=lambda p:-y*np.log(p)-(1-y)*np.log1p(-p)
    difference=losses(b)-losses(a) # positive => a is BETTER (lower NLL)
    rng=np.random.default_rng(seed)
    ix=rng.integers(0,len(y),size=(repeats,len(y)))
    ci=np.quantile(difference[ix].mean(axis=1),[.025,.975])
    return {"n":len(y),"mean_nll_gain_method_a_vs_b":float(difference.mean()),
            "paired_patient_ci95":ci.tolist(),"not_generator_level_ci":True}

def intervene(x,m,scenario):
    x=np.asarray(x,dtype=np.float64).copy()
    m=np.asarray(m,dtype=bool).copy()
    if scenario=="stable":pass
    elif scenario=="single_sensor_flip":x[:,:,5]*=-1
    elif scenario=="common_sensor_flip":x[:,:,4:7]*=-1
    elif scenario=="artifact_flip":x[:,:,9:11]*=-1
    elif scenario=="source_missing":m[:,:,4:7]=False
    else:raise ValueError(scenario)
    return x,m

def group_audit(vocab):
    found=[set(z.sensor_indices) for z in vocab.groups]
    truth=[set(z) for z in TRUTH_GROUPS]
    recovered=sum(any(found_group==true_group for found_group in found)
                  for true_group in truth)
    incorrect=[list(z) for z in found if z not in truth]
    return {
       "oracle_groups_evaluation_only":[list(z) for z in TRUTH_GROUPS],
       "exact_group_matches":recovered,
       "true_group_count":len(truth),
       "estimated_observable_source_K":len(found),
       "spurious_groups":incorrect,
       "unmatched_singleton_sensors":list(vocab.excluded_singletons),
       "not_a_causal_or_semantic_identity_metric":True
    }

def event_stats(model,x,m):
    z=model.predict(x,m)
    c=z["trajectories"]
    per_name={}
    for k,name in enumerate(z["names"]):
        intervals=[event_intervals(c[i,:,k]) for i in range(len(c))]
        per_name[name]={
            "fraction_episodes_with_2step_event":sum(bool(v) for v in intervals)/len(intervals),
            "mean_event_count":float(np.mean([len(v) for v in intervals])),
            "mean_activation":float(c[:,:,k].mean()),
            "mean_direct_absolute_logit_contribution":float(
                np.mean(np.abs(z["contributions"][:,k]))),
            "signed_decision_coefficient":float(z["betas"][k])
        }
    return per_name

def _test_masks(x,mode,seed):
    return masked_histories(x,mode,seed=seed)

def run_extended(draw,*,smoke=False):
    ntrain,nvalid,ntest=(360,160,160) if smoke else (TRAIN_N,VALID_N,TEST_N)
    data=extended_nonlinear_world(seed=draw,train_n=ntrain,
                                  valid_n=nvalid,test_n=ntest)
    x=data["x"]
    tr,v1,v2,te=[data[n] for n in ("train","valid_env1","valid_env2","test")]
    if len(set(tr)&set(te)) or len(set(v1)&set(te)) or len(set(v2)&set(te)):
        raise AssertionError("Overlapping TRAIN/VALID/TEST subjects")
    train_mask=_test_masks(x[tr],"MCAR20",draw+101)
    # Vocabulary generation sees only X_train, M_train. No manual source map.
    vocab=fit_temporal_vocabulary(x[tr],train_mask,seed=draw+131)
    stable={mode:_test_masks(x[te],mode,draw+201+i)
            for i,mode in enumerate(MODES)}
    src_models={}
    rows=[]
    uncertainty=[]
    passports={}
    for world in ("complete_nuisance","physical_extra","artifact_predictive"):
        y=data["outcomes"][world]
        static=AutonomousTemporalConceptFAN(vocab,include_surprise=False).fit_risk_head(
            x[tr],train_mask,y[tr])
        dynamic=AutonomousTemporalConceptFAN(vocab,include_surprise=True).fit_risk_head(
            x[tr],train_mask,y[tr])
        guarded=AutonomousTemporalConceptFAN(vocab,include_surprise=True,guard=True)
        # Guard is a post-fit safety decision. Never adapt risk weights to TEST faults.
        guarded.bias=dynamic.bias
        guarded.beta=dynamic.beta.copy()
        guarded.n_risk_train_labels=dynamic.n_risk_train_labels
        src_models[world]=(static,dynamic,guarded)
        train_raw=raw_features(x[tr],train_mask)
        raw=make_pipeline(StandardScaler(),
                          LogisticRegression(C=1.,max_iter=1200))
        raw.fit(train_raw,y[tr])
        pca=make_pipeline(StandardScaler(),
                          PCA(n_components=max(1,min(len(vocab.groups),train_raw.shape[-1]))),
                          LogisticRegression(C=1.,max_iter=1200))
        pca.fit(train_raw,y[tr])
        passports[world]={
            "static":static.passport(),"dynamic":dynamic.passport(),
            "guard":"exactly same fitted beta/bias as dynamic, source-only neutral fallback",
            "full_training_labels_for_all_heads":len(tr),
        }
        # No TEST inference performed until all heads are fitted for this world.
        for scenario in TREATMENTS:
            for mi,mode in enumerate(MODES):
                if scenario!="stable" and mode!="MCAR20":
                    continue
                mask=stable[mode]
                test_x,test_mask=intervene(x[te],mask,scenario)
                predictions={
                    "state":static.predict(test_x,test_mask),
                    "state_plus_surprise":dynamic.predict(test_x,test_mask),
                    "state_plus_surprise_guarded":guarded.predict(test_x,test_mask),
                }
                base_features=raw_features(test_x,test_mask)
                probabilities={
                    "raw_logistic":raw.predict_proba(base_features)[:,1],
                    "pca_auto_k":pca.predict_proba(base_features)[:,1],
                }
                for method,pred in predictions.items():
                    probabilities[method]=pred["probability"]
                    if pred["logit_additive_error"]>1e-12:
                        raise AssertionError("Explanations are not exact logit contributions")
                y_test=y[te]
                for method,pred in probabilities.items():
                    row={"draw":draw,"world":world,"scenario":scenario,
                         "stress":mode,"method":method,"n_test":len(te),
                         "selected_group_K":len(vocab.groups),
                         "concept_slots":len(vocab.groups)*(1 if method=="state" else 2)
                             if method.startswith("state") else None,
                         **metric(y_test,pred)}
                    if method.startswith("state"):
                        obs=predictions[method]
                        row["mean_measured_source_support"]=float(obs["support"].mean()) if obs["support"].size else 0.
                        row["mean_source_rejection_rate"]=float(
                            obs["coherence_rejected"].mean()) if obs["coherence_rejected"].size else 0.
                    else:
                        row["mean_measured_source_support"]=None
                        row["mean_source_rejection_rate"]=None
                    rows.append(row)
                if scenario=="stable" and mode=="MCAR20":
                    uncertainty.append({
                        "draw":draw,"world":world,
                        "dynamic_vs_state":matched_nll_bootstrap(
                            y_test,probabilities["state_plus_surprise"],
                            probabilities["state"],seed=draw+401),
                        "dynamic_vs_raw":matched_nll_bootstrap(
                            y_test,probabilities["state_plus_surprise"],
                            probabilities["raw_logistic"],seed=draw+402),
                    })
        passports[world]["temporal_event_counts_test_stable_MCAR20"] = event_stats(
            dynamic,x[te],stable["MCAR20"])
    # Synthetic oracle groups can be compared here and nowhere earlier.
    rec={
        "draw":draw,"source_family":"nonlinear_switching_laplace_extended_V30",
        "n_train":len(tr),"n_valid1":len(v1),"n_valid2":len(v2),"n_test":len(te),
        "vocabulary":vocab.as_dict(),"group_audit_oracle_after_fits":group_audit(vocab),
        "output_models":passports,
        "n_ordinary_oracle_concepts_used_for_training":0,
        "n_hand_assigned_K2_source_groups":0,
        "all_concepts_trained_before_binary_outcome_head":True,
        "test_cases_not_used_for_model_selection":True,
        "instrument_artifact_can_be_task_predictive":True,
        "source_physical_semantics_certified":False,
        "descriptive_nll_gain_intervals":uncertainty,
    }
    return rec,rows

def run_different_family(seed=20261241):
    """Independent generator design, specifically to falsify a clique assumption.

    No TEST on this generator: source generates frozen TRAIN and development
    VALID. This is exploratory OOD method analysis, NOT independent clinical
    external validation. The true latent factor count remains evaluator-only.
    """
    outcomes=[]
    for k in (3,5,7):
        data=generate_variable_factor_data(
            n_true_factors=k,seed=seed+k,split_seed=seed+k+97,
            n_samples=960,n_channels=8,input_steps=36,horizon=6)
        tr=data.train
        ev=data.validation
        vocab=fit_temporal_vocabulary(tr.x,tr.mask,seed=seed+k+13)
        model=AutonomousTemporalConceptFAN(vocab,include_surprise=True)
        model.fit_risk_head(tr.x,tr.mask,tr.y)
        pred=model.predict(ev.x,ev.mask)
        feat_train=raw_features(tr.x,tr.mask)
        feat_eval=raw_features(ev.x,ev.mask)
        raw=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1200))
        raw.fit(feat_train,tr.y)
        n_detected=len(vocab.groups)
        outcomes.append({
            "generator_type":"mixed_loading_variable_factor",
            "oracle_K_audit_only":k,
            "observable_group_K":n_detected,
            "extra_operational_slots":2*n_detected,
            "exact_group_semantics_identified":False,
            "model":metric(ev.y,pred["probability"]),
            "raw_logistic":metric(ev.y,raw.predict_proba(feat_eval)[:,1]),
            "source_label_free":True,
            "split_type":"DEVELOPMENT_VALID_ONLY_NO_SEALED_TEST",
            "notice":"Correlation cliques cannot universally identify overlapping mixed sources",
        })
    return outcomes

def run(out_dir:Path,*,smoke=False):
    if out_dir.exists():
        raise FileExistsError("Never overwrite experimental evidence")
    draws=DRAWS[:1] if smoke else DRAWS
    evaluations=[];rows=[]
    for seed in draws:
        report,part=run_extended(seed,smoke=smoke)
        evaluations.append(report)
        rows.extend(part)
        print(json.dumps({
            "event":"V53_DRAW_COMPLETED","draw":seed,
            "groups":[z["sensor_indices"] for z in report["vocabulary"]["groups"]],
            "n_test_rows":len(part),
            "primary":[r for r in part if r["world"]=="physical_extra"
                       and r["scenario"]=="stable" and r["stress"]=="MCAR20"],
        }),flush=True)
    family=[] if smoke else run_different_family()
    final={
        "status":"SMOKE_ONLY" if smoke else "V53_NEW_SYNTHETIC_EXPLORATORY_COMPLETE",
        "protocol":"docs/medical/CONCEPTFAN_V53_FROZEN_AUTONOMOUS_TEMPORAL_PROTOCOL_20261113.md",
        "all_train_measurement_concepts_auto_proposed":True,
        "known_groups_input":[],"true_concept_targets_used":False,
        "risk_head_label_budget_per_draw":TRAIN_N if not smoke else 360,
        "outcome_only_enters_post_definition_logistic_fit":True,
        "no_physionet_patient_test_used":True,
        "no_physical_semantics_certified":True,
        "no_superiority_guarantee":True,
        "two_draws_same_generator_family_not_two_hospitals":True,
        "studies":evaluations,
        "separate_mixed_loading_generator_family_development":family,
        "rows":rows,
    }
    out_dir.mkdir(parents=True,exist_ok=False)
    (out_dir/"STATUS.json").write_text(
        json.dumps(final,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    (out_dir/"SUMMARY.json").write_text(
        json.dumps(rows,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    with (out_dir/"SUMMARY.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({"event":"V53_DONE","status":final["status"],
                      "n_draws":len(evaluations),"n_test_metrics":len(rows),
                      "n_independent_mixed_loading_family_conditions":len(family)},
                     ensure_ascii=False),flush=True)
    return final

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",required=True,type=Path)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    run(args.out_dir,smoke=args.smoke)

if __name__=="__main__":main()
