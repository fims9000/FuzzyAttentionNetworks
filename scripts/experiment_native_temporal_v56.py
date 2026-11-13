#!/usr/bin/env python3
"""V56 executable self-supervised native temporal ConceptFAN comparison."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conceptfan_discovery.native_temporal_fuzzy_v56 import (
    NativeSelfSupervisedConceptFAN,TrainOnlySourceBank
)
from conceptfan_discovery.autonomous_adaptive_conceptfan_v55 import AdaptiveConceptFAN
from conceptfan_discovery.auto_measured_concept_benchmark import (
    extended_nonlinear_world,masked_histories
)
from conceptfan_discovery.variable_factor_data import generate_variable_factor_data
from scripts.experiment_autonomous_temporal_v53 import raw_features,metric

DRAWS=(20261393,20261397)
NET_SEEDS=(11,23)
OUTCOMES=("complete_nuisance","physical_extra","artifact_predictive")
CONDITIONS=("stable","single_sensor_flip","common_sensor_flip")
METHODS=("v55_static","v56_native_seed11","v56_native_seed23","raw_logistic")


def corrupt(x,m,scenario):
    a=np.asarray(x,dtype=float).copy()
    mm=np.asarray(m,dtype=bool).copy()
    if scenario=="stable":
        pass
    elif scenario=="single_sensor_flip":
        a[:,:,5]*=-1
    elif scenario=="common_sensor_flip":
        a[:,:,4:7]*=-1
    else:
        raise ValueError("Unregistered experimental corruption")
    return a,mm


def raw_classifier(x,m,y):
    model=make_pipeline(StandardScaler(),
                        LogisticRegression(C=1.,max_iter=1200))
    model.fit(raw_features(x,m),y)
    return model


def group_world(seed,*,smoke=False):
    ntrain,nvalid,ntest=(360,160,160) if smoke else (1400,600,700)
    data=extended_nonlinear_world(seed=seed,train_n=ntrain,
                                  valid_n=nvalid,test_n=ntest)
    x=data["x"]
    tr,te=data["train"],data["test"]
    mtr=masked_histories(x[tr],"MCAR20",seed=seed+101)
    mte=masked_histories(x[te],"MCAR20",seed=seed+102)
    bank=TrainOnlySourceBank(seed=seed+131,
              permutations=25 if smoke else 60,
              bootstraps=5 if smoke else 20).fit_unlabeled(x[tr],mtr)
    native={}
    for initial_seed in (NET_SEEDS[:1] if smoke else NET_SEEDS):
        native[initial_seed]=NativeSelfSupervisedConceptFAN(
            bank,neural_seed=initial_seed,epochs=2 if smoke else 8
        ).fit_source_unlabeled(x[tr],mtr)
    frozen_sha={str(seed):model.frozen_network_sha
                for seed,model in native.items()}
    receipt={
        "seed":seed,"N_TRAIN":len(tr),"N_VALID_UNUSED":nvalid,"N_TEST":len(te),
        "route":bank.route,"K":bank.k,
        "unlabeled_source_manifest":bank.public_passport(),
        "self_supervised_neural_fits":{
            str(k):{"source_sha":v.frozen_network_sha,
                    "unlabeled_loss_trace":v.loss_receipt}
            for k,v in native.items()},
        "frozen_concepts_before_any_outcome_head":True,
        "sensor_source_semantic_labels_certified":False,
        "risk_head_supervised_training_episode_count":len(tr),
    }
    rows=[]
    for world in OUTCOMES:
        y=data["outcomes"][world]
        static=AdaptiveConceptFAN(
            seed=seed+131,permutations=25 if smoke else 60,
            bootstraps=5 if smoke else 20).fit(x[tr],mtr,y[tr])
        if static.source_hash!=bank.sha or static.route!=bank.route:
            raise AssertionError("Native vs static source provenance mismatch")
        raw=raw_classifier(x[tr],mtr,y[tr])
        for model in native.values():
            model.fit_risk_head(x[tr],mtr,y[tr])
            if model.frozen_network_sha!=frozen_sha[str(model.neural_seed)]:
                raise AssertionError("Source network modified by supervised head")
        for condition in CONDITIONS:
            tx,tm=corrupt(x[te],mte,condition)
            methods={
                "v55_static":static.predict(tx,tm)["probability"],
                "raw_logistic":raw.predict_proba(raw_features(tx,tm))[:,1],
            }
            for ns,model in native.items():
                output=model.predict(tx,tm)
                if output["max_additive_logit_error"]>1e-12:
                    raise AssertionError("Native trained risk is non-additive")
                methods[f"v56_native_seed{ns}"]=output["probability"]
            for method,p in methods.items():
                rows.append({
                    "generator_seed":seed,"outcome_family":world,
                    "TEST_intervention":condition,"method":method,
                    "N_TEST":len(te),"source_route":bank.route,
                    "source_K":bank.k,
                    "outcome_gradients_used_in_native_concepts":False,
                    **metric(y[te],p)
                })
    return receipt,rows


def other_family(*,smoke=False):
    res=[];manifest=[]
    for k in ((3,) if smoke else (3,5)):
        seed=20261423+k
        data=generate_variable_factor_data(
            n_true_factors=k,seed=seed,split_seed=seed+91,
            n_samples=960,n_channels=8,input_steps=36,horizon=6)
        tr,ev=data.train,data.validation
        bank=TrainOnlySourceBank(
            seed=seed+131,permutations=25 if smoke else 60,
            bootstraps=5 if smoke else 20).fit_unlabeled(tr.x,tr.mask)
        native=NativeSelfSupervisedConceptFAN(
            bank,neural_seed=11,epochs=2 if smoke else 8)
        native.fit_source_unlabeled(tr.x,tr.mask)
        native.fit_risk_head(tr.x,tr.mask,tr.y)
        static=AdaptiveConceptFAN(
            seed=seed+131,permutations=25 if smoke else 60,
            bootstraps=5 if smoke else 20).fit(tr.x,tr.mask,tr.y)
        raw=raw_classifier(tr.x,tr.mask,tr.y)
        if static.source_hash!=bank.sha:
            raise AssertionError("Spectral source changed between risk arms")
        pred={
            "v55_static":static.predict(ev.x,ev.mask)["probability"],
            "v56_native_seed11":native.predict(ev.x,ev.mask)["probability"],
            "raw_logistic":raw.predict_proba(raw_features(ev.x,ev.mask))[:,1]
        }
        manifest.append({
            "oracle_true_K_external_audit_only":k,
            "selected_route":bank.route,"selected_K":bank.k,
            "TRAIN":len(tr.x),"DEVELOPMENT_VALID":len(ev.x),
            "original_sealed_TEST_untouched":True,
            "train_only_source":bank.public_passport(),
            "trained_dynamic_source":native.passport()
        })
        for method,p in pred.items():
            res.append({
                "generator_family":"MIXED_LOADING_VARIABLE_K_DEVELOPMENT",
                "true_K_audit_only":k,"method":method,
                "route":bank.route,"detected_K":bank.k,
                "n_valid":len(ev.x),"causal_physical_identity_proven":False,
                **metric(ev.y,p)
            })
    return manifest,res


def run(out:Path,*,smoke=False):
    if out.exists():
        raise FileExistsError("Do not overwrite research evidence")
    reports=[];rows=[]
    for draw in (DRAWS[:1] if smoke else DRAWS):
        report,case_rows=group_world(draw,smoke=smoke)
        reports.append(report);rows.extend(case_rows)
        event={
            "event":"V56_GROUP_DRAW","seed":draw,
            "route":report["route"],"K":report["K"],
            "primary":[z for z in case_rows
                       if z["outcome_family"]=="physical_extra"
                       and z["TEST_intervention"]=="single_sensor_flip"]
        }
        print(json.dumps(event),flush=True)
    mixed,other_rows=other_family(smoke=smoke)
    final={
        "status":"SMOKE_ONLY" if smoke else "V56_NEW_SYNTHETIC_EXPLORATORY_EXECUTED",
        "frozen_pre_outcome_protocol":"docs/medical/CONCEPTFAN_V56_FROZEN_NATIVE_DYNAMICS_PROTOCOL_20261113.md",
        "outcome_gradients_to_concepts":False,
        "unlabeled_train_only_source_definition":True,
        "native_trainable_fuzzy_temporal_concept_encoder":True,
        "classifier_has_no_direct_raw_input_bypass":True,
        "no_physical_or_clinical_semantics_certified":True,
        "no_external_site_validation":True,
        "group_draws":reports,
        "group_test_results":rows,
        "mixed_development_receipts":mixed,
        "mixed_development_results":other_rows,
        "no_guaranteed_improvement_over_static_or_raw":True
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"STATUS.json").write_text(
        json.dumps(final,indent=2,ensure_ascii=False,allow_nan=False)+"\n",
        encoding="utf8")
    for filename,cases in (("GROUP_RESULTS.csv",rows),("MIXED_RESULTS.csv",other_rows)):
        with (out/filename).open("w",newline="",encoding="utf8") as f:
            writer=csv.DictWriter(f,fieldnames=list(cases[0]))
            writer.writeheader();writer.writerows(cases)
    event={"event":"V56_DONE","status":final["status"],
           "N_group_test_rows":len(rows),"N_mixed_dev_rows":len(other_rows),
           "N_group_draws":len(reports)}
    print(json.dumps(event),flush=True)
    return final


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",required=True,type=Path)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    run(args.out_dir,smoke=args.smoke)


if __name__=="__main__":
    main()
