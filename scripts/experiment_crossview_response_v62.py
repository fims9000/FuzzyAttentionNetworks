#!/usr/bin/env python3
"""V62 exploratory reanalysis of the ALREADY VIEWED V61 physical gas TEST.

This compares a new fully unlabeled TRAIN-only cross-view response source bank
with the original positive-lag rank bank. V61 TEST was inspected before this
design: these rows are NOT new independently held-out confirmation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conceptfan_discovery.crossview_response_bank_v62 import TrainOnlyCrossViewBank
from conceptfan_discovery.native_temporal_fuzzy_v56 import NativeSelfSupervisedConceptFAN
from conceptfan_discovery.robust_observable_concepts_v54 import SpectralObservableConceptFAN
from scripts.experiment_autonomous_temporal_v53 import metric,raw_features
from scripts.experiment_real_gas_interventions_v61 import (
    load_official_runs,stratify_whole_runs,train_sensor_normalizer,
    full_run_spectral_bank,raw_exposure_features,semantic_audit,
    source_sha256
)

TASKS=("ethylene_present","co_present","methane_present")
METHODS=("original_v61_static_lag_rank","original_v61_native_lag_rank",
         "v62_static_crossview","v62_native_crossview",
         "raw_sensor_logistic","raw_exposure_logistic")


def _concept_head_predictions(bank,native,train_x,train_mask,y,
                              eval_x,eval_mask):
    static=SpectralObservableConceptFAN(bank.source).fit_risk_head(
        train_x,train_mask,y)
    native.fit_risk_head(train_x,train_mask,y)
    pred_static=static.predict(eval_x,eval_mask)
    pred_native=native.predict(eval_x,eval_mask)
    if pred_static["max_additive_error"]>1e-12 or pred_native["max_additive_logit_error"]>1e-12:
        raise AssertionError("Not an exact signed concept-only logit")
    return pred_static["probability"],pred_native["probability"]


def _baseline(x,m,y,xx,mm,*,exposure):
    train=(raw_exposure_features(x,m) if exposure else raw_features(x,m))
    evalu=(raw_exposure_features(xx,mm) if exposure else raw_features(xx,mm))
    model=make_pipeline(StandardScaler(),
                        LogisticRegression(C=1.,max_iter=1200))
    model.fit(train,y)
    return model.predict_proba(evalu)[:,1]


def run(archive:Path,out:Path):
    if out.exists():
        raise FileExistsError("Do not overwrite exploratory evidence")
    physical_x,records,source_receipt=load_official_runs(archive)
    splits=stratify_whole_runs(records)
    x,m,normalization=train_sensor_normalizer(physical_x,splits["TRAIN"])
    train=splits["TRAIN"]
    prior_bank=full_run_spectral_bank(x[train],m[train])
    candidate_bank=TrainOnlyCrossViewBank(
        seed=20261801,permutations=100,bootstraps=20
    ).fit_unlabeled(x[train],m[train])
    augmentation=x[train].reshape(-1,20,x.shape[-1])
    augmentation_mask=m[train].reshape(-1,20,x.shape[-1])
    assert len(augmentation)==360 and len(train)==120
    prior=NativeSelfSupervisedConceptFAN(
        prior_bank,neural_seed=11,epochs=8).fit_source_unlabeled(
            augmentation,augmentation_mask)
    candidate=NativeSelfSupervisedConceptFAN(
        candidate_bank,neural_seed=11,epochs=8).fit_source_unlabeled(
            augmentation,augmentation_mask)
    prior_sha=prior.frozen_network_sha
    candidate_sha=candidate.frozen_network_sha
    native_concepts={
        "original_v61":prior._transform(x,m)["trajectories"],
        "v62_crossview":candidate._transform(x,m)["trajectories"]
    }
    semantic={
        route:{
            cohort:semantic_audit(
                z,physical_x,records,splits[cohort],
                seed=20261871+(cohort=="TEST"))
            for cohort in ("VALID","TEST")
        } for route,z in native_concepts.items()
    }
    rows=[];task_prevalence={}
    for name in TASKS:
        y=np.asarray([r["targets"][name] for r in records],dtype=int)
        if set(np.unique(y[train]))!={0,1}:
            raise ValueError("Missing physical gas TRAIN task class")
        task_prevalence[name]={
            cohort:float(y[indices].mean()) for cohort,indices in splits.items()}
        for cohort in ("VALID","TEST"):
            idx=splits[cohort]
            prior_static,prior_native=_concept_head_predictions(
                prior_bank,prior,x[train],m[train],y[train],x[idx],m[idx])
            new_static,new_native=_concept_head_predictions(
                candidate_bank,candidate,x[train],m[train],y[train],x[idx],m[idx])
            if prior_sha!=prior.frozen_network_sha or candidate_sha!=candidate.frozen_network_sha:
                raise AssertionError("Gas task labels modified frozen unlabeled concept definitions")
            preds={
                "original_v61_static_lag_rank":prior_static,
                "original_v61_native_lag_rank":prior_native,
                "v62_static_crossview":new_static,
                "v62_native_crossview":new_native,
                "raw_sensor_logistic":_baseline(
                    x[train],m[train],y[train],x[idx],m[idx],exposure=False),
                "raw_exposure_logistic":_baseline(
                    x[train],m[train],y[train],x[idx],m[idx],exposure=True),
            }
            for method,p in preds.items():
                rows.append({
                    "target":name,"cohort":cohort,
                    "method":method,
                    "n_independent_whole_runs":len(idx),
                    "positive_fraction":float(y[idx].mean()),
                    "source_K":candidate_bank.k if method.startswith("v62")
                                else prior_bank.k if method.startswith("original_v61")
                                else None,
                    "evaluation_reuses_previously_inspected_v61_test":True,
                    **metric(y[idx],p)
                })
    evidence={
        "status":"V62_EXPLORATORY_CROSSVIEW_REAL_GAS_EXECUTED",
        "protocol":"docs/medical/CONCEPTFAN_V62_FROZEN_CROSSVIEW_CONTRAST_PROTOCOL_20261113.md",
        "dataset":"Original UCI gas sensor array exposed to turbulent gas mixtures ID309",
        "original_data_sha256":source_sha256(archive),
        "real_whole_run_counts":{name:len(ix) for name,ix in splits.items()},
        "same_v61_test_labels_already_viewed_before_v62_design":True,
        "new_model_benchmark_is_exploratory_not_independent_confirmation":True,
        "train_only_source_fitting_uses_zero_gas_labels":True,
        "train_only_sensor_normalization":normalization,
        "original_positive_lag_source":prior_bank.public_passport(),
        "crossview_source":candidate_bank.public_passport(),
        "original_native_hash":prior_sha,
        "crossview_native_hash":candidate_sha,
        "unlabeled_native_train_losses":{
            "original":prior.loss_receipt,"crossview":candidate.loss_receipt},
        "physical_gas_prevalence":task_prevalence,
        "fixed_valid_and_reused_test_metrics":rows,
        "frozen_run_level_semantic_audit":semantic,
        "physical_gas_axis_identity_proven":False,
        "physical_fault_instrument_provenance_verified":False,
        "dependent_training_crops_not_independent_cases":True,
        "n_independent_original_train_runs":len(train),
        "n_dependent_train_crops":len(augmentation),
        "raw_person_level_data_or_timestamps_exported":False,
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"STATUS.json").write_text(json.dumps(
        evidence,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
    with (out/"METRICS.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({
        "event":"V62_EXPLORATORY_DONE",
        "old_lag_rank_K":prior_bank.k,
        "new_crossview_K":candidate_bank.k,
        "new_crossview_bootstrap_interval":candidate_bank.train_receipt[
            "source_rank_evidence"]["train_whole_run_rank_interval_05_95"],
        "test_metrics":[row for row in rows if row["cohort"]=="TEST"],
        "test_source_associations":{
            key:val["TEST"]["max_abs_by_gas"] for key,val in semantic.items()
        },
    },ensure_ascii=False,allow_nan=False),flush=True)
    return evidence


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--uci-zip",required=True,type=Path)
    parser.add_argument("--out-dir",required=True,type=Path)
    args=parser.parse_args()
    run(args.uci_zip,args.out_dir)


if __name__=="__main__":
    main()
