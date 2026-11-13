#!/usr/bin/env python3
"""V71 deliberately SUPERVISED physical-dose anchored concepts, post V70.

Uses exactly the same original UCI308 58 experiments, 39/11/8 split,
unchanged original physical sensor measurements. TRAIN water-dilution dose
labels form two conditional source proxy axes: no autonomous chemical source
identity claim; gas-phase physical analyzer reference is still unavailable.
TEST8 was already viewed in V70, so this comparison is exploratory.
"""
from __future__ import annotations

import argparse,csv,json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conceptfan_discovery.trialwise_dualstream_fuzzy_v70 import TrialOnlyDualStreamBank
from conceptfan_discovery.reference_anchored_dose_v71 import SupervisedConditionalReferenceBank
from scripts.experiment_real_uci308_two_gas_v70 import (
    load_uci308_genuine_independent_trials,
    signed_concept_ridge,physical_trial_metrics,partial_spearman,
    original_source_sha256,ANALYTES,
)

METHODS=(
    "V71_SUPERVISED_REFERENCE_FUZZY",
    "V71_SUPERVISED_REFERENCE_LINEAR",
    "V71_LABEL_PERMUTATION_SOURCE_CONTROL",
    "V70_AUTONOMOUS_STATISTICAL_FUZZY",
    "ORIGINAL_PHYSICAL_SENSORS_RAW_RIDGE",
    "CONSTANT_TRAIN_MEAN",
)


def descriptive_specificity(bank,raw,reference,groups):
    z=bank.score_proxies(raw)
    table={}
    for cohort in ("TRAIN","VALID","TEST"):
        rows=groups[cohort]
        y=reference[rows]
        scores=z[rows]
        per=[]
        for j,name in enumerate(ANALYTES):
            match=partial_spearman(scores[:,j],y[:,j],y[:,1-j])
            crosstalk=partial_spearman(scores[:,j],y[:,1-j],y[:,j])
            margin=(abs(match)-abs(crosstalk)
                    if match is not None and crosstalk is not None else None)
            per.append({
                "axis_label_supplied_from_TRAIN_reference":name,
                "n_independent_original_trials":len(rows),
                "partial_spearman_matched_prep_dose":match,
                "partial_spearman_other_prep_dose":crosstalk,
                "abs_matched_minus_abs_other_specificity_margin":margin,
                "gas_analyte_molecular_identity_proven":False,
            })
        table[cohort]=per
    return table


def predict_refit(method,TRAIN_features,train_y,eval_features):
    if method.startswith("V71_SUPERVISED_REFERENCE_FUZZY") or method in (
        "V71_LABEL_PERMUTATION_SOURCE_CONTROL",
        "V70_AUTONOMOUS_STATISTICAL_FUZZY",
    ):
        return signed_concept_ridge(TRAIN_features,train_y,eval_features)
    reg=make_pipeline(StandardScaler(),Ridge(alpha=1.))
    reg.fit(TRAIN_features,train_y)
    return reg.predict(eval_features),None


def run(path:Path,out:Path):
    if out.exists():
        raise FileExistsError("Cannot overwrite already-computed V71 source proxy comparison")
    raw,target,batch,groups,schema=load_uci308_genuine_independent_trials(path)
    train=groups["TRAIN"]
    reference=TrialOnlyDualStreamBank().fit_unlabeled(raw[train])
    reference_hash=reference.sha
    raw32=reference.raw_features(raw)
    anchor=SupervisedConditionalReferenceBank().fit_reference_anchored(
        raw32[train],target[train])
    anchored_source_sha=anchor.sha
    shuffled=np.random.default_rng(20261971).permutation(len(train))
    null_anchor=SupervisedConditionalReferenceBank().fit_reference_anchored(
        raw32[train],target[train][shuffled])
    if null_anchor.sha==anchor.sha:
        raise AssertionError("Source direction did not respond to real supervised reference")
    features={
        "V71_SUPERVISED_REFERENCE_FUZZY":anchor.fuzzy(raw32),
        "V71_SUPERVISED_REFERENCE_LINEAR":anchor.score_proxies(raw32),
        "V71_LABEL_PERMUTATION_SOURCE_CONTROL":null_anchor.fuzzy(raw32),
        "V70_AUTONOMOUS_STATISTICAL_FUZZY":reference.fuzzy_features(raw),
        "ORIGINAL_PHYSICAL_SENSORS_RAW_RIDGE":raw32,
    }
    rows=[];heads=[]
    for j,name in enumerate(ANALYTES):
        train_y=target[train,j]
        if np.std(train_y)<=1e-8:
            raise ValueError("Nominal chemical preparation reference unobservable in TRAIN")
        for cohort in ("VALID","TEST"):
            ix=groups[cohort]
            for method in METHODS:
                if method=="CONSTANT_TRAIN_MEAN":
                    p=np.full(len(ix),train_y.mean());head=None
                else:
                    p,head=predict_refit(method,features[method][train],
                                         train_y,features[method][ix])
                rows.append({
                    "analyte_water_dilution_volpercent":name,
                    "cohort":cohort,
                    "method":method,
                    "n_original_independent_trials":len(ix),
                    "TEST_was_already_inspected_in_V70_before_design":True,
                    "training_labels_used_in_new_source_fit":
                        method.startswith("V71_"),
                    "target_is_nominal_water_dilution_not_airborne_gas_ppm":True,
                    **physical_trial_metrics(target[ix,j],p,np.std(train_y))})
                if cohort=="VALID" and head is not None:
                    heads.append({"reference":name,"method":method,
                                  "head_receipt":head})
    if reference.sha!=reference_hash or anchor.sha!=anchored_source_sha:
        raise AssertionError("Supervised heads rewrote frozen source")
    receipt={
        "status":"V71_POST_V70_EXPLORATORY_REFERENCE_ANCHORED_PROXY_EXECUTED",
        "not_a_new_independent_test":True,
        "V70_TEST8_already_inspected_when_hypothesis_formulated":True,
        "original_UCI308_source_zip_sha256":original_source_sha256(path),
        "original_data_trial_contract":schema,
        "original_physical_experiment_split":{
            k:len(v) for k,v in groups.items()},
        "original_V70_UNSUPERVISED_source_sha256":reference_hash,
        "anchored_source_SHA256":anchored_source_sha,
        "anchored_source_uses_actual_TRAIN_reference_preparation_doses":True,
        "anchor_is_supervised_water_dilution_proxy_not_autonomous_gas_factor":True,
        "anchored_source_passport":anchor.public_passport(),
        "permuted_reference_source_SHA256":null_anchor.sha,
        "all_preset_comparators_and_target_metrics":rows,
        "exact_signed_original_fuzzy_head_receipts":heads,
        "TRAIN_VALID_TEST_anchored_source_specificity":descriptive_specificity(
            anchor,raw32,target,groups),
        "physically_identifiable_gas_source_claimed":False,
        "reference_field_does_not_measure_gas_ppm":True,
        "same_39_train_physical_labels_used_to_anchor_and_head":True,
        "an_unlabeled_source_cannot_be_physically_named_from_K_alone":True,
        "protocol":"docs/medical/CONCEPTFAN_V71_FROZEN_SUPERVISED_CONDITIONAL_ANCHORS_20261113.md",
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"STATUS.json").write_text(json.dumps(
        receipt,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
    with (out/"METRICS.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({
        "event":"V71_ANCHORED_PREPARATION_DOSE_SOURCE_DONE",
        "split_original_independent_trials":{
            k:len(v) for k,v in groups.items()},
        "source_sha_supervised":anchored_source_sha,
        "TEST_not_new_confirmation":True,
        "test_scores":[{k:val for k,val in row.items()
                       if k in ("analyte_water_dilution_volpercent","method","R2",
                                "nominal_water_dilution_volpercent_MAE")}
                       for row in rows if row["cohort"]=="TEST"],
        "TEST_source_specificity":receipt["TRAIN_VALID_TEST_anchored_source_specificity"]["TEST"],
    },ensure_ascii=False,allow_nan=False),flush=True)
    return receipt


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--uci-zip",required=True,type=Path)
    p.add_argument("--out-dir",required=True,type=Path)
    a=p.parse_args()
    run(a.uci_zip,a.out_dir)


if __name__=="__main__":
    main()
