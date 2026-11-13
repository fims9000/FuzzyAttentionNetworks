#!/usr/bin/env python3
"""V62 brand-new UCI HAR activity dataset: conditional gyro beyond acceleration.

No UCI HAR TEST subject values/labels seen during source compilation,
training, or patient VALID+49 shuffled-null gate. UCI source is CC-BY 4.0.
Existing V61R train-only source residualizer, true random model initials,
fixed outcome-blind <=6 sources and native signed concept head are reused.
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
from sklearn.preprocessing import StandardScaler

from conceptfan_discovery.har_activity_v62 import (
    TASKS,STRESSES,SENSOR_NAMES,
    load_har_v62,rename_bank_as_har_v62,stress_har_v62,
    collapse_raw_axes_v62,
)
from conceptfan_discovery.program_compiler_v61 import (
    compile_allowed_programs_v61,select_conditioning_and_novel_programs_v61,
)
from conceptfan_discovery.program_compiler_v60 import infer_program_matrix_v60
from conceptfan_discovery.program_fan_v60 import (
    infer_program_head_v60,apply_program_admission_v60,
)
from scripts.experiment_conceptfan_conditional_v61 import (
    _fit_pulse_and_conditional,
    _fit_logistic_programs,_predict_logistic_programs,
    _as_metrics,_group_weight,
)
from scripts.experiment_conceptfan_bidmc_real_v58 import (
    _bootstrap_by_real_patient,
)

SEEDS=(11,23)
PROTOCOL="CONCEPTFAN_V62_UNTOUCHED_UCI_HAR_ACCEL_VS_GYRO_ACTIVITIES"


def _axis_features(axes,select):
    raw=np.asarray(axes,float)
    if raw.ndim!=3 or raw.shape[1:]!=(128,6):
        raise ValueError("HAR raw 128x6 inertial axes required")
    idx=list(select)
    d=raw[:,:,idx]
    # Same physical samples for both logistic baselines, no UCI 561-feature
    # externally engineered vector or TEST-fitted normalization.
    blocks=d.reshape(len(raw),16,8,len(idx))
    mean=blocks.mean(axis=2).reshape(len(raw),-1)
    std=blocks.std(axis=2).reshape(len(raw),-1)
    return np.hstack((mean,std,d.mean(axis=1),d.std(axis=1)))


def _fit_axis_logistic(data,select,*,seed):
    x=_axis_features(data["axes"],select)
    weights=_group_weight(data["patient"])
    standard=StandardScaler()
    xx=standard.fit_transform(x,sample_weight=weights)
    clf=LogisticRegression(
        max_iter=1800,solver="lbfgs",C=1.,random_state=seed)
    clf.fit(xx,data["y"],sample_weight=weights)
    return standard,clf,tuple(select)


def _infer_axis_logistic(model,axes):
    standard,clf,select=model
    x=_axis_features(axes,select)
    return np.asarray(clf.decision_function(standard.transform(x)),float)


def _train_anchor_mean_logistic(train,*,seed):
    x=train["anchor"]
    model=LogisticRegression(max_iter=1200,C=1.,random_state=seed)
    model.fit(x[:,None],train["y"],
              sample_weight=_group_weight(train["patient"]))
    return model


def _infer_anchor_mean_logistic(model,raw,available):
    val=np.asarray(raw[:,-4:,0],float).mean(axis=1)
    good=available[:,:,0].mean(axis=1)>=.5
    # A missing physical accelerometer must never be presented as a real 0g
    # sensor reading to a calibrated scalar accelerometer model. A learned
    # class-prior logit is used rather than invented medical/motion evidence.
    val=np.where(good,val,float(model._anchor_median)) if hasattr(
        model,"_anchor_median") else val
    return np.asarray(model.decision_function(val[:,None]),float)


def _metric(y,z,subject):
    from sklearn.metrics import accuracy_score
    q=_as_metrics(y,z,subject)
    q["accuracy_fixed_half"]=float(
        accuracy_score(y,expit(z)>=.5))
    return q


def execute(root:Path,out:Path,*,tasks=tuple(TASKS),seeds=SEEDS):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output=Path(out);output.mkdir(parents=True,exist_ok=True)
    cases=[];private=[];fitted=[];receipts=[]
    for task in tasks:
        sections,norm,manifest=load_har_v62(root,task)
        (output/f"uci_har_{task}_split_receipt.json").write_text(
            json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
        train,v1,v2,test=(sections[k] for k in (
            "train","valid1","valid2","test"))
        if any(np.unique(sections[k]["y"]).size<2
               for k in ("train","test")):
            raise RuntimeError(
                "Frozen HAR participant split is one class, do not resplit after TEST")
        bank=rename_bank_as_har_v62(
            compile_allowed_programs_v61(
                train["x"],train["mask"],train["patient"],
                allowed_channels=(0,1,2),seed=20266213))
        pi,ni=select_conditioning_and_novel_programs_v61(
            bank,(0,1,2))
        if not pi:
            raise RuntimeError("No body-accelerometer reference program")
        receipts.append({
            "task":task,"patient_groups":manifest["splits"],
            "original_physically_distinct_channels":SENSOR_NAMES,
            "all_compiler_candidate_passports":bank.manifest(),
            "accelerometer_only_indices":pi,
            "gyroscope_only_indices":ni,
            "precomputed_561_features_not_used":True,
            "source_compilation_used_activity_y":False,
        })
        zv,sv=infer_program_matrix_v60(
            bank,train["x"],train["mask"])
        for seed in seeds:
            pulse_head,mapping,gyro_head,ablation,gate=_fit_pulse_and_conditional(
                train,v1,v2,bank,pi,ni,seed=seed)
            lpulse=_fit_logistic_programs(
                zv,sv,train["y"],train["patient"],
                ids=pi,seed=20266220+seed)
            lall=_fit_logistic_programs(
                zv,sv,train["y"],train["patient"],
                ids=tuple(range(len(bank.candidates))),seed=20266230+seed)
            lacc=_fit_axis_logistic(
                train,(0,1,2),seed=20266240+seed)
            lgyro=_fit_axis_logistic(
                train,(3,4,5),seed=20266250+seed)
            lfull=_fit_axis_logistic(
                train,tuple(range(6)),seed=20266260+seed)
            lmean=_train_anchor_mean_logistic(train,seed=20266270+seed)
            lmean._anchor_median=float(np.median(train["anchor"]))
            fitted.append({
                "task":task,"seed":seed,
                "anchor_native_head":pulse_head.passport(),
                "gyro_source_contract":
                    None if mapping is None else mapping.passport(),
                "gyro_native_head":
                    None if gyro_head is None else gyro_head.passport(),
                "gyro_ablation_unconditioned":
                    None if ablation is None else ablation.passport(),
                "source_program_ids":[] if mapping is None else [
                    bank.candidates[j].candidate_id
                    for j in mapping.raw_source_indices],
                "dual_VALID_patient_and_accel_stratified_49_null_gate":gate,
                "training_labels_only_official_UCI_train_people":True,
                "physical_gyro_sensor_type_separate_but_causality_unproven":True,
            })
            for stress in STRESSES:
                x,m,raw,device_mask=stress_har_v62(
                    test,norm,stress,seed=20266290+seed)
                z,s=infer_program_matrix_v60(bank,x,m)
                pure=infer_program_head_v60(
                    pulse_head,z[:,pi],s[:,pi],np.zeros(len(z)))
                base=pure["logits"]
                if mapping is None:
                    q=np.zeros((len(z),0))
                    src=np.zeros_like(q)
                    support=np.zeros_like(q)
                    active=np.zeros((len(z),0),bool)
                    used=np.zeros(len(z),bool)
                    prop=base.copy()
                    deployed=base.copy()
                    ungated=base.copy()
                else:
                    from conceptfan_discovery.conditional_innovation_v61 import (
                        infer_conditional_sources_v61,
                    )
                    vz,vs=infer_conditional_sources_v61(mapping,z,s,m)
                    novelty=infer_program_head_v60(
                        gyro_head,vz,vs,base)
                    q=novelty["memberships"]
                    src=novelty["signed_contributions"]
                    support=novelty["support"]
                    active=novelty["used"][:,list(gyro_head.active_indices)]
                    prop=novelty["logits"]
                    deployed,used=apply_program_admission_v60(
                        base,prop,active,approved=gate["approved"])
                    raw_indices=mapping.raw_source_indices
                    ungated=infer_program_head_v60(
                        ablation,z[:,raw_indices],s[:,raw_indices],base
                    )["logits"]
                    if np.max(np.abs(prop-base-src.sum(axis=1)))>1e-9:
                        raise AssertionError("V62 gyro signed concept-only risk bypass")
                    if not gate["approved"]:
                        np.testing.assert_array_equal(deployed,base)
                raw_three=collapse_raw_axes_v62(raw)
                logits={
                    "AccelProgramFAN":base,
                    "ConditionalGyroFAN_proposed":prop,
                    "ConditionalGyroFAN_deployed":deployed,
                    "UnconditionalGyro_ablation":ungated,
                    "AccelProgramLogistic":_predict_logistic_programs(
                        lpulse,z,s,pi),
                    "AllProgramLogistic":_predict_logistic_programs(
                        lall,z,s,tuple(range(len(bank.candidates)))),
                    "AccelThreeAxisLogistic":_infer_axis_logistic(lacc,raw),
                    "GyroThreeAxisLogistic":_infer_axis_logistic(lgyro,raw),
                    "AllSixAxisLogistic":_infer_axis_logistic(lfull,raw),
                    "AccelNormMeanLogistic":_infer_anchor_mean_logistic(
                        lmean,raw_three,device_mask),
                }
                measured={
                    name:_metric(test["y"],value,test["patient"])
                    for name,value in logits.items()}
                key=f"UCIHAR_{task}_seed{seed}_{stress}"
                private_dir=output/"private_case_arrays"
                private_dir.mkdir(parents=True,exist_ok=True)
                path=private_dir/f"{key}.npz"
                groups=np.unique(test["patient"],return_inverse=True)[1]
                np.savez_compressed(path,
                    y=test["y"].astype(np.uint8),
                    subject_group=groups.astype(np.int16),
                    accel=base,proposed=prop,deployed=deployed,
                    unconditioned=ungated,
                    accel_program_logistic=logits["AccelProgramLogistic"],
                    all_program_logistic=logits["AllProgramLogistic"],
                    accel_axis_logistic=logits["AccelThreeAxisLogistic"],
                    gyro_axis_logistic=logits["GyroThreeAxisLogistic"],
                    all_axis_logistic=logits["AllSixAxisLogistic"],
                    accel_mean_logistic=logits["AccelNormMeanLogistic"],
                    gyro_memberships=q,gyro_support=support,
                    gyro_signed=src,active=active.astype(np.uint8),
                    admitted_used=used.astype(np.uint8),
                    anchor_memberships=pure["memberships"],
                    anchor_signed=pure["signed_contributions"],
                    anchor_support=pure["support"])
                private.append({
                    "case":key,
                    "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                    "private_per_participant_windows_not_uploaded":True,
                })
                cases.append({
                    "task":task,"seed":seed,"stress":stress,
                    "official_TEST_subjects":len(np.unique(test["patient"])),
                    "overlapping_windows":len(test["y"]),
                    "test_positive_fraction":float(test["y"].mean()),
                    "observed_gyro_source_fraction":
                        float(active.any(axis=1).mean()) if active.size else 0.,
                    "source_admitted_after_TRUE_subject_VALID_and_null":
                        bool(gate["approved"]),
                    "deployed_gyro_usage_fraction":float(used.mean()),
                    "metrics":measured,
                    "subject_bootstrap_deployed_vs_ALL_SIX_AXIS":
                        _bootstrap_by_real_patient(
                            test["y"],deployed,logits["AllSixAxisLogistic"],
                            test["patient"],seed=20266240+seed),
                    "subject_bootstrap_proposed_vs_ACCEL_THREE_AXIS":
                        _bootstrap_by_real_patient(
                            test["y"],prop,logits["AccelThreeAxisLogistic"],
                            test["patient"],seed=20266260+seed),
                    "original_har_TEST_used_for_fitting_or_source_admission":False,
                    "gyro_is_distinct_sensor_not_causal_concept_certificate":True,
                })
                print("V62",task,seed,stress,
                      "true_test_people",len(np.unique(test["patient"])),
                      "novel_sources",len(mapping.raw_source_indices) if mapping else 0,
                      "admitted",gate["approved"],
                      "ACC_AP",round(measured["AccelProgramFAN"]["auprc"],3),
                      "FULL_AXES_AP",round(measured["AllSixAxisLogistic"]["auprc"],3),
                      flush=True)
    report={
        "protocol":PROTOCOL,
        "status":"V62_EXECUTED_UNTOUCHED_UCI_HAR_SENSOR_NOVELTY",
        "original_dataset":"UCI HAR Using Smartphones, DOI 10.24432/C54S4K CC-BY 4.0",
        "task_activities_predeclared_before_TEST":list(tasks),
        "original_official_TEST_nine_independent_persons":True,
        "TRAIN_13_VALID1_4_VALID2_4_TEST_9_REAL_PARTICIPANTS":True,
        "base_body_acceleration_gyro_novel_sensor_independent_device_type":True,
        "all_sources_contemporaneous_window_no_future_leakage":True,
        "all_561_UCI_engineered_features_excluded":True,
        "conditional_gyro_source_ids_SELECTED_TRAIN_X_only":True,
        "significant_TEST_gain_does_not_override_FAILED_VALID":True,
        "no_physical_causal_source_interpretation":True,
        "two_true_stochastic_native_initializations":True,
        "n_trained_models":len(fitted),"n_test_conditions":len(cases),
        "n_original_official_TEST_windows":2947,
        "person_ids_and_full_windows_not_uploaded":True,
        "source_banks":receipts,"models":fitted,
        "test_cases":cases,"private_case_hashes":private,
        "limitations":"9 heldout participants, same phone model, video/protocol label, overlapping windows, exploratory multiple tasks no multiplicity adjustment",
    }
    (output/"report_v62.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    with (output/"aggregate_cases_v62.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.writer(f)
        names=("AccelProgramFAN","ConditionalGyroFAN_proposed",
               "ConditionalGyroFAN_deployed","AccelProgramLogistic",
               "AllProgramLogistic","AccelThreeAxisLogistic",
               "GyroThreeAxisLogistic","AllSixAxisLogistic")
        writer.writerow(["task","seed","stress","test_participants",
                         "n_windows","positive_fraction","gyro_admitted",
                         *[n+"_AUPRC" for n in names],
                         *[n+"_NLL" for n in names]])
        for case in cases:
            m=case["metrics"]
            writer.writerow([case["task"],case["seed"],case["stress"],
                             case["official_TEST_subjects"],
                             case["overlapping_windows"],
                             case["test_positive_fraction"],
                             case["source_admitted_after_TRUE_subject_VALID_and_null"],
                             *[m[n]["auprc"] for n in names],
                             *[m[n]["nll"] for n in names]])
    print(json.dumps({
        "status":report["status"],"models":len(fitted),
        "test_conditions":len(cases),
        "novel_gyro_approved":sum(
            q["dual_VALID_patient_and_accel_stratified_49_null_gate"]["approved"]
            for q in fitted)},indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--tasks",nargs="+",choices=TASKS,default=list(TASKS))
    p.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS))
    a=p.parse_args()
    execute(a.data,a.out,tasks=tuple(a.tasks),seeds=tuple(a.seeds))
