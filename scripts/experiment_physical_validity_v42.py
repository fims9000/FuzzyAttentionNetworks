"""V42 frozen independent-physical-reference test of adaptive ConceptFAN source.

This script does NOT read/train the outcome Y; all 4 new devices have
disjoint TRAIN1000 + VALID400+400 + TEST600, calibrated source-only by
measured ON/OFF and a NEW privileged noisy physical reference readout
R on 300 TRAIN + 200 VALID1 + 200 VALID2. At TEST there is no R input.
Latent E is read in a **separate post-inference AUDIT ONLY block**.
No native K2 retraining is needed for this source-semantic falsifier.
"""
from __future__ import annotations

import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

from conceptfan_discovery.physical_reference_world_v42 import (
    REGISTERED_DEVICES,CASES,generate_device_v42,
    render_observed_v42,masks_v42,measured_difference_v42,
)
from conceptfan_discovery.threeview_centered_v37 import (
    fit_centered_threeview_v37,
)
from conceptfan_discovery.adaptive_pair_consensus_v41 import (
    fit_adaptive_source_v41,adaptive_pair_feature_v41,ORDER,
)
from conceptfan_discovery.physical_reference_certificate_v42 import (
    fit_physical_reference_v42,predict_physical_reference_v42,
    save_physical_reference_v42,load_physical_reference_v42,
)

STRESSES=(
    "complete","MCAR20","MCAR40","MCAR50",
    "missing_view_A","missing_view_B","missing_view_C",
)
SCENARIOS=(
    "healthy_energy",
    "common_U8_matched_ON_OFF",
    "OFF_only_U4_drift",
    "OFF_source_leak_50pct",
    "OFF_reference_unavailable",
    "K2_complete_no_extra_source",
    "VALID2_known_OFF_drift",
    "TEST_A_ON_sensor_inversion",
)
REGISTERED_PAIRS=(
    ("healthy_energy","OFF_only_U4_drift","MCAR20"),
    ("healthy_energy","OFF_only_U4_drift","MCAR40"),
    ("healthy_energy","common_U8_matched_ON_OFF","MCAR20"),
    ("healthy_energy","common_U8_matched_ON_OFF","MCAR40"),
    ("healthy_energy","OFF_source_leak_50pct","MCAR20"),
    ("healthy_energy","OFF_source_leak_50pct","MCAR40"),
    ("healthy_energy","TEST_A_ON_sensor_inversion","MCAR20"),
    ("healthy_energy","TEST_A_ON_sensor_inversion","MCAR40"),
)
WRONG_FRACTION_THRESHOLD_TRAIN_IQR=.50
MIN_AUDIT_SUBGROUP=30
BOOTSTRAPS=399


def _observed(world,indices,*,fault,stress,seed):
    on,off=render_observed_v42(
        world,indices,fault=fault,seed=20264610+seed)
    mo,mr=masks_v42(
        on,off,stress,device_seed=world["device_seed"],
        mask_seed=20264650+STRESSES.index(stress))
    if fault=="no_reference":
        mr[:,:,4:10]=False
    return measured_difference_v42(on,off,mo,mr)


def _metric_group(est,low,high,used,truth,*,iqr):
    n=int(np.sum(used))
    if n==0:
        return {
            "accepted_count":0,"accepted_fraction":0.,
            "MAE_physical":None,"MAE_normalized_train_IQR":None,
            "wrong_physics_fraction_conditional_on_accepted":None,
            "true_E_in_noisy_R_interval_fraction":None,
            "Spearman_rho_estimated_vs_true_E":None,
            "insufficient_conditional_sample_count":True,
        }
    error=np.abs(est[used]-truth[used])
    cov=np.mean((truth[used]>=low[used])&
                (truth[used]<=high[used]))
    rho=None
    if n>=MIN_AUDIT_SUBGROUP and (
        np.std(est[used])>1e-9 and np.std(truth[used])>1e-9):
        z=float(spearmanr(est[used],truth[used]).statistic)
        if np.isfinite(z):
            rho=z
    return {
        "accepted_count":n,
        # Denominator for subgroup computed by calling code.
        "accepted_fraction":float(n/len(used)),
        "MAE_physical":float(error.mean()),
        "MAE_normalized_train_IQR":float(error.mean()/iqr),
        "wrong_physics_fraction_conditional_on_accepted":float(
            (error>WRONG_FRACTION_THRESHOLD_TRAIN_IQR*iqr).mean()),
        "true_E_in_noisy_R_interval_fraction":float(cov),
        "Spearman_rho_estimated_vs_true_E":rho,
        "insufficient_conditional_sample_count":n<MIN_AUDIT_SUBGROUP,
    }


def _classify(out,truth,*,device,family,case,stress,iqr):
    yes=np.asarray(out["physical_reference_allowed"])
    measured=np.asarray(out["support"])>0
    single=np.asarray(out["single_pair_uncorroborated"]) & yes
    multi=np.asarray(out["two_or_three_pairs_corroborated"]) & yes
    anchored=out["physical_reference_estimated_energy"]
    lo=out["physical_reference_interval_lower"]
    hi=out["physical_reference_interval_upper"]
    row={
        "device_seed":device,"family":family,
        "case":case,"stress":stress,
        "n_total_TEST_episodes":len(yes),
        "V41_statistical_measurement_support_fraction":float(measured.mean()),
        "V42_historical_reference_anchored_fraction":float(yes.mean()),
        "V42_all_anchored":_metric_group(anchored,lo,hi,yes,truth,iqr=iqr),
        "V42_single_uncorroborated":_metric_group(
            anchored,lo,hi,single,truth,iqr=iqr),
        "V42_two_or_three_correlated_views":_metric_group(
            anchored,lo,hi,multi,truth,iqr=iqr),
        "V42_anchored_by_pair":{
            k:_metric_group(
                anchored,lo,hi,
                yes&(out["selected_pair"]==k),truth,iqr=iqr)
            for k in ORDER
        },
        "selected_pair_counts":{
            k:int(np.sum(yes&(out["selected_pair"]==k)))
            for k in ORDER
        },
        "single_accepted_fraction_of_all_TEST":float(single.mean()),
        "multi_accepted_fraction_of_all_TEST":float(multi.mean()),
        "reject_no_measured_pair_count":int(
            np.sum(~measured)),
        "reject_reference_history_count":int(
            np.sum(measured & ~yes)),
        "no_lost_reference_sensor_input_imputed":True,
        "physical_truth_only_used_IN_THIS_POSTTEST_AUDIT":True,
        "confidence_interval_targets_noisy_reference_R_not_truth_E":True,
        "no_realworld_physical_semantics_certified":True,
    }
    return row


def _paired_test_bootstrap(a,b,truth,iqr,*,seed):
    yes=np.asarray(a["physical_reference_allowed"]) & np.asarray(
        b["physical_reference_allowed"])
    n=int(yes.sum())
    if n<MIN_AUDIT_SUBGROUP:
        return {
            "intersection_accepted_count":n,
            "delta_MAE_physical_fault_minus_healthy":None,
            "delta_normalized_MAE_fault_minus_healthy":None,
            "paired_TEST_subject_bootstrap_95pct_ci":None,
            "insufficient_subjects_for_comparison":True,
        }
    first=np.abs(a["physical_reference_estimated_energy"][yes]-truth[yes])
    second=np.abs(b["physical_reference_estimated_energy"][yes]-truth[yes])
    delta=(second-first)/iqr
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,n,size=(BOOTSTRAPS,n))
    ci=np.quantile(delta[idx].mean(axis=1),[.025,.975])
    return {
        "intersection_accepted_count":n,
        "delta_MAE_physical_fault_minus_healthy":float(
            (second-first).mean()),
        "delta_normalized_MAE_fault_minus_healthy":float(delta.mean()),
        "paired_TEST_subject_bootstrap_95pct_ci":[float(v) for v in ci],
        "insufficient_subjects_for_comparison":False,
        "bootstraps":BOOTSTRAPS,
        "paired_same_physical_source_true_E":True,
        "shared_fixed_TEST_subject_indices":True,
        "conditional_on_same_accepted_subjects":True,
    }


def run(out:Path,*,smoke:bool=False):
    if out.exists():
        raise FileExistsError("Never overwrite frozen physical source audit")
    ntr,nv,nt=(360,300,240) if smoke else (1000,800,600)
    devices=(20261607,) if smoke else tuple(REGISTERED_DEVICES)
    reftr,refval=(160,100) if smoke else (300,200)
    score=[];comparisons=[];certificates=[];patients={}
    json_model_bundle={}
    for device in devices:
        w=generate_device_v42(
            device_seed=device,n_train=ntr,n_valid=nv,n_test=nt)
        tr,v1,v2,te=(w[k] for k in (
            "train","valid1","valid2","test"))
        if np.intersect1d(tr,te).size or np.intersect1d(v1,te).size:
            raise AssertionError("TEST samples reused in source calibration")
        # This source fit is from exactly observed TRAIN ON/OFF ONLY.
        traind,trainm=_observed(
            w,tr,fault="none",stress="MCAR20",seed=device%101)
        blind=fit_centered_threeview_v37(
            traind,trainm,independent_view_noise_documented=True,
            seed=20264691+device%101,bootstrap=24)
        adaptive=fit_adaptive_source_v41(blind,traind,trainm)
        # *** PRIVILEGED R SENSOR ONLY IN 300+200+200 selected episodes.
        # Never pass this external sensor to adaptive source fitting.
        R=w["independent_reference_readout_CALIBRATION_ONLY"]
        tr_ref=tr[:reftr]; v1_ref=v1[:refval];v2_ref=v2[:refval]
        train_ref_d,train_ref_m=_observed(
            w,tr_ref,fault="none",stress="MCAR20",seed=device%101)
        v1d,v1m=_observed(
            w,v1_ref,fault="none",stress="MCAR20",seed=device%101)
        val2_candidates={}
        for mode in ("none","off_drift"):
            val2_candidates[mode]=_observed(
                w,v2_ref,fault=mode,stress="MCAR20",seed=device%101)
        ready={}
        for val_mode,(v2d,v2m) in val2_candidates.items():
            ready[val_mode]=fit_physical_reference_v42(
                adaptive,
                train_ref_d,train_ref_m,R[tr_ref],
                v1d,v1m,R[v1_ref],
                v2d,v2m,R[v2_ref],
                smoke=smoke,
            )
            certificates.append({
                "device_seed":device,"family":w["family"],
                "valid2_reference_fault":val_mode,
                "reference_budget_TRAIN":reftr,
                "reference_budget_VALID1":refval,
                "reference_budget_VALID2":refval,
                "certificate":ready[val_mode].audit(),
                "no_TEST_reference_available_to_source_prediction":True,
                "reference_is_extra_simulated_privileged_supervision":True,
            })
        # TEST truth is copied exclusively into the AUDIT code AFTER
        # predictions, NEVER passed to estimate or train physical source.
        e_true=w["true_slow_source_energy_POSTFIT_AUDIT_ONLY"][te].copy()
        observed_outputs={}
        for case,law,v2fault,testfault in CASES:
            for stress in STRESSES:
                d,mask=_observed(
                    w,te,fault=testfault,
                    stress=stress,seed=device%101)
                if testfault=="no_reference":
                    # Real absence: drop OFF from the *measured paired
                    # difference* itself, not a magic deletion of S.
                    # _observed already zeros all unavailable OFF.
                    assert not mask[:,:,4:10].any()
                cert=ready[v2fault]
                result=predict_physical_reference_v42(
                    cert,d,mask)
                # Freeze the actual in-inference output arrays BEFORE
                # invoking any ground-truth audit statistic below.
                key=(device,case,stress)
                observed_outputs[(case,stress)]=result
                for field in (
                    "q","support","selected_pair",
                    "physical_reference_estimated_energy",
                    "physical_reference_interval_lower",
                    "physical_reference_interval_upper",
                    "physical_reference_allowed",
                    "single_pair_uncorroborated",
                    "two_or_three_pairs_corroborated",
                    "physical_reference_reason",
                ):
                    patients[f"device{device}_{case}_{stress}_{field}"]=(
                        np.asarray(result[field]).copy())
                score.append(_classify(
                    result,e_true,device=device,family=w["family"],
                    case=case,stress=stress,
                    iqr=cert.train_reference_iqr))
        for k,(case_a,case_b,stress) in enumerate(REGISTERED_PAIRS):
            # For known VALID2 drift, specifically validating the
            # damaged reference changes the PRE-TEST certificate.
            comparisons.append({
                "device_seed":device,"family":w["family"],
                "healthy_case":case_a,"fault_case":case_b,
                "stress":stress,
                **_paired_test_bootstrap(
                    observed_outputs[(case_a,stress)],
                    observed_outputs[(case_b,stress)],
                    e_true,ready["none"].train_reference_iqr,
                    seed=20264790+device%101*100+k),
            })
        patients[f"device{device}_TEST_TRUE_SLOW_SOURCE_ENERGY_AUDIT_ONLY"]=e_true
        # Model state snapshots are generated only when all four
        # source calibration cohorts have passed independent processing.
        for val_mode,obj in ready.items():
            # Save to temporary in-memory mapping; actual JSON files
            # written only AFTER the study succeeds.
            json_model_bundle[(device,val_mode)]=obj
        print(json.dumps({
            "event":"V42_DEVICE_PHYSICAL_REFERENCE_VALIDITY_EXECUTED",
            "device_seed":device,"family":w["family"],
            "approved_pairs_normal_VALID2":{
                k:bool(v.two_VALIDs_approved)
                for k,v in ready["none"].pair_certs.items()},
            "approved_pairs_corrupted_VALID2":{
                k:bool(v.two_VALIDs_approved)
                for k,v in ready["off_drift"].pair_certs.items()},
            "healthy_MCAR20":next(z for z in score if (
                z["device_seed"]==device
                and z["case"]=="healthy_energy"
                and z["stress"]=="MCAR20")),
            "healthy_MCAR40":next(z for z in score if (
                z["device_seed"]==device
                and z["case"]=="healthy_energy"
                and z["stress"]=="MCAR40")),
            "OFF_drift_MCAR20":next(z for z in score if (
                z["device_seed"]==device
                and z["case"]=="OFF_only_U4_drift"
                and z["stress"]=="MCAR20")),
        }),flush=True)
    out.mkdir(parents=True,exist_ok=False)
    model_dir=out/"REFERENCE_MODELS"
    model_dir.mkdir()
    for (device,valid_mode),model in json_model_bundle.items():
        path=model_dir/f"device{device}_{valid_mode}_physical_reference_v42.json"
        save_physical_reference_v42(path,model)
        reopened=load_physical_reference_v42(path)
        # Verify no hidden physical truth or target-outcome in the
        # persisted runtime model.
        assert reopened.audit()==model.audit()
        s=path.read_text()
        assert ("TEST_TRUE" not in s and
                '"hidden_source"' not in s and
                '"outcomes"' not in s)
    status={
        "status":(
            "V42_SMOKE_ONLY" if smoke
            else "V42_PHYSICAL_REFERENCE_SEMANTIC_AUDIT_EXECUTED"),
        "devices":list(devices),
        "families":{str(k):REGISTERED_DEVICES[k] for k in devices},
        "n_train":ntr,"n_valid1":nv//2,"n_valid2":nv//2,
        "n_test":nt,
        "reference_budget_train_per_device":reftr,
        "reference_budget_valid1_per_device":refval,
        "reference_budget_valid2_per_device":refval,
        "n_fault_worlds":len(CASES),
        "n_observation_conditions":len(STRESSES),
        "n_saved_condition_truth_audits":len(score),
        "n_PRE_REGISTERED_paired_TEST_physical_error_comparisons":
            len(comparisons),
        "n_frozen_reference_inference_bundles":len(json_model_bundle),
        "reference_extra_supervision_NOT_autonomous_concept_discovery":True,
        "oracle_true_E_read_only_in_POST_TEST_AUDIT":True,
        "source_fit_does_NOT_receive_oracle_H_S_R_or_Y":True,
        "trained_with_R_only_300_TRAIN_200_VALID1_200_VALID2":
            not smoke,
        "current_episode_independent_reference_not_available":True,
        "interval_covers_noisy_R_under_assumptions_not_true_E":True,
        "not_clinically_validated":True,
        "source_measurement_audit":score,
        "pre_registered_physical_error_comparisons":comparisons,
        "reference_calibration_receipts":certificates,
    }
    (out/"STATUS.json").write_text(json.dumps(
        status,indent=2,ensure_ascii=False,allow_nan=False))
    (out/"PHYSICAL_CORRECTNESS_BY_CONDITION.json").write_text(
        json.dumps(score,indent=2,ensure_ascii=False,allow_nan=False))
    (out/"PAIRED_PHYSICAL_ERROR_COMPARISONS.json").write_text(
        json.dumps(comparisons,indent=2,ensure_ascii=False,allow_nan=False))
    np.savez_compressed(out/"FROZEN_TEST_MEASURED_SOURCE_AND_TRUTH_AUDIT.npz",
                        **patients)
    print(json.dumps({
        "event":"V42_COMPLETE",
        "status":status["status"],
        "n_source_truth_audits":len(score),
        "n_paired_comparisons":len(comparisons),
        "n_reference_bundles":len(json_model_bundle),
        "reference_extra_supervision":True,
        "no_TEST_R_model_inputs":True,
    }),flush=True)
    return status


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args()
    run(args.out_dir,smoke=args.smoke)
