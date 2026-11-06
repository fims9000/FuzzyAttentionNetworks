"""V43 full preregistered physical reference / independent live nuisance monitor.

Physical TEST truth E ONLY appears in score() and bootstrap() after all
inference predictions and accept/reject decisions have been frozen.
No TEST R, Y, H, S, or hidden U is supplied to either concept fitting,
monitor calibration or raw physical-R supervised regression.

The two TEST nuisance monitor readings are new physical observations
IN ADDITION to the target-source ON/OFF data; the simulator constructs
them from hidden U with independently generated sensor noise. Their
existence, gain and correctness would require actual hardware evidence.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr

from conceptfan_discovery.live_nuisance_world_v43 import (
    REGISTERED_DEVICES,generate_device_v43,render_observed_v43,
    masks_v43,measured_difference_v43,
    measured_current_nuisance_monitor_v43,
)
from conceptfan_discovery.threeview_centered_v37 import (
    fit_centered_threeview_v37,
)
from conceptfan_discovery.adaptive_pair_consensus_v41 import (
    fit_adaptive_source_v41,
)
from conceptfan_discovery.physical_reference_certificate_v42 import (
    fit_physical_reference_v42,predict_physical_reference_v42,
    save_physical_reference_v42,measured_external_on_off_difference_v42,
)
from conceptfan_discovery.live_nuisance_monitor_v43 import (
    fit_live_nuisance_monitor_v43,
    evaluate_live_nuisance_monitor_v43,
    gate_physical_reference_v43,save_live_monitor_v43,
)
from conceptfan_discovery.physical_reference_raw_baselines_v43 import (
    METHODS,features_measured_reference_v43,
    fit_raw_reference_v43,predict_raw_reference_v43,
)

STRESSES=("complete","MCAR20","MCAR40","missing_view_A")
# Historical reference calibrates on the same fixed healthy physical
# training environment; VALID2 corruption is tested as an explicit fault.
WORLDS=(
    ("healthy","none","honest","none"),
    ("matched_U8","common8","honest","none"),
    ("OFF_drift","off_drift","honest","none"),
    ("OFF_source_leak","leak","honest","none"),
    ("OFF_missing","no_reference","honest","none"),
    ("VALID2_known_drift","off_drift","honest","off_drift"),
    ("monitor_blind_drift","off_drift","blind_drift","none"),
    ("monitor_false_alarm","none","false_off_alarm","none"),
    ("monitor_missing","none","missing","none"),
    ("A_ON_inversion","invert_on_A","honest","none"),
)
METHODS_ALL=("V42_HISTORICAL","V43_LIVE_MONITORED")+tuple(
    f"{m}_{mode}" for m in METHODS for mode in ("UNGATED","GATED"))
METRICS=("coverage","wrong_given_accepted","wrong_and_accepted_per_all",
         "correct_and_accepted_per_all","norm_MAE_accepted")


def _read(world,ids,*,physical_fault,monitor_fault,stress,stage,monitor_seed):
    dev=world["device_seed"]
    seed=20264810+dev%101+stage
    on,off=render_observed_v43(
        world,ids,fault=physical_fault,seed=seed)
    mo,mr=masks_v43(
        on,off,stress,device_seed=dev,
        mask_seed=20264840+STRESSES.index(stress)+stage*31)
    if physical_fault=="no_reference":
        mr[:,:,4:10]=False
    c_on,c_off,cmo,cmr=measured_current_nuisance_monitor_v43(
        world,ids,physical_fault=physical_fault,
        monitor_fault=monitor_fault,same_seed_as_ON_OFF=seed)
    diff,md=measured_external_on_off_difference_v42(
        on,off,mo,mr)
    return {
        "on":on,"off":off,"mask_on":mo,"mask_off":mr,
        "diff":diff,"diff_mask":md,
        "control_on":c_on,"control_off":c_off,
        "control_mask_on":cmo,"control_mask_off":cmr,
    }


def _monitor_inputs(rec):
    return (rec["control_on"],rec["control_off"],
            rec["control_mask_on"],rec["control_mask_off"])


def _raw_model_features(rec):
    return features_measured_reference_v43(
        rec["on"],rec["off"],rec["mask_on"],rec["mask_off"])


def _score(guess,accepted,truth,*,iqr,baseline_interval=None):
    y=np.asarray(truth,dtype=np.float64)
    p=np.asarray(guess,dtype=np.float64)
    a=np.asarray(accepted,dtype=bool)
    if y.shape!=p.shape or y.shape!=a.shape or len(y)<1:
        raise AssertionError("Patient-level observation cohort mismatch")
    if not np.isfinite(y).all() or not np.isfinite(p[a]).all():
        raise AssertionError("Source model emitted non-finite accepted value")
    n=int(a.sum())
    wrong=np.zeros(len(y),dtype=bool)
    wrong[a]=np.abs(p[a]-y[a])>.5*iqr
    correct=a&~wrong
    result={
        "n_total":len(y),"n_accepted":n,
        "coverage":float(a.mean()),
        "wrong_given_accepted":(
            float(wrong[a].mean()) if n else None),
        "wrong_and_accepted_per_all":float(wrong.mean()),
        "correct_and_accepted_per_all":float(correct.mean()),
        "norm_MAE_accepted":(
            float(np.abs(p[a]-y[a]).mean()/iqr) if n else None),
        "n_physically_wrong_accepted":int(wrong.sum()),
        "n_physically_correct_accepted":int(correct.sum()),
        "true_E_in_noisy_R_interval_given_accepted":None,
        "small_accepted_sample":n<30,
    }
    if baseline_interval is not None and n:
        lower,upper=baseline_interval
        result["true_E_in_noisy_R_interval_given_accepted"]=float(
            ((y[a]>=lower[a])&(y[a]<=upper[a])).mean())
    return result


def _paired_differences(epA,epB,truth,*,iqr,seed):
    A=np.asarray(epA["accepted"],dtype=bool)
    B=np.asarray(epB["accepted"],dtype=bool)
    y=np.asarray(truth)
    p=np.asarray(epA["pred"]);q=np.asarray(epB["pred"])
    # Primary all-subject wrong-and-accepted + correct-and-accepted
    # differences. A reduced false-acceptance rate alone is trivial.
    wrongA=A&(np.abs(np.where(A,p,0)-y)>.5*iqr)
    wrongB=B&(np.abs(np.where(B,q,0)-y)>.5*iqr)
    correctA=A&~wrongA
    correctB=B&~wrongB
    dw=wrongB.astype(float)-wrongA.astype(float)
    dc=correctB.astype(float)-correctA.astype(float)
    rng=np.random.default_rng(seed)
    picks=rng.integers(0,len(y),size=(399,len(y)))
    ciW=np.quantile(dw[picks].mean(axis=1),[.025,.975]).tolist()
    ciC=np.quantile(dc[picks].mean(axis=1),[.025,.975]).tolist()
    common=A&B
    if int(common.sum())>=30:
        d=(np.abs(q[common]-y[common])-
           np.abs(p[common]-y[common]))/iqr
        rng=np.random.default_rng(seed+1)
        pp=rng.integers(0,len(d),size=(399,len(d)))
        ciMAE=np.quantile(d[pp].mean(axis=1),[.025,.975]).tolist()
        mae=float(d.mean())
    else:
        ciMAE=None;mae=None
    return {
        "delta_wrong_and_accepted_per_all_b_minus_a":float(dw.mean()),
        "paired_TEST_subject_95pct_wrong_delta_CI":ciW,
        "delta_correct_and_accepted_per_all_b_minus_a":float(dc.mean()),
        "paired_TEST_subject_95pct_correct_delta_CI":ciC,
        "intersection_n":int(common.sum()),
        "delta_normalized_MAE_on_same_accepted_b_minus_a":mae,
        "paired_95pct_normalized_MAE_on_same_accepted":ciMAE,
        "n_same_TEST_subjects":len(y),
        "bootstrap_reps":399,
        "fixed_thresholds_no_TEST_retuning":True,
    }


def run(out:Path,*,smoke:bool=False):
    if out.exists():
        raise FileExistsError("Original frozen V43 evidence cannot be overwritten")
    devices=(20261707,) if smoke else tuple(REGISTERED_DEVICES)
    ntest=220 if smoke else 600
    ntr=1000;nv=800
    report=[];paired=[];monitor_receipts=[];reference_receipts=[]
    raw_fits=[];frozen={};model_to_save={};raw_to_save={}
    for dev in devices:
        w=generate_device_v43(
            device_seed=dev,n_train=ntr,n_valid=nv,n_test=ntest)
        train,v1,v2,te=(w[k] for k in (
            "train","valid1","valid2","test"))
        assert not np.intersect1d(train,te).size
        # TRAIN and VALID1/2 split readings use healthy sensor/monitor
        # conditions; no current monitor derived from oracle TEST faults.
        tr=_read(w,train,physical_fault="none",
                 monitor_fault="honest",stress="MCAR20",
                 stage=2,monitor_seed=0)
        va1=_read(w,v1,physical_fault="none",
                  monitor_fault="honest",stress="MCAR20",
                  stage=3,monitor_seed=0)
        va2=_read(w,v2,physical_fault="none",
                  monitor_fault="honest",stress="MCAR20",
                  stage=4,monitor_seed=0)
        m=fit_live_nuisance_monitor_v43(
            _monitor_inputs(tr),_monitor_inputs(va1),_monitor_inputs(va2))
        monitor_receipts.append({
            "device_seed":dev,"family":w["family"],
            "monitor":m.audit(),"calibrated_only_TRAIN_and_dual_VALID":True,
        })
        if not m.passed_both_VALIDs:
            # A failed monitor does not magically become safe; study
            # can proceed and correctly refuse every V43 estimate.
            print(json.dumps({"event":"V43_MONITOR_REFUSED_DEVICE",
                              "device_seed":dev,
                              "reason":m.reason}),flush=True)
        v37=fit_centered_threeview_v37(
            tr["diff"],tr["diff_mask"],
            independent_view_noise_documented=True,
            seed=20264891+dev%101,bootstrap=24)
        adaptive=fit_adaptive_source_v41(
            v37,tr["diff"],tr["diff_mask"])
        R=w["independent_reference_readout_CALIBRATION_ONLY"]
        # Source fitting already finished without even handling R.
        tr_ref=train[:300];v1_ref=v1[:200];v2_ref=v2[:200]
        rt=_read(w,tr_ref,physical_fault="none",monitor_fault="honest",
                 stress="MCAR20",stage=2,monitor_seed=0)
        r1=_read(w,v1_ref,physical_fault="none",monitor_fault="honest",
                 stress="MCAR20",stage=3,monitor_seed=0)
        r2=_read(w,v2_ref,physical_fault="none",monitor_fault="honest",
                 stress="MCAR20",stage=4,monitor_seed=0)
        r2bad=_read(w,v2_ref,physical_fault="off_drift",
                    monitor_fault="honest",stress="MCAR20",
                    stage=4,monitor_seed=0)
        source_certs={}
        for validfault,rec in (("none",r2),("off_drift",r2bad)):
            obj=fit_physical_reference_v42(
                adaptive,rt["diff"],rt["diff_mask"],R[tr_ref],
                r1["diff"],r1["diff_mask"],R[v1_ref],
                rec["diff"],rec["diff_mask"],R[v2_ref])
            source_certs[validfault]=obj
            reference_receipts.append({
                "device_seed":dev,"VALID2_fault":validfault,
                "independent_reference_budget_TRAIN":300,
                "independent_reference_budget_VALID1":200,
                "independent_reference_budget_VALID2":200,
                "audit":obj.audit(),
                "no_latent_E_H_S_in_concept_source_fit":True,
            })
        x_train,_=_raw_model_features(rt)
        x_v1,_=_raw_model_features(r1)
        x_v2,_=_raw_model_features(r2)
        x_v2bad,_=_raw_model_features(r2bad)
        raw_models={}
        for validfault,second in (("none",x_v2),("off_drift",x_v2bad)):
            for name in METHODS:
                all700=name.endswith("R700")
                x=(np.vstack((x_train,x_v1,second))
                   if all700 else x_train)
                rr=(np.concatenate((R[tr_ref],R[v1_ref],R[v2_ref]))
                    if all700 else R[tr_ref])
                fitted=fit_raw_reference_v43(name,x,rr)
                raw_models[(validfault,name)]=fitted
                raw_fits.append({
                    "device_seed":dev,"valid2_fault":validfault,
                    "method":name,"noisy_R_training_labels":len(rr),
                    "source_sensor_input_features":int(x.shape[1]),
                    "TEST_R_or_physical_E_used":False,
                })
        model_to_save[(dev,"live")]=m
        for validfault,ref in source_certs.items():
            model_to_save[(dev,validfault)]=ref
        for key,rm in raw_models.items():
            raw_to_save[(dev,)+key]=rm
        truth=w["true_slow_source_energy_POSTFIT_AUDIT_ONLY"][te].copy()
        # Frozen V43 primary physical threshold from TRAIN true E is
        # known only to simulator post-fit AUDIT; no estimator sees it.
        e_tr=w["true_slow_source_energy_POSTFIT_AUDIT_ONLY"][train]
        truth_iqr=float(np.quantile(e_tr,.75)-np.quantile(e_tr,.25))
        if truth_iqr<=.01:
            raise AssertionError("Synthetic physical energy has no useful dynamic range")
        frozen[f"device{dev}_TRUE_E_TEST_AUDIT_ONLY"]=truth.copy()
        for case_index,(case,phys,mon_fault,val_fault) in enumerate(WORLDS):
            for stress in STRESSES:
                # For this particular TEST fault, the same physical
                # sensor cohort, source true E, monitor original noise
                # and dropout masks are used in each method comparison.
                rec=_read(w,te,physical_fault=phys,monitor_fault=mon_fault,
                          stress=stress,stage=10,monitor_seed=0)
                hist=predict_physical_reference_v42(
                    source_certs[val_fault],rec["diff"],rec["diff_mask"])
                mon=evaluate_live_nuisance_monitor_v43(
                    m,*_monitor_inputs(rec))
                live=gate_physical_reference_v43(hist,mon)
                rraw,source_available=_raw_model_features(rec)
                forecast={
                    "V42_HISTORICAL":{
                        "pred":hist["physical_reference_estimated_energy"],
                        "accepted":hist["physical_reference_allowed"],
                        "interval":(
                            hist["physical_reference_interval_lower"],
                            hist["physical_reference_interval_upper"]),
                    },
                    "V43_LIVE_MONITORED":{
                        "pred":live["physical_reference_estimated_energy"],
                        "accepted":live["physical_reference_allowed"],
                        "interval":(
                            live["physical_reference_interval_lower"],
                            live["physical_reference_interval_upper"]),
                    },
                }
                for name in METHODS:
                    p=predict_raw_reference_v43(
                        raw_models[(val_fault,name)],rraw)
                    forecast[name+"_UNGATED"]={
                        "pred":p,"accepted":np.ones(len(p),dtype=bool),
                        "interval":None}
                    forecast[name+"_GATED"]={
                        "pred":p,
                        "accepted":source_available&mon[
                            "accepted_matching_background"],
                        "interval":None}
                prefix=f"device{dev}_{case}_{stress}_"
                frozen[prefix+"nuisance_RMS"]=mon["current_MONITOR_RMS"].copy()
                frozen[prefix+"monitor_accepted"]=mon[
                    "accepted_matching_background"].copy()
                frozen[prefix+"monitor_measured"]=mon[
                    "measured_monitor_sufficient"].copy()
                frozen[prefix+"chosen_source_pair"]=hist[
                    "selected_pair"].copy()
                frozen[prefix+"single_pair_uncorroborated"]=hist[
                    "single_pair_uncorroborated"].copy()
                for method in METHODS_ALL:
                    f=forecast[method]
                    predictions=np.asarray(f["pred"],dtype=np.float64)
                    accepted=np.asarray(f["accepted"],dtype=bool)
                    frozen[prefix+method+"_pred"]=predictions.copy()
                    frozen[prefix+method+"_accepted"]=accepted.copy()
                    result=_score(
                        predictions,accepted,truth,iqr=truth_iqr,
                        baseline_interval=f["interval"])
                    report.append({
                        "device_seed":dev,"family":w["family"],
                        "case":case,"stress":stress,
                        "method":method,
                        "truth_train_IQR_for_AUDIT_ONLY":truth_iqr,
                        "current_monitor_detected_alarm_rate_ALL_TEST":float(
                            mon["alarm_TRUE_observed_background_difference"].mean()),
                        "current_monitor_measured_fraction":float(
                            mon["measured_monitor_sufficient"].mean()),
                        "current_monitor_accept_fraction":float(
                            mon["accepted_matching_background"].mean()),
                        "historical_reference_R_used_at_TEST":False,
                        "live_test_nuisance_control_is_NEW_external_information":
                            method=="V43_LIVE_MONITORED" or method.endswith("_GATED"),
                        **result,
                    })
                # Save main paired comparisons ALL patient denominators.
                if stress in ("MCAR20","MCAR40") and case in (
                    "healthy","OFF_drift","monitor_blind_drift",
                    "monitor_false_alarm","OFF_source_leak",
                ):
                    for j,(a,b) in enumerate((
                        ("V42_HISTORICAL","V43_LIVE_MONITORED"),
                        ("V43_LIVE_MONITORED","RAW_HGB_R300_GATED"),
                        ("V43_LIVE_MONITORED","RAW_HGB_R700_GATED"),
                    )):
                        paired.append({
                            "device_seed":dev,"case":case,"stress":stress,
                            "method_a":a,"method_b":b,
                            **_paired_differences(
                                forecast[a],forecast[b],truth,iqr=truth_iqr,
                                seed=20264911+dev%101*100+
                                     case_index*10+j+
                                     STRESSES.index(stress)*1000),
                        })
        print(json.dumps({
            "event":"V43_DEVICE_FINISHED","device_seed":dev,
            "family":w["family"],"monitor_training_threshold":m.train_threshold_RMS,
            "monitor_VALID_false_alarm":[
                m.valid1_false_alarm_fraction,m.valid2_false_alarm_fraction],
            "reference_valid2_normal_approved":{
                k:v.two_VALIDs_approved for k,v in
                source_certs["none"].pair_certs.items()},
            "reference_valid2_fault_approved":{
                k:v.two_VALIDs_approved for k,v in
                source_certs["off_drift"].pair_certs.items()},
            "main_MCAR20":[{
                "case":case,"method":name,
                "coverage":next(x["coverage"] for x in report if
                    x["device_seed"]==dev and x["case"]==case and
                    x["stress"]=="MCAR20" and x["method"]==name),
                "wrong_and_accepted_per_all":next(
                    x["wrong_and_accepted_per_all"] for x in report if
                    x["device_seed"]==dev and x["case"]==case and
                    x["stress"]=="MCAR20" and x["method"]==name)}
                for case in ("healthy","OFF_drift","monitor_blind_drift",
                             "OFF_source_leak")
                for name in ("V42_HISTORICAL","V43_LIVE_MONITORED",
                             "RAW_HGB_R700_GATED")],
        }),flush=True)

    out.mkdir(parents=True,exist_ok=False)
    models=out/"FROZEN_MODELS";models.mkdir()
    for (dev,key),model in model_to_save.items():
        path=models/f"device{dev}_{key}.json"
        if key=="live":
            save_live_monitor_v43(path,model)
        else:
            save_physical_reference_v42(path,model)
    # These joblib files are a trusted LOCAL SCIENTIFIC ARTIFACT only;
    # the separate public runtime loads JSON source/monitor, NOT joblib.
    for (dev,v2,name),fitted in raw_to_save.items():
        joblib.dump(fitted,models/f"device{dev}_{v2}_{name}.joblib")
    main={
        "status":"V43_SMOKE_ONLY" if smoke else
            "V43_LIVE_NUISANCE_AND_MATCHED_RAW_REFERENCE_EXECUTED",
        "devices":list(devices),"device_families":{
            str(k):REGISTERED_DEVICES[k] for k in devices},
        "n_train":ntr,"n_valid1":nv//2,"n_valid2":nv//2,
        "n_test":ntest,
        "n_physical_reference_labels_TRAIN":300,
        "n_physical_reference_labels_VALID1":200,
        "n_physical_reference_labels_VALID2":200,
        "n_worlds":len(WORLDS),"n_stresses":len(STRESSES),
        "n_methods":len(METHODS_ALL),
        "n_method_condition_metrics":len(report),
        "n_pre_registered_same_subject_comparisons":len(paired),
        "n_raw_models_fit":len(raw_fits),
        "n_monitor_calibrations":len(monitor_receipts),
        "n_historical_reference_calibrations":len(reference_receipts),
        "true_physical_E_used_ONLY_in_post_test_audit":True,
        "new_current_nuisance_control_is_external_side_information":True,
        "reference_R_never_available_for_TEST_inference":True,
        "raw_reference_regressors_use_same_300_or_700_R_label_budgets":True,
        "source_concepts_are_not_physically_certified_in_real_world":True,
        "no_OUTCOME_Y_used_in_source_monitor_or_raw_fit":True,
        "summary":report,
        "paired_TEST_comparisons":paired,
        "monitor_calibrations":monitor_receipts,
        "source_reference_calibrations":reference_receipts,
        "raw_reference_baseline_training":raw_fits,
    }
    (out/"STATUS.json").write_text(json.dumps(
        main,indent=2,ensure_ascii=False,allow_nan=False))
    (out/"SUMMARY.json").write_text(json.dumps(
        report,indent=2,ensure_ascii=False,allow_nan=False))
    (out/"PAIRED_TEST_COMPARISONS.json").write_text(json.dumps(
        paired,indent=2,ensure_ascii=False,allow_nan=False))
    np.savez_compressed(out/"FROZEN_TEST_PHYSICAL_PREDICTIONS.npz",**frozen)
    print(json.dumps({
        "event":"V43_COMPLETE",
        "status":main["status"],"n_conditions":len(report),
        "n_paired":len(paired),
        "n_models_json":len(model_to_save),
        "n_supervised_raw_regressors":len(raw_to_save),
    }),flush=True)
    return main


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--out-dir",type=Path,required=True)
    options=ap.parse_args()
    run(options.out_dir,smoke=options.smoke)
