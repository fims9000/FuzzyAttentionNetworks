"""V49 preregistered expert-intervention study on FROZEN base predictions.

Selection strictly uses MODEL and MEASURED support, not Y or independent
synthetic physical assay. The expensive virtual expert is consulted only
for the selected subset. Reject -> exact original K2. No retraining.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, log_loss, brier_score_loss
from conceptfan_discovery.cross_environment_transport import paired_test_ap_difference
from conceptfan_discovery.verified_concept_intervention_v49 import (
    fit_independent_reference_v49,synthetic_expert_verdict_v49,
    choose_review_patients_v49,apply_verified_concept_interventions_v49,
)

DEVICES=(20262907,20262909,20262921,20262923)
SEEDS=(11,23,37)
CASES=("healthy_energy","common_U8_matched","unknown_OFF_drift","OFF_source_leak")
STRESS=("MCAR20","MCAR40","MCAR50","missing_reference")
BUDGETS=(.10,.20)
N_RANDOM_REPEATS=3
PAIRS=(
    ("unknown_OFF_drift","MCAR40","ACTIVE20","V48_UNREVIEWED"),
    ("unknown_OFF_drift","MCAR40","ACTIVE20","RANDOM20"),
    ("healthy_energy","MCAR40","ACTIVE20","V48_UNREVIEWED"),
    ("common_U8_matched","MCAR40","ACTIVE20","V48_UNREVIEWED"),
    ("unknown_OFF_drift","MCAR40","HGB_ACTIVE20","HGB_UNREVIEWED"),
)


def _load_json(base:Path,name:str):
    return json.loads((base/name).read_text())


def _scores(y,p):
    y=np.asarray(y,dtype=int)
    p=np.asarray(p,dtype=float)
    p=np.clip(p,1e-7,1-1e-7)
    return {
        "n":len(y),"positives":int(y.sum()),
        "auprc":float(average_precision_score(y,p)),
        "nll":float(log_loss(y,p,labels=[0,1])),
        "brier":float(brier_score_loss(y,p)),
    }


def _rng_seed(device,seed,scenario,stress,budget,repeat,method):
    return (20264901+(device%100)*100000+seed*1000+
            CASES.index(scenario)*100+STRESS.index(stress)*20+
            int(budget*100)*5+repeat+('HGB'==method)*10000000)


def run(base:Path,out:Path,*,smoke:bool=False):
    if out.exists():
        raise FileExistsError("Never overwrite frozen reviewer TEST evidence")
    original=_load_json(base,"STATUS.json")
    devices=(DEVICES[0],) if smoke else DEVICES
    seeds=(SEEDS[0],) if smoke else SEEDS
    if (original["device_seeds"]!=list(devices) or
            original["native_seeds"]!=list(seeds)
            or original["status"]!=(
                "V49_SMOKE" if smoke else "V49_BASE_FROZEN_PREDICTIONS_EXECUTED")):
        raise AssertionError("Wrong fresh V49 TEST base / model weights")
    probs=np.load(base/"FROZEN_TEST_PREDICTIONS.npz",allow_pickle=False)
    masks=np.load(base/"FROZEN_SOURCE_AND_USE_FLAGS.npz",allow_pickle=False)
    q_evidence=np.load(base/"FROZEN_RISK_EVIDENCE.npz",allow_pickle=False)
    # This sealed file is a SYNTHETIC idealized EXPERT ASSAY. It does not
    # participate in model training or ANY review-patient selection.
    assay=np.load(base/"SEALED_IDEALIZED_EXPERT_ASSAY_AUDIT_ONLY.npz",
                  allow_pickle=False)
    if len(assay.files)!=len(devices)*4:
        raise AssertionError("Sealed physical expert assay incomplete")
    calibrated={}
    for dev in devices:
        calibrated[dev]=fit_independent_reference_v49(
            assay[f"device{dev}_TRAIN_measured_q"],
            assay[f"device{dev}_TRAIN_measured_support"],
            assay[f"device{dev}_TRAIN_energy"])
    outputs={};receipts={};test_rows=[];gate_audits=[];assays=[]
    count_reviewed=0;count_changed=0
    for device in devices:
        calib=calibrated[device]
        assays.append({"device":device,"TRAIN_only_calibration":calib.audit(),
                       "not_deployable_without_external_assay":True})
        for case in CASES:
            # The TEST hidden scalar is stored solely for simulated expert
            # adjudication *after* label-free review selection is fixed.
            true_e=assay[f"device{device}_TEST_energy"]
            yy=probs[f"device{device}_{case}_TEST_Y"]
            for stress in STRESS:
                key=f"device{device}_{case}_{stress}"
                q=q_evidence[key+"_source_q"]
                if len(q)!=len(yy) or len(true_e)!=len(yy):
                    raise AssertionError("Incomplete frozen subjects in physical assessor")
                for seed in seeds:
                    basep=probs[f"{key}_K2_seed{seed}"]
                    current=probs[f"{key}_V48_JOINT_LOGODDS_seed{seed}"]
                    hgb=probs[f"{key}_RAW_DIFF_HGB_ALL1800_seedNone"]
                    used=masks[f"{key}_V48_JOINT_LOGODDS_seed{seed}_used"]
                    if not np.array_equal(current[~used],basep[~used]):
                        raise AssertionError("Original V48 does not use exact K2 refusal")
                    candidates=int(np.count_nonzero(used))
                    baseline_recs=(
                        ("K2_UNREVIEWED",basep),
                        ("V48_UNREVIEWED",current),
                        ("HGB_UNREVIEWED",hgb),
                    )
                    for method,p in baseline_recs:
                        outputs[f"{key}_{method}_seed{seed}_rep0"]=p.copy()
                        test_rows.append({
                            "device":device,"seed":seed,"case":case,"stress":stress,
                            "method":method,"repeat":0,"metrics":_scores(yy,p),
                            "review_budget":0,"selected":0,"expert_rejected":0,
                            "eligible_source_count":candidates,
                        })
                    for budget_fraction in BUDGETS:
                        budget=int(np.floor(budget_fraction*len(yy)))
                        badge=int(budget_fraction*100)
                        for selection,repeat_count in (
                            ("active",1),("random",N_RANDOM_REPEATS)):
                            for repeat in range(repeat_count):
                                chosen=choose_review_patients_v49(
                                    current,basep,used,budget=budget,selection=selection,
                                    seed=_rng_seed(device,seed,case,stress,budget_fraction,
                                                   repeat,"V48"))
                                # Explicit external expensive reference only on chosen rows.
                                actions,expert=synthetic_expert_verdict_v49(
                                    calib,q,true_e,chosen)
                                refid=np.where(chosen,f"synthetic_assay_device{device}","")
                                amended=apply_verified_concept_interventions_v49(
                                    basep,current,q,used,actions,
                                    link="bounded_logodds",gamma=0.,
                                    reference_ids=refid)
                                method=f"{selection.upper()}{badge}"
                                output_key=f"{key}_{method}_seed{seed}_rep{repeat}"
                                outputs[output_key]=amended["probability"]
                                receipts[output_key+"_selected"]=chosen.copy()
                                receipts[output_key+"_actions"]=actions.astype("<U8")
                                count_reviewed+=expert["reviewed"]
                                count_changed+=expert["rejected"]
                                test_rows.append({
                                    "device":device,"seed":seed,"case":case,"stress":stress,
                                    "method":method,"repeat":repeat,
                                    "metrics":_scores(yy,amended["probability"]),
                                    "review_budget":budget,"selected":expert["reviewed"],
                                    "expert_rejected":expert["rejected"],
                                    "expert_confirmed":expert["confirmed"],
                                    "eligible_source_count":candidates,
                                    "independent_assay_certified":calib.certified_available,
                                    "selection_used_y_or_true_energy":False,
                                    "correction_uses_frozen_K2_without_retraining":True,
                                })
                        if badge==20:
                            chosen=choose_review_patients_v49(
                                hgb,basep,used,budget=budget,selection="active",
                                seed=_rng_seed(device,seed,case,stress,budget_fraction,0,"HGB"))
                            actions,expert=synthetic_expert_verdict_v49(
                                calib,q,true_e,chosen)
                            refid=np.where(chosen,f"synthetic_assay_device{device}","")
                            amended=apply_verified_concept_interventions_v49(
                                basep,hgb,q,used,actions,link="bounded_logodds",
                                gamma=0.,reference_ids=refid)
                            method="HGB_ACTIVE20"
                            output_key=f"{key}_{method}_seed{seed}_rep0"
                            outputs[output_key]=amended["probability"]
                            receipts[output_key+"_selected"]=chosen.copy()
                            receipts[output_key+"_actions"]=actions.astype("<U8")
                            test_rows.append({
                                "device":device,"seed":seed,"case":case,"stress":stress,
                                "method":method,"repeat":0,
                                "metrics":_scores(yy,amended["probability"]),
                                "review_budget":budget,"selected":expert["reviewed"],
                                "expert_rejected":expert["rejected"],
                                "expert_confirmed":expert["confirmed"],
                                "eligible_source_count":candidates,
                                "independent_assay_certified":calib.certified_available,
                                "HGB_is_hybrid_using_external_source_assay":True,
                            })
        print(json.dumps({"event":"V49_DEVICE_REVIEWED","device":device,
                          "assay_certified":calib.certified_available,
                          "n_TEST_review_rows":len(test_rows)}),flush=True)

    # Frozen comparison panels: one CI per device/contrast, paired over
    # TEST people and 3 native seeds; random is *mean of 3 fixed reruns*.
    def matrix(dev,case,stress,method):
        key=f"device{dev}_{case}_{stress}"
        rows=[]
        for seed in seeds:
            repetitions=range(N_RANDOM_REPEATS) if method=="RANDOM20" else (0,)
            for rep in repetitions:
                rows.append(outputs[f"{key}_{method}_seed{seed}_rep{rep}"])
        out=np.stack(rows,axis=0)
        if out.shape[0]==len(seeds):
            return np.repeat(out,N_RANDOM_REPEATS,axis=0)
        return out
    paired=[]
    bootstrap=199 if smoke else 399
    for device in devices:
        for i,(case,stress,a,b) in enumerate(PAIRS):
            y=probs[f"device{device}_{case}_TEST_Y"]
            paired.append({
                "device":device,"case":case,"stress":stress,"a":a,"b":b,
                **paired_test_ap_difference(
                    y,matrix(device,case,stress,a),
                    matrix(device,case,stress,b),
                    seed=20264981+(device%100)*100+i,resamples=bootstrap),
            })
    summary=[]
    for device in devices:
        for case in CASES:
            for stress in STRESS:
                for method in ("K2_UNREVIEWED","V48_UNREVIEWED","HGB_UNREVIEWED",
                               "ACTIVE10","ACTIVE20","RANDOM10","RANDOM20","HGB_ACTIVE20"):
                    rr=[r for r in test_rows if r["device"]==device and
                        r["case"]==case and r["stress"]==stress and r["method"]==method]
                    count=len(seeds)*(N_RANDOM_REPEATS if method.startswith("RANDOM") else 1)
                    if len(rr)!=count:
                        raise AssertionError("Dropped seed or random review repetition")
                    summary.append({
                        "device":device,"case":case,"stress":stress,"method":method,
                        "AUPRC":float(np.mean([r["metrics"]["auprc"] for r in rr])),
                        "NLL":float(np.mean([r["metrics"]["nll"] for r in rr])),
                        "Brier":float(np.mean([r["metrics"]["brier"] for r in rr])),
                        "reviewed":float(np.mean([r["selected"] for r in rr])),
                        "rejected":float(np.mean([r["expert_rejected"] for r in rr])),
                        "eligible":float(np.mean([r["eligible_source_count"] for r in rr])),
                    })
    result={
        "status":"V49_SMOKE_REVIEW" if smoke else "V49_REGISTERED_IDEALIZED_EXPERT_REVIEW_EXECUTED",
        "frozen_protocol":"docs/medical/CONCEPTFAN_V49_FROZEN_EXPERT_INTERVENTION_PROTOCOL_20261106.md",
        "device_seeds":list(devices),"native_seeds":list(seeds),
        "n_test_per_device":original["n_test"],
        "n_expert_action_arrays":len(receipts),
        "n_test_method_rows":len(test_rows),"n_summary_rows":len(summary),
        "n_preregistered_paired_CIs":len(paired),
        "n_selected_interventions_across_experiment":count_reviewed,
        "n_source_rejections_across_experiment":count_changed,
        "real_model_weights_not_retrained_at_review":True,
        "expert_sees_unavailable_hidden_synthetic_energy":True,
        "only_selected_cases_receive_expert_verdict":True,
        "model_review_selection_uses_no_label_y_or_reference_energy":True,
        "external_assay_not_present_in_real_data":True,
        "not_a_physical_semantics_certification_of_ConceptFAN":True,
    }
    out.mkdir(parents=True,exist_ok=False)
    for name,value in (
        ("STATUS.json",result),("TRAIN_ASSAY_CALIBRATION.json",assays),
        ("REVIEW_TEST_ROWS.json",test_rows),("REVIEW_SUMMARY.json",summary),
        ("PAIRED_CIS.json",paired)):
        (out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))
    np.savez_compressed(out/"FROZEN_REVIEW_PREDICTIONS.npz",**outputs)
    np.savez_compressed(out/"FROZEN_REVIEW_SELECTION_AND_ACTIONS.npz",**receipts)
    print(json.dumps({
        "event":"V49_REVIEW_COMPLETE",
        "healthy_MCAR40":[r for r in summary if
            r["case"]=="healthy_energy" and r["stress"]=="MCAR40"],
        "unknown_OFF_drift_MCAR40":[r for r in summary if
            r["case"]=="unknown_OFF_drift" and r["stress"]=="MCAR40"],
    }),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--base-dir",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    run(args.base_dir,args.out_dir,smoke=args.smoke)
