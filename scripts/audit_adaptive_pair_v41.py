"""Independent V41 post-run audit of all masks, gates, risk predictions.

Recomputes from immutable per-subject TEST observations/probabilities:
all 7680 AUPRC/AUROC/NLL/Brier, 3840 summary rows, EXACT K2 fallback
for 816 condition/model receipts if source OFF is unavailable, 52
pre-registered paired-test comparisons with additional EXPLORATORY
NLL confidence intervals, and 1280 metadata receipts for selected
real independent pair views (AB/AC/BC) including forbidden use of a
missing view.

Never fits/changes a model, TEST risk threshold, source gauge or gate.
Results are conditional on four simulated devices + 3 prefit K2s/device,
not confidence intervals across all possible institutions/devices.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,roc_auc_score,log_loss,brier_score_loss,
)

NATIVE={"K2","ON_AB","DIFF_AB","DIFF_ABC",
        "DIFF_ADAPTIVE","DIFF_ANY_UNCHECKED"}
SEEDS=(11,23,37)


def audit_v41(directory:Path,output:Path,*,reps=399):
    if output.exists():
        raise FileExistsError("Independent V41 audit must not rewrite receipts")
    if reps!=399:
        raise ValueError("Frozen 399 paired TEST subject resamples only")
    data=json.loads((directory/"STATUS.json").read_text())
    if (data["status"]!="V41_MATCHED_RAW_DIFFERENCE_NONLINEAR_EXECUTED"
        or len(data["devices"])!=4 or len(data["NN_seeds"])!=3):
        raise ValueError("Need complete executed immutable V41 evidence")
    archive=np.load(
        directory/"FROZEN_TEST_PREDICTIONS.npz",allow_pickle=False)
    maximum=0.
    for row in data["test_metric_rows"]:
        dev,case,stress,method,seed=(
            row["device_seed"],row["case"],row["stress"],
            row["method"],row["NN_seed"])
        y=archive[f"device{dev}_{case}_TEST_Y"]
        p=archive[f"device{dev}_{case}_{stress}_{method}_seed{seed}"]
        if (y.shape!=(600,) or p.shape!=(600,)
            or not np.isin(y,[0,1]).all()
            or not np.isfinite(p).all()):
            raise AssertionError("Missing or misaligned real TEST subject predictions")
        pp=np.clip(p,1e-7,1-1e-7)
        new={
            "auprc":average_precision_score(y,pp),
            "auroc":roc_auc_score(y,pp),
            "nll":log_loss(y,pp,labels=[0,1]),
            "brier":brier_score_loss(y,pp),
        }
        for k,v in new.items():
            maximum=max(maximum,abs(float(v)-row["metrics"][k]))
        budget=(1800 if method.endswith("ALL1800") else 1000)
        if row["n_outcome_training_labels"]!=budget:
            raise AssertionError("Misreported outcome-label budget")
    if maximum>=1e-12:
        raise AssertionError(f"Raw V41 TEST metric integrity error {maximum}")
    for summary in data["summary"]:
        chosen=[r for r in data["test_metric_rows"]
                if all(r[k]==summary[k] for k in (
                    "device_seed","case","stress","method"))]
        if len(chosen)!=(3 if summary["method"] in NATIVE else 1):
            raise AssertionError("Incomplete summary risk cohort")
        for key,original in (
            ("AUPRC","auprc"),("AUROC","auroc"),
            ("NLL","nll"),("Brier","brier"),
        ):
            if abs(float(np.mean([
                r["metrics"][original] for r in chosen
            ]))-summary[key])>=1e-12:
                raise AssertionError("TEST summary not reproducible from predictions")
    fallback=0
    for device in data["devices"]:
        for case in sorted(set(r["case"] for r in data["summary"])):
            for stress in sorted(set(r["stress"] for r in data["summary"])):
                if not(case=="OFF_reference_unavailable"
                       or stress=="missing_reference"):
                    continue
                for seed in SEEDS:
                    baseline=archive[
                        f"device{device}_{case}_{stress}_K2_seed{seed}"]
                    for method in (
                        "DIFF_AB","DIFF_ABC",
                        "DIFF_ADAPTIVE","DIFF_ANY_UNCHECKED"):
                        candidate=archive[
                            f"device{device}_{case}_{stress}_{method}_seed{seed}"]
                        np.testing.assert_array_equal(candidate,baseline)
                        fallback+=1
    if fallback!=4*(8+7)*3*4:
        raise AssertionError("At least one missing ON/OFF reference kept novel source")
    receipts=data["per_condition_source_receipts"]
    violations=[]
    allowed_when_view_missing={
        "missing_view_A":{"BC"},
        "missing_view_B":{"AC"},
        "missing_view_C":{"AB"},
    }
    for receipt in receipts:
        if not (receipt["no_imputed_sensor_values"] and
                receipt["no_Y_used_to_select_pair"] and
                receipt["source_not_physically_certified"]):
            raise AssertionError("Untrue source provenance receipt")
        if receipt["method"] not in ("DIFF_ADAPTIVE","DIFF_ANY_UNCHECKED"):
            continue
        must=allowed_when_view_missing.get(receipt["stress"])
        if must is not None:
            choices=receipt["chosen_measured_pair_counts"]
            for name,used in choices.items():
                if name not in must and used:
                    violations.append({
                        "device":receipt["device_seed"],
                        "stress":receipt["stress"],"method":receipt["method"],
                        "selected_missing_view_pair":name,
                        "n_invalid":used,
                    })
    if violations:
        raise AssertionError("An adaptive source claimed data from an absent sensor view")
    exploratory=[]
    for device in data["devices"]:
        ci=[c for c in data["paired_AP_CIs"]
            if c["device_seed"]==device]
        if len(ci)!=13:
            raise AssertionError("Registered 13 patient AP contrasts missing")
        for k,c in enumerate(ci):
            case,stress,a,b=(c[z] for z in (
                "case","stress","method_a","method_b"))
            y=archive[f"device{device}_{case}_TEST_Y"].astype(np.float64)
            def matrix(method):
                indices=SEEDS if method in NATIVE else (None,)
                arr=np.stack([
                    archive[f"device{device}_{case}_{stress}_{method}_seed{s}"]
                    for s in indices])
                if len(arr)==1:
                    arr=np.repeat(arr,len(SEEDS),axis=0)
                return np.clip(arr,1e-7,1-1e-7)
            pa,pb=matrix(a),matrix(b)
            y=y[None,:]
            la=-(y*np.log(pa)+(1-y)*np.log1p(-pa))
            lb=-(y*np.log(pb)+(1-y)*np.log1p(-pb))
            delta=la-lb
            rng=np.random.default_rng(20264601+device%101*100+k)
            ids=rng.integers(0,600,size=(reps,600))
            lower,upper=np.quantile(
                delta[:,ids].mean(axis=(0,2)),[.025,.975])
            exploratory.append({
                "device_seed":device,
                "case":case,"stress":stress,
                "method_a":a,"method_b":b,
                "delta_mean_NLL_a_minus_b":float(delta.mean()),
                "paired_TEST_subject_95pct_NLL_ci":[float(lower),float(upper)],
                "n_fixed_native_K2_seeds":3,
                "n_shared_TEST_subjects":600,
                "resamples":reps,
                "exploratory_additional_proper_score_not_preregistered_primary":True,
            })
    assert len(exploratory)==52
    if (len(data["test_metric_rows"])!=7680
        or len(data["summary"])!=3840
        or len(receipts)!=1280
        or len(data["dual_VALID_gates"])!=480):
        raise AssertionError("Incomplete frozen V41 evidence array")
    result={
        "status":"V41_INDEPENDENT_ALL_FROZEN_TEST_METRICS_AUDITED",
        "n_verified_original_TEST_rows":7680,
        "n_verified_aggregated_rows":3840,
        "max_abs_metric_difference":maximum,
        "n_verified_exact_missing_OFF_returns_to_original_K2":fallback,
        "n_source_measurement_provenance_receipts":len(receipts),
        "n_illegal_missing_view_pair_choices":len(violations),
        "n_saved_dual_VALID_risk_receipts":len(data["dual_VALID_gates"]),
        "n_registered_paired_AP_intervals":len(data["paired_AP_CIs"]),
        "n_exploratory_paired_NLL_intervals":len(exploratory),
        "no_source_H_S_or_y_admission_tuning_on_TEST":True,
        "single_pair_is_measurement_coverage_not_semantic_certification":True,
        "original_physical_ON_OFF_switch_is_external":True,
        "paired_NLL_exploratory":exploratory,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(
        result,indent=2,ensure_ascii=False,allow_nan=False))
    print(json.dumps({
        "event":"V41_COMPLETE_INDEPENDENT_AUDIT",
        "verified_metric_rows":7680,
        "verified_summary_rows":3840,
        "maximum_score_difference":maximum,
        "exact_fallback_checks":fallback,
        "illegal_pair_choices":len(violations),
        "registered_AP_intervals":len(data["paired_AP_CIs"]),
        "additional_NLL_intervals":len(exploratory),
    }),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--evidence-dir",type=Path,required=True)
    p.add_argument("--out-file",type=Path,required=True)
    args=p.parse_args()
    audit_v41(args.evidence_dir,args.out_file)
