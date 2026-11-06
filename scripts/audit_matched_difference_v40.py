"""Independent POST-hoc numerical verification of V40 FROZEN raw TEST evidence.

Recompute every AUPRC/AUROC/NLL/Brier from immutable per-subject
predictions for four NEW devices, without generator, train head or fit.

Check all 3240 TEST evaluations, 1800 group means, known Y-label
budgets, 312 exact K2 refusals if OFF is unobserved. NLL paired bootstrap
is an EXPLORATORY complement to the already preregistered 32 AP CIs,
conditional on four specific synthetic devices and frozen NN fits.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    average_precision_score,roc_auc_score,log_loss,brier_score_loss,
)

NATIVE_METHODS={"K2","ON_AB","DIFF_AB","DIFF_ABC"}
SEEDS=(11,23,37)


def audit_v40(evidence:Path,output:Path,*,bootstrap=399):
    if output.exists():
        raise FileExistsError("Never overwrite original immutable V40 audit")
    if bootstrap!=399:
        raise ValueError("Only frozen 399 draws allowed in independent analysis")
    d=json.loads((evidence/"STATUS.json").read_text())
    if (d["status"]!="V40_MATCHED_RAW_DIFFERENCE_NONLINEAR_EXECUTED"
        or len(d["devices"])!=4 or len(d["NN_seeds"])!=3):
        raise ValueError("Only completed four-device V40 study may be audited")
    archive=np.load(
        evidence/"FROZEN_TEST_PREDICTIONS.npz",allow_pickle=False)
    max_err=0.
    for row in d["test_metric_rows"]:
        device,case,stress,method,seed=(
            row["device_seed"],row["case"],row["stress"],
            row["method"],row["NN_seed"])
        y=archive[f"device{device}_{case}_TEST_Y"]
        x=archive[f"device{device}_{case}_{stress}_{method}_seed{seed}"]
        if (len(y)!=600 or y.shape!=x.shape or
            not np.isin(y,[0,1]).all() or
            not np.isfinite(x).all()):
            raise AssertionError("Misaligned TEST subject predictions")
        p=np.clip(np.asarray(x,dtype=np.float64),1e-7,1-1e-7)
        new={
            "auprc":average_precision_score(y,p),
            "auroc":roc_auc_score(y,p),
            "nll":log_loss(y,p,labels=[0,1]),
            "brier":brier_score_loss(y,p),
        }
        for k,v in new.items():
            max_err=max(max_err,abs(float(v)-row["metrics"][k]))
        expected=(
            d["n_train"]+d["n_valid1"]+d["n_valid2"]
            if method.endswith("ALL1800") else d["n_train"])
        if row["n_outcome_training_labels"]!=expected:
            raise AssertionError("Raw/NN labeled budget misreported")
    if max_err>=1e-12:
        raise AssertionError("TEST score cannot be reproduced from raw probabilities")
    for row in d["summary"]:
        underlying=[r for r in d["test_metric_rows"]
                    if all(r[k]==row[k] for k in
                           ("device_seed","case","stress","method"))]
        if len(underlying)!=(3 if row["method"] in NATIVE_METHODS else 1):
            raise AssertionError("Summary may omit or double-count NN models")
        for p,q in (("AUPRC","auprc"),("AUROC","auroc"),
                    ("NLL","nll"),("Brier","brier")):
            val=float(np.mean([r["metrics"][q] for r in underlying]))
            if abs(val-row[p])>=1e-12:
                raise AssertionError("Group summary not supported by frozen TEST")
    refusals=0
    for device in d["devices"]:
        for case in sorted(set(r["case"] for r in d["summary"])):
            for stress in sorted(set(r["stress"] for r in d["summary"])):
                if case!="OFF_reference_unavailable" and stress!="missing_reference":
                    continue
                for seed in SEEDS:
                    prior=archive[f"device{device}_{case}_{stress}_K2_seed{seed}"]
                    for method in ("DIFF_AB","DIFF_ABC"):
                        after=archive[
                            f"device{device}_{case}_{stress}_{method}_seed{seed}"]
                        np.testing.assert_array_equal(prior,after)
                        refusals+=1
    if refusals!=4*(5+8)*3*2:
        raise AssertionError("Missing reference exact-fallback count was altered")
    paired_nll=[]
    for device in d["devices"]:
        cons=[x for x in d["paired_AP_CIs"] if x["device_seed"]==device]
        if len(cons)!=8:
            raise AssertionError("Pre-registered 8 paired AP contrasts/device are missing")
        for i,row in enumerate(cons):
            case,stress,a,b=(row[k] for k in
                ("case","stress","method_a","method_b"))
            y=archive[f"device{device}_{case}_TEST_Y"].astype(np.float64)
            def forecasts(method):
                members=SEEDS if method in NATIVE_METHODS else (None,)
                a=np.stack([
                    archive[f"device{device}_{case}_{stress}_{method}_seed{s}"]
                    for s in members])
                if len(a)==1:
                    a=np.repeat(a,len(SEEDS),axis=0)
                return np.clip(a,1e-7,1-1e-7)
            a_pred,b_pred=forecasts(a),forecasts(b)
            ya=y[None,:]
            loss_a=-(ya*np.log(a_pred)+(1-ya)*np.log1p(-a_pred))
            loss_b=-(ya*np.log(b_pred)+(1-ya)*np.log1p(-b_pred))
            diff=loss_a-loss_b
            rng=np.random.default_rng(20264501+device%101*100+i)
            patient_ids=rng.integers(0,len(y),size=(bootstrap,len(y)))
            draws=diff[:,patient_ids].mean(axis=(0,2))
            low,high=np.quantile(draws,[.025,.975])
            paired_nll.append({
                "device_seed":device,"family":d["device_families"][str(device)],
                "case":case,"stress":stress,
                "method_a":a,"method_b":b,
                "delta_mean_NLL_a_minus_b":float(diff.mean()),
                "paired_95pct_NLL_ci":[float(low),float(high)],
                "n_shared_test_patients":600,
                "n_frozen_native_models":3,
                "resamples":bootstrap,
                "exploratory_complement_not_preregistered_primary_CI":True,
                "no_TEST_based_training_or_threshold_selection":True,
            })
    result={
        "status":"V40_INDEPENDENT_ALL_TEST_METRIC_RECOMPUTATION",
        "verified_original_TEST_metric_rows":len(d["test_metric_rows"]),
        "verified_aggregated_method_rows":len(d["summary"]),
        "max_absolute_metric_recomputation_error":max_err,
        "verified_exact_K2_missing_reference_fallbacks":refusals,
        "n_preregistered_matched_AP_contrasts":len(d["paired_AP_CIs"]),
        "n_additional_exploratory_paired_NLL_intervals":len(paired_nll),
        "labeled_budget_TRAIN":1000,
        "labeled_budget_TRAIN_and_VALID":1800,
        "all_TEST_scores_recomputed_from_original_saved_predictions":True,
        "no_H_S_or_TEST_borne_training":True,
        "no_clinical_source_identification_claim":True,
        "exploratory_NLL":paired_nll,
    }
    if (result["verified_original_TEST_metric_rows"]!=3240 or
        result["verified_aggregated_method_rows"]!=1800 or
        len(paired_nll)!=32):
        raise AssertionError("Missing complete V40 independent audit evidence")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(
        result,indent=2,ensure_ascii=False,allow_nan=False))
    print(json.dumps({
        "event":"V40_INDEPENDENT_AUDIT",
        "verified_TEST_rows":len(d["test_metric_rows"]),
        "verified_summary_rows":len(d["summary"]),
        "metric_max_error":max_err,
        "exact_missing_refusals":refusals,
        "n_extra_NLL_bootstraps":len(paired_nll),
    }),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--evidence-dir",type=Path,required=True)
    parser.add_argument("--out-file",type=Path,required=True)
    args=parser.parse_args()
    audit_v40(args.evidence_dir,args.out_file)
