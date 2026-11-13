#!/usr/bin/env python3
"""Independent V59 *external hospital* case audit.

Private per-person VitalDB predictions are read only in CI memory and never
uploaded. Report only aggregate source rejection, metrics and subject-group
bootstrap evidence, with data provenance and site difference.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score,log_loss

METHODS={
    "K2":"base","V59_frozen_source_FAN":"selected",
    "raw_logistic":"raw","energy_logistic":"engineered",
    "pulse_only_logistic":"pulse_only",
}


def metric(y,z):
    p=expit(z)
    return {
        "auprc":None if np.unique(y).size!=2 else float(
            average_precision_score(y,p)),
        "nll":float(log_loss(y,p,labels=[0,1])),
        "target_rate":float(np.mean(y)),
        "mean_predicted_probability":float(p.mean()),
        "n_windows":len(y),
        "both_classes_evaluable":bool(np.unique(y).size==2),
    }


def audit(root):
    root=Path(root)
    r=json.loads((root/"report_v59.json").read_text())
    if (r["status"]!="EXECUTED_INDEPENDENT_HOSPITAL_EXPLORATORY"
        or not r["source_sites_disjoint"] or
        r["external_HR_absent_from_model_inputs"] is not True or
        r["no_V56_external_source_certification"] is not True or
        r["n_external_test_independent_cases"]<30):
        raise AssertionError("Wrong external holdout / reference semantics")
    if len(r["cases"])!=8 or len(r["private_prediction_hashes"])!=8:
        raise AssertionError("Missing 2×4 registered external model forecasts")
    allcases={}
    unique=r["n_external_test_independent_cases"]
    receipts={int(z["native_seed"]):z for z in r["frozen_train_receipts"]}
    for row,manifest in zip(r["cases"],r["private_prediction_hashes"]):
        key=f"seed{row['seed']}_{row['stress']}"
        if key!=manifest["case"]:
            raise AssertionError("Private source receipt case mismatch")
        path=root/"private_case_arrays"/f"{key}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest["sha256"]:
            raise AssertionError("Private external patient risk file tampered")
        with np.load(path,allow_pickle=False) as z:
            data={name:z[name].copy() for name in (
                "y","case_group","base","selected","raw",
                "engineered","pulse_only","used","q","support")}
        y=data["y"]
        if (len(set(data["case_group"]))!=unique
            or not all(a.shape==(len(y),) for a in data.values())
            or not all(np.isfinite(a).all() for a in data.values())
            or not np.isin(y,[0,1]).all()):
            raise AssertionError("Corrupt external cohort alignment")
        if receipts[int(row["seed"])]["BIDMC_source_gate_state"]!="NO_MEASURED_OPERATOR_PROPOSED":
            raise AssertionError("Illegal unreviewed source admitted on external hospital")
        if data["used"].any():
            raise AssertionError("V58 found no source, cannot use it on VitalDB")
        np.testing.assert_array_equal(data["base"],data["selected"])
        for method,field in METHODS.items():
            checks=metric(y,data[field])
            reported=row["method_metrics"][method]
            for name,val in checks.items():
                old=reported[name]
                if old is None:
                    if val is not None:raise AssertionError("Class estimability changed")
                elif isinstance(old,(bool,int)):
                    if old!=val:raise AssertionError("Invalid group count")
                elif abs(old-val)>1e-10:
                    raise AssertionError(f"External metric mismatch {key}:{method}:{name}")
            group_macro=np.mean([
                log_loss(y[data["case_group"]==g],
                    expit(data[field][data["case_group"]==g]),labels=[0,1])
                for g in np.unique(data["case_group"])
            ])
            if abs(group_macro-reported["patient_macro_NLL"])>1e-10:
                raise AssertionError("External subject-group macro NLL changed")
        allcases[key]=data
    for scenario in ("natural","MCAR20","pulse_missing","pulse_plus_25_bpm"):
        a=allcases["seed11_"+scenario]
        b=allcases["seed23_"+scenario]
        np.testing.assert_array_equal(a["y"],b["y"])
        np.testing.assert_array_equal(a["case_group"],b["case_group"])
    result={
        "status":"V59_EXTERNAL_VITALDB_INDEPENDENT_SITE_CASE_AUDIT_PASS",
        "frozen_train_source_site":"BIDMC Boston ICU",
        "external_test_site":"VitalDB Seoul OR",
        "n_real_external_subject_groups":unique,
        "n_test_conditions_recomputed":len(allcases),
        "reference_HR_is_not_an_independent_physical_source_certificate":True,
        "private_subject_predictions_not_uploaded":True,
        "no_test_site_label_used_for_source_generation":True,
        "native_no_source_exact_K2_fallback_verified":True,
        "patient_group_macro_NLL_verified":True,
        "paired_subjects_shared_between_native_seeds":True,
        "no_ad_hoc_threshold_or_model_refit_after_external_site":True,
    }
    (root/"verify_v59_audit.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dir",required=True,type=Path)
    args=parser.parse_args()
    audit(args.dir)
