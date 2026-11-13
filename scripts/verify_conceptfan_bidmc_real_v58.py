#!/usr/bin/env python3
"""V58 independent audit of real BIDMC heldout model scores.

Reads 16 private runner-local TEST arrays (2 native seeds × 4 stresses? 8
arrays) and compares them to public aggregated metrics and source receipts.
The private arrays containing patient-level monitor-derived outcomes and risk
scores are NOT uploaded or printed. No raw patient IDs are stored.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score,log_loss

METHODS={
    "K2":"base",
    "V58_automatically_selected":"selected",
    "raw_logistic":"raw",
    "energy_logistic":"engineered",
    "pulse_only_logistic":"pulse_only",
}


def _metrics(y,z):
    p=expit(z)
    auprc=(None if np.unique(y).size<2
           else float(average_precision_score(y,p)))
    return {
        "auprc":auprc,
        "nll":float(log_loss(y,p,labels=[0,1])),
        "target_rate":float(y.mean()),
        "mean_predicted_probability":float(p.mean()),
        "n_windows":len(y),
        "both_classes_evaluable":bool(np.unique(y).size==2),
    }


def audit(root):
    root=Path(root)
    report=json.loads((root/"report_v58.json").read_text())
    manifest=json.loads((root/"group_split_manifest.json").read_text())
    cases=report["cases"]
    signatures=report["private_array_hash_manifest"]
    if (len(cases)!=8 or len(signatures)!=8
        or report["HR_is_both_training_label_and_test_reference"] is False
        or report["no_independent_assay_concept_certificate"] is False
        or manifest["identical_patient_in_multiple_splits"] is not False):
        raise AssertionError("Inadequate privacy/source scope or case count")
    risk_receipts={int(r["native_seed"]):r for r in report["case_receipts"]}
    valid_subjects=manifest["splits"]["test"]["patient_group_count"]
    if valid_subjects<6:raise AssertionError("Insufficient independent test patients")
    per_case={}
    for case,receipt in zip(cases,signatures):
        key=f"seed{case['seed']}_{case['stress']}"
        if receipt["case"]!=key:
            raise AssertionError("Mismatched masked case receipts")
        path=root/"private_case_arrays"/f"{key}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest()!=receipt["sha256"]:
            raise AssertionError("Private real TEST array hash mismatch")
        with np.load(path,allow_pickle=False) as z:
            data={k:z[k].copy() for k in (
                "y","base","selected","raw","engineered","pulse_only",
                "source_used","source_q","source_support","patient_group")}
        shapes={x.shape for x in data.values()}
        if len(shapes)!=1:
            raise AssertionError("Case mismatch: not all held-out windows aligned")
        y=data["y"]
        groups=data["patient_group"]
        if len(set(groups))!=valid_subjects:
            raise AssertionError("Patient-heldout group count drift")
        if not np.isin(y,[0,1]).all() or not all(
            np.isfinite(x).all() for x in data.values()):
            raise AssertionError("Invalid real TEST scores")
        candidate=risk_receipts[int(case["seed"])]["source_admitted"]
        used=data["source_used"].astype(bool)
        if np.any(used&(data["source_support"]<.5)):
            raise AssertionError("FAN source used without measured support")
        if not np.array_equal(data["base"][~used],data["selected"][~used]):
            raise AssertionError("Bitwise original K2 fallback violated")
        if candidate is None:
            if used.any():
                raise AssertionError("No admissible measured source, yet used one")
            np.testing.assert_array_equal(data["base"],data["selected"])
        else:
            gamma=float(candidate["gamma"])
            expected=data["base"]+gamma*np.where(used,data["source_q"]-.5,0.)
            np.testing.assert_allclose(
                data["selected"],expected,atol=2e-6,rtol=0)
        for name,field in METHODS.items():
            values=_metrics(y,data[field])
            for measure,val in values.items():
                old=case["test_metrics"][name][measure]
                if old is None:
                    if val is not None:raise AssertionError("Missing class estimability drift")
                elif isinstance(old,(bool,int)):
                    if old!=val:raise AssertionError("Class labels or counts drift")
                elif abs(old-val)>1e-10:
                    raise AssertionError(f"Real metric drift {key} {name} {measure}")
            patient_nll=np.mean([
                log_loss(y[groups==g],expit(data[field][groups==g]),labels=[0,1])
                for g in np.unique(groups)
            ])
            if abs(patient_nll-case["test_metrics"][name]["patient_macro_NLL"])>1e-10:
                raise AssertionError("Patient-group macro real NLL drift")
        per_case[key]=data
    # Both random native seeds use identical patient/HR labels for a
    # particular TEST stress: there are NOT 2× independent patients.
    for scenario in ("natural","MCAR20","pulse_missing","pulse_plus_25_bpm"):
        a=per_case[f"seed11_{scenario}"]
        b=per_case[f"seed23_{scenario}"]
        np.testing.assert_array_equal(a["y"],b["y"])
        np.testing.assert_array_equal(a["patient_group"],b["patient_group"])
    result={
        "status":"V58_REAL_BIDMC_INDEPENDENT_CASE_AUDIT_PASS",
        "validated_private_arrays":len(per_case),
        "private_arrays_not_uploaded":True,
        "n_unique_test_patient_groups":valid_subjects,
        "TEST_window_counts_not_independent_subject_counts":True,
        "native_seeds_share_same_REAL_test_patients":True,
        "ECG_HR_double_role_outcome_and_score_reference_disclosed":True,
        "source_is_not_physically_certified":True,
        "native_signed_risk_contributions_checked":True,
        "exact_K2_fallback_checked":True,
        "patient_group_macro_NLL_recomputed":True,
        "reported_candidate_counts":{
            str(r["native_seed"]):len(r["discovery_candidates_from_train_only"]["candidates"])
            for r in report["case_receipts"]},
    }
    (root/"verify_v58_audit.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dir",type=Path,required=True)
    a=p.parse_args()
    audit(a.dir)
