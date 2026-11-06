"""V43 independent audit of raw immutable TEST physical values and decisions.

This only loads frozen TEST prediction arrays and true E for scoring. No
source fit, no supervised reference model fit, no nuisance threshold
updates and no access to original device latent H/S/U.

Recomputes all 1600 method×fault×mask×device scores, 120 predeclared
paired patient-bootstrapped physical error/coverage comparisons and
mandatory OFF/monitor missingness abstentions.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

from scripts.experiment_live_nuisance_reference_v43 import (
    STRESSES,WORLDS,METHODS_ALL,_score,
    _paired_differences,
)

REPORT_METHODS=("V42_HISTORICAL","V43_LIVE_MONITORED")


def audit(folder:Path,output:Path):
    if output.exists():
        raise FileExistsError("Independent audit may not overwrite original result")
    data=json.loads((folder/"STATUS.json").read_text())
    if (data["status"]!="V43_LIVE_NUISANCE_AND_MATCHED_RAW_REFERENCE_EXECUTED"
        or len(data["devices"])!=4):
        raise ValueError("Need full frozen V43 TEST with all four NEW devices")
    npz=np.load(folder/"FROZEN_TEST_PHYSICAL_PREDICTIONS.npz",
                allow_pickle=False)
    maximum=0.; verified=0;missing_off=0;missing_mon=0
    for row in data["summary"]:
        dev,case,stress,method=(row[k] for k in (
            "device_seed","case","stress","method"))
        truth=npz[f"device{dev}_TRUE_E_TEST_AUDIT_ONLY"]
        prefix=f"device{dev}_{case}_{stress}_"
        pred=npz[prefix+method+"_pred"]
        used=npz[prefix+method+"_accepted"]
        if (truth.shape!=(600,) or pred.shape!=(600,)
            or used.shape!=(600,) or used.dtype!=np.bool_):
            raise AssertionError("Incorrect raw patient identity or 600 subject count")
        if not np.isfinite(truth).all():
            raise AssertionError("Missing AUDIT only true physical energy")
        if method in REPORT_METHODS:
            # Physical R interval lower/upper were evaluated in the
            # original source function but aren't necessary for the
            # core physical error recomputation in this independent audit.
            check=_score(pred,used,truth,
                iqr=row["truth_train_IQR_for_AUDIT_ONLY"])
        else:
            check=_score(pred,used,truth,
                iqr=row["truth_train_IQR_for_AUDIT_ONLY"])
        for name in (
            "coverage","wrong_given_accepted",
            "wrong_and_accepted_per_all",
            "correct_and_accepted_per_all","norm_MAE_accepted",
        ):
            a,b=check[name],row[name]
            if a is None or b is None:
                if a is not None or b is not None:
                    raise AssertionError(f"Bad absent physical statistic: {name}")
            else:
                maximum=max(maximum,abs(float(a)-float(b)))
        if (check["n_accepted"]!=row["n_accepted"]
            or check["n_physically_wrong_accepted"]!=
               row["n_physically_wrong_accepted"]):
            raise AssertionError("Wrong physical correctness denominators")
        if case=="OFF_missing" and method in REPORT_METHODS:
            if used.any():
                raise AssertionError("Named physical estimate despite missing source OFF")
            missing_off+=600
        if case=="monitor_missing" and (
            method=="V43_LIVE_MONITORED" or method.endswith("_GATED")):
            if used.any():
                raise AssertionError("External nuisance monitor was NOT available")
            missing_mon+=600
        if method=="V43_LIVE_MONITORED":
            prior=npz[prefix+"V42_HISTORICAL_accepted"]
            ctrl=npz[prefix+"monitor_accepted"]
            np.testing.assert_array_equal(used,prior&ctrl)
        if method.endswith("_GATED"):
            if np.any(used & ~npz[prefix+"monitor_accepted"]):
                raise AssertionError("Raw reference baseline ignores same nuisance monitor")
        verified+=1
    paired=0
    for row in data["paired_TEST_comparisons"]:
        dev,case,stress,ma,mb=(row[k] for k in (
            "device_seed","case","stress","method_a","method_b"))
        pref=f"device{dev}_{case}_{stress}_"
        a={"pred":npz[pref+ma+"_pred"],
           "accepted":npz[pref+ma+"_accepted"]}
        b={"pred":npz[pref+mb+"_pred"],
           "accepted":npz[pref+mb+"_accepted"]}
        truth=npz[f"device{dev}_TRUE_E_TEST_AUDIT_ONLY"]
        truth_iqr=next(x["truth_train_IQR_for_AUDIT_ONLY"]
                       for x in data["summary"]
                       if x["device_seed"]==dev and x["case"]==case
                       and x["stress"]==stress)
        ci_case=next(i for i,c in enumerate(WORLDS)
                     if c[0]==case)
        j=(
            0 if (ma,mb)==("V42_HISTORICAL","V43_LIVE_MONITORED")
            else 1 if mb=="RAW_HGB_R300_GATED"
            else 2 if mb=="RAW_HGB_R700_GATED" else None)
        if j is None:
            raise AssertionError("Unexpected unregistered comparison")
        rec=_paired_differences(
            a,b,truth,iqr=truth_iqr,
            seed=20264911+dev%101*100+
                 ci_case*10+j+STRESSES.index(stress)*1000)
        for k in (
            "delta_wrong_and_accepted_per_all_b_minus_a",
            "delta_correct_and_accepted_per_all_b_minus_a",
            "delta_normalized_MAE_on_same_accepted_b_minus_a",
        ):
            a0,b0=rec[k],row[k]
            if a0 is None or b0 is None:
                if a0 is not None or b0 is not None:
                    raise AssertionError("Recomputed same-population MAE support mismatch")
            else:
                maximum=max(maximum,abs(float(a0)-float(b0)))
        for k in (
            "paired_TEST_subject_95pct_wrong_delta_CI",
            "paired_TEST_subject_95pct_correct_delta_CI",
            "paired_95pct_normalized_MAE_on_same_accepted",
        ):
            a0,b0=rec[k],row[k]
            if a0 is None or b0 is None:
                if a0 is not None or b0 is not None:
                    raise AssertionError("Bootstrapped interval support mismatch")
            else:
                maximum=max(maximum,float(max(
                    np.abs(np.asarray(a0)-np.asarray(b0)))))
        if rec["intersection_n"]!=row["intersection_n"]:
            raise AssertionError("Paired conditional populations differ")
        paired+=1
    if (verified!=1600 or paired!=120
        or missing_off!=4*len(STRESSES)*2*600
        or missing_mon!=4*len(STRESSES)*5*600
        or maximum>=1e-12):
        raise AssertionError(
            f"Wrong V43 original receipts {verified}/{paired}/{missing_off}/{missing_mon}/{maximum}")
    state={
        "status":"V43_INDEPENDENT_PHYSICAL_ERROR_AND_CI_VERIFIED",
        "verified_1600_original_synthetic_TEST_method_conditions":verified,
        "verified_120_prespecified_paired_patient_contrasts":paired,
        "max_abs_original_vs_recalculated_error":maximum,
        "n_confirmed_no_OFF_concept_source_refusals":missing_off,
        "n_confirmed_no_live_monitor_gated_refusals":missing_mon,
        "no_ground_truth_used_for_monitor_fit":True,
        "no_TEST_physical_R_used_for_R300_R700_baseline_fit":True,
        "all_CI_conditional_on_four_simulated_device_cohorts":True,
        "not_a_physical_semantics_guarantee":True,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(
        state,indent=2,ensure_ascii=False,allow_nan=False))
    print(json.dumps({
        "event":"V43_INDEPENDENT_AUDIT",
        "verified_scores":verified,"paired_subject_intervals":paired,
        "max_abs_metric_error":maximum,
        "no_OFF_refusals":missing_off,
        "no_monitor_refusals":missing_mon,
    }),flush=True)
    return state


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--evidence-dir",type=Path,required=True)
    parser.add_argument("--out-file",type=Path,required=True)
    a=parser.parse_args()
    audit(a.evidence_dir,a.out_file)
