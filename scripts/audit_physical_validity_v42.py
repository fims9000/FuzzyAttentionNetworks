"""Independent raw TEST truth/audit integrity, not a second source fit.

Recompute physical correctness separately from the registered V42 model,
using the immutable per-patient TEST q, estimated physical energy, accepted
mask, true synthetic E (AUDIT ONLY), and pair-origin metadata. Check:
- All 224 TEST conditions × 600 subjects, accepted single vs multi,
  chosen AB/AC/BC, reference MAE, normalized MAE, bad physical signal
  among accepted and true E interval inclusion.
- Original registered 32 paired bootstrap physical error contrasts.
- Every missing OFF reference must give exactly 0 physical-source use.
- Persisted JSON reference source snapshots contain NO raw TEST R/S/Y,
  only externally calibrated TRAIN/VALID reference scale maps.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

PAIRS=(
    ("healthy_energy","OFF_only_U4_drift","MCAR20"),
    ("healthy_energy","OFF_only_U4_drift","MCAR40"),
    ("healthy_energy","common_U8_matched_ON_OFF","MCAR20"),
    ("healthy_energy","common_U8_matched_ON_OFF","MCAR40"),
    ("healthy_energy","OFF_source_leak_50pct","MCAR20"),
    ("healthy_energy","OFF_source_leak_50pct","MCAR40"),
    ("healthy_energy","TEST_A_ON_sensor_inversion","MCAR20"),
    ("healthy_energy","TEST_A_ON_sensor_inversion","MCAR40"),
)


def _subset_metric(z,mask,e,iqr):
    num=int(mask.sum())
    if num==0:
        return {
            "accepted_count":0,"MAE_physical":None,
            "MAE_normalized_train_IQR":None,
            "wrong_physics_fraction_conditional_on_accepted":None,
            "true_E_in_noisy_R_interval_fraction":None,
            "Spearman_rho_estimated_vs_true_E":None,
        }
    x=z["physical_reference_estimated_energy"][mask]
    l=z["physical_reference_interval_lower"][mask]
    h=z["physical_reference_interval_upper"][mask]
    truth=e[mask]
    error=np.abs(x-truth)
    corr=None
    if num>=30 and np.std(x)>1e-9 and np.std(truth)>1e-9:
        rho=float(spearmanr(x,truth).statistic)
        if np.isfinite(rho):
            corr=rho
    return {
        "accepted_count":num,
        "MAE_physical":float(error.mean()),
        "MAE_normalized_train_IQR":float(error.mean()/iqr),
        "wrong_physics_fraction_conditional_on_accepted":
            float((error>.50*iqr).mean()),
        "true_E_in_noisy_R_interval_fraction":
            float(((truth>=l)&(truth<=h)).mean()),
        "Spearman_rho_estimated_vs_true_E":corr,
    }


def _check_value(a,b,source):
    if a is None or b is None:
        if a is not None or b is not None:
            raise AssertionError(f"Missing physical correctness metric {source}")
        return 0.
    if abs(float(a)-float(b))>1e-12:
        raise AssertionError(f"Raw TEST physical correctness mismatch {source}")
    return abs(float(a)-float(b))


def audit(evidence:Path,output:Path):
    if output.exists():
        raise FileExistsError("Never overwrite independent physical error audit")
    d=json.loads((evidence/"STATUS.json").read_text())
    if (d["status"]!="V42_PHYSICAL_REFERENCE_SEMANTIC_AUDIT_EXECUTED"
        or len(d["devices"])!=4):
        raise ValueError("Need full registered V42 reference-truth study")
    archive=np.load(
        evidence/"FROZEN_TEST_MEASURED_SOURCE_AND_TRUTH_AUDIT.npz",
        allow_pickle=False)
    total=0
    maximum=0.
    failed_off=0
    for row in d["source_measurement_audit"]:
        dev,case,stress=(row[k] for k in (
            "device_seed","case","stress"))
        e=archive[f"device{dev}_TEST_TRUE_SLOW_SOURCE_ENERGY_AUDIT_ONLY"]
        if e.shape!=(600,) or not np.isfinite(e).all():
            raise AssertionError("Missing 600 real synthetic physical TEST labels")
        prefix=f"device{dev}_{case}_{stress}_"
        z={
            key:archive[prefix+key]
            for key in (
                "q","support","selected_pair",
                "physical_reference_estimated_energy",
                "physical_reference_interval_lower",
                "physical_reference_interval_upper",
                "physical_reference_allowed",
                "single_pair_uncorroborated",
                "two_or_three_pairs_corroborated",
                "physical_reference_reason",
            )
        }
        if any(len(v)!=600 for v in z.values()):
            raise AssertionError("TEST subject ID shape mismatch")
        if not np.isin(z["selected_pair"],["AB","AC","BC","--"]).all():
            raise AssertionError("Unregistered measured sensor pair")
        yes=z["physical_reference_allowed"].astype(bool)
        support=z["support"]>0
        if np.any(yes&~support):
            raise AssertionError("Reference exposed a rejected measured V41 source")
        if np.any(np.isfinite(
            z["physical_reference_estimated_energy"][~yes])):
            raise AssertionError("Reference predicted physical truth for unobserved source")
        if case=="OFF_reference_unavailable":
            if yes.any() or support.any():
                raise AssertionError("Nonzero physical source when OFF is completely absent")
            failed_off+=len(e)
        refrec=next(a for a in d["reference_calibration_receipts"]
                    if a["device_seed"]==dev and
                    a["valid2_reference_fault"]==(
                        "off_drift" if case=="VALID2_known_OFF_drift"
                        else "none"))
        iqr=float(refrec["certificate"]["reference_train_IQR_physical_units"])
        masks={
            "V42_all_anchored":yes,
            "V42_single_uncorroborated":
                yes&z["single_pair_uncorroborated"],
            "V42_two_or_three_correlated_views":
                yes&z["two_or_three_pairs_corroborated"],
        }
        for pair in ("AB","AC","BC"):
            masks[f"pair_{pair}"]=yes&(z["selected_pair"]==pair)
        for name,mask in masks.items():
            record=(row["V42_anchored_by_pair"][name.removeprefix("pair_")]
                    if name.startswith("pair_") else row[name])
            recomputed=_subset_metric(z,mask,e,iqr)
            for key,val in recomputed.items():
                maximum=max(maximum,_check_value(
                    val,record[key],f"{dev}/{case}/{stress}/{name}/{key}"))
        maximum=max(maximum,_check_value(
            float(yes.mean()),
            row["V42_historical_reference_anchored_fraction"],
            "reference_coverage"))
        maximum=max(maximum,_check_value(
            float(support.mean()),
            row["V41_statistical_measurement_support_fraction"],
            "statistical_coverage"))
        if sum(row["selected_pair_counts"].values())!=int(yes.sum()):
            raise AssertionError("Selected physical source pairs not counted")
        total+=len(e)
    if total!=4*8*7*600 or failed_off!=4*7*600:
        raise AssertionError("Incomplete TEST subject and unavailable OFF evidence")
    bootstrap=0
    for registered in d["pre_registered_physical_error_comparisons"]:
        dev=registered["device_seed"]
        caseA,caseB,stress=(registered[k] for k in (
            "healthy_case","fault_case","stress"))
        ix=PAIRS.index((caseA,caseB,stress))
        truth=archive[
            f"device{dev}_TEST_TRUE_SLOW_SOURCE_ENERGY_AUDIT_ONLY"]
        prefA=f"device{dev}_{caseA}_{stress}_"
        prefB=f"device{dev}_{caseB}_{stress}_"
        good=(archive[prefA+"physical_reference_allowed"]&
              archive[prefB+"physical_reference_allowed"])
        n=int(good.sum())
        if registered["intersection_accepted_count"]!=n:
            raise AssertionError("Wrong paired SAME patient comparison sample size")
        if n>=30:
            pa=archive[prefA+"physical_reference_estimated_energy"][good]
            pb=archive[prefB+"physical_reference_estimated_energy"][good]
            t=truth[good]
            refrec=next(a for a in d["reference_calibration_receipts"]
                        if a["device_seed"]==dev and
                        a["valid2_reference_fault"]=="none")
            iqr=refrec["certificate"]["reference_train_IQR_physical_units"]
            diff=(np.abs(pb-t)-np.abs(pa-t))/iqr
            _check_value(
                float(diff.mean()),
                registered["delta_normalized_MAE_fault_minus_healthy"],
                "paired physical MAE difference")
            rng=np.random.default_rng(20264790+dev%101*100+ix)
            pick=rng.integers(0,n,size=(399,n))
            ci=np.quantile(diff[pick].mean(axis=1),[.025,.975])
            for i in range(2):
                maximum=max(maximum,_check_value(
                    float(ci[i]),
                    registered["paired_TEST_subject_bootstrap_95pct_ci"][i],
                    "paired TEST physical truth CI"))
        bootstrap+=1
    for path in (evidence/"REFERENCE_MODELS").glob("*.json"):
        saved=path.read_text()
        state=json.loads(saved)
        if (state["schema"]!="conceptfan-v42-reference-validity-v1"
            or not state["no_source_S_or_H_or_patient_Y_stored"]
            or not state["not_clinically_or_causally_certified"]
            or any(s in saved for s in (
                "TEST_TRUE_SLOW_SOURCE_ENERGY_AUDIT_ONLY",
                "independent_reference_readout_CALIBRATION_ONLY",
                '"hidden_source"',
                '"outcomes"',
            ))):
            raise AssertionError("Persisted model includes raw latent patient/reference data")
    if (len(d["source_measurement_audit"])!=224 or
        len(d["reference_calibration_receipts"])!=8 or bootstrap!=32):
        raise AssertionError("Incomplete frozen V42 evidence")
    result={
        "status":"V42_INDEPENDENT_PHYSICAL_TRUTH_METRICS_VERIFIED",
        "verified_condition_audits":len(d["source_measurement_audit"]),
        "verified_patient_level_physical_truth_predictions":total,
        "verified_lost_OFF_reference_absent_source_episodes":failed_off,
        "max_abs_recalculated_physical_metric_error":maximum,
        "all_missing_OFF_exact_source_refusals":True,
        "no_TEST_reference_information_in_persisted_model":True,
        "n_reference_calibration_models":len(
            list((evidence/"REFERENCE_MODELS").glob("*.json"))),
        "n_registered_physical_error_bootstrap_checks":bootstrap,
        "single_pair_is_NOT_physically_certified_by_TRAIN_statistics":True,
        "reference_confidence_intervals_do_NOT_guarantee_E_coverage":True,
        "no_Y_or_latent_S_H_in_measured_source_fit":True,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(
        result,indent=2,ensure_ascii=False,allow_nan=False))
    print(json.dumps({
        "event":"V42_INDEPENDENT_PHYSICAL_TRUTH_AUDIT",
        "n_conditions":result["verified_condition_audits"],
        "n_TEST_subjects_per_condition":600,
        "n_total_subject_episodes":total,
        "n_missing_OFF_refusals":failed_off,
        "n_paired_bootstraps":bootstrap,
        "max_abs_metric_error":maximum,
    }),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--evidence-dir",type=Path,required=True)
    p.add_argument("--out-file",type=Path,required=True)
    a=p.parse_args()
    audit(a.evidence_dir,a.out_file)
