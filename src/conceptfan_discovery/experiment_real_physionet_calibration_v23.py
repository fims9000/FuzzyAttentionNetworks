"""V23: REAL PhysioNet Set A, exact original V10 source protocol.

A disjoint 2000-patient TRAIN cohort fits the existing native K8 V10
ConceptFAN. The remaining 400 TRAIN labels fit only a two-parameter
positive affine probability calibration of the original native risk
logit. The official separately designated 400 CALIBRATION and 600 TEST
patients are NEVER evaluated, nor is the second half of VALIDATION.

Only the EXACT original 300 already-inspected development subjects are
reported. All external logistic/HGB models get the same total 2400 TRAIN
labels, 36 historical hours and selected measured sensor/recorded masks.
No clinical physical concept source labels or causal identity are inferred.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict,replace
from hashlib import sha256
from pathlib import Path
import tempfile

import numpy as np
import torch
import yaml
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import train_test_split

from .clinical_v10_external_benchmark import _load_and_match_v10_patient_cohort
from .clinical_multiscale_v10 import fit_multiscale
from .data import DiscoveryData,DiscoverySplit
from .experiment_external_predictors_v22e import _predictive_metrics
from .experiment_unseen_test_v22c import _fingerprint
from .external_baselines_v10 import fit_tabular_baseline,predict_tabular
from .train_only_native_calibration_v22f import (
    NativeAffineProbabilityCalibration,
    MIN_SLOPE,MAX_SLOPE,MAX_ABS_INTERCEPT,REGULARIZATION,
    OPTIMIZATION_STEPS,
)
from conceptfan_realdata.data import sha256_file


EXPECTED_PUBLIC_SET_A_SHA256="1433638192fda622e8bb11a4559b8af2dbb1c69f0e1c3b5b606cc12aabd0a4c5"
EXPECTED_PUBLIC_OUTCOMES_A_SHA256="2613ea60ccda29f87571a7d6b09ad130858a8ccd66325bd073365925026883c2"
SPLIT_SEED=20261104
TRAIN_FIT_PATIENTS=2000
RESERVED_TRAIN_CALIBRATION=400
ORIGINAL_TRAIN_PATIENTS=2400
REPEATED_DEVELOPMENT_PATIENTS=300
EXPECTED_PHYSICAL_K=8
EXPECTED_CHANNELS=(
    "DiasABP","HR","MAP","NIDiasABP","NIMAP","NISysABP",
    "SysABP","Temp","Urine","Weight","RespRate","GCS",
)
MODEL_SEEDS=(11,23,37)
EXTERNAL_METHODS=("logistic_history","hist_gradient_boosting")


def split_original_train_for_disjoint_calibration(data:DiscoveryData):
    if (len(data.train.y)!=ORIGINAL_TRAIN_PATIENTS
            or data.train.oracle_states is not None
            or len(set(map(int,data.train.episode_ids)))!=ORIGINAL_TRAIN_PATIENTS):
        raise ValueError("Invalid original patient-independent real TRAIN partition")
    all_rows=np.arange(ORIGINAL_TRAIN_PATIENTS)
    fit,cal=train_test_split(
        all_rows,test_size=RESERVED_TRAIN_CALIBRATION,
        random_state=SPLIT_SEED,
        stratify=data.train.y.astype(int),
    )
    fit=np.sort(np.asarray(fit,dtype=int))
    cal=np.sort(np.asarray(cal,dtype=int))
    if (len(fit)!=TRAIN_FIT_PATIENTS or len(cal)!=RESERVED_TRAIN_CALIBRATION
            or len(set(fit)&set(cal))!=0
            or set(fit)|set(cal)!=set(all_rows)
            or len(np.unique(data.train.y[fit]))!=2
            or len(np.unique(data.train.y[cal]))!=2):
        raise AssertionError("Failed stratified disjoint real TRAIN/calibration cohort split")

    def subset(original:DiscoverySplit,indices)->DiscoverySplit:
        return DiscoverySplit(
            x=original.x[indices].copy(),
            mask=original.mask[indices].copy(),
            future=original.future[indices].copy(),
            future_mask=original.future_mask[indices].copy(),
            y=original.y[indices].copy(),
            oracle_states=None,
            episode_ids=original.episode_ids[indices].copy(),
        )

    model_fit=DiscoveryData(
        train=subset(data.train,fit),
        validation=data.validation,
        receipt={**data.receipt,"real_native_model_fit_patients":TRAIN_FIT_PATIENTS,
                 "separate_train_calibration_labels":RESERVED_TRAIN_CALIBRATION,
                 "old_original_calibration_and_test_still_sealed":True},
        medians=data.medians,
        scales=data.scales,
    )
    calibration=subset(data.train,cal)
    return model_fit,calibration,{
        "train_model_n":len(fit),
        "train_calibration_n":len(cal),
        "train_model_positive":int(data.train.y[fit].sum()),
        "train_calibration_positive":int(data.train.y[cal].sum()),
        "train_id_sha256":sha256(np.ascontiguousarray(
            np.sort(model_fit.train.episode_ids).astype("<i8")
        ).tobytes()).hexdigest(),
        "calibration_id_sha256":sha256(np.ascontiguousarray(
            np.sort(calibration.episode_ids).astype("<i8")
        ).tobytes()).hexdigest(),
        "same_subjects_across_model_and_calibration":False,
        "same_original_2400_patient_train_label_budget":True,
    }


def collect_native_real_logits(model,split:DiscoverySplit,calibrator=None):
    model.eval()
    original=[]
    probabilities=[]
    maximum_raw_error=0.
    maximum_scaled_error=0.
    with torch.no_grad():
        for first in range(0,len(split.y),128):
            last=min(first+128,len(split.y))
            x=torch.from_numpy(split.x[first:last])
            mask=torch.from_numpy(split.mask[first:last])
            out=model(x,mask,process_mask=mask)
            maximum_raw_error=max(maximum_raw_error,float(out.decomposition_error()))
            z=out.logits.detach().cpu().numpy().astype(np.float64)
            original.extend(z.tolist())
            if calibrator is None:
                probabilities.extend(expit(z).tolist())
            else:
                adjusted,bias,phys,care=calibrator.decomposed_native_contributions(out)
                maximum_scaled_error=max(
                    maximum_scaled_error,
                    float((adjusted-(bias+phys.sum(-1)+care.sum(-1))).abs().max()),
                    float((adjusted-(calibrator.slope*out.logits+calibrator.intercept)).abs().max()),
                )
                probabilities.extend(calibrator.probabilities(z).tolist())
    if maximum_raw_error>2e-5 or maximum_scaled_error>2e-5:
        raise AssertionError("Real-patient calibrated logit bypassed native concept mediation")
    return np.asarray(original),np.asarray(probabilities),maximum_raw_error,maximum_scaled_error


def fit_reserved_400_train_affine(
    model_logits:np.ndarray,
    calibration_targets:np.ndarray,
    *,
    calibration_episode_ids:np.ndarray,
    native_gradient_episode_ids:np.ndarray,
)->NativeAffineProbabilityCalibration:
    z=np.asarray(model_logits,dtype=np.float64).reshape(-1)
    y=np.asarray(calibration_targets,dtype=np.float64).reshape(-1)
    cal_ids=list(map(int,calibration_episode_ids))
    fit_ids=list(map(int,native_gradient_episode_ids))
    if (len(z)!=RESERVED_TRAIN_CALIBRATION
            or len(y)!=RESERVED_TRAIN_CALIBRATION
            or len(cal_ids)!=RESERVED_TRAIN_CALIBRATION
            or len(set(cal_ids))!=len(cal_ids)
            or len(set(fit_ids))!=TRAIN_FIT_PATIENTS
            or set(cal_ids)&set(fit_ids)
            or not np.isfinite(z).all() or not np.isfinite(y).all()
            or not np.isin(y,[0,1]).all()
            or len(np.unique(y))!=2 or z.std()<1e-10):
        raise ValueError("ABSTAIN: real calibration must use exactly 400 disjoint TRAIN subjects")
    def objective(v):
        logscale,shift=map(float,v)
        slope=np.exp(logscale)
        calibrated=slope*z+shift
        return (
            float(np.mean(np.logaddexp(0.,calibrated)-y*calibrated))
            +REGULARIZATION*(logscale**2+shift**2)
        )
    original=float(np.mean(np.logaddexp(0.,z)-y*z))
    opt=minimize(
        objective,np.array([0.,0.],dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(np.log(MIN_SLOPE),np.log(MAX_SLOPE)),
                (-MAX_ABS_INTERCEPT,MAX_ABS_INTERCEPT)],
        options={"maxiter":OPTIMIZATION_STEPS,"ftol":1e-11},
    )
    if not opt.success or not np.isfinite(opt.x).all():
        raise ValueError("ABSTAIN: independent TRAIN probability calibration failed")
    slope=float(np.exp(opt.x[0]))
    shift=float(opt.x[1])
    fitted=float(np.mean(np.logaddexp(0.,slope*z+shift)-y*(slope*z+shift)))
    if not MIN_SLOPE<=slope<=MAX_SLOPE or not -MAX_ABS_INTERCEPT<=shift<=MAX_ABS_INTERCEPT:
        raise AssertionError("Out of frozen V22f affine calibration bounds")
    return NativeAffineProbabilityCalibration(
        slope=slope,intercept=shift,
        n_independent_train_calibration_cases=RESERVED_TRAIN_CALIBRATION,
        initial_train_bce=original,fitted_train_bce=fitted,
        optimizer_success=True,
    )


def run(
    *,
    set_a_zip:Path,
    outcomes_a:Path,
    original_v10_status:Path,
    reference_v6_config:Path,
    reference_v7_config:Path,
    native_v10_config:Path,
    external_config:Path,
    outdir:Path,
):
    if outdir.exists():
        raise FileExistsError("Never overwrite real-patient V23 research evidence")
    if (sha256_file(set_a_zip)!=EXPECTED_PUBLIC_SET_A_SHA256
            or sha256_file(outcomes_a)!=EXPECTED_PUBLIC_OUTCOMES_A_SHA256):
        raise ValueError("Official Set A/public outcome files differ from frozen V10 original SHA256")
    prior=json.loads(original_v10_status.read_text())
    if (prior.get("status")!="EXPLORATORY_V10_MATCHED_TRAINING_COMPLETE"
            or tuple(prior.get("measured_channels",[]))!=EXPECTED_CHANNELS
            or prior.get("latent_dim_only_computational_budget")!=EXPECTED_PHYSICAL_K
            or prior.get("calibration_or_test_used") is not False):
        raise ValueError("Original V10 SOURCE selection status is not the frozen prior")
    v6=yaml.safe_load(reference_v6_config.read_text())
    v7=yaml.safe_load(reference_v7_config.read_text())
    v10=yaml.safe_load(native_v10_config.read_text())
    ext=yaml.safe_load(external_config.read_text())
    if (not v6["protocol"]["test_and_calibration_sealed"]
            or not ext["protocol"]["old_test_and_calibration_sealed"]
            or not v10["protocol"]["test_calibration_sealed"]
            or not v7["protocol"]["no_test_calibration_or_new_clinical_significance"]
            or tuple(map(int,v6["training"]["initialization_seeds"]))!=MODEL_SEEDS):
        raise ValueError("Cannot change original frozen PhysioNet clinical protocol")
    torch.set_num_threads(int(v6["training"]["cpu_threads"]))
    all_data,basis,dev,identity=_load_and_match_v10_patient_cohort(
        source_zip=set_a_zip,
        outcome_file=outcomes_a,
        base_cfg=v6,prior_receipt=prior,
    )
    if (len(dev.y)!=REPEATED_DEVELOPMENT_PATIENTS
            or len(all_data.train.y)!=ORIGINAL_TRAIN_PATIENTS
            or all_data.receipt["channel_names"]!=list(EXPECTED_CHANNELS)
            or basis.shape!=(len(EXPECTED_CHANNELS),EXPECTED_PHYSICAL_K)
            or all_data.train.oracle_states is not None
            or dev.oracle_states is not None):
        raise AssertionError("Real clinical selected source/patient identity changed")
    fit_data,calibration,cohorts=split_original_train_for_disjoint_calibration(all_data)
    baseline_models={}
    from time import perf_counter
    for method in EXTERNAL_METHODS:
        before=perf_counter()
        predictor=fit_tabular_baseline(
            method,all_data.train.x,all_data.train.mask,all_data.train.mask,
            all_data.train.y,
            augmentation_fraction=float(ext["training"]["augmentation_fraction"]),
            seed=int(ext["training"]["tabular_training_seed"]),
        )
        baseline_models[method]=(predictor,round(perf_counter()-before,3))
    cases=[]
    for seed in MODEL_SEEDS:
        model,receipt=fit_multiscale(
            fit_data,basis,seed,v6,v7,v10,
            consistency=True,feature_exploration=False,
            expert_feedback=None,native_head_refit=False,
        )
        model.eval()
        checksum=_fingerprint(model)
        calibration_logits,_,cal_decomp_error,_=collect_native_real_logits(
            model,calibration,
        )
        cal=fit_reserved_400_train_affine(
            calibration_logits,calibration.y,
            calibration_episode_ids=calibration.episode_ids,
            native_gradient_episode_ids=fit_data.train.episode_ids,
        )
        if _fingerprint(model)!=checksum:
            raise AssertionError("Real TRAIN-only calibration modified native ConceptFAN weights")
        dev_raw,dev_prob,max_raw,_=collect_native_real_logits(model,dev)
        after,dev_calibrated,_,max_scaled=collect_native_real_logits(
            model,dev,calibrator=cal,
        )
        if not np.array_equal(dev_raw,after):
            raise AssertionError("Real model native logit changed after probability calibration")
        raw_metrics=_predictive_metrics(dev.y,dev_prob)
        adj_metrics=_predictive_metrics(dev.y,dev_calibrated)
        if (abs(raw_metrics["test_auprc"]-adj_metrics["test_auprc"])>1e-9
                or abs(raw_metrics["test_auroc"]-adj_metrics["test_auroc"])>1e-9):
            raise AssertionError("Positive affine calibration changed real-patient risk ranking")
        if _fingerprint(model)!=checksum:
            raise AssertionError("Re-evaluation changed real native concept weights")
        for arm,metrics,error in (
            ("raw_native_v10",raw_metrics,max_raw),
            ("disjoint_train_affine_calibrated_v10",adj_metrics,max_scaled),
        ):
            cases.append({
                "model":"conceptfan_v10_multiscale_consistency",
                "seed":seed,"arm":arm,
                "original_train_available_labels":ORIGINAL_TRAIN_PATIENTS,
                "native_model_gradient_patients":TRAIN_FIT_PATIENTS,
                "reserved_train_calibration_patients":RESERVED_TRAIN_CALIBRATION,
                "original_calibration_and_test_sealed":True,
                "previously_inspected_development_patients":len(dev.y),
                "measured_sensor_count":len(EXPECTED_CHANNELS),
                "native_physical_slots":EXPECTED_PHYSICAL_K,
                "no_external_raw_to_risk_predictor":True,
                "native_calibration_scale":cal.slope,
                "native_calibration_shift":cal.intercept,
                "native_logit_exact_additivity_error":error,
                "native_state_sha256_before_and_after_identical":True,
                "physical_causal_source_identified":False,
                **{k.replace("test_","development_"):v for k,v in metrics.items()},
            })
        print("REAL_DEVELOPMENT",seed,
              "raw AUPRC",round(raw_metrics["test_auprc"],5),
              "raw Brier",round(raw_metrics["test_brier"],5),
              "cal Brier",round(adj_metrics["test_brier"],5),
              "train-only slope",round(cal.slope,4),
              flush=True)

    for method in EXTERNAL_METHODS:
        predictor,seconds=baseline_models[method]
        p=predict_tabular(predictor,dev.x,dev.mask,dev.mask)
        metrics=_predictive_metrics(dev.y,p)
        cases.append({
            "model":method,
            "seed":None,"arm":"standard_full_2400_train_labels",
            "original_train_available_labels":ORIGINAL_TRAIN_PATIENTS,
            "native_model_gradient_patients":None,
            "reserved_train_calibration_patients":None,
            "original_calibration_and_test_sealed":True,
            "previously_inspected_development_patients":len(dev.y),
            "measured_sensor_count":len(EXPECTED_CHANNELS),
            "native_physical_slots":None,
            "no_external_raw_to_risk_predictor":False,
            "native_calibration_scale":None,
            "native_calibration_shift":None,
            "native_logit_exact_additivity_error":None,
            "native_state_sha256_before_and_after_identical":True,
            "physical_causal_source_identified":False,
            "train_seconds":seconds,
            **{k.replace("test_","development_"):v for k,v in metrics.items()},
        })
    if len(cases)!=8:
        raise AssertionError("All 6 native + 2 independent external real arms required")
    grouped=[]
    for method,arm in (
        ("conceptfan_v10_multiscale_consistency","raw_native_v10"),
        ("conceptfan_v10_multiscale_consistency","disjoint_train_affine_calibrated_v10"),
        ("logistic_history","standard_full_2400_train_labels"),
        ("hist_gradient_boosting","standard_full_2400_train_labels"),
    ):
        selected=[r for r in cases if r["model"]==method and r["arm"]==arm]
        grouped.append({
            "model":method,"arm":arm,"n_model_initializations":len(selected),
            "mean_development_auprc":float(np.mean([r["development_auprc"] for r in selected])),
            "mean_development_auroc":float(np.mean([r["development_auroc"] for r in selected])),
            "mean_development_brier":float(np.mean([r["development_brier"] for r in selected])),
            "mean_development_logloss":float(np.mean([r["development_logloss"] for r in selected])),
            "original_test_and_calibration_sealed":True,
            "clinically_validated_source":False,
        })
    outdir.mkdir(parents=True,exist_ok=False)
    keys=sorted({k for r in cases for k in r})
    with (outdir/"ALL_EIGHT_REAL_DEVELOPMENT_PREDICTORS.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader();w.writerows(cases)
    keys=sorted({k for r in grouped for k in r})
    with (outdir/"FOUR_REAL_DEVELOPMENT_ARM_SUMMARIES.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader();w.writerows(grouped)
    status={
        "status":"V23_REAL_PHYSIONET_ORIGINAL_TRAIN_ONLY_NATIVE_CALIBRATION",
        "n_official_set_a_episodes":4000,
        "n_original_train_episodes":2400,
        "n_model_gradient_train_episodes":2000,
        "n_disjoint_train_calibration_episodes":400,
        "n_original_validation_episodes":600,
        "n_repeatedly_inspected_development_episodes":300,
        "n_original_calibration_sealed":400,
        "n_original_test_sealed":600,
        "source_selected_measured_channels":list(EXPECTED_CHANNELS),
        "native_K_computational_budget":EXPECTED_PHYSICAL_K,
        "n_native_full_fits":3,
        "n_native_raw_plus_calibrated_scores":6,
        "n_external_full_train_baselines":2,
        "n_all_reported_arms":len(cases),
        "all_concept_paths_same_original_V10":True,
        "all_native_state_hashes_unchanged_after_calibration":True,
        "positive_global_affine_does_not_change_risk_ranking":True,
        "no_new_claim_of_physically_validated_concept_identity":True,
        "no_future_values_or_outcomes_as_model_input":True,
        "calibration_patients_from_original_train_only":True,
        "source_selection_train_only_and_no_calibration_label_leakage":True,
        "existing_development_repeatedly_viewed_not_prospective":True,
        "no_patient_level_predictions_or_identifiers_exported":True,
        "no_external_hospital_or_clinical_intervention_evaluated":True,
        "official_set_a_sha256":EXPECTED_PUBLIC_SET_A_SHA256,
        "official_outcomes_a_sha256":EXPECTED_PUBLIC_OUTCOMES_A_SHA256,
        "native_training_cohorts":cohorts,
        "prior_original_V10_source_cohort_receipt":identity,
        "all_scores":cases,
        "aggregate":grouped,
    }
    (outdir/"STATUS.json").write_text(json.dumps(status,indent=2,allow_nan=False))
    return status


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--set-a-zip",type=Path,required=True)
    p.add_argument("--outcomes-a",type=Path,required=True)
    p.add_argument("--v10-status",type=Path,required=True)
    p.add_argument("--reference-v6-config",type=Path,required=True)
    p.add_argument("--reference-v7-config",type=Path,required=True)
    p.add_argument("--native-v10-config",type=Path,required=True)
    p.add_argument("--external-config",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    run(
        set_a_zip=args.set_a_zip,outcomes_a=args.outcomes_a,
        original_v10_status=args.v10_status,
        reference_v6_config=args.reference_v6_config,
        reference_v7_config=args.reference_v7_config,
        native_v10_config=args.native_v10_config,
        external_config=args.external_config,
        outdir=args.out_dir,
    )
