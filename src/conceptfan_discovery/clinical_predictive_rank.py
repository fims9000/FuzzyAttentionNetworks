"""Clinical stress audit for train-only unsupervised temporal rank.

Crucial negative control: a rank estimator that works on clean synthetic
linear systems need NOT identify physiological mechanisms in ICU records.
Inspect train patient subsample stability, missingness drift, and observable
predictive subspace reproducibility without consulting test/calibration data.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import yaml

from .lagged_rank import fit_lagged_predictive_rank
from .clinical_observability import (
    aggregate_hours, joint_pair_counts, support_selected_channels,
    support_selected_channels_across_subsamples,
)
from .physionet import prepare_physionet_discovery


def _basis_overlap(a: np.ndarray, b: np.ndarray, ka: int, kb: int) -> dict:
    if min(ka,kb)<1:
        return {"matched_dimensions":0,"mean_squared_principal_cosine":None}
    pa=a[:,:ka].T@b[:,:kb]
    similarity=np.linalg.svd(pa,compute_uv=False)
    return {
        "matched_dimensions":int(min(ka,kb)),
        "mean_squared_principal_cosine":float((similarity[:min(ka,kb)]**2).mean()),
        "principal_cosines":similarity[:min(ka,kb)].tolist(),
        "interpretation":"train-subsample observable predictive subspace agreement, not causal mechanism matching",
    }


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--raw-zip",type=Path,required=True)
    parser.add_argument("--outcomes",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error("Refusing to overwrite an existing audit")
    cfg=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not cfg["protocol"]["no_test_or_calibration_use"]:
        parser.error("This rank audit is train-only and cannot use sealed data")
    with tempfile.TemporaryDirectory(prefix="pd_clinical_rank_split_") as tmp:
        data=prepare_physionet_discovery(
            args.raw_zip,args.outcomes,Path(tmp),
            split_seed=int(cfg["data"]["split_seed"]),
            input_window=int(cfg["data"]["input_window"]),
            forecast_horizon=int(cfg["data"]["forecast_horizon"]),
            minimum_train_patient_coverage=float(cfg["data"]["minimum_train_patient_coverage"]),
        )
    x=data.train.x
    m=data.train.mask
    names=data.receipt["channel_names"]
    rank_cfg=cfg["rank"]
    rank_args={
        "lags":tuple(int(v) for v in rank_cfg["lags"]),
        "permutations":int(rank_cfg["permutations"]),
        "patient_bootstraps":int(rank_cfg["patient_bootstraps"]),
        "alpha":float(rank_cfg["alpha"]),
    }

    # Report this sensor/measurement drift independently of any rank outcome.
    mid=m.shape[1]//2
    coverage_early=m[:,:mid].mean(axis=(0,1))
    coverage_late=m[:,mid:].mean(axis=(0,1))
    mean_early=(x[:,:mid]*m[:,:mid]).sum((0,1))/np.maximum(m[:,:mid].sum((0,1)),1.)
    mean_late=(x[:,mid:]*m[:,mid:]).sum((0,1))/np.maximum(m[:,mid:].sum((0,1)),1.)
    rms=np.sqrt((x*x*m).sum((0,1))/np.maximum(m.sum((0,1)),1.))
    standardized_mean_drift=(mean_late-mean_early)/np.maximum(rms,.05)
    drift={
        "fraction_channels_with_abs_standardized_mean_drift_above_0_25":
            float((np.abs(standardized_mean_drift)>.25).mean()),
        "max_abs_standardized_mean_drift":float(np.max(np.abs(standardized_mean_drift))),
        "max_abs_observation_coverage_change":
            float(np.max(np.abs(coverage_late-coverage_early))),
    }

    n=len(x)
    shuffled=np.random.default_rng(int(rank_cfg["subsample_seed"])).permutation(n)
    half_indices=[shuffled[:n//2],shuffled[n//2:]]
    scales={}
    for hour_bin in (1,4,6):
        xx,mm=aggregate_hours(x,m,hour_bin)
        selection=support_selected_channels(
            mm,names,lags=rank_args["lags"],min_pairs=20,
        )
        record={
            "hour_bin":hour_bin,
            "hours_per_aggregated_observation":hour_bin,
            "n_time_bins":xx.shape[1],
            "train_only_measurement_support":selection,
            "validity_scope":"selected observable clinical signals only",
            "no_medical_ontology_claim":True,
        }
        keep=selection["selected_indices"]
        if len(keep)<2:
            record["status"]="NOT_IDENTIFIABLE_FROM_PAIRWISE_COVERAGE"
            record["diagnostic"]="Fewer than two channels have jointly observed time pairs"
            scales[str(hour_bin)]=record
            continue
        xs=xx[:,:,keep]
        ms=mm[:,:,keep]
        try:
            all_train=fit_lagged_predictive_rank(
                xs,ms,seed=int(rank_cfg["seed"])+hour_bin,**rank_args
            )
        except ValueError as exc:
            record["status"]="NOT_IDENTIFIABLE_FROM_MEASUREMENT_COVERAGE"
            record["diagnostic"]=str(exc)
            scales[str(hour_bin)]=record
            continue
        record["full_train"]=all_train.to_json()
        record["status"]="TRAIN_OBSERVABLE_RANK_ESTIMATED_NOT_CLINICALLY_VALIDATED"
        record["signed_observable_component_definitions"]=[]
        for concept in range(all_train.rank):
            loading=all_train.left_observable_loadings[:,concept]
            largest=np.argsort(-np.abs(loading))[:min(5,len(loading))]
            record["signed_observable_component_definitions"].append({
                "component":int(concept),
                "measurable_channel_coefficients":[
                    {"channel":selection["selected_channels"][j],
                     "signed_loading":float(loading[j])}
                    for j in largest
                ],
                "expert_clinical_name_provided":False,
            })
        halves=[]
        for i,idx in enumerate(half_indices):
            half_mask=ms[idx]
            joint=joint_pair_counts(half_mask,rank_args["lags"])
            if np.any(joint<20):
                halves.append({
                    "status":"NOT_IDENTIFIABLE_FROM_HALF_SAMPLE_COVERAGE",
                    "min_half_sample_joint_pairs":float(joint.min()),
                    "rank":None,
                })
                continue
            try:
                half=fit_lagged_predictive_rank(
                    xs[idx],half_mask,
                    seed=int(rank_cfg["seed"])+hour_bin*100+i+11,
                    **rank_args,
                )
            except ValueError as exc:
                halves.append({"status":"NOT_IDENTIFIABLE","reason":str(exc),"rank":None})
                continue
            halves.append({"status":"RANK_ESTIMATED",
                           "rank":int(half.rank),"estimated":half.to_json()})
        record["disjoint_train_halves"]=halves
        if len(halves)==2 and all(v["status"]=="RANK_ESTIMATED" for v in halves):
            record["half_train_subspace_overlap"]=_basis_overlap(
                np.asarray(halves[0]["estimated"]["left_observable_loadings"]),
                np.asarray(halves[1]["estimated"]["left_observable_loadings"]),
                halves[0]["rank"],halves[1]["rank"],
            )
        else:
            record["half_train_subspace_overlap"]={
                "status":"UNAVAILABLE","reason":"One or both halves insufficiently observed",
            }
        # Additional strict sensitivity: select OBSERVABLE CHANNELS shared
        # by both train halves before computing either spectral basis. The
        # masks influence feasibility, not K by reference to outcomes.
        strict=support_selected_channels_across_subsamples(
            (mm[half_indices[0]],mm[half_indices[1]]),
            names,lags=rank_args["lags"],min_pairs=20,
        )
        strict_report={"selection":strict}
        if strict["all_subsamples_joints_adequate"]:
            retained=strict["selected_indices"]
            halves_with_common_support=[]
            for i,idx in enumerate(half_indices):
                try:
                    rr=fit_lagged_predictive_rank(
                        xx[idx][:,:,retained],mm[idx][:,:,retained],
                        seed=int(rank_cfg["seed"])+hour_bin*101+i+503,
                        **rank_args,
                    )
                    halves_with_common_support.append({
                        "status":"RANK_ESTIMATED","rank":rr.rank,
                        "left_predictive_basis":rr.left_observable_loadings.tolist(),
                        "bootstrap_interval":list(rr.patient_bootstrap_interval),
                    })
                except ValueError as exc:
                    halves_with_common_support.append({
                        "status":"NOT_IDENTIFIABLE","reason":str(exc),"rank":None,
                    })
            strict_report["half_sample_results"]=halves_with_common_support
            if all(v["status"]=="RANK_ESTIMATED" for v in halves_with_common_support):
                ha,hb=halves_with_common_support
                strict_report["half_sample_subspace_overlap"]=_basis_overlap(
                    np.asarray(ha["left_predictive_basis"]),
                    np.asarray(hb["left_predictive_basis"]),
                    ha["rank"],hb["rank"],
                )
        else:
            strict_report["status"]="INSUFFICIENT_SENSORS_WITH_JOINT_TEMPORAL_SUPPORT"
        record["separate_train_subsample_robustness_analysis"]=strict_report
        scales[str(hour_bin)]=record
    success=[k for k,v in scales.items() if v["status"]=="TRAIN_OBSERVABLE_RANK_ESTIMATED_NOT_CLINICALLY_VALIDATED"]
    report={
        "status":("MEASURABLE_OBSERVABLE_RANK_DIAGNOSTIC_COMPLETE" if success
                  else "NO_IDENTIFIABLE_RANK_WITH_AVAILABLE_OBSERVATION_COVERAGE"),
        "official_train_data_receipt":data.receipt,
        "train_patient_count":len(x),
        "initial_train_observable_channels":len(names),
        "aggregation_sensitivity_by_hours":scales,
        "eligible_scales":[int(v) for v in success],
        "early_late_train_temporal_drift":drift,
        "no_individual_record_ids_or_predictions_published":True,
        "test_or_calibration_used":False,
        "clinical_concept_validity_inferred":False,
        "limitations":[
            "A clinical frequency-based channel subset is disclosed separately for each aggregation time scale",
            "Any observed rank may reflect patient mixture, nonstationarity, treatment or observation patterns",
            "Masks do not correct informative or missing-not-at-random sampling",
            "A change from 1h to 4h or 6h aggregation changes temporal resolution and estimated construct",
            "The previously inspected historical test split is not prospectively clean",
        ],
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({
        "clinical_concept_validity":False,
        "observable_ranks":{
            k:(v.get("full_train",{}).get("rank")
               if v["status"]=="TRAIN_OBSERVABLE_RANK_ESTIMATED_NOT_CLINICALLY_VALIDATED"
               else None)
            for k,v in scales.items()
        },
        "selected_channel_counts":{
            k:v["train_only_measurement_support"]["channels_selected"] for k,v in scales.items()
        },
        "support_statuses":{k:v["status"] for k,v in scales.items()},
        "measurement_drift":drift,
    },indent=2),flush=True)


if __name__=="__main__":
    main()
