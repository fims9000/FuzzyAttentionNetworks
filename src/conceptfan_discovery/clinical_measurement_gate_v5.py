"""Predeclared train-pilot sensor selection and independent diagnostic inference.

This protocol *deliberately* refuses to assign clinical names. The clinical
predictive rank only becomes a candidate if all pre-registered measurement
settings and distinct patient inference halves support the same dynamic
dimension, measured on a frozen pilot-selected sensor dictionary.

Unlike v3 clinical audit, the exact temporal sensor subset is selected on
the first disjoint 600 TRAIN patients and frozen BEFORE any spectral rank
is computed on remaining 1800 TRAIN patients. Test, calibration and
validation patients are never evaluated. This avoids selecting a good
number of factors by changing channels after seeing rank results.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from .clinical_observability import aggregate_hours
from .measurement_invariance_experiment_v5 import estimate
from .observation_regimes_v5 import (
    apply_measurement_environment, evaluate_rank_invariance,
)
from .physionet import prepare_physionet_discovery
from .pilot_temporal_sensors_v5 import choose_common_temporal_sensors


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True,type=Path)
    p.add_argument("--raw-zip",required=True,type=Path)
    p.add_argument("--outcomes",required=True,type=Path)
    p.add_argument("--out-dir",required=True,type=Path)
    args=p.parse_args()
    if args.out_dir.exists():
        p.error("Cannot overwrite a prior rank report")
    cfg=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not cfg["protocol"]["train_only"] or not cfg["protocol"]["no_test_calibration_or_validation"]:
        p.error("Official held-out medical records must remain sealed")
    temporal_cfg=cfg["measurement"]
    if temporal_cfg["admissible_policies"] != ["original","mcar_thin"]:
        p.error("The 65%-retention pilot guard applies ONLY to original/mcar_thin")
    if temporal_cfg["diagnostic_policy"]!="time_channel_mar":
        p.error("Unannounced observation policy change")
    if set(temporal_cfg["aggregation_hours"])!={1,4,6}:
        p.error("All 1h,4h,6h aggregations must be included")
    with tempfile.TemporaryDirectory(prefix="conceptfan_clinical_pilot_") as tmp:
        data=prepare_physionet_discovery(
            args.raw_zip,args.outcomes,Path(tmp),
            split_seed=int(cfg["data"]["split_seed"]),
            input_window=int(cfg["data"]["input_window"]),
            forecast_horizon=int(cfg["data"]["forecast_horizon"]),
            minimum_train_patient_coverage=float(cfg["data"]["minimum_train_patient_coverage"]),
        )
    x=data.train.x
    mask=data.train.mask
    if len(x)!=2400:
        raise RuntimeError("Historical train split changed")
    perm=np.random.default_rng(int(temporal_cfg["patient_split_seed"])).permutation(len(x))
    pilot_indices=perm[:int(temporal_cfg["pilot_patients"])]
    other=perm[len(pilot_indices):]
    first,second=np.array_split(other,2)
    if not (len(pilot_indices)==600 and len(first)==900 and len(second)==900):
        raise ValueError("Nonoverlapping 600 pilot + 900/900 inference protocol required")
    if len(set(pilot_indices)&set(other)) or len(set(first)&set(second)):
        raise AssertionError("Patient-level leakage between mask pilot and rank inference")

    names=data.receipt["channel_names"]
    # Only pilot observation MASKS decide the (potentially small) channel
    # set. The normalizing moments already come from TRAIN only; no
    # label, latent concept oracle or validation result selects channels.
    pilot=choose_common_temporal_sensors(
        mask[pilot_indices],names,
        lags=tuple(cfg["rank"]["lags"]),
        minimum_supported_strata=int(temporal_cfg["minimum_pilot_hour_strata"]),
        minimum_observed_pairs_per_stratum=int(temporal_cfg["pilot_min_pairs_per_hour"]),
        minimum_expected_null_pairs_per_stratum=float(
            temporal_cfg["pilot_min_expected_null_pairs"]
        ),
        worst_additional_observation_retention=float(
            temporal_cfg["admissible_worst_additional_retention"]
        ),
    )
    args.out_dir.mkdir(parents=True,exist_ok=False)
    receipt={
        "status":"RANK_AUDIT_COMPLETE" if pilot["status"]=="PILOT_SENSORS_ELIGIBLE"
                 else "ABSTAIN_INSUFFICIENT_PILOT_TEMPORAL_SUPPORT",
        "train_data_receipt":data.receipt,
        "independent_train_patient_count":{
            "selection_pilot":len(pilot_indices),
            "rank_inference_first":len(first),
            "rank_inference_second":len(second),
        },
        "selection_uses_only_pilot_masks":True,
        "test_validation_calibration_ever_used":False,
        "pilot_sensor_selection":pilot,
        "predeclared_research_scope":"observable dynamic rank only, not medically valid concept identities",
        "not_clinical_semantics":True,
    }
    records=[]
    checks=[]
    if pilot["status"]=="PILOT_SENSORS_ELIGIBLE":
        picked=np.asarray(pilot["selected_indices"],dtype=int)
        source_indices=np.concatenate((first,second))
        source_x=x[source_indices][:,:,picked]
        source_m=mask[source_indices][:,:,picked]
        partitions=(np.arange(len(source_x)),
                    np.arange(len(first)),
                    np.arange(len(first),len(source_x)))
        for hours in temporal_cfg["aggregation_hours"]:
            hours=int(hours)
            xa,ma=aggregate_hours(source_x,source_m,hours)
            by_method={"pooled_v3":{}, "equal_time_v5":{}}
            conditions=(*temporal_cfg["admissible_policies"],
                        temporal_cfg["diagnostic_policy"])
            for env_number,policy in enumerate(conditions):
                ec,em,envrec=apply_measurement_environment(
                    xa,ma,environment=policy,
                    seed=int(temporal_cfg["mask_seed"])+hours*31+env_number
                )
                for method_number,method in enumerate(by_method):
                    output_list=[]
                    for part_number,index in enumerate(partitions):
                        try:
                            estimated,rec=estimate(
                                method,ec[index],em[index],
                                options=cfg["rank"],
                                seed=int(cfg["rank"]["null_seed"])+hours*251+
                                     env_number*17+method_number*47+part_number
                            )
                            reason=""
                        except ValueError as exc:
                            estimated=None
                            rec={"rank":None}
                            reason=str(exc)
                        output_list.append(estimated)
                        records.append({
                            "hours_per_time_bin":hours,
                            "environment":policy,
                            "estimator":method,
                            "partition":("inference_train_1800","inference_half_A_900",
                                         "inference_half_B_900")[part_number],
                            "n_patients":len(index),
                            "fixed_sensor_count":len(picked),
                            "detected_k":rec.get("rank"),
                            "status":"RANK_ESTIMATED" if estimated is not None else "ABSTAIN",
                            "reason":reason,
                            "mask_fraction":float(em[index].mean()),
                            "no_patient_or_outcome_labels_used":True,
                        })
                    by_method[method][policy]={
                        "whole":output_list[0],"halves":tuple(output_list[1:]),
                    }
            for method,assessments in by_method.items():
                gate=evaluate_rank_invariance(
                    {name:assessments[name] for name in temporal_cfg["admissible_policies"]},
                    method=method,
                    min_squared_principal_cosine=float(
                        temporal_cfg["min_principal_squared_overlap"]
                    )
                )
                checks.append({
                    "hours_per_time_bin":hours,
                    "estimator":method,
                    "gate_status":gate.status,
                    "stable_k_if_any":gate.consensus_k,
                    "reasons":"|".join(gate.reasons),
                    "no_medical_concept_claim":True,
                })
                receipt[f"hour_{hours}_method_{method}"]={
                    "prespecified_admissible_measurement_gate":gate.json(),
                    "diagnostic_time_channel_mar_rank":
                        None if assessments[temporal_cfg["diagnostic_policy"]]["whole"] is None
                        else int(assessments[temporal_cfg["diagnostic_policy"]]["whole"].rank),
                }
    summary=pd.DataFrame(checks)
    pd.DataFrame(records).to_csv(args.out_dir/"clinical_observation_invariance_parts.csv",index=False)
    summary.to_csv(args.out_dir/"clinical_observation_invariance_gates.csv",index=False)
    if len(summary):
        for method,by_method in summary.groupby("estimator"):
            passed=by_method.loc[
                by_method.gate_status=="MEASUREMENT_INVARIANT_OBSERVABLE_RANK",
                "stable_k_if_any",
            ].dropna()
            all_scales_consistent=(
                len(passed)==len(temporal_cfg["aggregation_hours"])
                and passed.nunique()==1
            )
            receipt[f"clinical_multi_resolution_{method}"]={
                "status":("STRUCTURALLY_STABLE_OBSERVABLE_RANK_ONLY"
                          if all_scales_consistent else "ABSTAIN_CLINICAL_K_UNSTABLE"),
                "consensus_k":int(passed.iloc[0]) if all_scales_consistent else None,
                "n_passed_scales":len(passed),
                "medical_semantics_verified":False,
                "interpretation":"even a passed gate would NOT justify physiological labels",
            }
    receipt["limitations"]=[
        "the pilot channel set is selected on mask availability, not medical meaning",
        "synthetic thinning of clinical observations is not independent hospital data",
        "time-conditional MAR is assumed, not confirmed for clinical MNAR",
        "temporal binning changes the observed estimand",
        "a reliable clinical concept requires independent medical annotations or invariant environments",
        "previously inspected test patients remain sealed in this new research",
    ]
    (args.out_dir/"clinical_measurement_status.json").write_text(
        json.dumps(receipt,indent=2,allow_nan=False),encoding="utf-8"
    )
    print(json.dumps({
        "pilot_selected_n_channels":pilot["selected_indices"].__len__(),
        "gate_results":checks,
        "clinical_rank_verified":False,
    },indent=2,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
