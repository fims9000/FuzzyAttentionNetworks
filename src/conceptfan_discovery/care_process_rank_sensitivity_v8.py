"""Test rank robustness of observation-process concepts before clinical claims.

This is separate from the six fitted V8 classifiers, because the signed
operator dimension is an UNSUPERVISED property of train measurement masks.
A one-off K estimate from all patients is NOT independent evidence that K
is a stable number of healthcare observation mechanisms.

A predeclared 600-patient mask-only pilot fixes the same sensor dictionary
as V8. The remaining TRAIN masks are evaluated as two disjoint 900-patient
subsamples and pooled 1800-patient cohort. Full 2400 TRAIN estimate is also
reported as a sensitivity control, but it overlaps the pilot and is NOT
used as an independent validation.

Across original, additional MCAR and time-channel-specific MAR we test:
- full and disjoint-half K agreement,
- principal-subspace overlap of the actually learned signed mask operators,
- rank sensitivity to observation policy.

If not stable, return ABSTAIN_PROCESS_K_UNIDENTIFIABLE; do not imply the
full-train source K or the model's practical capacity equals real
care pathways or physician concepts. No validation/calibration/test used.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .care_process_rank_v8 import fit_care_policy_rank
from .observation_regimes_v5 import (
    apply_measurement_environment,principal_subspace_overlap,
)
from .physionet import prepare_physionet_discovery
from .pilot_temporal_sensors_v5 import choose_common_temporal_sensors


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--reference-v6-config",type=Path,required=True)
    p.add_argument("--raw-zip",type=Path,required=True)
    p.add_argument("--outcomes",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    if args.out_dir.exists():
        p.error("Do not overwrite a previously completed mask-rank sensitivity audit")
    cfg=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source=yaml.safe_load(args.reference_v6_config.read_text(encoding="utf-8"))
    if not cfg["protocol"]["historical_test_and_calibration_sealed"]:
        p.error("Requires historical test/calibration seal")
    with tempfile.TemporaryDirectory(prefix="pd_care_rank_mnar_sensitivity_") as tmp:
        data=prepare_physionet_discovery(
            args.raw_zip,args.outcomes,Path(tmp),
            split_seed=int(source["data"]["split_seed"]),
            input_window=int(source["data"]["input_window"]),
            forecast_horizon=int(source["data"]["forecast_horizon"]),
            minimum_train_patient_coverage=float(source["data"]["minimum_train_patient_coverage"]),
        )
    masks=data.train.mask
    n=len(masks)
    perm=np.random.default_rng(int(source["data"]["pilot_selection_seed"])).permutation(n)
    pilot=perm[:int(source["data"]["pilot_train_patients"])]
    inference=perm[len(pilot):]
    halves=np.array_split(inference,2)
    channels=choose_common_temporal_sensors(
        masks[pilot],data.receipt["channel_names"],
        lags=tuple(source["data"]["pilot_sensor_lags"]),
        minimum_supported_strata=int(source["data"]["pilot_min_time_strata"]),
        minimum_observed_pairs_per_stratum=int(source["data"]["pilot_min_hourly_pairs"]),
        minimum_expected_null_pairs_per_stratum=float(source["data"]["pilot_min_expected_null_pairs"]),
        worst_additional_observation_retention=float(source["data"]["pilot_worst_extra_thinning"]),
    )
    if channels["status"]!="PILOT_SENSORS_ELIGIBLE":
        raise ValueError("No adequate independent mask-only source dictionary")
    kept=np.asarray(channels["selected_indices"],dtype=int)
    if len(halves[0])!=900 or len(halves[1])!=900:
        raise ValueError("Expected independent 900+900 train patient groups")
    r=cfg["care_policy_rank"]
    method_args={
        "permutations":int(r["feature_permutations"]),
        "alpha":float(r["familywise_alpha"]),
        "max_slots":int(r["max_computational_policy_slots"]),
    }
    result_rows=[]
    source_details={}
    environment_basis={}
    for environment_index,env in enumerate(("original","mcar_thin","time_channel_mar")):
        _,policy_masks,policy_receipt=apply_measurement_environment(
            data.train.x[:,:,kept],masks[:,:,kept],
            environment=env,seed=int(r["null_seed"])+303+environment_index,
        )
        trials={}
        for i,(name,index) in enumerate((
            ("all_train_2400",np.arange(n)),
            ("disjoint_inference_1800",inference),
            ("inference_half_A_900",halves[0]),
            ("inference_half_B_900",halves[1]),
        )):
            try:
                rank=fit_care_policy_rank(
                    policy_masks[index],
                    seed=int(r["null_seed"])+101*environment_index+17*i,
                    **method_args,
                )
                record={
                    "status":"MASK_PROCESS_RANK_ESTIMATED",
                    "rank_uncapped":rank.detected_uncapped_k,
                    "computational_k":rank.chosen_k,
                    "permutation_threshold":rank.null_max_eigenvalue_cutoff,
                    "full_eigen_spectrum":rank.raw_eigenvalues.tolist(),
                    "source_signed_operator_basis":rank.signed_operator_basis.tolist(),
                }
                trials[name]=rank
            except ValueError as exc:
                record={
                    "status":"ABSTAIN_MASK_PROCESS_NOT_IDENTIFIABLE",
                    "reason":str(exc),
                    "rank_uncapped":None,"computational_k":None,
                }
                trials[name]=None
            result_rows.append({
                "environment":env,
                "patient_subset":name,
                "subset_patient_n":len(index),
                "jointly_supported_channel_count":len(kept),
                "observation_policy_retention_fraction":float(policy_receipt["relative_retained_observations"]),
                "uncapped_mask_process_k":record["rank_uncapped"],
                "practical_computational_k":record["computational_k"],
                "status":record["status"],
            })
            source_details[f"{env}_{name}"]=record
        a=trials["inference_half_A_900"]
        b=trials["inference_half_B_900"]
        pooled=trials["disjoint_inference_1800"]
        if a is None or b is None or pooled is None:
            overlap=None
            stable=False
        else:
            overlap=principal_subspace_overlap(
                a.signed_operator_basis,
                b.signed_operator_basis,
                a.chosen_k,b.chosen_k,
            )
            stable=bool(
                a.detected_uncapped_k==b.detected_uncapped_k==pooled.detected_uncapped_k
                and overlap is not None and overlap>=.75
            )
        environment_basis[env]={
            "inference_1800_uncapped_k":pooled.detected_uncapped_k if pooled is not None else None,
            "independent_half_900_ks":[
                a.detected_uncapped_k if a is not None else None,
                b.detected_uncapped_k if b is not None else None,
            ],
            "mean_squared_principal_subspace_overlap":overlap,
            "rank_accepted_within_this_environment":stable,
            "interpretation":"correlated observation-operator directions, not health care causality",
        }
    aggregate_ranks=[
        record["inference_1800_uncapped_k"]
        for record in environment_basis.values()
    ]
    stable_all=(
        all(x["rank_accepted_within_this_environment"] for x in environment_basis.values())
        and len(set(aggregate_ranks))==1
    )
    args.out_dir.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(result_rows).to_csv(
        args.out_dir/"care_policy_rank_sensitivity_by_train_subsample.csv",index=False
    )
    evidence={
        "method":"label-free patient-shuffled marginal null eigenvalue rank of train-only mask operator correlations",
        "status":"ACCEPT_PROCESS_OBSERVABLE_RANK_ONLY" if stable_all
                 else "ABSTAIN_PROCESS_K_UNIDENTIFIABLE",
        "strict_all_observation_environments_and_halves_agree":stable_all,
        "k_if_and_only_if_rank_stable":int(aggregate_ranks[0]) if stable_all else None,
        "measurement_value_and_outcome_data_used_for_rank":False,
        "label_and_latent_concept_supervision":False,
        "patient_disjoint_mask_only_pilot_600":True,
        "independent_rank_evaluation_halves_900_900":True,
        "all_2400_train_rank_control_overlaps_pilot":True,
        "test_calibration_validation_used":False,
        "pilot_sensor_selection":channels,
        "environment_results":environment_basis,
        "detailed_rank_spectra":source_details,
        "limitations":[
            "mask process rank is not the actual number of clinician workflows",
            "synthetic MCAR/MAR thinning uses the same patients, not external sites",
            "the chosen sensor dictionary was selected on an independent 600-patient mask pilot",
            "reference V8 logistic head and original clinical validation cannot certify unique causal concept identity",
            "an accepted observable subspace dimension still does not identify the unique semantic axes",
        ],
    }
    (args.out_dir/"STATUS.json").write_text(
        json.dumps(evidence,indent=2,allow_nan=False),encoding="utf-8"
    )
    print(json.dumps({
        "status":evidence["status"],
        "source":len(kept),
        "environments":environment_basis,
    },indent=2,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
