"""V9 real clinical TRAIN-only observation process COMMON SUBSPACE audit.

V8 measured 9 correlated mask directions on pooled 1800 TRAIN patients,
8 and 7 on independent halves, and 6/6 after temporal thinning. Treat
these as different finite-sample detectable subspaces, NOT proof of
9 or 6 unique clinician workflows.

Fit six frozen, disjoint TRAIN-half observation-mask subspaces. Use only
their PROJECTORS to build a cross-policy stable core under a strict,
prespecified threshold. No patient labels, physiological values, manual
concept names, validation/test or outcomes are used in determining K.
The 600-patient pilot still selects common measurable channels.

If no stable direction, report ABSTAIN rather than using the most
convenient full-cohort rank. A nonzero subspace is still not proof of
clinically named concepts or cross-hospital invariance.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import yaml

from .care_process_rank_v8 import fit_care_policy_rank
from .observation_regimes_v5 import apply_measurement_environment
from .physionet import prepare_physionet_discovery
from .pilot_temporal_sensors_v5 import choose_common_temporal_sensors
from .stable_care_subspace_v9 import find_common_mask_subspace


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--reference-v6-config",type=Path,required=True)
    p.add_argument("--raw-zip",type=Path,required=True)
    p.add_argument("--outcomes",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    if args.out_dir.exists():
        p.error("Refusing to overwrite earlier evidence")
    cfg=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source=yaml.safe_load(args.reference_v6_config.read_text(encoding="utf-8"))
    protocol=cfg["protocol"]
    if not protocol["old_test_calibration_validation_sealed"] or (
        protocol["rank_not_medical_ontology"] is not True
    ):
        p.error("This research cannot assert clinical latent concept identities")
    policy=tuple(cfg["evaluation"]["measurement_policies"])
    if policy!=("original","mcar_thin","time_channel_mar"):
        p.error("Unannounced or missing observation policy stress test")
    with tempfile.TemporaryDirectory(prefix="v9_mask_core_source_") as tmp:
        data=prepare_physionet_discovery(
            args.raw_zip,args.outcomes,Path(tmp),
            split_seed=int(source["data"]["split_seed"]),
            input_window=int(source["data"]["input_window"]),
            forecast_horizon=int(source["data"]["forecast_horizon"]),
            minimum_train_patient_coverage=float(
                source["data"]["minimum_train_patient_coverage"]
            ),
        )
    x=data.train.x
    m=data.train.mask
    if len(x)!=2400:
        raise RuntimeError("The frozen historical train split changed")
    split=np.random.default_rng(
        int(source["data"]["pilot_selection_seed"])
    ).permutation(len(m))
    pilot=split[:int(source["data"]["pilot_train_patients"])]
    inference=split[len(pilot):]
    halves=np.array_split(inference,2)
    if len(pilot)!=600 or len(halves[0])!=900 or len(halves[1])!=900:
        raise ValueError("Require disjoint train-only 600+900+900 cohort")
    check=choose_common_temporal_sensors(
        m[pilot],data.receipt["channel_names"],
        lags=tuple(source["data"]["pilot_sensor_lags"]),
        minimum_supported_strata=int(source["data"]["pilot_min_time_strata"]),
        minimum_observed_pairs_per_stratum=int(source["data"]["pilot_min_hourly_pairs"]),
        minimum_expected_null_pairs_per_stratum=float(
            source["data"]["pilot_min_expected_null_pairs"]
        ),
        worst_additional_observation_retention=float(
            source["data"]["pilot_worst_extra_thinning"]
        ),
    )
    if check["status"]!="PILOT_SENSORS_ELIGIBLE":
        raise RuntimeError("Pilot masks have no jointly observable source dictionary")
    indices=np.asarray(check["selected_indices"],dtype=int)
    params=cfg["estimator"]
    sources={}
    per_source={}
    for policy_i,environment in enumerate(policy):
        _,altered,receipt=apply_measurement_environment(
            x[:,:,indices],m[:,:,indices],
            environment=environment,
            seed=int(params["measurement_seed"])+policy_i,
        )
        for half_i,patients in enumerate(halves):
            source_id=f"{environment}_half_{half_i+1}"
            try:
                result=fit_care_policy_rank(
                    altered[patients],
                    permutations=int(params["patient_permutations"]),
                    seed=int(params["null_seed"])+101*policy_i+half_i,
                    alpha=float(params["alpha"]),
                    max_slots=int(params["max_source_k"]),
                )
                if result.truncated_by_budget:
                    raise ValueError("Rank exceeds frozen source capacity")
                sources[source_id]=result.signed_operator_basis
                per_source[source_id]={
                    "status":"MASK_SUBSPACE_ESTIMABLE",
                    "patient_count":len(patients),
                    "k_uncapped":int(result.detected_uncapped_k),
                    "operator_dimension":len(result.source_operator_center),
                    "source_explanation":"patient-specific observed-mask operators only",
                    "additional_measurement_retention":float(
                        receipt["relative_retained_observations"]
                    ),
                    "source_rank_eigenspectrum":result.raw_eigenvalues.tolist(),
                    "null_threshold":float(result.null_max_eigenvalue_cutoff),
                }
            except ValueError as exc:
                per_source[source_id]={
                    "status":"ABSTAIN_SOURCE_MASK_SUPPORT",
                    "reason":str(exc),
                }
            print(source_id,per_source[source_id]["status"],
                  per_source[source_id].get("k_uncapped"),flush=True)
    if len(sources)==2*len(policy):
        common=find_common_mask_subspace(
            sources,
            projector_threshold=float(params["projector_threshold"]),
            min_source_energy=float(params["min_source_energy"]),
            sensitivity_levels=tuple(float(v) for v in params["threshold_sensitivity"]),
        )
        core=common.receipt()
        if core["k_observable_common_core"]>=1:
            names=[
                f"{operator}|{channel}"
                for operator in (
                    "observation_fraction","recent_observation_fraction",
                    "normalized_time_since_last_observation",
                    "sampling_transition_fraction"
                ) for channel in check["selected_names"]
            ]
            core["canonical_source_operator_pivot_names"]=[
                names[index] for index in core["canonical_named_operator_pivots"]
            ]
            core["basis_identified_up_to_model_coordinate_convention"]=True
            core["exact_number_of_medical_concepts_claimed"]=False
    else:
        core={
            "status":"ABSTAIN_SOURCE_MASK_SUPPORT",
            "k_observable_common_core":None,
            "failed_sources":[k for k in per_source if k not in sources],
            "claim_limit":"No reproducible shared observed source subspace",
        }
    args.out_dir.mkdir(parents=True,exist_ok=False)
    status={
        "protocol":protocol["name"],
        "status":core["status"],
        "source_v8_negative_control":"https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41512934565",
        "source_patients":{"pilot_600":len(pilot),
                           "independent_halves_900_each":[len(h) for h in halves]},
        "n_pilot_selected_measurement_channels":len(indices),
        "pilot_sensor_provenance":check,
        "source_diagnostics":per_source,
        "cross_environment_common_core":core,
        "fit_on_original_train_masks_only":True,
        "patient_outcomes_not_used_for_selection":True,
        "validation_calibration_test_not_used":True,
        "same_patient_cohorts_appear_under_simulated_observation_policies":True,
        "clinical_semantic_identity_or_causal_mechanism_proved":False,
        "scientific_limitations":[
            "a common subspace does not resolve rotations to uniquely correct clinical semantic axes",
            "the selected named operator axes depend on a prespecified feature dictionary and pivot convention",
            "simulated observation changes are not independent hospitals",
            "missing-not-at-random sampling remains confounded",
            "nonzero common-core dimension is only a strict, protocol-dependent lower bound of observable mask dynamics",
            "lack of common directions entails abstention, not proof that true clinical factors do not exist",
        ],
    }
    (args.out_dir/"STATUS.json").write_text(
        json.dumps(status,indent=2,allow_nan=False),encoding="utf-8",
    )
    print(json.dumps({
        "status":status["status"],
        "common_core_k":core["k_observable_common_core"],
        "source_ks":{n:r.get("k_uncapped") for n,r in per_source.items()},
        "threshold_sensitivity":core.get("threshold_sensitivity_counts"),
        "named_operator_pivots":core.get("canonical_source_operator_pivot_names",[]),
    },indent=2,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
