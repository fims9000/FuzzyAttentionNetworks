"""V9 TRAIN-TIME care concepts from the robust disjoint-source common core.

IMPORTANT: V8 observed 9, 8, 7, 6 mask-dynamic directions depending on
sample/policy. V9 independently found a common 4D mask-operator subspace
across six TRAIN-half/policy views, using projector rotation invariance,
without outcome labels, clinical definitions or outcome-guided K.

This method trains those 4 measurable care-process factors AS concepts
inside the classifier rather than inventing a posthoc explanation.
Physiological measured-evidence concepts remain unknown-neutral.
All parameters contributing to the exact additive logit are inspectable.

V9 is *not* an independently clinically proven four-cause system; and
the previously reused validation is not a prospective test cohort.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .anchored_care_process_v8 import (
    MaskRankAnchoredDualConceptFAN,fit_train_only_care_task_prior,
)
from .clinical_anchored_care_experiment_v8 import fit_one
from .clinical_dual_process_experiment_v7 import evaluate_dual
from .clinical_unknown_neutral_experiment_v6 import subset_discovery_split
from .data import DiscoveryData
from .dual_process_stability_v7 import collect_observation_process_evidence
from .evaluation import compare_independent_discoveries,evaluate_discovery,concept_passports
from .lagged_rank import fit_lagged_predictive_rank
from .physionet import prepare_physionet_discovery
from .pilot_temporal_sensors_v5 import choose_common_temporal_sensors
from .selection import split_validation_for_cardinality
from .stable_care_subspace_v9 import StableCareAnchor


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--common-core-status",type=Path,required=True)
    parser.add_argument("--reference-v6-config",type=Path,required=True)
    parser.add_argument("--reference-v8-config",type=Path,required=True)
    parser.add_argument("--raw-zip",type=Path,required=True)
    parser.add_argument("--outcomes",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args()
    if args.out_dir.exists():
        parser.error("Refusing to overwrite any historical experiment")
    v9=json.loads(args.common_core_status.read_text(encoding="utf-8"))
    core=StableCareAnchor.from_v9_status(v9)
    if v9["status"]!="OBSERVATION_COMMON_SUBSPACE_ONLY":
        parser.error("K is unsupported; never force care concept slots")
    v6=yaml.safe_load(args.reference_v6_config.read_text(encoding="utf-8"))
    v8=yaml.safe_load(args.reference_v8_config.read_text(encoding="utf-8"))
    if (not v6["protocol"]["test_and_calibration_sealed"]
        or not v8["protocol"]["historical_test_and_calibration_sealed"]):
        parser.error("Test/calibration prohibition missing")
    methods=tuple(v8["training"]["methods"])
    if set(methods)!={"train_mask_source_only","train_mask_source_and_task_prior"}:
        parser.error("Both source-only and task-head-prior ablations required")

    with tempfile.TemporaryDirectory(prefix="v9_frozen_clinical_split_") as tmp:
        original=prepare_physionet_discovery(
            args.raw_zip,args.outcomes,Path(tmp),
            split_seed=int(v6["data"]["split_seed"]),
            input_window=int(v6["data"]["input_window"]),
            forecast_horizon=int(v6["data"]["forecast_horizon"]),
            minimum_train_patient_coverage=float(
                v6["data"]["minimum_train_patient_coverage"]
            ),
        )
    x=original.train.x
    mask=original.train.mask
    if len(x)!=2400 or len(original.validation.y)!=600:
        raise ValueError("Historical patient-level partition changed")
    order=np.random.default_rng(
        int(v6["data"]["pilot_selection_seed"])
    ).permutation(len(x))
    pilot=order[:int(v6["data"]["pilot_train_patients"])]
    independent_train=order[len(pilot):]
    feature=choose_common_temporal_sensors(
        mask[pilot],original.receipt["channel_names"],
        lags=tuple(v6["data"]["pilot_sensor_lags"]),
        minimum_supported_strata=int(v6["data"]["pilot_min_time_strata"]),
        minimum_observed_pairs_per_stratum=int(v6["data"]["pilot_min_hourly_pairs"]),
        minimum_expected_null_pairs_per_stratum=float(
            v6["data"]["pilot_min_expected_null_pairs"]),
        worst_additional_observation_retention=float(
            v6["data"]["pilot_worst_extra_thinning"]),
    )
    if feature["selected_names"]!=v9["pilot_sensor_provenance"]["selected_names"]:
        raise RuntimeError("V9 trained common-core source dictionary differs from its pinned TRAIN-only mask pilot")
    kept=np.asarray(feature["selected_indices"],dtype=int)
    if len(kept)!=core.n_observed_sensor_channels:
        raise RuntimeError("Source and evaluation sensor coordinates have different dimensions")
    r=v6["rank"]
    dynamic=fit_lagged_predictive_rank(
        x[independent_train][:,:,kept],mask[independent_train][:,:,kept],
        lags=tuple(r["lags"]),
        permutations=int(r["permutations"]),
        patient_bootstraps=int(r["patient_bootstraps"]),
        alpha=float(r["alpha"]),seed=int(r["seed"]),
    )
    physical_budget=min(int(dynamic.rank),int(r["max_computational_concept_slots"]))
    if physical_budget<2:
        raise RuntimeError("Cannot initialize nontrivial physical observed-evidence basis")
    observed_basis=dynamic.left_observable_loadings[:,:physical_budget]
    data=DiscoveryData(
        train=subset_discovery_split(original.train,kept),
        validation=subset_discovery_split(original.validation,kept),
        receipt={
            **original.receipt,
            "channel_names":feature["selected_names"],
            "n_channels":len(kept),
            "physical_K_unverified":True,
            "sampling_K_common_core_not_clinical_cause":True,
        },
        medians=original.medians[kept],
        scales=original.scales[kept],
    )
    _,outer=split_validation_for_cardinality(
        data.validation,
        random_seed=int(v6["evaluation"]["reported_validation_subset_seed"]),
    )
    torch.set_num_threads(int(v6["training"]["cpu_threads"]))
    reference=MaskRankAnchoredDualConceptFAN(
        physiological_observable_basis=observed_basis,
        care_rank=core,
        channels=tuple(data.receipt["channel_names"]),
        forecast_horizon=data.train.future.shape[1],
    )
    prior=fit_train_only_care_task_prior(
        reference,data.train.mask,data.train.y,
        c=float(v8["training"]["canonical_c"]),
        clip_beta=float(v8["training"]["max_canonical_beta"]),
    )
    args.out_dir.mkdir(parents=True,exist_ok=False)
    individual=[]
    pairwise=[]
    for method in methods:
        by_source={"measured_evidence":{},"care_observation_process":{}}
        for seed in v8["training"]["initialization_seeds"]:
            seed=int(seed)
            model=fit_one(
                method,seed,data,observed_basis,core,prior,v6,v8,
            )
            with torch.no_grad():
                frozen=model.care_operator_definition_error()
            if float(frozen)>1e-6:
                raise AssertionError("Train-derived care operator concept definitions drifted")
            physical=evaluate_discovery(model.physical,outer)
            care=collect_observation_process_evidence(
                model,outer.mask,outer.y,
            )
            by_source["measured_evidence"][seed]=physical
            by_source["care_observation_process"][seed]=care
            for policy_number,policy in enumerate(
                v8["evaluation"]["original_and_artificially_thinned_observations"]
            ):
                metrics=evaluate_dual(
                    model,outer,scenario=policy,
                    perturbation_seed=int(v6["evaluation"]["measurement_mask_seed"])+policy_number,
                )
                # Only aggregate metrics, never original predictions/rows.
                metrics.pop("_raw_mean_probability")
                metrics.pop("_raw_phys_contribution")
                metrics.pop("_raw_process_contribution")
                individual.append({
                    "method":method,
                    "initialization_seed":seed,
                    "evaluation_policy":policy,
                    "source_half_train_mask_common_care_K":core.chosen_k,
                    "source_pooled_uncapped_K_not_used_for_selection":None,
                    "physical_K_computational_not_clinical":physical_budget,
                    "same_300_historical_development_validation_cases":True,
                    "validated_clinical_concept_names":False,
                    **metrics,
                })
            if seed==int(v8["training"]["initialization_seeds"][0]):
                passports={
                    "method":method,
                    "frozen_mask_process_anchor_source":core.source_workflow,
                    "chosen_common_mask_subspace_k":core.chosen_k,
                    "no_direct_latent_or_clinical_concept_annotations":True,
                    "care_process_native_passports":
                        model.process.passports(tuple(data.receipt["channel_names"])),
                    "measured_evidence_passports":
                        concept_passports(
                            physical,model.physical,outer,
                            data.medians,data.scales,include_examples=False
                        ),
                    "no_individual_patient_explanations_published":True,
                }
                (args.out_dir/f"care_common_core_passports_{method}.json").write_text(
                    json.dumps(passports,indent=2,allow_nan=False),encoding="utf-8"
                )
            print(
                "V9",method,"seed",seed,"K",core.chosen_k,
                "validation_auprc",round(individual[-3]["auprc_full"],5),
                flush=True
            )
        for concept_family,evidence in by_source.items():
            seeds=sorted(evidence)
            for i,a in enumerate(seeds):
                for b in seeds[i+1:]:
                    agreement=compare_independent_discoveries(
                        evidence[a],evidence[b]
                    )
                    pairwise.append({
                        "method":method,"family":concept_family,
                        "seed_a":a,"seed_b":b,**agreement
                    })
    result=pd.DataFrame(individual)
    pairs=pd.DataFrame(pairwise)
    result.to_csv(args.out_dir/"v9_common_care_model_by_policy.csv",index=False)
    pairs.to_csv(args.out_dir/"v9_common_care_between_seeds.csv",index=False)
    summary=[]
    for method,group in result.groupby("method"):
        natural=group[group.evaluation_policy=="original"]
        between=pairs[
            (pairs.method==method)&(pairs.family=="care_observation_process")
        ]
        physical=pairs[
            (pairs.method==method)&(pairs.family=="measured_evidence")
        ]
        summary.append({
            "method":method,
            "mask_process_K_from_common_core":core.chosen_k,
            "independent_model_initializations":len(natural),
            "original_validation_auprc_mean":float(natural.auprc_full.mean()),
            "original_validation_auprc_std":float(natural.auprc_full.std(ddof=1)),
            "physical_only_auprc_mean":float(natural.auprc_physical_only.mean()),
            "care_only_auprc_mean":float(natural.auprc_observation_process_only.mean()),
            "mean_abs_logit_contribution_share_from_care":
                float(natural.mean_fraction_signed_abs_logit_from_process.mean()),
            "mean_case_specific_care_contributions_spearman_excess":
                float(between.excess_over_unmatched_episode_null_spearman.mean()),
            "mean_case_specific_physical_contributions_spearman_excess":
                float(physical.excess_over_unmatched_episode_null_spearman.mean()),
            "mean_care_operator_definition_similarity":
                float(between.definition_cosine_normalized.mean()),
            "mean_physical_operator_definition_similarity":
                float(physical.definition_cosine_normalized.mean()),
            "max_logit_decomposition_error":
                float(natural.max_logit_additivity_error.max()),
        })
    pd.DataFrame(summary).to_csv(args.out_dir/"v9_common_care_summary.csv",index=False)
    (args.out_dir/"STATUS.json").write_text(
        json.dumps({
            "status":"EXPLORATORY_COMMON_MASK_CORE_TRAINED_END_TO_END",
            "pinned_v9_common_basis":core.source_workflow,
            "train_only_common_K":core.chosen_k,
            "clinical_care_mechanism_count_claimed":False,
            "clinical_physiological_K_claimed":False,
            "frozen_independent_patient_mask_pilot":True,
            "patient_label_data_used_for_predictive_training_only":True,
            "test_or_calibration_data_used":False,
            "already_used_validation_not_independent_confirmation":True,
            "only_three_initialization_seeds":True,
            "both_families_exactly_concept_mediated":True,
            "all_results":summary,
        },indent=2,allow_nan=False),encoding="utf-8",
    )
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


if __name__=="__main__":
    main()
