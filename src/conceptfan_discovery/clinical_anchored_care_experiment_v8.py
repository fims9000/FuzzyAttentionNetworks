"""V8: automatic TRAIN-only care-process concept count and anchored source.

Contrast the unstable unconstrained V7 mask head with two precisely
prespecified anchored care-process variants:
- `train_mask_source_only`: K and operator direction derived by train-only
  mask covariance and patient-permuted feature-null; freeze signed source
  definitions; learn thresholds and signed task effect while training.
- `train_mask_source_and_task_prior`: same K/source; initialize and
  penalize gamma around a fixed TRAIN-only supervised logistic task head.

Physical evidence retains the already trained V7 UNKNOWN-neutral
bottleneck. Both physical and care concepts influence the task logit
during every training step, not post-hoc. No clinical concept annotations,
validation/tuning feedback or old test/calibration are supplied to K.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .anchored_care_process_v8 import (
    MaskRankAnchoredDualConceptFAN,
    fit_train_only_care_task_prior,initialize_task_prior,process_head_prior_loss,
)
from .care_process_rank_v8 import fit_care_policy_rank
from .clinical_dual_process_experiment_v7 import evaluate_dual
from .clinical_unknown_neutral_experiment_v6 import subset_discovery_split
from .data import DiscoveryData
from .dual_process_stability_v7 import collect_observation_process_evidence
from .evaluation import compare_independent_discoveries, evaluate_discovery
from .lagged_rank import fit_lagged_predictive_rank
from .model_v2 import masked_observation_view,v2_loss
from .observation_process_concepts_v7 import dual_training_objective
from .physionet import prepare_physionet_discovery
from .pilot_temporal_sensors_v5 import choose_common_temporal_sensors
from .selection import split_validation_for_cardinality


def fit_one(
    method:str,seed:int,data:DiscoveryData,
    basis:np.ndarray,care_rank,prior,
    v6:dict,v8:dict,
)->MaskRankAnchoredDualConceptFAN:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model=MaskRankAnchoredDualConceptFAN(
        physiological_observable_basis=basis,
        care_rank=care_rank,
        channels=tuple(data.receipt["channel_names"]),
        forecast_horizon=data.train.future.shape[1],
    )
    if method=="train_mask_source_and_task_prior":
        initialize_task_prior(model,prior)
    elif method!="train_mask_source_only":
        raise ValueError("Unknown frozen V8 method")
    t=v6["training"]
    optimizer=torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
    )
    positive=float(data.train.y.sum())
    pos_weight=torch.tensor(
        (len(data.train.y)-positive)/max(positive,1.),dtype=torch.float32,
    )
    batches=data.train.loader(
        int(t["batch_size"]),True,int(t["batch_order_seed"])
    )
    generator=torch.Generator().manual_seed(seed+71021)
    for epoch in range(int(t["epochs"])):
        model.train()
        for x,actual_mask,future,future_mask,y in batches:
            optimizer.zero_grad(set_to_none=True)
            measured,visible,withheld=masked_observation_view(
                x,actual_mask,
                fraction=float(t["main_mask_fraction"]),
                generator=generator,
            )
            # No contamination of care-process evidence by artificial
            # denoising omissions of physiological measurements.
            output=model(measured,visible,process_mask=actual_mask)
            old,components=v2_loss(
                model.physical,output.physical,y,future,future_mask,x,withheld,
                mode="full",pos_weight=pos_weight,
                loss_weights=t["loss_weights"],
            )
            loss,_=dual_training_objective(
                model,output,y,old,components["task"],
                pos_weight=pos_weight,
                mask_coeff_l1=float(v8["training"]["mask_coeff_l1"]),
                mask_redundancy_weight=float(v8["training"]["mask_definition_redundancy"]),
            )
            penalty=model.physical.spectral_definition_penalty()
            loss=(
                loss
                +float(t["spectral_anchor_weight"])*penalty["operator_displacement"]
                +float(t["spectral_axis_independence_weight"])*penalty["observable_axis_nonorthogonality"]
            )
            if method=="train_mask_source_and_task_prior":
                loss=(
                    loss+float(v8["training"]["signed_task_prior_weight"])*
                    process_head_prior_loss(model,prior)
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Nonfinite V8 model loss method={method} seed={seed} epoch={epoch}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),float(t["gradient_clip"])
            )
            optimizer.step()
    model.eval()
    if float(model.care_operator_definition_error())>1e-6:
        raise AssertionError("Frozen observed-mask source operator definitions were altered")
    return model


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--reference-v6-config",type=Path,required=True)
    p.add_argument("--raw-zip",type=Path,required=True)
    p.add_argument("--outcomes",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    if args.out_dir.exists():
        p.error("Refusing to overwrite completed research evidence")
    cfg=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    v6=yaml.safe_load(args.reference_v6_config.read_text(encoding="utf-8"))
    if not cfg["protocol"]["historical_test_and_calibration_sealed"] or not v6["protocol"]["test_and_calibration_sealed"]:
        p.error("Clinical test and calibration sets must remain sealed")
    expected={"train_mask_source_only","train_mask_source_and_task_prior"}
    if set(cfg["training"]["methods"])!=expected:
        p.error("All prespecified concept source and task-head controls required")
    with tempfile.TemporaryDirectory(prefix="anchored_icu_v8_") as tmp:
        full=prepare_physionet_discovery(
            args.raw_zip,args.outcomes,Path(tmp),
            split_seed=int(v6["data"]["split_seed"]),
            input_window=int(v6["data"]["input_window"]),
            forecast_horizon=int(v6["data"]["forecast_horizon"]),
            minimum_train_patient_coverage=float(
                v6["data"]["minimum_train_patient_coverage"]),
        )
    x=full.train.x
    m=full.train.mask
    order=np.random.default_rng(
        int(v6["data"]["pilot_selection_seed"])
    ).permutation(len(x))
    pilot=order[:int(v6["data"]["pilot_train_patients"])]
    nonpilot=order[len(pilot):]
    chosen=choose_common_temporal_sensors(
        m[pilot],full.receipt["channel_names"],
        lags=tuple(v6["data"]["pilot_sensor_lags"]),
        minimum_supported_strata=int(v6["data"]["pilot_min_time_strata"]),
        minimum_observed_pairs_per_stratum=int(v6["data"]["pilot_min_hourly_pairs"]),
        minimum_expected_null_pairs_per_stratum=float(v6["data"]["pilot_min_expected_null_pairs"]),
        worst_additional_observation_retention=float(v6["data"]["pilot_worst_extra_thinning"]),
    )
    if chosen["status"]!="PILOT_SENSORS_ELIGIBLE":
        raise RuntimeError("Train-only pilot selected no jointly observed clinical channels")
    retained=np.asarray(chosen["selected_indices"],dtype=int)
    r=v6["rank"]
    phys_rank=fit_lagged_predictive_rank(
        x[nonpilot][:,:,retained],m[nonpilot][:,:,retained],
        lags=tuple(r["lags"]),
        permutations=int(r["permutations"]),
        patient_bootstraps=int(r["patient_bootstraps"]),
        alpha=float(r["alpha"]),seed=int(r["seed"]),
    )
    phys_budget=min(int(phys_rank.rank),int(r["max_computational_concept_slots"]))
    if phys_budget<2:
        raise RuntimeError("No adequate measured-evidence computational capacity")
    phys_basis=phys_rank.left_observable_loadings[:,:phys_budget]
    care_rank=fit_care_policy_rank(
        m[:,:,retained],
        permutations=int(cfg["care_policy_rank"]["feature_permutations"]),
        seed=int(cfg["care_policy_rank"]["null_seed"]),
        alpha=float(cfg["care_policy_rank"]["familywise_alpha"]),
        max_slots=int(cfg["care_policy_rank"]["max_computational_policy_slots"]),
    )
    source=DiscoveryData(
        train=subset_discovery_split(full.train,retained),
        validation=subset_discovery_split(full.validation,retained),
        receipt={
            **full.receipt,
            "channel_names":chosen["selected_names"],
            "n_channels":len(retained),
            "unknown_medical_ontology":True,
        },
        medians=full.medians[retained],scales=full.scales[retained],
    )
    _,outer=split_validation_for_cardinality(
        source.validation,
        random_seed=int(v6["evaluation"]["reported_validation_subset_seed"])
    )
    torch.set_num_threads(int(v6["training"]["cpu_threads"]))
    reference=MaskRankAnchoredDualConceptFAN(
        physiological_observable_basis=phys_basis,care_rank=care_rank,
        channels=tuple(source.receipt["channel_names"]),
        forecast_horizon=source.train.future.shape[1],
    )
    prior=fit_train_only_care_task_prior(
        reference,source.train.mask,source.train.y,
        c=float(cfg["training"]["canonical_c"]),
        clip_beta=float(cfg["training"]["max_canonical_beta"]),
    )
    args.out_dir.mkdir(parents=True,exist_ok=False)
    (args.out_dir/"train_mask_rank_and_task_prior.json").write_text(
        json.dumps({
            "physiological_budget_not_verified_clinical_k":phys_budget,
            "care_process_rank":care_rank.receipt(),
            "shared_care_task_head_train_only":prior.receipt(),
            "clinical_concept_ontology_verified":False,
        },indent=2,allow_nan=False),encoding="utf-8",
    )
    per_run,per_environment,pairs=[],[],[]
    for method in cfg["training"]["methods"]:
        phys_evidence={}
        care_evidence={}
        for seed in cfg["training"]["initialization_seeds"]:
            seed=int(seed)
            model=fit_one(
                method,seed,source,phys_basis,care_rank,prior,v6,cfg
            )
            phys_evidence[seed]=evaluate_discovery(model.physical,outer)
            care_evidence[seed]=collect_observation_process_evidence(
                model,outer.mask,outer.y
            )
            clean=evaluate_dual(
                model,outer,scenario="original",
                perturbation_seed=int(v6["evaluation"]["measurement_mask_seed"]),
            )
            for scenario_i,scenario in enumerate(cfg["evaluation"]["original_and_artificially_thinned_observations"]):
                rec=evaluate_dual(
                    model,outer,scenario=scenario,
                    perturbation_seed=int(v6["evaluation"]["measurement_mask_seed"])+scenario_i,
                )
                raw=rec.pop("_raw_mean_probability")
                rec.pop("_raw_phys_contribution")
                rec.pop("_raw_process_contribution")
                per_environment.append({
                    "method":method,"init_seed":seed,"scenario":scenario,
                    "train_only_care_slots":care_rank.chosen_k,
                    "source_operator_frozen":True,
                    "care_source_treatment_effect_claim":False,
                    **rec,
                })
            per_run.append({
                "method":method,"init_seed":seed,
                "train_only_care_detected_k":care_rank.chosen_k,
                "candidate_phys_slots_not_medical_k":phys_budget,
                "outer_original_auprc":float(clean["auprc_full"]),
                "outer_physical_only_auprc":float(clean["auprc_physical_only"]),
                "outer_care_only_auprc":float(clean["auprc_observation_process_only"]),
                "care_signed_beta":model.process.direct_weights().detach().cpu().numpy().tolist(),
                "source_weight_error":float(model.care_operator_definition_error()),
                "source_process_logit_fraction":float(clean["mean_fraction_signed_abs_logit_from_process"]),
                "both_families_used_in_training":True,
            })
            print(
                "method",method,"seed",seed,
                "observed_mask_k",care_rank.chosen_k,
                "outer_auprc",round(clean["auprc_full"],4),
                flush=True,
            )
        for family,record in (
            ("physiological_measured_evidence",phys_evidence),
            ("care_sampling_process_only",care_evidence),
        ):
            seeds=sorted(record)
            for i,a in enumerate(seeds):
                for b in seeds[i+1:]:
                    pairs.append({
                        "method":method,"family":family,
                        "init_seed_a":a,"init_seed_b":b,
                        **compare_independent_discoveries(
                            record[a],record[b]
                        ),
                    })
    frame=pd.DataFrame(per_run)
    pairs_df=pd.DataFrame(pairs)
    environment_df=pd.DataFrame(per_environment)
    frame.to_csv(args.out_dir/"anchored_care_v8_per_model.csv",index=False)
    pairs_df.to_csv(args.out_dir/"anchored_care_v8_between_seeds.csv",index=False)
    environment_df.to_csv(args.out_dir/"anchored_care_v8_measurement_stress.csv",index=False)
    summaries=[]
    for method,group in frame.groupby("method"):
        for family,subset in pairs_df[pairs_df.method.eq(method)].groupby("family"):
            rho=subset["excess_over_unmatched_episode_null_spearman"].dropna()
            sign=subset["case_specific_excess_sign_agreement"].dropna()
            summaries.append({
                "method":method,"family":family,
                "independent_initializations":len(group),
                "n_original_model_pairs_dependently_shared":len(subset),
                "train_only_selected_care_k":care_rank.chosen_k,
                "mean_auprc":float(group.outer_original_auprc.mean()),
                "mean_care_logit_magnitude_fraction":float(group.source_process_logit_fraction.mean()),
                "mean_operator_definition_similarity":float(subset.definition_cosine_normalized.mean()),
                "mean_patient_specific_spearman_excess":
                    float(rho.mean()) if len(rho) else None,
                "mean_patient_specific_sign_excess":
                    float(sign.mean()) if len(sign) else None,
                "not_clinical_true_concepts":True,
            })
    pd.DataFrame(summaries).to_csv(args.out_dir/"anchored_care_v8_summary.csv",index=False)
    (args.out_dir/"STATUS.json").write_text(json.dumps({
        "status":"EXPLORATORY_TRAIN_MASK_CARDINALITY_AND_SOURCE_ANCHORING",
        "previous_v7_original_comparator":"https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41509956362",
        "care_process_concept_count_not_manual":True,
        "care_source_directions_train_masks_only":True,
        "train_task_beta_prior_outcomes_only":True,
        "physiology_medical_k_still_unidentified":True,
        "reused_historical_validation_not_confirmatory":True,
        "test_or_calibration_evaluated":False,
        "n_total_trained_models":len(frame),
        "results":summaries,
    },indent=2,allow_nan=False),encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)


if __name__=="__main__":
    main()
