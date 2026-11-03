"""V10 matched PhysioNet development ablation: V7 baseline vs native multi-scale.

Experiments differ ONLY in the physiological concept formation and its
optional within-run consistency loss. The mask-process head, train-only
sensor discovery, train-only rank, 36h history/12h forecast, optimizer,
18 epochs, initialization seeds, and validation patients are shared.
All methods re-train from scratch. No historical test/calibration access.

Exploratory science: 3 independently initialized networks per arm and
previously viewed 300 development patients DO NOT establish p-values.
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .clinical_dual_process_experiment_v7 import fit_dual, evaluate_dual
from .clinical_unknown_neutral_experiment_v6 import subset_discovery_split
from .data import DiscoveryData
from .dual_process_stability_v7 import collect_observation_process_evidence
from .evaluation import compare_independent_discoveries, concept_passports, evaluate_discovery
from .lagged_rank import fit_lagged_predictive_rank
from .model_v2 import masked_observation_view, v2_loss
from .multiscale_physiology_v10 import (
    MultiScaleDualProcessConceptFAN, MultiScaleUnknownNeutralConceptFAN,
    physiological_scale_entropy, expanded_multiscale_definition, SCALES,
)
from .observation_process_concepts_v7 import dual_training_objective
from .physionet import prepare_physionet_discovery
from .pilot_temporal_sensors_v5 import choose_common_temporal_sensors
from .selection import split_validation_for_cardinality
from .unknown_neutral_v6 import cross_view_concept_consistency


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def fit_multiscale(
    data: DiscoveryData,
    basis: np.ndarray,
    seed: int,
    base_config: dict,
    v7_config: dict,
    v10_config: dict,
    *,
    consistency: bool,
) -> tuple[MultiScaleDualProcessConceptFAN, dict]:
    _seed_everything(seed)
    t = base_config["training"]
    dual = v7_config["dual_head"]
    extra = v10_config["training"]
    model = MultiScaleDualProcessConceptFAN(
        observable_loadings=basis,
        channels=tuple(data.receipt["channel_names"]),
        forecast_horizon=data.train.future.shape[1],
        n_observation_concepts=int(dual["n_observation_process_concepts"]),
        scales=tuple(int(v) for v in v10_config["architecture"]["scales_hours"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
    )
    positives = float(data.train.y.sum())
    pos_weight = torch.tensor(
        (len(data.train.y)-positives)/max(positives,1.), dtype=torch.float32,
    )
    loader = data.train.loader(
        int(t["batch_size"]),True,int(t["batch_order_seed"]),
    )
    main_mask_rng=torch.Generator().manual_seed(seed+71021)
    second_mask_rng=torch.Generator().manual_seed(seed+71022)
    steps=0
    accumulation={"consistency":0., "joint_reliability":0.,
                  "scale_entropy":0.}
    for epoch in range(int(t["epochs"])):
        model.train()
        for x,original_mask,future,future_mask,y in loader:
            optimizer.zero_grad(set_to_none=True)
            corrupted,visible,heldout = masked_observation_view(
                x,original_mask,
                fraction=float(t["main_mask_fraction"]),
                generator=main_mask_rng,
            )
            out=model(corrupted,visible,process_mask=original_mask)
            base,parts=v2_loss(
                model.physical,out.physical,y,future,future_mask,x,heldout,
                mode="full",pos_weight=pos_weight,loss_weights=t["loss_weights"],
            )
            objective,receipt=dual_training_objective(
                model,out,y,base,parts["task"],
                pos_weight=pos_weight,
                mask_coeff_l1=float(dual["signed_mask_coefficient_l1"]),
                mask_redundancy_weight=float(dual["mask_definition_redundancy"]),
            )
            source=model.physical.spectral_definition_penalty()
            objective = (
                objective
                +float(t["spectral_anchor_weight"])*source["operator_displacement"]
                +float(t["spectral_axis_independence_weight"])*source["observable_axis_nonorthogonality"]
            )
            scale_entropy = physiological_scale_entropy(model)
            objective = objective + float(extra["scale_entropy_weight"])*scale_entropy
            # The consistency variant compares two independently thinned
            # physical views of the SAME patient, aligned by the native
            # concept dimensions and weighted by joint measurement support.
            consistency_value = objective.new_zeros(())
            joint_reliability = objective.new_zeros(())
            if consistency:
                altered,altered_mask,_ = masked_observation_view(
                    x,original_mask,
                    fraction=float(extra["second_mask_fraction"]),
                    generator=second_mask_rng,
                )
                out_second=model(altered,altered_mask,process_mask=original_mask)
                consistency_value,joint_reliability = cross_view_concept_consistency(
                    out.physical,out_second.physical,
                    minimum_joint_reliability=float(extra["minimum_joint_reliability"]),
                )
                objective=objective+float(extra["consistency_weight"])*consistency_value
            if not torch.isfinite(objective):
                raise FloatingPointError(f"Nonfinite V10 loss at seed {seed} epoch {epoch}")
            objective.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),float(t["gradient_clip"])
            )
            optimizer.step()
            accumulation["consistency"]+=float(consistency_value.detach())
            accumulation["joint_reliability"]+=float(joint_reliability.detach())
            accumulation["scale_entropy"]+=float(scale_entropy.detach())
            steps+=1
    model.eval()
    return model,{
        "n_training_minibatches":steps,
        "mean_training_consistency_penalty":accumulation["consistency"]/max(steps,1),
        "mean_training_joint_measurement_reliability":
            accumulation["joint_reliability"]/max(steps,1),
        "mean_train_scale_entropy":accumulation["scale_entropy"]/max(steps,1),
    }


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--reference-v6-config",type=Path,required=True)
    parser.add_argument("--reference-v7-config",type=Path,required=True)
    parser.add_argument("--raw-zip",type=Path,required=True)
    parser.add_argument("--outcomes",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args()
    if args.out_dir.exists():
        parser.error("Never overwrite prior evidence")
    cfg=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ref=yaml.safe_load(args.reference_v6_config.read_text(encoding="utf-8"))
    v7=yaml.safe_load(args.reference_v7_config.read_text(encoding="utf-8"))
    if not (cfg["protocol"]["test_calibration_sealed"] and
            cfg["protocol"]["clinical_semantics_not_identified"] and
            ref["protocol"]["test_and_calibration_sealed"] and
            v7["protocol"]["no_test_calibration_or_new_clinical_significance"]):
        parser.error("Original test/calibration and concept-identity restrictions are mandatory")
    methods=tuple(cfg["training"]["methods"])
    if methods != ("v7_retrained_baseline","v10_multiscale","v10_multiscale_consistency"):
        parser.error("All frozen, ordered ablations required")
    seeds=tuple(map(int,cfg["training"]["initialization_seeds"]))
    if len(set(seeds))!=len(seeds) or len(seeds)<3:
        parser.error("At least 3 distinct training initializations required")
    if seeds != tuple(map(int,ref["training"]["initialization_seeds"])):
        parser.error("Use the original V7 seeds for matched comparison")
    with tempfile.TemporaryDirectory(prefix="v10_clinical_split_") as tmp:
        initial=prepare_physionet_discovery(
            args.raw_zip,args.outcomes,Path(tmp),
            split_seed=int(ref["data"]["split_seed"]),
            input_window=int(ref["data"]["input_window"]),
            forecast_horizon=int(ref["data"]["forecast_horizon"]),
            minimum_train_patient_coverage=float(ref["data"]["minimum_train_patient_coverage"]),
        )
    if len(initial.train.y)!=2400 or len(initial.validation.y)!=600:
        raise ValueError("Official split changed")
    x,mask=initial.train.x,initial.train.mask
    order=np.random.default_rng(int(ref["data"]["pilot_selection_seed"])).permutation(len(x))
    pilot=order[:int(ref["data"]["pilot_train_patients"])]
    independent=order[len(pilot):]
    feature=choose_common_temporal_sensors(
        mask[pilot],initial.receipt["channel_names"],
        lags=tuple(ref["data"]["pilot_sensor_lags"]),
        minimum_supported_strata=int(ref["data"]["pilot_min_time_strata"]),
        minimum_observed_pairs_per_stratum=int(ref["data"]["pilot_min_hourly_pairs"]),
        minimum_expected_null_pairs_per_stratum=float(ref["data"]["pilot_min_expected_null_pairs"]),
        worst_additional_observation_retention=float(ref["data"]["pilot_worst_extra_thinning"]),
    )
    if feature["status"]!="PILOT_SENSORS_ELIGIBLE":
        raise RuntimeError("Train-only temporal sensors not eligible")
    selected=np.asarray(feature["selected_indices"],dtype=int)
    rankcfg=ref["rank"]
    rank=fit_lagged_predictive_rank(
        x[independent][:,:,selected],mask[independent][:,:,selected],
        lags=tuple(rankcfg["lags"]),permutations=int(rankcfg["permutations"]),
        patient_bootstraps=int(rankcfg["patient_bootstraps"]),
        alpha=float(rankcfg["alpha"]),seed=int(rankcfg["seed"]),
    )
    capacity=min(int(rank.rank),int(rankcfg["max_computational_concept_slots"]))
    if capacity<2:
        raise RuntimeError("Train-only computed capacity too small; do not force K")
    basis=rank.left_observable_loadings[:,:capacity]
    data=DiscoveryData(
        train=subset_discovery_split(initial.train,selected),
        validation=subset_discovery_split(initial.validation,selected),
        receipt={**initial.receipt,"channel_names":feature["selected_names"],
                 "n_channels":len(selected),"no_manually_named_concepts":True},
        medians=initial.medians[selected],scales=initial.scales[selected],
    )
    _,dev=split_validation_for_cardinality(
        data.validation,
        random_seed=int(ref["evaluation"]["reported_validation_subset_seed"]),
    )
    torch.set_num_threads(int(ref["training"]["cpu_threads"]))
    args.out_dir.mkdir(parents=True,exist_ok=False)
    rows=[]
    between=[]
    per_seed={}
    for method in methods:
        per_seed[method]={"phys":{},"care":{}}
        for seed in seeds:
            if method=="v7_retrained_baseline":
                model,training=fit_dual(data,basis,seed,ref,v7)
            else:
                model,training=fit_multiscale(
                    data,basis,seed,ref,v7,cfg,
                    consistency=(method=="v10_multiscale_consistency"),
                )
            # Match concept identities using BOTH the learned signed feature
            # dictionary and the learned temporal scale mixture. Otherwise
            # V10 could look falsely stable even when scale assignments drift.
            physical_evidence=evaluate_discovery(model.physical,dev)
            # Shared self-supervised quality audit, same seed and hidden
            # observed cells across every arm. All targets were recorded
            # in the development cohort; no future observations enter input.
            with torch.no_grad():
                original_x=torch.from_numpy(dev.x)
                recorded_mask=torch.from_numpy(dev.mask)
                hidden_x,visible_mask,withheld=masked_observation_view(
                    original_x,recorded_mask,
                    fraction=float(ref["training"]["main_mask_fraction"]),
                    generator=torch.Generator().manual_seed(
                        int(ref["evaluation"]["measurement_mask_seed"])+900
                    ),
                )
                hidden_result=model.physical(hidden_x,visible_mask)
                denoise_error=(hidden_result.reconstruction-original_x).square()
                denoise_rmse=float(
                    ((denoise_error*withheld).sum()/withheld.sum().clamp_min(1.)).sqrt()
                )
            per_seed[method]["phys"][seed]=replace(
                physical_evidence,
                weights=expanded_multiscale_definition(model),
            )
            per_seed[method]["care"][seed]=collect_observation_process_evidence(
                model,dev.mask,dev.y
            )
            for i,scenario in enumerate(cfg["evaluation"]["scenarios"]):
                metrics=evaluate_dual(
                    model,dev,scenario=scenario,
                    perturbation_seed=int(ref["evaluation"]["measurement_mask_seed"])+i,
                )
                metrics.pop("_raw_mean_probability")
                metrics.pop("_raw_phys_contribution")
                metrics.pop("_raw_process_contribution")
                rows.append({
                    "method":method,"seed":seed,
                    "development_scenario":scenario,
                    "physical_K_is_computational_budget":capacity,
                    "care_K_is_architectural_budget":model.process.n_concepts,
                    "same_300_previously_inspected_development_patients":True,
                    "physical_future_observed_rmse":float(
                        physical_evidence.metrics["future_observed_rmse"]
                    ),
                    "physical_masked_reconstruction_rmse":denoise_rmse,
                    "physical_mean_effective_concept_count":float(
                        physical_evidence.metrics["mean_effective_concept_count"]
                    ),
                    **metrics,**training,
                })
            if method!="v7_retrained_baseline" and seed==seeds[0]:
                item={
                    "method":method,
                    "physiology_multiscale":model.physical.multiscale_passport(),
                    "care_process":model.process.passports(tuple(data.receipt["channel_names"])),
                    "no_patient_ids_or_predictions_in_report":True,
                }
                (args.out_dir/f"{method}_definition_passport.json").write_text(
                    json.dumps(item,indent=2,allow_nan=False),encoding="utf-8"
                )
            print("V10",method,"seed",seed,"original AUPRC",round(rows[-3]["auprc_full"],5),flush=True)
        for family,group in per_seed[method].items():
            for a,b in itertools.combinations(seeds,2):
                agree=compare_independent_discoveries(group[a],group[b])
                between.append({"method":method,"concept_family":family,
                                "seed_a":a,"seed_b":b,**agree})
    records=pd.DataFrame(rows)
    agreement=pd.DataFrame(between)
    records.to_csv(args.out_dir/"v10_matched_development_by_scenario.csv",index=False)
    agreement.to_csv(args.out_dir/"v10_cross_seed_concept_agreement.csv",index=False)
    summary=[]
    for method,group in records.groupby("method",sort=False):
        source=group[group.development_scenario.eq("original")]
        phys=agreement[(agreement.method==method)&(agreement.concept_family=="phys")]
        care=agreement[(agreement.method==method)&(agreement.concept_family=="care")]
        summary.append({
            "method":method,
            "n_initializations":len(source),
            "auprc_original":float(source.auprc_full.mean()),
            "auprc_mcar":float(group[group.development_scenario.eq("mcar_thin")].auprc_full.mean()),
            "auprc_time_mar":float(group[group.development_scenario.eq("time_channel_mar")].auprc_full.mean()),
            "auprc_physical_only":float(source.auprc_physical_only.mean()),
            "auprc_care_only":float(source.auprc_observation_process_only.mean()),
            "physical_future_observed_rmse":float(
                source.physical_future_observed_rmse.mean()
            ),
            "physical_masked_reconstruction_rmse":float(
                source.physical_masked_reconstruction_rmse.mean()
            ),
            "physical_mean_effective_concept_count":float(
                source.physical_mean_effective_concept_count.mean()
            ),
            "physical_case_specific_spearman_excess":
                float(phys.excess_over_unmatched_episode_null_spearman.mean()),
            "care_case_specific_spearman_excess":
                float(care.excess_over_unmatched_episode_null_spearman.mean()),
            "physical_definition_cosine_normalized":
                float(phys.definition_cosine_normalized.mean()),
            "max_logit_additivity_error":float(group.max_logit_additivity_error.max()),
            "mean_mask_process_abs_logit_share":
                float(source.mean_fraction_signed_abs_logit_from_process.mean()),
        })
    result=pd.DataFrame(summary)
    result.to_csv(args.out_dir/"v10_ablation_summary.csv",index=False)
    receipt={
        "status":"EXPLORATORY_V10_MATCHED_TRAINING_COMPLETE",
        "reference":"V7 baseline, refit in same workflow",
        "new_method":"train-time causal multi-resolution physiological concepts",
        "methods":list(methods),
        "same_original_train_and_development_splits":True,
        "clinical_identity_claim":False,
        "number_of_real_physiological_causes_identified":False,
        "calibration_or_test_used":False,
        "confirmed_significance_claim":False,
        "n_independent_initializations":len(seeds),
        "n_independent_hospitals":1,
        "development_patients_repeatedly_inspected":True,
        "sensor_selection_train_only":True,
        "latent_dim_only_computational_budget":capacity,
        "measured_channels":list(data.receipt["channel_names"]),
        "results":summary,
    }
    (args.out_dir/"STATUS.json").write_text(
        json.dumps(receipt,indent=2,allow_nan=False),encoding="utf-8"
    )
    print(result.to_string(index=False),flush=True)


if __name__=="__main__":
    main()
