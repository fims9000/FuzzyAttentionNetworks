"""V7 real ICU: separate OBSERVATION PROCESS from native physiological evidence.

The verified rank-identification gate is ABSTAIN. Therefore K is only a
computational candidate capacity, not the number of real physiological
concepts. This study asks whether a mask-only, named sampling-process
concept family can recover label information lost when physiological
concepts correctly treat *unmeasured* as UNKNOWN.

TRAIN-only: mask pilot, spectral basis and outcome supervision.
Historical validation: exploratory scoring. Test/calibration: sealed.
No expert or synthetic concept labels and no black-box mask bypass.
"""
from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score,roc_auc_score

from .clinical_unknown_neutral_experiment_v6 import subset_discovery_split
from .data import DiscoveryData
from .evaluation import (
    concept_passports,evaluate_discovery,compare_independent_discoveries,
)
from .dual_process_stability_v7 import collect_observation_process_evidence
from .lagged_rank import fit_lagged_predictive_rank
from .model_v2 import masked_observation_view,v2_loss
from .observation_process_concepts_v7 import (
    DualProcessConceptFAN,dual_training_objective,
)
from .observation_regimes_v5 import apply_measurement_environment
from .physionet import prepare_physionet_discovery
from .pilot_temporal_sensors_v5 import choose_common_temporal_sensors
from .selection import split_validation_for_cardinality


def fit_dual(
    data:DiscoveryData,basis:np.ndarray,init_seed:int,
    train_cfg:dict,dual_cfg:dict,
)->tuple[DualProcessConceptFAN,dict]:
    random.seed(init_seed)
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    torch.use_deterministic_algorithms(True)
    t=train_cfg["training"]
    dual=dual_cfg["dual_head"]
    model=DualProcessConceptFAN(
        observable_loadings=basis,
        channels=tuple(data.receipt["channel_names"]),
        forecast_horizon=data.train.future.shape[1],
        n_observation_concepts=int(dual["n_observation_process_concepts"]),
    )
    optimizer=torch.optim.AdamW(
        model.parameters(),lr=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
    )
    positives=float(data.train.y.sum())
    pos_weight=torch.tensor(
        (len(data.train.y)-positives)/max(positives,1.),dtype=torch.float32,
    )
    order=data.train.loader(
        int(t["batch_size"]),True,int(t["batch_order_seed"]),
    )
    thinning=torch.Generator().manual_seed(init_seed+71021)
    steps=0
    mean_process_abs=0.
    for epoch in range(int(t["epochs"])):
        model.train()
        for x,recorded_mask,future,future_mask,y in order:
            optimizer.zero_grad(set_to_none=True)
            physical_x,physical_mask,hidden=masked_observation_view(
                x,recorded_mask,
                fraction=float(t["main_mask_fraction"]),
                generator=thinning,
            )
            # Artificial dropout affects *only physiological input*.
            # Original clinical measurement schedule remains intact for
            # dedicated, explicitly named care-process concepts.
            out=model(
                physical_x,physical_mask,
                process_mask=recorded_mask,
            )
            base,parts=v2_loss(
                model.physical,out.physical,y,future,future_mask,x,hidden,
                mode="full",pos_weight=pos_weight,
                loss_weights=t["loss_weights"],
            )
            objective,receipt=dual_training_objective(
                model,out,y,base,parts["task"],
                pos_weight=pos_weight,
                mask_coeff_l1=float(dual["signed_mask_coefficient_l1"]),
                mask_redundancy_weight=float(dual["mask_definition_redundancy"]),
            )
            spectral=model.physical.spectral_definition_penalty()
            objective=(
                objective
                +float(t["spectral_anchor_weight"])*spectral["operator_displacement"]
                +float(t["spectral_axis_independence_weight"])*spectral["observable_axis_nonorthogonality"]
            )
            if not torch.isfinite(objective):
                raise FloatingPointError("Nonfinite dual clinical loss")
            objective.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),float(t["gradient_clip"])
            )
            optimizer.step()
            mean_process_abs+=float(out.observation_contributions.detach().abs().mean())
            steps+=1
    model.eval()
    return model,{
        "n_training_minibatches":steps,
        "training_mean_abs_care_process_logit_contribution":
            mean_process_abs/max(steps,1),
    }


def evaluate_dual(
    model:DualProcessConceptFAN,
    original,
    *,
    scenario:str,
    perturbation_seed:int,
)->dict:
    if scenario=="original":
        values,mask=original.x,original.mask
        retained=1.
    else:
        values,mask,receipt=apply_measurement_environment(
            original.x,original.mask,
            environment=scenario,seed=perturbation_seed,
        )
        retained=receipt["relative_retained_observations"]
    with torch.no_grad():
        out=model(
            torch.from_numpy(values),torch.from_numpy(mask),
            process_mask=torch.from_numpy(original.mask),
        )
        total=out.probabilities.detach().numpy()
        phys=out.physical.probability.detach().numpy()
        mask_only=torch.sigmoid(
            out.physical.decision_bias+out.observation_contributions.sum(dim=-1)
        ).detach().numpy()
        direct_physical=out.physical.contributions.detach().numpy()
        direct_process=out.observation_contributions.detach().numpy()
        max_error=float(out.decomposition_error())
    truth=original.y
    if np.unique(truth).size!=2:
        raise ValueError("Both clinical outcomes needed in evaluation")
    physical_mag=np.abs(direct_physical).sum(axis=1)
    process_mag=np.abs(direct_process).sum(axis=1)
    process_share=np.divide(
        process_mag,physical_mag+process_mag,
        out=np.zeros_like(process_mag),
        where=(physical_mag+process_mag)>1e-8,
    )
    return {
        "scenario":scenario,
        "retained_original_measurements":float(retained),
        "auprc_full":float(average_precision_score(truth,total)),
        "auprc_physical_only":float(average_precision_score(truth,phys)),
        "auprc_observation_process_only":float(average_precision_score(truth,mask_only)),
        "auroc_full":float(roc_auc_score(truth,total)),
        "mean_probability_full":float(total.mean()),
        "mean_abs_physical_contribution":float(physical_mag.mean()),
        "mean_abs_process_contribution":float(process_mag.mean()),
        "mean_fraction_signed_abs_logit_from_process":float(process_share.mean()),
        "max_logit_additivity_error":max_error,
        "patient_values_not_exported":True,
        "_raw_mean_probability":total,
        "_raw_phys_contribution":direct_physical,
        "_raw_process_contribution":direct_process,
    }


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--reference-v6-config",type=Path,required=True)
    p.add_argument("--raw-zip",type=Path,required=True)
    p.add_argument("--outcomes",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    if args.out_dir.exists():
        p.error("Do not overwrite previous real ICU evidence")
    cfg=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ref=yaml.safe_load(args.reference_v6_config.read_text(encoding="utf-8"))
    if (not cfg["protocol"]["no_test_calibration_or_new_clinical_significance"]
        or cfg["protocol"]["old_strict_rank_gate"]!="ABSTAIN_CLINICAL_K_UNSTABLE"
        or not ref["protocol"]["test_and_calibration_sealed"]):
        p.error("Historical medical rank abstention and test seal required")
    if len(ref["training"]["initialization_seeds"])<3:
        raise ValueError("Require at least 3 independently initialized training models")
    with tempfile.TemporaryDirectory(prefix="clinical_v7_original_split_") as tmp:
        all_data=prepare_physionet_discovery(
            args.raw_zip,args.outcomes,Path(tmp),
            split_seed=int(ref["data"]["split_seed"]),
            input_window=int(ref["data"]["input_window"]),
            forecast_horizon=int(ref["data"]["forecast_horizon"]),
            minimum_train_patient_coverage=float(
                ref["data"]["minimum_train_patient_coverage"]),
        )
    x=all_data.train.x
    mask=all_data.train.mask
    if len(x)!=2400 or len(all_data.validation.y)!=600:
        raise ValueError("Original official train-only clinical protocol changed")
    order=np.random.default_rng(
        int(ref["data"]["pilot_selection_seed"])
    ).permutation(len(x))
    pilot=order[:int(ref["data"]["pilot_train_patients"])]
    inference=order[len(pilot):]
    feature=choose_common_temporal_sensors(
        mask[pilot],all_data.receipt["channel_names"],
        lags=tuple(ref["data"]["pilot_sensor_lags"]),
        minimum_supported_strata=int(ref["data"]["pilot_min_time_strata"]),
        minimum_observed_pairs_per_stratum=int(ref["data"]["pilot_min_hourly_pairs"]),
        minimum_expected_null_pairs_per_stratum=float(
            ref["data"]["pilot_min_expected_null_pairs"]),
        worst_additional_observation_retention=float(
            ref["data"]["pilot_worst_extra_thinning"]),
    )
    if feature["status"]!="PILOT_SENSORS_ELIGIBLE":
        raise RuntimeError("Insufficient independently observable clinical sensors")
    selected=np.asarray(feature["selected_indices"],dtype=int)
    rankcfg=ref["rank"]
    rank=fit_lagged_predictive_rank(
        x[inference][:,:,selected],mask[inference][:,:,selected],
        lags=tuple(rankcfg["lags"]),
        permutations=int(rankcfg["permutations"]),
        patient_bootstraps=int(rankcfg["patient_bootstraps"]),
        alpha=float(rankcfg["alpha"]),seed=int(rankcfg["seed"]),
    )
    capacity=min(int(rank.rank),int(rankcfg["max_computational_concept_slots"]))
    if capacity<2:
        raise RuntimeError("No suitable train-only computational concept budget")
    basis=rank.left_observable_loadings[:,:capacity]
    channels=feature["selected_names"]
    data=DiscoveryData(
        train=subset_discovery_split(all_data.train,selected),
        validation=subset_discovery_split(all_data.validation,selected),
        receipt={
            **all_data.receipt,
            "channel_names":channels,
            "n_channels":len(selected),
            "no_manually_named_targets":True,
        },
        medians=all_data.medians[selected],
        scales=all_data.scales[selected],
    )
    _,outer=split_validation_for_cardinality(
        data.validation,random_seed=int(ref["evaluation"]["reported_validation_subset_seed"]),
    )
    torch.set_num_threads(int(ref["training"]["cpu_threads"]))
    args.out_dir.mkdir(parents=True,exist_ok=False)
    rows=[]
    phys_by_seed={}
    sampling_by_seed={}
    for seed in ref["training"]["initialization_seeds"]:
        seed=int(seed)
        model,training=fit_dual(data,basis,seed,ref,cfg)
        scenarios={}
        clean=evaluate_dual(
            model,outer,scenario="original",
            perturbation_seed=int(ref["evaluation"]["measurement_mask_seed"])
        )
        # Patient-level trajectories remain in-memory. The aggregate
        # across-retraining comparisons align native definitions and use
        # empirical patient-frequency nulls, not raw uncorrected agreement.
        phys_by_seed[seed]=evaluate_discovery(model.physical,outer)
        sampling_by_seed[seed]=collect_observation_process_evidence(
            model,outer.mask,outer.y
        )
        for i,scenario in enumerate(cfg["evaluation"]["scenarios"]):
            result=evaluate_dual(
                model,outer,scenario=scenario,
                perturbation_seed=int(ref["evaluation"]["measurement_mask_seed"])+i,
            )
            # PRIVATE arrays used only to compute aggregate robustness; never
            # write individual patient records in evidence artifacts.
            probability=result.pop("_raw_mean_probability")
            direct_phys=result.pop("_raw_phys_contribution")
            direct_process=result.pop("_raw_process_contribution")
            if scenario=="original":
                source_probability=probability
                source_physical=direct_phys
                source_process=direct_process
            result.update({
                "seed":seed,
                "candidate_physical_K_not_clinical_truth":capacity,
                "care_process_K":model.process.n_concepts,
                "mean_abs_risk_shift_from_original":float(
                    np.abs(probability-source_probability).mean()
                ),
                "mean_abs_physical_contribution_shift_from_original":float(
                    np.abs(direct_phys-source_physical).sum(axis=1).mean()
                ),
                "mean_abs_process_contribution_shift_from_original":float(
                    np.abs(direct_process-source_process).sum(axis=1).mean()
                ),
                "all_patient_records_sealed":True,
                **training,
            })
            rows.append(result)
        if seed==int(ref["training"]["initialization_seeds"][0]):
            phys=evaluate_discovery(model.physical,outer)
            physical_passports=concept_passports(
                phys,model.physical,outer,data.medians,data.scales,
                include_examples=False,
            )
            (args.out_dir/"dual_family_concept_passports_no_patient_examples.json").write_text(
                json.dumps({
                    "source_training_labels":"original_mortality_outcome_only",
                    "physical_slots_are_training_native_not_medical_ontology":True,
                    "care_process_slots_are_training_native_not_diagnoses":True,
                    "candidate_capacity":capacity,
                    "physical":physical_passports,
                    "care_process":model.process.passports(tuple(channels)),
                    "no_patient_specific_records":True,
                    "all_paths_exact_logit_attribution":True,
                },indent=2,allow_nan=False),encoding="utf-8",
            )
        print("seed",seed,
              "AUPRC full",round(clean["auprc_full"],4),
              "phys-only",round(clean["auprc_physical_only"],4),
              "process-only",round(clean["auprc_observation_process_only"],4),
              flush=True,
        )
    # Three trained models produce three OVERLAPPING comparison pairs.
    # These comparisons are descriptive and NEVER independently bootstrapped
    # as though they were independent patients or models.
    comparisons=[]
    original_seeds=sorted(phys_by_seed)
    for i,a in enumerate(original_seeds):
        for b in original_seeds[i+1:]:
            for group,source in (
                ("physiological_measured_evidence",phys_by_seed),
                ("care_sampling_process_only",sampling_by_seed),
            ):
                paired=compare_independent_discoveries(source[a],source[b])
                comparisons.append({
                    "concept_family":group,
                    "init_seed_a":a,"init_seed_b":b,
                    "same_original_outer_validation_patients":True,
                    "test_not_evaluated":True,
                    **paired,
                })
    pair_df=pd.DataFrame(comparisons)
    pair_df.to_csv(
        args.out_dir/"dual_process_v7_between_seeds.csv",index=False
    )
    data_frame=pd.DataFrame(rows)
    data_frame.to_csv(args.out_dir/"dual_process_v7_per_policy.csv",index=False)
    aggregate=[]
    for scenario,sub in data_frame.groupby("scenario"):
        aggregate.append({
            "scenario":scenario,
            "three_independent_model_initializations":len(sub),
            "candidate_physical_K":capacity,
            "n_observation_process_concepts":int(cfg["dual_head"]["n_observation_process_concepts"]),
            "mean_auprc_full":float(sub.auprc_full.mean()),
            "mean_auprc_physical_only":float(sub.auprc_physical_only.mean()),
            "mean_auprc_process_only":float(sub.auprc_observation_process_only.mean()),
            "mean_physical_contribution_shift":float(
                sub.mean_abs_physical_contribution_shift_from_original.mean()
            ),
            "mean_process_contribution_shift":float(
                sub.mean_abs_process_contribution_shift_from_original.mean()
            ),
            "mean_process_magnitude_share":float(
                sub.mean_fraction_signed_abs_logit_from_process.mean()
            ),
            "mean_risk_shift":float(sub.mean_abs_risk_shift_from_original.mean()),
            "max_logit_error":float(sub.max_logit_additivity_error.max()),
        })
    pd.DataFrame(aggregate).to_csv(args.out_dir/"dual_process_v7_summary.csv",index=False)
    status={
        "status":"EXPLORATORY_DUAL_ONTOLOGY_ICU_BOTTLENECK",
        "prior_strict_rank_validation":"ABSTAIN_CLINICAL_K_UNSTABLE",
        "cannot_claim_n_physiological_factors":True,
        "train_only_spectral_computational_slots":capacity,
        "number_physical_sensors":len(selected),
        "number_separately_identified_sampling_process_slots":int(
            cfg["dual_head"]["n_observation_process_concepts"]
        ),
        "sampling_policy_mask_never_as_unlabeled_physiology":True,
        "actual_measurement_process_mask_preserved_during_artificial_denoising":True,
        "no_hidden_classifier_bypass":True,
        "train_patient_count":2400,
        "historic_validation_development_only":True,
        "calibration_and_test_sealed":True,
        "medical_concept_labels_provided":False,
        "no_clinical_semantic_validation":True,
        "no_patient_level_predictions_saved":True,
        "results":aggregate,
        "independent_retraining_concept_family_diagnostics":[
            {
                "concept_family":str(family),
                "three_overlapping_pairs":len(frame),
                "mean_case_specific_spearman_excess":(
                    float(frame["excess_over_unmatched_episode_null_spearman"].dropna().mean())
                    if frame["excess_over_unmatched_episode_null_spearman"].notna().any()
                    else None
                ),
                "mean_case_specific_sign_excess":(
                    float(frame["case_specific_excess_sign_agreement"].dropna().mean())
                    if frame["case_specific_excess_sign_agreement"].notna().any()
                    else None
                ),
                "mean_operator_definition_similarity":float(frame["definition_cosine_normalized"].mean()),
                "not_independent_pairwise_trials":True,
            }
            for family,frame in pair_df.groupby("concept_family")
        ],
        "remaining_limitations":[
            "only one ICU challenge source and three independently initialized models",
            "case-dependent measurement intensity can reflect unmeasured severity or clinician decisions",
            "patient-level observation process is not a causal intervention or pathology definition",
            "same historical validation split used during iterative exploratory research",
            "external medical ground truth and prospectively frozen evaluation still unavailable",
        ],
    }
    (args.out_dir/"STATUS.json").write_text(
        json.dumps(status,indent=2,allow_nan=False),encoding="utf-8"
    )
    print(json.dumps(aggregate,indent=2,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
