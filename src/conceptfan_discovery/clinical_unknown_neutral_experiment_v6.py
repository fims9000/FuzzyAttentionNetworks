"""Exploratory *real ICU* test of whether absent measurements act as evidence.

This DOES NOT interpret the concept slots as medical pathologies:
the previously verified clinical dimension gate ABSTAINED. Accordingly
we name K only a train-derived computational concept BUDGET, capped at
a frozen maximum; no physiological factor count is asserted.

Use TRAIN data and outcomes to fit all three concept-mediated models.
The old validation subset is used for EXPLORATORY comparison; old
calibration and test cases remain sealed. Clinical diagnoses and
the five manually engineered proxy concepts never supervise slots.
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

from .data import DiscoveryData, DiscoverySplit
from .evaluation import (
    _episode_excess_agreement,compare_independent_discoveries,
    concept_passports,evaluate_discovery,
)
from .lagged_rank import fit_lagged_predictive_rank
from .observation_regimes_v5 import apply_measurement_environment
from .physionet import prepare_physionet_discovery
from .pilot_temporal_sensors_v5 import choose_common_temporal_sensors
from .selection import split_validation_for_cardinality
from .unknown_neutral_experiment_v6 import fit_one,sign_excess_same_patient


def subset_discovery_split(split:DiscoverySplit,channels:np.ndarray)->DiscoverySplit:
    return DiscoverySplit(
        x=split.x[:,:,channels].copy(),
        mask=split.mask[:,:,channels].copy(),
        future=split.future[:,:,channels].copy(),
        future_mask=split.future_mask[:,:,channels].copy(),
        y=split.y.copy(),
        oracle_states=None,
        episode_ids=split.episode_ids.copy(),
    )


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--raw-zip",type=Path,required=True)
    p.add_argument("--outcomes",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    if args.out_dir.exists():
        p.error("Refusing to overwrite existing clinical model results")
    cfg=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if (not cfg["protocol"]["test_and_calibration_sealed"]
        or not cfg["protocol"]["not_clinical_concept_discovery"]
        or cfg["protocol"]["prior_strict_clinical_rank_gate"]!="ABSTAIN_CLINICAL_K_UNSTABLE"):
        p.error("Do not suppress known clinical rank nonidentifiability")
    if set(cfg["training"]["methods"])!={"multiplicative_original",
                                         "unknown_neutral",
                                         "unknown_neutral_consistency"}:
        p.error("Complete three-arm comparison is required")
    with tempfile.TemporaryDirectory(prefix="pd_v6_clinical_split_") as tmp:
        full=prepare_physionet_discovery(
            args.raw_zip,args.outcomes,Path(tmp),
            split_seed=int(cfg["data"]["split_seed"]),
            input_window=int(cfg["data"]["input_window"]),
            forecast_horizon=int(cfg["data"]["forecast_horizon"]),
            minimum_train_patient_coverage=float(cfg["data"]["minimum_train_patient_coverage"]),
        )
    x=full.train.x
    mask=full.train.mask
    n=len(x)
    if n!=2400 or len(full.validation.y)!=600:
        raise ValueError("Official train/validation protocol altered")
    rng=np.random.default_rng(int(cfg["data"]["pilot_selection_seed"]))
    order=rng.permutation(n)
    pilot_ids=order[:int(cfg["data"]["pilot_train_patients"])]
    inference_ids=order[len(pilot_ids):]
    if len(pilot_ids)!=600 or len(inference_ids)!=1800:
        raise ValueError("Published 600/1800 independent train-only mask protocol changed")
    names=full.receipt["channel_names"]
    chosen=choose_common_temporal_sensors(
        mask[pilot_ids],names,
        lags=tuple(cfg["data"]["pilot_sensor_lags"]),
        minimum_supported_strata=int(cfg["data"]["pilot_min_time_strata"]),
        minimum_observed_pairs_per_stratum=int(cfg["data"]["pilot_min_hourly_pairs"]),
        minimum_expected_null_pairs_per_stratum=float(
            cfg["data"]["pilot_min_expected_null_pairs"]),
        worst_additional_observation_retention=float(
            cfg["data"]["pilot_worst_extra_thinning"]),
    )
    if chosen["status"]!="PILOT_SENSORS_ELIGIBLE":
        raise RuntimeError("Clinical pilot does not support any verifiable multichannel concept dictionary")
    selected=np.asarray(chosen["selected_indices"],dtype=int)
    rank_cfg=cfg["rank"]
    rank=fit_lagged_predictive_rank(
        x[inference_ids][:,:,selected],mask[inference_ids][:,:,selected],
        lags=tuple(rank_cfg["lags"]),
        permutations=int(rank_cfg["permutations"]),
        patient_bootstraps=int(rank_cfg["patient_bootstraps"]),
        alpha=float(rank_cfg["alpha"]),
        seed=int(rank_cfg["seed"]),
    )
    if rank.rank<2:
        raise RuntimeError("Candidate computational concepts cannot be initialized from observed temporal rank")
    capacity=int(min(rank.rank,int(rank_cfg["max_computational_concept_slots"])))
    basis=rank.left_observable_loadings[:,:capacity]
    train=subset_discovery_split(full.train,selected)
    validation=subset_discovery_split(full.validation,selected)
    receipt={
        **full.receipt,
        "channel_names":chosen["selected_names"],
        "n_channels":len(selected),
        "manual_five_proxy_concept_targets_used":False,
    }
    clipped=DiscoveryData(
        train,validation,receipt,full.medians[selected],full.scales[selected]
    )
    _,outer=split_validation_for_cardinality(
        validation,random_seed=int(cfg["evaluation"]["reported_validation_subset_seed"]),
    )
    if np.unique(outer.y).size!=2:
        raise ValueError("Both clinical label classes required on evaluation subset")
    torch.set_num_threads(int(cfg["training"]["cpu_threads"]))
    args.out_dir.mkdir(parents=True,exist_ok=False)
    cases=[]
    stress=[]
    crossfit=[]
    by_method={}
    for method in cfg["training"]["methods"]:
        by_method[method]={}
        for init_seed in cfg["training"]["initialization_seeds"]:
            init=int(init_seed)
            model,training=fit_one(method,init,clipped,cfg,basis)
            out=evaluate_discovery(model,outer)
            blank_input=torch.zeros_like(torch.from_numpy(outer.x[:8]))
            with torch.no_grad():
                out_blank=model(blank_input,torch.zeros_like(blank_input))
                blank_direct=float(out_blank.contributions.abs().max())
            cases.append({
                "method":method,
                "initialization_seed":init,
                "candidate_concept_slots":capacity,
                "pilot_observable_channels":len(selected),
                "mean_outer_validation_auprc":float(out.metrics["auprc"]),
                "outer_validation_auroc":float(out.metrics["auroc"]),
                "outer_validation_future_rmse":float(out.metrics["future_observed_rmse"]),
                "zero_measurement_max_abs_contribution":blank_direct,
                "additive_logit_error":float(out.metrics["max_additive_error"]),
                "patient_specific_clinical_semantics":False,
                "candidate_K_verified_as_physio_dimension":False,
                **training,
            })
            by_method[method][init]=out
            for scenario_i,scenario in enumerate(cfg["evaluation"]["scenarios"]):
                vx,vm,obs=apply_measurement_environment(
                    outer.x,outer.mask,
                    environment=scenario,
                    seed=int(cfg["evaluation"]["measurement_mask_seed"])+scenario_i
                )
                changed=DiscoverySplit(
                    x=vx,mask=vm,
                    future=outer.future,
                    future_mask=outer.future_mask,
                    y=outer.y,oracle_states=None,
                    episode_ids=outer.episode_ids,
                )
                new=evaluate_discovery(model,changed)
                _,rho_excess,_,top1_excess=_episode_excess_agreement(
                    out.contributions,new.contributions
                )
                stress.append({
                    "method":method,
                    "initialization_seed":init,
                    "scenario":scenario,
                    "retained_original_measurement_fraction":obs["relative_retained_observations"],
                    "scenario_auprc":float(new.metrics["auprc"]),
                    "delta_auprc":float(new.metrics["auprc"]-out.metrics["auprc"]),
                    "mean_abs_predicted_risk_change":float(
                        np.abs(new.probabilities-out.probabilities).mean()
                    ),
                    "mean_abs_direct_contribution_change":float(
                        np.abs(new.contributions-out.contributions).sum(axis=1).mean()
                    ),
                    "paired_episode_concept_ranking_excess":float(rho_excess),
                    "paired_episode_sign_excess":
                        sign_excess_same_patient(out.contributions,new.contributions),
                    "original_validation_labels_used_for_model_selection":False,
                    "old_test_or_calibration_queried":False,
                })
            if init==int(cfg["training"]["initialization_seeds"][0]):
                passport=concept_passports(
                    out,model,outer,clipped.medians,clipped.scales,
                    include_examples=False,
                )
                (args.out_dir/f"clinical_nonsemantic_passports_{method}.json").write_text(
                    json.dumps({
                        "method":method,
                        "candidate_not_true_clinical_concept_K":capacity,
                        "training_only_measured_operator_labels":chosen["selected_names"],
                        "concepts_have_no_verified_medical_name":True,
                        "model_contributions_used_in_fitting":True,
                        "individual_patient_examples_published":False,
                        "nonsemantic_auditable_concept_passports":passport,
                    },indent=2,allow_nan=False),encoding="utf-8",
                )
            print(method,init,
                  "AUPRC",round(out.metrics["auprc"],4),
                  "blank_direct_effect",round(blank_direct,7),
                  flush=True)
        seeds=sorted(by_method[method])
        for i,a in enumerate(seeds):
            for b in seeds[i+1:]:
                comparison=compare_independent_discoveries(
                    by_method[method][a],by_method[method][b]
                )
                crossfit.append({
                    "method":method,"seed_a":a,"seed_b":b,**comparison
                })
    df=pd.DataFrame(cases)
    g=pd.DataFrame(stress)
    h=pd.DataFrame(crossfit)
    df.to_csv(args.out_dir/"clinical_v6_per_run.csv",index=False)
    g.to_csv(args.out_dir/"clinical_v6_observation_stress.csv",index=False)
    h.to_csv(args.out_dir/"clinical_v6_between_seeds.csv",index=False)
    summary=[]
    for method,sub in df.groupby("method"):
        p=g[g.method==method]
        m=p[p.scenario=="time_channel_mar"]
        between=h[h.method==method]
        rho=between["excess_over_unmatched_episode_null_spearman"].dropna()
        summary.append({
            "method":method,
            "trained_independent_initializations":len(sub),
            "candidate_K_not_true_clinical_K":capacity,
            "mean_outer_validation_auprc":float(sub.mean_outer_validation_auprc.mean()),
            "mean_outer_auroc":float(sub.outer_validation_auroc.mean()),
            "mean_blank_direct_effect":float(sub.zero_measurement_max_abs_contribution.mean()),
            "mean_time_mar_risk_shift":float(m.mean_abs_predicted_risk_change.mean()),
            "mean_time_mar_contribution_shift":float(m.mean_abs_direct_contribution_change.mean()),
            "mean_crossseed_episode_spearman_excess":
                float(rho.mean()) if len(rho) else None,
            "no_clinical_concept_validity_claim":True,
        })
    pd.DataFrame(summary).to_csv(args.out_dir/"clinical_v6_summary.csv",index=False)
    status={
        "protocol":cfg["protocol"]["name"],
        "research_status":"EXPLORATORY_PREDICTION_AND_MISSINGNESS_NOT_CLINICAL_ONTOLOGY",
        "earlier_clinical_identifiability_gate":"ABSTAIN_CLINICAL_K_UNSTABLE",
        "auto_rank_from_heldout_train_1800":int(rank.rank),
        "candidate_computational_slots_capped":capacity,
        "candidate_slots_dont_equal_verified_medical_concepts":True,
        "pilot_sensor_selection":chosen,
        "official_split_counts":{"train":2400,"validation":600,
                                 "calibration_sealed":400,"test_sealed":600},
        "only_original_train_used_for_training":True,
        "only_original_validation_used_for_development_scoring":True,
        "historical_test_or_calibration_labels_accessed":False,
        "validation_repeatedly_used_in_program_not_clean_confirmation":True,
        "no_human_concept_supervision":True,
        "no_patient_level_export":True,
        "study_results":summary,
        "next_required_evidence":"external clinically annotated or prospectively held-out independent hospital and a stable identifiable missingness-aware dynamic basis",
    }
    (args.out_dir/"STATUS.json").write_text(
        json.dumps(status,indent=2,allow_nan=False),encoding="utf-8"
    )
    print(json.dumps(summary,indent=2,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
