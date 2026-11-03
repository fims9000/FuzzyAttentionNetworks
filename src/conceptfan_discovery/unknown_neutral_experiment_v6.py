"""V6: does neutral UNKNOWN + dual-view regularization stabilize native concepts?

Only TRAIN observations are used to estimate K and construct signed sensor
operators. TRAIN outcomes influence ONLY the jointly trained task objective.
Synthetic planted latent states are never training labels or concept targets.

Four fresh-seed replicates per arm, three dynamic factor counts, 36 models:
- multiplicative_original (v3 trained through real decision bottleneck);
- unknown_neutral (missing concept state has exactly zero direct task effect);
- unknown_neutral_consistency (same plus training-time dual observation
  consistency, evaluated on disjoint outer validation).

The same patient cohort is viewed under four extra missingness policies.
These are stress tests, not independent site-level replications.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .data import DiscoverySplit
from .evaluation import (
    _episode_excess_agreement, compare_independent_discoveries,
    evaluate_discovery, oracle_correspondence,
)
from .lagged_rank import fit_lagged_predictive_rank
from .model_v2 import masked_observation_view,v2_loss
from .observation_regimes_v5 import apply_measurement_environment
from .selection import split_validation_for_cardinality
from .spectral_concepts_v3 import SpectralConceptFAN
from .unknown_neutral_v6 import (
    UnknownNeutralSpectralConceptFAN,cross_view_concept_consistency,
)
from .variable_factor_data import generate_variable_factor_data


def fit_one(
    method:str,init_seed:int,data,cfg:dict,
    train_observable_basis:np.ndarray,
):
    t=cfg["training"]
    random.seed(int(init_seed))
    np.random.seed(int(init_seed))
    torch.manual_seed(int(init_seed))
    torch.use_deterministic_algorithms(True)
    clazz=(SpectralConceptFAN if method=="multiplicative_original"
           else UnknownNeutralSpectralConceptFAN)
    model=clazz(
        observable_loadings=train_observable_basis,
        channels=tuple(data.receipt["channel_names"]),
        forecast_horizon=data.train.future.shape[1],
        freeze_definitions=False,
    )
    optimizer=torch.optim.AdamW(
        model.parameters(),lr=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
    )
    positives=float(data.train.y.sum())
    pos_weight=torch.tensor(
        (len(data.train.y)-positives)/max(positives,1.),
        dtype=torch.float32,
    )
    loader=data.train.loader(
        int(t["batch_size"]),True,int(t["batch_order_seed"]),
    )
    main_rng=torch.Generator().manual_seed(int(init_seed)+71021)
    other_rng=torch.Generator().manual_seed(int(init_seed)+71031)
    accumulated_loss=0.
    accumulated_consistency=0.
    steps=0
    for epoch in range(int(t["epochs"])):
        model.train()
        for x,mask,future,future_mask,y in loader:
            optimizer.zero_grad(set_to_none=True)
            first,first_mask,withheld=masked_observation_view(
                x,mask,fraction=float(t["main_mask_fraction"]),
                generator=main_rng,
            )
            o1=model(first,first_mask)
            objective,parts=v2_loss(
                model,o1,y,future,future_mask,x,withheld,
                mode="full",pos_weight=pos_weight,
                loss_weights=t["loss_weights"],
            )
            anchor=model.spectral_definition_penalty()
            objective=(
                objective
                +float(t["spectral_anchor_weight"])*anchor["operator_displacement"]
                +float(t["spectral_axis_independence_weight"])*anchor["observable_axis_nonorthogonality"]
            )
            consistency=objective.new_zeros(())
            if method=="unknown_neutral_consistency":
                second,second_mask,_=masked_observation_view(
                    x,mask,fraction=float(t["second_mask_fraction"]),
                    generator=other_rng,
                )
                o2=model(second,second_mask)
                consistency,confidence=cross_view_concept_consistency(
                    o1,o2,
                    minimum_joint_reliability=float(
                        t["consistency_min_joint_reliability"]
                    ),
                )
                objective=objective+float(t["consistency_weight"])*consistency
            if not torch.isfinite(objective):
                raise FloatingPointError(
                    f"Nonfinite {method} epoch={epoch} seed={init_seed}"
                )
            objective.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),float(t["gradient_clip"]),
            )
            optimizer.step()
            accumulated_loss+=float(objective.detach())
            accumulated_consistency+=float(consistency.detach())
            steps+=1
    model.eval()
    return model,{
        "mean_train_minibatch_objective":accumulated_loss/max(steps,1),
        "mean_dual_view_training_penalty":accumulated_consistency/max(steps,1),
        "training_minibatches":steps,
    }


def _split_with_observation_policy(
    split:DiscoverySplit, policy:str,seed:int,
)->tuple[DiscoverySplit,dict]:
    x,mask,receipt=apply_measurement_environment(
        split.x,split.mask,environment=policy,seed=int(seed),
    )
    return DiscoverySplit(
        x=x,mask=mask,
        future=split.future,future_mask=split.future_mask,y=split.y,
        oracle_states=split.oracle_states,
        episode_ids=split.episode_ids,
    ),receipt


def sign_excess_same_patient(l:np.ndarray,r:np.ndarray)->float:
    if l.shape!=r.shape:
        raise ValueError("Contributions must be from identical patient and concept coordinates")
    a,b=np.sign(l),np.sign(r)
    obs=float((a==b).mean())
    chance=float(np.mean(sum(
        (a==sign).mean(axis=0)*(b==sign).mean(axis=0)
        for sign in (-1,0,1)
    )))
    return obs-chance


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    if args.out_dir.exists():
        p.error("No overwriting previously audited experiments")
    cfg=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not cfg["protocol"]["no_test_use"] or not cfg["evaluation"]["all_risk_predictions_concept_mediated"]:
        p.error("Only sealed-test, direct concept bottleneck is permitted")
    methods=tuple(cfg["training"]["methods"])
    if set(methods)!={
        "multiplicative_original","unknown_neutral","unknown_neutral_consistency"
    }:
        p.error("All three prespecified architecture controls are required")
    assert set(cfg["evaluation"]["scenarios"])=={
        "original","mcar_thin","time_channel_mar","observed_value_mnar",
    }
    torch.set_num_threads(int(cfg["training"]["cpu_threads"]))
    args.out_dir.mkdir(parents=True,exist_ok=False)
    per_run,per_policy,pairwise=[],[],[]
    rank_provenance={}
    for true_k in cfg["data"]["planted_k_external_evaluation_only"]:
        ds=generate_variable_factor_data(
            n_true_factors=int(true_k),
            seed=int(cfg["data"]["generator_seed_base"])+int(true_k),
            split_seed=int(cfg["data"]["split_seed_base"])+int(true_k),
            n_samples=int(cfg["data"]["n_samples"]),
            n_channels=int(cfg["data"]["n_channels"]),
            input_steps=int(cfg["data"]["input_window"]),
            horizon=int(cfg["data"]["horizon"]),
        )
        c=cfg["rank"]
        rank=fit_lagged_predictive_rank(
            ds.train.x,ds.train.mask,
            lags=tuple(c["train_only_lags"]),
            permutations=int(c["permutations"]),
            patient_bootstraps=int(c["patient_bootstraps"]),
            alpha=float(c["alpha"]),
            seed=int(c["rank_seed"])+int(true_k),
        )
        if rank.rank<2:
            raise RuntimeError(
                f"No valid independently observable concept basis K={rank.rank}; refuse manual fallback"
            )
        basis=rank.left_observable_loadings[:,:rank.rank]
        rank_provenance[str(true_k)]=rank.to_json()
        _,audit=split_validation_for_cardinality(
            ds.validation,random_seed=int(cfg["evaluation"]["outer_validation_split_seed"]),
        )
        by_method={}
        for method in methods:
            by_method[method]={}
            for init_seed in cfg["training"]["initialization_seeds"]:
                init_seed=int(init_seed)
                model,training_metrics=fit_one(method,init_seed,ds,cfg,basis)
                clean=evaluate_discovery(model,audit)
                oracle=oracle_correspondence(clean,audit.oracle_states)
                with torch.no_grad():
                    # Completely unmeasured measurements cannot be mistaken
                    # for a signed clinical concept in the new variant.
                    blank=model(
                        torch.zeros_like(torch.from_numpy(audit.x[:5])),
                        torch.zeros_like(torch.from_numpy(audit.mask[:5])),
                    )
                    blank_mean_direct_effect=float(blank.contributions.abs().mean())
                per_run.append({
                    "hidden_k_external_evaluation_only":int(true_k),
                    "train_only_detected_k":int(rank.rank),
                    "method":method,
                    "seed":init_seed,
                    "outer_validation_original_auprc":float(clean.metrics["auprc"]),
                    "outer_validation_original_future_rmse":float(clean.metrics["future_observed_rmse"]),
                    "oracle_factor_coverage":float(oracle["oracle_factor_coverage"]),
                    "coverage_adjusted_latent_correspondence":
                        float(oracle["oracle_coverage_adjusted_abs_corr"]),
                    "zero_observation_mean_abs_direct_contribution":
                        blank_mean_direct_effect,
                    "exact_logit_additivity_error":float(clean.metrics["max_additive_error"]),
                    **training_metrics,
                })
                by_method[method][init_seed]=clean
                scenarios={}
                for scenario_i,scenario in enumerate(cfg["evaluation"]["scenarios"]):
                    corrupted,receipt=_split_with_observation_policy(
                        audit,scenario,seed=int(cfg["evaluation"]["mask_seed_base"])+
                        int(true_k)*101+scenario_i,
                    )
                    evidence=evaluate_discovery(model,corrupted)
                    raw,extra,top1,extra_top1=_episode_excess_agreement(
                        clean.contributions,evidence.contributions
                    )
                    per_policy.append({
                        "hidden_k_external_evaluation_only":int(true_k),
                        "method":method,
                        "seed":init_seed,
                        "scenario":scenario,
                        "retained_observation_fraction":receipt["relative_retained_observations"],
                        "scenario_depends_on_observed_value":receipt["selection_depends_on_current_observed_value"],
                        "outer_auprc":float(evidence.metrics["auprc"]),
                        "relative_auprc_to_undisturbed":
                            float(evidence.metrics["auprc"]-clean.metrics["auprc"]),
                        "mean_abs_risk_shift":float(np.abs(evidence.probabilities-clean.probabilities).mean()),
                        "mean_abs_total_concept_contribution_shift":float(
                            np.abs(evidence.contributions-clean.contributions).sum(axis=1).mean()
                        ),
                        "case_specific_spearman_excess_over_independent_patient_null":
                            float(extra),
                        "case_specific_sign_agreement_excess":
                            sign_excess_same_patient(clean.contributions,evidence.contributions),
                        "max_logit_decomposition_error":float(evidence.metrics["max_additive_error"]),
                        "no_patient_level_release":True,
                    })
                print(
                    "trueK",true_k,"detectedK",rank.rank,"method",method,
                    "seed",init_seed,"AUPRC",round(clean.metrics["auprc"],4),
                    "no_measurement_contribution",round(blank_mean_direct_effect,5),
                    flush=True,
                )
            for i,a in enumerate(sorted(by_method[method])):
                for b in sorted(by_method[method])[i+1:]:
                    pair=compare_independent_discoveries(
                        by_method[method][a],by_method[method][b]
                    )
                    pairwise.append({
                        "hidden_k_external_evaluation_only":int(true_k),
                        "method":method,"seed_a":a,"seed_b":b,**pair
                    })
    runs=pd.DataFrame(per_run)
    policies=pd.DataFrame(per_policy)
    between=pd.DataFrame(pairwise)
    runs.to_csv(args.out_dir/"unknown_neutral_v6_per_run.csv",index=False)
    policies.to_csv(args.out_dir/"unknown_neutral_v6_by_measurement.csv",index=False)
    between.to_csv(args.out_dir/"unknown_neutral_v6_between_seeds.csv",index=False)
    summary=[]
    for (truth,method),group in runs.groupby(
        ["hidden_k_external_evaluation_only","method"]
    ):
        for scenario,altered in policies[
            (policies.hidden_k_external_evaluation_only==truth)
            &(policies.method==method)
        ].groupby("scenario"):
            source=between[
                (between.hidden_k_external_evaluation_only==truth)
                &(between.method==method)
            ]["excess_over_unmatched_episode_null_spearman"].dropna()
            summary.append({
                "true_k_external_audit_only":int(truth),
                "method":method,
                "evaluation_policy":scenario,
                "number_initialization_seeds":len(group),
                "train_detected_k":int(group.train_only_detected_k.iloc[0]),
                "original_auprc":float(group.outer_validation_original_auprc.mean()),
                "scenario_auprc":float(altered.outer_auprc.mean()),
                "delta_auprc":float(altered.relative_auprc_to_undisturbed.mean()),
                "mean_abs_risk_shift":float(altered.mean_abs_risk_shift.mean()),
                "mean_abs_contribution_shift":
                    float(altered.mean_abs_total_concept_contribution_shift.mean()),
                "mean_case_specific_spearman_excess":
                    float(altered.case_specific_spearman_excess_over_independent_patient_null.mean()),
                "mean_case_specific_sign_excess":
                    float(altered.case_specific_sign_agreement_excess.mean()),
                "mean_blank_observation_direct_effect":
                    float(group.zero_observation_mean_abs_direct_contribution.mean()),
                "crossseed_case_specific_spearman_excess":
                    float(source.mean()) if len(source) else None,
                "oracle_coverage_adjusted_abs_corr":
                    float(group.coverage_adjusted_latent_correspondence.mean()),
            })
    pd.DataFrame(summary).to_csv(args.out_dir/"unknown_neutral_v6_summary.csv",index=False)
    (args.out_dir/"train_only_rank_source.json").write_text(
        json.dumps(rank_provenance,indent=2,allow_nan=False),encoding="utf-8",
    )
    status={
        "protocol":cfg["protocol"]["name"],
        "n_trained_models":len(runs),
        "no_manual_concept_labels":True,
        "no_planted_latent_labels_given_to_network":True,
        "rank_estimated_from_train_measurements_only":True,
        "validation_evaluation_not_test":True,
        "no_patient_level_data_published":True,
        "blank_observation_means_unknown_not_healthy":True,
        "original_and_thinned_views_same_patients_not_external_replications":True,
        "interpretation":"compare task accuracy and within-patient explanation reliability; no physiological causal claim",
        "limitations":[
            "nonlinear dynamics and sensor missingness can violate rank identifiability",
            "missing-not-at-random values are not identifiable from masks alone",
            "stability by enforcing a common definition may mask predictive underfitting",
            "single synthetic generator per hidden K and four neural initializations are exploratory",
            "valid clinical concept identity requires independent medical reference measurements",
        ],
        "results":summary,
    }
    (args.out_dir/"STATUS.json").write_text(
        json.dumps(status,indent=2,allow_nan=False),encoding="utf-8"
    )
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


if __name__=="__main__":
    main()
