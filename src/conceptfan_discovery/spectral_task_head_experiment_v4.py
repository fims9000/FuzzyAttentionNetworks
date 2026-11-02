"""Explore whether train-only canonical signed task weights stabilize concepts.

Previously seen v1/v2/v3 results are NOT overwritten. Fresh synthetic
generators are used. All head references are fitted using TRAIN outcomes
and unsupervised TRAIN-observable concepts. No concept annotation, oracle
latent state, validation label tuning or test examples enter fitting.
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

from .evaluation import compare_independent_discoveries, evaluate_discovery, oracle_correspondence
from .lagged_rank import fit_lagged_predictive_rank
from .model_v2 import masked_observation_view, v2_loss
from .selection import split_validation_for_cardinality
from .spectral_concepts_v3 import SpectralConceptFAN
from .spectral_task_head import (
    fit_canonical_task_head, initialize_head, task_head_alignment_penalty,
)
from .variable_factor_data import generate_variable_factor_data


def fit_one(
    method: str, seed: int, dataset, cfg: dict,
    loading: np.ndarray, head,
)->SpectralConceptFAN:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    t=cfg["training"]
    model=SpectralConceptFAN(
        observable_loadings=loading,
        channels=tuple(dataset.receipt["channel_names"]),
        forecast_horizon=dataset.train.future.shape[1],
        freeze_definitions=method=="task_head_fixed",
    )
    if method!="spectral_finetuned":
        initialize_head(model,head,freeze_head=method=="task_head_fixed")
    parameters=[v for v in model.parameters() if v.requires_grad]
    optimizer=torch.optim.AdamW(
        parameters,lr=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
    )
    positives=float(dataset.train.y.sum())
    imbalance=torch.tensor((len(dataset.train.y)-positives)/max(positives,1.),dtype=torch.float32)
    loader=dataset.train.loader(
        int(t["batch_size"]),True,int(t["order_seed"])
    )
    generator=torch.Generator().manual_seed(seed+51019)
    for epoch in range(int(t["epochs"])):
        model.train()
        for x,mask,future,future_mask,y in loader:
            optimizer.zero_grad(set_to_none=True)
            damaged,visible,withheld=masked_observation_view(
                x,mask,fraction=float(t["mask_fraction"]),generator=generator
            )
            out=model(damaged,visible)
            loss,_=v2_loss(
                model,out,y,future,future_mask,x,withheld,
                mode="full",pos_weight=imbalance,
                loss_weights=t["loss_weights"],
            )
            prior=model.spectral_definition_penalty()
            loss=loss+float(t["feature_prior_weight"])*prior["operator_displacement"]
            loss=loss+float(t["feature_orthogonality_weight"])*prior["observable_axis_nonorthogonality"]
            if method=="task_head_prior":
                loss=loss+float(t["head_prior_weight"])*task_head_alignment_penalty(model,head)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite "+method+" seed="+str(seed))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters,float(t["gradient_clip"]),
            )
            optimizer.step()
    model.eval()
    return model


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True,type=Path)
    p.add_argument("--out-dir",required=True,type=Path)
    args=p.parse_args()
    if args.out_dir.exists():
        p.error("Refusing to overwrite a previous experiment")
    cfg=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not cfg["protocol"]["no_test_evaluation"] or not cfg["evaluation"]["no_test"]:
        p.error("Test split must remain sealed")
    if set(cfg["training"]["methods"])!={"spectral_finetuned","task_head_prior","task_head_fixed"}:
        p.error("Three-arm methodological comparison required")
    torch.set_num_threads(int(cfg["training"]["cpu_threads"]))
    args.out_dir.mkdir(parents=True,exist_ok=False)
    runs,pairs=[],[]
    head_receipts={}
    for planted_k in cfg["data"]["true_factors_external_audit_only"]:
        data=generate_variable_factor_data(
            n_true_factors=int(planted_k),
            seed=int(cfg["data"]["generator_seed_base"])+int(planted_k),
            split_seed=int(cfg["data"]["split_seed_base"])+int(planted_k),
            n_samples=int(cfg["data"]["n_samples"]),
            n_channels=int(cfg["data"]["n_channels"]),
            input_steps=int(cfg["data"]["input_steps"]),
            horizon=int(cfg["data"]["horizon"]),
        )
        sc=cfg["rank"]
        rank=fit_lagged_predictive_rank(
            data.train.x,data.train.mask,
            lags=tuple(sc["lags"]),
            permutations=int(sc["permutations"]),
            patient_bootstraps=int(sc["patient_bootstraps"]),
            alpha=float(sc["alpha"]),
            seed=int(sc["seed"])+int(planted_k),
        )
        if rank.rank<2:
            raise RuntimeError("Detectable dynamic rank below two; no manual K fallback allowed")
        basis=rank.left_observable_loadings[:,:rank.rank]
        head=fit_canonical_task_head(
            data.train.x,data.train.mask,data.train.y,
            observable_basis=basis,
            channels=tuple(data.receipt["channel_names"]),
            forecast_horizon=data.train.future.shape[1],
            regularization_c=float(cfg["training"]["logistic_head_c"]),
            max_beta=float(cfg["training"]["signed_beta_max"]),
        )
        head_receipts[str(planted_k)]={
            "train_only_rank":rank.to_json(),
            "canonical_train_only_head":head.to_json(),
            "number_initialization_seeds":len(cfg["training"]["initialization_seeds"]),
        }
        _,audit=split_validation_for_cardinality(
            data.validation,
            random_seed=int(cfg["evaluation"]["inner_split_seed"]),
            selection_fraction=float(cfg["evaluation"]["outer_validation_fraction"]),
        )
        by_method={}
        for method in cfg["training"]["methods"]:
            by_method[method]={}
            for seed in cfg["training"]["initialization_seeds"]:
                initial=int(seed)
                model=fit_one(method,initial,data,cfg,basis,head)
                with torch.no_grad():
                    e=evaluate_discovery(model,audit)
                oracle=oracle_correspondence(e,audit.oracle_states)
                rec={
                    "true_k_external_audit_only":int(planted_k),
                    "train_observed_k":int(rank.rank),
                    "method":method,"init_seed":initial,
                    "outer_auprc":float(e.metrics["auprc"]),
                    "future_observed_rmse":float(e.metrics["future_observed_rmse"]),
                    "matched_oracle_factor_coverage":float(oracle["oracle_factor_coverage"]),
                    "oracle_coverage_adjusted_abs_corr":float(oracle["oracle_coverage_adjusted_abs_corr"]),
                    "max_additive_logit_error":float(e.metrics["max_additive_error"]),
                    "signed_beta":json.dumps(e.decision_weights.tolist()),
                    "signed_beta_cosine_vs_train_head":float(
                        e.decision_weights@head.signed_coefficient/
                        max(np.linalg.norm(e.decision_weights)*np.linalg.norm(head.signed_coefficient),1e-12)
                    ),
                    "target_concept_annotations_used":False,
                    "historical_test_used":False,
                }
                runs.append(rec)
                by_method[method][initial]=e
                print("K",planted_k,"method",method,"seed",initial,
                      "AUPRC",round(rec["outer_auprc"],4),
                      "beta_head_cos",round(rec["signed_beta_cosine_vs_train_head"],3),flush=True)
            for a,b in itertools.combinations(sorted(by_method[method]),2):
                metrics=compare_independent_discoveries(
                    by_method[method][a],by_method[method][b]
                )
                pairs.append({
                    "true_k_external_audit_only":int(planted_k),
                    "method":method,"init_a":a,"init_b":b,
                    **metrics,
                })
    per_run=pd.DataFrame(runs)
    pair_df=pd.DataFrame(pairs)
    per_run.to_csv(args.out_dir/"task_head_v4_per_run.csv",index=False)
    pair_df.to_csv(args.out_dir/"task_head_v4_pairwise.csv",index=False)
    summary=[]
    for (truth,method),group in per_run.groupby(["true_k_external_audit_only","method"]):
        comparison=pair_df.loc[
            pair_df.true_k_external_audit_only.eq(truth)
            &pair_df.method.eq(method)
        ]
        def mean_col(name:str)->float|None:
            finite=comparison[name].replace([np.inf,-np.inf],np.nan).dropna()
            return float(finite.mean()) if len(finite) else None
        summary.append({
            "true_k_oracle_audit_only":int(truth),
            "method":method,
            "detected_train_k":int(group.train_observed_k.iloc[0]),
            "runs":len(group),
            "outer_auprc_mean":float(group.outer_auprc.mean()),
            "outer_auprc_std":float(group.outer_auprc.std(ddof=1)),
            "oracle_coverage_adjusted_corr":float(group.oracle_coverage_adjusted_abs_corr.mean()),
            "mean_beta_cosine_to_train_canonical":float(group.signed_beta_cosine_vs_train_head.mean()),
            "case_specific_spearman_excess":mean_col("excess_over_unmatched_episode_null_spearman"),
            "case_specific_sign_agreement_excess":mean_col("case_specific_excess_sign_agreement"),
            "chance_corrected_top1_excess":mean_col("excess_over_unmatched_episode_null_top1"),
            "mean_definition_similarity":mean_col("definition_cosine_normalized"),
        })
    pd.DataFrame(summary).to_csv(args.out_dir/"task_head_v4_summary.csv",index=False)
    (args.out_dir/"train_only_task_head_receipts.json").write_text(json.dumps(
        head_receipts,indent=2,allow_nan=False),encoding="utf-8")
    (args.out_dir/"STATUS.json").write_text(json.dumps({
        "status":"EXPLORATORY_TASK_HEAD_PRIOR_STUDY_COMPLETE",
        "test_never_evaluated":True,
        "train_only_rank_and_label_fit":True,
        "human_named_concept_targets":False,
        "synthetic_latent_oracle_used_only_to_evaluate":True,
        "fixed_head_by_design_does_not_prove_semantic_stability":True,
        "claim_limit":"five seeds per method, new synthetic generators, not clinical significant effects",
        "summary":summary,
    },indent=2,allow_nan=False),encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


if __name__=="__main__":
    main()
