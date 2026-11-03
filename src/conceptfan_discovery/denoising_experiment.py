"""Self-supervised PD-ConceptFAN v2: masked observed-channel reconstruction.

Frozen synthetic trials vary actual hidden dynamics in 3/5/7 dimensions.
Only observed signals and class labels are fed to training. Latent truth is
available only to audit recovered factors after each model is frozen.
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

from .evaluation import evaluate_discovery, oracle_correspondence
from .model_v2 import DenoisingPredictiveConceptFAN, masked_observation_view, v2_loss
from .selection import split_validation_for_cardinality
from .selection_v2 import select_v2_cardinality
from .variable_factor_data import generate_variable_factor_data


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def fit_one(mode: str, seed: int, data, cfg: dict) -> DenoisingPredictiveConceptFAN:
    fix_seed(seed)
    t=cfg["training"]
    model=DenoisingPredictiveConceptFAN(
        n_slots=int(cfg["architecture"]["n_candidate_slots"]),
        n_channels=data.train.x.shape[-1],
        channels=tuple(data.receipt["channel_names"]),
        forecast_horizon=data.train.future.shape[1],
        min_width=float(cfg["architecture"]["min_concept_width"]),
    )
    optimizer=torch.optim.AdamW(
        model.parameters(),lr=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
    )
    n_pos=float(data.train.y.sum())
    weight=torch.tensor((len(data.train.y)-n_pos)/max(n_pos,1.),dtype=torch.float32)
    loader=data.train.loader(int(t["batch_size"]),True,int(t["order_seed"]))
    generator=torch.Generator().manual_seed(seed+51019)
    for epoch in range(int(t["epochs"])):
        model.train()
        for x,mask,future,future_mask,y in loader:
            optimizer.zero_grad(set_to_none=True)
            if mode=="full":
                corrupted,visible,hidden=masked_observation_view(
                    x,mask,fraction=float(t["mask_fraction"]),generator=generator
                )
            else:
                corrupted,visible=x,mask
                hidden=torch.zeros_like(mask)
            output=model(corrupted,visible)
            loss,parts=v2_loss(
                model,output,y,future,future_mask,x,hidden,
                mode=mode,pos_weight=weight,loss_weights=t["loss_weights"],
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {mode} epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),float(t["gradient_clip"]))
            optimizer.step()
    model.eval()
    return model


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    if args.out_dir.exists():
        p.error("Refusing to overwrite frozen experiment")
    cfg=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not cfg["protocol"]["no_test_evaluation"] or set(cfg["training"]["modes"])!={"task_only","full_without_denoising","full"}:
        p.error("Protocol ablation or test-seal violation")
    torch.set_num_threads(int(cfg["training"]["cpu_threads"]))
    args.out_dir.mkdir(parents=True,exist_ok=False)
    rows=[]
    for true_k in cfg["data"]["n_true_factors"]:
        ds=generate_variable_factor_data(
            n_true_factors=int(true_k),
            seed=int(cfg["data"]["generator_base_seed"])+int(true_k),
            split_seed=int(cfg["data"]["split_seed"])+int(true_k),
            n_samples=int(cfg["data"]["n_samples"]),
            n_channels=int(cfg["data"]["n_channels"]),
            input_steps=int(cfg["data"]["input_steps"]),
            horizon=int(cfg["data"]["horizon"]),
        )
        selecting,auditing=split_validation_for_cardinality(
            ds.validation,random_seed=int(cfg["selection"]["random_seed"])
        )
        for mode,seed in itertools.product(cfg["training"]["modes"],cfg["training"]["init_seeds"]):
            model=fit_one(mode,int(seed),ds,cfg)
            model.eval()
            selector=select_v2_cardinality(
                model,selecting,auditing,
                mode=mode,
                seed=int(cfg["selection"]["masking_seed"]),
                max_auprc_drop=float(cfg["selection"]["auprc_tolerance"]),
                max_nll_increase=float(cfg["selection"]["nll_tolerance"]),
                max_future_rmse_increase=float(cfg["selection"]["forecast_tolerance"]),
                max_masked_rmse_increase=float(cfg["selection"]["denoising_tolerance"]),
            )
            before=selector.path[0]["selection"]
            outer=selector.audit_metrics
            with torch.no_grad():
                e=evaluate_discovery(model,auditing,
                    selected_slots=torch.from_numpy(selector.selected_mask))
            oracle=oracle_correspondence(e,auditing.oracle_states)
            rows.append({
                "true_k_posttraining_audit_only":int(true_k),"mode":mode,
                "init_seed":int(seed),
                "k_selected":int(selector.selected_mask.sum()),
                "factor_coverage":float(oracle["oracle_factor_coverage"]),
                "oracle_coverage_adjusted_abs_corr":float(oracle["oracle_coverage_adjusted_abs_corr"]),
                "outer_auprc":float(outer["auprc"]),
                "outer_nll":float(outer["nll"]),
                "outer_masked_rmse":float(outer["masked_reconstruction_rmse"]),
                "outer_future_rmse":float(outer["future_rmse"]),
                "selection_full_masked_rmse":float(before["masked_reconstruction_rmse"]),
                "prediction_additivity_error":float(e.metrics["max_additive_error"]),
                "true_latent_labels_in_training":False,
                "test_evaluated":False,
            })
            print("trueK",true_k,"method",mode,"seed",seed,"selectedK",rows[-1]["k_selected"],
                  "coverage",round(rows[-1]["factor_coverage"],3),
                  "masked_rmse",round(rows[-1]["outer_masked_rmse"],3),flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(args.out_dir/"denoising_v2_per_run.csv",index=False)
    summary=[]
    for (true_k,mode),sub in frame.groupby(["true_k_posttraining_audit_only","mode"]):
        summary.append({
            "true_k_oracle_only":int(true_k),"method":mode,"runs":len(sub),
            "mean_selected_k":float(sub.k_selected.mean()),
            "mean_abs_k_error":float(np.abs(sub.k_selected-true_k).mean()),
            "mean_factor_coverage":float(sub.factor_coverage.mean()),
            "mean_coverage_adjusted_abs_corr":float(sub.oracle_coverage_adjusted_abs_corr.mean()),
            "mean_auprc":float(sub.outer_auprc.mean()),
            "mean_masked_reconstruction_rmse":float(sub.outer_masked_rmse.mean()),
            "mean_future_rmse":float(sub.outer_future_rmse.mean()),
            "max_additivity_error":float(sub.prediction_additivity_error.max()),
        })
    pd.DataFrame(summary).to_csv(args.out_dir/"denoising_v2_summary.csv",index=False)
    (args.out_dir/"STATUS.json").write_text(json.dumps({
        "status":"EXPLORATORY_V2_FACTORS_REPLICATION_COMPLETE",
        "learned_concepts_are_training_time_native":True,
        "concept_oracle_supervision":False,
        "masked_values_cannot_enter_their_own_reconstruction_inputs":True,
        "selection_and_audit_are_disjoint":True,
        "test_ever_used":False,
        "not_claimed": [
            "globally optimal concept cardinality",
            "unique causal factors or intervention effects",
            "statistical significance across repeated overlapping model pairs",
            "clinical model improvement",
        ],
        "results":summary,
    },indent=2),encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


if __name__=="__main__":
    main()
