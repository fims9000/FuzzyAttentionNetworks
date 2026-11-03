"""Falsification study: does learned concept count follow 3, 5 or 7 latent factors?

Ground truth is never used for concept training, cardinality selection or
matching between runs. It only appears in a separate post-fitting score.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .evaluation import compare_independent_discoveries, evaluate_discovery, oracle_correspondence
from .experiment import _run_one
from .model import PredictiveDynamicsConceptFAN
from .selection import select_concept_cardinality, split_validation_for_cardinality
from .variable_factor_data import generate_variable_factor_data


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True,type=Path)
    p.add_argument("--out-dir",required=True,type=Path)
    args=p.parse_args()
    if args.out_dir.exists():
        p.error("Never overwrite a completed research result")
    cfg=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not cfg["protocol"]["no_test_evaluation"] or set(cfg["training"]["modes"]) != {"task_only","task_future","full"}:
        p.error("Protocol rejected: test or ablation contract")
    torch.set_num_threads(int(cfg["training"]["cpu_threads"]))
    rows,pairs=[],[]
    args.out_dir.mkdir(parents=True,exist_ok=False)
    for k in cfg["data"]["n_true_factors"]:
        data=generate_variable_factor_data(
            n_true_factors=int(k),
            seed=int(cfg["data"]["generator_base_seed"])+int(k),
            split_seed=int(cfg["data"]["split_seed"])+int(k),
            n_samples=int(cfg["data"]["n_samples"]),
            n_channels=int(cfg["data"]["n_channels"]),
            input_steps=int(cfg["data"]["input_steps"]),
            horizon=int(cfg["data"]["horizon"]),
        )
        factor_cfg={
            "architecture":cfg["architecture"],
            "data":{"forecast_horizon":int(cfg["data"]["horizon"])},
            "training":cfg["training"],
        }
        root=args.out_dir/f"true_factors_{k}"
        root.mkdir()
        (root/"receipt.json").write_text(json.dumps(data.receipt,indent=2),encoding="utf-8")
        selecting,auditing=split_validation_for_cardinality(
            data.validation,random_seed=int(cfg["reporting"]["cardinality_selection_seed"]),
        )
        for method in cfg["training"]["modes"]:
            evidence={}
            for seed,order in itertools.product(cfg["training"]["init_seeds"],cfg["training"]["order_seeds"]):
                record,_=_run_one(factor_cfg,data,method,int(seed),int(order),root)
                checkpoint=torch.load(
                    root/"runs"/method/record["run_id"]/"model.pt",
                    map_location="cpu",weights_only=True,
                )
                model=PredictiveDynamicsConceptFAN(
                    n_slots=int(cfg["architecture"]["n_candidate_slots"]),
                    n_channels=len(checkpoint["channels"]),
                    channels=tuple(checkpoint["channels"]),
                    forecast_horizon=int(cfg["data"]["horizon"]),
                    min_width=float(cfg["architecture"]["min_concept_width"]),
                )
                model.load_state_dict(checkpoint["model_state_dict"],strict=True)
                model.eval()
                selected=select_concept_cardinality(
                    model,selecting,auditing,mode=method,
                    auprc_tolerance=float(cfg["reporting"]["auprc_tolerance"]),
                    nll_tolerance=float(cfg["reporting"]["nll_tolerance"]),
                    future_rmse_tolerance=float(cfg["reporting"]["future_rmse_tolerance"]),
                )
                with torch.no_grad():
                    ev=evaluate_discovery(model,auditing,
                        selected_slots=torch.from_numpy(selected.selected_mask))
                oracle=oracle_correspondence(ev,auditing.oracle_states)
                evidence[record["run_id"]]=ev
                rows.append({
                    "true_factors":int(k),"method":method,"run_id":record["run_id"],
                    "selected_k":int(selected.selected_mask.sum()),
                    "k_abs_error":int(abs(int(selected.selected_mask.sum())-int(k))),
                    "outer_auprc":ev.metrics["auprc"],
                    "outer_future_rmse":ev.metrics["future_observed_rmse"],
                    "oracle_matched":oracle["oracle_matched"],
                    "oracle_factor_coverage":oracle["oracle_factor_coverage"],
                    "oracle_coverage_adjusted_abs_corr":oracle["oracle_coverage_adjusted_abs_corr"],
                    "exact_additive_error":ev.metrics["max_additive_error"],
                })
                print("latent_K",k,"method",method,"init",seed,
                      "discovered_K",int(selected.selected_mask.sum()),
                      "oracle_coverage",round(oracle["oracle_factor_coverage"],3),flush=True)
            for left,right in itertools.combinations(sorted(evidence),2):
                pairs.append({
                    "true_factors":int(k),"method":method,"run_a":left,"run_b":right,
                    **compare_independent_discoveries(evidence[left],evidence[right]),
                })
    runs=pd.DataFrame(rows)
    comparison=pd.DataFrame(pairs)
    runs.to_csv(args.out_dir/"variable_true_k_runs.csv",index=False)
    comparison.to_csv(args.out_dir/"variable_true_k_pairwise.csv",index=False)
    aggregated=[]
    for (k,method),group in runs.groupby(["true_factors","method"]):
        pair_subset=comparison[
            comparison["true_factors"].eq(k)&comparison["method"].eq(method)
        ]
        finite=pair_subset["excess_over_unmatched_episode_null_spearman"].dropna()
        aggregated.append({
            "true_k_for_posthoc_oracle_only":int(k),
            "method":method,
            "runs":len(group),
            "selected_k_mean":float(group.selected_k.mean()),
            "selected_k_min":int(group.selected_k.min()),
            "selected_k_max":int(group.selected_k.max()),
            "mean_abs_k_error":float(group.k_abs_error.mean()),
            "mean_oracle_factor_coverage":float(group.oracle_factor_coverage.mean()),
            "mean_oracle_coverage_adjusted_corr":float(group.oracle_coverage_adjusted_abs_corr.mean()),
            "validation_auprc":float(group.outer_auprc.mean()),
            "validation_future_rmse":float(group.outer_future_rmse.mean()),
            "case_specific_excess_spearman":float(finite.mean()) if len(finite) else None,
        })
    pd.DataFrame(aggregated).to_csv(args.out_dir/"variable_factor_count_summary.csv",index=False)
    (args.out_dir/"RESEARCH_STATUS.json").write_text(json.dumps({
        "interpretation":"exploratory falsification of latent-factor recovery and variable concept cardinality",
        "true_factor_count_used_for_training":False,
        "test_ever_evaluated":False,
        "aggregate_runs":len(runs),
        "methods":list(cfg["training"]["modes"]),
        "experiments":aggregated,
        "warning":"Selected predictive K need not equal true physical-state dimension. Do not label matched slots as discovered clinical pathologies.",
    },indent=2,allow_nan=False),encoding="utf-8")
    print(pd.DataFrame(aggregated).to_string(index=False),flush=True)


if __name__=="__main__":
    main()
