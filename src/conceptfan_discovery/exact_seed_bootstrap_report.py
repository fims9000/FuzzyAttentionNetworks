"""Recompute exact model-dependence-aware v4 intervals from frozen CSV evidence.

Never fit or rerun neural networks, never overwrite accepted paper tables.
Enumerates all 5**5 bootstrap model-index draws (exclude all-same when
there is no independently trained pair). Same sampled source models are used
in both methods for paired rank, sign, and AUPRC differences.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .exact_seed_bootstrap import all_pairs_model_bootstrap,exact_paired_run_bootstrap


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--source-dir",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    if args.out_dir.exists():
        p.error("Never overwrite old research evidence")
    sources=args.source_dir
    if (sources/"published_task_head").is_dir():
        sources=sources/"published_task_head"
    pairs=pd.read_csv(sources/"task_head_v4_pairwise.csv")
    runs=pd.read_csv(sources/"task_head_v4_per_run.csv")
    results=[]
    metrics=("excess_over_unmatched_episode_null_spearman",
             "case_specific_excess_sign_agreement",
             "excess_over_unmatched_episode_null_top1",
             "definition_cosine_normalized")
    methods=(("spectral_finetuned","task_head_fixed"),
             ("spectral_finetuned","task_head_prior"),
             ("task_head_fixed","task_head_prior"))
    for true_k in (3,5,7):
        subset_pairs=pairs[pairs.true_k_external_audit_only==true_k]
        subset_runs=runs[runs.true_k_external_audit_only==true_k]
        if len(subset_pairs)!=30 or len(subset_runs)!=15:
            raise ValueError("Expected exactly 30 dependent model pair metrics and 15 per-model rows per true K")
        for reference,comparator in methods:
            for metric in metrics:
                result=all_pairs_model_bootstrap(
                    subset_pairs,metric=metric,
                    methods=(reference,comparator),n_fits=5,
                )
                results.append({
                    "planted_k_external_audit_only":true_k,
                    "analysis_type":"model_pair_agreement",
                    **result,
                })
            for metric in ("outer_auprc","oracle_coverage_adjusted_abs_corr",
                           "future_observed_rmse"):
                result=exact_paired_run_bootstrap(
                    subset_runs,metric=metric,
                    methods=(reference,comparator),n_fits=5,
                )
                results.append({
                    "planted_k_external_audit_only":true_k,
                    "analysis_type":"paired_run_metric",
                    **result,
                })
    args.out_dir.mkdir(parents=True,exist_ok=False)
    pd.DataFrame([{
        "planted_k_external_audit_only":row["planted_k_external_audit_only"],
        "metric":row["metric"],
        "analysis_type":row["analysis_type"],
        "reference":row["method_reference"],
        "comparator":row["method_comparator"],
        "mean_paired_delta":row["mean_paired_difference"],
        "ci95_model_only_low":row["model_only_bootstrap_ci95"][0],
        "ci95_model_only_high":row["model_only_bootstrap_ci95"][1],
        "valid_resamples":row.get("valid_distinct_model_resamples",
                                  row.get("resamples_enumerated")),
    } for row in results]).to_csv(
        args.out_dir/"exact_five_model_bootstrap_v4.csv",index=False,
    )
    (args.out_dir/"full_bootstrap_receipt.json").write_text(json.dumps({
        "source_v4_workflow":"https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41501300941",
        "exact_original_train_model_units":5,
        "overlapping_source_pairs_per_arm":10,
        "original_independent_generator_replications_per_k":1,
        "bootstrap_exact_draws":5**5,
        "test_ever_evaluated":False,
        "model_only_uncertainty":True,
        "patient_sampling_uncertainty_included":False,
        "causal_or_clinical_significance_claim":False,
        "original_frozen_sources_modified":False,
        "methods":list(methods),
        "rows":results,
    },indent=2),encoding="utf-8")
    print(pd.read_csv(args.out_dir/"exact_five_model_bootstrap_v4.csv").to_string(index=False))


if __name__=="__main__":
    main()
