"""Exact model-level bootstrap of the frozen, 36-fit v6 observation study.

Four initialization seeds yield six overlapping cross-model pairs per
architecture. They are NOT six independent replicates. Use all 4**4=256
ordered model-index resamples (including their multinomial multiplicity)
and a COMMON resampled set of seeds for paired method contrasts.

The resulting conditional model-only intervals do NOT include variation
over patient cohorts or distinct generators, and are NOT clinical p-values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .exact_seed_bootstrap import (
    all_pairs_model_bootstrap,exact_paired_run_bootstrap,
)


METHODS=(
    ("multiplicative_original","unknown_neutral"),
    ("multiplicative_original","unknown_neutral_consistency"),
    ("unknown_neutral","unknown_neutral_consistency"),
)


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--source-dir",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    if args.out_dir.exists():
        p.error("Do not overwrite a completed model-bootstrap audit")
    source=args.source_dir
    if (source/"published_neutral").is_dir():
        source=source/"published_neutral"
    run=pd.read_csv(source/"unknown_neutral_v6_per_run.csv").rename(
        columns={"seed":"init_seed"}
    )
    pairs=pd.read_csv(source/"unknown_neutral_v6_between_seeds.csv").rename(
        columns={"seed_a":"init_a","seed_b":"init_b"}
    )
    env=pd.read_csv(source/"unknown_neutral_v6_by_measurement.csv").rename(
        columns={"seed":"init_seed"}
    )
    all_results=[]
    for true_k in (3,5,7):
        per_run=run[run.hidden_k_external_evaluation_only==true_k]
        per_pairs=pairs[pairs.hidden_k_external_evaluation_only==true_k]
        per_env=env[env.hidden_k_external_evaluation_only==true_k]
        if len(per_run)!=12 or len(per_pairs)!=18 or len(per_env)!=48:
            raise ValueError("Expected 12 fitted models,18 dependent model pairs and48 model-policy observations per K")
        for control,modified in METHODS:
            for metric in (
                "excess_over_unmatched_episode_null_spearman",
                "case_specific_excess_sign_agreement",
            ):
                result=all_pairs_model_bootstrap(
                    per_pairs,metric=metric,
                    methods=(control,modified),n_fits=4,
                )
                all_results.append({
                    "true_k_external_evaluation_only":true_k,
                    "data_part":"crossfit_explanation",
                    "observation_policy":"original",
                    **result,
                })
            for metric in (
                "outer_validation_original_auprc",
                "coverage_adjusted_latent_correspondence",
                "zero_observation_mean_abs_direct_contribution",
            ):
                result=exact_paired_run_bootstrap(
                    per_run,metric=metric,
                    methods=(control,modified),n_fits=4,
                )
                all_results.append({
                    "true_k_external_evaluation_only":true_k,
                    "data_part":"model_outcome",
                    "observation_policy":"original",
                    **result,
                })
            for policy in ("mcar_thin","time_channel_mar","observed_value_mnar"):
                table=per_env[per_env.scenario.eq(policy)]
                for metric in (
                    "outer_auprc","mean_abs_risk_shift",
                    "mean_abs_total_concept_contribution_shift",
                    "case_specific_spearman_excess_over_independent_patient_null",
                    "case_specific_sign_agreement_excess",
                ):
                    result=exact_paired_run_bootstrap(
                        table,metric=metric,
                        methods=(control,modified),n_fits=4,
                    )
                    all_results.append({
                        "true_k_external_evaluation_only":true_k,
                        "data_part":"observational_stress",
                        "observation_policy":policy,
                        **result,
                    })
    args.out_dir.mkdir(parents=True,exist_ok=False)
    frame=pd.DataFrame([{
        "true_k":r["true_k_external_evaluation_only"],
        "scope":r["data_part"],
        "measurement_policy":r["observation_policy"],
        "reference":r["method_reference"],
        "alternative":r["method_comparator"],
        "metric":r["metric"],
        "mean_paired_delta":r["mean_paired_difference"],
        "model_only_95ci_low":r["model_only_bootstrap_ci95"][0],
        "model_only_95ci_high":r["model_only_bootstrap_ci95"][1],
        "n_independent_init_seeds":4,
        "n_exact_resamples":r.get("valid_distinct_model_resamples",256),
        "dependent_pairs_are_not_independent":r["data_part"]=="crossfit_explanation",
    } for r in all_results])
    frame.to_csv(args.out_dir/"unknown_neutral_v6_exact_four_model_bootstrap.csv",index=False)
    receipt={
        "pinned_original_workflow":"https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41506778924",
        "pinned_input_artifact":"unknown-neutral-v6-1",
        "n_fitted_independent_init_seeds":4,
        "overlapping_crossfit_pairs_per_arm":6,
        "all_ordered_bootstrap_draws":4**4,
        "valid_distinct_pair_resamples":4**4-4,
        "patient_cohort_uncertainty_included":False,
        "independent_generator_uncertainty_included":False,
        "test_set_inspected":False,
        "no_confirmatory_significance_or_causal_claim":True,
        "results":all_results,
    }
    (args.out_dir/"full_exact_four_model_receipt.json").write_text(
        json.dumps(receipt,indent=2,allow_nan=False),encoding="utf-8"
    )
    print(frame.loc[
        frame.metric.isin((
            "excess_over_unmatched_episode_null_spearman",
            "outer_validation_original_auprc",
            "mean_abs_total_concept_contribution_shift",
        ))
    ].to_string(index=False),flush=True)


if __name__=="__main__":
    main()
