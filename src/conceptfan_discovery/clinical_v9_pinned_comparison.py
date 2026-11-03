"""Pinned three-seed comparison of V7 three care slots, V8 nine learned-rank
care slots, and V9 four common-core care slots.

All outcomes derive from the same historical 300-case development validation,
same original TRAIN/validation split and same 11/23/37 source initialization
IDs. This is a *model-only* descriptive experiment; all 3**3 exact ordered
bootstrap draws are enumerated, but neither patients nor hospitals are
independent replications.

No patient-level predictions, true medical concept labels, later
calibration, or held-out historical test cases enter this report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .exact_seed_bootstrap import exact_paired_run_bootstrap


def locate(root:Path,subdir:str,filename:str)->Path:
    for path in (root/subdir/filename,root/filename):
        if path.exists():
            return path
    raise FileNotFoundError(f"Expected frozen {subdir}/{filename} in {root}")


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--v7-dir",type=Path,required=True)
    parser.add_argument("--v8-dir",type=Path,required=True)
    parser.add_argument("--v9-dir",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args()
    if args.out_dir.exists():
        parser.error("Refusing to replace frozen model evidence")
    v7=pd.read_csv(locate(args.v7_dir,"published_dual","dual_process_v7_per_policy.csv"))
    v8=pd.read_csv(locate(args.v8_dir,"published_care_v8","anchored_care_v8_measurement_stress.csv"))
    v9=pd.read_csv(locate(args.v9_dir,"published_v9_trained","v9_common_care_model_by_policy.csv"))
    methods={
        "v7_manual_three_care_slots":v7.assign(
            method="v7_manual_three_care_slots",init_seed=v7.seed
        ),
        "v8_unsupervised_mask_rank_9_free_head":v8[v8.method.eq("train_mask_source_only")].copy().assign(
            method="v8_unsupervised_mask_rank_9_free_head"
        ),
        "v8_unsupervised_mask_rank_9_task_prior":v8[v8.method.eq("train_mask_source_and_task_prior")].copy().assign(
            method="v8_unsupervised_mask_rank_9_task_prior"
        ),
        "v9_common_mask_core_4_free_head":v9[v9.method.eq("train_mask_source_only")].copy().assign(
            method="v9_common_mask_core_4_free_head",
            init_seed=lambda df:df.initialization_seed,
        ),
        "v9_common_mask_core_4_task_prior":v9[v9.method.eq("train_mask_source_and_task_prior")].copy().assign(
            method="v9_common_mask_core_4_task_prior",
            init_seed=lambda df:df.initialization_seed,
        ),
    }
    all_rows=[]
    for name,frame in methods.items():
        for record in frame.itertuples(index=False):
            policy=getattr(record,"scenario",None)
            if policy is None:
                policy=getattr(record,"evaluation_policy",None)
            if policy not in ("original","mcar_thin","time_channel_mar"):
                raise ValueError("Unannounced validation observation regime")
            all_rows.append({
                "method":name,
                "scenario":policy,
                "init_seed":int(record.init_seed),
                "auprc_full":float(record.auprc_full),
                "auprc_physiological_branch":float(record.auprc_physical_only),
                "auprc_observation_process_branch":float(record.auprc_observation_process_only),
                "mask_process_abs_logit_fraction":float(
                    record.mean_fraction_signed_abs_logit_from_process
                ),
            })
    aligned=pd.DataFrame(all_rows)
    for (name,policy),frame in aligned.groupby(["method","scenario"]):
        if len(frame)!=3 or set(frame.init_seed)!={11,23,37}:
            raise ValueError(f"Not a matched 3-independent-model experiment: {name}/{policy}")
    comparisons=(
        ("v7_manual_three_care_slots","v9_common_mask_core_4_free_head"),
        ("v7_manual_three_care_slots","v9_common_mask_core_4_task_prior"),
        ("v8_unsupervised_mask_rank_9_free_head","v9_common_mask_core_4_free_head"),
        ("v8_unsupervised_mask_rank_9_task_prior","v9_common_mask_core_4_task_prior"),
    )
    rows=[]
    for policy in ("original","mcar_thin","time_channel_mar"):
        part=aligned[aligned.scenario.eq(policy)].copy()
        for base,experiment in comparisons:
            for metric in ("auprc_full","auprc_physiological_branch",
                           "auprc_observation_process_branch",
                           "mask_process_abs_logit_fraction"):
                result=exact_paired_run_bootstrap(
                    part,metric=metric,methods=(base,experiment),n_fits=3
                )
                rows.append({
                    "policy":policy,"baseline":base,"experiment":experiment,
                    "metric":metric,
                    "paired_delta":result["mean_paired_difference"],
                    "model_bootstrap_95ci_low":result["model_only_bootstrap_ci95"][0],
                    "model_bootstrap_95ci_high":result["model_only_bootstrap_ci95"][1],
                    "three_paired_source_model_seeds":True,
                    "independent_patient_bootstrap":False,
                    "confirmatory_inference":False,
                })
    args.out_dir.mkdir(parents=True,exist_ok=False)
    grouped=aligned.groupby(["method","scenario"]).agg(
        independent_seeds=("init_seed","nunique"),
        mean_auprc=("auprc_full","mean"),
        std_auprc=("auprc_full","std"),
        mean_physiological_auprc=("auprc_physiological_branch","mean"),
        mean_care_only_auprc=("auprc_observation_process_branch","mean"),
        mean_care_logit_magnitude_fraction=("mask_process_abs_logit_fraction","mean"),
    ).reset_index()
    aligned.to_csv(args.out_dir/"v7_v8_v9_paired_model_auprc.csv",index=False)
    grouped.to_csv(args.out_dir/"v7_v8_v9_summary_by_observation_regime.csv",index=False)
    pd.DataFrame(rows).to_csv(args.out_dir/"v9_exact_three_model_bootstrap.csv",index=False)
    receipt={
        "status":"COMPARISON_FROM_PINNED_ORIGINAL_MODEL_RUNS",
        "source_v7_workflow":"https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41509956362",
        "source_v8_workflow":"https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41512252657",
        "source_v9_workflow":"https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41610735337",
        "fixed_three_training_seeds":[11,23,37],
        "same_historical_300_validation_patients":True,
        "original_train_only_rank_and_feature_dictionary":True,
        "original_physical_slots_8_and_mask_slots_3_9_4_respectively":True,
        "all_3_to_the_third_bootstrap_draws_enumerated":True,
        "not_independent_patient_site_or_generator_uncertainty":True,
        "not_confirmatory_due_to_development_reuse":True,
        "clinical_causal_or_physiological_meaning_proved":False,
        "medical_concept_cardinality_claimed":False,
        "records":rows,
    }
    (args.out_dir/"STATUS.json").write_text(
        json.dumps(receipt,indent=2,allow_nan=False),encoding="utf-8"
    )
    print(grouped.to_string(index=False),flush=True)
    print("BASELINE_V7_V8_TO_V9",flush=True)
    print(pd.DataFrame(rows).query("metric=='auprc_full'").to_string(index=False),flush=True)


if __name__=="__main__":
    main()
