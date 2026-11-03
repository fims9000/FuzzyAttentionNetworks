"""Model-level paired bootstrap from pinned real PhysioNet V6/V7 receipts.

Both the old and new experiments used identical TRAIN/outer VALIDATION
patient indices, same three source initialization-seed identifiers and
three exact predeclared observation scenarios. This audit treats the unit
as a trained model seed (3), NOT a patient, channel, concept or
overlapping model pair.

Enumerate all 3**3 ordered paired source-model bootstrap resamples. The
results are descriptive uncertainty conditional on this historical
reused development validation. They are NOT medical generalization
intervals, p-values or patient-level bootstrap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .exact_seed_bootstrap import exact_paired_run_bootstrap


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--frozen-v6",type=Path,required=True)
    p.add_argument("--frozen-v7",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    if args.out_dir.exists():
        p.error("Refusing to overwrite historical bootstrap results")
    old_dir=args.frozen_v6
    new_dir=args.frozen_v7
    if (old_dir/"published_clinical_neutral").is_dir():
        old_dir=old_dir/"published_clinical_neutral"
    if (new_dir/"published_dual").is_dir():
        new_dir=new_dir/"published_dual"
    old=pd.read_csv(old_dir/"clinical_v6_observation_stress.csv")
    new=pd.read_csv(new_dir/"dual_process_v7_per_policy.csv")
    if set(old.scenario)!={"original","mcar_thin","time_channel_mar"} or (
        set(new.scenario)!={"original","mcar_thin","time_channel_mar"}
    ):
        raise ValueError("Different observation-stress protocols cannot be compared")
    methods={
        "v6_old_physiology":("multiplicative_original","scenario_auprc"),
        "v6_neutral_physiology":("unknown_neutral","scenario_auprc"),
        "v6_neutral_consistency":("unknown_neutral_consistency","scenario_auprc"),
        "v7_full_dual_process":(None,"auprc_full"),
        "v7_physiology_only":(None,"auprc_physical_only"),
        "v7_care_process_only":(None,"auprc_observation_process_only"),
    }
    contrasts=(
        ("v6_old_physiology","v7_full_dual_process"),
        ("v6_neutral_physiology","v7_full_dual_process"),
        ("v6_neutral_consistency","v7_full_dual_process"),
        ("v7_physiology_only","v7_full_dual_process"),
        ("v7_care_process_only","v7_full_dual_process"),
    )
    samples=[]
    audit=[]
    for scenario in ("original","mcar_thin","time_channel_mar"):
        for display,(source_method,column) in methods.items():
            if source_method is None:
                part=new[new.scenario.eq(scenario)]
                ids=part.seed.astype(int)
            else:
                part=old[old.scenario.eq(scenario)&old.method.eq(source_method)]
                ids=part.initialization_seed.astype(int)
            if len(part)!=3 or ids.nunique()!=3:
                raise ValueError(f"Expected three independent source models in {display}/{scenario}")
            for seed,value in zip(ids,part[column]):
                samples.append({
                    "method":display,
                    "scenario":scenario,
                    "init_seed":int(seed),
                    "outer_auprc":float(value),
                })
        partition=pd.DataFrame(samples)
        partition=partition[partition.scenario.eq(scenario)]
        for baseline,alternative in contrasts:
            report=exact_paired_run_bootstrap(
                partition,metric="outer_auprc",
                methods=(baseline,alternative),n_fits=3,
            )
            audit.append({
                "scenario":scenario,
                "reference":baseline,
                "alternative":alternative,
                "delta_auprc":report["mean_paired_difference"],
                "model_only_ci95_lo":report["model_only_bootstrap_ci95"][0],
                "model_only_ci95_hi":report["model_only_bootstrap_ci95"][1],
                "n_independent_init_seeds":3,
                "n_ordered_exact_bootstrap_draws":3**3,
                "historical_validation_previously_used":True,
                "no_patient_level_uncertainty":True,
                "not_clinical_significance":True,
            })
    args.out_dir.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(audit).to_csv(
        args.out_dir/"physionet_v7_vs_v6_exact_model_bootstrap.csv",index=False,
    )
    (args.out_dir/"STATUS.json").write_text(json.dumps({
        "status":"EXPLORATORY_THREE_MODEL_BOOTSTRAP_NO_CONFIRMATORY_PVALUE",
        "v6_source_workflow":"https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41507973119",
        "v7_source_workflow":"https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41508831155",
        "source_v6_artifact":"physionet-unknown-neutral-v6-1",
        "source_v7_artifact":"physionet-separate-care-process-concepts-v7-1",
        "n_source_models":3,
        "paired_bootstrap_resamples":3**3,
        "uncertainty_scope":"trained model initialization only, reused fixed validation patients",
        "not_corrected_for_repeated_model_development_or_patient_sampling":True,
        "no_clinical_concept_identity_claim":True,
        "no_test_or_calibration_prediction":True,
        "comparisons":audit,
    },indent=2,allow_nan=False),encoding="utf-8")
    print(pd.DataFrame(audit).to_string(index=False),flush=True)


if __name__=="__main__":
    main()
