"""Reproduce frozen v6 synthetic concept formulas from first model seed.

Each saved concept is native to the trained classifier (not post-hoc):
measured signed operator mixture, fuzzy threshold and width, reliability,
signed logit coefficient, activation frequency and task contribution.

No patient-level record is published. A pinned earlier model run CSV is
checked for reproduced held-out AUPRC to catch silent implementation drift.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .evaluation import (
    concept_passports,evaluate_discovery,oracle_correspondence,
)
from .lagged_rank import fit_lagged_predictive_rank
from .selection import split_validation_for_cardinality
from .unknown_neutral_experiment_v6 import fit_one
from .variable_factor_data import generate_variable_factor_data


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--frozen-v6-runs",type=Path,required=True)
    args=p.parse_args()
    if args.out_dir.exists():
        p.error("Never rewrite a previously produced concept definition audit")
    cfg=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    original=pd.read_csv(args.frozen_v6_runs)
    torch.set_num_threads(int(cfg["training"]["cpu_threads"]))
    run_seed=int(cfg["training"]["initialization_seeds"][0])
    args.out_dir.mkdir(parents=True,exist_ok=False)
    all_results=[]
    for true_k in cfg["data"]["planted_k_external_evaluation_only"]:
        data=generate_variable_factor_data(
            n_true_factors=int(true_k),
            seed=int(cfg["data"]["generator_seed_base"])+int(true_k),
            split_seed=int(cfg["data"]["split_seed_base"])+int(true_k),
            n_samples=int(cfg["data"]["n_samples"]),
            n_channels=int(cfg["data"]["n_channels"]),
            input_steps=int(cfg["data"]["input_window"]),
            horizon=int(cfg["data"]["horizon"]),
        )
        r=cfg["rank"]
        rank=fit_lagged_predictive_rank(
            data.train.x,data.train.mask,
            lags=tuple(r["train_only_lags"]),
            permutations=int(r["permutations"]),
            patient_bootstraps=int(r["patient_bootstraps"]),
            alpha=float(r["alpha"]),
            seed=int(r["rank_seed"])+int(true_k),
        )
        if rank.rank<2:
            raise RuntimeError("No trustworthy train-only concept dimension")
        basis=rank.left_observable_loadings[:,:rank.rank]
        _,outer=split_validation_for_cardinality(
            data.validation,random_seed=int(cfg["evaluation"]["outer_validation_split_seed"])
        )
        for method in cfg["training"]["methods"]:
            model,_=fit_one(method,run_seed,data,cfg,basis)
            evaluation=evaluate_discovery(model,outer)
            expected=original[
                original.hidden_k_external_evaluation_only.eq(int(true_k))
                & original.method.eq(method)
                & original.seed.eq(run_seed)
            ]
            if len(expected)!=1:
                raise ValueError("No uniquely matching frozen reference seed in v6")
            discrepancy=abs(
                float(expected.outer_validation_original_auprc.iloc[0])
                -float(evaluation.metrics["auprc"])
            )
            if discrepancy>1e-7:
                raise AssertionError(
                    f"Frozen method AUPRC reproduction mismatch {method} k={true_k} delta={discrepancy:.6g}"
                )
            passports=concept_passports(
                evaluation,model,outer,data.medians,data.scales,
                include_examples=False,
            )
            if len(passports)!=rank.rank:
                raise AssertionError("Model number of explicit concept passports differs from data-derived rank")
            with torch.no_grad():
                missing=model(
                    torch.zeros_like(torch.from_numpy(outer.x[:3])),
                    torch.zeros_like(torch.from_numpy(outer.mask[:3])),
                )
                direct_if_no_observation=float(missing.contributions.abs().max())
            output={
                "source_frozen_workflow":"https://github.com/fims9000/FuzzyAttentionNetworks/actions/runs/41506778924",
                "reproduces_frozen_init_seed":run_seed,
                "frozen_auprc_discrepancy":discrepancy,
                "method":method,
                "oracle_planted_factor_count_evaluation_only":int(true_k),
                "measured_train_selected_rank":int(rank.rank),
                "external_planted_states_used_for_training":False,
                "manually_provided_concept_definitions":False,
                "final_outcome_head_uses_only_native_concept_summaries":True,
                "trained_fuzzy_thresholds_and_widths":True,
                "mean_validation_auprc":float(evaluation.metrics["auprc"]),
                "reliability_aware_unknown_state":method!="multiplicative_original",
                "max_abs_direct_contribution_when_no_sensor_observed":
                    direct_if_no_observation,
                "concept_operators":model.concept_basis_manifest(),
                "definition_passports":passports,
                "patient_level_traces_published":False,
                "physiological_concept_labels_verified":False,
                "causal_physiological_intervention_claim":False,
            }
            filename=f"concept_passports_k{true_k}_{method}_seed{run_seed}.json"
            (args.out_dir/filename).write_text(
                json.dumps(output,indent=2,allow_nan=False),encoding="utf-8"
            )
            all_results.append({
                "planted_k_audit_only":int(true_k),
                "model":method,
                "rank_from_train":int(rank.rank),
                "n_usable_native_concepts":len(passports),
                "seed":run_seed,
                "frozen_auprc_discrepancy":discrepancy,
                "largest_missing_measurement_direct_effect":direct_if_no_observation,
                "all_passports_include_exact_signed_beta":
                    all(isinstance(x["signed_direct_logit_coefficient"],float)
                        for x in passports),
                "all_passports_no_patient_examples":
                    all(not x["illustrative_validation_episodes"] for x in passports),
            })
            print(filename,"reproduction_error",discrepancy,flush=True)
    pd.DataFrame(all_results).to_csv(
        args.out_dir/"audited_concept_passport_index.csv",index=False
    )
    (args.out_dir/"STATUS.json").write_text(json.dumps({
        "status":"ALL_V6_FIRST_SEEDS_REPRODUCED_FROM_TRAINING",
        "nine_native_concept_models":len(all_results),
        "true_states_and_clinical_targets_unused":True,
        "all_coefficients_influence_prediction_during_training":True,
        "no_patient_identifiers_in_passports":True,
        "no_reliable_medical_concept_names_available":True,
        "models":all_results,
    },indent=2,allow_nan=False),encoding="utf-8")


if __name__=="__main__":
    main()
