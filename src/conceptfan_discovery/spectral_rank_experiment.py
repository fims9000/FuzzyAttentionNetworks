"""Falsify unsupervised predictive-rank identification on independent generators.

The algorithm sees only TRAIN observations and masks. "true_k" appears solely
in the synthetic *generator* and in evaluation columns after fitting.
Competing predeclared estimators are both reported, not winner-selected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .lagged_rank import fit_lagged_predictive_rank
from .predictive_rank import estimate_predictive_hankel_rank
from .variable_factor_data import generate_variable_factor_data


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args=p.parse_args()
    if args.out_dir.exists():
        p.error("Refusing to overwrite an existing research result")
    cfg=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not cfg["protocol"]["no_test_evaluation"] or not cfg["rank"]["no_true_k_use_for_selection"]:
        p.error("Oracle/test isolation contract missing")

    data_cfg=cfg["data"]
    rank_cfg=cfg["rank"]
    outputs=[]
    details={}
    for true_k in data_cfg["true_factor_counts_evaluation_only"]:
        for seed in data_cfg["generator_seeds"]:
            # Synthetic generator definition necessarily knows K; the rank
            # estimator receives measured train arrays only.
            data=generate_variable_factor_data(
                n_true_factors=int(true_k),
                seed=int(seed)+int(true_k),
                split_seed=int(data_cfg["split_seed_offset"])+int(seed)%701+int(true_k),
                n_samples=int(data_cfg["n_samples"]),
                n_channels=int(data_cfg["n_channels"]),
                input_steps=int(data_cfg["input_steps"]),
                horizon=int(data_cfg["horizon"]),
            )
            observed=data.train.x
            mask=data.train.mask
            rank=fit_lagged_predictive_rank(
                observed,mask,
                lags=tuple(int(v) for v in rank_cfg["positive_lags"]),
                permutations=int(rank_cfg["permutation_repetitions"]),
                patient_bootstraps=int(rank_cfg["patient_bootstrap_repetitions"]),
                alpha=float(rank_cfg["familywise_alpha"]),
                seed=int(rank_cfg["seed"])+int(seed)%541+int(true_k),
            )
            alternative=estimate_predictive_hankel_rank(
                observed,mask,
                block_lags=3,
                null_repetitions=int(rank_cfg["permutation_repetitions"]),
                max_rank=int(data_cfg["n_channels"]),
                significance_level=float(rank_cfg["familywise_alpha"]),
                seed=int(rank_cfg["seed"])+int(seed)%541+int(true_k),
            )
            key=f"truth_{true_k}_generator_{seed}"
            details[key]={
                "stacked_lagged_covariance":rank.to_json(),
                "square_block_hankel_comparator":alternative.summary(),
                "train_observable_provenance":{
                    "number_train_patients":len(observed),
                    "n_channels":observed.shape[-1],
                    "channel_names":data.receipt["channel_names"],
                    "original_training_patient_id_sha256":data.receipt["train_ids_sha256"],
                },
                "external_evaluation_only":{
                    "synthetic_planted_latent_dimension":int(true_k),
                    "lagged_rank_error":int(rank.rank-int(true_k)),
                    "block_hankel_rank_error":int(alternative.rank_estimate-int(true_k)),
                },
                "oracle_supplied_to_rank_estimator":False,
                "clinical_generalization_claim":False,
            }
            outputs.append({
                "true_k_external_audit_only":int(true_k),
                "generator_seed":int(seed),
                "n_train_patients":len(observed),
                "lagged_predictive_k":int(rank.rank),
                "lagged_patient_bootstrap_lo":rank.patient_bootstrap_interval[0],
                "lagged_patient_bootstrap_hi":rank.patient_bootstrap_interval[1],
                "lagged_abs_error":abs(int(rank.rank)-int(true_k)),
                "hankel_comparator_k":int(alternative.rank_estimate),
                "hankel_abs_error":abs(int(alternative.rank_estimate)-int(true_k)),
                "signal_singular_value_at_true_k":
                    float(rank.singular_values[int(true_k)-1]),
                "signal_singular_value_after_true_k":
                    float(rank.singular_values[int(true_k)]) if int(true_k)<len(rank.singular_values) else float("nan"),
                "familywise_null_threshold":float(rank.noise_max_threshold),
                "oracle_leakage":False,"test_used":False,
            })
            print(key,"stacked positive-lag K",rank.rank,
                  "Hankel K",alternative.rank_estimate,
                  "95% patient-bootstrap",rank.patient_bootstrap_interval,flush=True)
    args.out_dir.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(outputs).to_csv(args.out_dir/"spectral_rank_external_oracle_audit.csv",index=False)
    (args.out_dir/"spectral_rank_diagnostics.json").write_text(
        json.dumps(details,indent=2,allow_nan=False),encoding="utf-8"
    )
    content={
        "experiment":"train-only observable dynamical rank across independent 3/5/7 planted factor systems",
        "run_count":len(outputs),
        "no_hidden_concept_oracle_in_training":True,
        "test_set_locked":True,
        "selection_rule":"stack positive-lag pairwise-observed channel covariance; count singular values above 95th percentile permuted-patient largest singular value",
        "alternative_rule":"block-Hankel patient-shuffled spectral rank",
        "n_exact_lagged_recoveries":sum(x["lagged_abs_error"]==0 for x in outputs),
        "n_exact_hankel_recoveries":sum(x["hankel_abs_error"]==0 for x in outputs),
        "limitations":[
            "detectable observable temporal predictive rank need not equal latent state dimension without observability and independent measurement noise",
            "nonlinear dynamics, nonstationarity, informative missingness and patient-dependent observations can violate assumptions",
            "bootstrap rank interval with fixed training null threshold is sensitivity, not an unbiased confidence set",
            "this pilot follows earlier selected methods; no confirmatory statistical inference",
        ],
    }
    (args.out_dir/"STATUS.json").write_text(json.dumps(content,indent=2),encoding="utf-8")
    print(json.dumps(content,indent=2),flush=True)


if __name__=="__main__":
    main()
