"""Train-only falsification of dynamic-rank robustness to observation policy.

Compare pooled observed-pair covariance (v3) against equal-time averaged,
patient-shuffle-calibrated covariance (v5). For each *predeclared* synthetic
measurement policy, estimate whole-TRAIN and two disjoint TRAIN-half ranks.

The gate must ABSTAIN when any admissible policy fails joint observations,
rank agreement, or patient-specific subspace agreement. Diagnostic MNAR
value-dependent thinning is reported as a negative stress, not counted as
proof that the MAR rank estimator is invalid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from .lagged_rank import fit_lagged_predictive_rank
from .observation_regimes_v5 import (
    apply_measurement_environment, evaluate_rank_invariance,
)
from .time_stratified_rank_v5 import fit_time_stratified_rank
from .variable_factor_data import generate_variable_factor_data


def estimate(
    name:str,x:np.ndarray,m:np.ndarray,
    *,
    options:dict,seed:int,
)->tuple[SimpleNamespace,dict]:
    if name=="pooled_v3":
        out=fit_lagged_predictive_rank(
            x,m,lags=tuple(options["lags"]),
            permutations=int(options["permutation_repetitions"]),
            patient_bootstraps=int(options["baseline_patient_bootstrap_repetitions"]),
            alpha=float(options["alpha"]),seed=seed,
        )
        return SimpleNamespace(rank=out.rank,
                               source_direction=out.left_observable_loadings),{
            "rank":out.rank,"estimator":"pooled_v3",
            "noise_threshold":out.noise_max_threshold,
            "spectrum":out.singular_values.tolist(),
            "patient_bootstrap_interval":list(out.patient_bootstrap_interval),
        }
    if name=="equal_time_v5":
        out=fit_time_stratified_rank(
            x,m,lags=tuple(options["lags"]),
            permutations=int(options["permutation_repetitions"]),
            min_pairs_per_hour=int(options["min_pairs_per_hour"]),
            min_time_strata=int(options["min_time_strata"]),
            alpha=float(options["alpha"]),seed=seed,
        )
        return SimpleNamespace(rank=out.rank,
                               source_direction=out.source_direction),{
            "rank":out.rank,"estimator":"equal_time_v5",
            "noise_threshold":out.familywise_null_threshold,
            "spectrum":out.singular_values.tolist(),
            "minimum_observed_hourly_pair_strata":int(out.observed_time_strata_per_channel_pair.min()),
            "minimum_null_hourly_pair_strata":int(out.null_min_supported_time_strata_per_draw.min()),
        }
    raise ValueError("Unknown rank estimator")


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args()
    if args.out_dir.exists():
        parser.error("Refusing to overwrite prior exploratory results")
    cfg=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not cfg["protocol"]["no_test_or_calibration_evaluation"] or not cfg["reporting"]["no_oracle_for_any_rank_selection"]:
        parser.error("No test or planted-state leakage allowed")
    environments=tuple(cfg["measurement"]["admissible_environments"])
    diagnostic=cfg["measurement"]["diagnostic_mnar_environment"]
    if set(environments)!={"original","mcar_thin","time_channel_mar"} or diagnostic!="observed_value_mnar":
        parser.error("Missing required predeclared observation-policy controls")
    if set(cfg["data"]["planted_k_external_audit_only"])!={3,5,7}:
        parser.error("Expected frozen 3/5/7 independent factor configurations")

    details={}
    rows=[]
    gates=[]
    for true_k in cfg["data"]["planted_k_external_audit_only"]:
        for generator_seed in cfg["data"]["generator_seeds"]:
            ds=generate_variable_factor_data(
                n_true_factors=int(true_k),
                seed=int(generator_seed)+int(true_k),
                split_seed=int(cfg["data"]["split_seed_base"])+int(true_k),
                n_samples=int(cfg["data"]["n_samples"]),
                n_channels=int(cfg["data"]["n_channels"]),
                input_steps=int(cfg["data"]["input_steps"]),
                horizon=int(cfg["data"]["horizon"]),
            )
            x=ds.train.x
            m=ds.train.mask
            n=len(x)
            shuffle=np.random.default_rng(
                int(cfg["measurement"]["train_halves_seed"])+int(true_k)
            ).permutation(n)
            halves=(shuffle[:n//2],shuffle[n//2:])
            key=f"external_hidden_{true_k}_generator_{generator_seed}"
            details[key]={
                "source_train_receipt":ds.receipt,
                "all_predictions_and_training_targets_excluded_from_rank":True,
                "measurement_environments":{},
            }
            by_method={"pooled_v3":{},"equal_time_v5":{}}
            for env_i,environment in enumerate((*environments,diagnostic)):
                value,mask,env_receipt=apply_measurement_environment(
                    x,m,environment=environment,
                    seed=int(cfg["measurement"]["mask_seed_base"])+int(true_k)*41+env_i,
                )
                details[key]["measurement_environments"][environment]=env_receipt
                for method_i,method in enumerate(by_method):
                    results=[]
                    for split_i,ids in enumerate((None,*halves)):
                        xs=value if ids is None else value[ids]
                        ms=mask if ids is None else mask[ids]
                        label=("whole_train","half_train_A","half_train_B")[split_i]
                        try:
                            output,rec=estimate(
                                method,xs,ms,options=cfg["rank"],
                                seed=int(cfg["rank"]["null_seed"])+
                                int(true_k)*101+env_i*7+method_i*17+split_i,
                            )
                            status="RANK_ESTIMATED"
                            reason=None
                        except ValueError as exc:
                            output=None
                            rec={"rank":None,"estimator":method}
                            status="ABSTAIN_SUPPORT_OR_ESTIMATOR"
                            reason=str(exc)
                        results.append(output)
                        rows.append({
                            "planted_factor_count_external_audit_only":int(true_k),
                            "generator_seed":int(generator_seed),
                            "environment":environment,
                            "estimator":method,
                            "train_partition":label,
                            "n_train_patients":len(xs),
                            "new_missingness_policy_observation_fraction":float(mask.mean()),
                            "status":status,
                            "detected_observable_k":rec["rank"],
                            "absolute_error_external_audit_only":
                                abs(int(rec["rank"])-int(true_k)) if output is not None else np.nan,
                            "failure_reason":reason,
                            "permutation_noise_threshold":rec.get("noise_threshold"),
                            "minimum_supported_time_strata":rec.get("minimum_observed_hourly_pair_strata"),
                            "minimum_null_time_strata":rec.get("minimum_null_hourly_pair_strata"),
                            "latent_oracle_and_outcome_used_for_rank":False,
                        })
                        print(
                            "oracle-eval K",true_k,"environment",environment,
                            "rank-algorithm",method,label,"K",rec["rank"],
                            "status",status,flush=True,
                        )
                    by_method[method][environment]={
                        "whole":results[0],
                        "halves":tuple(results[1:]),
                    }
            for method,by_env in by_method.items():
                # The diagnostic MNAR environment is always reported but it
                # violates the explicitly permitted time-conditional MAR
                # assumption, so success is not required there.
                gate=evaluate_rank_invariance(
                    {env:by_env[env] for env in environments},
                    method=method,
                    min_squared_principal_cosine=float(
                        cfg["measurement"]["minimum_half_subspace_squared_cosine"]
                    ),
                )
                gates.append({
                    "true_k_for_external_audit_only":int(true_k),
                    "generator_seed":int(generator_seed),
                    "estimator":method,
                    "gate_status":gate.status,
                    "gate_chosen_k":gate.consensus_k,
                    "gate_correct_if_accepted":
                        (gate.consensus_k==int(true_k)) if gate.consensus_k is not None else None,
                    "gate_reasons":"|".join(gate.reasons),
                    "pass_not_forced":gate.status=="MEASUREMENT_INVARIANT_OBSERVABLE_RANK",
                })
                details[key][method]={
                    "admissible_environment_invariance":gate.json(),
                    "diagnostic_mnar_full_rank":
                        None if by_env[diagnostic]["whole"] is None
                        else int(by_env[diagnostic]["whole"].rank),
                    "mnar_not_assumed_recoverable":True,
                }
    args.out_dir.mkdir(parents=True,exist_ok=False)
    frame=pd.DataFrame(rows)
    gate_frame=pd.DataFrame(gates)
    frame.to_csv(args.out_dir/"measurement_rank_per_partition.csv",index=False)
    gate_frame.to_csv(args.out_dir/"measurement_invariance_gates.csv",index=False)
    (args.out_dir/"measurement_rank_diagnostics.json").write_text(
        json.dumps(details,indent=2,allow_nan=False),encoding="utf-8",
    )
    success={
        method:{
            "complete_synthetic_generator_cases":len(part),
            "accepted_k_count":int(part.pass_not_forced.sum()),
            "accepted_with_correct_k":int(
                part.loc[part.pass_not_forced,"gate_correct_if_accepted"].eq(True).sum()
            ),
            "rejected_due_to_nonidentifiability":int((~part.pass_not_forced).sum()),
        }
        for method,part in gate_frame.groupby("estimator")
    }
    status={
        "protocol":cfg["protocol"]["name"],
        "training_concepts_supplied_by_humans":False,
        "true_latent_states_given_to_estimators":False,
        "outcome_labels_given_to_estimators":False,
        "validation_or_test_used":False,
        "new_experiment_after_earlier_physionet_failure":True,
        "external_oracle_only_in_postfit_audit":True,
        "known_mnar_violation_is_a_separate_diagnostic":True,
        "all_environments_same_patients_not_independent_data":True,
        "results":success,
        "interpretation":"reject unstable dimension, never force true K from external generator",
        "limitation":"neither time-stratified nor pooled method proves physiological semantics under latent-dependent sampling",
    }
    (args.out_dir/"STATUS.json").write_text(
        json.dumps(status,indent=2,allow_nan=False),encoding="utf-8"
    )
    print(json.dumps(status,indent=2),flush=True)


if __name__=="__main__":
    main()
