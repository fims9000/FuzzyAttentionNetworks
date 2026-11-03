"""PhysioNet: one V10 versus independently defined real-model baselines.

Measures logistic history, sklearn histogram gradient boosting, Che-style
GRU-D, and a 2-layer time-series Transformer against ONLY V10. Previous
ConceptFAN versions are not used as scientific comparators.

Same official 4000-patient train/validation partition, V10's train-only
temporal-sensor selection, 2400 actual training patients, same prior-viewed
300 development patients, 36 historical hours and fixed 12-hour future
horizon (unused by external classifiers). No existing calibration/test
patient is loaded into an evaluation tensor. No per-patient predictions
or identifiers are exported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .clinical_multiscale_v10 import fit_multiscale
from .clinical_unknown_neutral_experiment_v6 import subset_discovery_split
from .data import DiscoveryData
from .external_baselines_v10 import (
    fit_neural_baseline, fit_tabular_baseline,
    predict_neural, predict_tabular,
)
from .lagged_rank import fit_lagged_predictive_rank
from .observation_regimes_v5 import apply_measurement_environment
from .physionet import prepare_physionet_discovery
from .pilot_temporal_sensors_v5 import choose_common_temporal_sensors
from .selection import split_validation_for_cardinality


EXTERNAL_METHODS=(
    "logistic_history",
    "hist_gradient_boosting",
    "gru_d_style",
    "transformer_timeseries",
)
V10_METHOD="conceptfan_v10_multiscale_consistency"
SCENARIOS=("original","mcar_thin","time_channel_mar")


def _load_and_match_v10_patient_cohort(
    *,
    source_zip:Path,
    outcome_file:Path,
    base_cfg:dict,
    prior_receipt:dict,
) -> tuple[DiscoveryData,np.ndarray,object,dict]:
    with tempfile.TemporaryDirectory(prefix="v10_external_orig_split_") as path:
        original=prepare_physionet_discovery(
            source_zip,outcome_file,Path(path),
            split_seed=int(base_cfg["data"]["split_seed"]),
            input_window=int(base_cfg["data"]["input_window"]),
            forecast_horizon=int(base_cfg["data"]["forecast_horizon"]),
            minimum_train_patient_coverage=float(
                base_cfg["data"]["minimum_train_patient_coverage"]
            ),
        )
    assert len(original.train.y)==2400
    assert len(original.validation.y)==600
    source_order=np.random.default_rng(
        int(base_cfg["data"]["pilot_selection_seed"])
    ).permutation(len(original.train.y))
    pilot=source_order[:int(base_cfg["data"]["pilot_train_patients"])]
    independent=source_order[len(pilot):]
    feature=choose_common_temporal_sensors(
        original.train.mask[pilot],original.receipt["channel_names"],
        lags=tuple(base_cfg["data"]["pilot_sensor_lags"]),
        minimum_supported_strata=int(base_cfg["data"]["pilot_min_time_strata"]),
        minimum_observed_pairs_per_stratum=int(
            base_cfg["data"]["pilot_min_hourly_pairs"]
        ),
        minimum_expected_null_pairs_per_stratum=float(
            base_cfg["data"]["pilot_min_expected_null_pairs"]
        ),
        worst_additional_observation_retention=float(
            base_cfg["data"]["pilot_worst_extra_thinning"]
        ),
    )
    if feature["status"]!="PILOT_SENSORS_ELIGIBLE":
        raise RuntimeError("Exact V10 train-only sensors not eligible")
    selected=np.asarray(feature["selected_indices"],dtype=int)
    rc=base_cfg["rank"]
    rank=fit_lagged_predictive_rank(
        original.train.x[independent][:,:,selected],
        original.train.mask[independent][:,:,selected],
        lags=tuple(rc["lags"]),permutations=int(rc["permutations"]),
        patient_bootstraps=int(rc["patient_bootstraps"]),
        alpha=float(rc["alpha"]),seed=int(rc["seed"]),
    )
    capacity=min(int(rank.rank),int(rc["max_computational_concept_slots"]))
    if capacity<2:
        raise RuntimeError("Cannot inflate an unsupported physiological dimension")
    observed=DiscoveryData(
        train=subset_discovery_split(original.train,selected),
        validation=subset_discovery_split(original.validation,selected),
        receipt={**original.receipt,
                 "channel_names":feature["selected_names"],
                 "n_channels":len(selected),
                 "not_predefined_clinical_concepts":True},
        medians=original.medians[selected],
        scales=original.scales[selected],
    )
    _,development=split_validation_for_cardinality(
        observed.validation,
        random_seed=int(
            base_cfg["evaluation"]["reported_validation_subset_seed"]
        ),
    )
    if prior_receipt.get("measured_channels")!=observed.receipt["channel_names"]:
        raise ValueError("Baseline patient channels differ from frozen V10 run")
    if capacity!=int(prior_receipt["latent_dim_only_computational_budget"]):
        raise ValueError("Baseline rank capacity differs from frozen V10 run")
    if len(development.y)!=300 or len(development.x[0])!=36:
        raise ValueError("Historical comparison cohort/time horizon altered")
    identity={
        "n_official_set_a_patients":4000,
        "training_patients":2400,
        "development_patients_previously_seen":300,
        "original_validation_patients":600,
        "old_test_sealed":600,
        "old_calibration_sealed":400,
        "observation_hours":36,
        "forecast_horizon_hours_only_for_v10":12,
        "channel_names":list(observed.receipt["channel_names"]),
        "n_channels":len(observed.receipt["channel_names"]),
        "physiological_K_only_computational_budget":capacity,
        "development_episode_index_sha256":hashlib.sha256(
            np.ascontiguousarray(development.episode_ids,dtype="<i8").tobytes()
        ).hexdigest(),
        "train_only_sensor_rank_and_normalization":True,
        "no_prior_version_compared":True,
        "published_reference_run":"41623853277",
        "previously_viewed_development_not_confirmatory":True,
    }
    return observed,rank.left_observable_loadings[:,:capacity],development,identity


def _model_score(y:np.ndarray,p:np.ndarray)->dict:
    y=np.asarray(y,dtype=np.float64)
    p=np.asarray(p,dtype=np.float64)
    if y.ndim!=1 or p.shape!=y.shape or np.unique(y).size!=2:
        raise ValueError("Scoring requires both classes on same patients")
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError("A valid probability for each evaluation case is required")
    return {
        "auprc":float(average_precision_score(y,p)),
        "auroc":float(roc_auc_score(y,p)),
        "brier":float(brier_score_loss(y,p)),
        "n":len(y),
        "n_positive":int(y.sum()),
        "prevalence":float(y.mean()),
    }


def _perturbation_views(development,seed:int)->dict:
    views={}
    for i,scenario in enumerate(SCENARIOS):
        if scenario=="original":
            values,mask=development.x,development.mask
        else:
            values,mask,_=apply_measurement_environment(
                development.x,development.mask,
                environment=scenario,seed=int(seed)+i,
            )
        views[scenario]=(values,mask,development.mask)
    return views


def _paired_bootstrap(
    y:np.ndarray,
    candidate:dict[int,np.ndarray],
    external:dict[int,np.ndarray],
    *,
    repetitions:int,seed:int,
)->dict:
    """Patient-cluster paired resamples AND independent training-run resamples.

    Descriptive, conditional on a repeatedly selected 300-case cohort and
    a fixed method/hyperparameter search. Never use it as a confirmation.
    """
    if repetitions<100:
        raise ValueError("Too few finite descriptive bootstrap draws")
    y=np.asarray(y,dtype=int)
    positives=np.flatnonzero(y==1)
    negatives=np.flatnonzero(y==0)
    if not len(positives) or not len(negatives):
        raise ValueError("Both clinical classes required")
    for methods in (candidate,external):
        if not methods or any(len(p)!=len(y) for p in methods.values()):
            raise ValueError("Every independent run must share identical patient set")
    rng=np.random.default_rng(seed)
    c_list=np.stack([candidate[k] for k in sorted(candidate)])
    b_list=np.stack([external[k] for k in sorted(external)])
    observed_c=float(np.mean([average_precision_score(y,p) for p in c_list]))
    observed_b=float(np.mean([average_precision_score(y,p) for p in b_list]))
    sampled=[]
    for _ in range(int(repetitions)):
        # Paired within-stratum patient indices retain the original disease
        # prevalence. One shared sample is used for both architectures.
        take=np.r_[rng.choice(positives,len(positives),replace=True),
                   rng.choice(negatives,len(negatives),replace=True)]
        run_c=rng.choice(len(c_list),size=len(c_list),replace=True)
        run_b=rng.choice(len(b_list),size=len(b_list),replace=True)
        val_c=np.mean([
            average_precision_score(y[take],c_list[idx,take])
            for idx in run_c
        ])
        val_b=np.mean([
            average_precision_score(y[take],b_list[idx,take])
            for idx in run_b
        ])
        sampled.append(float(val_c-val_b))
    return {
        "delta_auprc_v10_minus_external":observed_c-observed_b,
        "descriptive_patient_and_training_run_bootstrap_ci95":
            [float(q) for q in np.quantile(sampled,[.025,.975])],
        "paired_patient_resample":True,
        "one_fixed_validation_population":True,
        "model_seed_resample":True,
        "external_distinct_models":len(b_list),
        "v10_distinct_models":len(c_list),
        "repetitions":int(repetitions),
        "does_not_include_method_selection_or_external_hospitals":True,
        "confirmatory_p_value":None,
    }


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True,type=Path)
    parser.add_argument("--reference-v6-config",required=True,type=Path)
    parser.add_argument("--reference-v7-config",required=True,type=Path)
    parser.add_argument("--reference-v10-config",required=True,type=Path)
    parser.add_argument("--frozen-v10-receipt",required=True,type=Path)
    parser.add_argument("--raw-zip",required=True,type=Path)
    parser.add_argument("--outcomes",required=True,type=Path)
    parser.add_argument("--out-dir",required=True,type=Path)
    args=parser.parse_args()
    if args.out_dir.exists():
        parser.error("Never overwrite a prior comparison")
    cfg=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base=yaml.safe_load(args.reference_v6_config.read_text(encoding="utf-8"))
    v7=yaml.safe_load(args.reference_v7_config.read_text(encoding="utf-8"))
    v10=yaml.safe_load(args.reference_v10_config.read_text(encoding="utf-8"))
    prior=json.loads(args.frozen_v10_receipt.read_text(encoding="utf-8"))
    if not (
        cfg["protocol"]["no_v7_v8_v9_comparisons"]
        and cfg["protocol"]["old_test_and_calibration_sealed"]
        and cfg["protocol"]["no_external_claim_of_significance"]
        and prior["status"]=="EXPLORATORY_V10_MATCHED_TRAINING_COMPLETE"
        and base["protocol"]["test_and_calibration_sealed"]
        and v10["protocol"]["test_calibration_sealed"]
        and v7["protocol"]["no_test_calibration_or_new_clinical_significance"]
    ):
        parser.error("Original V10 cohort or statistical safety contract violated")
    if tuple(cfg["baselines"]["external_methods"])!=EXTERNAL_METHODS:
        parser.error("All four predeclared independent baselines are required")
    seeds=tuple(int(x) for x in cfg["training"]["initialization_seeds"])
    if seeds!=(11,23,37) or seeds!=tuple(v10["training"]["initialization_seeds"]):
        parser.error("Use same frozen three V10 initialization seeds")
    torch.set_num_threads(int(base["training"]["cpu_threads"]))
    data,basis,dev,cohort=_load_and_match_v10_patient_cohort(
        source_zip=args.raw_zip,outcome_file=args.outcomes,
        base_cfg=base,prior_receipt=prior,
    )
    if len(data.train.y)!=2400 or len(dev.y)!=300:
        raise AssertionError("Matched clinical patient counts have changed")
    views=_perturbation_views(
        dev,int(base["evaluation"]["measurement_mask_seed"])
    )
    args.out_dir.mkdir(parents=True,exist_ok=False)
    evidence={}
    fit_records=[]
    scores=[]
    methods=(V10_METHOD,)+EXTERNAL_METHODS
    for method in methods:
        is_deterministic=method in ("logistic_history","hist_gradient_boosting")
        model_seeds=(0,) if is_deterministic else seeds
        evidence[method]={}
        for seed in model_seeds:
            started=time.perf_counter()
            if method==V10_METHOD:
                model,training=fit_multiscale(
                    data,basis,seed,base,v7,v10,consistency=True
                )
                params=sum(p.numel() for p in model.parameters())
            elif is_deterministic:
                model=fit_tabular_baseline(
                    method,data.train.x,data.train.mask,data.train.mask,
                    data.train.y,
                    augmentation_fraction=float(cfg["training"]["augmentation_fraction"]),
                    seed=int(cfg["training"]["tabular_training_seed"]),
                )
                training={"architecture":method,"epochs":None,
                          "tabular_augmented_duplicate_training_view":True}
                params=None
            else:
                model,training=fit_neural_baseline(
                    method,data.train,seed=seed,
                    epochs=int(cfg["training"]["epochs"]),
                    batch_size=int(cfg["training"]["batch_size"]),
                    learning_rate=float(cfg["training"]["learning_rate"]),
                    weight_decay=float(cfg["training"]["weight_decay"]),
                    augmentation_fraction=float(cfg["training"]["augmentation_fraction"]),
                )
                params=int(training["parameters"])
            fit_elapsed=time.perf_counter()-started
            predictions={}
            timings=[]
            for scenario,(values,visible,source_mask) in views.items():
                started=time.perf_counter()
                if method==V10_METHOD:
                    with torch.no_grad():
                        result=model(
                            torch.from_numpy(values),
                            torch.from_numpy(visible),
                            process_mask=torch.from_numpy(source_mask),
                        )
                        p=result.probabilities.detach().cpu().numpy().astype(np.float64)
                        assert float(result.decomposition_error().detach())<1e-5
                elif is_deterministic:
                    p=predict_tabular(model,values,visible,source_mask)
                else:
                    p=predict_neural(model,values,visible,source_mask)
                elapsed=time.perf_counter()-started
                predictions[scenario]=p
                timings.append(elapsed)
                scores.append({
                    "model":method,"seed":seed,"scenario":scenario,
                    "train_patients":len(data.train.y),
                    "development_patients":len(dev.y),
                    "n_observed_sensors":dev.x.shape[-1],
                    "observed_hours":dev.x.shape[1],
                    "train_seconds":fit_elapsed,
                    "inference_seconds_for_300":elapsed,
                    "trainable_parameters":params,
                    "original_recorded_mask_given_to_all":True,
                    "old_test_and_calibration_used":False,
                    **_model_score(dev.y,p),
                })
            evidence[method][seed]=predictions
            fit_records.append({
                "method":method,"seed":seed,"fit_seconds":fit_elapsed,
                "parameters":params,**training,
            })
            print(method,"seed",seed,"original AUPRC",round(
                _model_score(dev.y,predictions["original"])["auprc"],4
            ),"fit_s",round(fit_elapsed,2),flush=True)

    raw=pd.DataFrame(scores)
    fit=pd.DataFrame(fit_records)
    grouped=[]
    for method in methods:
        subset=raw[raw.model.eq(method)]
        for scenario in SCENARIOS:
            sample=subset[subset.scenario.eq(scenario)]
            grouped.append({
                "model":method,"scenario":scenario,
                "n_independent_model_initializations":len(sample),
                "mean_auprc":float(sample.auprc.mean()),
                "sd_across_initializations_auprc":float(
                    sample.auprc.std(ddof=1)) if len(sample)>1 else None,
                "mean_auroc":float(sample.auroc.mean()),
                "mean_brier_raw_unadjusted":float(sample.brier.mean()),
                "mean_train_seconds":float(sample.train_seconds.mean()),
                "mean_inference_seconds_for_300":float(
                    sample.inference_seconds_for_300.mean()
                ),
                "patient_prevalence":float(sample.prevalence.iloc[0]),
                "input_channels":int(sample.n_observed_sensors.iloc[0]),
                "history_hours":int(sample.observed_hours.iloc[0]),
                "ranking_is_exploratory":True,
            })
    comparisons=[]
    for scenario in SCENARIOS:
        for j,method in enumerate(EXTERNAL_METHODS):
            stats=_paired_bootstrap(
                dev.y,
                {seed:evidence[V10_METHOD][seed][scenario] for seed in seeds},
                {seed:evidence[method][seed][scenario] for seed in evidence[method]},
                repetitions=int(cfg["statistics"]["patient_bootstrap_repetitions"]),
                seed=int(cfg["statistics"]["seed"])+100*SCENARIOS.index(scenario)+j,
            )
            comparisons.append({
                "scenario":scenario,
                "reference":"ConceptFAN-V10",
                "other_model":method,
                **stats,
            })
    raw.to_csv(args.out_dir/"v10_vs_external_all_run_metrics.csv",index=False)
    pd.DataFrame(grouped).to_csv(args.out_dir/"v10_vs_external_summary.csv",index=False)
    pd.DataFrame(comparisons).to_csv(
        args.out_dir/"v10_vs_external_descriptive_paired_bootstrap.csv",index=False
    )
    fit.to_csv(args.out_dir/"v10_vs_external_training_runtime.csv",index=False)
    (args.out_dir/"STATUS.json").write_text(json.dumps({
        "status":"V10_EXTERNAL_ARCHITECTURE_BASELINES_DEV_COMPLETED",
        "comparison_methods":list(methods),
        "study_basis":"same official PhysioNet 2012 Set A train/36-hour historical dev",
        "frozen_clinical_cohort":cohort,
        "standard_rnn_type":"locally implemented GRU-D-style, NOT official published weights",
        "transformer_type":"locally trained two-layer time-series Transformer, no pretrained weights",
        "tabular_methods":"StandardScaler LogisticRegression / sklearn HistGradientBoosting",
        "confidence_intervals":"descriptive fixed development patients, resampled patient IDs and available model seeds",
        "test_used":False,
        "calibration_used":False,
        "independent_external_hospital_test":False,
        "new_method_selected_on_repeated_development":True,
        "no_independent_scientific_significance_claim":True,
        "no_patient_level_predictions_published":True,
        "results":grouped,
        "paired_descriptive":comparisons,
    },indent=2,allow_nan=False),encoding="utf-8")
    print(pd.DataFrame(grouped).to_string(index=False),flush=True)


if __name__=="__main__":
    main()
