"""Train-time unsupervised spectral concepts with exact logit decomposition.

Compare three predeclared approaches on NEW synthetic generators:
  learned_10: original v2 concept discovery with 10 freely trained slots;
  spectral_frozen: TRAIN-only detectable temporal rank and fixed signed
      measured-channel definitions, with task/future/denoising heads trained;
  spectral_finetuned: same K and initialization, feature definitions allowed
      to adapt under regularized deviation and orthogonality.

No planted-state supervision enters spectral fit or neural optimization.
All evidence is exploratory because the proposed spectral basis was developed
after seeing prior negative v1/v2 pilot results.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import yaml

from .denoising_experiment import fit_one as fit_v2
from .evaluation import (
    compare_independent_discoveries, concept_passports, evaluate_discovery,
    oracle_correspondence,
)
from .lagged_rank import fit_lagged_predictive_rank
from .model_v2 import masked_observation_view, v2_loss
from .selection import split_validation_for_cardinality
from .selection_v2 import select_v2_cardinality
from .spectral_concepts_v3 import SpectralConceptFAN
from .variable_factor_data import generate_variable_factor_data


def initialize_all(seed:int)->None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def fit_spectral(
    method:str,
    seed:int,
    dataset,
    cfg:dict,
    left_basis:np.ndarray,
)->SpectralConceptFAN:
    initialize_all(seed)
    training=cfg["training"]
    model=SpectralConceptFAN(
        observable_loadings=left_basis,
        channels=tuple(dataset.receipt["channel_names"]),
        forecast_horizon=dataset.train.future.shape[1],
        freeze_definitions=method=="spectral_frozen",
    )
    active=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(
        active,lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    positives=float(dataset.train.y.sum())
    pos_weight=torch.tensor(
        (len(dataset.train.y)-positives)/max(positives,1.),
        dtype=torch.float32,
    )
    with torch.no_grad():
        model.decision_bias.fill_(
            float(np.log(np.clip(positives/(len(dataset.train.y)-positives),1e-5,1e5)))
        )
    loader=dataset.train.loader(
        int(training["batch_size"]),True,int(training["order_seed"])
    )
    random_masking=torch.Generator().manual_seed(seed+51019)
    for epoch in range(int(training["epochs"])):
        model.train()
        for x,mask,future,future_mask,label in loader:
            optimizer.zero_grad(set_to_none=True)
            corrupted,visible,withheld=masked_observation_view(
                x,mask,fraction=float(training["mask_fraction"]),
                generator=random_masking,
            )
            output=model(corrupted,visible)
            objective,parts=v2_loss(
                model,output,label,future,future_mask,x,withheld,
                mode="full",pos_weight=pos_weight,
                loss_weights=training["loss_weights"],
            )
            if method=="spectral_finetuned":
                penalty=model.spectral_definition_penalty()
                extra=training["spectral_regularizers"]
                objective=(
                    objective
                    +float(extra["alignment"])*penalty["operator_displacement"]
                    +float(extra["axis_orthogonality"])*penalty["observable_axis_nonorthogonality"]
                )
            if not torch.isfinite(objective):
                raise FloatingPointError(f"Nonfinite {method} seed={seed} epoch={epoch}")
            objective.backward()
            torch.nn.utils.clip_grad_norm_(
                active,float(training["gradient_clip"])
            )
            optimizer.step()
    model.eval()
    return model


def audit_denoising_rmse(model,split,seed:int=20261291)->float:
    generator=torch.Generator().manual_seed(int(seed))
    x=torch.from_numpy(split.x)
    mask=torch.from_numpy(split.mask)
    damaged,visible,withheld=masked_observation_view(
        x,mask,fraction=.2,generator=generator
    )
    with torch.no_grad():
        out=model(damaged,visible)
        return float(
            (((out.reconstruction-x).square()*withheld).sum() /
             withheld.sum().clamp_min(1.)).sqrt()
        )


def write(path:Path,value:dict|list)->None:
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    if args.out_dir.exists():
        p.error("Refusing to overwrite historical experimental records")
    cfg=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not cfg["protocol"]["no_test_evaluation"] or not cfg["evaluation"]["no_oracle_in_optimization"]:
        p.error("No-test and no-oracle training contracts required")
    if set(cfg["training"]["methods"])!={"learned_10","spectral_frozen","spectral_finetuned"}:
        p.error("All independent baseline and spectral ablations are required")
    torch.set_num_threads(int(cfg["training"]["cpu_threads"]))
    args.out_dir.mkdir(parents=True,exist_ok=False)
    rows,models_comparisons=[],[]
    rank_receipts={}
    all_methods=tuple(cfg["training"]["methods"])
    for planted_k in cfg["data"]["planted_factors_for_evaluation_only"]:
        dataset=generate_variable_factor_data(
            n_true_factors=int(planted_k),
            seed=int(cfg["data"]["generator_seed_base"])+int(planted_k),
            split_seed=int(cfg["data"]["split_seed_base"])+int(planted_k),
            n_samples=int(cfg["data"]["n_samples"]),
            n_channels=int(cfg["data"]["n_channels"]),
            input_steps=int(cfg["data"]["input_steps"]),
            horizon=int(cfg["data"]["horizon"]),
        )
        spectral=cfg["spectral"]
        # IMPORTANT: the rank algorithm sees TRAIN OBSERVATIONS ONLY, not
        # planted K, labels, validation or other oracle state coordinates.
        detected=fit_lagged_predictive_rank(
            dataset.train.x,dataset.train.mask,
            lags=tuple(spectral["lags"]),
            permutations=int(spectral["permutations"]),
            patient_bootstraps=int(spectral["patient_bootstraps"]),
            alpha=float(spectral["alpha"]),
            seed=int(spectral["seed"])+int(planted_k),
        )
        rank_receipts[f"planted_{planted_k}"]=detected.to_json()
        if detected.rank<2:
            raise RuntimeError(
                f"Train-observed dynamic rank={detected.rank}. Reject, do not invent concepts."
            )
        left_basis=detected.left_observable_loadings[:,:detected.rank]
        selection,audit=split_validation_for_cardinality(
            dataset.validation,
            random_seed=int(cfg["evaluation"]["inner_outer_validation_split_seed"]),
        )
        by_method={}
        for method in all_methods:
            by_method[method]={}
            for init in cfg["training"]["init_seeds"]:
                initial=int(init)
                if method=="learned_10":
                    model=fit_v2("full",initial,dataset,cfg)
                    # Existing v2 count determined on the private SELECTION
                    # half only; the spectral arms use K found on TRAIN only.
                    choice=select_v2_cardinality(
                        model,selection,audit,mode="full",
                        max_auprc_drop=float(cfg["evaluation"]["baseline_auprc_tolerance"]),
                        max_nll_increase=float(cfg["evaluation"]["baseline_nll_tolerance"]),
                        max_future_rmse_increase=float(cfg["evaluation"]["baseline_future_tolerance"]),
                        max_masked_rmse_increase=float(cfg["evaluation"]["baseline_denoise_tolerance"]),
                    )
                    picked=torch.from_numpy(choice.selected_mask)
                    selected_k=int(choice.selected_mask.sum())
                    with torch.no_grad():
                        e=evaluate_discovery(model,audit,selected_slots=picked)
                    # To evaluate baseline masked reconstruction, use the
                    # same pruned subset (not the unpruned full v2 model).
                    generator=torch.Generator().manual_seed(20261291)
                    x=torch.from_numpy(audit.x)
                    mask=torch.from_numpy(audit.mask)
                    corrupted,visible,withheld=masked_observation_view(
                        x,mask,fraction=.20,generator=generator,
                    )
                    with torch.no_grad():
                        reconstructed=model(corrupted,visible,selected_slots=picked).reconstruction
                        masked_rmse=float(
                            (((reconstructed-x).square()*withheld).sum() /
                             withheld.sum().clamp_min(1.)).sqrt()
                        )
                else:
                    model=fit_spectral(method,initial,dataset,cfg,left_basis)
                    selected_k=int(model.n_slots)
                    with torch.no_grad():
                        e=evaluate_discovery(model,audit)
                    masked_rmse=audit_denoising_rmse(model,audit)
                oracle=oracle_correspondence(e,audit.oracle_states)
                definition=e.weights
                per_patient_contrib=e.contributions
                if not np.isfinite(per_patient_contrib).all():
                    raise FloatingPointError("Nonfinite audited concept contributions")
                record={
                    "hidden_factor_count_for_external_audit_only":int(planted_k),
                    "spectral_detected_k_train_only":int(detected.rank),
                    "method":method,"initialization_seed":initial,
                    "chosen_native_concepts":selected_k,
                    "active_nonconstant_concepts":int(e.metrics["n_active_and_nonconstant_slots"]),
                    "outer_validation_auprc":float(e.metrics["auprc"]),
                    "outer_validation_future_rmse":float(e.metrics["future_observed_rmse"]),
                    "outer_validation_denoising_rmse":masked_rmse,
                    "oracle_factor_coverage":float(oracle["oracle_factor_coverage"]),
                    "oracle_coverage_adjusted_abs_corr":float(oracle["oracle_coverage_adjusted_abs_corr"]),
                    "max_logit_additivity_error":float(e.metrics["max_additive_error"]),
                    "human_concept_labels_used":False,
                    "test_used":False,
                }
                rows.append(record)
                by_method[method][initial]=e
                if initial==int(cfg["training"]["init_seeds"][0]):
                    key=f"truth_{planted_k}_{method}"
                    selected_passports=concept_passports(
                        e,model,audit,dataset.medians,dataset.scales,
                        include_examples=False,
                    )
                    write(args.out_dir/f"concept_definitions_{key}.json",
                          {"observed_rank":detected.rank,
                           "training_label_free":True,
                           "concepts":selected_passports})
                print(
                    "oracleK",planted_k,"rankK",detected.rank,method,
                    "seed",initial,"AUCPR",round(record["outer_validation_auprc"],4),
                    "true-state-coverage",round(record["oracle_factor_coverage"],3),
                    flush=True,
                )
            for seed_a,seed_b in itertools.combinations(
                sorted(by_method[method]),2
            ):
                pair=compare_independent_discoveries(
                    by_method[method][seed_a],by_method[method][seed_b]
                )
                models_comparisons.append({
                    "hidden_factor_count_for_external_audit_only":int(planted_k),
                    "method":method,"init_seed_a":seed_a,"init_seed_b":seed_b,
                    **pair,
                })
    results=pd.DataFrame(rows)
    comparisons=pd.DataFrame(models_comparisons)
    results.to_csv(args.out_dir/"spectral_native_v3_per_run.csv",index=False)
    comparisons.to_csv(args.out_dir/"spectral_native_v3_between_seeds.csv",index=False)
    summary=[]
    for (truth,method),group in results.groupby(
        ["hidden_factor_count_for_external_audit_only","method"]
    ):
        pair=comparisons.loc[
            comparisons["hidden_factor_count_for_external_audit_only"].eq(truth)
            & comparisons["method"].eq(method)
        ]
        rho=pair["excess_over_unmatched_episode_null_spearman"].dropna()
        summary.append({
            "true_k_evaluation_only":int(truth),"method":method,
            "runs":len(group),
            "train_only_rank":int(group.spectral_detected_k_train_only.iloc[0]),
            "mean_selected_k":float(group.chosen_native_concepts.mean()),
            "mean_effectively_varying_k":float(group.active_nonconstant_concepts.mean()),
            "mean_outer_auprc":float(group.outer_validation_auprc.mean()),
            "mean_outer_future_rmse":float(group.outer_validation_future_rmse.mean()),
            "mean_outer_denoise_rmse":float(group.outer_validation_denoising_rmse.mean()),
            "mean_oracle_factor_coverage":float(group.oracle_factor_coverage.mean()),
            "mean_coverage_adjusted_oracle_corr":float(group.oracle_coverage_adjusted_abs_corr.mean()),
            "mean_case_specific_excess_spearman":float(rho.mean()) if len(rho) else None,
            "between_seed_pairs":len(pair),
        })
    pd.DataFrame(summary).to_csv(args.out_dir/"spectral_native_v3_summary.csv",index=False)
    write(args.out_dir/"train_only_rank_receipts.json",rank_receipts)
    write(args.out_dir/"STATUS.json",{
        "status":"V3_SPECTRAL_NATIVE_EXPLORATORY_COMPARISON",
        "train_only_rank_estimator":True,
        "observables_only_no_expert_concept_labels":True,
        "heldout_synthetic_oracle_used_only_after_fit_for_audit":True,
        "test_ever_evaluated":False,
        "comparisons":summary,
        "not_claimed":[
            "medically validated semantic concepts",
            "exact identifiability for general nonlinear missing clinical series",
            "causal interventions on patients",
            "statistical significance with only four independent initializations",
            "shared frozen basis itself proving nontrivial explanation reproducibility",
        ],
    })
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


if __name__=="__main__":
    main()
