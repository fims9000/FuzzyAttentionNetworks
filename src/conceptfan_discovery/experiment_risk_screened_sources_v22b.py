"""V22b: distinguish recovered concepts from independently predictive concepts.

Compare two observed source bases (PCA, environment-separated) under the
exact SAME 600-core-TRAIN native V10 model, with or without a dedicated
200-case independent TRAIN source-risk screen. This never trains a post-hoc
risk model; only original native decision coefficients and structural no-risk
policy code 2 are changed at INITIALIZATION.

Validation oracle physical/artifact identity is evaluated after ALL model
fitting; no synthetic TEST is generated or unsealed.
"""
from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .mixed_source_data_v22 import (
    FAMILIES,SEEDS,MODEL_CHANNELS,make_observable_v22,
    train_only_whitening,build_native_v22_data,
    load_oracle_v22_validation_for_postfit_audit,
)
from .environmental_unmixing_v22 import (
    fit_train_only_environment_source_directions,
    native_basis_from_separation,separation_passport,
)
from .native_source_risk_screen_v22b import (
    screen_v22_native_sources_on_disjoint_train_labels,
    risk_plan_passport,
)
from .experiment_mixed_source_v22 import (
    _config,_native_readout,_oracle_alignment_no_training_feedback,
    _rho,
)
from .clinical_multiscale_v10 import fit_multiscale


MODEL_SEEDS=(11,23,37)
SOURCE_METHODS=("pooled_pca_basis","environment_contrast_basis")
RISK_ARMS=("arbitrary_native_start","independently_screened_risk")
CORE_TRAIN_N=600
INDEPENDENT_RISK_SCREEN_N=200
TRAIN_EPOCHS=10
BATCH_SIZE=96


def run(outdir:Path):
    if outdir.exists():
        raise FileExistsError("Never overwrite frozen V22b scientific evidence")
    torch.set_num_threads(1)
    opportunities=[]
    source_receipts={}
    risk_receipts={}
    signed_by_family_method={}
    for family in FAMILIES:
        original=make_observable_v22(family)
        if len(original.train_episode_ids)!=(CORE_TRAIN_N+INDEPENDENT_RISK_SCREEN_N):
            raise AssertionError("Unregistered TRAIN/core/source cohort denominator")
        # Only these 600 TRAIN episodes may fit normalization, observable
        # source subspace and the actual native prediction model.
        core=replace(
            original,
            train_x_full=original.train_x_full[:CORE_TRAIN_N].copy(),
            train_environment=original.train_environment[:CORE_TRAIN_N].copy(),
            train_labels=original.train_labels[:CORE_TRAIN_N].copy(),
            train_episode_ids=original.train_episode_ids[:CORE_TRAIN_N].copy(),
        )
        whitener=train_only_whitening(core)
        sep=fit_train_only_environment_source_directions(
            core,whiten=whitener,
            seed=522221,permutations=199,bootstraps=48,
        )
        source_receipts[family]=separation_passport(sep)
        data=build_native_v22_data(core,whitener)
        if set(data.train.episode_ids) & set(original.train_episode_ids[CORE_TRAIN_N:]):
            raise AssertionError("Source/target risk screening reused native TRAIN cohort")
        if (data.train.oracle_states is not None
                or data.validation.oracle_states is not None):
            raise AssertionError("Source oracle accidentally exposed")
        screen_observed=whitener.transform(
            original.train_x_full[CORE_TRAIN_N:],
        )[:,:24]
        screen_masks=np.ones_like(screen_observed,dtype=np.float32)
        screen_labels=original.train_labels[CORE_TRAIN_N:]
        screen_ids=original.train_episode_ids[CORE_TRAIN_N:]

        pca=np.eye(len(MODEL_CHANNELS),2,dtype=np.float64)
        source_basis={
            "pooled_pca_basis":pca,
            "environment_contrast_basis":(
                native_basis_from_separation(sep)
                if sep.status=="ACCEPT_OBSERVATIONAL_ENVIRONMENT_UNMIXING"
                else None
            ),
        }
        config,dual,v10=_config()
        config["training"]["epochs"]=TRAIN_EPOCHS
        config["training"]["batch_size"]=BATCH_SIZE
        fitted=[]
        signed_by_family_method[family]={}
        for method in SOURCE_METHODS:
            basis=source_basis[method]
            risk=None
            risk_receipts[f"{family}__{method}"]=None
            signed_by_family_method[family][method]={arm:[] for arm in RISK_ARMS}
            if basis is not None:
                risk=screen_v22_native_sources_on_disjoint_train_labels(
                    screen_observed,screen_masks,screen_labels,basis,
                    original_source_train_episode_ids=data.train.episode_ids,
                    screen_episode_ids=screen_ids,
                    seed=522231,permutations=199,bootstraps=199,
                )
                risk_receipts[f"{family}__{method}"]=risk_plan_passport(risk)

            for arm in RISK_ARMS:
                for seed in MODEL_SEEDS:
                    if basis is None:
                        opportunities.append({
                            "family":family,"source_method":method,
                            "risk_init_arm":arm,"seed":seed,
                            "status":sep.status,"trained":False,
                            "validation_auprc":None,
                            "validation_auroc":None,
                            "validation_brier":None,
                            "max_native_logit_decomposition_error":None,
                            "oracle_alignment_postfit_eval_only":None,
                            "physical_identity_established":False,
                        })
                        continue
                    is_screened=arm=="independently_screened_risk"
                    init=(risk.active_native_raw_decision_initialization
                          if is_screened else None)
                    descriptive=(risk.structurally_descriptive_only_native_slots
                                 if is_screened else None)
                    model,receipt=fit_multiscale(
                        data,basis,seed,config,dual,v10,
                        consistency=False,
                        feature_exploration=False,
                        expert_feedback=None,
                        native_head_refit=False,
                        initial_native_raw_decision_weights=(
                            dict(init) if init else None
                        ),
                        initial_native_descriptive_only_slots=(
                            tuple(descriptive) if descriptive else None
                        ),
                    )
                    if is_screened:
                        for slot in (0,1):
                            expected=2 if slot in descriptive else 0
                            actual=int(model.physical.signed_risk_policy[slot])
                            if expected!=actual:
                                raise AssertionError(
                                    "TRAIN-only source-risk gate not persisted in native model"
                                )
                            if expected==2 and float(model.physical.decision_weights()[slot])!=0:
                                raise AssertionError("Non-risk concept leaked to native prediction")
                    else:
                        if not (model.physical.signed_risk_policy==0).all():
                            raise AssertionError("Original unrestricted native baseline changed")
                    stats,activations,effects=_native_readout(model,data.validation)
                    row={
                        "family":family,"source_method":method,
                        "risk_init_arm":arm,"seed":seed,
                        "status":"TRAINED","trained":True,
                        "source_identifiability_status":sep.status,
                        "n_core_train":CORE_TRAIN_N,
                        "n_disjoint_risk_screen":INDEPENDENT_RISK_SCREEN_N,
                        "initial_source_risk_signs":(
                            {str(k):v for k,v in init.items()}
                            if is_screened and init else {}
                        ),
                        "native_descriptive_risk_zero_slots":(
                            list(descriptive) if is_screened and descriptive else []
                        ),
                        "risk_screen_disjoint_from_model_fit":True,
                        "no_oracle_used_in_native_training":True,
                        "oracle_alignment_postfit_eval_only":None,
                        "physical_identity_established":False,
                        **stats,
                    }
                    opportunities.append(row)
                    fitted.append((row,activations))
                    signed_by_family_method[family][method][arm].append(effects)
                    print(family,method,arm,seed,
                          "AUPRC",round(stats["validation_auprc"],6),
                          "excluded",row["native_descriptive_risk_zero_slots"],
                          flush=True)
        # Only AFTER all model fits and risk-screen actions have completed:
        # hidden synthetic source truth may be inspected for audit metrics.
        truth=load_oracle_v22_validation_for_postfit_audit(family)
        true_physical=truth["generated_physical_source"][:,-6:].mean(1)
        true_instrument=truth["generated_instrument_source"][:,-6:].mean(1)
        env=original.validation_environment
        for row,activation in fitted:
            row["oracle_alignment_postfit_eval_only"]=_oracle_alignment_no_training_feedback(
                activation,true_physical,true_instrument,env,
            )
        del truth,true_physical,true_instrument,fitted
    summary=[]
    for family in FAMILIES:
        for method in SOURCE_METHODS:
            for arm in RISK_ARMS:
                rows=[r for r in opportunities if r["family"]==family
                      and r["source_method"]==method
                      and r["risk_init_arm"]==arm]
                completed=[r for r in rows if r["trained"]]
                effect=signed_by_family_method[family][method][arm]
                pairs=[
                    _rho(effect[i][:,k],effect[j][:,k])
                    for i in range(len(effect)) for j in range(i+1,len(effect))
                    for k in (0,1)
                ]
                summary.append({
                    "family":family,"source_method":method,
                    "risk_init_arm":arm,
                    "opportunities":len(rows),
                    "trained":len(completed),
                    "abstentions":len(rows)-len(completed),
                    "mean_validation_auprc":(
                        float(np.mean([r["validation_auprc"] for r in completed]))
                        if completed else None
                    ),
                    "mean_validation_brier":(
                        float(np.mean([r["validation_brier"] for r in completed]))
                        if completed else None
                    ),
                    "mean_oracle_two_source_alignment_postfit":(
                        float(np.mean([
                            r["oracle_alignment_postfit_eval_only"]["mean_abs_oracle_source_spearman_eval_only"]
                            for r in completed
                        ])) if completed else None
                    ),
                    "mean_signed_patient_contribution_crossseed_spearman_no_alignment":(
                        float(np.mean(pairs)) if pairs else None
                    ),
                    "train_only_risk_plan":risk_receipts[f"{family}__{method}"],
                    "clinical_source_correctness_proven":False,
                })
    if len(opportunities)!=48 or len(summary)!=16 or len(source_receipts)!=4:
        raise AssertionError("Incomplete V22b preregistered experiment denominator")
    outdir.mkdir(parents=True,exist_ok=False)
    for filename,records in (
        ("ALL_48_TRAIN_SIGNED_SOURCE_OPPORTUNITIES.csv",opportunities),
        ("ALL_16_MODEL_ARM_SUMMARIES.csv",summary),
    ):
        keys=sorted({k for r in records for k in r})
        with (outdir/filename).open("w",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=keys)
            writer.writeheader()
            for row in records:
                writer.writerow({
                    k:json.dumps(row.get(k),sort_keys=True) if isinstance(row.get(k),(dict,list))
                    else row.get(k)
                    for k in keys
                })
    (outdir/"TRAIN_ONLY_RISK_PLAN_PASSports.json").write_text(
        json.dumps(risk_receipts,indent=2,allow_nan=False)
    )
    (outdir/"SOURCE_IDENTIFIABILITY_CORE_TRAIN_PASSports.json").write_text(
        json.dumps(source_receipts,indent=2,allow_nan=False)
    )
    status={
        "status":"V22B_DISJOINT_TRAIN_SCREENED_NATIVE_SOURCE_RISK_ADMISSION",
        "number_registered_opportunities":len(opportunities),
        "number_fitted_native_models":sum(r["trained"] for r in opportunities),
        "number_identifiability_abstentions":sum(not r["trained"] for r in opportunities),
        "n_generator_families":len(FAMILIES),
        "n_initialization_seeds":len(MODEL_SEEDS),
        "source_core_train_n":CORE_TRAIN_N,
        "risk_independent_train_n":INDEPENDENT_RISK_SCREEN_N,
        "all_source_and_task_risk_labels_disjoint":True,
        "native_description_risk_veto_persists_across_training":True,
        "same_native_model_and_obs_after_screening":True,
        "no_external_logit_predictor_or_oracle_teacher":True,
        "synthetic_test_seed_522203_not_evaluated":True,
        "all_synthetic_validation_families_previously_inspected_in_V22a":True,
        "no_physically_validated_concepts_claimed":True,
        "all_negative_and_abstained_cases_retained":True,
        "per_fit_or_abstention":opportunities,
        "summary":summary,
        "source_evidence":source_receipts,
        "task_risk_evidence":risk_receipts,
    }
    (outdir/"STATUS.json").write_text(json.dumps(status,indent=2,allow_nan=False))
    return status


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",required=True,type=Path)
    args=parser.parse_args()
    run(args.out_dir)
