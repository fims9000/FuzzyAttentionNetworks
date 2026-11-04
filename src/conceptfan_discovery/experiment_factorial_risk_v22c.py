"""V22c: separate native risk-weight sign and descriptive-only risk masking.

All four arms begin from the exact same native source basis and original
torch seed, differing ONLY in the registered initial decision coefficient
and/or serialized zero-risk structural policy. This is a controlled
within-synthetic-family *factorial mechanism ablation*, not medical proof.
"""
from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from .mixed_source_data_v22 import (
    FAMILIES,MODEL_CHANNELS,make_observable_v22,
    train_only_whitening,build_native_v22_data,
    load_oracle_v22_validation_for_postfit_audit,
)
from .environmental_unmixing_v22 import (
    fit_train_only_environment_source_directions,
    native_basis_from_separation,separation_passport,
)
from .native_source_risk_screen_v22b import (
    screen_v22_native_sources_on_disjoint_train_labels,risk_plan_passport,
)
from .experiment_mixed_source_v22 import _native_readout,_oracle_alignment_no_training_feedback
from .experiment_native_risk_v18 import _config
from .clinical_multiscale_v10 import fit_multiscale


MODEL_SEEDS=(11,23,37)
ARMS=(
    "random_original",
    "signed_start_only",
    "descriptive_mask_only",
    "signed_start_and_mask",
)
CORE_TRAIN=600
RISK_SCREEN=200
EPOCHS=10
BATCH=96


def _valid_spearman(a,b):
    a=np.asarray(a,dtype=np.float64)
    b=np.asarray(b,dtype=np.float64)
    if a.shape!=b.shape or a.ndim!=1:
        raise ValueError("Comparable paired patient signed effects required")
    if min(float(np.std(a)),float(np.std(b)))<1e-9:
        return None
    statistic=spearmanr(a,b).statistic
    return float(statistic) if np.isfinite(statistic) else None


def run(outdir:Path):
    if outdir.exists():
        raise FileExistsError("Refuse to overwrite executed risk-mechanism attribution")
    torch.set_num_threads(1)
    receipts={}
    registered=[]
    signed_vectors={}
    for family in FAMILIES:
        full=make_observable_v22(family)
        core=replace(
            full,
            train_x_full=full.train_x_full[:CORE_TRAIN].copy(),
            train_environment=full.train_environment[:CORE_TRAIN].copy(),
            train_labels=full.train_labels[:CORE_TRAIN].copy(),
            train_episode_ids=full.train_episode_ids[:CORE_TRAIN].copy(),
        )
        whitening=train_only_whitening(core)
        source=fit_train_only_environment_source_directions(
            core,whiten=whitening,
            seed=522221,permutations=199,bootstraps=48,
        )
        input_data=build_native_v22_data(core,whitening)
        signed_vectors[family]={arm:[] for arm in ARMS}
        receipts[family]={"source":separation_passport(source),"risk_screen":None}
        active=[]
        if source.status=="ACCEPT_OBSERVATIONAL_ENVIRONMENT_UNMIXING":
            basis=native_basis_from_separation(source)
            measured_screen=whitening.transform(full.train_x_full[CORE_TRAIN:])[:,:24]
            screen=screen_v22_native_sources_on_disjoint_train_labels(
                measured_screen,np.ones_like(measured_screen),
                full.train_labels[CORE_TRAIN:],
                basis,
                original_source_train_episode_ids=input_data.train.episode_ids,
                screen_episode_ids=full.train_episode_ids[CORE_TRAIN:],
                seed=522231,permutations=199,bootstraps=199,
            )
            receipts[family]["risk_screen"]=risk_plan_passport(screen)
            config,dual,v10=_config()
            config["training"]["epochs"]=EPOCHS
            config["training"]["batch_size"]=BATCH
        else:
            basis=None
            screen=None
        for seed in MODEL_SEEDS:
            for arm in ARMS:
                if basis is None:
                    registered.append({
                        "family":family,"seed":seed,"arm":arm,
                        "status":source.status,"trained":False,
                        "validation_auprc":None,
                        "validation_auroc":None,
                        "validation_brier":None,
                        "max_native_logit_decomposition_error":None,
                        "active_native_signed_effect_rank_crossseed":None,
                        "oracle_alignment_eval_only":None,
                        "physical_meaning_certified":False,
                    })
                    continue
                use_sign=arm in ("signed_start_only","signed_start_and_mask")
                use_mask=arm in ("descriptive_mask_only","signed_start_and_mask")
                initial=(
                    dict(screen.active_native_raw_decision_initialization)
                    if use_sign else {}
                )
                descriptive=(
                    tuple(screen.structurally_descriptive_only_native_slots)
                    if use_mask else ()
                )
                model,receipt=fit_multiscale(
                    input_data,basis,seed,config,dual,v10,
                    consistency=False,feature_exploration=False,
                    expert_feedback=None,
                    native_head_refit=False,
                    initial_native_raw_decision_weights=initial or None,
                    initial_native_descriptive_only_slots=descriptive or None,
                )
                for slot in (0,1):
                    expected_policy=2 if slot in descriptive else 0
                    actual_policy=int(model.physical.signed_risk_policy[slot])
                    if actual_policy!=expected_policy:
                        raise AssertionError("Risk source mask not persistent in native graph")
                    if actual_policy==2 and float(model.physical.decision_weights()[slot])!=0.:
                        raise AssertionError("Descriptive-only source leaked to native risk head")
                metrics,activation,effects=_native_readout(
                    model,input_data.validation,
                )
                weights=model.physical.decision_weights().detach().cpu().numpy().tolist()
                row={
                    "family":family,"seed":seed,"arm":arm,
                    "status":"TRAINED","trained":True,
                    "source_identifiability_status":source.status,
                    "original_core_train_episodes":CORE_TRAIN,
                    "independent_risk_screen_episodes":RISK_SCREEN,
                    "screen_signed_initial_weights":(
                        {str(k):v for k,v in initial.items()}
                        if use_sign else {}
                    ),
                    "descriptive_source_zero_risk_slots":list(descriptive),
                    "native_task_coefficient_after_training":list(map(float,weights)),
                    "native_source_risk_policy_after_training":[
                        int(v) for v in model.physical.signed_risk_policy.cpu().numpy()
                    ],
                    "source_filters_identical_across_four_factorial_arms":True,
                    "source_oracle_used_only_after_all_fits":True,
                    "oracle_alignment_eval_only":None,
                    "physical_meaning_certified":False,
                    **metrics,
                }
                registered.append(row)
                signed_vectors[family][arm].append({
                    "seed":seed,"effects":effects,
                    "native_zero_mask":set(descriptive),
                })
                active.append((row,activation))
                print(family,seed,arm,"AUPRC",round(metrics["validation_auprc"],5),
                      "risk weights",np.round(weights,3).tolist(),flush=True)
        # The physical/instrument oracle cannot affect source, risk or
        # native training; its explicit load is AFTER ALL four arms/seeds.
        oracle=load_oracle_v22_validation_for_postfit_audit(family)
        phys=oracle["generated_physical_source"][:,-6:].mean(1)
        instr=oracle["generated_instrument_source"][:,-6:].mean(1)
        for row,act in active:
            row["oracle_alignment_eval_only"]=_oracle_alignment_no_training_feedback(
                act,phys,instr,full.validation_environment,
            )
        del oracle,phys,instr,active
    aggregate=[]
    for family in FAMILIES:
        for arm in ARMS:
            rows=[r for r in registered if r["family"]==family and r["arm"]==arm]
            completed=[r for r in rows if r["trained"]]
            views=signed_vectors[family][arm]
            all_source_ranks=[]
            active_only_ranks=[]
            for i in range(len(views)):
                for j in range(i+1,len(views)):
                    if views[i]["native_zero_mask"]!=views[j]["native_zero_mask"]:
                        raise AssertionError("Same arm must preserve the risk-eligible concept set")
                    for slot in (0,1):
                        a=views[i]["effects"][:,slot]
                        b=views[j]["effects"][:,slot]
                        statistic=_valid_spearman(a,b)
                        if statistic is not None:
                            all_source_ranks.append(statistic)
                            if slot not in views[i]["native_zero_mask"]:
                                active_only_ranks.append(statistic)
            aggregate.append({
                "family":family,"arm":arm,
                "n_registered_opportunities":len(rows),
                "n_fitted_models":len(completed),
                "n_source_identifiability_abstentions":len(rows)-len(completed),
                "mean_validation_auprc":(
                    float(np.mean([r["validation_auprc"] for r in completed]))
                    if completed else None
                ),
                "mean_validation_brier":(
                    float(np.mean([r["validation_brier"] for r in completed]))
                    if completed else None
                ),
                "mean_crossseed_risk_eligible_signed_effect_spearman":(
                    float(np.mean(active_only_ranks)) if active_only_ranks else None
                ),
                "number_actual_risk_eligible_pairwise_contributions":len(active_only_ranks),
                "masked_zero_concept_spearman_was_excluded_not_forced_zero":True,
                "mean_source_oracle_alignment_only_postfit":(
                    float(np.mean([
                        r["oracle_alignment_eval_only"]["mean_abs_oracle_source_spearman_eval_only"]
                        for r in completed
                    ])) if completed else None
                ),
                "causal_source_identity_certified":False,
            })
    if len(registered)!=48 or len(aggregate)!=16 or len(receipts)!=4:
        raise AssertionError("Missing factorial risk signs, zero-mask or identifiability arms")
    outdir.mkdir(parents=True,exist_ok=False)
    for name,rows in (
        ("ALL_48_SIGN_AND_MASK_FACTORIAL_OPPORTUNITIES.csv",registered),
        ("ALL_16_FACTORIAL_SUMMARIES.csv",aggregate),
    ):
        keys=sorted({k for r in rows for k in r})
        with (outdir/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    k:json.dumps(row.get(k),sort_keys=True) if isinstance(row.get(k),(dict,list))
                    else row.get(k)
                    for k in keys
                })
    (outdir/"FAMILY_SOURCE_RISK_PLAN_PASSports.json").write_text(
        json.dumps(receipts,indent=2,allow_nan=False)
    )
    manifest={
        "status":"V22C_NATIVE_RISK_SIGN_VERSUS_DESCRIPTIVE_EXCLUSION_FACTORIAL",
        "n_registered_factorial_opportunities":len(registered),
        "n_native_fits":sum(r["trained"] for r in registered),
        "n_statistical_source_refusals":sum(not r["trained"] for r in registered),
        "source_core_train_episode_count":CORE_TRAIN,
        "risk_screen_train_episode_count":RISK_SCREEN,
        "no_oracle_used_in_source_or_task_training":True,
        "true_sources_consulted_only_after_all_family_fits":True,
        "all_model_native_code_and_source_axes_same_across_arms":True,
        "masked_constant_sources_not_counted_as_zero_stability":True,
        "test_seed_522203_still_sealed":True,
        "all_four_factorial_arms_and_negative_cases_reported":True,
        "not_real_medical_concept_certification":True,
        "per_opportunity":registered,
        "aggregate":aggregate,
        "source_risk_passports":receipts,
    }
    (outdir/"STATUS.json").write_text(json.dumps(manifest,indent=2,allow_nan=False))
    return manifest


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args()
    run(args.out_dir)
