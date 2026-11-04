"""V22d: label-budget-controlled native source/risk comparison on new TEST.

Fairly allocate the *same* 800 labeled TRAIN episodes to either:
(1) standard native joint optimization on all 800, or
(2) native optimization on 600 plus independent 200 TRAIN-only labels
    for per-source task-risk screening, starting-sign and no-risk masks.

The whitened observable representation and candidate BSS/PCA basis are
estimated from the FIRST 600 TRAIN histories for both arms. Exactly 36
K2 native models and 12 source-identifiability refusals are registered.
Brand-new synthetic TEST seed 522313 is generated only AFTER all fits.

Extra task-risk screening is observational, not physiological certification.
This is label-budget-controlled but NOT optimizer-compute matched.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import csv
import json

import numpy as np
import torch

from .data import DiscoverySplit
from .mixed_source_data_v22 import (
    FAMILIES,HISTORY,HORIZON,MODEL_CHANNELS,
    _simulate,make_observable_v22,
    train_only_whitening,build_native_v22_data,
)
from .environmental_unmixing_v22 import (
    fit_train_only_environment_source_directions,
    native_basis_from_separation,separation_passport,
)
from .native_source_risk_screen_v22b import (
    screen_v22_native_sources_on_disjoint_train_labels,
    risk_plan_passport,
)
from .experiment_risk_screened_sources_v22b import (
    CORE_TRAIN_N,INDEPENDENT_RISK_SCREEN_N,
    TRAIN_EPOCHS,BATCH_SIZE,MODEL_SEEDS,SOURCE_METHODS,
)
from .experiment_unseen_test_v22c import _fingerprint
from .experiment_mixed_source_v22 import (
    _config,_native_readout,
    _oracle_alignment_no_training_feedback,
)
from .clinical_multiscale_v10 import fit_multiscale


BUDGET_ARMS=("joint_800_labels","screened_600_fit_200_gate")
FRESH_HOLDOUT_SEED=522313
FRESH_HOLDOUT_N=300


def _generate_test_after_all_training(family,whitener):
    # Must be called only AFTER the 48-opportunity training/screen barrier.
    raw,environment,labels,latent=_simulate(
        family,FRESH_HOLDOUT_SEED,FRESH_HOLDOUT_N,
    )
    transformed=whitener.transform(raw)
    mask=np.ones_like(transformed,dtype=np.float32)
    split=DiscoverySplit(
        x=transformed[:,:HISTORY],
        mask=mask[:,:HISTORY],
        future=transformed[:,HISTORY:HISTORY+HORIZON],
        future_mask=mask[:,HISTORY:HISTORY+HORIZON],
        y=labels.astype(np.float32),
        oracle_states=None,
        episode_ids=np.arange(60000,60000+FRESH_HOLDOUT_N,dtype=np.int64),
    )
    return split,environment.astype(np.int8),latent[:,:HISTORY].astype(np.float32)


def run(outdir:Path):
    if outdir.exists():
        raise FileExistsError("Never overwrite frozen V22d equal-label-budget TEST")
    torch.set_num_threads(1)
    records=[]
    ready={}
    whitening={}
    proposals={}
    screens={}
    n_fitted=0
    n_abstained=0
    for family in FAMILIES:
        full=make_observable_v22(family)
        if len(full.train_episode_ids)!=800:
            raise AssertionError("Original full-label TRAIN budget changed")
        core=replace(
            full,
            train_x_full=full.train_x_full[:CORE_TRAIN_N].copy(),
            train_environment=full.train_environment[:CORE_TRAIN_N].copy(),
            train_labels=full.train_labels[:CORE_TRAIN_N].copy(),
            train_episode_ids=full.train_episode_ids[:CORE_TRAIN_N].copy(),
        )
        w=train_only_whitening(core)
        whitening[family]=w
        sep=fit_train_only_environment_source_directions(
            core,whiten=w,
            seed=522221,permutations=199,bootstraps=48,
        )
        proposals[family]=separation_passport(sep)
        fit600=build_native_v22_data(core,w)
        fit800=build_native_v22_data(full,w)
        if (len(fit600.train.y)!=600 or len(fit800.train.y)!=800
                or fit600.train.oracle_states is not None
                or fit800.train.oracle_states is not None):
            raise AssertionError("Source/model fit cohort or hidden-oracle boundary changed")
        if (not np.array_equal(fit600.train.x,fit800.train.x[:CORE_TRAIN_N])
                or not np.array_equal(
                    fit600.train.episode_ids,
                    fit800.train.episode_ids[:CORE_TRAIN_N],
                )):
            raise AssertionError("Both label-budget arms must share identical source histories")

        independent=full.train_x_full[CORE_TRAIN_N:]
        screen_inputs=w.transform(independent)[:,:HISTORY]
        screen_masks=np.ones_like(screen_inputs,dtype=np.float32)
        screen_labels=full.train_labels[CORE_TRAIN_N:]
        screen_ids=full.train_episode_ids[CORE_TRAIN_N:]
        if (len(screen_labels)!=INDEPENDENT_RISK_SCREEN_N
                or set(map(int,fit600.train.episode_ids))
                   &set(map(int,screen_ids))):
            raise AssertionError("Independent TRAIN screen must not reuse core-fit IDs")
        pca_basis=np.eye(len(MODEL_CHANNELS),2,dtype=np.float64)
        candidate_basis={
            "pooled_pca_basis":pca_basis,
            "environment_contrast_basis":(
                native_basis_from_separation(sep)
                if sep.status=="ACCEPT_OBSERVATIONAL_ENVIRONMENT_UNMIXING"
                else None
            ),
        }
        config,dual,v10=_config()
        config["training"]["epochs"]=TRAIN_EPOCHS
        config["training"]["batch_size"]=BATCH_SIZE
        ready[family]=[]
        for method in SOURCE_METHODS:
            basis=candidate_basis[method]
            taskplan=(
                screen_v22_native_sources_on_disjoint_train_labels(
                    screen_inputs,screen_masks,screen_labels,basis,
                    original_source_train_episode_ids=fit600.train.episode_ids,
                    screen_episode_ids=screen_ids,
                    seed=522231,permutations=199,bootstraps=199,
                ) if basis is not None else None
            )
            screens[f"{family}__{method}"]=(
                risk_plan_passport(taskplan) if taskplan is not None else None
            )
            for arm in BUDGET_ARMS:
                for seed in MODEL_SEEDS:
                    identity={
                        "family":family,"source_method":method,
                        "label_budget_arm":arm,"seed":seed,
                    }
                    if basis is None:
                        records.append({
                            **identity,"status":sep.status,
                            "trained":False,"test_evaluated":False,
                            "test_auprc":None,"test_auroc":None,
                            "test_brier":None,
                            "max_native_logit_decomposition_error":None,
                            "postfit_hidden_source_alignment_only":None,
                            "physiological_causality_certified":False,
                        })
                        n_abstained+=1
                        continue
                    screened=arm=="screened_600_fit_200_gate"
                    fit_data=fit600 if screened else fit800
                    signed=(taskplan.active_native_raw_decision_initialization
                            if screened else None)
                    no_risk=(taskplan.structurally_descriptive_only_native_slots
                             if screened else None)
                    model,receipt=fit_multiscale(
                        fit_data,basis,seed,config,dual,v10,
                        consistency=False,
                        feature_exploration=False,
                        expert_feedback=None,
                        native_head_refit=False,
                        initial_native_raw_decision_weights=(
                            dict(signed) if signed else None
                        ),
                        initial_native_descriptive_only_slots=(
                            tuple(no_risk) if no_risk else None
                        ),
                    )
                    model.eval()
                    if screened:
                        for slot in (0,1):
                            expected=2 if slot in no_risk else 0
                            if int(model.physical.signed_risk_policy[slot])!=expected:
                                raise AssertionError("Source relevance mask changed")
                            if expected==2 and float(model.physical.decision_weights()[slot])!=0:
                                raise AssertionError("Non-task source leaked into native risk")
                    elif not bool((model.physical.signed_risk_policy==0).all()):
                        raise AssertionError("Full 800-label joint baseline is not unrestricted")
                    sha=_fingerprint(model)
                    ready[family].append((identity,model,sha))
                    records.append({
                        **identity,"status":"TRAINED",
                        "trained":True,"test_evaluated":False,
                        "train_labels_total_available":800,
                        "native_joint_fit_episodes":600 if screened else 800,
                        "independent_risk_screen_episodes":200 if screened else 0,
                        "common_native_concept_basis_fitted_on_600_train_only":True,
                        "native_model_sha256_before_test":sha,
                        "signed_task_initial_weights":(
                            {str(k):v for k,v in signed.items()}
                            if screened and signed else {}
                        ),
                        "descriptive_only_source_slots":(
                            list(no_risk) if screened and no_risk else []
                        ),
                        "test_auprc":None,"test_auroc":None,
                        "test_brier":None,
                        "max_native_logit_decomposition_error":None,
                        "postfit_hidden_source_alignment_only":None,
                        "physiological_causality_certified":False,
                    })
                    n_fitted+=1
                    print("TRAIN_FINISHED_WITH_NO_TEST",
                          family,method,arm,seed,flush=True)

    if len(records)!=48 or n_fitted!=36 or n_abstained!=12:
        raise AssertionError("Do not unseal new TEST until all budget arms finish")

    # First and only creation of the untouched TEST generator seed 522313.
    # All 36 model weights, 8 independent risk-screen decisions, and
    # all source bases already exist and are frozen.
    test_generation_after_complete_training=True
    for family in FAMILIES:
        test,environment,hidden=_generate_test_after_all_training(
            family,whitening[family],
        )
        if test.oracle_states is not None or len(test.y)!=FRESH_HOLDOUT_N:
            raise AssertionError("TEST hidden physical source leaked to native model")
        physical=hidden[:,-6:,0].mean(1)
        instrument=hidden[:,-6:,1].mean(1)
        for identity,model,sha in ready[family]:
            if _fingerprint(model)!=sha:
                raise AssertionError("Native model changed before independent TEST")
            score,raw,effect=_native_readout(model,test)
            audit=_oracle_alignment_no_training_feedback(
                raw,physical,instrument,environment,
            )
            matching=[
                r for r in records
                if all(r[k]==v for k,v in identity.items())
            ]
            if len(matching)!=1 or matching[0]["test_evaluated"]:
                raise AssertionError("Duplicate or missing registered TEST evaluation")
            row=matching[0]
            row.update({
                "test_evaluated":True,
                "test_auprc":score["validation_auprc"],
                "test_auroc":score["validation_auroc"],
                "test_brier":score["validation_brier"],
                "max_native_logit_decomposition_error":score[
                    "max_native_logit_decomposition_error"
                ],
                "postfit_hidden_source_alignment_only":audit,
                "native_model_sha256_after_test":_fingerprint(model),
            })
            if row["native_model_sha256_after_test"]!=sha:
                raise AssertionError("Native model mutated by TEST scoring")
        del test,environment,hidden,physical,instrument

    aggregate=[]
    for family in FAMILIES:
        for method in SOURCE_METHODS:
            for arm in BUDGET_ARMS:
                selected=[
                    r for r in records
                    if (r["family"],r["source_method"],r["label_budget_arm"])
                    ==(family,method,arm)
                ]
                fitted=[r for r in selected if r["test_evaluated"]]
                aggregate.append({
                    "family":family,"source_method":method,
                    "label_budget_arm":arm,
                    "registered_opportunities":len(selected),
                    "fitted_and_test_evaluated":len(fitted),
                    "identifiability_abstentions":len(selected)-len(fitted),
                    "mean_new_test_auprc":(
                        float(np.mean([r["test_auprc"] for r in fitted]))
                        if fitted else None
                    ),
                    "mean_new_test_brier":(
                        float(np.mean([r["test_brier"] for r in fitted]))
                        if fitted else None
                    ),
                    "mean_new_test_source_alignment_only":(
                        float(np.mean([
                            r["postfit_hidden_source_alignment_only"][
                                "mean_abs_oracle_source_spearman_eval_only"
                            ] for r in fitted
                        ])) if fitted else None
                    ),
                    "compute_budget_strictly_matched":False,
                    "both_arms_consumed_800_labeled_train_subjects":True,
                    "physiological_causality_claimed":False,
                })

    if (len(aggregate)!=16 or sum(r["test_evaluated"] for r in records)!=36):
        raise AssertionError("Cannot publish incomplete V22d sample denominator")
    outdir.mkdir(parents=True,exist_ok=False)
    for name,rows in (
        ("ALL_48_EQUAL_LABEL_BUDGET_CASES.csv",records),
        ("ALL_16_NEW_TEST_BUDGET_SUMMARIES.csv",aggregate),
    ):
        keys=sorted({k for row in rows for k in row})
        with (outdir/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    k:json.dumps(row.get(k),sort_keys=True)
                    if isinstance(row.get(k),(dict,list)) else row.get(k)
                    for k in keys
                })
    receipt={
        "status":"V22D_EQUAL_800_LABEL_BUDGET_NATIVE_RISK_SOURCE_COMPARISON",
        "n_registered_native_cases":len(records),
        "n_trained_native_models":n_fitted,
        "n_structural_identifiability_abstentions":n_abstained,
        "n_frozen_test_evaluations":sum(r["test_evaluated"] for r in records),
        "core_source_proposal_episodes":600,
        "joint_native_fit_episodes":800,
        "screened_native_fit_episodes":600,
        "independent_train_task_risk_screen_episodes":200,
        "fresh_new_test_seed":FRESH_HOLDOUT_SEED,
        "fresh_test_episodes_per_family":FRESH_HOLDOUT_N,
        "fresh_test_instantiated_after_every_model_fit":test_generation_after_complete_training,
        "two_arms_consume_same_total_labeled_train_episodes":True,
        "source_proposal_and_whitener_identical_both_arms":True,
        "optimization_compute_exposures_not_matched":True,
        "test_scores_never_select_or_update_model":True,
        "all_model_state_sha256_unchanged_by_test":True,
        "all_source_alignment_truth_loaded_postfit_only":True,
        "new_test_draw_but_generating_families_already_inspected":True,
        "no_physician_clinical_or_causal_concept_identity_proven":True,
        "all_negative_and_abstained_cases_reported":True,
        "per_fit":records,
        "summary":aggregate,
        "measured_source_evidence":proposals,
        "train_only_predictive_risk_evidence":screens,
    }
    (outdir/"SOURCE_PROPOSALS.json").write_text(
        json.dumps(proposals,indent=2,allow_nan=False)
    )
    (outdir/"TRAIN_ONLY_RISK_SCREENS.json").write_text(
        json.dumps(screens,indent=2,allow_nan=False)
    )
    (outdir/"STATUS.json").write_text(
        json.dumps(receipt,indent=2,allow_nan=False)
    )
    return receipt


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",required=True,type=Path)
    args=parser.parse_args()
    run(args.out_dir)
