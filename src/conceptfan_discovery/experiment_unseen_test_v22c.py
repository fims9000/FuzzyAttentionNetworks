"""V22c: frozen, independently seeded TEST draw for V22b native risk experiments.

Exactly the V22b 600 CORE TRAIN / independent 200 TRAIN source risk screen
and all 48 registered conditions. The previously unseen TEST generator
is NOT INSTANTIATED before all original 36 native models have finished
training, including the two source-identifiability ABSTAIN families.

TEST outcomes and true synthetic hidden factors are then used ONLY once
for descriptive scoring of every predeclared model; no updated models,
data-driven selection, optimized thresholds or clinical source identity.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import torch

from .data import DiscoverySplit
from .mixed_source_data_v22 import (
    FAMILIES,SEEDS,SIZES,HISTORY,HORIZON,MODEL_CHANNELS,
    _simulate,make_observable_v22,train_only_whitening,build_native_v22_data,
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
    CORE_TRAIN_N,INDEPENDENT_RISK_SCREEN_N,TRAIN_EPOCHS,BATCH_SIZE,
    MODEL_SEEDS,SOURCE_METHODS,RISK_ARMS,
)
from .experiment_mixed_source_v22 import (
    _config,_native_readout,_oracle_alignment_no_training_feedback,_rho,
)
from .clinical_multiscale_v10 import fit_multiscale


FRESH_TEST_SEED=SEEDS["test"]
FRESH_TEST_N=SIZES["test"]
EXPECTED_OPPORTUNITIES=len(FAMILIES)*len(SOURCE_METHODS)*len(RISK_ARMS)*len(MODEL_SEEDS)


def _fingerprint(model)->str:
    checksum=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        checksum.update(name.encode("utf-8"))
        checksum.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def _test_split_without_oracle(family,whitener):
    # This function must only be called AFTER every registered model fit
    # has completed across ALL four source-generating conditions.
    raw,environment,y,latent=_simulate(
        family,FRESH_TEST_SEED,FRESH_TEST_N,
    )
    transformed=whitener.transform(raw)
    mask=np.ones_like(transformed,dtype=np.float32)
    testing=DiscoverySplit(
        x=transformed[:,:HISTORY],
        mask=mask[:,:HISTORY],
        future=transformed[:,HISTORY:HISTORY+HORIZON],
        future_mask=mask[:,HISTORY:HISTORY+HORIZON],
        y=y.astype(np.float32),
        oracle_states=None,
        episode_ids=np.arange(50000,50000+FRESH_TEST_N,dtype=np.int64),
    )
    return testing,environment.astype(np.int8),latent[:,:HISTORY].astype(np.float32)


def run(outdir:Path):
    if outdir.exists():
        raise FileExistsError("Frozen V22c TEST must never overwrite previous evidence")
    torch.set_num_threads(1)

    ready_models={}
    whitening={}
    source_evidence={}
    task_risk={}
    registered=[]
    n_trained=0
    n_refusals=0
    first_raw_test_instantiate_at=None

    for family in FAMILIES:
        original=make_observable_v22(family)
        core=replace(
            original,
            train_x_full=original.train_x_full[:CORE_TRAIN_N].copy(),
            train_environment=original.train_environment[:CORE_TRAIN_N].copy(),
            train_labels=original.train_labels[:CORE_TRAIN_N].copy(),
            train_episode_ids=original.train_episode_ids[:CORE_TRAIN_N].copy(),
        )
        whitener=train_only_whitening(core)
        whitening[family]=whitener
        sep=fit_train_only_environment_source_directions(
            core,whiten=whitener,
            seed=522221,permutations=199,bootstraps=48,
        )
        source_evidence[family]=separation_passport(sep)
        data=build_native_v22_data(core,whitener)
        source_ids=set(map(int,data.train.episode_ids))
        screen_ids=original.train_episode_ids[CORE_TRAIN_N:]
        if source_ids & set(map(int,screen_ids)):
            raise AssertionError("TRAIN source-risk and source/model fit cohorts overlapped")
        screen_history=whitener.transform(
            original.train_x_full[CORE_TRAIN_N:],
        )[:,:HISTORY]
        screen_mask=np.ones_like(screen_history,dtype=np.float32)
        screen_y=original.train_labels[CORE_TRAIN_N:]
        if (len(data.train.y)!=CORE_TRAIN_N
                or len(screen_y)!=INDEPENDENT_RISK_SCREEN_N):
            raise AssertionError("V22b training/gate sample counts were modified")

        original_basis=np.eye(len(MODEL_CHANNELS),2,dtype=np.float64)
        basis_by_method={
            "pooled_pca_basis":original_basis,
            "environment_contrast_basis":(
                native_basis_from_separation(sep)
                if sep.status=="ACCEPT_OBSERVATIONAL_ENVIRONMENT_UNMIXING"
                else None
            ),
        }
        config,dual,v10=_config()
        config["training"]["epochs"]=TRAIN_EPOCHS
        config["training"]["batch_size"]=BATCH_SIZE
        ready_models[family]=[]

        for method in SOURCE_METHODS:
            basis=basis_by_method[method]
            plan=(
                screen_v22_native_sources_on_disjoint_train_labels(
                    screen_history,screen_mask,screen_y,basis,
                    original_source_train_episode_ids=data.train.episode_ids,
                    screen_episode_ids=screen_ids,
                    seed=522231,permutations=199,bootstraps=199,
                ) if basis is not None else None
            )
            task_risk[f"{family}__{method}"]=(
                risk_plan_passport(plan) if plan is not None else None
            )
            for arm in RISK_ARMS:
                for seed in MODEL_SEEDS:
                    identity={"family":family,"source_method":method,
                              "risk_init_arm":arm,"seed":seed}
                    if basis is None:
                        registered.append({
                            **identity,"status":sep.status,"trained":False,
                            "test_evaluated":False,
                            "test_auprc":None,
                            "test_auroc":None,
                            "test_brier":None,
                            "max_native_logit_decomposition_error":None,
                            "oracle_alignment_postfit_test_audit_only":None,
                            "physical_source_certification":False,
                        })
                        n_refusals+=1
                        continue
                    apply_plan=arm=="independently_screened_risk"
                    weights=(plan.active_native_raw_decision_initialization
                             if apply_plan else None)
                    descriptive=(plan.structurally_descriptive_only_native_slots
                                 if apply_plan else None)
                    model,receipt=fit_multiscale(
                        data,basis,seed,config,dual,v10,
                        consistency=False,
                        feature_exploration=False,
                        expert_feedback=None,
                        native_head_refit=False,
                        initial_native_raw_decision_weights=(
                            dict(weights) if weights else None
                        ),
                        initial_native_descriptive_only_slots=(
                            tuple(descriptive) if descriptive else None
                        ),
                    )
                    model.eval()
                    model_fingerprint=_fingerprint(model)
                    if apply_plan:
                        for slot in (0,1):
                            expected=2 if slot in descriptive else 0
                            if int(model.physical.signed_risk_policy[slot])!=expected:
                                raise AssertionError(
                                    "V22c native risk source mask differs from V22b"
                                )
                            if expected==2 and float(model.physical.decision_weights()[slot])!=0:
                                raise AssertionError("TEST model has leaked descriptive source risk")
                    else:
                        if not bool((model.physical.signed_risk_policy==0).all()):
                            raise AssertionError("Arbitrary-start V22b baseline mutated")
                    ready_models[family].append((identity,model,plan,model_fingerprint))
                    registered.append({
                        **identity,"status":"TRAINED","trained":True,
                        "test_evaluated":False,
                        "core_train":CORE_TRAIN_N,
                        "independent_train_risk_screen":INDEPENDENT_RISK_SCREEN_N,
                        "native_model_sha256_before_any_test_data":model_fingerprint,
                        "risk_init_weights":(
                            {str(k):v for k,v in weights.items()}
                            if apply_plan and weights else {}
                        ),
                        "descriptive_zero_risk_slots":(
                            list(descriptive) if apply_plan and descriptive else []
                        ),
                        "test_auprc":None,
                        "test_auroc":None,
                        "test_brier":None,
                        "max_native_logit_decomposition_error":None,
                        "oracle_alignment_postfit_test_audit_only":None,
                        "physical_source_certification":False,
                    })
                    n_trained+=1
                    print("FITTED_WITHOUT_TEST",family,method,arm,seed,flush=True)

    if (n_trained!=36 or n_refusals!=12 or
            len(registered)!=EXPECTED_OPPORTUNITIES):
        raise AssertionError(
            "Do not unseal V22c TEST if 36 fits and 12 refusals are incomplete"
        )

    # SEALED DATA BOUNDARY: All 36 neural predictions, 8 source-risk gates,
    # 4 source proposals and all frozen arm decisions already determined.
    # This is the FIRST place TEST seed 522203 is simulated or exposed.
    test_unsealed_after_full_training=True
    for family in FAMILIES:
        test,environment,truth=_test_split_without_oracle(
            family,whitening[family]
        )
        if test.oracle_states is not None or len(test.y)!=FRESH_TEST_N:
            raise AssertionError("V22c test data contract broken")
        if not set(test.episode_ids).isdisjoint(range(0,CORE_TRAIN_N+INDEPENDENT_RISK_SCREEN_N)):
            raise AssertionError("TEST episode IDs reused a TRAIN subject")
        physical=truth[:,-6:,0].mean(1)
        instrument=truth[:,-6:,1].mean(1)
        for identity,model,plan,training_sha in ready_models[family]:
            if _fingerprint(model)!=training_sha:
                raise AssertionError("Model was changed after TEST was introduced")
            stats,activation,effects=_native_readout(model,test)
            matched=_oracle_alignment_no_training_feedback(
                activation,physical,instrument,environment,
            )
            matching=[row for row in registered if
                      all(row[k]==v for k,v in identity.items())]
            if len(matching)!=1 or matching[0]["test_evaluated"]:
                raise AssertionError("Registered TEST case scored twice/missing")
            row=matching[0]
            row.update({
                "test_evaluated":True,
                "test_auprc":stats["validation_auprc"],
                "test_auroc":stats["validation_auroc"],
                "test_brier":stats["validation_brier"],
                "max_native_logit_decomposition_error":stats[
                    "max_native_logit_decomposition_error"
                ],
                "mean_abs_native_concept_effect":stats[
                    "average_absolute_native_concept_effect"
                ],
                "oracle_alignment_postfit_test_audit_only":matched,
                "native_model_sha256_after_test":_fingerprint(model),
            })
            if row["native_model_sha256_after_test"]!=training_sha:
                raise AssertionError("TEST evaluation changed model parameters")
        del test,environment,truth,physical,instrument

    summary=[]
    for family in FAMILIES:
        for method in SOURCE_METHODS:
            for arm in RISK_ARMS:
                cases=[r for r in registered if
                       (r["family"],r["source_method"],r["risk_init_arm"])
                       ==(family,method,arm)]
                evaluated=[r for r in cases if r["test_evaluated"]]
                summary.append({
                    "family":family,"source_method":method,
                    "risk_init_arm":arm,
                    "n_registered":len(cases),
                    "n_test_evaluated":len(evaluated),
                    "n_abstained":len(cases)-len(evaluated),
                    "mean_new_seed_test_auprc":(
                        float(np.mean([r["test_auprc"] for r in evaluated]))
                        if evaluated else None
                    ),
                    "mean_new_seed_test_brier":(
                        float(np.mean([r["test_brier"] for r in evaluated]))
                        if evaluated else None
                    ),
                    "mean_test_oracle_concept_alignment_postfit":(
                        float(np.mean([
                            r["oracle_alignment_postfit_test_audit_only"][
                                "mean_abs_oracle_source_spearman_eval_only"
                            ] for r in evaluated
                        ])) if evaluated else None
                    ),
                    "genuine_new_random_test_cohort_but_inspected_generator_families":True,
                    "physical_or_clinical_source_identity_certified":False,
                })
    if len(summary)!=16 or sum(r["test_evaluated"] for r in registered)!=36:
        raise AssertionError("All predeclared V22c TEST outcomes must be reported")

    outdir.mkdir(parents=True,exist_ok=False)
    for filename,records in (
        ("ALL_48_FROZEN_TEST_CASES.csv",registered),
        ("ALL_16_TEST_ARM_SUMMARIES.csv",summary),
    ):
        keys=sorted({key for row in records for key in row})
        with (outdir/filename).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=keys)
            writer.writeheader()
            for r in records:
                writer.writerow({
                    k:json.dumps(r.get(k),sort_keys=True)
                    if isinstance(r.get(k),(dict,list)) else r.get(k)
                    for k in keys
                })

    receipt={
        "status":"V22C_FROZEN_FRESH_TEST_522203_NATIVE_SOURCE_AND_RISK",
        "registered_opportunities":len(registered),
        "fitted_models":n_trained,
        "predeclared_identifiability_refusals":n_refusals,
        "test_evaluated_frozen_models":sum(r["test_evaluated"] for r in registered),
        "test_seed":FRESH_TEST_SEED,
        "new_test_episodes_per_family":FRESH_TEST_N,
        "old_train_fit_seed":SEEDS["train"],
        "old_inspected_validation_seed_not_used_for_test":SEEDS["validation"],
        "frozen_test_instantiated_only_after_every_model_fit":test_unsealed_after_full_training,
        "identical_independent_train_screening_policy_v22b":True,
        "exact_same_predeclared_native_model_architecture_and_epochs":True,
        "no_test_retraining_threshold_tuning_or_model_selection":True,
        "test_evaluation_never_mutates_native_state":True,
        "source_oracle_loaded_only_for_postfit_test_audit":True,
        "new_random_test_draw_within_previously_inspected_simulator_families":True,
        "no_clinical_causality_or_human_feedback_claimed":True,
        "synthetic_latent_truth_not_in_native_input_or_risk_screen":True,
        "all_negative_and_refused_cases_preserved":True,
        "per_fit_and_abstention":registered,
        "aggregate":summary,
        "source_evidence":source_evidence,
        "train_only_task_risk_screens":task_risk,
    }
    (outdir/"SOURCE_ENVIRONMENT_PROPOSALS.json").write_text(
        json.dumps(source_evidence,indent=2,allow_nan=False)
    )
    (outdir/"TRAIN_ONLY_RISK_PLAN_PASSports.json").write_text(
        json.dumps(task_risk,indent=2,allow_nan=False)
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
