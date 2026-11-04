"""V22f: independent TRAIN-only risk calibration versus raw native logits.

Native K2 source formation uses 600 TRAIN histories without outcomes from
the 150 calibration patients. Actual native model gradients see only first
450 TRAIN episodes. A disjoint 200 TRAIN label screen assigns native task
risk roles, exactly as V22b/d/e. A distinct 150 episode TRAIN subset fits
ONLY a standard positive affine logit calibration (not new concepts).

All 18 native model fits, 18 scalar calibrations, and four full-800-label
external logistic controls finish BEFORE the new TEST seed 522333 is
generated. Native model weights never change after native training. Each
scored patient retains exact additive physical and observation-process
contributions, all scaled by the same positive calibration parameter.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import csv
import json

import numpy as np
import torch
from scipy.special import expit

from .data import DiscoverySplit
from .mixed_source_data_v22 import (
    FAMILIES,HISTORY,HORIZON,MODEL_CHANNELS,
    _simulate,make_observable_v22,train_only_whitening,
    build_native_v22_data,
)
from .environmental_unmixing_v22 import (
    fit_train_only_environment_source_directions,
    native_basis_from_separation,separation_passport,
)
from .native_source_risk_screen_v22b import (
    screen_v22_native_sources_on_disjoint_train_labels,
    risk_plan_passport,
)
from .train_only_native_calibration_v22f import (
    fit_train_only_native_affine_calibration,
    native_calibration_passport,
)
from .experiment_risk_screened_sources_v22b import (
    TRAIN_EPOCHS,BATCH_SIZE,MODEL_SEEDS,
)
from .experiment_mixed_source_v22 import (
    _config,_oracle_alignment_no_training_feedback,
)
from .experiment_unseen_test_v22c import _fingerprint
from .experiment_external_predictors_v22e import (
    observed_temporal_summaries,_external_estimators,
    _pickle_fingerprint,_predictive_metrics,
)
from .clinical_multiscale_v10 import fit_multiscale


SOURCE_TRAIN_HISTORIES=600
NATIVE_FIT_LABELS=450
CALIBRATION_LABELS=150
SOURCE_RISK_LABELS=200
FROZEN_NEW_TEST_SEED=522333
FROZEN_NEW_TEST_N=300
NATIVE_METHODS=("native_pca","native_environment_unmixing")
NATIVE_SCORING_ARMS=("original_native_probabilities",
                     "train_affine_calibrated_probabilities")


def _make_split(raw,y,ids,w):
    transformed=w.transform(raw)
    mask=np.ones_like(transformed,dtype=np.float32)
    return DiscoverySplit(
        x=transformed[:,:HISTORY],
        mask=mask[:,:HISTORY],
        future=transformed[:,HISTORY:HISTORY+HORIZON],
        future_mask=mask[:,HISTORY:HISTORY+HORIZON],
        y=np.asarray(y,dtype=np.float32),
        oracle_states=None,
        episode_ids=np.asarray(ids,dtype=np.int64),
    )


def _native_logits(model,split,calibrator=None):
    model.eval()
    logits=[]
    probs=[]
    raw_sources=[]
    max_native_error=0.
    max_calibrated_error=0.
    with torch.no_grad():
        for first in range(0,len(split.y),128):
            last=min(first+128,len(split.y))
            x=torch.from_numpy(split.x[first:last])
            mask=torch.from_numpy(split.mask[first:last])
            out=model(x,mask,process_mask=mask)
            max_native_error=max(max_native_error,float(out.decomposition_error()))
            original_logits=out.logits.detach().cpu().numpy().astype(np.float64)
            logits.extend(original_logits.tolist())
            if calibrator is not None:
                z,bias,phys,care=calibrator.decomposed_native_contributions(out)
                z2=calibrator.slope*out.logits+calibrator.intercept
                max_calibrated_error=max(
                    max_calibrated_error,
                    float((z-z2).abs().max()),
                    float((z-(bias+phys.sum(dim=-1)+care.sum(dim=-1))).abs().max()),
                )
                probs.extend(torch.sigmoid(z).cpu().numpy().tolist())
            else:
                probs.extend(out.probabilities.detach().cpu().numpy().tolist())
            raw=out.physical.raw_activation[:,-6:,:]
            availability=out.physical.availability[:,-6:,:]
            supported=(raw*availability).sum(dim=1)/availability.sum(dim=1).clamp_min(1e-8)
            raw_sources.extend(supported.cpu().numpy().tolist())
    if max_native_error>2e-5 or max_calibrated_error>2e-5:
        raise AssertionError("Probability calibration broke exact native concept logit mediation")
    return (
        np.asarray(logits,dtype=np.float64),
        np.asarray(probs,dtype=np.float64),
        np.asarray(raw_sources,dtype=np.float64),
        max_native_error,max_calibrated_error,
    )


def run(outdir:Path):
    if outdir.exists():
        raise FileExistsError("Never replace frozen V22f calibration evidence")
    torch.set_num_threads(1)
    cases=[]
    train_source_evidence={}
    train_risk_evidence={}
    retained={}
    n_models=0
    n_calibrators=0
    n_ext=0
    n_refusals=0
    for family in FAMILIES:
        observed=make_observable_v22(family)
        if len(observed.train_labels)!=800:
            raise AssertionError("Underlying V22 TRAIN budget changed")
        history=replace(
            observed,
            train_x_full=observed.train_x_full[:SOURCE_TRAIN_HISTORIES].copy(),
            train_labels=observed.train_labels[:SOURCE_TRAIN_HISTORIES].copy(),
            train_environment=observed.train_environment[:SOURCE_TRAIN_HISTORIES].copy(),
            train_episode_ids=observed.train_episode_ids[:SOURCE_TRAIN_HISTORIES].copy(),
        )
        net_part=replace(
            history,
            train_x_full=history.train_x_full[:NATIVE_FIT_LABELS].copy(),
            train_labels=history.train_labels[:NATIVE_FIT_LABELS].copy(),
            train_environment=history.train_environment[:NATIVE_FIT_LABELS].copy(),
            train_episode_ids=history.train_episode_ids[:NATIVE_FIT_LABELS].copy(),
        )
        white=train_only_whitening(history)
        sep=fit_train_only_environment_source_directions(
            history,whiten=white,
            seed=522221,permutations=199,bootstraps=48,
        )
        train_source_evidence[family]=separation_passport(sep)
        model_data=build_native_v22_data(net_part,white)
        cal_ids=observed.train_episode_ids[NATIVE_FIT_LABELS:SOURCE_TRAIN_HISTORIES]
        cal_y=observed.train_labels[NATIVE_FIT_LABELS:SOURCE_TRAIN_HISTORIES]
        cal_split=_make_split(
            observed.train_x_full[NATIVE_FIT_LABELS:SOURCE_TRAIN_HISTORIES],
            cal_y,cal_ids,white,
        )
        screen_ids=observed.train_episode_ids[SOURCE_TRAIN_HISTORIES:]
        screen_y=observed.train_labels[SOURCE_TRAIN_HISTORIES:]
        screen_history=white.transform(
            observed.train_x_full[SOURCE_TRAIN_HISTORIES:],
        )[:,:HISTORY]
        if (len(model_data.train.y)!=450 or len(cal_split.y)!=150
                or len(screen_y)!=200):
            raise AssertionError("Native TRAIN/calibration/risk screen budget not frozen")
        partitions=[
            set(map(int,model_data.train.episode_ids)),
            set(map(int,cal_split.episode_ids)),
            set(map(int,screen_ids)),
        ]
        if any(partitions[i]&partitions[j] for i in range(3) for j in range(i+1,3)):
            raise AssertionError("Calibration labels share patients with native gradients or task screen")
        if model_data.train.oracle_states is not None or cal_split.oracle_states is not None:
            raise AssertionError("Ground-truth source unexpectedly exposed to model")
        bases={
            "native_pca":np.eye(len(MODEL_CHANNELS),2,dtype=np.float64),
            "native_environment_unmixing":(
                native_basis_from_separation(sep)
                if sep.status=="ACCEPT_OBSERVATIONAL_ENVIRONMENT_UNMIXING"
                else None
            ),
        }
        config,dual,v10=_config()
        config["training"]["epochs"]=TRAIN_EPOCHS
        config["training"]["batch_size"]=BATCH_SIZE
        retained[family]={"whitener":white,"native":[],"logistic":None}

        for method,basis in bases.items():
            risk=(screen_v22_native_sources_on_disjoint_train_labels(
                screen_history,np.ones_like(screen_history,dtype=np.float32),
                screen_y,basis,
                original_source_train_episode_ids=model_data.train.episode_ids,
                screen_episode_ids=screen_ids,
                seed=522231,permutations=199,bootstraps=199,
            ) if basis is not None else None)
            train_risk_evidence[f"{family}__{method}"]=(
                risk_plan_passport(risk) if risk is not None else None
            )
            for seed in MODEL_SEEDS:
                identity={"family":family,"method":method,"seed":seed}
                if basis is None:
                    for arm in NATIVE_SCORING_ARMS:
                        cases.append({
                            **identity,"arm":arm,"status":sep.status,
                            "trained":False,"test_evaluated":False,
                            "test_auprc":None,"test_auroc":None,
                            "test_brier":None,"test_logloss":None,
                            "physical_source_certified":False,
                        })
                        n_refusals+=2
                    continue
                signed=risk.active_native_raw_decision_initialization
                descriptive=risk.structurally_descriptive_only_native_slots
                model,fit_receipt=fit_multiscale(
                    model_data,basis,seed,config,dual,v10,
                    consistency=False,feature_exploration=False,
                    expert_feedback=None,native_head_refit=False,
                    initial_native_raw_decision_weights=(
                        dict(signed) if signed else None
                    ),
                    initial_native_descriptive_only_slots=(
                        tuple(descriptive) if descriptive else None
                    ),
                )
                model.eval()
                original_checksum=_fingerprint(model)
                cal_logit,_,_,max_native_error,_=_native_logits(
                    model,cal_split,
                )
                calibrator=fit_train_only_native_affine_calibration(
                    cal_logit,cal_y,
                    independent_calibration_episode_ids=cal_ids,
                    native_model_fit_episode_ids=model_data.train.episode_ids,
                    source_risk_screen_episode_ids=screen_ids,
                )
                if _fingerprint(model)!=original_checksum:
                    raise AssertionError("Calibration unexpectedly modified native ConceptFAN")
                retained[family]["native"].append(
                    (identity,model,calibrator,original_checksum)
                )
                for arm in NATIVE_SCORING_ARMS:
                    cases.append({
                        **identity,"arm":arm,"status":"TRAINED",
                        "trained":True,"test_evaluated":False,
                        "n_native_gradients_patient_cases":450,
                        "n_disjoint_probability_calibration_cases":150,
                        "n_disjoint_risk_screen_cases":200,
                        "n_unsupervised_source_measurement_histories":600,
                        "descriptive_zero_risk_slots":list(descriptive),
                        "calibration_slope_train_only":calibrator.slope,
                        "calibration_intercept_train_only":calibrator.intercept,
                        "native_model_sha256_before_test":original_checksum,
                        "native_model_changed_by_calibration":False,
                        "test_auprc":None,"test_auroc":None,
                        "test_brier":None,"test_logloss":None,
                        "physical_source_certified":False,
                    })
                n_models+=1
                n_calibrators+=1
                print("NATIVE_AND_150_TRAIN_CALIBRATION_FINISHED",
                      family,method,seed,
                      "scale",round(calibrator.slope,3),
                      flush=True)

        logistic=_external_estimators()["logistic_30_temporal_summaries"]
        logistic.fit(
            observed_temporal_summaries(observed.train_x_full),
            observed.train_labels.astype(int),
        )
        native_digest=_pickle_fingerprint(logistic)
        retained[family]["logistic"]=(logistic,native_digest)
        cases.append({
            "family":family,"method":"logistic_30_temporal_summaries",
            "seed":None,"arm":"external_full_800_train_labels",
            "status":"TRAINED","trained":True,"test_evaluated":False,
            "n_train_labels":800,
            "predictor_sha256_before_test":native_digest,
            "test_auprc":None,"test_auroc":None,
            "test_brier":None,"test_logloss":None,
            "physical_source_certified":False,
        })
        n_ext+=1
        print("EXTERNAL_LOGISTIC_FIT_FINISHED_BEFORE_TEST",family,flush=True)

    # TEST CANNOT EXIST before the complete fit+calibration barrier.
    if (len(cases)!=52 or n_models!=18 or n_calibrators!=18
            or n_ext!=4 or n_refusals!=12):
        raise AssertionError("No TEST data may be opened before 18+18+4 fits complete")

    new_test_opened_after_all_models_and_calibrators=True
    for family in FAMILIES:
        raw,env,y,hidden=_simulate(
            family,FROZEN_NEW_TEST_SEED,FROZEN_NEW_TEST_N,
        )
        split=_make_split(
            raw,y,np.arange(80000,80000+FROZEN_NEW_TEST_N),
            retained[family]["whitener"],
        )
        physical=hidden[:,HISTORY-6:HISTORY,0].mean(1)
        instrument=hidden[:,HISTORY-6:HISTORY,1].mean(1)
        for identity,model,calibrator,checksum in retained[family]["native"]:
            if _fingerprint(model)!=checksum:
                raise AssertionError("Native source-model state changed before new TEST")
            original_logits,original_probs,concept,old_error,_=_native_logits(
                model,split,
            )
            calibrated_logits,calibrated_probs,_,_,cal_error=_native_logits(
                model,split,calibrator=calibrator,
            )
            if not np.array_equal(original_logits,calibrated_logits):
                raise AssertionError("Probability calibrator changed original native logits")
            original_metrics=_predictive_metrics(y,original_probs)
            calibrated_metrics=_predictive_metrics(y,calibrated_probs)
            if (abs(original_metrics["test_auprc"]-calibrated_metrics["test_auprc"])>1e-9
                    or abs(original_metrics["test_auroc"]-calibrated_metrics["test_auroc"])>1e-9):
                raise AssertionError("Positive affine probability calibration unexpectedly changed risk ranking")
            alignment=_oracle_alignment_no_training_feedback(
                concept,physical,instrument,env,
            )
            for arm,metrics,max_error in (
                ("original_native_probabilities",original_metrics,old_error),
                ("train_affine_calibrated_probabilities",calibrated_metrics,cal_error),
            ):
                record=[r for r in cases if
                        r["family"]==family and r["method"]==identity["method"]
                        and r["seed"]==identity["seed"] and r["arm"]==arm]
                if len(record)!=1 or record[0]["test_evaluated"]:
                    raise AssertionError("Duplicate or missing registered native calibration comparison")
                row=record[0]
                row.update({
                    **metrics,
                    "test_evaluated":True,
                    "native_exact_additive_logit_error":max_error,
                    "postfit_source_truth_alignment_only":alignment,
                    "native_model_sha256_after_test":_fingerprint(model),
                })
                if row["native_model_sha256_after_test"]!=checksum:
                    raise AssertionError("New TEST scoring mutated native risk mediation")

        logistic,sha=retained[family]["logistic"]
        if _pickle_fingerprint(logistic)!=sha:
            raise AssertionError("External logistic changed before testing")
        logitscore=logistic.predict_proba(observed_temporal_summaries(raw))[:,1]
        row=[x for x in cases if x["family"]==family and
             x["method"]=="logistic_30_temporal_summaries"]
        if len(row)!=1 or row[0]["test_evaluated"]:
            raise AssertionError("Expected exactly one full-budget logistic per family")
        row[0].update({
            **_predictive_metrics(y,logitscore),
            "test_evaluated":True,
            "predictor_sha256_after_test":_pickle_fingerprint(logistic),
        })
        if row[0]["predictor_sha256_after_test"]!=sha:
            raise AssertionError("External logistic changed on TEST")
        del split,raw,env,y,hidden,physical,instrument

    aggregates=[]
    for family in FAMILIES:
        for method in (*NATIVE_METHODS,"logistic_30_temporal_summaries"):
            arms=(NATIVE_SCORING_ARMS if method in NATIVE_METHODS
                  else ("external_full_800_train_labels",))
            for arm in arms:
                selected=[x for x in cases if x["family"]==family
                          and x["method"]==method and x["arm"]==arm]
                successful=[x for x in selected if x["test_evaluated"]]
                aggregates.append({
                    "family":family,"method":method,"arm":arm,
                    "registered":len(selected),
                    "n_fitted_and_test_evaluated":len(successful),
                    "n_source_refusals":len(selected)-len(successful),
                    "mean_new_test_auprc":(
                        float(np.mean([x["test_auprc"] for x in successful]))
                        if successful else None
                    ),
                    "mean_new_test_auroc":(
                        float(np.mean([x["test_auroc"] for x in successful]))
                        if successful else None
                    ),
                    "mean_new_test_brier":(
                        float(np.mean([x["test_brier"] for x in successful]))
                        if successful else None
                    ),
                    "mean_new_test_logloss":(
                        float(np.mean([x["test_logloss"] for x in successful]))
                        if successful else None
                    ),
                    "real_clinical_physical_identity_proven":False,
                })
    if (len(aggregates)!=20 or sum(x["test_evaluated"] for x in cases)!=40
            or sum(x["trained"] and x["method"] in NATIVE_METHODS for x in cases)!=36):
        raise AssertionError("V22f TEST evaluation denominator incomplete")

    outdir.mkdir(parents=True,exist_ok=False)
    for name,rows in (
        ("ALL_52_CALIBRATION_AND_LOGISTIC_CASES.csv",cases),
        ("ALL_20_CALIBRATION_ARM_SUMMARIES.csv",aggregates),
    ):
        keys=sorted({k for row in rows for k in row})
        with (outdir/name).open("w",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    k:json.dumps(row.get(k),sort_keys=True)
                    if isinstance(row.get(k),(dict,list)) else row.get(k)
                    for k in keys
                })
    result={
        "status":"V22F_DISJOINT_TRAIN_NATIVE_POSITIVE_AFFINE_CALIBRATION",
        "n_registered_total_score_cases":len(cases),
        "n_native_models_fit":n_models,
        "n_train_only_affine_calibrators_fit":n_calibrators,
        "n_external_logistic_models_fit":n_ext,
        "n_native_score_opportunities":48,
        "n_native_calibration_method_abstentions":n_refusals,
        "n_total_frozen_test_scores":sum(x["test_evaluated"] for x in cases),
        "new_test_seed":FROZEN_NEW_TEST_SEED,
        "new_test_patients_per_family":FROZEN_NEW_TEST_N,
        "original_train_patients_per_family":800,
        "native_gradient_train_patients":450,
        "disjoint_train_calibration_patients":150,
        "disjoint_native_task_screen_patients":200,
        "calibration_patient_histories_used_in_unsupervised_source_formation":True,
        "calibration_patient_labels_used_in_source_formation":False,
        "test_data_first_generated_after_all_fits_and_calibrators":new_test_opened_after_all_models_and_calibrators,
        "no_test_fitting_tuning_or_selection":True,
        "original_native_risk_model_unchanged_after_calibration_and_test":True,
        "native_exact_concept_decomposition_preserved_after_calibration":True,
        "all_probability_scaling_slopes_positive":True,
        "all_AUPRC_AUROC_native_pairs_unchanged_by_calibration":True,
        "not_a_novel_probability_calibration_algorithm":True,
        "no_physician_real_clinical_source_or_causal_claim":True,
        "all_negative_cases_and_abstentions_retained":True,
        "per_case":cases,"aggregate":aggregates,
        "unsupervised_source_evidence":train_source_evidence,
        "independent_train_task_role_evidence":train_risk_evidence,
    }
    (outdir/"SOURCE_IDENTIFICATION.json").write_text(
        json.dumps(train_source_evidence,indent=2,allow_nan=False)
    )
    (outdir/"DISJOINT_TRAIN_TASK_SCREEN.json").write_text(
        json.dumps(train_risk_evidence,indent=2,allow_nan=False)
    )
    (outdir/"STATUS.json").write_text(
        json.dumps(result,indent=2,allow_nan=False)
    )
    return result


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args()
    run(args.out_dir)
