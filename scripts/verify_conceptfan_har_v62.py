#!/usr/bin/env python3
"""Independent UCI HAR nine-participant TEST evaluation of native ProgramFAN.

Reads private original per-participant TEST arrays exclusively inside CI.
Verifies every saved metric, native signed contribution from frozen TRAIN
passports, robust device-missing fallback, participant-macro NLL and
exact dual-VALID + 49-stratified-source-null admission. No original
participant motion stream or risk array is included in public artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score,accuracy_score,log_loss

MODELS={
    "AccelProgramFAN":"accel",
    "ConditionalGyroFAN_proposed":"proposed",
    "ConditionalGyroFAN_deployed":"deployed",
    "UnconditionalGyro_ablation":"unconditioned",
    "AccelProgramLogistic":"accel_program_logistic",
    "AllProgramLogistic":"all_program_logistic",
    "AccelThreeAxisLogistic":"accel_axis_logistic",
    "GyroThreeAxisLogistic":"gyro_axis_logistic",
    "AllSixAxisLogistic":"all_axis_logistic",
    "AccelNormMeanLogistic":"accel_mean_logistic",
}


def metrics(y,z,people):
    p=expit(z)
    return {
        "auprc":float(average_precision_score(y,p))
                 if len(set(y.tolist()))==2 else None,
        "nll":float(log_loss(y,p,labels=[0,1])),
        "target_rate":float(y.mean()),
        "mean_predicted_probability":float(p.mean()),
        "n_windows":len(y),
        "both_classes_evaluable":bool(len(set(y.tolist()))==2),
        "patient_macro_NLL":float(np.mean([
            log_loss(y[people==g],p[people==g],labels=[0,1])
            for g in np.unique(people)])),
        "real_patient_groups":len(np.unique(people)),
        "accuracy_fixed_half":float(accuracy_score(y,p>=.5)),
    }


def audit(root):
    root=Path(root)
    r=json.loads((root/"report_v62.json").read_text())
    models={(x["task"],int(x["seed"])):x for x in r["models"]}
    banks={x["task"]:x for x in r["source_banks"]}
    if (r["status"]!="V62_EXECUTED_UNTOUCHED_UCI_HAR_SENSOR_NOVELTY"
        or not r["ALL_seed_training_and_VALID_gates_frozen_before_TEST_opened"]
        or not r["original_official_TEST_nine_independent_persons"]
        or not r["conditional_gyro_source_ids_SELECTED_TRAIN_X_only"]
        or not r["two_true_stochastic_native_initializations"]
        or r["person_ids_and_full_windows_not_uploaded"] is not True):
        raise AssertionError("UCI HAR untouched source/model promise incomplete")
    if (len(r["test_cases"])!=len(r["private_case_hashes"])
        or len(r["test_cases"])!=len(models)*5
        or len(models)!=4 or len(banks)!=2):
        raise AssertionError("Not all two tasks × two seeds × five stresses executed")
    private={}
    for case,item in zip(r["test_cases"],r["private_case_hashes"]):
        task=case["task"];seed=int(case["seed"]);stress=case["stress"]
        key=f"UCIHAR_{task}_seed{seed}_{stress}"
        if item["case"]!=key or not item["private_per_participant_windows_not_uploaded"]:
            raise AssertionError("Individual UCI participant evidence disclosure invalid")
        path=root/"private_case_arrays"/f"{key}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item["sha256"]:
            raise AssertionError("Original external UCI HAR TEST predictions tampered")
        with np.load(path,allow_pickle=False) as npz:
            x={f:npz[f].copy() for f in (
                "y","subject_group","accel","proposed","deployed",
                "unconditioned","accel_program_logistic",
                "all_program_logistic","accel_axis_logistic",
                "gyro_axis_logistic","all_axis_logistic",
                "accel_mean_logistic","gyro_memberships","gyro_support",
                "gyro_signed","active","admitted_used",
                "anchor_memberships","anchor_signed","anchor_support")}
        y=x["y"];ids=x["subject_group"];n=len(y)
        if (len(np.unique(ids))!=9
            or case["official_TEST_subjects"]!=9
            or len(y)!=case["overlapping_windows"]
            or not np.isin(y,[0,1]).all()
            or not all(np.isfinite(v).all() for v in x.values())):
            raise AssertionError("Invalid independent participant or raw-score evidence")
        record=models[task,seed]
        bank=banks[task]
        if set(bank["original_physically_distinct_channels"])!={
            "BODY_ACCELEROMETER_NORM","BODY_GYROSCOPE_NORM",
            "BODY_GYROSCOPE_SIGNED_Y"
        }:
            raise AssertionError("Wrong source sensor modalities")
        for j in bank["gyroscope_only_indices"]:
            if 0 in bank["all_compiler_candidate_passports"]["candidates"][j]["channels"]:
                raise AssertionError("Novel gyro program still read original accelerometer")
        anchor=record["anchor_native_head"]
        alpha=np.asarray(anchor["gamma"],float)
        member=x["anchor_memberships"]
        signed=x["anchor_signed"]
        if (signed.shape!=(n,len(alpha)) or
            member.shape!=signed.shape):
            raise AssertionError("Native anchor-concept dimension mismatch")
        expected=alpha[None,:]*(member-.5)
        np.testing.assert_allclose(signed,expected,atol=2e-9,rtol=0)
        np.testing.assert_allclose(
            x["accel"],anchor["train_prior_log_odds"]+signed.sum(axis=1),
            atol=2e-9,rtol=0)
        if not(anchor["fixed_outcome_blind_source_shortlist"]
               and anchor["stochastic_seeded_initialization_std"]==.03):
            raise AssertionError("V61R fixed accel dictionary/model initialization not applied")
        source=record["gyro_native_head"]
        gate=record["dual_VALID_patient_and_accel_stratified_49_null_gate"]
        if source is None:
            assert x["gyro_memberships"].shape==(n,0)
            np.testing.assert_array_equal(x["accel"],x["proposed"])
            np.testing.assert_array_equal(x["accel"],x["deployed"])
            assert not gate["approved"]
        else:
            if (not source["fixed_outcome_blind_source_shortlist"] or
                source["n_candidates"]>6 or
                source["stochastic_seeded_initialization_std"]!=.03):
                raise AssertionError("Source candidate selected posthoc using activity Y")
            src_id=record["gyro_source_contract"]["raw_source_indices"]
            expected_ids=[
                bank["all_compiler_candidate_passports"]["candidates"][j]["candidate_id"]
                for j in src_id]
            if expected_ids!=record["source_program_ids"]:
                raise AssertionError("TRAIN source IDs do not match heldout audit")
            gamma=np.asarray(source["gamma"],float)
            q=x["gyro_memberships"];support=x["gyro_support"];signed=x["gyro_signed"]
            if (q.shape!=(n,len(gamma)) or signed.shape!=q.shape
                or support.shape!=q.shape):
                raise AssertionError("Gyro source native dimension changed on TEST")
            if np.any((q<0)|(q>1)) or np.any((support<0)|(support>1)):
                raise AssertionError("Gyro memberships not bounded")
            needed=np.where(support>=.5,gamma[None,:]*(q-.5),0.)
            np.testing.assert_allclose(signed,needed,atol=2e-9,rtol=0)
            np.testing.assert_allclose(
                x["proposed"],x["accel"]+signed.sum(axis=1),atol=2e-9,rtol=0)
            eligible=x["active"].astype(bool).any(axis=1)
            approved=bool(gate["patient_dual_VALID"]["approved"] and
                          gate["conditional_null_VALID1"]["passed"] and
                          gate["conditional_null_VALID2"]["passed"])
            if approved!=gate["approved"]:
                raise AssertionError("Dual-VALID and conditional null verdict inconsistent")
            used=eligible&approved
            np.testing.assert_array_equal(x["admitted_used"].astype(bool),used)
            np.testing.assert_array_equal(
                x["deployed"],np.where(used,x["proposed"],x["accel"]))
        if stress in ("gyro_missing","accel_missing"):
            # Missing one full physical sensor cannot silently allow an
            # unsupported conditional gyroscope source.
            np.testing.assert_array_equal(x["accel"],x["deployed"])
            assert not x["admitted_used"].any()
        for label,field in MODELS.items():
            orig=case["metrics"][label]
            computed=metrics(y,x[field],ids)
            for metric,value in computed.items():
                recorded=orig[metric]
                if recorded is None:
                    if value is not None:raise AssertionError("AUPRC estimability mismatch")
                elif isinstance(recorded,(bool,int)):
                    if recorded!=value:raise AssertionError("UCI TEST counts drift")
                elif abs(value-recorded)>1e-10:
                    raise AssertionError(f"UCI HAR original TEST metric drift {key}/{label}/{metric}")
        private[key]={"y":y,"group":ids}
    # Both seeds share SAME nine official TEST people. Do not treat 2×
    # trained models as 18 independent heldout real participants.
    for task in ("STAIRS_UP_VS_DOWN","STAND_VS_SIT"):
        for stress in ("natural","MCAR20","gyro_missing","accel_missing","gyro_gain_1p5"):
            a=private[f"UCIHAR_{task}_seed11_{stress}"]
            b=private[f"UCIHAR_{task}_seed23_{stress}"]
            np.testing.assert_array_equal(a["y"],b["y"])
            np.testing.assert_array_equal(a["group"],b["group"])
    result={
        "status":"V62_UCI_HAR_INDEPENDENT_SENSOR_MODALITY_AUDIT_PASS",
        "original_official_subjects":"21 development, 9 untouched TEST",
        "real_test_participants":9,
        "n_verified_individual_private_test_arrays":len(private),
        "original_TEST_loaded_only_after_all_train_valid_freeze":True,
        "accelerometer_gyro_different_measured_modalities":True,
        "conditional_gyro_source_never_reads_anchor_acceleration":True,
        "native_accelerometer_and_gyro_signed_contributions_verified":True,
        "strict_dual_TRUE_participant_validation_and_49_null_checks":True,
        "exact_accel_fallback_when_gyro_not_admitted_or_source_missing":True,
        "both_stochastic_native_seeds_share_same_nine_TEST_people":True,
        "all_reported_AUPRC_NLL_accuracy_and_subject_macro_NLL_recomputed":True,
        "no_original_participant_signal_or_individual_risk_uploaded":True,
        "programs_are_statistical_not_certified_physical_causes":True,
    }
    (root/"verify_v62_audit.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dir",required=True,type=Path)
    a=p.parse_args()
    audit(a.dir)
