#!/usr/bin/env python3
"""V60 real ProgramFAN end-to-end from outcome-blind programs to native risk.

This supersedes neither V58/V59 metrics nor the V56/V57 assay gates.
- BIDMC TRAIN only: typed program compilation + sparse signed torch head.
- BIDMC VALID1 + VALID2: strict whole-patient admission (no window bootstrap).
- BIDMC/VitalDB TEST only: paired, real, site-specific descriptive outcomes.
- No semantic physical concept or clinical/prospective-device certification.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from conceptfan_discovery.program_compiler_v60 import (
    compile_measured_programs_v60,infer_program_matrix_v60,
    fingerprint_program_bank_v60,
)
from conceptfan_discovery.program_fan_v60 import (
    fit_program_head_v60,infer_program_head_v60,
    group_macro_nll_v60,admit_program_head_v60,
    apply_program_admission_v60,
)
from conceptfan_discovery.real_future_tasks_v60 import (
    load_bidmc_task_v60,load_vitaldb_task_v60,
)
from scripts.experiment_conceptfan_bidmc_real_v58 import (
    _test_variant,_scores,_pulse_score,_train_pulse_logistic,
    _pulse_logit,_aggregate_patient_nll,_bootstrap_by_real_patient,
)
from scripts.experiment_conceptfan_operator_library_v54 import (
    native_k2,native_logits,run_joint_epochs_v47,
    _baseline,_apply_baseline,
)

TASKS=("simultaneous","future_60s")
STRESS=("natural","MCAR20","pulse_missing","pulse_plus_25_bpm")
SITES=("BIDMC","VitalDB")
SEEDS=(11,23)
PROTOCOL="CONCEPTFAN_V60_TYPED_PROGRAM_COMPILER_SUBJECT_GATED_NATIVE_RISK"


def _check_native_output(pred):
    e=np.max(np.abs(pred["logits"]-pred["baseline"]-
                    pred["signed_contributions"].sum(axis=1)))
    if e>1e-9:
        raise AssertionError("Native signed measured-program risk decomposition failed")
    absent=~pred["used"].any(axis=1)
    if not np.array_equal(pred["logits"][absent],pred["baseline"][absent]):
        raise AssertionError("Program head risk bypass while all sources unavailable")
    return float(e)


def _active_use(fit,pred):
    used=np.asarray(pred["used"],bool)
    if not fit.active_indices:
        return np.zeros((len(used),1),bool)
    return used[:,list(fit.active_indices)]


def _trained_heads(train,valid1,valid2,bank,model,*,seed):
    xtr,atr=infer_program_matrix_v60(
        bank,train["x"][:,:,:3],train["mask"][:,:,:3])
    train_base=native_logits(model,train["x"],train["mask"])
    fitting={}
    for variant in ("PURE","RESIDUAL"):
        fit=fit_program_head_v60(
            xtr,atr,train_base,train["y"],train["patient"],
            variant=variant,seed=20266021+seed)
        fitting[variant]=fit
    selected=min(fitting.values(),key=lambda f:(
        f.train_patient_macro_NLL,f.variant))
    valid_episodes=[]
    for section in (valid1,valid2):
        v,a=infer_program_matrix_v60(
            bank,section["x"][:,:,:3],section["mask"][:,:,:3])
        orig=native_logits(model,section["x"],section["mask"])
        out=infer_program_head_v60(selected,v,a,orig)
        _check_native_output(out)
        valid_episodes.append((
            section["y"],orig,out["logits"],
            _active_use(selected,out),section["patient"]))
    admission=admit_program_head_v60(
        selected,valid_episodes[0],valid_episodes[1])
    return fitting,selected,admission


def _patient_metadata_summary(section):
    return {
        "n_real_patient_groups":len(np.unique(section["patient"])),
        "n_overlapping_windows":len(section["y"]),
        "positive_window_fraction":float(np.mean(section["y"])),
        "positive_patients":int(sum(
            section["y"][section["patient"]==p].any()
            for p in np.unique(section["patient"]))),
        "mean_natural_input_observation_fraction":
            float(section["mask"][:,:,:3].mean()),
    }


def _run_test(heldout,norm,model,bank,fits,selected,admission,
              raw_fit,pulse_fit,*,site,task,seed,output):
    cases=[];secret_manifest=[]
    for stress in STRESS:
        x,mask,pulse=_test_variant(
            heldout,norm,stress,seed=20266090+seed)
        frozen=native_logits(model,x,mask)
        v,a=infer_program_matrix_v60(
            bank,x[:,:,:3],mask[:,:,:3])
        predictions={}
        outputs={}
        for mode in ("PURE","RESIDUAL"):
            out=infer_program_head_v60(fits[mode],v,a,frozen)
            _check_native_output(out)
            outputs[mode]=out
            predictions[mode]=out["logits"]
        target=outputs[selected.variant]
        used=_active_use(selected,target)
        accepted,actually_used=apply_program_admission_v60(
            frozen,target["logits"],used,approved=admission["approved"])
        if not admission["approved"]:
            np.testing.assert_array_equal(accepted,frozen)
            if actually_used.any():
                raise AssertionError("Unadmitted model modified original K2")
        zraw=_apply_baseline(raw_fit[0],x,mask)
        zeng=_apply_baseline(raw_fit[1],x,mask,energy=True)
        zpulse=_pulse_logit(pulse,pulse_fit)
        scores={
            "K2":frozen,
            "ProgramFAN_PURE":predictions["PURE"],
            "ProgramFAN_RESIDUAL":predictions["RESIDUAL"],
            "ProgramFAN_DUAL_VALID_OR_K2":accepted,
            "raw_logistic":zraw,
            "energy_logistic":zeng,
            "pulse_only_logistic":zpulse,
        }
        score={}
        for name,z in scores.items():
            value=_scores(heldout["y"],z)
            macro,npatients=_aggregate_patient_nll(
                heldout["y"],z,heldout["patient"])
            value["patient_macro_NLL"]=macro
            value["real_patient_groups"]=npatients
            score[name]=value
        codes=np.unique(heldout["patient"],return_inverse=True)[1]
        private=output/"private_case_arrays"
        private.mkdir(parents=True,exist_ok=True)
        key=f"{site}_{task}_seed{seed}_{stress}"
        file=private/f"{key}.npz"
        np.savez_compressed(
            file,
            y=heldout["y"].astype(np.uint8),
            patient_group=codes.astype(np.int16),
            frozen=frozen,
            pure=predictions["PURE"],residual=predictions["RESIDUAL"],
            selected=target["logits"],deployed=accepted,
            raw=zraw,engineered=zeng,pulse_only=zpulse,
            used_active=used.astype(np.uint8),
            actually_used=actually_used.astype(np.uint8),
            source_memberships=target["memberships"],
            signed_contributions=target["signed_contributions"],
            source_support=target["support"],
        )
        secret_manifest.append({
            "case":key,"sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
            "private_person_risks_uploaded":False})
        cases.append({
            "site":site,"task":task,"seed":seed,"stress":stress,
            "selected_program_variant":selected.variant,
            "total_program_candidates":len(bank.candidates),
            "active_selected_program_indices":list(selected.active_indices),
            "accepted_after_dual_VALID":bool(admission["approved"]),
            "proposed_source_support_fraction":float(used.any(axis=1).mean()),
            "deployed_source_usage_fraction":float(actually_used.mean()),
            "test_population":_patient_metadata_summary(heldout),
            "direct_PPG_PULSE_vs_outcome_HR":_pulse_score(
                heldout["y"],pulse,heldout["hr"]),
            "metrics":score,
            "patient_cluster_bootstrap_proposed_vs_energy":
                _bootstrap_by_real_patient(
                    heldout["y"],target["logits"],zeng,
                    heldout["patient"],seed=20266050+seed),
            "patient_cluster_bootstrap_deployed_vs_energy":
                _bootstrap_by_real_patient(
                    heldout["y"],accepted,zeng,
                    heldout["patient"],seed=20266070+seed),
            "no_patient_group_used_for_TEST_model_selection":True,
            "no_clinical_or_causal_concept_identity_inferred":True,
        })
        print("V60",site,task,seed,stress,
              "candidates",len(bank.candidates),
              "selected",selected.variant,
              "admitted",admission["approved"],
              "AUPRC",score["ProgramFAN_PURE"]["auprc"],
              "logreg",score["pulse_only_logistic"]["auprc"],
              flush=True)
    return cases,secret_manifest


def execute(bidmc,output,*,vitaldb=None,seeds=SEEDS,tasks=TASKS):
    torch.set_num_threads(2)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    all_cases=[];all_private=[];receipts=[];source_manifests=[]
    for task in tasks:
        data,norm,manifest=load_bidmc_task_v60(Path(bidmc),task)
        (output/f"bidmc_{task}_split_receipt.json").write_text(
            json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
        tr,v1,v2=(data[k] for k in ("train","valid1","valid2"))
        if np.unique(tr["y"]).size!=2:
            raise RuntimeError("Frozen BIDMC TRAIN one-class task: no posthoc split change")
        bank=compile_measured_programs_v60(
            tr["x"][:,:,:3],tr["mask"][:,:,:3],
            tr["patient"],seed=20266013)
        if not bank.candidates:
            raise RuntimeError("No measurable real TRAIN programs, record failed capability")
        raw_fit=_baseline(tr["x"],tr["mask"],tr["y"])
        pulse_fit=_train_pulse_logistic(tr)
        sites={"BIDMC":data["test"]}
        if vitaldb is not None:
            external,receipt=load_vitaldb_task_v60(
                Path(vitaldb),norm,task)
            sites["VitalDB"]=external
        source_manifests.append({
            "task":task,"program_bank":bank.manifest(),
            "program_fingerprint":fingerprint_program_bank_v60(bank),
            "BIDMC_split_counts":{
                key:{"subjects":len(np.unique(row["patient"])),
                     "windows":len(row["y"]),
                     "positive_fraction":float(row["y"].mean())}
                for key,row in data.items()},
            "source_operator_proposal_used_outcome_y":False,
            "source_physical_identity_proven":False,
            "future_task_no_target_in_input":
                task!="future_60s" or
                manifest["target_begins_after_input"] is True,
        })
        for seed in seeds:
            model=native_k2(seed)
            training=run_joint_epochs_v47(
                model,tr["x"][:,:,:4],tr["mask"][:,:,:4],
                np.full(len(tr["y"]),.5,np.float32),
                np.zeros(len(tr["y"]),bool),tr["y"],
                seed=seed,consistency_weight=0.,epochs=5)
            fits,choice,gate=_trained_heads(
                tr,v1,v2,bank,model,seed=seed)
            receipts.append({
                "task":task,"native_seed":seed,
                "native_K2_training":training,
                "all_program_fits_TRAIN_only":{
                    k:v.passport() for k,v in fits.items()},
                "selected_variant_TRAIN_only":choice.variant,
                "chosen_head_patient_VALID_receipt":gate,
                "source_program_passport_count":len(bank.candidates),
                "source_physical_identity_verified":False,
            })
            for site,hold in sites.items():
                scores,secret=_run_test(
                    hold,norm,model,bank,fits,choice,gate,
                    raw_fit,pulse_fit,site=site,task=task,
                    seed=seed,output=output)
                all_cases.extend(scores);all_private.extend(secret)
    report={
        "protocol":PROTOCOL,
        "status":"EXECUTED_EXPLORATORY_REAL_HOSPITAL_MEASUREMENT",
        "training_site":"BIDMC 53 original official recordings",
        "test_sites":list(sites),
        "test_sites_previously_inspected_in_V58_V59":True,
        "n_real_tasks":len(tasks),"real_tasks":list(tasks),
        "n_native_K2_models":len(receipts),
        "n_test_cases":len(all_cases),"expected_cases_per_site_task_seed":len(STRESS),
        "n_individual_TEST_array_manifests":len(all_private),
        "source_discovery_outcome_blind":True,
        "source_predictive_head_TRAIN_Y_only":True,
        "whole_real_patient_VALID_admission":True,
        "VALID1_and_VALID2_both_required":True,
        "exact_native_K2_on_abstention":True,
        "no_direct_raw_to_risk_shortcut":True,
        "no_medical_utility_or_physical_causal_semantic_claim":True,
        "patient_identifiers_exported":False,
        "patient_real_vitals_exported":False,
        "individual_risk_predictions_in_public_artifact":False,
        "source_manifests":source_manifests,
        "fit_and_gate_receipts":receipts,
        "test_cases":all_cases,
        "private_case_sha256":all_private,
        "limitations":"Previously inspected BIDMC/VitalDB sites; overlapping windows within subjects; external RESP modality shift; concurrent monitor label is almost same as PULSE; +60s is model development not prospective clinical study."
    }
    (output/"report_v60.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    with (output/"aggregate_cases_v60.csv").open(
        "w",newline="",encoding="utf8") as f:
        w=csv.writer(f)
        methods=("K2","ProgramFAN_PURE","ProgramFAN_RESIDUAL",
                 "ProgramFAN_DUAL_VALID_OR_K2",
                 "raw_logistic","energy_logistic","pulse_only_logistic")
        w.writerow(["task","site","seed","stress","source_candidates",
                    "trained_model_choice","dual_valid_accepted",
                    "source_used",
                    *[k+"_AUPRC" for k in methods],
                    *[k+"_NLL" for k in methods]])
        for case in all_cases:
            m=case["metrics"]
            w.writerow([
                case["task"],case["site"],case["seed"],case["stress"],
                case["total_program_candidates"],
                case["selected_program_variant"],
                case["accepted_after_dual_VALID"],
                case["deployed_source_usage_fraction"],
                *[m[k]["auprc"] for k in methods],
                *[m[k]["nll"] for k in methods]])
    print(json.dumps({
        "status":report["status"],
        "sites":report["test_sites"],"native_models":len(receipts),
        "registered_test_cases":len(all_cases),
        "programs_by_task":[len(t["program_bank"]["candidates"])
                            for t in source_manifests],
    },indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--bidmc",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--vitaldb",type=Path,default=None)
    p.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS))
    p.add_argument("--tasks",nargs="+",choices=TASKS,default=list(TASKS))
    a=p.parse_args()
    execute(a.bidmc,a.out,vitaldb=a.vitaldb,
            seeds=tuple(a.seeds),tasks=tuple(a.tasks))
