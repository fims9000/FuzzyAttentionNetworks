#!/usr/bin/env python3
"""V61 outcome-blind conditional novelty, train-only signed risk, patient VALID.

Real BIDMC TRAIN, real BIDMC TEST, VitalDB external hospital TEST. This is
exploratory: both sites were previously inspected; no physical semantics.
The strong primary competitor is the *same pulse-program logistic* and the
same allowed pulse+nonpulse full-program logistic, not the weak original K2.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from conceptfan_discovery.program_compiler_v60 import (
    infer_program_matrix_v60,
)
from conceptfan_discovery.program_compiler_v61 import (
    compile_allowed_programs_v61,
    select_conditioning_and_novel_programs_v61,
)
from conceptfan_discovery.conditional_innovation_v61 import (
    fit_conditional_sources_v61,infer_conditional_sources_v61,
    conditional_shuffle_gate_v61,
)
from conceptfan_discovery.program_fan_v60 import (
    fit_program_head_v60,infer_program_head_v60,
    admit_program_head_v60,apply_program_admission_v60,
)
from conceptfan_discovery.real_future_targets_v61 import (
    TASKS,TARGETS,load_bidmc_future_v61,
    load_vitaldb_future_v61,stress_v61,
)
from scripts.experiment_conceptfan_bidmc_real_v58 import (
    _scores,_bootstrap_by_real_patient,_pulse_logit,
    _train_pulse_logistic,_aggregate_patient_nll,
)

SEEDS=(11,23)
STRESS=("natural","MCAR20","pulse_missing","pulse_plus_25_bpm")
PROTOCOL="CONCEPTFAN_V61_CONDITIONAL_NONPULSE_NOVELTY_NATIVE_PATIENT_AUDIT"


def _group_weight(group):
    patients,inv,cnt=np.unique(group,return_inverse=True,return_counts=True)
    weight=1./cnt[inv]
    return weight/weight.mean()


def _fit_logistic_programs(z,s,y,patient,*,ids,seed):
    idx=np.asarray(ids,dtype=int)
    x=np.column_stack([z[:,idx],s[:,idx]])
    model=LogisticRegression(
        C=1.,max_iter=1200,solver="lbfgs",random_state=seed)
    model.fit(x,y,sample_weight=_group_weight(patient))
    return model


def _predict_logistic_programs(model,z,s,ids):
    idx=np.asarray(ids,dtype=int)
    x=np.column_stack([z[:,idx],s[:,idx]])
    return np.asarray(model.decision_function(x),float)


def _as_metrics(y,z,patient):
    m=_scores(y,z)
    macro,n=_aggregate_patient_nll(y,z,patient)
    m["patient_macro_NLL"]=macro
    m["real_patient_groups"]=n
    return m


def _novel_all_sources(train,bank,source_indices,pulse_indices):
    v,s=infer_program_matrix_v60(
        bank,train["x"],train["mask"])
    mapped=fit_conditional_sources_v61(
        v,s,train["mask"],train["patient"],
        pulse_indices=pulse_indices,source_indices=source_indices)
    return v,s,mapped


def _evaluate_valid(section,bank,pulse_idx,pulse_head,source_map,risk_head):
    v,s=infer_program_matrix_v60(
        bank,section["x"],section["mask"])
    pulse=infer_program_head_v60(
        pulse_head,v[:,pulse_idx],s[:,pulse_idx],
        np.zeros(len(v)))
    if risk_head is None or source_map is None:
        return pulse["logits"],None,None,None,None
    innovative,support=infer_conditional_sources_v61(
        source_map,v,s,section["mask"])
    proposed=infer_program_head_v60(
        risk_head,innovative,support,pulse["logits"])
    used=(proposed["used"][:,list(risk_head.active_indices)]
          if risk_head.active_indices else
          np.zeros((len(v),1),bool))
    return pulse["logits"],proposed,innovative,support,used


def _fit_pulse_and_conditional(train,v1,v2,bank,pulse_idx,source_idx,*,seed):
    values,support,source_map=_novel_all_sources(
        train,bank,source_idx,pulse_idx)
    pulse_head=fit_program_head_v60(
        values[:,pulse_idx],support[:,pulse_idx],
        np.zeros(len(train["y"])),train["y"],
        train["patient"],variant="PURE",seed=20266110+seed)
    pulse_train=infer_program_head_v60(
        pulse_head,values[:,pulse_idx],support[:,pulse_idx],
        np.zeros(len(values)))["logits"]
    if source_map is not None:
        residual,residual_support=infer_conditional_sources_v61(
            source_map,values,support,train["mask"])
        risk_head=fit_program_head_v60(
            residual,residual_support,pulse_train,train["y"],
            train["patient"],variant="RESIDUAL",seed=20266170+seed)
        raw_uncond=values[:,list(source_map.raw_source_indices)]
        raw_sup=support[:,list(source_map.raw_source_indices)]
        raw_ablation_head=fit_program_head_v60(
            raw_uncond,raw_sup,pulse_train,train["y"],
            train["patient"],variant="RESIDUAL",seed=20266190+seed)
    else:
        risk_head=None;raw_ablation_head=None
    events=[]
    for split in (v1,v2):
        pulse,proposed,residual,res_sup,used=_evaluate_valid(
            split,bank,pulse_idx,pulse_head,source_map,risk_head)
        if risk_head is None:
            events.append(None);continue
        patient=split["patient"]
        events.append({
            "tuple":(split["y"],pulse,proposed["logits"],used,patient),
            "pulse":pulse,"source":residual,"support":res_sup,
            "proposed":proposed["logits"],"labels":split["y"],
            "patient":patient,
        })
    if risk_head is not None:
        patient_gate=admit_program_head_v60(
            risk_head,events[0]["tuple"],events[1]["tuple"])
        n1=conditional_shuffle_gate_v61(
            risk_head,events[0]["source"],events[0]["support"],
            events[0]["pulse"],events[0]["labels"],
            events[0]["patient"],observed_proposed=events[0]["proposed"],
            seed=20266121+seed)
        n2=conditional_shuffle_gate_v61(
            risk_head,events[1]["source"],events[1]["support"],
            events[1]["pulse"],events[1]["labels"],
            events[1]["patient"],observed_proposed=events[1]["proposed"],
            seed=20266171+seed)
        approved=bool(patient_gate["approved"] and n1["passed"] and n2["passed"])
        gate={
            "approved":approved,
            "reason":("CONDITIONALLY_NOVEL_BOTH_PATIENT_VALID_AND_NULLS"
                      if approved else "ABSTAIN_NO_REPLICATED_CONDITIONAL_SENSOR_EVIDENCE"),
            "patient_dual_VALID":patient_gate,
            "conditional_null_VALID1":n1,
            "conditional_null_VALID2":n2,
            "model_parameters_and_source_mapping_frozen_before_valid":True,
        }
    else:
        gate={
            "approved":False,"reason":"ABSTAIN_NO_SOURCE_BEYOND_PULSE_AFTER_RESIDUALIZATION",
            "patient_dual_VALID":None,
            "conditional_null_VALID1":None,
            "conditional_null_VALID2":None,
            "model_parameters_and_source_mapping_frozen_before_valid":True,
        }
    return pulse_head,source_map,risk_head,raw_ablation_head,gate


def _test_one(test,norm,bank,pulse_idx,source_idx,pulse_head,
              source_map,risk_head,unconditioned_head,gate,
              pulse_fit,lpulse,lall,*,seed,task,site,output):
    rows=[];private=[]
    for perturbation in STRESS:
        x,m=stress_v61(test,norm,perturbation,
                       seed=20266100+seed)
        v,s=infer_program_matrix_v60(bank,x,m)
        pulse=infer_program_head_v60(
            pulse_head,v[:,pulse_idx],s[:,pulse_idx],
            np.zeros(len(v)))
        z_pulse=pulse["logits"]
        if source_map is None or risk_head is None:
            # No measured *conditional innovation*; no hallucinated sources.
            r=np.zeros((len(v),0));r_sup=np.zeros_like(r)
            proposed=z_pulse.copy()
            deployed=z_pulse.copy()
            signed=np.zeros((len(v),0))
            q=np.zeros((len(v),0))
            active=np.zeros((len(v),0),bool)
            actually_used=np.zeros(len(v),bool)
            unconditional=z_pulse.copy()
        else:
            r,r_sup=infer_conditional_sources_v61(
                source_map,v,s,m)
            out=infer_program_head_v60(
                risk_head,r,r_sup,z_pulse)
            proposed=out["logits"]
            signed=out["signed_contributions"];q=out["memberships"]
            active=(out["used"][:,list(risk_head.active_indices)]
                    if risk_head.active_indices else
                    np.zeros((len(v),1),bool))
            deployed,actually_used=apply_program_admission_v60(
                z_pulse,proposed,active,approved=gate["approved"])
            raw_indices=source_map.raw_source_indices
            raw=infer_program_head_v60(
                unconditioned_head,
                v[:,raw_indices],s[:,raw_indices],z_pulse)
            unconditional=raw["logits"]
            if np.max(np.abs(proposed-z_pulse-signed.sum(axis=1)))>1e-9:
                raise AssertionError("Native signed V61 risk attribution missing")
            if not gate["approved"]:
                np.testing.assert_array_equal(deployed,z_pulse)
        # Exactly the same source data and TRAIN label budget for logistic.
        z_logpulse=_predict_logistic_programs(lpulse,v,s,pulse_idx)
        z_logall=_predict_logistic_programs(
            lall,v,s,tuple(range(len(bank.candidates))))
        actual_pulse=test["pulse"].copy()
        if perturbation=="pulse_missing":
            actual_pulse[:]=np.nan
        elif perturbation=="pulse_plus_25_bpm":
            actual_pulse=np.where(np.isfinite(actual_pulse),
                                  actual_pulse+25, np.nan)
        z_legacy=_pulse_logit(actual_pulse,pulse_fit)
        scores={
            "PulseProgramFAN":z_pulse,
            "ConditionalProgramFAN_proposed":proposed,
            "ConditionalProgramFAN_deployed":deployed,
            "UnconditionalNonpulse_ablation":unconditional,
            "PulseProgramLogistic":z_logpulse,
            "FullAllowedProgramLogistic":z_logall,
            "PulseMeanLogistic":z_legacy,
        }
        y=test["y"]
        measured={name:_as_metrics(y,logits,test["patient"])
                  for name,logits in scores.items()}
        pp=output/"private_case_arrays"
        pp.mkdir(parents=True,exist_ok=True)
        key=f"{site}_{task}_seed{seed}_{perturbation}"
        path=pp/f"{key}.npz"
        codes=np.unique(test["patient"],return_inverse=True)[1]
        np.savez_compressed(
            path,y=y.astype(np.uint8),patient_group=codes.astype(np.int16),
            pulse=z_pulse,proposed=proposed,deployed=deployed,
            unconditioned=unconditional,logistic_pulse=z_logpulse,
            logistic_full=z_logall,logistic_mean=z_legacy,
            source_q=q,source_signed=signed,
            source_support=r_sup,active_used=active.astype(np.uint8),
            admitted_used=actually_used.astype(np.uint8),
            pulse_q=pulse["memberships"],
            pulse_signed=pulse["signed_contributions"],
            pulse_support=pulse["support"],
        )
        private.append({
            "name":key,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "patient_level_predictions_not_uploaded":True,
        })
        rows.append({
            "site":site,"task":task,"seed":seed,"stress":perturbation,
            "n_patient_groups":len(np.unique(test["patient"])),
            "n_overlapping_windows":len(y),
            "positive_fraction":float(y.mean()),
            "available_conditional_source_fraction":
                float(active.any(axis=1).mean()) if active.size else 0.,
            "deployed_new_source_fraction":float(actually_used.mean()),
            "conditional_new_source_admitted":bool(gate["approved"]),
            "n_source_programs":0 if source_map is None else len(source_map.raw_source_indices),
            "metrics":measured,
            "paired_subject_bootstrap_deployed_vs_full_logistic":
                _bootstrap_by_real_patient(
                    y,deployed,z_logall,test["patient"],
                    seed=20266145+seed),
            "paired_subject_bootstrap_proposed_vs_pulse_program_logistic":
                _bootstrap_by_real_patient(
                    y,proposed,z_logpulse,test["patient"],
                    seed=20266155+seed),
            "source_is_not_physical_causal_measurement":True,
        })
        print("V61",site,task,seed,perturbation,
              "new",len(source_map.raw_source_indices) if source_map else 0,
              "admitted",gate["approved"],
              "PULSE",round(float(measured["PulseProgramFAN"]["nll"]),5),
              "proposed",round(float(measured["ConditionalProgramFAN_proposed"]["nll"]),5),
              flush=True)
    return rows,private


def execute(bidmc,out,*,vitaldb=None,tasks=TASKS,seeds=SEEDS):
    torch.set_num_threads(2)
    output=Path(out);output.mkdir(parents=True,exist_ok=True)
    entries=[];secrets=[];models=[];banks=[];abstained=[]
    for task in tasks:
        d,norm,manifest=load_bidmc_future_v61(Path(bidmc),task)
        (output/f"bidmc_{task}_split_receipt.json").write_text(
            json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
        train=d["train"]
        if np.unique(train["y"]).size!=2:
            abstained.append({"task":task,"reason":"ONE_CLASS_TRAIN_NO_TUNING"})
            continue
        allowed=TARGETS[task]["allowed_channels"]
        bank=compile_allowed_programs_v61(
            train["x"],train["mask"],train["patient"],
            allowed_channels=allowed,seed=20266113)
        pi,ni=select_conditioning_and_novel_programs_v61(
            bank,allowed)
        banks.append({
            "task":task,
            "bank":bank.manifest(),
            "pulse_program_indices":pi,
            "nonpulse_raw_indices":ni,
            "target_channel_prohibited":task!="future_resp_60s" or
                                         all(1 not in c.channels for c in bank.candidates),
            "risk_source_task_Y_used_for_discovery":False,
        })
        tr_z,tr_s=infer_program_matrix_v60(
            bank,train["x"],train["mask"])
        pulse_fit=_train_pulse_logistic(train)
        sites={"BIDMC":d["test"]}
        if vitaldb is not None:
            external,source_receipt=load_vitaldb_future_v61(
                Path(vitaldb),task,norm)
            sites["VitalDB"]=external
        for seed in seeds:
            lpulse=_fit_logistic_programs(
                tr_z,tr_s,train["y"],train["patient"],
                ids=pi,seed=20266112+seed)
            lall=_fit_logistic_programs(
                tr_z,tr_s,train["y"],train["patient"],
                ids=tuple(range(len(bank.candidates))),seed=20266132+seed)
            ph,source_map,head,ungated,gate=_fit_pulse_and_conditional(
                train,d["valid1"],d["valid2"],bank,
                pi,ni,seed=seed)
            models.append({
                "task":task,"seed":seed,
                "pulse_native_PyTorch_head":ph.passport(),
                "nonpulse_train_only_source_map":(
                    None if source_map is None else source_map.passport()),
                "conditional_native_PyTorch_head":(
                    None if head is None else head.passport()),
                "unconditioned_ablation_head":(
                    None if ungated is None else ungated.passport()),
                "two_patient_VALID_and_conditional_shuffle_gate":gate,
                "target_channel_available_to_any_model":False
                    if task=="future_resp_60s" else None,
                "TRAIN_Y_used_to_fit_program_definition":False,
            })
            for site,heldout in sites.items():
                lines,files=_test_one(
                    heldout,norm,bank,pi,ni,ph,source_map,head,ungated,
                    gate,pulse_fit,lpulse,lall,
                    seed=seed,task=task,site=site,output=output)
                entries.extend(lines);secrets.extend(files)
    report={
        "protocol":PROTOCOL,
        "status":"V61_EXECUTED_REAL_CONDITIONAL_NOVELTY_EXPLORATORY",
        "real_sites":sorted({x["site"] for x in entries}),
        "registered_tasks":list(tasks),
        "tasks_not_estimable_on_TRAIN":abstained,
        "n_trained_models":len(models),
        "n_test_scenarios":len(entries),
        "each_native_seed_reuses_same_real_patients":True,
        "outcome_label_always_starts_after_end_plus_60s":True,
        "future_resp_target_excluded_from_input_and_scaling":True,
        "new_programs_do_not_read_PULSE_directly":True,
        "source_conditional_on_pulse_TRAIN_values_only":True,
        "new_source_claims_only_statistical_incremental_value":True,
        "source_gate_patient_cluster_and_pulse_stratified_null":True,
        "sites_previously_inspected_NOT_confirmatory":True,
        "person_ids_raw_observations_and_predictions_not_uploaded":True,
        "program_banks":banks,"frozen_model_and_gate_receipts":models,
        "test_cases":entries,"private_prediction_hashes":secrets,
    }
    (output/"report_v61.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    with (output/"aggregate_cases_v61.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.writer(f)
        named=("PulseProgramFAN","ConditionalProgramFAN_proposed",
               "ConditionalProgramFAN_deployed",
               "PulseProgramLogistic","FullAllowedProgramLogistic",
               "UnconditionalNonpulse_ablation","PulseMeanLogistic")
        writer.writerow(["task","site","seed","stress","patients",
                         "windows","positive_rate","nonpulse_admitted",
                         *[x+"_AUPRC" for x in named],
                         *[x+"_NLL" for x in named]])
        for item in entries:
            metrics=item["metrics"]
            writer.writerow([
                item["task"],item["site"],item["seed"],item["stress"],
                item["n_patient_groups"],item["n_overlapping_windows"],
                item["positive_fraction"],
                item["conditional_new_source_admitted"],
                *[metrics[x]["auprc"] for x in named],
                *[metrics[x]["nll"] for x in named],
            ])
    print(json.dumps({
        "status":report["status"],"n_models":len(models),
        "scenarios":len(entries),"no_train_class":abstained,
        "admitted":sum(x["two_patient_VALID_and_conditional_shuffle_gate"]["approved"]
                       for x in models)},indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--bidmc",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--vitaldb",type=Path,default=None)
    p.add_argument("--tasks",nargs="+",choices=TASKS,default=list(TASKS))
    p.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS))
    args=p.parse_args()
    execute(args.bidmc,args.out,vitaldb=args.vitaldb,
            tasks=tuple(args.tasks),seeds=tuple(args.seeds))
