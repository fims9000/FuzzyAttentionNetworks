#!/usr/bin/env python3
"""Posthoc descriptive V60 audit versus strongest PULSE-only baseline.

This file was added AFTER the preregistered V60 BIDMC/VitalDB TEST was first
inspected. It does NOT select, train, alter or admit any source model or
amend primary experiment thresholds. Its inference bands are diagnostic and
not confirmatory; resampling unit is a whole real patient (not windows).
Reports source-program dependency, exposing whether all programs still rely
on a single PULSE monitor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score

BOOTSTRAP_REPS=399
METHODS=("pure","residual","deployed")


def subject_resampling(y,patient,a,b,*,seed):
    rng=np.random.default_rng(seed)
    groups=np.unique(patient)
    ids=[np.flatnonzero(patient==p) for p in groups]
    auprc=[]
    nll=[]
    for _ in range(BOOTSTRAP_REPS):
        selected=rng.integers(0,len(groups),len(groups))
        ix=np.concatenate([ids[j] for j in selected])
        yy=y[ix]
        if np.unique(yy).size<2:continue
        auprc.append(float(
            average_precision_score(yy,expit(a[ix]))-
            average_precision_score(yy,expit(b[ix]))))
        # Patient-macro NLL difference, not pooled overlapping windows.
        nll.append(float(np.mean([
            (np.logaddexp(0.,a[ids[j]])-y[ids[j]]*a[ids[j]]
             -np.logaddexp(0.,b[ids[j]])+y[ids[j]]*b[ids[j]]
            ).mean() for j in selected])))
    if len(auprc)<100:return {
        "status":"NOT_ESTIMABLE_FEW_TWO_CLASS_GROUP_DRAWS",
        "valid_draws":len(auprc)}
    return {
        "status":"POSTHOC_DESCRIPTIVE_ONLY",
        "positive_AUPRC_difference":float(
            average_precision_score(y,expit(a))-
            average_precision_score(y,expit(b))) if np.unique(y).size==2 else None,
        "AUPRC_paired_subject_95pct":list(map(float,np.quantile(auprc,[.025,.975]))),
        "patient_macro_NLL_paired_95pct":list(map(float,np.quantile(nll,[.025,.975]))),
        "actual_real_subjects":len(groups),
        "valid_bootstrap_draws":len(auprc),
        "bootstrap_unit":"distinct real subject",
    }


def execute(path):
    path=Path(path)
    original=json.loads((path/"report_v60.json").read_text())
    models={(x["task"],int(x["native_seed"])):x
            for x in original["fit_and_gate_receipts"]}
    banks={x["task"]:x["program_bank"]["candidates"]
           for x in original["source_manifests"]}
    results=[]
    for entry in original["test_cases"]:
        if entry["stress"]!="natural":continue
        task=entry["task"];seed=int(entry["seed"])
        case=f"{entry['site']}_{task}_seed{seed}_natural"
        with np.load(path/"private_case_arrays"/f"{case}.npz",
                     allow_pickle=False) as z:
            yy=z["y"].copy()
            groups=z["patient_group"].copy()
            pulse=z["pulse_only"].copy()
            variants={key:z[key].copy() for key in METHODS}
        rec=models[task,seed]
        fit=rec["all_program_fits_TRAIN_only"][
            rec["selected_variant_TRAIN_only"]]
        chosen=[banks[task][i] for i in fit["active_indices"]]
        n_require_pulse=sum(0 in p["channels"] for p in chosen)
        contrast={
            name:subject_resampling(
                yy,groups,logits,pulse,
                seed=20266033+seed+sum(ord(c) for c in entry["site"]+task+name))
            for name,logits in variants.items()
        }
        results.append({
            "site":entry["site"],"task":task,"native_seed":seed,
            "selected_variant":fit["variant"],
            "dual_VALID_approved":rec["chosen_head_patient_VALID_receipt"]["approved"],
            "n_active_programs":len(chosen),
            "active_operator_chains":[
                {"operator":p["operator"],
                 "measured_channel_names":p["channel_names"],
                 "train_gamma":float(fit["gamma"][i]),
                 "device_causal_identity_claim":False}
                for i,p in zip(fit["active_indices"],chosen)],
            "n_active_programs_reading_PULSE":n_require_pulse,
            "n_active_programs_not_reading_PULSE":len(chosen)-n_require_pulse,
            "all_active_programs_depend_on_PULSE":n_require_pulse==len(chosen),
            "only_measured_source_programs_not_new_physical_identities":True,
            "paired_subject_comparison_vs_PULSE_only":contrast,
        })
    receipt={
        "status":"V60_POSTHOC_DEPENDENCE_AND_BASELINE_AUDIT_DONE",
        "posthoc_after_primary_TEST_seen":True,
        "no_model_training_selection_or_admission_change":True,
        "new_observed_external_site_or_independent_confirmatory_data":False,
        "sample_unit_distinct_subjects_not_overlapping_windows":True,
        "no_per_person_scores_published":True,
        "comparison_groups":results,
    }
    (path/"posthoc_v60_source_dependence.json").write_text(
        json.dumps(receipt,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
        "status":receipt["status"],
        "tested_site_task_seed_groups":len(results),
        "all_active_concepts_still_PULSE_related":
        all(x["all_active_programs_depend_on_PULSE"] for x in results),
        "no_model_refit":True},indent=2))
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dir",type=Path,required=True)
    a=p.parse_args()
    execute(a.dir)
