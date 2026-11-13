#!/usr/bin/env python3
"""V61R independent PROCESS replay: exact source IDs and numerical logits.

This is only a posthoc engineering reproducibility check. It does NOT turn
already inspected hospital TEST data into a new confirmatory clinical study.
Private patient-level arrays must remain in GitHub Actions runner memory;
only aggregate max residual/logit drift and hashes are uploaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LOGIT_TOL=1.e-4
FIELDS=("pulse","proposed","deployed","unconditioned",
        "logistic_pulse","logistic_full","logistic_mean")
SOURCE_FIELDS=("source_q","source_signed","source_support","pulse_q")


def verify(first:Path,second:Path):
    a=json.loads((first/"report_v61.json").read_text())
    b=json.loads((second/"report_v61.json").read_text())
    if any(r["status"]!="V61R_EXECUTED_POSTHOC_STABILITY_RESEARCH" for r in (a,b)):
        raise AssertionError("Two replay runs not V61R frozen posthoc stability protocol")
    if (len(a["test_cases"])!=len(b["test_cases"]) or
        len(a["frozen_model_and_gate_receipts"])!=len(b["frozen_model_and_gate_receipts"])):
        raise AssertionError("V61R replay changed task or source enrollment")
    model_a={(r["task"],int(r["seed"])):r
             for r in a["frozen_model_and_gate_receipts"]}
    model_b={(r["task"],int(r["seed"])):r
             for r in b["frozen_model_and_gate_receipts"]}
    for key in model_a:
        r=model_a[key];s=model_b[key]
        if r["stable_outcome_blind_program_ids"]!=s["stable_outcome_blind_program_ids"]:
            raise AssertionError("TRAIN-only new concept identities unstable across independent runs")
        if r["stable_source_contract_SHA256"]!=s["stable_source_contract_SHA256"]:
            raise AssertionError("12-digit canonical source residualizer still nonreproducible")
        if (r["two_patient_VALID_and_conditional_shuffle_gate"]["approved"]
            !=s["two_patient_VALID_and_conditional_shuffle_gate"]["approved"]):
            raise AssertionError("Patient-level source admission changed after replay")
    max_drift={x:0. for x in FIELDS+SOURCE_FIELDS}
    for row in a["test_cases"]:
        key=f"{row['site']}_{row['task']}_seed{row['seed']}_{row['stress']}"
        x=first/"private_case_arrays"/f"{key}.npz"
        y=second/"private_case_arrays"/f"{key}.npz"
        if not x.exists() or not y.exists():raise AssertionError("Missing private TEST replay")
        with np.load(x,allow_pickle=False) as p,np.load(y,allow_pickle=False) as q:
            for field in ("y","patient_group","active_used","admitted_used"):
                np.testing.assert_array_equal(p[field],q[field])
            for field in FIELDS+SOURCE_FIELDS:
                if p[field].shape!=q[field].shape:
                    raise AssertionError("Source dimension changed between fresh processes")
                diff=float(np.max(np.abs(p[field]-q[field]))) if p[field].size else 0.
                max_drift[field]=max(max_drift[field],diff)
                if diff>LOGIT_TOL:
                    raise AssertionError(
                        f"UNSTABLE_V61R_REPLAY: {key} {field} max_drift={diff}")
    report={
        "status":"V61R_INDEPENDENT_PROCESS_REPLAY_WITHIN_TOLERANCE",
        "posthoc_not_new_confirmatory_data":True,
        "compared_true_independent_python_processes":True,
        "canonical_source_contracts_identical_sha256":True,
        "frozen_dual_patient_gate_decisions_replay_identical":True,
        "identical_patient_group_and_test_labels":True,
        "real_sites":a["real_sites"],
        "n_test_arrays_compared":len(a["test_cases"]),
        "max_abs_allowed_error":LOGIT_TOL,
        "observed_max_abs_drift":max_drift,
        "actual_private_patient_risks_exported":False,
    }
    (first/"verify_v61r_replay.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--first",required=True,type=Path)
    p.add_argument("--second",required=True,type=Path)
    args=p.parse_args()
    verify(args.first,args.second)
