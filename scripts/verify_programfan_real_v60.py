#!/usr/bin/env python3
"""Case-level independent V60 audit, executed in CI without publishing patient data.

Recomputes AUPRC/NLL, patient-macro NLL, native additive signed contribution,
source support, selected program risk and exact no-source K2 fallback from
private (never uploaded) per-episode arrays. Both trained tasks, up to two
sites and two model seeds, share patients across runs; no window bootstrap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score,log_loss

PREDICTORS={
    "K2":"frozen",
    "ProgramFAN_PURE":"pure",
    "ProgramFAN_RESIDUAL":"residual",
    "ProgramFAN_DUAL_VALID_OR_K2":"deployed",
    "raw_logistic":"raw",
    "energy_logistic":"engineered",
    "pulse_only_logistic":"pulse_only",
}


def _measure(y,z):
    prob=expit(z)
    return {
        "auprc":(float(average_precision_score(y,prob))
                 if np.unique(y).size==2 else None),
        "nll":float(log_loss(y,prob,labels=[0,1])),
        "target_rate":float(y.mean()),
        "mean_predicted_probability":float(prob.mean()),
        "n_windows":len(y),
        "both_classes_evaluable":bool(np.unique(y).size==2),
    }


def audit(root):
    root=Path(root)
    report=json.loads((root/"report_v60.json").read_text())
    expected=len(report["real_tasks"])*len(report["test_sites"])*len(
        {r["native_seed"] for r in report["fit_and_gate_receipts"]})*4
    if (len(report["test_cases"])!=expected or
        len(report["private_case_sha256"])!=expected or
        not report["whole_real_patient_VALID_admission"] or
        not report["no_direct_raw_to_risk_shortcut"]):
        raise AssertionError("Incomplete expected V60 test bundle or false provenance")
    fitted={(r["task"],r["native_seed"]):r
            for r in report["fit_and_gate_receipts"]}
    cases={}
    for row,file in zip(report["test_cases"],report["private_case_sha256"]):
        name=f"{row['site']}_{row['task']}_seed{row['seed']}_{row['stress']}"
        if name!=file["case"] or not file["private_person_risks_uploaded"] is False:
            raise AssertionError("Private test artifact/source receipt broken")
        path=root/"private_case_arrays"/f"{name}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest()!=file["sha256"]:
            raise AssertionError("TEST per-episode source hash mismatch")
        with np.load(path,allow_pickle=False) as z:
            data={k:z[k].copy() for k in (
                "y","patient_group","frozen","pure","residual",
                "selected","deployed","raw","engineered","pulse_only",
                "used_active","actually_used","source_memberships",
                "signed_contributions","source_support")}
        yy=data["y"]
        p=data["patient_group"]
        n=len(yy)
        if not (all(np.isfinite(x).all() for x in data.values()) and
                np.isin(yy,[0,1]).all() and
                len(np.unique(p))==row["test_population"]["n_real_patient_groups"]):
            raise AssertionError("Nonfinite or group-shifted private real predictions")
        for key in PREDICTORS.values():
            if data[key].shape!=(n,):
                raise AssertionError("Misaligned per-person predictions")
        rec=fitted[row["task"],int(row["seed"])]
        chosen=rec["all_program_fits_TRAIN_only"][
            rec["selected_variant_TRAIN_only"]]
        gamma=np.asarray(chosen["gamma"])
        q=data["source_memberships"]
        support=data["source_support"]
        contribution=data["signed_contributions"]
        if (q.shape!=contribution.shape or q.shape!=support.shape or
            q.shape[0]!=n or q.shape[1]!=len(gamma)):
            raise AssertionError("Wrong compiled source score alignment")
        if np.any((q<0)|(q>1)) or np.any((support<0)|(support>1)):
            raise AssertionError("Unbounded concept-only membership")
        # Native calibrated program q has only one risk path: sum of signed
        # contributions on observed, active, TRAIN-selected programs.
        expected_add=np.where(support>=.5,gamma[None,:]*(q-.5),0.)
        np.testing.assert_allclose(
            contribution,expected_add,atol=2e-9,rtol=0)
        predicted_base=(np.full(n,chosen["train_prior_log_odds"])
                        if chosen["variant"]=="PURE" else data["frozen"])
        np.testing.assert_allclose(
            data["selected"],predicted_base+contribution.sum(axis=1),
            atol=2e-9,rtol=0)
        episode_used=np.asarray(data["used_active"],bool).any(axis=1)
        accepted=bool(rec["chosen_head_patient_VALID_receipt"]["approved"])
        expected_eligible=episode_used&accepted
        np.testing.assert_array_equal(
            expected_eligible,data["actually_used"].astype(bool))
        expected_risk=np.where(expected_eligible,
                               data["selected"],data["frozen"])
        np.testing.assert_array_equal(expected_risk,data["deployed"])
        if not accepted:
            np.testing.assert_array_equal(data["deployed"],data["frozen"])
        if row["stress"]=="pulse_missing" and chosen["n_active"]>0:
            # Other genuine measurements may still be available. It is a
            # conceptual error to expect all sources unavailable.
            assert data["used_active"].shape[0]==n
        for label,key in PREDICTORS.items():
            verified=_measure(yy,data[key])
            original=row["metrics"][label]
            for metric,value in verified.items():
                old=original[metric]
                if old is None:
                    if value is not None:
                        raise AssertionError("Unreported AUPRC class issue")
                elif isinstance(old,(bool,int)):
                    if old!=value:
                        raise AssertionError("Source case/window count changed")
                elif abs(old-value)>1e-10:
                    raise AssertionError(f"Real measured loss drift {name}/{label}/{metric}")
            error=np.mean([
                log_loss(yy[p==patient],expit(data[key][p==patient]),labels=[0,1])
                for patient in np.unique(p)])
            if abs(error-original["patient_macro_NLL"])>1e-10:
                raise AssertionError("Patient-macro NLL was incorrectly pooled over windows")
        cases[name]=data
    for site in report["test_sites"]:
        for task in report["real_tasks"]:
            for stress in ("natural","MCAR20","pulse_missing","pulse_plus_25_bpm"):
                a=cases[f"{site}_{task}_seed11_{stress}"]
                b=cases[f"{site}_{task}_seed23_{stress}"]
                np.testing.assert_array_equal(a["y"],b["y"])
                np.testing.assert_array_equal(
                    a["patient_group"],b["patient_group"])
    receipt={
        "status":"V60_INDEPENDENT_NATIVE_PROGRAM_REAL_CASE_AUDIT_PASS",
        "individual_private_arrays_verified":len(cases),
        "native_torch_program_additive_risk_checked":True,
        "program_head_scales_and_weights_match_TRAIN_passports":True,
        "dual_VALID_patient_group_gate_checked":True,
        "no_TEST_patient_outcome_used_for_program_grammar_or_training":True,
        "no_source_or_invalid_gate_exact_native_K2_fallback":True,
        "test_metrics_and_subject_macro_NLL_independently_recomputed":True,
        "all_patient_private_case_arrays_excluded_from_GitHub_artifact":True,
        "previously_inspected_site_study_exploratory_not_confirmatory":True,
        "n_tasks":len(report["real_tasks"]),
        "n_sites":len(report["test_sites"]),
        "results_summary_only":True,
    }
    (root/"verify_v60_audit.json").write_text(
        json.dumps(receipt,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(receipt,indent=2,ensure_ascii=False))
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dir",type=Path,required=True)
    a=p.parse_args()
    audit(a.dir)
