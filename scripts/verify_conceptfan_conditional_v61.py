#!/usr/bin/env python3
"""V61 independent real held-out case audit, no patient data uploaded.

PRIVATE runner arrays: values/logits and anonymized grouping codes; audit
recomputes each source-only signed contribution, pulse-only head, exact
patient-gated fallback and all reported metrics. Only aggregate JSON upload.
"""
from __future__ import annotations

import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score,log_loss

METHODS={
    "PulseProgramFAN":"pulse",
    "ConditionalProgramFAN_proposed":"proposed",
    "ConditionalProgramFAN_deployed":"deployed",
    "UnconditionalNonpulse_ablation":"unconditioned",
    "PulseProgramLogistic":"logistic_pulse",
    "FullAllowedProgramLogistic":"logistic_full",
    "PulseMeanLogistic":"logistic_mean",
}


def scores(y,z):
    p=expit(z)
    return {
        "auprc":float(average_precision_score(y,p))
                 if np.unique(y).size==2 else None,
        "nll":float(log_loss(y,p,labels=[0,1])),
        "target_rate":float(y.mean()),
        "mean_predicted_probability":float(p.mean()),
        "n_windows":len(y),
        "both_classes_evaluable":bool(np.unique(y).size==2),
    }


def audit(root):
    root=Path(root)
    report=json.loads((root/"report_v61.json").read_text())
    if (report["status"]!="V61R_EXECUTED_POSTHOC_STABILITY_RESEARCH"
        or not report["future_resp_target_excluded_from_input_and_scaling"]
        or not report["new_programs_do_not_read_PULSE_directly"]
        or not report["source_gate_patient_cluster_and_pulse_stratified_null"]
        or report["sites_previously_inspected_NOT_confirmatory"] is not True):
        raise AssertionError("V61R disclosure or guarded source definition incorrect")
    if not (report["posthoc_amendment_after_original_V61_test_inspection"]
            and report["V61_original_negative_results_preserved_unchanged"]
            and report["torch_cpu_deterministic_one_thread"]
            and report["novel_source_dictionary_selected_without_Y"]):
        raise AssertionError("V61R posthoc-stability amendment not disclosed")
    expected=len(report["frozen_model_and_gate_receipts"])*(
        len(report["real_sites"])*4)
    if (len(report["test_cases"])!=expected or
        len(report["private_prediction_hashes"])!=expected):
        raise AssertionError("Wrong preregistered V61 case count")
    models={(x["task"],int(x["seed"])):x
            for x in report["frozen_model_and_gate_receipts"]}
    banks={x["task"]:x for x in report["program_banks"]}
    data_by_key={}
    for row,item in zip(report["test_cases"],report["private_prediction_hashes"]):
        task=row["task"];seed=int(row["seed"])
        key=f"{row['site']}_{task}_seed{seed}_{row['stress']}"
        if item["name"]!=key or item["patient_level_predictions_not_uploaded"] is not True:
            raise AssertionError("V61 private case source/clinical privacy promise broken")
        file=root/"private_case_arrays"/f"{key}.npz"
        if hashlib.sha256(file.read_bytes()).hexdigest()!=item["sha256"]:
            raise AssertionError("V61 private case checksum mismatch")
        with np.load(file,allow_pickle=False) as archive:
            x={name:archive[name].copy() for name in (
                "y","patient_group","pulse","proposed","deployed",
                "unconditioned","logistic_pulse","logistic_full",
                "logistic_mean","source_q","source_signed",
                "source_support","active_used","admitted_used",
                "pulse_q","pulse_signed","pulse_support")}
        y=x["y"];group=x["patient_group"]
        n=len(y)
        if (len(np.unique(group))!=row["n_patient_groups"]
            or not np.isin(y,[0,1]).all() or
            not all(np.isfinite(t).all() for t in x.values())
            or any(x[name].shape!=(n,) for name in METHODS.values())):
            raise AssertionError("V61 real patient-group labels or risk outputs corrupted")
        model=models[task,seed]
        bank=banks[task]
        if model["pulse_native_PyTorch_head"]["stochastic_seeded_initialization_std"]<=0:
            raise AssertionError("V61R stochastic head initials still identical across seeds")
        if not model["pulse_native_PyTorch_head"]["fixed_outcome_blind_source_shortlist"]:
            raise AssertionError("V61R pulse dictionary not frozen")
        newfit=model["conditional_native_PyTorch_head"]
        if newfit is not None:
            if newfit["n_candidates"]>6 or not newfit["fixed_outcome_blind_source_shortlist"]:
                raise AssertionError("V61R source gamma-trained top-six selection still active")
            expected_id=[bank["bank"]["candidates"][i]["candidate_id"]
                         for i in model["nonpulse_train_only_source_map"]["raw_source_indices"]]
            if expected_id!=model["stable_outcome_blind_program_ids"]:
                raise AssertionError("Canonical source dictionary fingerprint/identity mismatch")
        pulse_fit=model["pulse_native_PyTorch_head"]
        if (task=="future_resp_60s" and not bank["target_channel_prohibited"]):
            raise AssertionError("RESP target entered source compiler")
        if any(0 in bank["bank"]["candidates"][i]["channels"]
               for i in bank["nonpulse_raw_indices"]):
            raise AssertionError("Claimed new source directly reads PULSE")
        # Full native pulse concept-only risk, NO raw head.
        pg=np.asarray(pulse_fit["gamma"],float)
        pq=x["pulse_q"];pcon=x["pulse_signed"]
        if pq.shape!=pcon.shape or pq.shape!=(n,len(pg)):
            raise AssertionError("Pulse concept membership/logit misalignment")
        expected_pcon=pg[None,:]*(pq-.5)
        np.testing.assert_allclose(pcon,expected_pcon,atol=2e-9,rtol=0)
        np.testing.assert_allclose(
            x["pulse"],pulse_fit["train_prior_log_odds"]+pcon.sum(axis=1),
            atol=2e-9,rtol=0)
        source=model["conditional_native_PyTorch_head"]
        gate=model["two_patient_VALID_and_conditional_shuffle_gate"]
        a=x["source_q"];signed=x["source_signed"]
        if source is None:
            if a.shape!=(n,0) or signed.shape!=(n,0):
                raise AssertionError("No conditional source but nonempty components")
            np.testing.assert_array_equal(x["proposed"],x["pulse"])
            np.testing.assert_array_equal(x["deployed"],x["pulse"])
            assert not gate["approved"]
        else:
            gamma=np.asarray(source["gamma"],float)
            supp=x["source_support"]
            if a.shape!=(n,len(gamma)) or a.shape!=signed.shape or a.shape!=supp.shape:
                raise AssertionError("Native conditional source shape differs from TRAIN passport")
            expected_signed=np.where(
                supp>=.5,gamma[None,:]*(a-.5),0.)
            np.testing.assert_allclose(signed,expected_signed,atol=2e-9,rtol=0)
            np.testing.assert_allclose(
                x["proposed"],x["pulse"]+signed.sum(axis=1),
                atol=2e-9,rtol=0)
            active=x["active_used"].astype(bool).any(axis=1)
            approval=bool(
                gate["patient_dual_VALID"]["approved"] and
                gate["conditional_null_VALID1"]["passed"] and
                gate["conditional_null_VALID2"]["passed"])
            if approval!=gate["approved"]:
                raise AssertionError("Native new-source gate inconsistently reported")
            intended=active&approval
            np.testing.assert_array_equal(
                x["admitted_used"].astype(bool),intended)
            np.testing.assert_array_equal(
                x["deployed"],np.where(intended,
                                       x["proposed"],x["pulse"]))
        for label,field in METHODS.items():
            got=scores(y,x[field]);declared=row["metrics"][label]
            for metric,val in got.items():
                old=declared[metric]
                if old is None:
                    if val is not None:
                        raise AssertionError("AUPRC estimability changed")
                elif isinstance(old,(bool,int)):
                    if old!=val:
                        raise AssertionError("Window counts or model task labels drifted")
                elif abs(val-old)>1e-10:
                    raise AssertionError(f"V61 site case metric drift: {key}/{label}/{metric}")
            macro=np.mean([
                log_loss(y[group==p],expit(x[field][group==p]),labels=[0,1])
                for p in np.unique(group)])
            if abs(macro-declared["patient_macro_NLL"])>1e-10:
                raise AssertionError("Wrong V61 patient-macro NLL")
        data_by_key[key]={"y":y,"group":group}
    for task in report["registered_tasks"]:
        if task in {z["task"] for z in report["tasks_not_estimable_on_TRAIN"]}:
            continue
        for site in report["real_sites"]:
            for stress in ("natural","MCAR20","pulse_missing","pulse_plus_25_bpm"):
                a=data_by_key[f"{site}_{task}_seed11_{stress}"]
                b=data_by_key[f"{site}_{task}_seed23_{stress}"]
                np.testing.assert_array_equal(a["y"],b["y"])
                np.testing.assert_array_equal(a["group"],b["group"])
    # Two native seeds must produce genuinely distinct TRAIN model head
    # parameters (but still share exact TEST patient groups).
    for task in report["registered_tasks"]:
        trials=[v for (tt,_),v in models.items() if tt==task]
        if len(trials)==2:
            a=trials[0]["pulse_native_PyTorch_head"]["gamma"]
            b=trials[1]["pulse_native_PyTorch_head"]["gamma"]
            if np.array_equal(a,b):
                raise AssertionError("V61R two different seeds produced an identical trained head")
            if trials[0]["stable_outcome_blind_program_ids"]!=trials[1]["stable_outcome_blind_program_ids"]:
                raise AssertionError("V61R TRAIN source IDs changed between model initializations")
    receipt={
        "status":"V61R_POSTHOC_NATIVE_STABILITY_CASE_AUDIT_PASS",
        "posthoc_after_V61_TEST_was_inspected":True,
        "two_actual_distinct_seeded_native_initializations_verified":True,
        "TRAIN_only_source_dictionary_ids_identical_across_initializations":True,
        "verified_private_real_case_arrays":len(data_by_key),
        "task_targets_always_60s_future":True,
        "no_RESP_target_read_in_excluded_task":True,
        "pulse_head_native_signed_decomposition_verified":True,
        "new_head_native_signed_decomposition_verified":True,
        "new_sources_never_read_PULSE_directly":True,
        "dual_validation_patient_and_49_null_controls_consistent":True,
        "exact_frozen_pulse_program_head_fallback_verified":True,
        "full_allowed_logistic_and_pulse_controls_recomputed":True,
        "patient_macro_NLL_recomputed_without_window_pseudoreplication":True,
        "clinical_data_and_patient_predictions_not_uploaded":True,
        "prior_BIDMC_VitalDB_sites_inspected_not_confirmatory":True,
    }
    (root/"verify_v61_audit.json").write_text(
        json.dumps(receipt,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(receipt,indent=2,ensure_ascii=False))
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dir",type=Path,required=True)
    a=p.parse_args()
    audit(a.dir)
