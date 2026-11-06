"""V51 NEW fresh-device energy bias correction from metrologically noisy assay.

Two independent VALIDs approve or reject TRAIN-only corrections, with exact
old raw fallback. TEST hidden physical energy is used ONLY for scoring,
never for fitting maps, selecting model, or adjusting a decision threshold.
"""
from __future__ import annotations

import argparse,json
from pathlib import Path
import numpy as np

from conceptfan_discovery.observational_twins_v51 import (
    DEVICES_V51,make_observational_twins_v51,
)
from conceptfan_discovery.independent_anchor_certificate_v50 import (
    fit_attested_source_calibration_v50,assess_semantic_scope_v50,
)
from conceptfan_discovery.train_only_energy_debias_v51 import (
    fit_train_only_energy_debias_v51,infer_energy_with_train_only_debias_v51,
)

BUDGETS=(.05,.20,1.)
METHODS=("V50_RAW","V51_SCALE","V51_AFFINE")
REGIMES=("healthy_MCAR20","healthy_MCAR40","amplifier_shift_MCAR20")
CONTRASTS=(("V51_SCALE","V50_RAW"),
           ("V51_AFFINE","V50_RAW"),
           ("V51_AFFINE","V51_SCALE"))


def _source_energy_metrics(y,p):
    return {
        "n":len(y),
        "mae":float(np.mean(np.abs(p-y))),
        "rmse":float(np.sqrt(np.mean((p-y)**2))),
        "pearson":float(np.corrcoef(p,y)[0,1])
    }


def _paired_ci_absolute_error(y,a,b,*,seed:int,bootstrap:int):
    rng=np.random.default_rng(seed)
    errors=np.abs(a-y)-np.abs(b-y)
    n=len(y)
    observed=float(np.mean(errors))
    dist=np.array([np.mean(errors[rng.integers(0,n,n)])
                   for _ in range(bootstrap)])
    return {
        "difference_MAE_a_minus_b":observed,
        "paired_TEST_subject_bootstrap_95pct_ci":[float(np.quantile(dist,.025)),
                                                 float(np.quantile(dist,.975))],
        "resamples":bootstrap,
        "untouched_TEST_subjects":n,
    }


def run(out:Path,*,smoke:bool=False):
    if out.exists():
        raise FileExistsError("Never overwrite V51 independent TEST findings")
    devices=(DEVICES_V51[0],) if smoke else DEVICES_V51
    ntr,nv1,nv2,nte=(500,200,200,240) if smoke else (1200,400,400,600)
    bstrap=199 if smoke else 399
    rows=[];calibrations=[];comparisons=[];proofs=[];receipts={}
    for device in devices:
        world=make_observational_twins_v51(
            device,n_train=ntr,n_valid1=nv1,n_valid2=nv2,n_test=nte)
        tr,v1,v2,te=(world[z] for z in ("train","valid1","valid2","test"))
        x=world["shared_observed"]
        m=world["measured_boolean_masks"]
        receipts[f"device{device}_TRAIN_X"]=x[tr].copy()
        receipts[f"device{device}_TRAIN_mask"]=m["TRAIN_MCAR20"][tr].copy()
        receipts[f"device{device}_VALID1_X"]=x[v1].copy()
        receipts[f"device{device}_VALID1_mask"]=m["VALID1_MCAR20"][v1].copy()
        receipts[f"device{device}_VALID2_X"]=x[v2].copy()
        receipts[f"device{device}_VALID2_mask"]=m["VALID2_MCAR40"][v2].copy()
        receipts[f"device{device}_TEST_X"]=x[te].copy()
        receipts[f"device{device}_TEST_shifted_X"]=world["test_only_amplifier_readout"][te].copy()
        receipts[f"device{device}_TEST_mask20"]=m["TEST_MCAR20"][te].copy()
        receipts[f"device{device}_TEST_mask40"]=m["TEST_MCAR40"][te].copy()
        receipt_energies=[]
        for gauge_world in world["worlds"]:
            gauge=gauge_world["gauge_id"]
            assay=gauge_world["attested_external_assay"]
            energy_true=gauge_world[
                "hidden_physical_source_energy_TEST_AUDIT_ONLY"][te]
            receipts[f"device{device}_gauge{gauge}_TRAIN_assay"]=assay[tr].copy()
            receipts[f"device{device}_gauge{gauge}_VALID1_assay"]=assay[v1].copy()
            receipts[f"device{device}_gauge{gauge}_VALID2_assay"]=assay[v2].copy()
            receipts[f"device{device}_gauge{gauge}_TEST_E_AUDIT_ONLY"]=energy_true.copy()
            receipt_energies.append(energy_true)
            for budget in BUDGETS:
                seed=20265041+device%101
                source=fit_attested_source_calibration_v50(
                    x[tr],m["TRAIN_MCAR20"][tr],assay[tr],
                    x[v1],m["VALID1_MCAR20"][v1],assay[v1],
                    x[v2],m["VALID2_MCAR40"][v2],assay[v2],
                    physical_device_id=gauge_world["physical_device_id"],
                    independent_reference_id=gauge_world["external_reference_id"],
                    train_reference_budget_fraction=budget,seed=seed)
                debias=fit_train_only_energy_debias_v51(
                    source,
                    x[tr],m["TRAIN_MCAR20"][tr],assay[tr],
                    x[v1],m["VALID1_MCAR20"][v1],assay[v1],
                    x[v2],m["VALID2_MCAR40"][v2],assay[v2],
                    train_reference_budget_fraction=budget,seed=seed)
                calibrations.append({
                    "device":device,"gauge":gauge,"budget":budget,
                    "external_assay_noise_known_before_train":.16**2,
                    "TRAIN_only_source_calibration":source.audit(),
                    "TRAIN_and_two_VALID_energy_correction":debias.audit(),
                    "TEST_source_truth_not_used_in_calibration":True,
                })
                for regime in REGIMES:
                    xt=(world["test_only_amplifier_readout"][te]
                        if regime.startswith("amplifier") else x[te])
                    mt=(m["TEST_MCAR40"][te] if regime=="healthy_MCAR40"
                        else m["TEST_MCAR20"][te])
                    scope=assess_semantic_scope_v50(
                        source,xt,mt,device_id=gauge_world["physical_device_id"],
                        external_reference_id=gauge_world["external_reference_id"])
                    for method in METHODS:
                        pred=infer_energy_with_train_only_debias_v51(
                            source,debias,xt,mt,mode=method)
                        values=pred["energy"]
                        key=f"device{device}_gauge{gauge}_budget{int(budget*100)}_{regime}_{method}"
                        receipts[key+"_source_energy"]=values.copy()
                        rows.append({
                            "device":device,"gauge":gauge,"budget":budget,
                            "regime":regime,"method":method,
                            "correction_validated":pred["correction_admitted"],
                            "corrected_energy_used":pred["correction_actually_used"],
                            "source_semantic_scope":scope["status"],
                            "physical_source_name_allowed":scope["physical_label_allowed"],
                            "measured_TEST_covariance_shift":scope.get("covariance_shift"),
                            "TRAIN_assay_count":source.train_assay_episodes,
                            "extra_VALID_assay_count":160,
                            "TEST_oracle_used_in_fit":False,
                            "TEST_postfit_hidden_true_energy_metrics":
                                _source_energy_metrics(energy_true,values),
                        })
                    if regime=="healthy_MCAR40":
                        for comp,(first,second) in enumerate(CONTRASTS):
                            a=receipts[f"device{device}_gauge{gauge}_budget{int(budget*100)}_{regime}_{first}_source_energy"]
                            b=receipts[f"device{device}_gauge{gauge}_budget{int(budget*100)}_{regime}_{second}_source_energy"]
                            comparisons.append({
                                "device":device,"gauge":gauge,"budget":budget,
                                "regime":regime,"a":first,"b":second,
                                **_paired_ci_absolute_error(
                                    energy_true,a,b,
                                    seed=20265181+(device%101)*100+(0 if gauge=="A" else 25)+
                                         int(budget*100)*3+comp,
                                    bootstrap=bstrap),
                            })
        truth_gap=float(np.mean(np.abs(receipt_energies[0]-receipt_energies[1])))
        if truth_gap<=.05:
            raise AssertionError("Twinned physical coordinates lost ambiguity")
        proofs.append({
            "device":device,"twin_observed_X_exact":True,
            "twin_outcomes_and_masks_exact":True,
            "average_physical_source1_energy_disagreement":truth_gap,
            "blind_two_world_absolute_error_lower_bound":truth_gap/2.,
            "physical_gauge_not_identifiable_from_X_or_Y":True,
        })
        print(json.dumps({"event":"V51_DEVICE_COMPLETE","device":device,
                          "energy_rows":len([a for a in rows if a["device"]==device]),
                          "gauge_disagreement":truth_gap}),flush=True)
    result={
        "status":("V51_ENERGY_DEBIAS_SMOKE" if smoke else
                  "V51_TRAIN_ONLY_ENERGY_DEBIAS_EXECUTED"),
        "protocol_frozen_before_new_TEST":
            "docs/medical/CONCEPTFAN_V51_FROZEN_TRAIN_ONLY_ENERGY_DEBIAS_PROTOCOL_20261106.md",
        "new_device_seeds":list(devices),
        "n_train":ntr,"n_valid1":nv1,"n_valid2":nv2,"n_test":nte,
        "n_external_source_calibrations":len(calibrations),
        "n_frozen_TEST_source_energy_rows":len(rows),
        "n_preregistered_paired_MAE_CIs":len(comparisons),
        "n_observational_equivalence_receipts":len(proofs),
        "no_change_to_native_model_or_original_K2_risk":True,
        "all_energy_correction_parameters_from_TRAIN_only":True,
        "both_VALID_energy_gates_fixed_before_TEST":True,
        "latent_source_truth_used_ONLY_in_postfit_TEST_metrics":True,
        "physical_semantics_not_guaranteed_outside_attested_instrument":True,
        "not_real_patient_data":True,
    }
    out.mkdir(parents=True,exist_ok=False)
    for name,data in (
        ("STATUS.json",result),
        ("TRAIN_VALID_CALIBRATION_RECEIPTS.json",calibrations),
        ("TEST_ENERGY_ROWS.json",rows),
        ("PAIRED_TEST_MAE_CIS.json",comparisons),
        ("OBSERVATIONAL_TWIN_PROOFS.json",proofs)):
        (out/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))
    np.savez_compressed(out/"FROZEN_TEST_SOURCE_ENERGY_AND_ASSAY_RECEIPTS.npz",**receipts)
    print(json.dumps({"event":"V51_COMPLETED","status":result["status"],
                      "healthy_MCAR40":[x for x in rows
                         if x["regime"]=="healthy_MCAR40" and x["budget"]==.05]}),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args()
    run(args.out_dir,smoke=args.smoke)
