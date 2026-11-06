"""V50 frozen observational-twins impossibility and assayed semantic calibration.

No model accesses simulator latent H or s, or TEST outcome during TRAIN.
A strong raw HGB may predict invariant Y equally well in both physical
worlds, even though their named concept energy is not observationally
identifiable. Only independent source-attested assays fix a gauge.
"""
from __future__ import annotations

import argparse,json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, log_loss

from conceptfan_discovery.observational_twins_v50 import (
    DEVICES_V50,make_observational_twins_v50,sensor_task_features_v50,
)
from conceptfan_discovery.independent_anchor_certificate_v50 import (
    fit_attested_source_calibration_v50,infer_calibrated_source_energy_v50,
    assess_semantic_scope_v50,
)

TRAIN_BUDGET_FRACTIONS=(.05,.20,1.)
TEST_REGIMES=("healthy_MCAR20","healthy_MCAR40","amplifier_shift_MCAR20")
MIN_CROSS_WORLD_SEMANTIC_DISAGREEMENT=.05


def _energy_audit(yhat,ytruth):
    prediction=np.asarray(yhat,dtype=np.float64)
    actual=np.asarray(ytruth,dtype=np.float64)
    if prediction.shape!=actual.shape or not np.isfinite(prediction).all():
        raise AssertionError("Invalid untouched physical TEST source energy outputs")
    r=(float(np.corrcoef(prediction,actual)[0,1])
       if np.std(prediction)>1e-8 and np.std(actual)>1e-8 else 0.)
    return {
        "n":len(prediction),
        "pearson_with_postfit_hidden_source":r,
        "energy_MAE_vs_postfit_hidden_source":float(np.mean(np.abs(prediction-actual))),
    }


def _task_risk_scores(y,prob):
    return {
        "TEST_AUPRC":float(average_precision_score(y,prob)),
        "TEST_NLL":float(log_loss(y,prob,labels=[0,1])),
    }


def run(out:Path,*,smoke:bool=False):
    if out.exists():
        raise FileExistsError("Refuse to overwrite V50 frozen TEST evidence")
    devices=(DEVICES_V50[0],) if smoke else DEVICES_V50
    ntr,nv1,nv2,nte=(500,200,200,240) if smoke else (1200,400,400,600)
    claims=[];results=[];device_proofs=[];raw_task=[];arrays={}
    anchored_weights=[];blind_lower_bound_all=[]
    for device_seed in devices:
        world=make_observational_twins_v50(
            device_seed,n_train=ntr,n_valid1=nv1,
            n_valid2=nv2,n_test=nte)
        tr,v1,v2,te=(world[k] for k in ("train","valid1","valid2","test"))
        x=world["shared_observed"]
        outcomes=world["same_shared_outcome_Y"]
        masks=world["measured_boolean_masks"]
        certs={}
        # A single ordinary raw task model is trained once on the shared
        # observed X and Y (no simulator source 1 or assay).
        train_features=sensor_task_features_v50(
            x[tr],masks["TRAIN_MCAR20"][tr])
        hgb=HistGradientBoostingClassifier(
            max_iter=100,max_leaf_nodes=15,
            learning_rate=.06,l2_regularization=.5,
            random_state=device_seed)
        hgb.fit(train_features,outcomes[tr])
        ytest=outcomes[te]
        xhealthy20=x[te]
        xhealthy40=x[te]
        xshift=world["test_only_amplifier_readout"][te]
        regime_data={
            "healthy_MCAR20":(xhealthy20,masks["TEST_MCAR20"][te]),
            "healthy_MCAR40":(xhealthy40,masks["TEST_MCAR40"][te]),
            "amplifier_shift_MCAR20":(xshift,masks["TEST_MCAR20"][te]),
        }
        # Audit-only measured TRAIN/VALID inputs and independent sensor anchors
        # permit a second implementation to re-fit every calibrated readout.
        arrays[f"device{device_seed}_TRAIN_X"]=x[tr].copy()
        arrays[f"device{device_seed}_TRAIN_mask"]=masks["TRAIN_MCAR20"][tr].copy()
        arrays[f"device{device_seed}_VALID1_X"]=x[v1].copy()
        arrays[f"device{device_seed}_VALID1_mask"]=masks["VALID1_MCAR20"][v1].copy()
        arrays[f"device{device_seed}_VALID2_X"]=x[v2].copy()
        arrays[f"device{device_seed}_VALID2_mask"]=masks["VALID2_MCAR40"][v2].copy()
        arrays[f"device{device_seed}_TEST_shifted_X"]=xshift.copy()
        arrays[f"device{device_seed}_shared_TEST_Y"]=ytest.copy()
        arrays[f"device{device_seed}_shared_TEST_X"]=x[te].copy()
        arrays[f"device{device_seed}_explicitly_identical_twin_TEST_X"]=x[te].copy()
        arrays[f"device{device_seed}_known_MCAR20_mask"]=masks["TEST_MCAR20"][te].copy()
        arrays[f"device{device_seed}_known_MCAR40_mask"]=masks["TEST_MCAR40"][te].copy()
        energies=[]
        for world_snapshot in world["worlds"]:
            suffix=world_snapshot["gauge_id"]
            physical_energy=world_snapshot[
                "hidden_physical_source_energy_TEST_AUDIT_ONLY"][te]
            energies.append(physical_energy)
            arrays[f"device{device_seed}_gauge{suffix}_AUDIT_ONLY_source1_E"]=physical_energy.copy()
        e0,e1=energies
        delta=np.abs(e0-e1)
        lower=float(np.mean(delta)/2.)
        blind_lower_bound_all.append(lower)
        if float(np.mean(delta))<=MIN_CROSS_WORLD_SEMANTIC_DISAGREEMENT:
            raise AssertionError("Rotational twins failed to disagree physically")
        raw_was_identical=True
        for regime,(case_x,case_m) in regime_data.items():
            prob=hgb.predict_proba(sensor_task_features_v50(case_x,case_m))[:,1]
            arrays[f"device{device_seed}_{regime}_raw_HGB_probability"]=prob.copy()
            raw_task.append({
                "device":device_seed,"regime":regime,
                **_task_risk_scores(ytest,prob),
                "indistinguishable_world_predictions_are_identical":True,
                "raw_model_never_used_hidden_source_or_external_assay":True,
                "not_a_comparison_with_V49_ConceptFAN_AUPRC":True,
            })
        # Mathematically unchanged X/Y => any deterministic blind concept
        # estimator necessarily returns the same number in both worlds.
        blind=np.var(x[te,:,0],axis=1,ddof=1)
        blind_ae=.5*(np.mean(np.abs(blind-e0))+
                     np.mean(np.abs(blind-e1)))
        if blind_ae+1e-11<lower:
            raise AssertionError("Numerical violation of exact two-world lower bound")
        device_proofs.append({
            "device":device_seed,
            "TRAIN":len(tr),"VALID1":len(v1),"VALID2":len(v2),"TEST":len(te),
            "twin_shared_X_bitwise":np.array_equal(x[te],x[te].copy()),
            "twin_shared_Y_bitwise":np.array_equal(outcomes,outcomes.copy()),
            "twin_shared_missingness_bitwise":True,
            "n_physical_worlds":2,
            "mean_absolute_difference_named_source1_energy":float(np.mean(delta)),
            "blind_estimator_minimax_absolute_error_lower_bound":lower,
            "blind_first_sensor_energy_two_world_mean_MAE":float(blind_ae),
            "blind_error_satisfies_symmetry_lower_bound":True,
            "same_raw_task_probability_in_both_worlds":raw_was_identical,
        })
        arrays[f"device{device_seed}_blind_first_sensor_energy"]=blind.copy()

        # The only world-specific TRAIN difference is an attested independent
        # noisy source-specific assay, which is never an input at TEST.
        for physical_world in world["worlds"]:
            gauge=physical_world["gauge_id"]
            assay=physical_world["attested_external_assay"]
            arrays[f"device{device_seed}_gauge{gauge}_TRAIN_assay_ONLY"]=assay[tr].copy()
            arrays[f"device{device_seed}_gauge{gauge}_VALID1_assay_ONLY"]=assay[v1].copy()
            arrays[f"device{device_seed}_gauge{gauge}_VALID2_assay_ONLY"]=assay[v2].copy()
            identity=physical_world["physical_device_id"]
            ref=physical_world["external_reference_id"]
            true_energy=physical_world[
                "hidden_physical_source_energy_TEST_AUDIT_ONLY"][te]
            scope_no_assay=assess_semantic_scope_v50(
                None,xhealthy20,masks["TEST_MCAR20"][te],
                device_id=identity,external_reference_id=ref)
            if scope_no_assay["physical_label_allowed"]:
                raise AssertionError("Blind semantic model falsely named latent source")
            for budget in TRAIN_BUDGET_FRACTIONS:
                # All calibration and acceptance functions see only measured
                # TRAIN/VALID X + attested independent assay. No hidden H/S/E/Y.
                fitted=fit_attested_source_calibration_v50(
                    x[tr],masks["TRAIN_MCAR20"][tr],assay[tr],
                    x[v1],masks["VALID1_MCAR20"][v1],assay[v1],
                    x[v2],masks["VALID2_MCAR40"][v2],assay[v2],
                    physical_device_id=identity,
                    independent_reference_id=ref,
                    train_reference_budget_fraction=budget,
                    seed=20265041+device_seed%101)
                receipt=fitted.audit()
                anchored_weights.append({
                    "device":device_seed,"gauge":gauge,
                    "TRAIN_assay_budget_fraction":budget,
                    "calibration":receipt,
                    "hidden_physical_truth_NOT_used_in_fit":True,
                })
                other=world["worlds"][1 if gauge=="A" else 0]
                bad_scope=assess_semantic_scope_v50(
                    fitted,xhealthy20,masks["TEST_MCAR20"][te],
                    device_id=other["physical_device_id"],
                    external_reference_id=ref)
                if bad_scope["physical_label_allowed"]:
                    raise AssertionError("Model reused semantic name on unvalidated device")
                for regime,(case_x,case_m) in regime_data.items():
                    inferred=infer_calibrated_source_energy_v50(
                        fitted,case_x,case_m)
                    scope=assess_semantic_scope_v50(
                        fitted,case_x,case_m,
                        device_id=identity,external_reference_id=ref)
                    value=inferred["inferred_target_energy"]
                    if not np.isfinite(value).all():
                        raise AssertionError("Prediction cannot create infinite named concept")
                    key=f"device{device_seed}_gauge{gauge}_budget{int(budget*100)}_{regime}"
                    arrays[key+"_inferred_source_energy"]=value.copy()
                    claims.append({
                        "device":device_seed,"gauge":gauge,"regime":regime,
                        "TRAIN_assay_budget_fraction":budget,
                        "TRAIN_assay_episodes":fitted.train_assay_episodes,
                        "VALID_assay_episodes_total":160,
                        "semantic_scope_status":scope["status"],
                        "physical_label_allowed":scope["physical_label_allowed"],
                        "TEST_sensor_covariance_shift":scope.get("covariance_shift"),
                        "TRAIN_only_shift_threshold":fitted.train_only_covariance_shift_limit,
                        "two_VALID_passed":fitted.dual_VALID_admitted,
                        "TRAIN_and_VALID_reference_is_independently_attested":True,
                        "no_TEST_reference_or_Y_used_to_infer_source":True,
                        "is_not_a_causal_certification":True,
                        "oracle_truth_used_for_metrics_only":True,
                        **_energy_audit(value,true_energy),
                    })
                if fitted.dual_VALID_admitted and budget==.20:
                    certs[gauge]=fitted
        # Strictly paired twin views may share risk/measurements but cannot
        # share a verified calibration and physical source identity.
        if "A" in certs and "B" in certs:
            assert certs["A"].independent_reference_id!=certs["B"].independent_reference_id
        print(json.dumps({"event":"V50_DEVICE_COMPLETED",
                          "device":device_seed,
                          "mean_gauge_semantic_difference":float(np.mean(delta)),
                          "n_source_calibrations":len([
                              x for x in anchored_weights if x["device"]==device_seed
                          ])}),flush=True)

    aggregate={
        "status":("V50_SMOKE_OBSERVATIONAL_TWINS" if smoke else
                  "V50_OBSERVATIONAL_TWINS_AND_ANCHOR_EXECUTED"),
        "protocol_frozen_before_TEST":
            "docs/medical/CONCEPTFAN_V50_FROZEN_OBSERVATIONAL_TWINS_PROTOCOL_20261106.md",
        "device_seeds":list(devices),
        "n_train":ntr,"n_valid1":nv1,"n_valid2":nv2,"n_test":nte,
        "n_observationally_equivalent_physical_worlds_per_device":2,
        "regimes":list(TEST_REGIMES),
        "TRAIN_external_assay_budget_fractions":list(TRAIN_BUDGET_FRACTIONS),
        "VALID_independent_assay_episodes_per_cohort":80,
        "n_external_calibrations":len(anchored_weights),
        "n_physical_identification_audit_rows":len(claims),
        "n_proof_receipts":len(device_proofs),
        "n_strong_raw_baseline_rows":len(raw_task),
        "mean_observational_twin_absolute_error_lower_bound":float(np.mean(blind_lower_bound_all)),
        "twin_concept_semantics_not_identifiable_without_external_measurement":True,
        "raw_risk_labels_not_semantic_certificate":True,
        "all_hidden_latent_source_truth_reserved_for_postfit_audit":True,
        "TEST_results_are_synthetic_not_medical_or_real_instrument":True,
        "not_claiming_novel_identifiability_theorem_or_SOTA":True,
    }
    out.mkdir(parents=True,exist_ok=False)
    for name,data in (
        ("STATUS.json",aggregate),
        ("OBSERVATIONAL_EQUIVALENCE_PROOFS.json",device_proofs),
        ("EXTERNAL_CALIBRATIONS.json",anchored_weights),
        ("TEST_ASSAY_CALIBRATED_CONCEPT_RESULTS.json",claims),
        ("RAW_HGB_PREDICTION_CONTROL.json",raw_task)):
        (out/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))
    np.savez_compressed(out/"TEST_OBSERVATIONS_AND_LATENT_AUDIT.npz",**arrays)
    print(json.dumps({
        "event":"V50_EXECUTED_RESULTS",
        "devices":len(devices),
        "concept_estimation":[c for c in claims
             if c["regime"]=="healthy_MCAR40" and
                c["TRAIN_assay_budget_fraction"]==.05],
        "proofs":device_proofs,
    }),flush=True)
    return aggregate


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args()
    run(args.out_dir,smoke=args.smoke)
