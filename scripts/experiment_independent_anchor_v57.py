#!/usr/bin/env python3
"""V57 independent external measurement anchor: frozen source-fault falsification.

Test-only faults are applied only after TRAIN-only source and reference alignment,
unsupervised fuzzy encoder training, and supervised risk-head fitting. The
reference comes from an *independent simulator sensor*, not an oracle label
given to any model.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conceptfan_discovery.auto_measured_concept_benchmark import (
    extended_nonlinear_world,masked_histories
)
from conceptfan_discovery.independent_anchor_verification_v57 import IndependentAnchorVerifier
from conceptfan_discovery.native_temporal_fuzzy_v56 import (
    NativeSelfSupervisedConceptFAN, TrainOnlySourceBank
)
from scripts.experiment_autonomous_temporal_v53 import (
    raw_features, metric, matched_nll_bootstrap
)

DRAWS=(20261471,20261473)
WORLDS=("physical_extra","complete_nuisance","artifact_predictive")
FAULTS=("stable","source_common_sign_flip","single_sensor_sign_flip",
        "anchor_only_sign_flip","joint_anchor_and_source_flip",
        "anchor_missing","source_missing")
METHODS=("native_v56","external_anchor_guard_v57",
         "raw_12_logistic","raw_plus_independent_reference_logistic")


def separately_measured_anchor(simulation,*,seed:int,
                               independent_noise_sigma:float=.27):
    """Generate real measured X_ref, not give latent z to any estimator.

    The simulator alone derives the independent reference from its physical
    factor. Source/model/verifier never sees this privileged generator value.
    This is a synthetic provenance assumption, NOT a real medical assay.
    """
    true_c=np.asarray(
        simulation["oracle_concepts_for_posthoc_TEST_audit_only"][:,:,2],dtype=float)
    latent=logit(np.clip(true_c,1e-5,1-1e-5))/1.1
    rng=np.random.default_rng(seed+50001)
    reference=(latent+rng.normal(scale=independent_noise_sigma,size=latent.shape)
               ).astype(np.float32)
    observed=rng.random(reference.shape)>=.20
    return reference,observed


def apply_test_fault(x,m,anchor,am,scenario):
    a=np.asarray(x,dtype=np.float64).copy()
    mask=np.asarray(m,dtype=bool).copy()
    ref=np.asarray(anchor,dtype=np.float64).copy()
    valid=np.asarray(am,dtype=bool).copy()
    if scenario=="source_common_sign_flip":
        a[:,:,4:7]*=-1
    elif scenario=="single_sensor_sign_flip":
        a[:,:,5]*=-1
    elif scenario=="anchor_only_sign_flip":
        ref*=-1
    elif scenario=="joint_anchor_and_source_flip":
        a[:,:,4:7]*=-1
        ref*=-1
    elif scenario=="anchor_missing":
        valid[:]=False
    elif scenario=="source_missing":
        mask[:,:,4:7]=False
    elif scenario!="stable":
        raise ValueError("Unregistered V57 fault mechanism")
    return a,mask,ref,valid


def reference_features(a,m,ref,ref_m):
    base=raw_features(a,m)
    anchor=np.asarray(ref)
    observed=np.asarray(ref_m,dtype=bool)
    if anchor.shape!=observed.shape or len(base)!=len(anchor):
        raise ValueError("Mismatched external reference and raw features")
    safe=np.where(observed,anchor,0.)
    whole=safe.sum(axis=1)/np.maximum(1,observed.sum(axis=1))
    late=safe[:,-6:].sum(axis=1)/np.maximum(1,observed[:,-6:].sum(axis=1))
    return np.column_stack((base,whole,late,observed.mean(axis=1)))


def _binary_head(x,y):
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1200))
    model.fit(x,y)
    return model


def run_draw(draw:int,*,smoke:bool=False):
    train_n,valid_n,test_n=(360,160,160) if smoke else (1400,600,700)
    data=extended_nonlinear_world(
        seed=draw,train_n=train_n,valid_n=valid_n,test_n=test_n)
    x=data["x"]
    ref,ref_m=separately_measured_anchor(data,seed=draw)
    tr,te=data["train"],data["test"]
    if set(tr)&set(te) or set(data["valid_env1"])&set(te):
        raise AssertionError("Train and TEST patients must be disjoint")
    train_m=masked_histories(x[tr],"MCAR20",seed=draw+101)
    test_m=masked_histories(x[te],"MCAR20",seed=draw+102)
    bank=TrainOnlySourceBank(
        seed=draw+131,permutations=25 if smoke else 60,
        bootstraps=5 if smoke else 20
    ).fit_unlabeled(x[tr],train_m)
    native=NativeSelfSupervisedConceptFAN(
        bank,neural_seed=11,epochs=2 if smoke else 8
    ).fit_source_unlabeled(x[tr],train_m)
    # This calibration happens BEFORE any TRAIN Y head is fitted.
    verifier=IndependentAnchorVerifier(
        native,seed=draw+151,
        anchor_provenance="SYNTHETIC_SEPARATELY_CALIBRATED_SENSOR_WITH_INDEPENDENT_NOISE"
    ).fit_from_unlabeled(x[tr],train_m,ref[tr],ref_m[tr])
    if verifier.evidence.matched_concept_index is None:
        raise AssertionError("Primary physical-reference hypothesis did not match a group")
    true_group=set(
        data["evaluation_only_true_groups"]["physical_extra"])
    detected_group=set(verifier.evidence.detected_sensor_group)
    matched_true_group=(detected_group==true_group)
    # This comparison is postfit oracle evaluation ONLY.
    receipt={
        "draw":draw,"N_TRAIN":len(tr),"N_VALID_UNUSED":valid_n,"N_TEST":len(te),
        "source_route":bank.route,"source_K":bank.k,
        "physical_reference_simulated_noise_sigma":.27,
        "physical_reference_independent_mask_probability":.80,
        "independent_device_assumption_not_real_world_proven":True,
        "native_unlabeled_source_sha":native.frozen_network_sha,
        "source_discovery":bank.public_passport(),
        "reference_verification":verifier.passport(),
        "evaluator_only_truth_sensor_group":sorted(true_group),
        "detected_group_matches_evaluator_only_truth":matched_true_group,
        "reference_linkage_fit_before_any_task_labels":True,
        "native_training_losses":native.loss_receipt,
    }
    output=[];pair=[]
    for world in WORLDS:
        y=data["outcomes"][world]
        native.fit_risk_head(x[tr],train_m,y[tr])
        original_sha=native.frozen_network_sha
        train_raw=raw_features(x[tr],train_m)
        basic=_binary_head(train_raw,y[tr])
        extra=_binary_head(reference_features(
            x[tr],train_m,ref[tr],ref_m[tr]),y[tr])
        for scenario in FAULTS:
            xx,mm,rr,rm=apply_test_fault(
                x[te],test_m,ref[te],ref_m[te],scenario)
            augmented=verifier.predict(xx,mm,rr,rm)
            if native.frozen_network_sha!=original_sha:
                raise AssertionError("Reference gate changed trained neural concept definition")
            pp={
                "native_v56":augmented["baseline_probability"],
                "external_anchor_guard_v57":augmented["probability"],
                "raw_12_logistic":basic.predict_proba(raw_features(xx,mm))[:,1],
                "raw_plus_independent_reference_logistic":extra.predict_proba(
                    reference_features(xx,mm,rr,rm))[:,1],
            }
            if augmented["max_additive_error"]>1e-12:
                raise AssertionError("Reference guard lost exact concept-only logit")
            for method,p in pp.items():
                output.append({
                    "draw":draw,"outcome":world,"scenario":scenario,
                    "method":method,"N_TEST":len(te),
                    "reference_matched_to_correct_group_audit_only":matched_true_group,
                    "source_K":bank.k,
                    "reference_rejection_rate":float(
                        augmented["reference_rejected"].mean()) if
                        method=="external_anchor_guard_v57" else None,
                    "reference_no_data_fraction":float(np.mean(
                        augmented["reference_status"]=="NO_REFERENCE")) if
                        method=="external_anchor_guard_v57" else None,
                    "max_additive_error":augmented["max_additive_error"]
                    if method=="external_anchor_guard_v57" else None,
                    **metric(y[te],p)
                })
            if world=="physical_extra":
                pair.append({
                    "draw":draw,"scenario":scenario,
                    "fraction_reference_rejected":float(
                        augmented["reference_rejected"].mean()),
                    "fraction_reference_supported":float(np.mean(
                        augmented["reference_status"]=="SUPPORTED")),
                    "v57_versus_v56_paired_nll":matched_nll_bootstrap(
                        y[te],pp["external_anchor_guard_v57"],pp["native_v56"],
                        seed=draw+301+FAULTS.index(scenario)),
                    "v57_versus_raw_plus_ref_paired_nll":matched_nll_bootstrap(
                        y[te],pp["external_anchor_guard_v57"],
                        pp["raw_plus_independent_reference_logistic"],
                        seed=draw+351+FAULTS.index(scenario)),
                })
    receipt["physical_extra_fault_gate_audit"]=pair
    return receipt,output


def run(out:Path,*,smoke=False):
    if out.exists():
        raise FileExistsError("Never overwrite a pinned source-audit experiment")
    records=[];rows=[]
    for seed in (DRAWS[:1] if smoke else DRAWS):
        receipt,cases=run_draw(seed,smoke=smoke)
        records.append(receipt);rows.extend(cases)
        print(json.dumps({
            "event":"V57_REFERENCE_DRAW","draw":seed,
            "matched_truth_group":receipt[
                "detected_group_matches_evaluator_only_truth"],
            "group":receipt["reference_verification"][
                "reference_fit"]["detected_sensor_group"],
            "primary":[r for r in cases if r["outcome"]=="physical_extra"
                       and r["scenario"]=="source_common_sign_flip"],
        }),flush=True)
    final={
        "status":"SMOKE_ONLY" if smoke else "V57_NEW_SYNTHETIC_INDEPENDENT_SENSOR_EXECUTED",
        "frozen_pre_outcome_protocol":"docs/medical/CONCEPTFAN_V57_FROZEN_INDEPENDENT_ANCHOR_PROTOCOL_20261113.md",
        "source_and_reference_link_fitted_from_train_observations_only":True,
        "reference_channel_excluded_from_concept_encoder_and_native_task_head":True,
        "external_reference_provenance_is_synthetic_assumption_not_medical_proof":True,
        "different_generator_family_confirmation":False,
        "common_mode_joint_fault_is_possible_negative_control":True,
        "draws":records,"rows":rows,
        "no_experiment_or_test_tuning":True,
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"STATUS.json").write_text(
        json.dumps(final,indent=2,ensure_ascii=False,allow_nan=False)+"\n",
        encoding="utf8")
    with (out/"RESULTS.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({
        "event":"V57_DONE","status":final["status"],
        "draws":len(records),"test_rows":len(rows)
    }),flush=True)
    return final


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",required=True,type=Path)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    run(a.out_dir,smoke=a.smoke)


if __name__=="__main__":
    main()
