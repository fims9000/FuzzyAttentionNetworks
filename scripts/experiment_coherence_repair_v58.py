#!/usr/bin/env python3
"""V58 frozen new-draw sensor-fusion experiment with concept-level repair.

No TEST observations/labels are used in source, anchor map, coherence
threshold, native fuzzy encoder training, or risk-head fit. Extra independent
measurement is a SYNTHETIC provenance assumption, not clinical proof.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from conceptfan_discovery.auto_measured_concept_benchmark import (
    extended_nonlinear_world,masked_histories
)
from conceptfan_discovery.native_temporal_fuzzy_v56 import (
    NativeSelfSupervisedConceptFAN,TrainOnlySourceBank
)
from conceptfan_discovery.independent_anchor_verification_v57 import (
    IndependentAnchorVerifier
)
from conceptfan_discovery.coherence_aware_anchor_repair_v58 import (
    CoherenceAwareAnchorRepair
)
from scripts.experiment_independent_anchor_v57 import (
    separately_measured_anchor,reference_features,_binary_head
)
from scripts.experiment_autonomous_temporal_v53 import (
    metric,raw_features,matched_nll_bootstrap
)

DRAWS=(20261491,20261493)
WORLDS=("physical_extra","complete_nuisance","artifact_predictive")
SCENARIOS=(
    "stable","single_sensor_sign_flip","two_sensor_sign_flip",
    "source_common_sign_flip","anchor_only_sign_flip",
    "joint_anchor_and_source_flip","anchor_missing","source_missing",
)
METHODS=(
    "native_v56","v57_reference_zeroing",
    "v58_coherence_aware_repair","raw_12_logistic",
    "raw_plus_reference_logistic",
)


def test_fault(x,mask,anchor,am,scenario):
    a=np.asarray(x,dtype=float).copy()
    m=np.asarray(mask,dtype=bool).copy()
    r=np.asarray(anchor,dtype=float).copy()
    rm=np.asarray(am,dtype=bool).copy()
    if scenario=="single_sensor_sign_flip":
        a[:,:,5]*=-1
    elif scenario=="two_sensor_sign_flip":
        a[:,:,4:6]*=-1
    elif scenario=="source_common_sign_flip":
        a[:,:,4:7]*=-1
    elif scenario=="anchor_only_sign_flip":
        r*=-1
    elif scenario=="joint_anchor_and_source_flip":
        a[:,:,4:7]*=-1
        r*=-1
    elif scenario=="anchor_missing":
        rm[:]=False
    elif scenario=="source_missing":
        m[:,:,4:7]=False
    elif scenario!="stable":
        raise ValueError("Unknown preregistered test condition")
    return a,m,r,rm


def execute_draw(draw:int,*,smoke:bool=False):
    ntrain,nvalid,ntest=(360,160,160) if smoke else (1400,600,700)
    data=extended_nonlinear_world(
        seed=draw,train_n=ntrain,valid_n=nvalid,test_n=ntest)
    x=data["x"]
    anchor,anchor_m=separately_measured_anchor(data,seed=draw)
    tr,te=data["train"],data["test"]
    train_m=masked_histories(x[tr],"MCAR20",seed=draw+101)
    test_m=masked_histories(x[te],"MCAR20",seed=draw+102)
    source=TrainOnlySourceBank(
        seed=draw+131,permutations=25 if smoke else 60,
        bootstraps=5 if smoke else 20).fit_unlabeled(x[tr],train_m)
    net=NativeSelfSupervisedConceptFAN(
        source,neural_seed=11,epochs=2 if smoke else 8
    ).fit_source_unlabeled(x[tr],train_m)
    verifier=IndependentAnchorVerifier(
        net,seed=draw+151,
        anchor_provenance="SYNTHETIC_INDEPENDENT_CALIBRATED_REFERENCE_SENSOR"
    ).fit_from_unlabeled(x[tr],train_m,anchor[tr],anchor_m[tr])
    repaired=CoherenceAwareAnchorRepair(verifier).fit_from_unlabeled(
        x[tr],train_m)
    # Every reference mapping and coherence threshold has been calibrated
    # without task outcomes, before ANY risk head fit.
    assert source.sha==net.bank.sha
    e=verifier.evidence
    if e.matched_concept_index is None:
        raise AssertionError("The frozen independent anchor could not be calibrated")
    truth_group=set(data["evaluation_only_true_groups"]["physical_extra"])
    detected_group=set(e.detected_sensor_group)
    source_truth_match=(truth_group==detected_group)
    rows=[];paired=[];risk_receipts={}
    for world in WORLDS:
        y=data["outcomes"][world]
        net.fit_risk_head(x[tr],train_m,y[tr])
        trained_sha=net.frozen_network_sha
        beta_before=net.beta.copy()
        raw=_binary_head(raw_features(x[tr],train_m),y[tr])
        raw_plus=_binary_head(reference_features(
            x[tr],train_m,anchor[tr],anchor_m[tr]),y[tr])
        risk_receipts[world]={
            "train_outcomes":len(tr),
            "frozen_source_sha256":trained_sha,
            "risk_head_beta":beta_before.tolist(),
            "anchor_mapping_never_touched_train_outcome":True
        }
        for scenario in SCENARIOS:
            xx,mm,rr,rm=test_fault(x[te],test_m,anchor[te],anchor_m[te],scenario)
            old=net.predict(xx,mm)
            excluded=verifier.predict(xx,mm,rr,rm)
            corrected=repaired.predict(xx,mm,rr,rm)
            if net.frozen_network_sha!=trained_sha:
                raise AssertionError("Native fuzzy encoder mutated under repair")
            np.testing.assert_array_equal(net.beta,beta_before)
            if corrected["max_additive_logit_error"]>1e-12:
                raise AssertionError("Concept repair lost exact signed decision")
            predictions={
                "native_v56":old["probability"],
                "v57_reference_zeroing":excluded["probability"],
                "v58_coherence_aware_repair":corrected["probability"],
                "raw_12_logistic":raw.predict_proba(raw_features(xx,mm))[:,1],
                "raw_plus_reference_logistic":raw_plus.predict_proba(
                    reference_features(xx,mm,rr,rm))[:,1],
            }
            statuses=dict(Counter(corrected["repair_reason"]))
            for method,p in predictions.items():
                rows.append({
                    "draw":draw,"outcome":world,"scenario":scenario,
                    "method":method,"N_TEST":len(te),
                    "source_K":source.k,
                    "source_truth_group_match_audit_only":source_truth_match,
                    "repair_rate":float(corrected["repaired_source"].mean())
                                   if method=="v58_coherence_aware_repair" else None,
                    "reference_zeroing_rate":float(excluded["reference_rejected"].mean())
                                   if method=="v57_reference_zeroing" else None,
                    "coherence_preserved_original_fraction":float(np.mean(
                        corrected["repair_reason"]=="SENSOR_CONFLICT_KEPT_ROBUST_SOURCE"))
                        if method=="v58_coherence_aware_repair" else None,
                    "source_repair_status_counts":(
                        json.dumps(statuses,sort_keys=True)
                        if method=="v58_coherence_aware_repair" else None),
                    "max_additive_error":corrected["max_additive_logit_error"]
                        if method=="v58_coherence_aware_repair" else None,
                    **metric(y[te],p)
                })
            if world=="physical_extra":
                paired.append({
                    "draw":draw,"scenario":scenario,
                    "reference_flag_rate":float(excluded["reference_rejected"].mean()),
                    "selective_repair_rate":float(corrected["repaired_source"].mean()),
                    "reason_counts":statuses,
                    "v58_vs_v56_nll_gain":matched_nll_bootstrap(
                        y[te],predictions["v58_coherence_aware_repair"],
                        predictions["native_v56"],
                        seed=draw+310+SCENARIOS.index(scenario)),
                    "v58_vs_v57_nll_gain":matched_nll_bootstrap(
                        y[te],predictions["v58_coherence_aware_repair"],
                        predictions["v57_reference_zeroing"],
                        seed=draw+350+SCENARIOS.index(scenario)),
                    "v58_vs_raw_plus_anchor_nll_gain":matched_nll_bootstrap(
                        y[te],predictions["v58_coherence_aware_repair"],
                        predictions["raw_plus_reference_logistic"],
                        seed=draw+390+SCENARIOS.index(scenario)),
                })
    receipt={
        "draw":draw,"N_TRAIN":len(tr),"N_VALID_UNUSED":nvalid,"N_TEST":len(te),
        "source_router":source.route,"source_K":source.k,
        "source_discovery_sha":source.sha,"unlabeled_native_sha":net.frozen_network_sha,
        "source_truth_group_match_evaluator_only":source_truth_match,
        "verified_sensor_group":list(e.detected_sensor_group),
        "anchor_calibration":verifier.passport(),
        "coherence_calibration":repaired.passport(),
        "unlabeled_source_losses":net.loss_receipt,
        "risk_training_receipts":risk_receipts,
        "paired_test_subject_NLL_descriptive":paired,
        "task_head_never_saw_reference_or_original_raw_bypass":True,
        "reference_provenance_is_synthetic_independent_noise_assumption":True,
    }
    return receipt,rows


def run(output:Path,*,smoke:bool=False):
    if output.exists():
        raise FileExistsError("Do not overwrite existing pinned V58 evidence")
    receipts=[];results=[]
    for draw in (DRAWS[:1] if smoke else DRAWS):
        receipt,rows=execute_draw(draw,smoke=smoke)
        receipts.append(receipt);results.extend(rows)
        print(json.dumps({
            "event":"V58_NEW_DRAW","draw":draw,
            "true_group_match":receipt["source_truth_group_match_evaluator_only"],
            "primary":[r for r in rows if r["outcome"]=="physical_extra"
                       and r["scenario"]=="source_common_sign_flip"],
        }),flush=True)
    report={
        "status":"SMOKE_ONLY" if smoke else "V58_NEW_EXPLORATORY_COHERENCE_REPAIR_EXECUTED",
        "frozen_protocol":"docs/medical/CONCEPTFAN_V58_FROZEN_COHERENT_REPAIR_PROTOCOL_20261113.md",
        "unlabeled_source_and_coherence_reference_training_only":True,
        "reference_affects_only_one_concept_input_not_task_head":True,
        "v57_test_previously_viewed_and_not_used_in_v58_fits":True,
        "no_test_hyperparameter_tuning":True,
        "assumed_externally_verified_independent_sensor":True,
        "joint_anchor_primary_sensor_failures_not_guaranteed_detectable":True,
        "draws":receipts,"rows":results,
        "physiological_semantics_proven":False,
    }
    output.mkdir(parents=True,exist_ok=False)
    (output/"STATUS.json").write_text(json.dumps(
        report,indent=2,allow_nan=False,ensure_ascii=False)+"\n",encoding="utf8")
    with (output/"RESULTS.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader();writer.writerows(results)
    print(json.dumps({
        "event":"V58_DONE","status":report["status"],
        "N_draws":len(receipts),"N_test_rows":len(results)},flush=True))
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    run(a.out_dir,smoke=a.smoke)

if __name__=="__main__":
    main()
