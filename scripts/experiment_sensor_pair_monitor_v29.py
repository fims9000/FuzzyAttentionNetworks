"""V29: train-only sensor agreement corroborates V28 predictive source admission.

A TEST-only one-sensor fault may be detected without TEST outcomes by pair
correlation + paired readout gap, but a both-sensor common-mode sign flip is
observationally indistinguishable under this pair-only monitor.
The original native K2 concept path and its decision head remain unchanged.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from conceptfan_discovery.sensor_pair_provenance_v29 import (
    fit_unlabelled_pair_guard,pair_is_credible,
)
from conceptfan_discovery.isolated_source_admission_v28 import (
    isolated_source_prediction,measured_candidate,train_source_coefficient,
    two_environment_admission,
)
from scripts.experiment_frozen_concept_head_refit_v26 import refit_linear_concept_head
from scripts.experiment_concept_leakage_guard_v27 import fit_concepts_without_outcome_gradients
from scripts.experiment_trained_conceptfan_omission_v25 import (
    EPOCHS,NN_SEEDS,STRESS,
    external_logistic,flatten_measured_inputs,frozen_world,
    make_model,mask_mechanism,metrics,predict_native,sha_state,
)

SEED=20261109
TRAIN_N,VALID_N,TEST_N=1800,1200,900
TRAIN_MASK_SEED,VALID1_SEED,VALID2_SEED=20265101,20265102,20265103
TEST_MASK_BASE=20265900
STRESS_MODES=("MCAR20","MCAR40","MNAR","block8")
SCENARIOS=("stable","single_sensor_sign_flip","single_sensor_bias",
           "both_sensor_sign_flip","nuisance")
METHODS=("raw_logistic","oracle_true_k2","oracle_true_k3",
         "frozen_k2","v28_outcome_validated","v29_sensor_consistent")


def changed_test(x:np.ndarray,scenario:str)->np.ndarray:
    out=np.asarray(x,dtype=np.float32).copy()
    if scenario=="single_sensor_sign_flip":
        out[:,:,5]*=-1.0
    elif scenario=="single_sensor_bias":
        out[:,:,5]+=2.0
    elif scenario=="both_sensor_sign_flip":
        out[:,:,4:6]*=-1.0
    elif scenario not in ("stable","nuisance"):
        raise ValueError(scenario)
    return out


def _head_logit(head,q):
    logits,parts=head(q)
    if np.max(np.abs(logits-(head.bias+parts.sum(axis=1))))>1e-12:
        raise AssertionError("Non-exact concept-only source risk")
    return logits


def evaluate(out_dir:Path,*,smoke:bool=False):
    if out_dir.exists():
        raise FileExistsError("V29 evidence must be preserved, not overwritten")
    torch.set_num_threads(2)
    ntr,nva,nte=(240,240,240) if smoke else (TRAIN_N,VALID_N,TEST_N)
    seeds=(11,) if smoke else NN_SEEDS
    epochs=2 if smoke else EPOCHS
    boots=199 if smoke else 399
    source=frozen_world(seed=SEED,n_train=ntr,n_cal=nva,n_test=nte)
    tr,va,te=source["index_train"],source["index_cal"],source["index_test"]
    va1,va2=va[:nva//2],va[nva//2:]
    if (len(set(tr)&set(va)) or len(set(va)&set(te)) or len(set(tr)&set(te))):
        raise AssertionError("TRAIN, VALID and TEST must have disjoint subjects")
    x,c=source["x"],source["concept_trajectories"]
    mtrain=mask_mechanism(x[tr],"MCAR20",seed=TRAIN_MASK_SEED)
    mval1=mask_mechanism(x[va1],"MCAR20",seed=VALID1_SEED)
    mval2=mask_mechanism(x[va2],"MCAR20",seed=VALID2_SEED)
    pair_policy=fit_unlabelled_pair_guard(x[tr],mtrain)
    q_train_source,source_train_support=measured_candidate(x[tr],mtrain)
    train_quality=pair_is_credible(pair_policy,x[tr],mtrain)
    rows,gates,model_states=[],[],[]
    normal_ref_cache={}
    for seed in seeds:
        core=make_model("native",2,seed)
        fit=fit_concepts_without_outcome_gradients(
            core,x[tr],mtrain,c[tr],seed=seed,epochs=epochs)
        core_hash=sha_state(core)
        model_states.append({"seed":seed,"native_K2_sha256":core_hash,
                             "label_free_first_stage":True,**fit})
        qtrain=predict_native(core,x[tr],mtrain,c[tr])["concept_summary"]
        qva1=predict_native(core,x[va1],mval1,c[va1])["concept_summary"]
        qva2=predict_native(core,x[va2],mval2,c[va2])["concept_summary"]
        qnew1,support1=measured_candidate(x[va1],mval1)
        qnew2,support2=measured_candidate(x[va2],mval2)
        for law in ("predictive","nuisance"):
            y=(source["y_omitted"] if law=="predictive" else source["y_complete"])
            head=refit_linear_concept_head(qtrain,y[tr])
            old_original_weights=np.asarray(head.weights).copy()
            train_logits=_head_logit(head,qtrain)
            gamma=train_source_coefficient(train_logits,q_train_source,y[tr])
            gate=two_environment_admission(
                gamma,
                environments=(
                    ("VALID1",y[va1],_head_logit(head,qva1),qnew1,support1),
                    ("VALID2",y[va2],_head_logit(head,qva2),qnew2,support2),
                ),
                minimum_gain=.002,minimum_supported_fraction=.75,
                per_subject_min_source_support=.5,
                bootstrap_replicates=boots,seed=SEED+seed,
            )
            gates.append({"law":law,"seed":seed,**gate.as_dict(),
                          "added_validation_label_budget":nva})
            oracle2=refit_linear_concept_head(c[tr,:,:2].mean(axis=1),y[tr])
            oracle3=refit_linear_concept_head(c[tr,:,:3].mean(axis=1),y[tr])
            raw=external_logistic(x[tr],mtrain,y[tr])
            for scenario in SCENARIOS:
                if scenario=="nuisance" and law!="nuisance":
                    continue
                if scenario!="nuisance" and law!="predictive":
                    continue
                for stress in STRESS_MODES:
                    transformed=changed_test(x[te],scenario)
                    mask=mask_mechanism(transformed,stress,
                                        seed=TEST_MASK_BASE+STRESS.index(stress))
                    core_pred=predict_native(core,transformed,mask,c[te])
                    if core_pred["max_additive_error"]>2e-5:
                        raise AssertionError("Original native concept graph lost additive structure")
                    q3,coverage=measured_candidate(transformed,mask)
                    base=_head_logit(head,core_pred["concept_summary"])
                    prior=isolated_source_prediction(
                        base,q3,coverage,coefficient=gamma,
                        globally_admitted=gate.admitted,minimum_subject_support=.5)
                    monitored=pair_is_credible(pair_policy,transformed,mask)
                    confidence=prior["used_source"] & monitored["integrity_passed"]
                    monitored_signed=np.where(confidence,
                                               prior["source_signed_contribution"],0.)
                    monitored_logit=base+monitored_signed
                    np.testing.assert_allclose(
                        monitored_logit,base+monitored_signed,rtol=0,atol=0)
                    np.testing.assert_array_equal(monitored_logit[~confidence],
                                                  base[~confidence])
                    candidate_valid=float(prior["used_source"].mean())
                    pair_valid=float(monitored["integrity_passed"].mean())
                    accepted=float(confidence.mean())
                    result_by_method={
                        "frozen_k2":(expit(base),0.),
                        "v28_outcome_validated":(expit(prior["gated_logit"]),candidate_valid),
                        "v29_sensor_consistent":(expit(monitored_logit),accepted),
                        "oracle_true_k2":(expit(_head_logit(oracle2,c[te,:,:2].mean(axis=1))),0.),
                        "oracle_true_k3":(expit(_head_logit(oracle3,c[te,:,:3].mean(axis=1))),0.),
                        "raw_logistic":(raw.predict_proba(flatten_measured_inputs(transformed,mask))[:,1],0.),
                    }
                    for method,(prob,source_usage) in result_by_method.items():
                        rows.append({"scenario":scenario,"seed":seed,
                                     "method":method,"stress":stress,
                                     "metrics":metrics(y[te],prob),
                                     "accepted_source_fraction":source_usage,
                                     "pair_integrity_pass_fraction":pair_valid,
                                     "source_eligible_fraction":float(
                                         prior["source_supported"].mean()),
                                     "original_k2_concept_mse":core_pred["concept_mse"],
                                     "original_model_additive_error":core_pred["max_additive_error"],
                                     "source_decision_head_does_not_change_original_coefficients":
                                         method in ("frozen_k2","v28_outcome_validated",
                                                    "v29_sensor_consistent")})
            if sha_state(core)!=core_hash or not np.array_equal(old_original_weights,np.asarray(head.weights)):
                raise AssertionError("Outcome branch modified frozen concept formation/old K2 weights")
        print(json.dumps({
            "event":"V29_SEED","seed":seed,
            "native_sha":core_hash,
            "gate_results":[{"law":z["law"],"accepted":z["admitted"],
                            "gains":[v["mean_paired_logloss_gain"] for v in z["environments"]]}
                           for z in gates if z["seed"]==seed]},ensure_ascii=False),flush=True)
    # Each seed applied the same raw/oracle baseline to identical subjects.
    # Deduplicate repeated baseline displays rather than pretending 3
    # independent logistic/oracle model fits were performed.
    dedup={}
    for row in rows:
        if row["method"] in ("raw_logistic","oracle_true_k2","oracle_true_k3"):
            key=(row["scenario"],row["stress"],row["method"])
            if key not in dedup:
                dedup[key]=row
    rows=[r for r in rows if r["method"] not in ("raw_logistic","oracle_true_k2","oracle_true_k3")]+list(dedup.values())
    # Observationally stable instrument interpretation is identical to
    # measured physical, not a new cohort or repeated neural fit.
    for item in list(rows):
        if item["scenario"]=="stable":
            rows.append({**item,"scenario":"stable_instrument_interpretation",
                         "identical_measurements_labels_and_models":True})
    summary=[]
    for scenario in SCENARIOS+("stable_instrument_interpretation",):
        for stress in STRESS_MODES:
            for method in METHODS:
                items=[r for r in rows if r["scenario"]==scenario
                       and r["stress"]==stress and r["method"]==method]
                expected=1 if method in ("raw_logistic","oracle_true_k2","oracle_true_k3") else len(seeds)
                if len(items)!=expected:
                    raise AssertionError(f"Wrong number of paired model fits in {scenario}/{stress}/{method}")
                summary.append({
                    "scenario":scenario,"stress":stress,"method":method,
                    "n_unique_models_for_this_method":len(items),
                    "mean_auprc":float(np.mean([q["metrics"]["auprc"] for q in items])),
                    "mean_nll":float(np.mean([q["metrics"]["nll"] for q in items])),
                    "mean_brier":float(np.mean([q["metrics"]["brier"] for q in items])),
                    "mean_source_usage":float(np.mean([q["accepted_source_fraction"] for q in items])),
                    "mean_sensor_pair_credible":float(np.mean([q["pair_integrity_pass_fraction"] for q in items])),
                })
    out={
        "status":"V29_SMOKE_ONLY" if smoke else "V29_NATIVE_UNLABELLED_SENSOR_PAIR_MONITOR",
        "frozen_protocol":"docs/medical/IOI_V29_FROZEN_SENSOR_PROVENANCE_PROTOCOL_20261106.md",
        "generator_seed":SEED,
        "same_prior_simulator_family":True,
        "subject_splits":{"train":ntr,"valid_two_environments":nva,
                          "each_validation_environment":nva//2,"test":nte},
        "nn_seeds":list(seeds),"unique_native_K2_models":len(seeds),
        "train_only_sensor_pair_policy":pair_policy.as_dict(),
        "train_clean_sensor_pair_pass_fraction":float(train_quality["integrity_passed"].mean()),
        "test_outcomes_used_for_training_source_gate_or_sensor_monitor":False,
        "sensor_monitor_uses_outcome_labels":False,
        "global_source_gate_uses_extra_validation_outcomes":True,
        "common_mode_fault_not_detectable_from_pair_agreement":True,
        "synthetic_train_concept_oracle_supervision_used":True,
        "no_automatic_physical_semantic_certification":True,
        "no_external_clinical_validation":True,
        "sample_sha256":sha256(np.ascontiguousarray(x).tobytes()+
                               np.ascontiguousarray(c).tobytes()+
                               np.ascontiguousarray(source["y_omitted"]).tobytes()+
                               np.ascontiguousarray(source["y_complete"]).tobytes()).hexdigest(),
        "gates":gates,"native_states":model_states,"rows":rows,"summary":summary,
    }
    out_dir.mkdir(parents=True,exist_ok=False)
    for name,body in (("STATUS.json",out),("SUMMARY.json",summary),
                      ("GATES.json",gates)):
        (out_dir/name).write_text(json.dumps(body,indent=2,allow_nan=False))
    primary=[r for r in summary if r["stress"]=="MCAR20"
             and r["method"] in ("frozen_k2","v28_outcome_validated",
                                 "v29_sensor_consistent")
             and r["scenario"]!="stable_instrument_interpretation"]
    print(json.dumps({"event":"V29_DONE","status":out["status"],
                      "n_unique_native_models":len(seeds),
                      "n_rows":len(rows),"n_summary":len(summary),
                      "train_sensor_pair_pass_fraction":out["train_clean_sensor_pair_pass_fraction"],
                      "thresholds":pair_policy.as_dict(),
                      "primary":primary},ensure_ascii=False),flush=True)
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    evaluate(args.out_dir,smoke=args.smoke)


if __name__=="__main__":
    main()
