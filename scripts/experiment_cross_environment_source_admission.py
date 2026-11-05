"""Out-of-environment candidate admission for real (oracle-free) native ConceptFAN.

A new synthetic draw tests actual VALID2 sensor-distribution changes, rather
than treating two IID validation patient cohorts as different environments.
Reuses automatic TRAIN-only masked source grouping from PR26, same native K2
known-concept training from observed pair proxies, and frozen additive K2 head.

Known-shift refusal and unseen-shift failure are BOTH primary outcomes.
Train/VALID/TEST roles and sensor interventions are frozen in protocol.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conceptfan_discovery.auto_measured_concept_benchmark import (
    extended_nonlinear_world, masked_histories, raw_measured_features, STRESS,
)
from conceptfan_discovery.auto_measured_concept_proposals import (
    form_train_measured_groups, measured_group_concept,
    group_provenance_fingerprint,
)
from conceptfan_discovery.auto_measured_concept_admission import (
    fixed_unobserved_pca, pca_unknown_values, admitted_test_logit,
    evaluate_two_independent_environments, select_at_most_one_candidate,
)
from conceptfan_discovery.cross_environment_transport import (
    CASES, changed_sensor_environment, paired_test_ap_difference, case_manifest,
)
from conceptfan_discovery.isolated_source_admission_v28 import (
    train_source_coefficient,
)
from scripts.experiment_automatic_measured_concepts import (
    train_masked_measured_known_concepts, candidate_with_support,
    logit_from_head, _oracle_head_for_outcome,
)
from scripts.experiment_nonlinear_anchor_repair_v30 import make_firewalled_k2
from scripts.experiment_frozen_concept_head_refit_v26 import refit_linear_concept_head
from scripts.experiment_trained_conceptfan_omission_v25 import (
    EPOCHS, NN_SEEDS, metrics, predict_native, sha_state,
)

GENERATOR_SEED=20261114
TRAIN_N,VALID_N,TEST_N=1700,1200,900
TRAIN_MASK_SEED,VALID1_MASK_SEED,VALID2_MASK_SEED=20268201,20268202,20268203
TEST_MASK_SEED=20268900
PROPOSAL_SEED=20268204
GATE_SEED=20268210
METHODS=(
    "frozen_K2",
    "AUTO_INVARIANT",
    "AUTO_IID_GATE",
    "MANUAL_PAIR",
    "PCA_UNKNOWN",
    "ALL_GROUPS_REFIT",
    "RAW_TRAIN1700",
    "RAW_ALL2900",
    "ORACLE_TRUE_K2",
    "ORACLE_TRUE_RELEVANT_EXTRA",
)
NN_METHODS=METHODS[:6]
SINGLE_FIT_METHODS=METHODS[6:]
SMOKE_SIZES=(300,300,300)


def _baseline_raw(x,mask,y):
    model=make_pipeline(
        StandardScaler(),LogisticRegression(C=1.,max_iter=1200))
    model.fit(raw_measured_features(x,mask),y)
    return model


def _one_candidate(
    name,source, *,
    train_x,train_m,valid1_x,valid1_m,valid2_iid_x,valid2_shift_x,valid2_m,
    base_train,base_val1,base_val2,
    y_train,y_val1,y_val2,
    seed,
):
    qtrain,_=candidate_with_support(name,source,train_x,train_m)
    gamma=train_source_coefficient(base_train,qtrain,y_train)
    q1,s1=candidate_with_support(name,source,valid1_x,valid1_m)
    qi,si=candidate_with_support(name,source,valid2_iid_x,valid2_m)
    qs,ss=candidate_with_support(name,source,valid2_shift_x,valid2_m)
    common=("VALID1",y_val1,base_val1,q1,s1)
    shifted=evaluate_two_independent_environments(
        name,gamma,common,("SHIFTED_VALID2",y_val2,base_val2,qs,ss),
        train_label_count=len(y_train),
        min_mean_gain=.003,min_support_fraction=.75,
        min_subject_support=.50,bootstrap_reps=399,
        seed=seed)
    iid=evaluate_two_independent_environments(
        name,gamma,common,("UNSHIFTED_VALID2_CONTROL",y_val2,base_val2,qi,si),
        train_label_count=len(y_train),
        min_mean_gain=.003,min_support_fraction=.75,
        min_subject_support=.50,bootstrap_reps=399,
        seed=seed)
    # Only VALID measurements differ in the two gates. Same TRAIN gamma,
    # original fixed K2 head and exact same subject IDs / Y.
    if shifted.gamma_train_only!=iid.gamma_train_only:
        raise AssertionError("Gate comparison illegally trained risk on VALID")
    return {"shifted":shifted,"iid":iid}


def _summarize(rows,case_name,stress,method,nseed):
    matched=[r for r in rows
       if r["case"]==case_name and r["stress"]==stress and r["method"]==method]
    expected=nseed if method in NN_METHODS else 1
    if len(matched)!=expected:
        raise AssertionError(f"Expected {expected} {case_name}/{stress}/{method}; have {len(matched)}")
    return {
        "case":case_name,"stress":stress,"method":method,
        "native_seed_fits":expected if method in NN_METHODS else 0,
        "AUPRC":float(np.mean([r["metrics"]["auprc"] for r in matched])),
        "AUROC":float(np.mean([r["metrics"]["auroc"] for r in matched])),
        "NLL":float(np.mean([r["metrics"]["nll"] for r in matched])),
        "Brier":float(np.mean([r["metrics"]["brier"] for r in matched])),
        "source_use":float(np.mean([r["source_usage"] for r in matched])),
    }


def run(out_dir:Path,*,smoke:bool=False):
    if out_dir.exists():
        raise FileExistsError("Preserve exact execution receipt: results path already exists")
    torch.set_num_threads(2)
    ntr,nval,nte=SMOKE_SIZES if smoke else (TRAIN_N,VALID_N,TEST_N)
    epochs=2 if smoke else EPOCHS
    seeds=(11,) if smoke else NN_SEEDS
    data=extended_nonlinear_world(
        seed=GENERATOR_SEED,train_n=ntr,valid_n=nval,test_n=nte)
    x=data["x"]
    tr,v1,v2,te=(data["train"],data["valid_env1"],
                  data["valid_env2"],data["test"])
    assert len(set(tr)&set(v1))==len(set(v1)&set(v2))==len(set(v2)&set(te))==0
    train_m=masked_histories(x[tr],"MCAR20",seed=TRAIN_MASK_SEED)
    val1_m=masked_histories(x[v1],"MCAR20",seed=VALID1_MASK_SEED)
    val2_m=masked_histories(x[v2],"MCAR20",seed=VALID2_MASK_SEED)
    manifest=form_train_measured_groups(
        x[tr],train_m,known_sensor_indices=(0,1,2,3),
        seed=PROPOSAL_SEED,min_pair_abs_corr=.60,
        min_joint_cells_per_fold=300,max_candidates=5)
    proposed=manifest.candidates
    if len(proposed)<2:
        raise AssertionError("Source proposal failed; must report as failed run")
    pca=fixed_unobserved_pca(x[tr],train_m,manifest.searched_sensors)
    source_map={c.candidate_id:c for c in proposed}
    source_map["MANUAL_PAIR"]=None
    source_map["PCA_UNKNOWN"]=pca
    # independent TEST episodes; each intervention/mask reuses patient IDs
    test_x={}
    test_mask={}
    for case in CASES:
        shifted=changed_sensor_environment(x[te],case.test_device_regime)
        test_x[case.name]=shifted
        for k,stress in enumerate(STRESS):
            test_mask[(case.name,stress)]=masked_histories(
                shifted,stress,seed=TEST_MASK_SEED+k)
    rows,gates,selections,training=[],[],[],[]
    samples_for_ci={}
    baseline_refs={}
    for case in CASES:
        law=case.outcome_law
        y=data["outcomes"][law]
        shifted2=changed_sensor_environment(x[v2],case.validation2_device_regime)
        all_x=np.concatenate([x[tr],x[v1],shifted2],axis=0)
        all_m=np.concatenate([train_m,val1_m,val2_m],axis=0)
        all_y=np.concatenate([y[tr],y[v1],y[v2]],axis=0)
        baseline_refs[case.name]={
            "RAW_TRAIN1700":_baseline_raw(x[tr],train_m,y[tr]),
            "RAW_ALL2900":_baseline_raw(all_x,all_m,all_y),
            "oracle":_oracle_head_for_outcome(data,law,tr),
        }
    for seed in seeds:
        core=make_firewalled_k2(seed)
        receipt=train_masked_measured_known_concepts(
            core,x[tr,:,:4],train_m[:,:,:4],seed=seed,epochs=epochs)
        k2_hash=sha_state(core)
        training.append({"nn_seed":seed,"k2_sha256":k2_hash,**receipt})
        qtr=predict_native(core,x[tr,:,:4],train_m[:,:,:4])["concept_summary"]
        q1=predict_native(core,x[v1,:,:4],val1_m[:,:,:4])["concept_summary"]
        q2=predict_native(core,x[v2,:,:4],val2_m[:,:,:4])["concept_summary"]
        for case_idx,case in enumerate(CASES):
            law=case.outcome_law
            y=data["outcomes"][law]
            head=refit_linear_concept_head(qtr,y[tr])
            frozen_bias=float(head.bias)
            frozen_coeffs=np.asarray(head.weights).copy()
            ztrain=logit_from_head(head,qtr)
            zval1=logit_from_head(head,q1)
            zval2=logit_from_head(head,q2)
            shifted_v2=changed_sensor_environment(
                x[v2],case.validation2_device_regime)
            candidate_gates={}
            for j,source in enumerate(proposed):
                candidate_gates[source.candidate_id]=_one_candidate(
                    source.candidate_id,source,
                    train_x=x[tr],train_m=train_m,
                    valid1_x=x[v1],valid1_m=val1_m,
                    valid2_iid_x=x[v2],valid2_shift_x=shifted_v2,
                    valid2_m=val2_m,
                    base_train=ztrain,base_val1=zval1,base_val2=zval2,
                    y_train=y[tr],y_val1=y[v1],y_val2=y[v2],
                    seed=GATE_SEED+17*j+97*case_idx,
                )
                gates.append({
                    "case":case.name,"nn_seed":seed,
                    "candidate_id":source.candidate_id,
                    "source_family":"AUTO_UNLABELED_TRAIN",
                    "shifted":candidate_gates[source.candidate_id]["shifted"].as_dict(),
                    "iid":candidate_gates[source.candidate_id]["iid"].as_dict(),
                })
            shifted_choice=select_at_most_one_candidate(
                tuple(g["shifted"] for g in candidate_gates.values()))
            iid_choice=select_at_most_one_candidate(
                tuple(g["iid"] for g in candidate_gates.values()))
            extra_gates={}
            for j,method in enumerate(("MANUAL_PAIR","PCA_UNKNOWN")):
                extra_gates[method]=_one_candidate(
                    method,source_map[method],
                    train_x=x[tr],train_m=train_m,
                    valid1_x=x[v1],valid1_m=val1_m,
                    valid2_iid_x=x[v2],valid2_shift_x=shifted_v2,
                    valid2_m=val2_m,
                    base_train=ztrain,base_val1=zval1,base_val2=zval2,
                    y_train=y[tr],y_val1=y[v1],y_val2=y[v2],
                    seed=GATE_SEED+211+17*j+97*case_idx,
                )
                gates.append({
                    "case":case.name,"nn_seed":seed,
                    "candidate_id":method,"source_family":"BASELINE",
                    "shifted":extra_gates[method]["shifted"].as_dict(),
                    "iid":extra_gates[method]["iid"].as_dict(),
                })
            selections.append({
                "case":case.name,"law":law,"nn_seed":seed,
                "validation2_regime":case.validation2_device_regime,
                "test_regime":case.test_device_regime,
                "shifted_valid_admitted_source":shifted_choice,
                "iid_valid_admitted_source":iid_choice,
                "same_VALID2_patient_labels_both_gates":True,
                "native_K2_unchanged_sha256":k2_hash,
            })
            train_group_features=[
                measured_group_concept(group,x[tr],train_m)[0]
                for group in proposed]
            refit_head=refit_linear_concept_head(
                np.column_stack([qtr]+train_group_features),y[tr])
            for stress in STRESS:
                xte=test_x[case.name]
                mte=test_mask[(case.name,stress)]
                # The known concept neural path does NOT see changed sensors
                # 4..11 and never consumes an observed value in missing cells.
                pred=predict_native(core,xte[:,:,:4],mte[:,:,:4])
                base=logit_from_head(head,pred["concept_summary"])
                variants={
                    "frozen_K2":(base.copy(),np.zeros(len(te),bool))
                }
                def _selected(choice,which):
                    if choice is None:
                        return base.copy(),np.zeros(len(te),bool)
                    gate=candidate_gates[choice][which]
                    candidate,support=candidate_with_support(
                        choice,source_map[choice],xte,mte)
                    return admitted_test_logit(
                        base,candidate,support,
                        gamma=gate.gamma_train_only,
                        approved=gate.approved,minimum_fraction=.50)
                variants["AUTO_INVARIANT"]=_selected(shifted_choice,"shifted")
                variants["AUTO_IID_GATE"]=_selected(iid_choice,"iid")
                for method in ("MANUAL_PAIR","PCA_UNKNOWN"):
                    gate=extra_gates[method]["shifted"]
                    candidate,support=candidate_with_support(
                        method,source_map[method],xte,mte)
                    variants[method]=admitted_test_logit(
                        base,candidate,support,
                        gamma=gate.gamma_train_only,
                        approved=gate.approved,
                        minimum_fraction=.50)
                qgroup=np.column_stack([
                    pred["concept_summary"]]+[
                        measured_group_concept(p,xte,mte)[0]
                        for p in proposed])
                variants["ALL_GROUPS_REFIT"]=(
                    logit_from_head(refit_head,qgroup),
                    np.zeros(len(te),bool))
                for method,(z,used) in variants.items():
                    if method!="ALL_GROUPS_REFIT":
                        np.testing.assert_array_equal(
                            z[~used],base[~used])
                    if not np.isfinite(z).all():
                        raise AssertionError("Nonfinite TEST decision")
                    p=expit(z)
                    rows.append({
                        "case":case.name,"stress":stress,"nn_seed":seed,
                        "method":method,
                        "metrics":metrics(y[te],p),
                        "source_usage":float(used.mean()),
                        "extra_source_selected":(
                            shifted_choice if method=="AUTO_INVARIANT"
                            else iid_choice if method=="AUTO_IID_GATE"
                            else method if method in ("MANUAL_PAIR","PCA_UNKNOWN") and used.any()
                            else None),
                        "original_K2_sha256":k2_hash,
                        "original_head_preserved":method!="ALL_GROUPS_REFIT",
                    })
                    if stress=="MCAR20" and method in (
                        "frozen_K2","AUTO_INVARIANT","AUTO_IID_GATE"):
                        samples_for_ci[(case.name,method,seed)]=p.copy()
            if (sha_state(core)!=k2_hash or
                    float(head.bias)!=frozen_bias or
                    not np.array_equal(np.asarray(head.weights),frozen_coeffs)):
                raise AssertionError("Candidate source fitting mutated frozen K2")
        print(json.dumps({
            "event":"ENV_SEED",
            "seed":seed,"k2_sha256":k2_hash,
            "candidate_groups":[list(p.sensor_indices) for p in proposed],
            "decisions":[{"case":s["case"],
                          "shifted":s["shifted_valid_admitted_source"],
                          "iid":s["iid_valid_admitted_source"]}
                         for s in selections if s["nn_seed"]==seed],
        }),flush=True)
    # Ordinary raw baselines and privileged true-state audit are fit once
    # per scenario, NEVER duplicated as separate NN initialization trials.
    for case in CASES:
        y=data["outcomes"][case.outcome_law]
        xte=test_x[case.name]
        refs=baseline_refs[case.name]
        h2,h3,c2,c3=refs["oracle"]
        for stress in STRESS:
            mte=test_mask[(case.name,stress)]
            scores={
                "RAW_TRAIN1700":refs["RAW_TRAIN1700"].predict_proba(
                    raw_measured_features(xte,mte))[:,1],
                "RAW_ALL2900":refs["RAW_ALL2900"].predict_proba(
                    raw_measured_features(xte,mte))[:,1],
                "ORACLE_TRUE_K2":expit(logit_from_head(h2,c2[te])),
                "ORACLE_TRUE_RELEVANT_EXTRA":expit(logit_from_head(h3,c3[te])),
            }
            for method,prob in scores.items():
                rows.append({
                    "case":case.name,"stress":stress,"nn_seed":None,
                    "method":method,"metrics":metrics(y[te],prob),
                    "source_usage":0.,
                    "privileged_source_reference":method.startswith("ORACLE"),
                })
    summary=[
        _summarize(rows,c.name,s,m,len(seeds))
        for c in CASES for s in STRESS for m in METHODS
    ]
    ci=[]
    for caseidx,case in enumerate(CASES):
        y=data["outcomes"][case.outcome_law][te]
        arr=lambda name:np.stack(
            [samples_for_ci[(case.name,name,seed)] for seed in seeds])
        for j,(a,b) in enumerate((
            ("AUTO_INVARIANT","frozen_K2"),
            ("AUTO_INVARIANT","AUTO_IID_GATE"),
        )):
            outcome=paired_test_ap_difference(
                y,arr(a),arr(b),seed=20268300+37*caseidx+j,
                resamples=399)
            ci.append({"case":case.name,"stress":"MCAR20",
                       "method_a":a,"method_b":b,**outcome})
    result={
        "status":"SMOKE_ONLY" if smoke else "SHIFTED_VALID_REAL_ENVIRONMENT_ADMISSION_EVIDENCE",
        "frozen_protocol":"docs/medical/IOI_CROSS_ENVIRONMENT_ADMISSION_FROZEN_PROTOCOL_20261106.md",
        "simulator_family":"extended_nonlinear_V30_same_prior_family",
        "new_generator_seed":GENERATOR_SEED,
        "train_valid_test_disjoint":True,
        "actually_shifted_VALID2_values":True,
        "VALID1_and_VALID2_are_different_measurement_regimes":True,
        "TEST_outcomes_used_for_training_or_admission":False,
        "data_sizes":{"TRAIN":ntr,"VALID1":nval//2,"VALID2":nval//2,"TEST":nte},
        "source_proposals_train_only":True,
        "known_k2_supervised_only_by_masked_measured_proxies":True,
        "fixed_known_sensor_map_supplied":True,
        "no_auto_causal_physical_semantics":True,
        "same_total_2900_labeled_cases_available_to_all_comparators":True,
        "same_total_train_optimizer_budget_across_comparators":False,
        "TEST_subject_bootstrap_CI_conditional_on_frozen_models":True,
        "NN_seed_count":len(seeds),"NN_seeds":list(seeds),
        "NN_training_epochs":epochs,
        "cases":case_manifest(),
        "proposal_manifest":manifest.as_dict(),
        "proposal_sha256":group_provenance_fingerprint(manifest),
        "dataset_sha256":sha256(
            np.ascontiguousarray(x).tobytes()+
            np.ascontiguousarray(data["outcomes"]["physical_extra"]).tobytes()+
            np.ascontiguousarray(data["outcomes"]["artifact_predictive"]).tobytes()
        ).hexdigest(),
        "model_training":training,
        "gate_differences":gates,
        "source_selection":selections,
        "ci95_paired_TEST_AUPRC":ci,
        "rows":rows,"summary":summary,
    }
    out_dir.mkdir(parents=True,exist_ok=False)
    for name,d in (
        ("STATUS.json",result),("GATES.json",gates),
        ("SELECTIONS.json",selections),("SUMMARY.json",summary),
        ("CI95.json",ci),("TRAINING.json",training),
    ):
        (out_dir/name).write_text(
            json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False))
    # Synthetic reference predictions for future inspection, not human data
    # or user-health records. The successful fit and TEST metrics are fixed.
    saved={}
    for case in CASES:
        saved[case.name+"_TEST_labels"]=data["outcomes"][case.outcome_law][te]
        for method in ("frozen_K2","AUTO_INVARIANT","AUTO_IID_GATE"):
            for seed in seeds:
                saved[f"{case.name}_{method}_seed{seed}"]=samples_for_ci[
                    (case.name,method,seed)]
    np.savez_compressed(out_dir/"FROZEN_TEST_PREDICTIONS.npz",**saved)
    primary=[s for s in summary if s["stress"]=="MCAR20"]
    print(json.dumps({
        "event":"CROSS_ENV_DONE","status":result["status"],
        "n_unique_native_K2_fits":len(seeds),
        "n_rows":len(rows),"n_summary":len(summary),
        "n_shifted_gate_rows":len(gates),
        "candidate_groups":[p.sensor_indices for p in proposed],
        "selections":selections,
        "MCAR20":primary,
        "ap_differences":ci,
    }),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--out-dir",required=True,type=Path)
    a=p.parse_args()
    run(a.out_dir,smoke=a.smoke)
