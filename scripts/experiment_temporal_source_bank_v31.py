"""V31: contrast-based measured-concept proposal vs mean-only source ablation.

New synthetic NON-AR pulse-order simulator; native temporal ConceptFAN K2
concepts remain frozen and trained from observed source-pair proxies ONLY.
The unassigned source group formation is TRAIN-only; candidate risk gamma is
TRAIN-only and selection uses two disjoint 500-subject VALID environments,
one optionally sensor-shifted. Entire 800-subject TEST held out.

Strong baselines include a raw temporal (early/late) logistic with ALL 2500
available labels and an all-group measured CBM with exactly the same number
of labels. Published IB-CBM/Res-SCBM are NOT claimed to be reproduced.
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

from conceptfan_discovery.phase_pulse_benchmark_v31 import (
    SEED,TRAIN_N,VALID_N,TEST_N,CASES,STRESS,
    phase_pulse_world,device_environment,measured_mask,
    temporal_sensor_features,
)
from conceptfan_discovery.temporal_source_candidate_bank_v31 import (
    TemporalCandidateSpec,candidate_bank,measured_temporal_activation,
)
from conceptfan_discovery.auto_measured_concept_proposals import (
    form_train_measured_groups,group_provenance_fingerprint,
)
from conceptfan_discovery.auto_measured_concept_admission import (
    evaluate_two_independent_environments,select_at_most_one_candidate,
    admitted_test_logit,
)
from conceptfan_discovery.isolated_source_admission_v28 import (
    train_source_coefficient,
)
from conceptfan_discovery.cross_environment_transport import (
    paired_test_ap_difference,
)
from scripts.experiment_automatic_measured_concepts import (
    train_masked_measured_known_concepts,logit_from_head,
)
from scripts.experiment_nonlinear_anchor_repair_v30 import make_firewalled_k2
from scripts.experiment_frozen_concept_head_refit_v26 import (
    refit_linear_concept_head,
)
from scripts.experiment_trained_conceptfan_omission_v25 import (
    NN_SEEDS,EPOCHS,predict_native,sha_state,metrics,
)

TRAIN_MASK_SEED=20269301
VALID1_MASK_SEED=20269302
VALID2_MASK_SEED=20269303
TEST_MASK_BASE=20269900
GROUP_FOLD_SEED=20269304
GATE_SEED=20269310
METHODS=(
    "frozen_K2",
    "AUTO_STATIC",
    "AUTO_HALF_CONTRAST",
    "AUTO_BANK",
    "CBM_ALL_TRAIN1500",
    "CBM_ALL_ALL2500",
    "RAW_TEMPORAL_TRAIN1500",
    "RAW_TEMPORAL_ALL2500",
    "ORACLE_TRUE_K2",
    "ORACLE_TRUE_EXTRA",
)
NN_METHODS=METHODS[:6]
SINGLE_FIT_METHODS=METHODS[6:]


def _fit_raw(x,m,y):
    p=make_pipeline(StandardScaler(),LogisticRegression(
        C=1.,max_iter=1200))
    p.fit(temporal_sensor_features(x,m),y)
    return p


def _source_feature_matrix(groups,bank,x,m):
    group_map={g.candidate_id:g for g in groups}
    return np.column_stack([
        measured_temporal_activation(
            group_map[spec.source_group_id],x,m,
            statistic=spec.statistic)[0]
        for spec in bank
    ])


def _selected_prediction(choice,bank_map,group_map,gate_map,
                         test_x,test_mask,old_logit):
    if choice is None:
        return old_logit.copy(),np.zeros(len(old_logit),dtype=bool)
    spec=bank_map[choice]
    group=group_map[spec.source_group_id]
    q,s=measured_temporal_activation(
        group,test_x,test_mask,statistic=spec.statistic)
    gate=gate_map[choice]
    return admitted_test_logit(
        old_logit,q,s,gamma=gate.gamma_train_only,
        approved=gate.approved,minimum_fraction=.50)


def _summarize(records,case,stress,method,n_fits):
    selected=[r for r in records
              if r["case"]==case and r["stress"]==stress
              and r["method"]==method]
    expected=n_fits if method in NN_METHODS else 1
    if len(selected)!=expected:
        raise AssertionError(f"{case}/{stress}/{method}: {len(selected)} not {expected}")
    return {
        "case":case,"stress":stress,"method":method,
        "neural_k2_fits":expected if method in NN_METHODS else 0,
        "mean_auprc":float(np.mean([r["metrics"]["auprc"] for r in selected])),
        "mean_auroc":float(np.mean([r["metrics"]["auroc"] for r in selected])),
        "mean_nll":float(np.mean([r["metrics"]["nll"] for r in selected])),
        "mean_brier":float(np.mean([r["metrics"]["brier"] for r in selected])),
        "mean_source_usage":float(np.mean([r.get("source_usage",0.) for r in selected])),
    }


def execute(out_dir:Path,*,smoke:bool=False):
    if out_dir.exists():
        raise FileExistsError("Refuse to overwrite frozen run evidence")
    torch.set_num_threads(2)
    ntr,nval,nte=(300,300,240) if smoke else (TRAIN_N,VALID_N,TEST_N)
    seeds=(11,) if smoke else NN_SEEDS
    epochs=2 if smoke else EPOCHS
    bootstraps=199 if smoke else 399
    data=phase_pulse_world(seed=SEED,n_train=ntr,n_valid=nval,n_test=nte)
    x=data["x"]
    tr,v1,v2,te=[data[k] for k in ("train","valid1","valid2","test")]
    if any(np.intersect1d(a,b).size for i,a in enumerate((tr,v1,v2,te))
           for b in (tr,v1,v2,te)[i+1:]):
        raise AssertionError("TRAIN VALID TEST subjects overlap")
    mtr=measured_mask(x[tr],"MCAR20",seed=TRAIN_MASK_SEED)
    mv1=measured_mask(x[v1],"MCAR20",seed=VALID1_MASK_SEED)
    mv2=measured_mask(x[v2],"MCAR20",seed=VALID2_MASK_SEED)
    manifest=form_train_measured_groups(
        x[tr],mtr,known_sensor_indices=(0,1,2,3),
        seed=GROUP_FOLD_SEED,min_pair_abs_corr=.60,
        min_joint_cells_per_fold=300,max_candidates=5)
    groups=manifest.candidates
    if len(groups)<2:
        raise AssertionError("Missing multichannel source candidates, no result allowed")
    bank=candidate_bank(groups)
    bank_map={z.candidate_id:z for z in bank}
    group_map={g.candidate_id:g for g in groups}
    kinds={"AUTO_STATIC":"static_mean",
           "AUTO_HALF_CONTRAST":"half_contrast",
           "AUTO_BANK":None}
    test_map={}
    for case_name,law,v2_regime,test_regime in CASES:
        observed=device_environment(x[te],test_regime)
        for j,stress in enumerate(STRESS):
            test_map[(case_name,stress)]=(observed,
                measured_mask(observed,stress,seed=TEST_MASK_BASE+j))
    raw_refs={}
    oracle_refs={}
    for case_name,law,v2_regime,test_regime in CASES:
        y=data["outcomes"][law]
        val2=device_environment(x[v2],v2_regime)
        fullx=np.concatenate([x[tr],x[v1],val2],axis=0)
        fullm=np.concatenate([mtr,mv1,mv2],axis=0)
        fully=np.concatenate([y[tr],y[v1],y[v2]])
        raw_refs[case_name]={
            "RAW_TEMPORAL_TRAIN1500":_fit_raw(x[tr],mtr,y[tr]),
            "RAW_TEMPORAL_ALL2500":_fit_raw(fullx,fullm,fully),
        }
        true_c=data["true_known_CONCEPTS_AUDIT_ONLY"].mean(axis=1)
        extra=data["amplitude_CONCEPTS_AUDIT_ONLY"].get(
            law,np.zeros(len(x)))
        oracle_refs[case_name]={
            "ORACLE_TRUE_K2":(
                refit_linear_concept_head(true_c[tr],y[tr]),true_c),
            "ORACLE_TRUE_EXTRA":(
                refit_linear_concept_head(
                    np.column_stack([true_c[tr],extra[tr]]),y[tr]),
                np.column_stack([true_c,extra])),
        }
    records,gates,selections,model_receipts=[],[],[],[]
    bootstrap_prediction={}
    for seed in seeds:
        native=make_firewalled_k2(seed)
        fit=train_masked_measured_known_concepts(
            native,x[tr,:,:4],mtr[:,:,:4],seed=seed,epochs=epochs)
        native_sha=sha_state(native)
        model_receipts.append({"seed":seed,"native_k2_sha256":native_sha,**fit})
        qtrain=predict_native(native,x[tr,:,:4],mtr[:,:,:4])["concept_summary"]
        qv1=predict_native(native,x[v1,:,:4],mv1[:,:,:4])["concept_summary"]
        qv2=predict_native(native,x[v2,:,:4],mv2[:,:,:4])["concept_summary"]
        qtest={}
        for j,stress in enumerate(STRESS):
            observed,mask=test_map[(CASES[0][0],stress)]
            qtest[stress]=predict_native(
                native,observed[:,:,:4],mask[:,:,:4])["concept_summary"]
        for ci,(case_name,law,v2_regime,test_regime) in enumerate(CASES):
            y=data["outcomes"][law]
            head=refit_linear_concept_head(qtrain,y[tr])
            old_bias=float(head.bias)
            old_weights=np.asarray(head.weights,dtype=np.float64).copy()
            ztrain=logit_from_head(head,qtrain)
            zv1=logit_from_head(head,qv1)
            zv2=logit_from_head(head,qv2)
            modified_val2=device_environment(x[v2],v2_regime)
            candidate_gates={}
            for i,spec in enumerate(bank):
                source=group_map[spec.source_group_id]
                tq,_=measured_temporal_activation(
                    source,x[tr],mtr,statistic=spec.statistic)
                gamma=train_source_coefficient(ztrain,tq,y[tr])
                q1,s1=measured_temporal_activation(
                    source,x[v1],mv1,statistic=spec.statistic)
                q2,s2=measured_temporal_activation(
                    source,modified_val2,mv2,statistic=spec.statistic)
                receipt=evaluate_two_independent_environments(
                    spec.candidate_id,gamma,
                    ("VALID1",y[v1],zv1,q1,s1),
                    ("SHIFTED_VALID2",y[v2],zv2,q2,s2),
                    train_label_count=ntr,
                    min_mean_gain=.003,min_support_fraction=.75,
                    min_subject_support=.50,
                    bootstrap_reps=bootstraps,
                    seed=GATE_SEED+19*i+101*ci,
                )
                candidate_gates[spec.candidate_id]=receipt
                gates.append({"case":case_name,"seed":seed,
                              "candidate_spec":spec.as_dict(),
                              **receipt.as_dict()})
            selections_by_method={}
            for method,statistic in kinds.items():
                chosen=select_at_most_one_candidate(tuple(
                    g for spec in bank if statistic is None
                    or spec.statistic==statistic
                    for g in (candidate_gates[spec.candidate_id],)
                ))
                selections_by_method[method]=chosen
            selections.append({
                "case":case_name,"seed":seed,"source_decisions":selections_by_method,
                "known_K2_hash":native_sha,
                "validation2_device_regime":v2_regime,
            })
            train_src=_source_feature_matrix(groups,bank,x[tr],mtr)
            val1_src=_source_feature_matrix(groups,bank,x[v1],mv1)
            val2_src=_source_feature_matrix(groups,bank,modified_val2,mv2)
            cbm_train=refit_linear_concept_head(
                np.column_stack([qtrain,train_src]),y[tr])
            cbm_all=refit_linear_concept_head(
                np.concatenate([
                    np.column_stack([qtrain,train_src]),
                    np.column_stack([qv1,val1_src]),
                    np.column_stack([qv2,val2_src]),
                ],axis=0),
                np.concatenate([y[tr],y[v1],y[v2]]))
            for stress in STRESS:
                observed,mask=test_map[(case_name,stress)]
                known=qtest[stress]
                old=logit_from_head(head,known)
                variants={"frozen_K2":(old.copy(),np.zeros(nte,bool))}
                for method in kinds:
                    variants[method]=_selected_prediction(
                        selections_by_method[method],bank_map,group_map,
                        candidate_gates,observed,mask,old)
                src=_source_feature_matrix(groups,bank,observed,mask)
                cbm_features=np.column_stack([known,src])
                variants["CBM_ALL_TRAIN1500"]=(
                    logit_from_head(cbm_train,cbm_features),
                    np.zeros(nte,bool))
                variants["CBM_ALL_ALL2500"]=(
                    logit_from_head(cbm_all,cbm_features),
                    np.zeros(nte,bool))
                for method,(logits,used) in variants.items():
                    if not np.isfinite(logits).all():
                        raise AssertionError("Nonfinite temporal source decision")
                    if method not in ("CBM_ALL_TRAIN1500","CBM_ALL_ALL2500"):
                        np.testing.assert_array_equal(logits[~used],old[~used])
                    prob=expit(logits)
                    records.append({
                        "case":case_name,"stress":stress,"method":method,
                        "seed":seed,"metrics":metrics(y[te],prob),
                        "source_usage":float(used.mean()),
                        "selected":selections_by_method.get(method),
                        "original_K2_sha256":native_sha,
                        "K2_coefficients_frozen":(
                            method not in ("CBM_ALL_TRAIN1500","CBM_ALL_ALL2500")),
                    })
                    if stress=="MCAR20" and method in (
                        "frozen_K2","AUTO_STATIC","AUTO_HALF_CONTRAST",
                        "AUTO_BANK"):
                        bootstrap_prediction[(case_name,method,seed)]=prob
            if sha_state(native)!=native_sha or float(head.bias)!=old_bias or (
                not np.array_equal(np.asarray(head.weights),old_weights)):
                raise AssertionError("Source admission altered original K2")
        print(json.dumps({
            "event":"V31_NATIVE_SEED","seed":seed,
            "native_K2_sha256":native_sha,
            "candidate_groups":[list(q.sensor_indices) for q in groups],
            "selections":[{"case":s["case"],"methods":s["source_decisions"]}
                          for s in selections if s["seed"]==seed],
        }),flush=True)
    # Conventional raw models + privileged oracle ceilings each fitted
    # ONCE per case, not counted as three independent NN initialization fits.
    for case_name,law,v2_regime,test_regime in CASES:
        y=data["outcomes"][law]
        for stress in STRESS:
            observed,mask=test_map[(case_name,stress)]
            for method in SINGLE_FIT_METHODS:
                if method.startswith("RAW_TEMPORAL"):
                    prob=raw_refs[case_name][method].predict_proba(
                        temporal_sensor_features(observed,mask))[:,1]
                else:
                    oracle,original=oracle_refs[case_name][method]
                    prob=expit(logit_from_head(oracle,original[te]))
                records.append({
                    "case":case_name,"stress":stress,"method":method,
                    "seed":None,"metrics":metrics(y[te],prob),
                    "source_usage":0.,
                    "privileged_oracle_only":method.startswith("ORACLE"),
                })
    summary=[_summarize(records,c[0],stress,method,len(seeds))
             for c in CASES for stress in STRESS for method in METHODS]
    comparisons=[]
    for ci,(case_name,law,v2_regime,test_regime) in enumerate(CASES):
        y=data["outcomes"][law][te]
        def probs(method):
            return np.stack([bootstrap_prediction[(case_name,method,s)]
                             for s in seeds])
        for j,(a,b) in enumerate((
            ("AUTO_HALF_CONTRAST","AUTO_STATIC"),
            ("AUTO_HALF_CONTRAST","frozen_K2"),
            ("AUTO_BANK","AUTO_STATIC"),
        )):
            comparisons.append({
                "case":case_name,"stress":"MCAR20",
                "method_a":a,"method_b":b,
                **paired_test_ap_difference(
                    y,probs(a),probs(b),
                    seed=20269400+100*ci+j,resamples=bootstraps),
            })
    result={
        "status":"SMOKE_ONLY" if smoke else "V31_TEMPORAL_SOURCE_BANK_EXECUTED",
        "protocol":"docs/medical/IOI_V31_FROZEN_PHASE_PULSE_TEMPORAL_SOURCE_PROTOCOL_20261106.md",
        "distinct_generator_family":data["generator_family"],
        "not_previous_AR_source_generator":True,
        "generator_seed":SEED,
        "split_sizes":{"TRAIN":ntr,"VALID1":nval//2,"VALID2":nval//2,"TEST":nte},
        "NN_seeds":list(seeds),"NN_epochs":epochs,
        "n_unique_native_K2_models":len(seeds),
        "known_concepts_trained_from_observable_proxies_only":True,
        "known_groups_supplied_externally":[[0,1],[2,3]],
        "candidate_definition_and_grouping_train_only":True,
        "target_gradients_to_concept_path":False,
        "oracle_true_latent_concepts_for_training":False,
        "oracle_latent_states_for_audit_only":True,
        "total_label_cases_available_to_all_methods":ntr+nval,
        "compute_matching_all_published_CBMs":False,
        "IB_CBM_and_ResSCBM_original_implementations_executed":False,
        "TEST_outcomes_used_for_training_or_source_gate":False,
        "paired_TEST_bootstrap_conditional_on_frozen_K2_fits":True,
        "source_semantic_physical_certification":False,
        "phase_order_hidden_by_full_history_mean_in_generator":(
            data["pulse_has_zero_true_full_history_mean"]),
        "source_proposals":manifest.as_dict(),
        "source_proposals_sha256":group_provenance_fingerprint(manifest),
        "temporal_bank_spec":[s.as_dict() for s in bank],
        "case_spec":list(CASES),
        "dataset_sha256":sha256(np.ascontiguousarray(x).tobytes()+
            np.ascontiguousarray(data["outcomes"]["physical"]).tobytes()+
            np.ascontiguousarray(data["outcomes"]["artifact"]).tobytes()).hexdigest(),
        "training":model_receipts,
        "gates":gates,"selections":selections,
        "rows":records,"summary":summary,
        "confidence_intervals":comparisons,
    }
    out_dir.mkdir(parents=True,exist_ok=False)
    for filename,body in (
        ("STATUS.json",result),
        ("SUMMARY.json",summary),
        ("GATES.json",gates),
        ("SOURCE_SELECTION.json",selections),
        ("BOOTSTRAP_CI.json",comparisons),
        ("TRAINED_K2.json",model_receipts),
    ):
        (out_dir/filename).write_text(json.dumps(body,indent=2,allow_nan=False))
    saved={}
    for case_name,law,_,_ in CASES:
        saved[case_name+"_TEST_labels"]=data["outcomes"][law][te]
        for method in ("frozen_K2","AUTO_STATIC","AUTO_HALF_CONTRAST","AUTO_BANK"):
            for seed in seeds:
                saved[f"{case_name}_{method}_seed{seed}"]=bootstrap_prediction[
                    (case_name,method,seed)]
    np.savez_compressed(out_dir/"FROZEN_TEST_PREDICTIONS.npz",**saved)
    print(json.dumps({
        "event":"V31_FINISHED","status":result["status"],
        "trained_native_k2_models":len(seeds),
        "n_candidate_groups":len(groups),
        "candidate_groups":[q.sensor_indices for q in groups],
        "n_candidate_features":len(bank),
        "n_test_rows":len(records),"n_aggregates":len(summary),
        "n_gate_receipts":len(gates),
        "selections":selections,
        "primary_MCAR20":[s for s in summary if s["stress"]=="MCAR20"],
        "ap_intervals":comparisons,
    }),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    execute(args.out_dir,smoke=args.smoke)
