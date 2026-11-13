#!/usr/bin/env python3
"""V67 genuinely new sensor-drift regime, six physical gases, batch-level holdout.

Static source transfer only: 128 extracted response descriptors are NOT 128
real time steps. Fully label-free anonymous ICA/fuzzy source formation first;
then equal TRAIN label-budget classification and VALID-only semantic matching.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,balanced_accuracy_score,f1_score,log_loss,
    roc_auc_score,confusion_matrix
)

from conceptfan_discovery.static_gas_drift_concepts_v67 import (
    GAS_NAMES, FrozenSixGasSourceBank,
    read_original_ten_batches,fix_semantic_alignment_from_valid,
    audit_fixed_mapping
)

LABELS=np.arange(1,7)
METHODS=(
    "V67_SOURCE_MULTIFUZZY_EXACT_ADDITIVE",
    "V67_SOURCE_ICA_LINEAR",
    "POOLED_32_SENSOR_DESCRIPTORS_LOGISTIC",
    "RAW_128_LOGISTIC",
    "RAW_128_EXTRA_TREES",
    "MAJORITY_TRAIN_CLASS",
)


def measure_multiclass(y,p):
    yy=np.asarray(y,int)
    pr=np.asarray(p,float)
    if (pr.shape!=(len(yy),6) or not np.isfinite(pr).all() or
        not np.allclose(pr.sum(axis=1),1.,atol=1e-9)):
        raise AssertionError("Six-class probabilities invalid")
    pred=LABELS[np.argmax(pr,axis=1)]
    return {
        "n_exposure_samples_not_n_independent_batch_environments":len(yy),
        "class_support":[int(np.sum(yy==j)) for j in LABELS],
        "accuracy":float(accuracy_score(yy,pred)),
        "balanced_accuracy":float(balanced_accuracy_score(yy,pred)),
        "macro_F1":float(f1_score(yy,pred,labels=LABELS,
                                    average="macro",zero_division=0)),
        "weighted_F1":float(f1_score(yy,pred,labels=LABELS,
                                       average="weighted",zero_division=0)),
        "multiclass_log_loss":float(log_loss(yy,pr,labels=LABELS)),
        "multiclass_macro_ovr_auc":float(roc_auc_score(
            yy,pr,labels=LABELS,multi_class="ovr",average="macro")),
        "confusion_matrix_gas_1_to_6":confusion_matrix(
            yy,pred,labels=LABELS).tolist(),
    }


def exact_concept_head(train_source,train_y,all_source):
    """Identical label budget, TRAIN-only scaling, exact original-state logits."""
    source=np.asarray(train_source,float)
    future=np.asarray(all_source,float)
    if source.ndim!=2 or future.ndim!=2 or source.shape[1]!=future.shape[1]:
        raise ValueError("TRAIN/test source feature contract broken")
    if source.min()<0 or source.max()>1 or future.min()<0 or future.max()>1:
        raise ValueError("Bounded fuzzy memberships required")
    scale=StandardScaler().fit(source)
    model=LogisticRegression(C=1.,max_iter=1500,solver="lbfgs",
                             random_state=20261967)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(scale.transform(source),train_y)
    if list(model.classes_)!=list(LABELS):
        raise ValueError("All six class heads must be present and ordered")
    beta=model.coef_/scale.scale_[None,:]
    # clf logit=b_raw+x@beta, rewrite in centered inspectable [0,1] states
    bias=model.intercept_ - beta@scale.mean_ + .5*beta.sum(axis=1)
    centered=future-.5
    original_logits=centered@beta.T+bias[None,:]
    recomputed=softmax(original_logits,axis=1)
    actual=model.predict_proba(scale.transform(future))
    maxerror=float(np.max(np.abs(actual-recomputed)))
    if maxerror>1e-9:
        raise AssertionError("Signed class-wise concept explanation is not exact")
    return recomputed,{
        "fitted_only_train_gas_class_labels":len(train_y),
        "no_raw_input_to_risk_head_bypass":True,
        "fuzzy_states_are_bounded_and_exactly_signed_additive":True,
        "per_class_bias_in_original_centered_fuzzy_coordinates":bias.tolist(),
        "per_class_beta_per_fuzzy_state":beta.tolist(),
        "head_train_only_feature_standardization":True,
        "max_abs_probability_additive_reconstruction_error":maxerror,
        "training_sklearn_warnings":[str(w.message) for w in caught],
        "physical_gas_semantic_coordinate_identity_proven":False,
    }


def fit_baselines(raw,gas,idx,features):
    result={}
    issue={}
    models=(
        ("V67_SOURCE_ICA_LINEAR",features["source_ica_linear"],
         make_pipeline(StandardScaler(),LogisticRegression(
             C=1.,max_iter=1500,solver="lbfgs",random_state=20261967))),
        ("POOLED_32_SENSOR_DESCRIPTORS_LOGISTIC",features["pooled32"],
         make_pipeline(StandardScaler(),LogisticRegression(
             C=1.,max_iter=1500,solver="lbfgs",random_state=20261967))),
        ("RAW_128_LOGISTIC",features["raw128"],
         make_pipeline(StandardScaler(),LogisticRegression(
             C=1.,max_iter=1500,solver="lbfgs",random_state=20261967))),
        ("RAW_128_EXTRA_TREES",features["raw128"],
         ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,
                              random_state=20261967,n_jobs=2)),
    )
    for name,all_values,model in models:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(all_values[idx],gas[idx])
        if list(model.classes_)!=list(LABELS):
            raise AssertionError("Missing class in baseline fit")
        result[name]=model.predict_proba(all_values)
        issue[name]=[str(x.message) for x in w]
    majority=int(LABELS[np.argmax(np.bincount(gas[idx],minlength=7)[1:])])
    allproba=np.zeros((len(gas),6),dtype=float)
    allproba[:,majority-1]=1.
    result["MAJORITY_TRAIN_CLASS"]=allproba
    return result,issue,majority


def drift_diagnostic(x,batch,idx,levels,fuzzy,bank):
    reference_raw=bank.normalize(x[idx])
    reference_concept=fuzzy[idx]
    reports=[]
    for b in range(1,11):
        selected=batch==b
        standardized=bank.normalize(x[selected])
        source=fuzzy[selected]
        reports.append({
            "batch_id":b,
            "n_exposures":int(np.sum(selected)),
            "median_absolute_128_feature_shift_from_train":
                float(np.median(np.abs(
                    np.median(standardized,axis=0)-
                    np.median(reference_raw,axis=0)))),
            "mean_abs_fuzzy_membership_shift_from_train":
                float(np.mean(np.abs(
                    source.mean(axis=0)-reference_concept.mean(axis=0)))),
            "batch_drift_assessed_without_any_gas_labels":True,
        })
    return reports


def physical_archive_sha256(path):
    h=sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(1024*1024),b""):
            h.update(part)
    return h.hexdigest()


def run(archive:Path,out:Path):
    if out.exists():
        raise FileExistsError("Never overwrite executed V67 independent final TEST")
    x,y,concentrations,batches,cohorts,schema=read_original_ten_batches(archive)
    train=cohorts["TRAIN"];valid=cohorts["VALID"];test=cohorts["TEST"]
    bank=FrozenSixGasSourceBank().fit_unlabeled(x[train])
    source_manifest_sha=bank.sha
    source_features={
        "raw128":bank.normalize(x),
    }
    source_features["pooled32"]=bank.sensor_views(source_features["raw128"])
    source_features["source_ica_linear"]=bank.source_levels(x)
    source_features["fuzzy"]=bank.fuzzy(x)
    phead,headreceipt=exact_concept_head(
        source_features["fuzzy"][train],y[train],source_features["fuzzy"])
    predictive={
        "V67_SOURCE_MULTIFUZZY_EXACT_ADDITIVE":phead}
    base,warnings_receipt,majority=fit_baselines(x,y,train,source_features)
    predictive.update(base)
    if set(predictive)!=set(METHODS):
        raise AssertionError("Incomplete preregistered class-model comparisons")
    semantic_valid=source_features["source_ica_linear"][valid]
    alignment,valid_matrix=fix_semantic_alignment_from_valid(semantic_valid,y[valid])
    semantic=audit_fixed_mapping(
        alignment,valid_matrix,
        source_features["source_ica_linear"][test],
        y[test],concentrations[test],permutation_seed=20261967)
    subsets={
        "VALID_BATCH8_9":valid,
        "VALID_BATCH8_ONLY":np.flatnonzero(batches==8),
        "VALID_BATCH9_ONLY":np.flatnonzero(batches==9),
        "TEST_BATCH10_FINAL":test,
    }
    rows=[]
    for name in METHODS:
        for cohort,which in subsets.items():
            met=measure_multiclass(y[which],predictive[name][which])
            rows.append({
                "method":name,"evaluation_cohort":cohort,
                "n_independent_drift_batches":2 if cohort=="VALID_BATCH8_9" else 1,
                "model_selection_never_sees_test_batch10":True,
                "unseen_data_family_vs_V53_V66":True,
                **met})
    drift=drift_diagnostic(
        x,batches,train,source_features["source_ica_linear"],
        source_features["fuzzy"],bank)
    if bank.sha!=source_manifest_sha:
        raise AssertionError("Reading supervised labels mutated source definition")
    truth={
        "status":"V67_NEW_UCI270_TEMPORAL_BATCH_TRANSFER_EXECUTED",
        "official_data":"UCI Gas Sensor Array Drift at Different Concentrations, ID270",
        "unseen_data_family_V67_vs_V60_V66":True,
        "source_zip_sha256":physical_archive_sha256(archive),
        "licensing":"ACADEMIC_RESEARCH_ONLY",
        "original_sample_provenance":schema,
        "split_sample_sizes":{k:len(v) for k,v in cohorts.items()},
        "n_independent_hardware_installations":1,
        "n_temporal_drift_batch_periods":10,
        "eight_descriptors_per_sensor_are_NOT_actual_sequence_timesteps":True,
        "V65_native_temporal_recurrence_not_claimed_tested":True,
        "static_label_free_source_bank":bank.public_passport(),
        "concept_head_exact_additive_receipt":headreceipt,
        "baseline_fit_warnings":warnings_receipt,
        "baseline_majority_train_gas_class":majority,
        "fixed_after_VALID_postfit_semantic_axis_assignment":alignment,
        "gas_specific_semantic_audit":semantic,
        "all_precommitted_multiclass_scores_and_confusions":rows,
        "unlabeled_sensor_batch_drift_diagnostics":drift,
        "concentration_setpoints_and_gas_id_excluded_from_source_fit":True,
        "six_pure_gases_not_simultaneous_multisource_exposures":True,
        "semantic_label_matching_IS_VALID_supervised_not_autonomous_discovery":True,
        "chemical_source_identifiability_or_unique_physics_proven":False,
        "paper_claims_original_IOI_PIERE_unmodified":True,
        "protocol":"docs/medical/CONCEPTFAN_V67_FROZEN_UNSEEN_SIX_GAS_DRIFT_PROTOCOL_20261113.md",
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"STATUS.json").write_text(
        json.dumps(truth,ensure_ascii=False,allow_nan=False,indent=2)+"\n",
        encoding="utf8")
    rowsflat=[]
    for item in rows:
        rowsflat.append({k:(json.dumps(v,ensure_ascii=False)
                            if isinstance(v,(list,dict)) else v)
                         for k,v in item.items()})
    with (out/"METRICS.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rowsflat[0]))
        writer.writeheader();writer.writerows(rowsflat)
    print(json.dumps({
        "event":"V67_REAL_UNSEEN_SIX_GAS_B10_DONE",
        "source_K":bank.K,
        "source_ICA_converged":bank.receipt["ica_converged"],
        "train_valid_test":{k:len(v) for k,v in cohorts.items()},
        "all_test":[{k:v for k,v in row.items()
                     if k in ("method","macro_F1","accuracy",
                              "balanced_accuracy","multiclass_log_loss")}
                    for row in rows if row["evaluation_cohort"]=="TEST_BATCH10_FINAL"],
        "semantic_fixed_mapping_test":{
            gas:{k:v for k,v in item.items()
                 if k in ("status","VALID_oriented_auc","TEST_oriented_auc_no_rematching")}
            for gas,item in semantic["mapping"].items()},
    },ensure_ascii=False,allow_nan=False),flush=True)
    return truth


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--uci-zip",required=True,type=Path)
    parser.add_argument("--out-dir",required=True,type=Path)
    args=parser.parse_args()
    run(args.uci_zip,args.out_dir)


if __name__=="__main__":
    main()
