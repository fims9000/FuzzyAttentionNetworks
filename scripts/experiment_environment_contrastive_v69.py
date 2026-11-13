#!/usr/bin/env python3
"""V69 execution — label-free drift-contrastive source, already seen TEST exploratory.

Same original UCI270 physical batch split, same K and supervised TRAIN labels
as V67. No gas identity/ppm influences the new source objective. V69 TEST is
NOT independent after V67/V68 results were inspected.
"""
from __future__ import annotations

import argparse,csv,json,warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conceptfan_discovery.environment_contrastive_source_v69 import TrainOnlyDriftContrastiveBank
from conceptfan_discovery.static_gas_drift_concepts_v67 import (
    FrozenSixGasSourceBank,read_original_ten_batches,
    fix_semantic_alignment_from_valid,audit_fixed_mapping
)
from scripts.experiment_real_six_gas_drift_v67 import (
    exact_concept_head,fit_baselines,measure_multiclass,
    physical_archive_sha256,LABELS
)
from scripts.experiment_drift_semantic_gate_v68 import source_subspace_agreement


def environment_variance_ratio(scores,environment):
    x=np.asarray(scores,dtype=np.float64)
    b=np.asarray(environment)
    grand=x.mean(axis=0)
    w=0.;between=0.
    for z in np.unique(b):
        row=x[b==z]
        mu=row.mean(axis=0)
        w+=float(np.sum((row-mu)**2))
        between+=len(row)*float(np.sum((mu-grand)**2))
    if w<=0:
        return None
    return {
        "train_within_batch_sum_squared_response":w,
        "train_between_batch_sum_squared_mean_shift":between,
        "between_over_within_variance":between/w,
        "gas_class_labels_used_in_ratio":False,
        "ratio_smaller_does_not_certify_chemical_identity":True,
    }


def fit_linear_fair(train_values,train_labels,all_values):
    if train_values.ndim!=2 or all_values.shape[1]!=train_values.shape[1]:
        raise ValueError("Source-head dimensions do not match")
    model=make_pipeline(StandardScaler(),LogisticRegression(
        C=1.,max_iter=1500,solver="lbfgs",random_state=20261967))
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        model.fit(train_values,train_labels)
    if list(model[-1].classes_)!=list(LABELS):
        raise AssertionError("All six original gas labels must be fitted")
    p=model.predict_proba(all_values)
    return p, [str(w.message) for w in got]


def frozen_valid_mapping(bank,x,y,ppm,valid,test,seed):
    levels=bank.source_levels(x)
    mapping,matrix=fix_semantic_alignment_from_valid(levels[valid],y[valid])
    audit=audit_fixed_mapping(mapping,matrix,levels[test],y[test],
                              ppm[test],permutation_seed=seed)
    return mapping,audit


def source_train_batch_loo(x,b,training,original):
    previous=original.source_levels(x[training])
    rows=[]
    for left in range(1,8):
        new_training=training[b[training]!=left]
        fit=TrainOnlyDriftContrastiveBank().fit_unlabeled(
            x[new_training],b[new_training])
        probe=fit.source_levels(x[training])
        rows.append({
            "original_historical_batch_left_out":left,
            "K_train_without_batch":fit.K,
            "source_sha256":fit.sha,
            "n_current_unlabeled_training_rows":len(new_training),
            "score_subspace_overlap_on_original_TRAIN":source_subspace_agreement(
                previous,probe),
            "source_learning_never_receives_gas_identity":True,
        })
    return rows


def run(path:Path,out:Path):
    if out.exists():
        raise FileExistsError("Cannot overwrite post-selection V69 record")
    x,y,ppm,batches,cohorts,schema=read_original_ten_batches(path)
    train=cohorts["TRAIN"];valid=cohorts["VALID"];test=cohorts["TEST"]
    fresh=TrainOnlyDriftContrastiveBank().fit_unlabeled(
        x[train],batches[train])
    old=FrozenSixGasSourceBank().fit_unlabeled(x[train])
    if fresh.K!=old.K:
        raise AssertionError("New direction design must hold rank equal to V67")
    frozen_new=fresh.sha;frozen_old=old.sha
    new_fuzzy=fresh.fuzzy(x);old_fuzzy=old.fuzzy(x)
    new_scores=fresh.source_levels(x)
    old_scores=old.source_levels(x)

    new_risk,new_risk_receipt=exact_concept_head(
        new_fuzzy[train],y[train],new_fuzzy)
    old_risk,old_risk_receipt=exact_concept_head(
        old_fuzzy[train],y[train],old_fuzzy)
    new_linear,linear_warnings=fit_linear_fair(
        new_scores[train],y[train],new_scores)
    original_data={
        "raw128":old.normalize(x),
        "pooled32":old.sensor_views(old.normalize(x)),
        "source_ica_linear":old_scores,
    }
    comparator,warns,majority=fit_baselines(
        x,y,train,original_data)
    candidates={
        "V69_DRIFT_CONTRASTIVE_FUZZY_3K":new_risk,
        "V69_DRIFT_CONTRASTIVE_LINEAR_K":new_linear,
        "V67_GLOBAL_ICA_FUZZY_3K_REPRODUCED":old_risk,
        **comparator,
    }
    expect={
        "V69_DRIFT_CONTRASTIVE_FUZZY_3K",
        "V69_DRIFT_CONTRASTIVE_LINEAR_K",
        "V67_GLOBAL_ICA_FUZZY_3K_REPRODUCED",
        "V67_SOURCE_ICA_LINEAR",
        "POOLED_32_SENSOR_DESCRIPTORS_LOGISTIC",
        "RAW_128_LOGISTIC",
        "RAW_128_EXTRA_TREES",
        "MAJORITY_TRAIN_CLASS",
    }
    if set(candidates)!=expect:
        raise AssertionError("Missing preregistered controls")

    subsets={
        "VALID_BATCH8_9":valid,
        "VALID_BATCH8_ONLY":np.flatnonzero(batches==8),
        "VALID_BATCH9_ONLY":np.flatnonzero(batches==9),
        "TEST_BATCH10_PREVIOUSLY_INSPECTED":test,
    }
    rows=[]
    for method,prob in candidates.items():
        for name,indices in subsets.items():
            result=measure_multiclass(y[indices],prob[indices])
            rows.append({
                "method":method,"cohort":name,
                "n_independent_acquisition_period_batches":
                    2 if name=="VALID_BATCH8_9" else 1,
                "TEST_batch10_already_viewed_before_V69_design":True,
                "method_selected_only_on_prior_V67_failures_not_current_TEST":True,
                **result})
    mapping_new,semantic_new=frozen_valid_mapping(
        fresh,x,y,ppm,valid,test,seed=20261969)
    mapping_old,semantic_old=frozen_valid_mapping(
        old,x,y,ppm,valid,test,seed=20261969)
    loo=source_train_batch_loo(x,batches,train,fresh)
    drift_indices={
        "V69_DRIFT_CONTRASTIVE":environment_variance_ratio(
            new_scores[train],batches[train]),
        "V67_GLOBAL_ICA":environment_variance_ratio(
            old_scores[train],batches[train]),
    }
    if fresh.sha!=frozen_new or old.sha!=frozen_old:
        raise AssertionError("Source changed during supervised downstream heads")
    receipt={
        "status":"V69_POST_V67_EXPLORATORY_DRIFT_CONTRASTIVE_EXECUTED",
        "TEST_batch10_was_already_viewed_before_design":True,
        "claim_independent_external_confirmation":False,
        "protocol":"docs/medical/CONCEPTFAN_V69_FROZEN_DRIFT_CONTRASTIVE_SOURCE_20261113.md",
        "original_zip_sha256":physical_archive_sha256(path),
        "data":"UCI270 six real single-gas exposures with chronological drift",
        "source_data_provenance":schema,
        "unchanged_split_sizes":{k:len(v) for k,v in cohorts.items()},
        "source_fit_uses_historical_batch_ID_but_no_gas_label_or_ppm":True,
        "original_source_K_and_new_source_K":{"old":old.K,"new":fresh.K},
        "old_source":old.public_passport(),
        "new_drift_contrastive_source":fresh.public_passport(),
        "train_only_observable_environment_contrast_ratios":drift_indices,
        "exact_signed_additive_new_source_head":new_risk_receipt,
        "exact_signed_additive_original_source_head":old_risk_receipt,
        "new_source_linear_logistic_fit_warnings":linear_warnings,
        "other_original_method_train_fit_warnings":warns,
        "majority_baseline_train_class":majority,
        "predeclared_all_method_classification_scores":rows,
        "fixed_VALID_postfit_gas_axis_mapping_new":mapping_new,
        "fixed_VALID_postfit_gas_axis_mapping_old":mapping_old,
        "new_source_gas_semantic_audit":semantic_new,
        "old_source_gas_semantic_audit":semantic_old,
        "TRAIN_only_leave_one_batch_out_source_stability":loo,
        "only_one_sensor_apparatus_and_ten_acquisition_periods":True,
        "false_physical_semantic_identity_claimed":False,
        "no_original_IOI_PIERE_camera_ready_file_changed":True,
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"STATUS.json").write_text(json.dumps(
        receipt,indent=2,ensure_ascii=False,allow_nan=False)+"\n",
        encoding="utf8")
    import csv
    with (out/"METRICS.csv").open("w",newline="",encoding="utf8") as f:
        flat=[{key:(json.dumps(val,ensure_ascii=False) if
                    isinstance(val,(list,dict)) else val)
               for key,val in item.items()} for item in rows]
        writer=csv.DictWriter(f,fieldnames=list(flat[0]))
        writer.writeheader();writer.writerows(flat)
    print(json.dumps({
        "event":"V69_POSTHOC_SOURCE_COMPARE_DONE",
        "K":fresh.K,
        "old_vs_new_batch_drift":{
            name:data["between_over_within_variance"] if data else None
            for name,data in drift_indices.items()},
        "test":[{key:row[key] for key in ("method","accuracy","macro_F1",
                                        "multiclass_log_loss")}
                for row in rows if row["cohort"]=="TEST_BATCH10_PREVIOUSLY_INSPECTED"],
        "source_rank_loo":[item["K_train_without_batch"] for item in loo],
        "valid_fixed_semantic_new_test":{
            name:val.get("TEST_oriented_auc_no_rematching")
            for name,val in semantic_new["mapping"].items()},
    },allow_nan=False,ensure_ascii=False),flush=True)
    return receipt


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--uci-zip",required=True,type=Path)
    p.add_argument("--out-dir",required=True,type=Path)
    a=p.parse_args()
    run(a.uci_zip,a.out_dir)


if __name__=="__main__":
    main()
