#!/usr/bin/env python3
"""V68 RETROSPECTIVE semantic gate across separate VALID batches 8 and 9.

The V67 batch10 TEST was already inspected. This is therefore not a new
sealed test, regardless of whether the gate succeeds or fails. Fixed TRAIN
source geometry and VALID-only label matching; no batch10 adaptation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from conceptfan_discovery.static_gas_drift_concepts_v67 import (
    FrozenSixGasSourceBank,GAS_NAMES, read_original_ten_batches,
    fix_semantic_alignment_from_valid
)
from scripts.experiment_real_six_gas_drift_v67 import physical_archive_sha256

MIN_GAS_SUPPORT_PER_VALID_BATCH=20
MIN_VALID_ORIENTED_AUC=.80
MAX_VALID_BATCH_AUC_DIFFERENCE=.15
DESCRIPTIVE_TEST_AUC_REFERENCE=.75
RNG_SEED=20261968


def source_subspace_agreement(baseline_scores,alternative_scores):
    """Canonical squared correlations; invariant to ICA axis sign/permutation."""
    x=np.asarray(baseline_scores,float)
    y=np.asarray(alternative_scores,float)
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y):
        raise ValueError("Full same TRAIN probe required for rank comparison")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite source subspace")
    q1=np.linalg.qr(x-x.mean(axis=0),mode="reduced")[0]
    q2=np.linalg.qr(y-y.mean(axis=0),mode="reduced")[0]
    singular=np.linalg.svd(q1.T@q2,compute_uv=False)
    return {
        "reference_rank":x.shape[1],
        "left_one_batch_out_rank":y.shape[1],
        "all_squared_canonical_correlations":(singular**2).tolist(),
        "mean_squared_canonical_correlation_over_common_rank":float(
            np.mean(singular**2)),
        "comparison_invariant_to_source_sign_and_permutation":True,
        "overlap_proves_chemical_identity":False,
    }


def train_only_leave_one_batch_out(raw,batch,train_idx,original):
    source_scores=original.source_levels(raw[train_idx])
    results=[]
    for omitted in range(1,8):
        selected=train_idx[batch[train_idx]!=omitted]
        if len(selected)<1000:
            raise ValueError("Not enough physical TRAIN exposures after exclusion")
        source=FrozenSixGasSourceBank().fit_unlabeled(raw[selected])
        scores=source.source_levels(raw[train_idx])
        results.append({
            "omitted_historical_train_batch":omitted,
            "n_unlabeled_train_exposures":len(selected),
            "source_rank_K":source.K,
            "ICA_converged":source.receipt["ica_converged"],
            "unlabeled_source_sha256":source.sha,
            "original_train_batch_gas_labels_or_ppm_used":False,
            "probe_is_original_train_sensor_measurements_only":True,
            "source_subspace_agreement":source_subspace_agreement(
                source_scores,scores)
        })
    return results


def auc_direction(reference,positive,sign:int):
    if sign not in (-1,1):
        raise ValueError("Frozen axis sign must be -1 or 1")
    return float(roc_auc_score(np.asarray(positive,int),
                               sign*np.asarray(reference,float)))


def predeclared_early_valid_gate(mapping,raw_scores,gas,batches,
                                *,rank_is_stable:bool):
    if raw_scores.shape[1]<1:
        raise ValueError("Need original frozen unlabeled source directions")
    results={}
    rng=np.random.default_rng(RNG_SEED)
    testidx=np.flatnonzero(batches==10)
    shuffled=rng.permutation(gas[testidx])
    for cls in range(1,7):
        gas_name=GAS_NAMES[cls-1]
        counts={str(b):int(np.sum((batches==b)&(gas==cls)))
                for b in (8,9,10)}
        if str(cls) not in mapping:
            results[str(cls)]={
                "gas_name":gas_name,
                "status":"ABSTAIN_UNMATCHED_SOURCE_K_SMALLER_THAN_CLASS_COUNT",
                "VALID8_VALID9_TEST10_counts":counts,
                "K_source_axes_not_forced_to_six":True}
            continue
        selected=mapping[str(cls)]
        k=selected["anonymous_ica_source_axis"]
        sign=selected["valid_direction"]
        auc={}
        for b in (8,9,10):
            idx=batches==b
            auc[str(b)]=auc_direction(raw_scores[idx,k],gas[idx]==cls,sign)
        shuffled_auc=auc_direction(
            raw_scores[testidx,k],shuffled==cls,sign)
        support=(counts["8"]>=MIN_GAS_SUPPORT_PER_VALID_BATCH and
                 counts["9"]>=MIN_GAS_SUPPORT_PER_VALID_BATCH)
        performance=(auc["8"]>=MIN_VALID_ORIENTED_AUC and
                     auc["9"]>=MIN_VALID_ORIENTED_AUC)
        stability=abs(auc["8"]-auc["9"])<=MAX_VALID_BATCH_AUC_DIFFERENCE
        admitted=rank_is_stable and support and performance and stability
        failures=[]
        if not rank_is_stable:failures.append("RANK_VARIES_ACROSS_TRAIN_HISTORY")
        if not support:failures.append("INSUFFICIENT_VALID_GAS_REFERENCES")
        if not performance:failures.append("LOW_VALID_GAS_ASSOCIATION")
        if not stability:failures.append("VALID_BATCH8_TO_BATCH9_SIGN_OR_AUC_DRIFT")
        results[str(cls)]={
            "gas_name":gas_name,
            "anonymous_ICA_source_axis":k,
            "fixed_sign_chosen_from_batch8_only":sign,
            "status":("EARLY_VALID_ASSOCIATION_NOT_PHYSICAL_IDENTITY"
                      if admitted else "ABSTAIN_EARLY_ASSOCIATION_UNSTABLE"),
            "VALID8_VALID9_TEST10_counts":counts,
            "AUC_VALID8":auc["8"],
            "AUC_VALID9_no_rematch":auc["9"],
            "AUC_TEST10_no_rematch_ALREADY_SEEN":auc["10"],
            "AUC_TEST10_shuffled_label_control":shuffled_auc,
            "VALID_stability_gate_failures":failures,
            "frozen_min_VALID_gas_support":MIN_GAS_SUPPORT_PER_VALID_BATCH,
            "frozen_min_VALID_oriented_AUC":MIN_VALID_ORIENTED_AUC,
            "frozen_max_VALID_auc_change":MAX_VALID_BATCH_AUC_DIFFERENCE,
            "descriptive_test_auc_above_075":bool(
                auc["10"]>=DESCRIPTIVE_TEST_AUC_REFERENCE),
            "test_not_used_to_choose_gate_or_source":True,
            "physical_source_identity_proven":False,
        }
    return results


def marginal_drift(raw,train_idx,batches,source):
    train_m=source.fuzzy(raw[train_idx]).mean(axis=0)
    res={}
    for b in (8,9,10):
        sample=source.fuzzy(raw[batches==b])
        res[str(b)]={
            "n_exposures":len(sample),
            "mean_abs_bounded_concept_shift_from_TRAIN":float(
                np.mean(np.abs(sample.mean(axis=0)-train_m))),
            "does_not_use_any_gas_class_for_drift":True,
            "marginal_drift_cannot_certify_conditional_semantics":True,
        }
    return res


def run(path:Path,out:Path):
    if out.exists():
        raise FileExistsError("Do not overwrite retrospective V68 receipt")
    raw,gas,ppm,batch,cohorts,schema=read_original_ten_batches(path)
    train=cohorts["TRAIN"]
    source=FrozenSixGasSourceBank().fit_unlabeled(raw[train])
    source_hash=source.sha
    folds=train_only_leave_one_batch_out(raw,batch,train,source)
    rank_stable=all(r["source_rank_K"]==source.K for r in folds)
    # Critical: batch8 ONLY fixes the axis/gas relationship. Batch9 confirms
    # or forces abstention and batch10 is a retrospective evaluation only.
    idx8=np.flatnonzero(batch==8)
    if set(np.unique(gas[idx8]))!=set(range(1,7)):
        raise ValueError("Batch8 physical gas controls incomplete")
    all_scores=source.source_levels(raw)
    mapping,auc8=fix_semantic_alignment_from_valid(all_scores[idx8],gas[idx8])
    report=predeclared_early_valid_gate(
        mapping,all_scores,gas,batch,rank_is_stable=rank_stable)
    if source.sha!=source_hash:
        raise AssertionError("A downstream gas label changed source geometry")
    early_valid=[
        g for g,item in report.items()
        if item["status"]=="EARLY_VALID_ASSOCIATION_NOT_PHYSICAL_IDENTITY"
    ]
    safe={
        "status":"V68_RETROSPECTIVE_DRIFT_SEMANTIC_ABSTENTION_EXECUTED",
        "not_an_independent_sealed_TEST_because_V67_seen":True,
        "V68_thresholds_set_after_V67_test_was_inspected":True,
        "original_UCI270_source_zip_sha256":physical_archive_sha256(path),
        "read_only_source_original_class_labels_are_never_fit_inputs":True,
        "physical_dataset":"UCI270, single-gas exposures, one installation",
        "original_batch_split":schema["time_split_batches"],
        "TRAIN_VALID_TEST_source_counts":{
            n:len(ids) for n,ids in cohorts.items()},
        "original_frozen_V67_statistical_source_bank_sha256":source_hash,
        "unlabeled_original_source_K":source.K,
        "leave_one_batch_out_same_K_all_seven_training_periods":rank_stable,
        "n_train_historical_batch_stability_tests":len(folds),
        "all_unlabeled_SOURCE_subspace_stability":folds,
        "source_axis_matching_from_batch8_only":mapping,
        "matching_batch8_original_unoriented_auc":auc8.tolist(),
        "reference_support_AUC_and_abstention_by_gas":report,
        "n_valid_stable_gases_not_certified":len(early_valid),
        "gas_names_with_validated_only_association":[
            GAS_NAMES[int(k)-1] for k in early_valid],
        "unlabeled_batch8_9_10_marginal_drift":marginal_drift(
            raw,train,batch,source),
        "physical_source_specific_semantic_identity_proven":False,
        "all_gas_names_from_VALID_labels_postfit_not_autonomous":True,
        "original_V65_recurrent_dynamics_unverified_here":True,
        "protocol":"docs/medical/CONCEPTFAN_V68_FROZEN_POSTHOC_DRIFT_SEMANTIC_GATE_20261113.md",
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"STATUS.json").write_text(json.dumps(
        safe,indent=2,ensure_ascii=False,allow_nan=False)+"\n",
        encoding="utf8")
    rows=[]
    for gas_id,details in report.items():
        rows.append({
            "gas_id":gas_id,
            "gas_name":details["gas_name"],
            "status":details["status"],
            "valid_AUC8":details.get("AUC_VALID8"),
            "valid_AUC9_no_rematch":details.get("AUC_VALID9_no_rematch"),
            "previously_seen_TEST_AUC10_no_rematch":
                details.get("AUC_TEST10_no_rematch_ALREADY_SEEN"),
            "n_batch8":details["VALID8_VALID9_TEST10_counts"]["8"],
            "n_batch9":details["VALID8_VALID9_TEST10_counts"]["9"],
            "n_batch10":details["VALID8_VALID9_TEST10_counts"]["10"],
            "failure_reasons":";".join(details.get(
                "VALID_stability_gate_failures",[])),
        })
    with (out/"SEMANTIC_ABSTENTION.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({
        "event":"V68_POSTHOC_VALID_ONLY_SEMANTIC_GATE",
        "source_original_K":source.K,
        "train_source_K_stable":rank_stable,
        "train_leave_one_batch_out": [
            {"omitted":s["omitted_historical_train_batch"],
             "rank":s["source_rank_K"],
             "subspace_overlap":s["source_subspace_agreement"][
                 "mean_squared_canonical_correlation_over_common_rank"]}
            for s in folds],
        "gas_valid_semantic_results":[
            {k:val for k,val in row.items()
             if k in ("gas_name","status","valid_AUC8",
                      "valid_AUC9_no_rematch",
                      "previously_seen_TEST_AUC10_no_rematch",
                      "failure_reasons")}
            for row in rows],
    },ensure_ascii=False,allow_nan=False),flush=True)
    return safe


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--uci-zip",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args()
    run(args.uci_zip,args.out_dir)


if __name__=="__main__":
    main()
