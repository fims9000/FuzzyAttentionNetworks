#!/usr/bin/env python3
"""V64: causal baseline subtraction / rank compression / fuzzy bottleneck audit.

V63 data's TEST was ALREADY viewed. This report is posthoc diagnostic, NOT an
independent source-identity confirmation or a positive architecture selection.
"""
from __future__ import annotations

import argparse,csv,json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conceptfan_discovery.train_innovation_ica_bank_v63 import TrainOnlyInnovationBank
from scripts.experiment_real_dynamic_gas_ica_v63 import (
    build_real_windows,train_only_sensor_scaling,reg_metrics,DATASET_FILES,
    SECOND_COLS,source_sha256
)


def physical_sources(x,m,banks,absolute_pca):
    """Seven distinct information-preserving/lossy stages, no gas labels."""
    if not m.all():
        raise ValueError("No missing observed source timestamps in official UCI322")
    raw=x.mean(axis=1)
    late=x[:,90:120,:].mean(axis=1)
    diff=x-x[:,0:1,:]
    original=np.concatenate((raw,late),axis=1)
    delta=np.concatenate((diff.mean(axis=1),diff[:,90:120,:].mean(axis=1)),
                         axis=1)
    represented={
        "raw_absolute_32":original,
        "raw_initial_differenced_32":delta,
        "absolute_pca_K":absolute_pca.transform(original),
    }
    for source,bank in banks.items():
        basis=bank.loading
        absolute=np.einsum("ntd,dk->ntk",x,basis,optimize=True)
        projected,support=bank.transform(x,m)
        fuzzy=bank.static_fuzzy(x,m)
        if (projected.shape!=fuzzy.shape or
                not np.isfinite(absolute).all() or
                not np.isfinite(fuzzy).all()):
            raise AssertionError("Inconsistent source transformation stage")
        represented.update({
            source+"_projected_absolute_linear_K":np.column_stack(
                (absolute.mean(axis=1),absolute[:,90:120].mean(axis=1))),
            source+"_projected_delta_linear_K":np.column_stack(
                (projected.mean(axis=1),projected[:,90:120].mean(axis=1))),
            source+"_projected_delta_fuzzy_all_K":fuzzy.mean(axis=1),
            source+"_projected_delta_fuzzy_late_K":fuzzy[:,90:120].mean(axis=1),
        })
    if len(represented)!=11:
        raise AssertionError("Expected 11 preregistered information-loss stages")
    return represented


def _train_only_bank(x,m,groups,basis):
    return TrainOnlyInnovationBank(basis=basis,seed=20261913,
        explained_variance_target=.95,max_k=8,block_bootstraps=20
        ).fit_unlabeled(x,m,train_halfhour_blocks=groups)


def run(path:Path,out:Path):
    if out.exists():
        raise FileExistsError("Pinned posthoc research artifact must never be overwritten")
    runs,schema=build_real_windows(path)
    x,m,refs,session,groups,normalizer=train_only_sensor_scaling(runs)
    banks={basis:_train_only_bank(x["TRAIN"],m["TRAIN"],groups,basis)
           for basis in ("pca","ica")}
    if banks["ica"].k!=banks["pca"].k:
        raise AssertionError("Rank-matched source definitions changed")
    k=banks["ica"].k
    xtr=np.column_stack((x["TRAIN"].mean(axis=1),
                         x["TRAIN"][:,90:120].mean(axis=1)))
    absolute_pca=make_pipeline(StandardScaler(),PCA(n_components=k,random_state=3))
    absolute_pca.fit(xtr)
    represented={cohort:physical_sources(x[cohort],m[cohort],banks,absolute_pca)
                 for cohort in ("TRAIN","VALID","TEST")}
    methods=tuple(represented["TRAIN"])
    rows=[]
    for sidx,record_name in enumerate(DATASET_FILES):
        for reference_index,target_name in ((0,SECOND_COLS[sidx]),(1,"ethylene")):
            tr=np.flatnonzero(session["TRAIN"]==sidx)
            ytr=refs["TRAIN"][tr,reference_index]
            if np.std(ytr)<1e-7:
                raise ValueError("No gas setpoint variation in TRAIN controls")
            for cohort in ("VALID","TEST"):
                ids=np.flatnonzero(session[cohort]==sidx)
                y=refs[cohort][ids,reference_index]
                for method in methods:
                    trainf=represented["TRAIN"][method][tr]
                    evalf=represented[cohort][method][ids]
                    reg=make_pipeline(StandardScaler(),Ridge(alpha=1.))
                    reg.fit(trainf,ytr)
                    pred=reg.predict(evalf)
                    rows.append({
                        "reference_target":record_name+":"+target_name,
                        "cohort":cohort,"stage":method,
                        "train_only_unsupervised_selected_K":k,
                        "n_correlated_120s_windows":len(ids),
                        "source_already_inspected_V63_TEST":True,
                        **reg_metrics(y,pred,float(np.std(ytr)))})
    report={
        "status":"V64_POSTHOC_REAL_GAS_INFORMATION_BOTTLENECK_EXECUTED",
        "declared_post_V63_TEST_inspection":True,
        "new_independent_confirmatory_data":False,
        "single_source_testbed_two_recordings":True,
        "source_ICA_PCA_still_fit_without_setpoints":True,
        "protocol":"docs/medical/CONCEPTFAN_V64_FROZEN_BOTTLENECK_ABLATION_20261113.md",
        "official_UCI322_zip_sha256":source_sha256(path),
        "source_120s_time_embargo":schema,
        "original_v63_number_trajectory_windows":{
            name:len(samples) for name,samples in runs.items()},
        "train_only_normalization":normalizer,
        "source_rank_K":k,
        "source_train_sha256":{
            name:bank.sha for name,bank in banks.items()},
        "normalization_and_pca_fit_only_train":True,
        "no_concentration_labels_enter_statistical_source_space":True,
        "all_stage_results":rows,
        "real_physical_source_semantics_identified":False,
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"STATUS.json").write_text(json.dumps(
        report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",
        encoding="utf-8")
    with (out/"METRICS.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({
        "event":"V64_INFO_BOTTLENECK_EXECUTED",
        "K":k,
        "all_test":[{key:value for key,value in row.items()
                     if key in ("reference_target","stage","R2","MAE_ppm")}
                    for row in rows if row["cohort"]=="TEST"]
    },ensure_ascii=False,allow_nan=False),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--uci-zip",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    a=parser.parse_args()
    run(a.uci_zip,a.out_dir)


if __name__=="__main__":
    main()
