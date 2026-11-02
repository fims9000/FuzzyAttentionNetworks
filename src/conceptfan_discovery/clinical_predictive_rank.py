"""Clinical stress audit for train-only unsupervised temporal rank.

Crucial negative control: a rank estimator that works on clean synthetic
linear systems need NOT identify physiological mechanisms in ICU records.
Inspect train patient subsample stability, missingness drift, and observable
predictive subspace reproducibility without consulting test/calibration data.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import yaml

from .lagged_rank import fit_lagged_predictive_rank
from .physionet import prepare_physionet_discovery


def _basis_overlap(a: np.ndarray, b: np.ndarray, ka: int, kb: int) -> dict:
    if min(ka,kb)<1:
        return {"matched_dimensions":0,"mean_squared_principal_cosine":None}
    pa=a[:,:ka].T@b[:,:kb]
    similarity=np.linalg.svd(pa,compute_uv=False)
    return {
        "matched_dimensions":int(min(ka,kb)),
        "mean_squared_principal_cosine":float((similarity[:min(ka,kb)]**2).mean()),
        "principal_cosines":similarity[:min(ka,kb)].tolist(),
        "interpretation":"train-subsample observable predictive subspace agreement, not causal mechanism matching",
    }


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--raw-zip",type=Path,required=True)
    parser.add_argument("--outcomes",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error("Refusing to overwrite an existing audit")
    cfg=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not cfg["protocol"]["no_test_or_calibration_use"]:
        parser.error("This rank audit is train-only and cannot use sealed data")
    with tempfile.TemporaryDirectory(prefix="pd_clinical_rank_split_") as tmp:
        data=prepare_physionet_discovery(
            args.raw_zip,args.outcomes,Path(tmp),
            split_seed=int(cfg["data"]["split_seed"]),
            input_window=int(cfg["data"]["input_window"]),
            forecast_horizon=int(cfg["data"]["forecast_horizon"]),
            minimum_train_patient_coverage=float(cfg["data"]["minimum_train_patient_coverage"]),
        )
    x=data.train.x
    m=data.train.mask
    rank_cfg=cfg["rank"]
    rank_args={
        "lags":tuple(int(v) for v in rank_cfg["lags"]),
        "permutations":int(rank_cfg["permutations"]),
        "patient_bootstraps":int(rank_cfg["patient_bootstraps"]),
        "alpha":float(rank_cfg["alpha"]),
    }
    full=fit_lagged_predictive_rank(
        x,m,seed=int(rank_cfg["seed"]),**rank_args
    )
    n=len(x)
    rng=np.random.default_rng(int(rank_cfg["subsample_seed"]))
    shuffled=rng.permutation(n)
    sub=[]
    for part,idx in enumerate((shuffled[:n//2],shuffled[n//2:])):
        sub.append(fit_lagged_predictive_rank(
            x[idx],m[idx],seed=int(rank_cfg["seed"])+part+11,**rank_args
        ))
    full_rank=full.rank
    names=data.receipt["channel_names"]
    axis=[]
    for concept in range(full_rank):
        u=full.left_observable_loadings[:,concept]
        positions=np.argsort(-np.abs(u))[:min(5,len(u))]
        axis.append({
            "rank_index":int(concept),
            "observed_signed_channel_loadings":[
                {"channel":names[j],"signed_loading":float(u[j])}
                for j in positions
            ],
            "observable_basis_only":True,
            "clinical_concept_label_assigned":False,
        })
    mid=m.shape[1]//2
    coverage_early=m[:,:mid].mean(axis=(0,1))
    coverage_late=m[:,mid:].mean(axis=(0,1))
    mean_early=(x[:,:mid]*m[:,:mid]).sum((0,1)) / np.maximum(m[:,:mid].sum((0,1)),1.)
    mean_late=(x[:,mid:]*m[:,mid:]).sum((0,1)) / np.maximum(m[:,mid:].sum((0,1)),1.)
    rms=np.sqrt((x*x*m).sum((0,1)) / np.maximum(m.sum((0,1)),1.))
    standardized_mean_drift=(mean_late-mean_early)/np.maximum(rms,.05)
    clinically_unvalidated={
        "status":"CLINICAL_TRAIN_OBSERVABLE_RANK_DIAGNOSTIC",
        "official_train_data_receipt":data.receipt,
        "full_train_rank":full.to_json(),
        "two_independent_train_subsamples":[x.to_json() for x in sub],
        "half_train_subspace_agreement":_basis_overlap(
            sub[0].left_observable_loadings,
            sub[1].left_observable_loadings,
            sub[0].rank,sub[1].rank,
        ),
        "top_observable_channel_loadings_per_axis":axis,
        "early_late_train_temporal_drift":{
            "fraction_channels_with_abs_standardized_mean_drift_above_0_25":
                float((np.abs(standardized_mean_drift)>.25).mean()),
            "max_abs_standardized_mean_drift":float(np.max(np.abs(standardized_mean_drift))),
            "max_abs_observation_coverage_change":float(
                np.max(np.abs(coverage_late-coverage_early))
            ),
        },
        "no_individual_record_ids_or_predictions_published":True,
        "test_or_calibration_used":False,
        "clinical_concept_validity_inferred":False,
        "limitations":[
            "Positive-lag covariance is confounded by informative missingness, treatment decisions, nonstationarity and patient mixture",
            "Subsample rank variability signals lack of stable observable dimension; no rank result is physiological ontology",
            "Lagged covariance masks correct only pairwise coverage, not missing-not-at-random bias",
            "The 600 already inspected historical test cases are not a clean prospective evaluation set",
        ],
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(clinically_unvalidated,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({
        "full_train_k":full.rank,
        "half_train_ks":[v.rank for v in sub],
        "half_subspace_agreement":clinically_unvalidated["half_train_subspace_agreement"],
        "measurement_drift":clinically_unvalidated["early_late_train_temporal_drift"],
        "clinical_validity_claim":False,
    },indent=2),flush=True)


if __name__=="__main__":
    main()
