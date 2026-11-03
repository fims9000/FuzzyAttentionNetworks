"""Exact finite 3-initialization sensitivity for frozen V10 development evidence.

Enumerates 2^3 sign flips and 3^3 paired seed resamples. This is *not*
patient bootstrap and not confirmatory inference, regardless of CI width.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIOS = ("original", "mcar_thin", "time_channel_mar")
METHODS = ("v7_retrained_baseline", "v10_multiscale", "v10_multiscale_consistency")
METRICS = (
    "auprc_full",
    "auprc_physical_only",
    "physical_masked_reconstruction_rmse",
    "physical_future_observed_rmse",
    "mean_fraction_signed_abs_logit_from_process",
)


def exact_seed_sensitivity(differences: np.ndarray) -> dict:
    d = np.asarray(differences,dtype=np.float64)
    if d.shape != (3,) or not np.isfinite(d).all():
        raise ValueError("Exactly 3 finite paired source-seed differences required")
    boot=np.asarray([
        float(d[np.asarray(draw,dtype=int)].mean())
        for draw in itertools.product(range(3),repeat=3)
    ])
    flips=np.asarray([
        float((d*np.asarray(signs,dtype=float)).mean())
        for signs in itertools.product((-1.,1.),repeat=3)
    ])
    return {
        "mean_paired_difference":float(d.mean()),
        "per_initialization_differences":d.tolist(),
        "model_only_seed_resample_ci95_low":float(np.quantile(boot,.025)),
        "model_only_seed_resample_ci95_high":float(np.quantile(boot,.975)),
        "two_sided_exact_signflip_p":float(
            np.mean(np.abs(flips)>=abs(float(d.mean()))-1e-12)
        ),
        "n_independent_initializations":3,
        "n_exhaustive_seed_bootstraps":27,
        "n_exhaustive_sign_flips":8,
        "patients_resampled":False,
        "independent_hospitals":False,
        "confirmatory_inference":False,
    }


def audit_frozen_tables(records:pd.DataFrame,pairs:pd.DataFrame)->tuple[pd.DataFrame,dict]:
    required={"method","seed","development_scenario",*METRICS}
    if not required.issubset(records.columns):
        raise ValueError(f"Frozen metrics missing required columns: {required-set(records.columns)}")
    if not {"method","concept_family","seed_a","seed_b",
            "definition_cosine_normalized",
            "excess_over_unmatched_episode_null_spearman"}.issubset(pairs.columns):
        raise ValueError("Frozen cross-seed alignment table missing required columns")
    expected={(m,s,k) for m in METHODS for s in (11,23,37) for k in SCENARIOS}
    actual=set(zip(records.method,records.seed,records.development_scenario))
    if actual!=expected or len(records)!=len(expected):
        raise ValueError("Expected exact matched methods, source seeds and scenarios with no duplicate")
    rows=[]
    for scenario in SCENARIOS:
        for challenger in METHODS[1:]:
            for metric in METRICS:
                base=records[(records.method==METHODS[0])&
                             (records.development_scenario==scenario)].sort_values("seed")
                proposed=records[(records.method==challenger)&
                                 (records.development_scenario==scenario)].sort_values("seed")
                assert list(base.seed)==list(proposed.seed)==[11,23,37]
                z=exact_seed_sensitivity(
                    proposed[metric].to_numpy(dtype=float)-base[metric].to_numpy(dtype=float)
                )
                rows.append({"scenario":scenario,"challenger":challenger,
                             "baseline":METHODS[0],"metric":metric,**z})
    stability={}
    for method in METHODS:
        stability[method]={}
        for family in ("phys","care"):
            subset=pairs[(pairs.method==method)&(pairs.concept_family==family)]
            keyset={tuple(sorted((int(a),int(b)))) for a,b in
                    subset[["seed_a","seed_b"]].itertuples(index=False,name=None)}
            if keyset!={(11,23),(11,37),(23,37)} or len(subset)!=3:
                raise ValueError("Expected 3 overlapping pair comparisons, not independent samples")
            stability[method][family]={
                "mean_definition_cosine_normalized":
                    float(subset.definition_cosine_normalized.mean()),
                "mean_patient_specific_spearman_excess":
                    float(subset.excess_over_unmatched_episode_null_spearman.mean()),
                "n_overlapping_model_pairs":3,
                "pairwise_p_value_reported":False,
            }
    return pd.DataFrame(rows),stability


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--frozen-results",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    a=parser.parse_args()
    if a.out_dir.exists():
        parser.error("Refusing to overwrite existing audit")
    src=a.frozen_results
    table=pd.read_csv(src/"v10_matched_development_by_scenario.csv")
    pairs=pd.read_csv(src/"v10_cross_seed_concept_agreement.csv")
    sensitivity,stability=audit_frozen_tables(table,pairs)
    a.out_dir.mkdir(parents=True,exist_ok=False)
    sensitivity.to_csv(a.out_dir/"v10_exact_three_seed_sensitivity.csv",index=False)
    (a.out_dir/"STATUS.json").write_text(json.dumps({
        "status":"EXACT_EXHAUSTIVE_THREE_SEED_DESCRIPTIVE_SENSITIVITY",
        "baseline":METHODS[0],
        "candidate_methods":list(METHODS[1:]),
        "statistics":"paired seed exact signflip + seed-only exhaustive bootstrap",
        "small_n_limit":"With 3 sign-symmetric independent initializations, minimum two-sided exact p is 0.25",
        "n_independent_initializations":3,
        "not_included":"patient, hospital, data-order, method selection uncertainty",
        "already_repeated_development_validation":True,
        "not_confirmatory_statistical_significance":True,
        "concept_stability":stability,
    },indent=2),encoding="utf-8")
    print(sensitivity[sensitivity.metric.eq("auprc_full")][
        ["scenario","challenger","mean_paired_difference",
         "model_only_seed_resample_ci95_low","model_only_seed_resample_ci95_high",
         "two_sided_exact_signflip_p"]].to_string(index=False))


if __name__=="__main__":
    main()
