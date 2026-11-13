#!/usr/bin/env python3
"""V31 replication: two *new draws* of the existing V27 synthetic generator.

This tests initialization/sample sensitivity of task leakage, NOT transport to
new mechanisms, external hospitals or causal identification. The generator,
training arms, probe, architecture and evaluation stay fixed from V27.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from scripts import experiment_concept_leakage_guard_v27 as v27

DRAW_CONFIGS=(
    (20261113,20262113,20263313),
    (20261114,20262114,20263314),
)
NN_SEEDS=(11,23,37)

def aggregate(root,draws):
    matrix=[]
    for draw_seed,report in draws:
        if report["generator_seed"]!=draw_seed or report["n_training_runs"]!=12:
            raise AssertionError("V27 source protocol not respected")
        for item in report["leakage_audit"]:
            if item["family"] not in ("complete","omitted"):
                continue
            matrix.append({
                "draw":draw_seed,"world":item["family"],"training":item["method"],
                "seed":item["seed"],"nll_gain":item["incremental_learned_concept_nll_gain"],
                "ci_lower":item["paired_test_subject_bootstrap_ci95"][0],
                "ci_upper":item["paired_test_subject_bootstrap_ci95"][1],
                "hidden_alignment":item["max_abs_hidden_factor_correlation"],
            })
    outcomes={}
    for world in ("complete","omitted"):
        for training in ("joint","staged"):
            subset=[z for z in matrix if z["world"]==world and z["training"]==training]
            if len(subset)!=len(DRAW_CONFIGS)*len(NN_SEEDS):
                raise AssertionError("missing or duplicated experiment fit")
            vals=np.array([z["nll_gain"] for z in subset])
            outcomes[f"{world}:{training}"]={
                "n_distinct_fit_outputs":len(vals),
                "n_distinct_test_draws":len(DRAW_CONFIGS),
                "mean_gain":float(vals.mean()),
                "median_gain":float(np.median(vals)),
                "min_gain":float(vals.min()),
                "max_gain":float(vals.max()),
                "positive_gains":int((vals>0).sum()),
                "positive_subject_bootstrap_intervals":sum(z["ci_lower"]>0 for z in subset),
            }
    for draw_seed,_ in draws:
        joint=[r["nll_gain"] for r in matrix if r["draw"]==draw_seed and
               r["world"]=="omitted" and r["training"]=="joint"]
        staged=[r["nll_gain"] for r in matrix if r["draw"]==draw_seed and
                r["world"]=="omitted" and r["training"]=="staged"]
        outcomes[f"draw:{draw_seed}"]={
            "omitted_joint_mean_gain":float(np.mean(joint)),
            "omitted_staged_mean_gain":float(np.mean(staged)),
            "joint_minus_staged_gain":float(np.mean(joint)-np.mean(staged))
        }
    result={
        "status":"REPLICATION_ONLY_SAME_GENERATOR_FAMILY",
        "generator_draws":[z[0] for z in DRAW_CONFIGS],
        "network_initialization_seeds":list(NN_SEEDS),
        "true_external_mechanism_replications":0,
        "independent_hospital_datasets":0,
        "n_new_real_data_subjects":0,
        "n_fitted_training_runs":sum(report["n_training_runs"] for _,report in draws),
        "n_distinct_test_draws":len(draws),
        "cannot_treat_all_fit_seeds_as_independent_test_cohorts":True,
        "base_protocol":"V27 JOINT vs STAGED with identical TRAIN-only additive head",
        "measures_extra_conditional_prediction_not_physical_causality":True,
        "outcomes":outcomes,
        "per_fit":matrix
    }
    (root/"V31_SUMMARY.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2),flush=True)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    root=a.out_dir
    if root.exists():
        raise FileExistsError(str(root))
    draws=[]
    for generator_seed,training_mask_seed,test_mask_base in DRAW_CONFIGS:
        v27.NEW_SEED=generator_seed
        v27.TRAIN_MASK_SEED=training_mask_seed
        v27.TEST_MASK_BASE=test_mask_base
        dest=root.parent/(root.name+"_draw_"+str(generator_seed))
        report=v27.run(out_dir=dest,smoke=a.smoke)
        draws.append((generator_seed,report))
    root.mkdir(parents=True,exist_ok=False)
    if not a.smoke:
        aggregate(root,draws)
    else:
        (root/"SMOKE_ONLY.json").write_text(json.dumps({
            "status":"SMOKE_ONLY","draws":len(draws)},indent=2))

if __name__=="__main__":
    main()
