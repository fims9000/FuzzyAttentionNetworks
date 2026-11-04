"""Small frozen V17 falsification: a new factor versus noise-only decoys.

All nine cases, including ABSTAIN, are reported. Generator truth is used only
to classify conditions in this *evaluation* script, never for fitting.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from .candidate_admission_v17 import (
    propose_observable_residual_concept,candidate_passport,
)

SEEDS = (2507,2607,2707)
MODES = ("new_dynamic_factor","white_sensor_only","colored_sensor_only")
EXISTING_K = 2


def generate_case(mode,seed,n=440,t=17,d=5):
    rng=np.random.default_rng(seed)
    x=rng.normal(scale=.09,size=(n,t,d))
    for i in (0,1):
        x[:,:,i]+=rng.normal(size=(n,t))*.3
    shock=rng.normal(size=(n,t))
    state=np.zeros((n,t))
    if mode in ("new_dynamic_factor","colored_sensor_only"):
        for j in range(t):
            state[:,j]=(shock[:,j] if j==0 else
                         .78*state[:,j-1]+np.sqrt(1-.78**2)*shock[:,j])
        x[:,:,2]+=.9*state
    else:
        state=shock
        x[:,:,2]+=.18*state
    return x,np.ones_like(x),np.eye(d,EXISTING_K)


def run(output:Path):
    if output.exists():
        raise FileExistsError("Never overwrite completed evidence")
    rows=[]
    for seed in SEEDS:
        for mode in MODES:
            x,m,b=generate_case(mode,seed)
            try:
                candidate=propose_observable_residual_concept(
                    x,m,b,tuple(f"sensor_{i}" for i in range(x.shape[-1])),
                    seed=701,permutations=99,bootstraps=24,
                )
                status="PROPOSED"
                p=float(candidate.permutation_p)
                cosine=float(candidate.bootstrap_abs_cosine_q10)
                oracle_abs_cosine=float(abs(candidate.direction[2]))
                identifier=candidate.candidate_id
            except ValueError as exc:
                if "ABSTAIN" not in str(exc):
                    raise
                status="ABSTAIN"
                p=None; cosine=None; oracle_abs_cosine=None;identifier=None
            rows.append({
                "generator_seed":seed,"mode":mode,
                "true_dynamic_latent_added":mode=="new_dynamic_factor",
                "temporally_correlated_sensor_noise_added":mode=="colored_sensor_only",
                "status":status,
                "train_only_candidate_p":p,
                "bootstrap_direction_cosine_q10":cosine,
                "oracle_sensor2_direction_cosine_eval_only":oracle_abs_cosine,
                "candidate_id":identifier,
                "physically_identified":False,
            })
            print(seed,mode,status,flush=True)
    summary=[]
    for mode in MODES:
        sub=[r for r in rows if r["mode"]==mode]
        summary.append({
            "mode":mode,"total":len(sub),
            "proposals":sum(r["status"]=="PROPOSED" for r in sub),
            "abstentions":sum(r["status"]=="ABSTAIN" for r in sub),
            "semantics_confirmed":False,
        })
    output.mkdir(parents=True,exist_ok=False)
    with (output/"ALL_CASES.csv").open("w",newline="") as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    with (output/"SUMMARY.csv").open("w",newline="") as file:
        w=csv.DictWriter(file,fieldnames=list(summary[0]))
        w.writeheader();w.writerows(summary)
    receipt={
        "status":"V17_RESIDUAL_CONCEPT_SYNTHETIC_FALSIFICATION",
        "n_cases":len(rows),"no_oracle_training":True,
        "no_medical_semantic_claim":True,
        "all_proposals_and_refusals_included":True,
        "missingness_not_tested_because_algorithm_abstains":True,
        "candidate_is_not_automatically_admitted":True,
        "results":rows,"summary":summary,
    }
    (output/"STATUS.json").write_text(json.dumps(receipt,indent=2,allow_nan=False))
    return receipt


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    run(args.out_dir)
