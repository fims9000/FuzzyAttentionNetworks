"""Frozen V18: independent TRAIN risk gate with source/noise counterfactuals.

A source may be temporally real but not predictive, and a predictive sensor
artifact may still be physically false. This experiment reports *all* cases
and does not select hyperparameters on these outcomes.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .candidate_admission_v17 import propose_observable_residual_concept
from .risk_screen_v18 import screen_risk_association,risk_evidence_passport

SEEDS=(2507,2607,2707)
MODES=(
    "new_source_predictive_positive",
    "new_source_predictive_negative",
    "new_source_outcome_irrelevant",
    "colored_sensor_noise_predictive",
    "colored_sensor_noise_irrelevant",
    "white_sensor_noise_irrelevant",
)
DISCOVER_N=360
RISK_SCREEN_N=260
N_STEPS=17
N_CHANNELS=5
BASIS=np.eye(N_CHANNELS,2)
CHANNEL_NAMES=tuple(f"sensor_{i}" for i in range(N_CHANNELS))


def _generate(seed,mode):
    rng=np.random.default_rng(seed)
    n=DISCOVER_N+RISK_SCREEN_N
    x=rng.normal(scale=.09,size=(n,N_STEPS,N_CHANNELS))
    for i in (0,1):
        x[:,:,i]+=rng.normal(scale=.3,size=(n,N_STEPS))
    shock=rng.normal(size=(n,N_STEPS))
    state=np.zeros_like(shock)
    if mode=="white_sensor_noise_irrelevant":
        state=shock
        x[:,:,2]+=.18*state
    else:
        for t in range(N_STEPS):
            state[:,t]=(shock[:,t] if t==0 else
                        .78*state[:,t-1]+np.sqrt(1-.78**2)*shock[:,t])
        x[:,:,2]+=.9*state
    last=state[DISCOVER_N:,-6:].mean(axis=1)
    if mode in ("new_source_predictive_positive","colored_sensor_noise_predictive"):
        py=1/(1+np.exp(-3.8*last))
    elif mode=="new_source_predictive_negative":
        py=1/(1+np.exp(3.8*last))
    else:
        py=np.full(RISK_SCREEN_N,.5)
    # independent TRAIN screen outcomes; never used in V17 candidate fit
    y=(rng.random(RISK_SCREEN_N)<py).astype(np.float64)
    return x[:DISCOVER_N],x[DISCOVER_N:],y


def run(directory:Path):
    if directory.exists():
        raise FileExistsError("Never overwrite a completed experiment")
    result=[]
    for seed in SEEDS:
        for mode in MODES:
            discovery,screen,y=_generate(seed,mode)
            common=dict(
                seed=seed,mode=mode,
                predictor_is_physical_new_factor=mode.startswith("new_source"),
                predictor_is_sensor_artifact=mode.startswith("colored_sensor"),
                sensor_noise_temporally_white=mode.startswith("white_sensor"),
                all_screen_labels_train_only=True,
                candidate_status="ABSTAIN",
                risk_status="NOT_SCREENED",signed_orientation=0,
                candidate_permutation_p=None,candidate_bootstrap_q10=None,
                association_r=None,risk_p=None,risk_ci95_lo=None,
                risk_ci95_hi=None,
                signed_candidate_sensor_2=None,
                estimated_physical_cause_certified=False,
            )
            try:
                candidate=propose_observable_residual_concept(
                    discovery,np.ones_like(discovery),BASIS,CHANNEL_NAMES,
                    seed=701,permutations=99,bootstraps=24,
                )
            except ValueError as e:
                if "ABSTAIN" not in str(e):
                    raise
            else:
                ev=screen_risk_association(
                    candidate,screen,np.ones_like(screen),y,
                    screening_episode_ids=range(DISCOVER_N,DISCOVER_N+RISK_SCREEN_N),
                    discovery_episode_ids=range(DISCOVER_N),
                    seed=701,permutations=199,bootstraps=199,
                )
                common.update(
                    candidate_status="PROPOSED",
                    risk_status=ev.status,
                    signed_orientation=ev.signed_orientation,
                    candidate_permutation_p=candidate.permutation_p,
                    candidate_bootstrap_q10=candidate.bootstrap_abs_cosine_q10,
                    association_r=ev.point_biserial_r,
                    risk_p=ev.permutation_p_two_sided,
                    risk_ci95_lo=ev.bootstrap_ci95[0],
                    risk_ci95_hi=ev.bootstrap_ci95[1],
                    signed_candidate_sensor_2=float(candidate.direction[2]),
                )
            result.append(common)
            print(seed,mode,common["candidate_status"],common["risk_status"],
                  common["signed_orientation"],flush=True)
    summary=[]
    for mode in MODES:
        sub=[x for x in result if x["mode"]==mode]
        summary.append({
            "mode":mode,"n":len(sub),
            "candidate_proposals":sum(x["candidate_status"]=="PROPOSED" for x in sub),
            "risk_sign_accepted":sum(x["risk_status"]=="ACCEPT_PREDICTIVE_ASSOCIATION" for x in sub),
            "risk_abstentions":sum(x["risk_status"]=="ABSTAIN_UNSTABLE_OR_WEAK_ASSOCIATION" for x in sub),
            "no_candidate_abstentions":sum(x["candidate_status"]=="ABSTAIN" for x in sub),
            "causality_verified":False,
        })
    receipt={
        "status":"V18_TRAIN_ONLY_INDEPENDENT_RISK_ASSOCIATION_SCREEN",
        "number_cases":len(result),
        "complete_case_reporting":True,
        "discovery_samples_train_only":DISCOVER_N,
        "screen_samples_train_only":RISK_SCREEN_N,
        "discovery_and_risk_screen_disjoint":True,
        "clinical_ground_truth_absent":True,
        "source_and_sensor_noise_causality_not_identifiable":True,
        "does_not_change_accepted_PIERE":True,
        "per_case":result,"summary":summary,
    }
    directory.mkdir(parents=True,exist_ok=False)
    for filename,records in (("ALL_CASES.csv",result),("SUMMARY.csv",summary)):
        with (directory/filename).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (directory/"STATUS.json").write_text(json.dumps(receipt,indent=2,allow_nan=False))
    return receipt


if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-dir",required=True,type=Path)
    args=ap.parse_args()
    run(args.out_dir)
