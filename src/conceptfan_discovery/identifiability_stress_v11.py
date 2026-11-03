"""Controlled positive and negative identifiability stress tests (NO concept training labels).

Planted latent state dimension = 3, 5 or 7; independent linear temporal
mechanisms, with distinct versus repeated transition rates. Repeated
eigenvalues admit arbitrary rotations of a latent subspace: labeling every
SVD axis a different independent disease would be false even at N=infinity.

We evaluate the learned *observable span* against the simulator's true
loading span only AFTER fitting without oracle labels. Independent-time
shuffle destroys autocorrelation while preserving static channel marginals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .identifiability_v11 import fit_predictive_group_certificate, projector_overlap
from .lagged_rank import fit_lagged_predictive_rank


def simulate_linear_dynamics(
    *,
    k:int,regime:str,seed:int,n_patients:int=900,
    steps:int=36,channels:int=10,
    observed_fraction:float=.94,
) -> tuple[np.ndarray,np.ndarray,np.ndarray,dict]:
    if k not in (3,5,7) or regime not in ("distinct","degenerate","time_shuffled","iid_temporal_null"):
        raise ValueError("Only predeclared 3/5/7-factor mechanisms are supported")
    if channels<k+1 or n_patients<100 or steps<10 or not .5<observed_fraction<=1.:
        raise ValueError("Insufficient generator resolution")
    rng=np.random.default_rng(seed)
    random_matrix=rng.normal(size=(channels,k))
    emission=np.linalg.qr(random_matrix,mode="reduced")[0]
    # Fixed stationary marginal variance prevents simple variance from
    # disclosing K; predictive lag structure must carry the signal.
    rate=np.linspace(.94,.35,k)
    if regime=="degenerate":
        rate[1]=rate[0]
    latent=np.zeros((n_patients,steps,k),dtype=np.float64)
    latent[:,0]=rng.normal(size=(n_patients,k))
    innovation_scale=np.sqrt(1.-rate**2)
    for t in range(1,steps):
        latent[:,t]=rate*latent[:,t-1]+rng.normal(size=(n_patients,k))*innovation_scale
    if regime=="iid_temporal_null":
        # A STRICT negative control: independent latent innovations at EACH
        # time, retaining cross-sensor instantaneous mixing and marginals,
        # but removing all past-to-future latent dependence. This is stronger
        # than shuffling each patient's times (which preserves episode means).
        latent=rng.normal(size=(n_patients,steps,k))
    raw=latent@emission.T+rng.normal(scale=.09,size=(n_patients,steps,channels))
    mask=(rng.random(size=raw.shape)<observed_fraction).astype(np.float64)
    if regime=="time_shuffled":
        for patient in range(n_patients):
            order=rng.permutation(steps)
            raw[patient]=raw[patient,order]
            mask[patient]=mask[patient,order]
    x=np.where(mask>0,raw,0.).astype(np.float32)
    receipt={
        "n_true_factors_oracle_evaluation_only":k,
        "regime":regime,
        "n_train_patients":n_patients,
        "observed_steps":steps,
        "channels":channels,
        "rate_values_used_only_in_generator":rate.tolist(),
        "oracle_emission_never_used_for_fit":True,
        "oracle_latent_traces_never_used_for_fit":True,
        "time_shuffled_control_preserves_channel_distribution":regime=="time_shuffled",
        "strict_no_temporal_dependence_control":regime=="iid_temporal_null",
        "seed":seed,
    }
    return x,mask.astype(np.float32),emission,receipt


def run_stress(cfg:dict)->tuple[pd.DataFrame,dict]:
    rows=[]
    all_groups={}
    k_list=cfg["data"]["k_factors"]
    for k in k_list:
        for regime in cfg["data"]["regimes"]:
            x,m,emission,receipt=simulate_linear_dynamics(
                k=int(k),regime=regime,
                seed=int(cfg["data"]["seed"])+int(k)*100+
                     list(cfg["data"]["regimes"]).index(regime),
                n_patients=int(cfg["data"]["n_train_patients"]),
                steps=int(cfg["data"]["history_hours"]),
                channels=int(cfg["data"]["n_channels"]),
                observed_fraction=float(cfg["data"]["observed_fraction"]),
            )
            r=cfg["rank"]
            rank=fit_lagged_predictive_rank(
                x,m,lags=tuple(r["lags"]),
                permutations=int(r["permutations"]),
                patient_bootstraps=int(r["patient_bootstraps"]),
                alpha=float(r["alpha"]),
                seed=int(r["seed"])+int(k),
            )
            count=int(min(rank.rank,int(r["max_computational_slots"])))
            row={
                **receipt,
                "detected_predictive_rank":count,
                "rank_bootstrap_interval":list(rank.patient_bootstrap_interval),
                "familywise_signal_noise_threshold":float(rank.noise_max_threshold),
                "leading_singular_value":float(rank.singular_values[0]),
                "oracle_span_overlap":None,
                "n_stable_singletons":0,
                "n_stable_groups":0,
                "n_abstained_groups":0,
                "group_dimension_list":[],
            }
            if count>=1:
                cert=fit_predictive_group_certificate(
                    x,m,computational_rank=count,
                    lags=tuple(r["lags"]),
                    bootstraps=int(cfg["certificate"]["bootstraps"]),
                    seed=int(cfg["certificate"]["seed"])+int(k),
                    minimum_subspace_overlap=float(cfg["certificate"]["min_overlap"]),
                    gap_noise_quantile=float(cfg["certificate"]["gap_noise_quantile"]),
                )
                comparison_rank=min(int(k),count)
                # Report projection onto true observable emission span only
                # when K is matched; mismatched rank has no equal-dimensional
                # projector overlap and should NOT be called recovered.
                if count==k and regime in ("distinct","degenerate"):
                    row["oracle_span_overlap"]=projector_overlap(
                        cert.reference_basis[:,:k],emission
                    )
                row.update({
                    "n_stable_singletons":sum(g.status=="INDIVIDUAL_OBSERVABLE_AXIS_STABLE"
                                              for g in cert.groups),
                    "n_stable_groups":sum(g.status=="GROUP_SUBSPACE_STABLE_AXES_UNRESOLVED"
                                          for g in cert.groups),
                    "n_abstained_groups":sum(g.status.startswith("ABSTAIN")
                                             for g in cert.groups),
                    "group_dimension_list":[g.stop-g.first for g in cert.groups],
                })
                all_groups[f"true_{k}_{regime}"]=cert.to_json()
            print(f"trueK={k} regime={regime} detectK={count} "
                  f"n_axis_cert={row['n_stable_singletons']} "
                  f"n_group_cert={row['n_stable_groups']}",flush=True)
            rows.append(row)
    return pd.DataFrame(rows),all_groups


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    if a.out_dir.exists():
        p.error("Refusing to overwrite prior falsification results")
    cfg=yaml.safe_load(a.config.read_text(encoding="utf-8"))
    if not (cfg["protocol"]["oracle_only_after_fit"] and
            cfg["protocol"]["no_test_or_clinical_data"] and
            cfg["protocol"]["negative_time_shuffle_required"]):
        p.error("Falsification protocol not frozen")
    if set(cfg["data"]["regimes"])!={"distinct","degenerate","time_shuffled","iid_temporal_null"}:
        p.error("Do not omit the negative time-shuffle control")
    table,details=run_stress(cfg)
    a.out_dir.mkdir(parents=True,exist_ok=False)
    table.to_csv(a.out_dir/"predictive_identifiability_3_5_7_stress.csv",index=False)
    (a.out_dir/"TRAIN_ONLY_CERTIFICATES.json").write_text(
        json.dumps(details,indent=2,allow_nan=False),encoding="utf-8"
    )
    status={
        "status":"V11_PREDICTIVE_IDENTIFIABILITY_STRUCTURAL_STRESS_COMPLETE",
        "no_true_factors_supplied_to_estimator":True,
        "simulation_hidden_state_identities_are_evaluation_only":True,
        "null_condition":"Within-patient time shuffle (weak null) AND independent iid latent time draws (strict null)",
        "measured_metrics_are_empirical_not_a_new_identifiability_theorem":True,
        "n_factors_evaluated":list(cfg["data"]["k_factors"]),
        "results":table.astype(object).where(pd.notna(table),None).to_dict(orient="records"),
        "limitations":[
            "Linear stationary AR toy, not real physiology",
            "Finite-data gaps and group thresholds are heuristic certificates",
            "True factor count may not equal observable predictive rank",
            "Repeated eigenvalues make individual directions non-unique",
            "Within-patient time shuffle retains patient-specific episode means and can create spurious positive-lag structure",
            "All clinical claims need independent assessment",
        ],
    }
    (a.out_dir/"STATUS.json").write_text(
        json.dumps(status,indent=2,allow_nan=False),encoding="utf-8"
    )


if __name__=="__main__":
    main()
