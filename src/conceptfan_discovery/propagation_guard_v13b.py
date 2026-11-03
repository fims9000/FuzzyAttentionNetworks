"""V13b: experimental propagation counterexample and guarded graph correction.

True direct intervention can affect OTHER latent factors downstream.
Given latent z = (I-B)^-1 u, observed x=A z, and an intervention
that directly shifts only u_j by delta_j:
    D = A (I-B)^-1 diag(delta).
Thus verified single-target interventions alone do NOT identify A_j.
Only independently verified absence of propagation (B=0) or a trusted
known B (with calibrated delta) permits a direct-source statement.
Never infer B, intervention targets, or direct causes from D alone.

This standalone synthetic experiment contains ZERO PhysioNet data.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .intervention_identifiability_v13 import (
    InterventionCertificate, estimate_intervention_certificate,
)


@dataclass(frozen=True)
class EnvironmentObservations:
    x: np.ndarray
    observed: np.ndarray
    environment: np.ndarray


@dataclass(frozen=True)
class SimulatorOnlyTruth:
    loading: np.ndarray
    downstream_graph: np.ndarray
    shift_amplitude: float
    latent_total_response: np.ndarray


def simulate_single_target_with_downstream(
    *, k: int, seed: int, n_per_environment: int = 180,
    steps: int = 36, observed_fraction: float = .95,
    intervention_strength: float = 1.5,
    propagation: bool = True,
) -> tuple[EnvironmentObservations, SimulatorOnlyTruth]:
    """Randomized direct-target interventions with optional latent propagation."""
    if (k not in (3,5,7) or n_per_environment < 50 or steps < 12
            or not .7 <= observed_fraction <= 1.
            or intervention_strength <= 0):
        raise ValueError("Unsupported seeded synthetic intervention mechanism")
    rng=np.random.default_rng(seed)
    d=k+4
    loading=np.linalg.qr(rng.normal(size=(d,k)),mode="reduced")[0]
    b=np.zeros((k,k),dtype=np.float64)
    if propagation:
        for j in range(1,k):
            b[j,j-1]=.80
        for j in range(2,k):
            b[j,j-2]=.18
    total=np.linalg.inv(np.eye(k)-b)
    n=(k+1)*n_per_environment
    environments=np.repeat(np.arange(k+1),n_per_environment)
    rng.shuffle(environments)
    disturbances=np.empty((n,steps,k),dtype=np.float64)
    disturbances[:,0]=rng.normal(size=(n,k))
    rate=.68
    for t in range(1,steps):
        disturbances[:,t]=(rate*disturbances[:,t-1]
            +np.sqrt(1-rate*rate)*rng.normal(size=(n,k)))
    # Intervene directly on only ONE exogenous latent mechanism u_j,
    # but with propagation downstream z=(I-B)^-1 u.
    for j in range(k):
        disturbances[environments==j+1,:,j]+=intervention_strength
    latent=disturbances@total.T
    x=latent@loading.T+rng.normal(scale=.12,size=(n,steps,d))
    mask=(rng.random(x.shape)<observed_fraction).astype(np.float32)
    observed=np.where(mask>0,x,0).astype(np.float32)
    return (
        EnvironmentObservations(observed,mask,environments.astype(np.int32)),
        SimulatorOnlyTruth(loading,b,intervention_strength,total),
    )


def target_order_column_cosine(
    recovered:np.ndarray,oracle_loadings:np.ndarray,
)->float:
    r=np.asarray(recovered,dtype=np.float64)
    a=np.asarray(oracle_loadings,dtype=np.float64)
    if (r.shape!=a.shape or r.ndim!=2
            or not np.isfinite(r).all() or not np.isfinite(a).all()
            or min(np.linalg.norm(r,axis=0))<1e-8
            or min(np.linalg.norm(a,axis=0))<1e-8):
        raise ValueError("Finite nonzero equal-size direct response columns required")
    x=r/np.linalg.norm(r,axis=0,keepdims=True)
    y=a/np.linalg.norm(a,axis=0,keepdims=True)
    return float(np.abs(np.sum(x*y,axis=0)).mean())


def recover_direct_loadings_given_verified_graph(
    contrasts:np.ndarray,graph:np.ndarray,
    intervention_amplitudes:np.ndarray, *,
    graph_and_amplitudes_verified_externally:bool,
)->np.ndarray:
    """Invert known total-response propagation without inspecting latent oracle.

    Explicitly disallow unknown or guessed graphs: observational response D
    alone cannot distinguish A from A(I-B)^-1. This is ordinary algebra and
    NOT an identification theorem for unknown causal graphs.
    """
    d=np.asarray(contrasts,dtype=np.float64)
    b=np.asarray(graph,dtype=np.float64)
    amplitude=np.asarray(intervention_amplitudes,dtype=np.float64)
    if not isinstance(graph_and_amplitudes_verified_externally,(bool,np.bool_)):
        raise ValueError("External graph proof must be an actual boolean")
    if not bool(graph_and_amplitudes_verified_externally):
        raise ValueError("ABSTAIN_UNKNOWN_DOWNSTREAM_GRAPH_AND_INTERVENTION_SCALES")
    if (d.ndim!=2 or b.shape!=(d.shape[1],d.shape[1])
            or amplitude.shape!=(d.shape[1],)
            or not np.isfinite(d).all() or not np.isfinite(b).all()
            or not np.isfinite(amplitude).all() or np.any(np.abs(amplitude)<1e-5)
            or not np.allclose(np.diag(b),0,atol=1e-10)):
        raise ValueError("Incorrect known-graph/contrast/shift dimensions")
    # Important ordering if different shift magnitudes:
    # D=A(I-B)^(-1) diag(delta)
    # A=D diag(delta)^(-1) (I-B).
    recover=(d/amplitude[None,:])@(np.eye(len(amplitude))-b)
    if not np.isfinite(recover).all() or np.linalg.matrix_rank(recover)<len(amplitude):
        raise ValueError("Graph-corrected direct sensor source rank unresolved")
    return recover


def run_propagation_study(
    *,seed:int=20261104,sizes:tuple[int,...]=(3,5,7),
    n_per_environment:int=180,bootstraps:int=64,
)->pd.DataFrame:
    """No model-score tuning, no actual patient causal interventions."""
    rows=[]
    for k in sizes:
        for propagated in (False,True):
            observed,hidden=simulate_single_target_with_downstream(
                k=k,seed=seed+100*k,
                n_per_environment=n_per_environment,
                propagation=propagated,
            )
            # The mechanism really does directly target exactly one u_j.
            # But its effect can propagate through B into OTHER z_i.
            cert=estimate_intervention_certificate(
                observed.x,observed.observed,observed.environment,
                exclusive_single_target_provenance=True,
                no_downstream_propagation_provenance=not propagated,
                bootstraps=bootstraps,seed=seed+k,
            )
            # Audit what happens under an ASSUMPTION FALSELY asserting B=0.
            fake=estimate_intervention_certificate(
                observed.x,observed.observed,observed.environment,
                exclusive_single_target_provenance=True,
                no_downstream_propagation_provenance=True,
                bootstraps=bootstraps,seed=seed+k,
            )
            # Oracle A/B enter ONLY this evaluation block; estimator above
            # has no access to the graph, direct loadings, or latent states.
            observed_total=target_order_column_cosine(
                cert.response,hidden.loading,
            )
            corrected=recover_direct_loadings_given_verified_graph(
                cert.response,hidden.downstream_graph,
                np.full(k,hidden.shift_amplitude),
                graph_and_amplitudes_verified_externally=True,
            )
            recovered=target_order_column_cosine(corrected,hidden.loading)
            # A graph guessed as zero is not valid external graph evidence:
            # do NOT call the no-propagation response a certified direct A.
            rows.append({
                "true_factor_count_evaluation_only":k,
                "has_downstream_propagation":propagated,
                "true_direct_exogenous_target_is_exclusive":True,
                "no_downstream_propagation_provenance":not propagated,
                "actual_certificate_status":cert.status,
                "wrongly_assumed_no_propagation_status":fake.status,
                "empirical_rank":cert.rank_estimate,
                "empirical_numeric_resolution":cert.receipt[
                    "empirical_response_resolution_independent_of_provenance"],
                "min_singular_over_bootstrap_noise":cert.min_singular_value/
                    max(cert.q90_bootstrap_spectral_noise,1e-12),
                "min_bootstrap_q10_direction_cosine":float(
                    cert.bootstrap_q10_direction_cosine.min()),
                "oracle_total_response_vs_direct_loading_cosine":observed_total,
                "oracle_DIRECT_recovery_with_EXTERNALLY_known_graph_cosine":recovered,
                "oracle_direct_graph_used_in_certificate":False,
                "known_graph_used_only_in_separate_conditional_correction":True,
                "no_clinical_data":True,
                "seed":seed,
            })
            print("V13b PROPAGATION",k,propagated,
                  "claim",cert.status,"wrong_source_cos",round(observed_total,4),
                  "known_graph_corrected",round(recovered,4),flush=True)
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    if args.out_dir.exists():
        p.error("Never overwrite a scientific validation artifact")
    import yaml
    cfg=yaml.safe_load(args.config.read_text())
    if (not cfg["protocol"]["synthetic_only"]
            or not cfg["protocol"]["no_clinical_data"]
            or not cfg["protocol"]["oracle_never_used_in_certificate"]
            or tuple(cfg["data"]["k"])!=(3,5,7)
            or cfg["evaluation"]["scenarios"]!=["none","propagated"]
            or not cfg["evaluation"]["mixed_propagation_must_abstain"]):
        p.error("Frozen V13b experimental protocol invalid")
    report=run_propagation_study(
        seed=int(cfg["data"]["seed"]),
        sizes=tuple(cfg["data"]["k"]),
        n_per_environment=int(cfg["data"]["train_patients_per_environment"]),
        bootstraps=int(cfg["certificate"]["patient_bootstraps"]),
    )
    args.out_dir.mkdir(parents=True,exist_ok=False)
    report.to_csv(args.out_dir/"V13B_DOWNSTREAM_PROPAGATION.csv",index=False)
    status={
        "status":"V13B_CONDITIONAL_DOWNSTREAM_PROPAGATION_FALSIFICATION_COMPLETE",
        "no_clinical_data":True,
        "n_scenarios":len(report),
        "seed":int(cfg["data"]["seed"]),
        "first_v13_unseen_run_NOT_REUSED":41728978959,
        "individual_factor_identity_without_known_graph_NOT_CLAIMED":True,
        "results":report.astype(object).where(pd.notna(report),None).to_dict("records"),
    }
    (args.out_dir/"STATUS.json").write_text(
        json.dumps(status,indent=2,allow_nan=False)
    )


if __name__=="__main__":
    main()
