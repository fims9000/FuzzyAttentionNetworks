"""V13 standalone synthetic falsification with unseen patient episodes.

No PhysioNet files, no real patient outcomes. The true mixing A and latent z
are returned in a separate evaluation-oracle object and NEVER passed into
certificate fitting or classifier training. Real clinical interventions must
be verified externally before these structural guarantees can apply.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from .intervention_identifiability_v13 import (
    InterventionCertificate, estimate_intervention_certificate,
    train_intervention_concept_model,
)


@dataclass(frozen=True)
class TrialObservations:
    x: np.ndarray
    mask: np.ndarray
    environment: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class TrialOracle:
    latent_final: np.ndarray
    emission: np.ndarray
    intervention_targets: np.ndarray


def simulate_randomized_intervention_trials(
    *, k: int, seed: int, regime: str, n_per_environment: int,
    steps: int = 36, observed_fraction: float = .95,
    intervention_amplitude: float = 1.5,
) -> tuple[TrialObservations, TrialOracle]:
    if (k not in (3, 5, 7) or regime not in
            ("exclusive", "multitarget", "duplicated", "sham")
            or n_per_environment < 40 or steps < 12
            or not .7 <= observed_fraction <= 1 or intervention_amplitude <= 0):
        raise ValueError("Unsupported synthetic intervention design")
    rng = np.random.default_rng(seed)
    d = k + 4
    a = np.linalg.qr(rng.normal(size=(d,k)),mode="reduced")[0]
    # These intervention mechanism coefficients are hidden ground truth.
    b = np.eye(k)
    if regime == "multitarget":
        # All environments target more than one latent cause; invertible.
        # Learned response directions can be stable BUT non-causal-factor axes.
        b = np.linalg.qr(rng.normal(size=(k,k)), mode="reduced")[0]
    elif regime == "duplicated":
        b[:,1] = b[:,0]
    elif regime == "sham":
        b[:] = 0
    n = (k+1)*n_per_environment
    environment = np.repeat(np.arange(k+1), n_per_environment)
    rng.shuffle(environment)
    latent = np.zeros((n,steps,k),dtype=np.float64)
    # Isotropic independent latent dynamics: observational rotations allowed.
    latent[:,0] = rng.normal(size=(n,k))
    rate=.73
    for t in range(1,steps):
        latent[:,t] = rate*latent[:,t-1] + np.sqrt(1-rate**2)*rng.normal(size=(n,k))
    for i in range(1,k+1):
        latent[environment==i] += intervention_amplitude*b[:,i-1]
    raw = latent@a.T + rng.normal(scale=.12,size=(n,steps,d))
    mask = (rng.random(size=raw.shape)<observed_fraction).astype(np.float32)
    observed = np.where(mask>0,raw,0).astype(np.float32)
    beta = np.where(np.arange(k)%2==0, 1.0,-1.0)*np.linspace(.9,1.4,k)
    recent = latent[:,-6:,:].mean(axis=1)
    risk = -.20 + recent@beta
    p = 1/(1+np.exp(-np.clip(risk,-20,20)))
    y = (rng.random(n)<p).astype(np.float32)
    return TrialObservations(observed,mask,environment,y),TrialOracle(
        latent_final=recent.astype(np.float32),emission=a,intervention_targets=b,
    )


def same_emission_new_episodes(
    oracle: TrialOracle, *, seed: int, n: int = 300, steps: int = 36,
    observed_fraction: float = .95,
) -> tuple[TrialObservations,TrialOracle]:
    """New control-env patients with SAME hidden data mechanism; never train on them."""
    rng = np.random.default_rng(seed)
    k=oracle.emission.shape[1]
    d=oracle.emission.shape[0]
    latent=np.zeros((n,steps,k),dtype=np.float64)
    latent[:,0]=rng.normal(size=(n,k))
    for t in range(1,steps):
        latent[:,t]=.73*latent[:,t-1]+np.sqrt(1-.73**2)*rng.normal(size=(n,k))
    raw=latent@oracle.emission.T+rng.normal(scale=.12,size=(n,steps,d))
    mask=(rng.random(raw.shape)<observed_fraction).astype(np.float32)
    x=np.where(mask>0,raw,0).astype(np.float32)
    recent=latent[:,-6:,:].mean(axis=1)
    beta=np.where(np.arange(k)%2==0,1.0,-1.0)*np.linspace(.9,1.4,k)
    logit=-.20+recent@beta
    y=(rng.random(n)<1/(1+np.exp(-np.clip(logit,-20,20)))).astype(np.float32)
    return TrialObservations(x,mask,np.zeros(n,dtype=int),y),TrialOracle(
        recent.astype(np.float32),oracle.emission.copy(),oracle.intervention_targets.copy(),
    )


def observable_pca_axis_overlap(train: TrialObservations, oracle: TrialOracle) -> float:
    """Optimistic PCA baseline on ONLY control TRAIN patients, no interventions."""
    k=len(np.unique(train.environment))-1
    x=train.x[train.environment==0,-6:,:]
    m=train.mask[train.environment==0,-6:,:]
    mean=(x*m).sum(axis=1)/np.maximum(m.sum(axis=1),1)
    pca=PCA(n_components=k,svd_solver="full").fit(mean)
    a=np.abs(pca.components_@oracle.emission)
    rows,cols=linear_sum_assignment(-a)
    return float(a[rows,cols].mean())


def oracle_direction_alignment(cert: InterventionCertificate,oracle: TrialOracle)->dict:
    """Evaluation only; never used for rank, fitting, tuning or attribution sign."""
    overlap=np.abs(cert.unit_directions.T@oracle.emission)
    rows,cols=linear_sum_assignment(-overlap)
    return {
        "oracle_target_order_column_cosine":float(np.diag(overlap).mean()),
        "oracle_optimistic_permuted_column_cosine":float(overlap[rows,cols].mean()),
        "true_intervention_mechanism_condition":float(np.linalg.cond(oracle.intervention_targets))
            if np.linalg.matrix_rank(oracle.intervention_targets)==oracle.intervention_targets.shape[1]
            else None,
        "oracle_used_in_estimation":False,
    }


def _score(y:np.ndarray,p:np.ndarray)->dict:
    return {
        "auprc":float(average_precision_score(y,p)),
        "auroc":float(roc_auc_score(y,p)),
    }


def run_study(
    *, seed: int = 20261104, sizes: tuple[int,...]=(3,5,7),
    n_train_per_env: int = 180, n_validation: int = 260,
    train_epochs: int = 40, bootstraps: int = 64,
    observable_response_weight: float = .8,
)->tuple[pd.DataFrame,dict]:
    torch.set_num_threads(1)
    results=[]
    receipts={}
    for k in sizes:
        for regime in ("exclusive","multitarget","duplicated","sham"):
            train,truth=simulate_randomized_intervention_trials(
                k=k,seed=seed+100*k,
                regime=regime,n_per_environment=n_train_per_env,
            )
            # The following list is accepted as proven ONLY under an actual
            # controlled experimental intervention protocol. This metadata
            # cannot be identified from observational contrast matrices.
            verified=(regime=="exclusive")
            cert=estimate_intervention_certificate(
                train.x,train.mask,train.environment,
                exclusive_single_target_provenance=verified,
                bootstraps=bootstraps,seed=seed+k,
            )
            row={"true_k_for_evaluation_only":k,"regime":regime,
                 "individually_identifiable_claim_supported":cert.status.startswith("CERTIFIED"),
                 "certificate_status":cert.status,
                 "empirical_rank":cert.rank_estimate,
                 "smallest_singular_value":cert.min_singular_value,
                 "q90_perturbation":cert.q90_bootstrap_spectral_noise,
                 "worst_bootstrap_direction_cosine":float(cert.bootstrap_q10_direction_cosine.min()),
                 **oracle_direction_alignment(cert,truth),
                 "pca_train_control_only_oracle_axis_cosine_OPTIMISTIC":observable_pca_axis_overlap(train,truth),
            }
            # Provide an intentionally FALSE target-exclusivity assertion to
            # demonstrate that an invertible multi-target design can look
            # strongly identified yet the latent identity conclusion is wrong.
            if regime=="multitarget":
                false_cert=estimate_intervention_certificate(
                    train.x,train.mask,train.environment,
                    exclusive_single_target_provenance=True,bootstraps=bootstraps,seed=seed+k,
                )
                row["counterfactual_FALSE_provenance_status"]=false_cert.status
                row["counterfactual_FALSE_provenance_would_claim_identity"]=false_cert.status.startswith("CERTIFIED")
            if cert.status.startswith("CERTIFIED"):
                validation,valtruth=same_emission_new_episodes(
                    truth,seed=seed+3000+k,n=n_validation,
                )
                # Independently trained models share a train-only anchor.
                # Semantic stability of FIXED definitions is not a learning gain;
                # evaluate risk coefficient/contribution ranking separately.
                contributions=[]
                losses=[]
                weights=[]
                definition_shift=[]
                predictions=[]
                intervention_model=None
                mean=train.x[:,-6:,:].mean(axis=1)
                lr=LogisticRegression(C=1.,max_iter=500).fit(mean,train.y)
                raw=validation.x[:,-6:,:].mean(axis=1)
                row.update({"raw_logistic_control_auprc":float(
                    average_precision_score(validation.y,lr.predict_proba(raw)[:,1])
                )})
                for model_seed in (11,23,37):
                    model,receipt=train_intervention_concept_model(
                        train.x,train.mask,train.y,cert,seed=model_seed,epochs=train_epochs,
                        observable_response_weight=observable_response_weight,
                    )
                    with torch.no_grad():
                        outputs=model(torch.from_numpy(validation.x),torch.from_numpy(validation.mask))
                        p=torch.sigmoid(outputs["logits"]).numpy()
                        c=outputs["concepts"].numpy()
                        phi=outputs["contributions"].numpy()
                        mixing=model.definition.detach().numpy()@valtruth.emission
                    scores=_score(validation.y,p)
                    predictions.append(scores)
                    if model_seed == 11:
                        intervention_model=model
                    contributions.append(phi)
                    weights.append(mixing)
                    losses.append(receipt["train_loss"])
                    definition_shift.append(receipt["definition_movement_from_intervention_initializer"])
                signed=[]
                definitions=[]
                for i,j in itertools.combinations(range(3),2):
                    for slot in range(k):
                        a,b=contributions[i][:,slot],contributions[j][:,slot]
                        signed.append(float(spearmanr(a,b).statistic) if (
                            np.std(a)>1e-7 and np.std(b)>1e-7
                        ) else float("nan"))
                    cos=np.sum(weights[i]*weights[j],axis=1)/(
                        np.linalg.norm(weights[i],axis=1)*np.linalg.norm(weights[j],axis=1)+1e-12
                    )
                    definitions.append(cos)
                if not np.isfinite(signed).all():
                    row["sign_effect_stability_status"]="ABSTAIN_NEAR_CONSTANT"
                else:
                    row["min_signed_same_patient_spearman"] = float(np.min(signed))
                    row["signed_effect_stability_status"]=(
                        "ALL_PAIRS_RHO_GE_0.5" if min(signed)>=.5 else "FAIL_ANY_PAIR_RHO_LT_0.5"
                    )
                # New evaluation-only paired perturbations. The true sensor
                # response loading is never available to the training model;
                # it is used HERE to synthesize counterfactual observed values
                # and independently verify direct concept responsiveness.
                with torch.no_grad():
                    x0=torch.from_numpy(validation.x)
                    m0=torch.from_numpy(validation.mask)
                    base=intervention_model(x0,m0)["concepts"].numpy()
                    delta=np.zeros((k,k),dtype=np.float64)
                    for j in range(k):
                        sensor_shift=(1.5*valtruth.emission[:,j]).astype(np.float32)
                        altered=validation.x.copy()
                        altered[:,-6:,:] += m0[:,-6:,:].numpy()*sensor_shift[None,None,:]
                        counterfactual=intervention_model(torch.from_numpy(altered),m0)["concepts"].numpy()
                        delta[:,j]=np.mean(counterfactual-base,axis=0)
                ratio=(np.abs(delta).sum()-np.abs(np.diag(delta)).sum())/max(np.abs(delta).sum(),1e-12)
                row["eval_only_paired_intervention_offtarget_concept_fraction_seed11"] = float(ratio)
                no_probe_scores=[]
                for ablated_seed in (11,23,37):
                    no_probe,_=train_intervention_concept_model(
                        train.x,train.mask,train.y,cert,seed=ablated_seed,
                        epochs=train_epochs,observable_response_weight=0.,
                    )
                    with torch.no_grad():
                        no_probe_out=no_probe(
                            torch.from_numpy(validation.x),torch.from_numpy(validation.mask)
                        )
                        no_probe_p=torch.sigmoid(no_probe_out["logits"]).numpy()
                    no_probe_scores.append(average_precision_score(validation.y,no_probe_p))
                    if ablated_seed==11:
                        with torch.no_grad():
                            base_no_probe=no_probe(
                                torch.from_numpy(validation.x),torch.from_numpy(validation.mask)
                            )["concepts"].numpy()
                            ablated_delta=np.zeros((k,k),dtype=np.float64)
                            for j in range(k):
                                altered=validation.x.copy()
                                sensor_shift=(1.5*valtruth.emission[:,j]).astype(np.float32)
                                altered[:,-6:,:] += validation.mask[:,-6:,:]*sensor_shift[None,None,:]
                                shifted=no_probe(
                                    torch.from_numpy(altered),torch.from_numpy(validation.mask)
                                )["concepts"].numpy()
                                ablated_delta[:,j]=np.mean(shifted-base_no_probe,axis=0)
                            row["eval_only_paired_intervention_offtarget_fraction_NO_probe_seed11"]=(
                                float((np.abs(ablated_delta).sum()
                                       -np.abs(np.diag(ablated_delta)).sum())/
                                      max(np.abs(ablated_delta).sum(),1e-12))
                            )
                row["mean_auprc_native_NO_response_probe"] = float(np.mean(no_probe_scores))
                row.update({
                    "mean_auprc_new_patients":float(np.mean([a["auprc"] for a in predictions])),
                    "mean_auroc_new_patients":float(np.mean([a["auroc"] for a in predictions])),
                    "mean_train_loss":float(np.mean(losses)),
                    "mean_trainable_definition_movement":float(np.mean(definition_shift)),
                    "mean_cross_seed_learned_definition_cosine":float(np.mean(definitions)),
                    "mean_offtarget_mixing_ratio":float(np.mean([
                        (np.abs(w).sum()-np.abs(np.diag(w)).sum()) / max(np.abs(w).sum(),1e-12)
                        for w in weights
                    ])),
                    "n_train_model_seeds":3,
                    "train_response_equivariance_weight":observable_response_weight,
                    "no_oracle_latent_truth_in_model_fit":True,
                })
            receipts[f"K{k}_{regime}"]=cert.to_json()
            results.append(row)
            print(f"K={k} regime={regime} status={cert.status} "
                  f"target_cos={row['oracle_target_order_column_cosine']:.3f} "
                  f"pca_axis_cos={row['pca_train_control_only_oracle_axis_cosine_OPTIMISTIC']:.3f}"
                  ,flush=True)
    return pd.DataFrame(results),receipts


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    if args.out_dir.exists():
        p.error("Do not overwrite a previous experimental artifact")
    import yaml
    cfg=yaml.safe_load(args.config.read_text())
    if (not cfg["protocol"]["synthetic_only_no_patient_data"]
            or not cfg["protocol"]["no_oracle_in_training"]
            or cfg["evaluation"]["regimes"] != ["exclusive","multitarget","duplicated","sham"]):
        p.error("Predeclared intervention falsification protocol invalid")
    table,certificates=run_study(
        seed=int(cfg["data"]["seed"]),sizes=tuple(cfg["data"]["k_true"]),
        n_train_per_env=int(cfg["data"]["n_train_per_environment"]),
        n_validation=int(cfg["data"]["n_holdout_control"]),
        train_epochs=int(cfg["training"]["epochs"]),
        bootstraps=int(cfg["certificate"]["patient_bootstraps"]),
        observable_response_weight=float(cfg["training"]["observable_response_weight"]),
    )
    args.out_dir.mkdir(parents=True,exist_ok=False)
    table.to_csv(args.out_dir/"V13_INTERVENTION_EXCLUSIVITY_STRESS.csv",index=False)
    (args.out_dir/"TRAIN_ONLY_INTERVENTION_CERTIFICATES.json").write_text(
        json.dumps(certificates,indent=2,allow_nan=False))
    status={
        "status":"V13_CONDITIONAL_INTERVENTION_IDENTIFIABILITY_SYNTHETIC_ONLY",
        "n_scenarios":len(table),
        "no_clinical_data_accessed":True,
        "interventions_are_experimental_ASSUMPTIONS_not_verified_from_data":True,
        "model_native_concepts_not_posthoc":True,
        "original_Piere_V11_V12_evidence_unchanged":True,
        "results":table.astype(object).where(pd.notna(table),None).to_dict("records"),
    }
    (args.out_dir/"STATUS.json").write_text(json.dumps(status,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
