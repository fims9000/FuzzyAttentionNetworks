"""V13 exogenous-action identification: new synthetic generator seeds.

No access to PhysioNet, Med-CircuitBench, existing V10/V12 checkpoints or
earlier Set B holdout. True latent trajectories and mixing are evaluation-only.
Three independent 800-of-1000 TRAIN subsamples (without replacement) and 300
distinct same-seed validation cases for each factor-count regime.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

from .instrumented_v13 import (
    InterventionProtocol, identify_action_anchored_concepts,
    fit_instrumented_bottleneck,
)
from .intervention_synthetic_v13 import make_intervention_scm

FACTOR_COUNTS=(3,5,7)
TRAINING_SEEDS=(11,23,37)
REGISTERED_GENERATOR_SEEDS=(20261104,20261204)

def rho(x,y):
    if np.std(x)<1e-8 or np.std(y)<1e-8:
        return float("nan")
    return float(spearmanr(x,y).statistic)

def baseline_pca_oracle(train,validation,oracle,k):
    n,t,d=train.shape
    p=PCA(n_components=k,svd_solver="full").fit(train.reshape(-1,d))
    a=p.transform(validation.reshape(-1,d)).reshape(len(validation),t,k)[:,-4:].mean(1)
    b=oracle[:,-4:].mean(1)
    corr=np.array([[rho(a[:,i],b[:,j]) for j in range(k)] for i in range(k)])
    i,j=linear_sum_assignment(-np.abs(corr))
    return float(np.abs(corr[i,j]).mean())

def run(output_dir:Path,*,n_total:int=1300,epochs:int=85,
        generator_seed_base:int=20261104):
    if generator_seed_base not in REGISTERED_GENERATOR_SEEDS:
        raise ValueError("Unknown generator seed, refusing unregistered adaptation")
    if output_dir.exists():
        raise FileExistsError("Cannot overwrite prior scientific audit")
    if n_total!=1300 or epochs!=85:
        raise ValueError("Registered cohort and optimizer settings are frozen")
    rows,pairs,negatives=[],[],[]
    for k in FACTOR_COUNTS:
        d=k+3
        data=make_intervention_scm(n=n_total,k=k,d=d,
            seed=generator_seed_base+100*k,history_steps=12,horizon=2,
            action_mode="randomized")
        x=data.observations[:,:12].astype(np.float32)
        u=data.actions[:,:11].astype(np.float32)
        y=data.labels.astype(np.float32)
        protocol=InterventionProtocol(True,True,True,True,
            tuple(f"randomized_target_{i}" for i in range(k)))
        ids=np.random.default_rng(20261120+k).permutation(len(x))
        train,validation=ids[:1000],ids[1000:]
        if len(validation)!=300 or np.unique(y[validation]).size!=2:
            raise ValueError("Incorrect sealed validation split")
        # Evaluation-only optimistic PCA oracle reference.
        pca_rho=baseline_pca_oracle(x[train],x[validation],
            data.oracle_factors[validation,:12],k)
        fits={}
        for seed in TRAINING_SEEDS:
            selected=np.random.default_rng(seed+1000*k).choice(
                train,size=800,replace=False)
            cert=identify_action_anchored_concepts(
                x[selected],u[selected],protocol,seed=seed+3*k,
                permutations=199,bootstraps=48)
            model,receipt=fit_instrumented_bottleneck(
                cert,x[selected],y[selected],seed=seed,epochs=epochs)
            with torch.no_grad():
                out=model(torch.from_numpy(x[validation]))
                score=out["probabilities"].numpy()
                effects=out["contributions"].numpy()
                anchors=out["anchors"].numpy()
                beta=(3*torch.tanh(model.raw_beta)).numpy()
                exact=float((out["logits"]-effects.sum(-1)-model.bias).abs().max())
            if exact>1e-6:
                raise AssertionError("Native concept-only path failed")
            # Hidden oracle is read ONLY after model fit and score creation.
            oracle=data.oracle_factors[validation,:12]
            same=[rho(anchors[:,-4:,j].mean(1),oracle[:,-4:,j].mean(1))
                  for j in range(k)]
            true=data.planted_mixing
            true=true/np.linalg.norm(true,axis=0)
            signature_cos=np.sum(cert.signature*true,axis=0)
            rows.append({
                "true_factor_count_evaluation_only":k,"seed":seed,
                "n_train_used":len(selected),"n_heldout_evaluation":len(validation),
                "mean_signed_oracle_factor_spearman":float(np.mean(same)),
                "min_signed_oracle_factor_spearman":float(np.min(same)),
                "min_true_response_signature_cosine":float(np.min(signature_cos)),
                "min_bootstrap_response_orientation_cosine_q10":float(
                    min(cert.worst_bootstrap_oriented_cosine_q10)),
                "pretreatment_placebo_p":cert.pretreatment_balance_permutation_p,
                "auprc":float(average_precision_score(y[validation],score)),
                "auroc":float(roc_auc_score(y[validation],score)),
                "brier":float(brier_score_loss(y[validation],score)),
                "exact_logit_error":exact,
                "pca_optimistic_hungarian_oracle_abs_spearman":pca_rho,
                "true_factor_oracle_used_for_training":False,
                "source_mixing_used_for_training":False,
                "causal_patient_risk_interpretation":False,
            })
            fits[seed]={"effect":effects,"signature":cert.signature,"beta":beta}
            print("K",k,"seed",seed,"AUPRC",round(rows[-1]["auprc"],5),
                  "signed oracle rho",round(rows[-1]["mean_signed_oracle_factor_spearman"],5),
                  flush=True)
        for a,b in itertools.combinations(TRAINING_SEEDS,2):
            for j in range(k):
                left,right=fits[a],fits[b]
                pairs.append({
                    "k":k,"slot":j,"seed_a":a,"seed_b":b,
                    "action_signature_cosine":float(np.dot(left["signature"][:,j],
                                                           right["signature"][:,j])),
                    "signed_effect_spearman":rho(left["effect"][:,j],right["effect"][:,j]),
                    "both_beta_nonzero":bool(
                        min(abs(left["beta"][j]),abs(right["beta"][j]))>.08),
                    "same_beta_sign":bool(np.sign(left["beta"][j])==np.sign(right["beta"][j])),
                    "oracle_matched_to_align_effects":False,
                })
        for bad_mode in ("confounded","collinear","none"):
            bad=make_intervention_scm(n=320,k=k,d=d,seed=7121+k,
                history_steps=12,action_mode=bad_mode,mixing=data.planted_mixing)
            abstained=False
            reason=""
            try:
                identify_action_anchored_concepts(bad.observations[:,:12],
                    bad.actions[:,:11],protocol,seed=151+k,
                    permutations=99,bootstraps=24)
            except ValueError as exc:
                abstained=True
                reason=str(exc)
            negatives.append({"k":k,"negative_control":bad_mode,
                              "abstained":abstained,"reason":reason})
            if not abstained:
                raise AssertionError("Failed negative control: "+bad_mode)
    frame=pd.DataFrame(rows)
    pair_frame=pd.DataFrame(pairs)
    controls=pd.DataFrame(negatives)
    slot_rows=[]
    for k in FACTOR_COUNTS:
        x=pair_frame[pair_frame.k==k].groupby("slot").agg(
            min_signature_cosine=("action_signature_cosine","min"),
            min_signed_effect_spearman=("signed_effect_spearman","min"),
            all_beta_nonzero=("both_beta_nonzero","all"),
            all_beta_signs_agree=("same_beta_sign","all"),
        ).reset_index()
        x["k"]=k
        x["strict_stable_individual_effect"]=(
            (x.min_signature_cosine>.9)&(x.min_signed_effect_spearman>.5)
            &x.all_beta_nonzero&x.all_beta_signs_agree)
        slot_rows.append(x)
    slots=pd.concat(slot_rows,ignore_index=True)
    summary=frame.groupby("true_factor_count_evaluation_only").agg(
        mean_auprc=("auprc","mean"),mean_auroc=("auroc","mean"),
        mean_signed_oracle_factor_spearman=("mean_signed_oracle_factor_spearman","mean"),
        minimum_true_signature_cosine=("min_true_response_signature_cosine","min"),
        pca_optimistic_oracle_abs_spearman=("pca_optimistic_hungarian_oracle_abs_spearman","mean")
    ).reset_index()
    summary["stable_individual_effect_count"]=[
        int(slots[slots.k==k].strict_stable_individual_effect.sum())
        for k in FACTOR_COUNTS
    ]
    status={
        "status":"CONDITIONAL_SYNTHETIC_INTERVENTION_V13_COMPLETE",
        "generator_seed_base":generator_seed_base,
        "stage":"new_generator_confirmation" if generator_seed_base==20261204 else "exploratory",
        "registered_train_size":1000,"registered_dev_size":300,
        "per_run_training_subsample_without_replacement":800,
        "three_independent_model_initializations":True,
        "independently_known_action_target_dictionary":True,
        "oracle_latent_factors_and_mixing_used_only_for_postfit_evaluation":True,
        "all_9_invalid_intervention_controls_rejected":bool(controls.abstained.all()),
        "no_physionet_or_previous_holdout_access":True,
        "not_a_clinical_or_observational_identifiability_claim":True,
        "limits":[
            "Randomized independent and singly targeted actions imposed by generator",
            "Source scale unidentified under unknown action gain",
            "PCA matching is evaluation-only and optimistic",
            "Native signed prediction contribution is not causal treatment effect",
            "Same generator family and three fits; no broad distributional guarantee",
        ],
        "results":summary.to_dict("records"),
    }
    output_dir.mkdir(parents=True,exist_ok=False)
    frame.to_csv(output_dir/"PER_SEED_AGGREGATES.csv",index=False)
    pair_frame.to_csv(output_dir/"PAIRWISE_SIGNED_EFFECTS.csv",index=False)
    controls.to_csv(output_dir/"NEGATIVE_CONTROLS.csv",index=False)
    slots.to_csv(output_dir/"SIGNED_EFFECT_CERTIFICATES.csv",index=False)
    summary.to_csv(output_dir/"GROUP_SUMMARY.csv",index=False)
    (output_dir/"STATUS.json").write_text(json.dumps(status,indent=2,allow_nan=False))
    print(summary.to_string(index=False),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",required=True,type=Path)
    p.add_argument("--epochs",type=int,default=85)
    p.add_argument("--generator-seed-base",type=int,default=20261104)
    a=p.parse_args()
    run(a.out_dir,epochs=a.epochs,generator_seed_base=a.generator_seed_base)

if __name__=="__main__":
    main()
