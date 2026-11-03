"""New-seed synthetic oracle falsification of V12, with 3/5/7 factors.

Generator oracle states are explicitly removed from the training object.
Train-only permutation-calibrated rank and bootstrap structural groups choose
K WITHOUT the known synthetic ground-truth count. The planted latent states
are read ONLY after all matched models for a regime are fit. This does not
validate clinical semantics or independently prove a mathematical theorem.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from .clinical_identifiability_v11 import _definition_group_overlap
from .clinical_signed_effect_v12 import predict_native
from .clinical_v10_external_benchmark import _model_score
from .effect_certification_v11 import joint_concept_screen
from .identifiability_v11 import (
    compare_group_attributions,fit_predictive_group_certificate,projector_overlap,
)
from .lagged_rank import fit_lagged_predictive_rank
from .magnets_binary_baseline_v12 import (
    fit_magnets_binary_adaptation,
    MaskAggregateBinaryConcepts,
)
from .multiscale_physiology_v10 import expanded_multiscale_definition
from .signed_effect_v12 import fit_train_only_signed_anchors
from .train_identifiability_v11 import fit_identifiability_aware_v11
from .train_signed_effect_v12 import fit_signed_effect_v12
from .variable_factor_data import generate_variable_factor_data


NATIVE=("v11_projector_group_view","v12_observable_anchor","v12_anchor_signed_effect")
METHODS=NATIVE+("magnets_style_binary_adaptation",)
SEEDS=(11,23,37)


def _prediction_and_concepts(model,values:np.ndarray,mask:np.ndarray,method:str):
    model.eval()
    probabilities,concepts,contributions=[],[],[]
    with torch.no_grad():
        for i in range(0,len(values),128):
            x=torch.from_numpy(values[i:i+128])
            m=torch.from_numpy(mask[i:i+128])
            if method in NATIVE:
                o=model(x,m,process_mask=m)
                if float(o.decomposition_error())>1e-5:
                    raise AssertionError("Invalid native concept-only additive logit")
                p=o.probabilities.cpu().numpy()
                c=o.physical.concept_summary.cpu().numpy()
                e=o.physical.contributions.cpu().numpy()
            else:
                o=model(x,m,m)
                if float(o.decomposition_error(model.decision_bias))>1e-5:
                    raise AssertionError("MAGNETS-style logit bypass detected")
                p=torch.sigmoid(o.logits).cpu().numpy()
                c=o.concepts.cpu().numpy()
                e=o.contributions.cpu().numpy()
            probabilities.append(p)
            concepts.append(c)
            contributions.append(e)
    return (np.concatenate(probabilities).astype(np.float64),
            np.concatenate(concepts).astype(np.float64),
            np.concatenate(contributions).astype(np.float64))


def optimistic_oracle_match(
    learned:np.ndarray,oracle:np.ndarray,*,seed:int,
)->dict:
    """Evaluation-only optimal assignment, correctly labelled optimistic.

    Report a permuted-patient oracle control with the same optimistic matching
    so random correlations cannot masquerade as recovered semantic concepts.
    """
    a=np.asarray(learned,dtype=np.float64)
    b=np.asarray(oracle,dtype=np.float64)
    if (a.ndim!=2 or b.ndim!=2 or a.shape[0]!=b.shape[0] or
        a.shape[0]<30 or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("Matched validation cases and finite concept arrays required")
    def corrs(left,right):
        ans=np.zeros((left.shape[1],right.shape[1]),dtype=np.float64)
        for j in range(left.shape[1]):
            for k in range(right.shape[1]):
                if np.std(left[:,j])>1e-9 and np.std(right[:,k])>1e-9:
                    v=float(spearmanr(left[:,j],right[:,k]).statistic)
                    ans[j,k]=v if np.isfinite(v) else 0.
        row,col=linear_sum_assignment(-np.abs(ans))
        return np.asarray([ans[i,j] for i,j in zip(row,col)]),list(map(int,row)),list(map(int,col))
    observed,row,col=corrs(a,b)
    rng=np.random.default_rng(seed)
    null,_,_=corrs(a,b[rng.permutation(len(b))])
    return {
        "observed_mean_abs_spearman_OPTIMISTIC":float(np.mean(np.abs(observed))),
        "shuffled_patient_oracle_mean_abs_spearman_OPTIMISTIC_NULL":float(
            np.mean(np.abs(null))
        ),
        "matched_signed_spearman":observed.tolist(),
        "assigned_learned_slots_evaluation_only":row,
        "assigned_true_factor_slots_evaluation_only":col,
        "n_true_factors":b.shape[1],
        "n_learned_slots":a.shape[1],
        "rank_matches_true_oracle_count":a.shape[1]==b.shape[1],
        "oracle_used_in_optimization":False,
        "independent_validity_cannot_be_inferred_from_optimal_matching":True,
    }


def run(a):
    cfg=yaml.safe_load(a.config.read_text())
    ref=yaml.safe_load(a.reference_v6_config.read_text())
    v7=yaml.safe_load(a.reference_v7_config.read_text())
    v10=yaml.safe_load(a.reference_v10_config.read_text())
    v11=yaml.safe_load(a.reference_v11_config.read_text())
    v12=yaml.safe_load(a.reference_v12_config.read_text())
    if a.out_dir.exists():
        raise FileExistsError("Previous synthetic oracle audit must never be overwritten")
    if not (cfg["protocol"]["latent_factor_labels_for_evaluation_only"]
            and cfg["protocol"]["no_oracle_parameters_in_optimizer"]
            and cfg["protocol"]["no_validation_outcome_in_selection"]
            and tuple(cfg["data"]["k_true"])==(3,5,7)
            and tuple(cfg["evaluation"]["methods"])==METHODS
            and tuple(cfg["evaluation"]["seeds"])==SEEDS
            and cfg["protocol"]["synthetic_only_no_clinical_data"]):
        raise ValueError("Frozen new-seed 3/5/7 oracle contract violated")
    torch.set_num_threads(int(ref["training"]["cpu_threads"]))
    rows=[]
    predictions={}
    group_pairs=[]
    group_screen={}
    source_receipts={}
    for true_k in (3,5,7):
        d=cfg["data"]
        data=generate_variable_factor_data(
            n_true_factors=true_k,
            seed=int(d["generator_seed_base"])+true_k,
            split_seed=int(d["split_seed_base"])+true_k,
            n_samples=int(d["n_samples"]),n_channels=int(d["channels"]),
            input_steps=int(d["history_hours"]),horizon=int(d["forecast_hours"]),
        )
        # The only path to latent oracle truth is explicitly reopened LATER.
        oracle=data.validation.oracle_states
        safe_train=replace(data.train,oracle_states=None)
        safe_val=replace(data.validation,oracle_states=None)
        data=replace(data,train=safe_train,validation=safe_val)
        r=cfg["rank"]
        rank=fit_lagged_predictive_rank(
            data.train.x,data.train.mask,
            lags=tuple(r["lags"]),permutations=int(r["permutations"]),
            patient_bootstraps=int(r["patient_bootstraps"]),
            alpha=float(r["alpha"]),seed=int(r["seed_base"])+true_k,
        )
        k=min(int(rank.rank),int(r["max_computational_slots"]))
        if k<2:
            rows.append({
                "true_oracle_factor_count":true_k,
                "detected_train_only_predictive_rank":k,
                "decision":"ABSTAIN_NO_MULTISLOT_COMPUTATIONAL_IDENTIFIABILITY",
            })
            source_receipts[str(true_k)]={"status":"ABSTAIN_RANK_TOO_LOW",
                                          "rank":rank.to_json()}
            continue
        c=cfg["certificate"]
        certificate=fit_predictive_group_certificate(
            data.train.x,data.train.mask,computational_rank=k,
            lags=tuple(r["lags"]),bootstraps=int(c["patient_bootstraps"]),
            seed=int(c["seed_base"])+true_k,
            minimum_subspace_overlap=float(c["min_subspace_overlap"]),
            gap_noise_quantile=float(c["gap_noise_quantile"]),
        )
        if projector_overlap(
            rank.left_observable_loadings[:,:k],
            certificate.reference_basis[:,:k],
        )<.999:
            raise AssertionError("New synthetic source rank/certificate mismatch")
        h=v12["source_anchors"]
        anchors=fit_train_only_signed_anchors(
            data.train.x,data.train.mask,data.train.y,certificate,
            bootstraps=int(h["patient_bootstraps"]),
            permutations=int(h["train_only_label_permutations"]),
            seed=int(h["seed"])+true_k,
            familywise_alpha=float(h["bonferroni_familywise_alpha"]),
            min_absolute_class_difference=float(h["minimum_absolute_train_class_difference"]),
            history_steps=int(v12["training"]["anchor_history_hours"]),
        )
        source_receipts[str(true_k)]={
            "train_only_rank":rank.to_json(),
            "train_only_groups":certificate.to_json(),
            "train_only_signs":anchors.to_json(),
            "n_training_examples":len(data.train.y),
            "n_heldout_validation_examples":len(data.validation.y),
            "latent_oracle_states_present_in_training_object":False,
        }
        members={}
        definitions={}
        model_outcomes={}
        for method in METHODS:
            members[method],definitions[method],model_outcomes[method]={},{},{}
            for seed in SEEDS:
                before=time.perf_counter()
                if method=="v11_projector_group_view":
                    model,train=fit_identifiability_aware_v11(
                        data,certificate.reference_basis[:,:k],certificate,
                        seed,ref,v7,v10,v11,variant=method,
                    )
                elif method in NATIVE:
                    model,train=fit_signed_effect_v12(
                        data,anchors.basis,certificate,anchors,seed,
                        ref,v7,v10,v11,v12,variant=method,
                    )
                else:
                    model,train=fit_magnets_binary_adaptation(
                        data.train,seed,n_concepts=k,
                        epochs=int(ref["training"]["epochs"]),
                        batch_size=int(ref["training"]["batch_size"]),
                        learning_rate=float(ref["training"]["learning_rate"]),
                        weight_decay=float(ref["training"]["weight_decay"]),
                        augmentation_fraction=float(ref["training"]["main_mask_fraction"]),
                        n_masks=int(v12["training"]["mask_aggregate_architecture"]["n_masks"]),
                    )
                elapsed=time.perf_counter()-before
                p,c,e=_prediction_and_concepts(
                    model,data.validation.x,data.validation.mask,method,
                )
                members[method][seed]=e
                if method in NATIVE:
                    definitions[method][seed]=expanded_multiscale_definition(model)
                # Oracle is opened for EVALUATION ONLY, after optimizer.run.
                known=oracle[:,-6:,:].mean(axis=1)
                match=optimistic_oracle_match(c,known,seed=20261211+true_k*100+seed)
                rows.append({
                    "method":method,"seed":seed,
                    "true_oracle_factor_count":true_k,
                    "detected_train_only_predictive_rank":k,
                    "source_singletons_train_only":int(anchors.singleton.sum()),
                    "train_risk_signs_noncausal":int(np.count_nonzero(anchors.risk_sign)),
                    "train_seconds":elapsed,
                    **_model_score(data.validation.y,p),
                    **match,
                    "heldout_synthetic_validation_only":True,
                    "oracle_used_in_training":False,
                    "clinical_concept_meaning_proven":False,
                })
                model_outcomes[method][seed]=p
                print("SYNTHETIC V12 ORACLE",true_k,"detected",k,method,seed,
                      "validation AUPRC",round(_model_score(data.validation.y,p)["auprc"],5),
                      "optimistic oracle rho",round(match["observed_mean_abs_spearman_OPTIMISTIC"],4),
                      flush=True)
        raw=[]
        for method in NATIVE:
            for sa,sb in itertools.combinations(SEEDS,2):
                effects=compare_group_attributions(
                    members[method][sa],members[method][sb],certificate.groups,
                )
                definition=_definition_group_overlap(
                    definitions[method][sa],definitions[method][sb],certificate,
                )
                for e,f in zip(effects,definition):
                    raw.append({
                        "method":method,"seed_a":sa,"seed_b":sb,**e,
                        "definition_subspace_overlap":f["definition_subspace_overlap"],
                        "oracle_true_k_hidden_during_train":true_k,
                    })
        native=pd.DataFrame(raw)
        screen=joint_concept_screen(
            certificate.to_json(),native,expected_seeds=SEEDS,
            already_repeatedly_inspected_development=False,
            min_every_pair_spearman=float(
                v12["evaluation"]["signed_pairwise_min_spearman"]
            ),
            min_every_pair_definition_overlap=float(
                v12["evaluation"]["learned_definition_min_overlap"]
            ),
        )
        group_pairs.extend(raw)
        group_screen[str(true_k)]=screen
    a.out_dir.mkdir(parents=True,exist_ok=False)
    frame=pd.DataFrame(rows)
    frame.to_csv(a.out_dir/"SYNTHETIC_NEW_SEED_3_5_7_ORACLE_PER_RUN.csv",index=False)
    pd.DataFrame(group_pairs).to_csv(
        a.out_dir/"SYNTHETIC_NEW_SEED_SIGNED_GROUP_PAIRS.csv",index=False
    )
    (a.out_dir/"TRAIN_ONLY_3_5_7_RANK_AND_SOURCE.json").write_text(json.dumps(
        source_receipts,indent=2,allow_nan=False
    ),encoding="utf-8")
    (a.out_dir/"NEW_SEED_GROUP_FUNCTIONAL_SCREEN.json").write_text(json.dumps(
        group_screen,indent=2,allow_nan=False
    ),encoding="utf-8")
    status={
        "status":"V12_NEW_SEED_VARIABLE_FACTOR_3_5_7_ORACLE_AUDIT_COMPLETE",
        "no_real_physionet_or_set_b_loaded":True,
        "true_latent_states_never_entered_optimizer_or_train_loader":True,
        "oracle_comparisons_are_optimistically_hungarian_matched_at_evaluation":True,
        "clinical_causality_not_demonstrated":True,
        "n_factor_regimes":3,
        "n_model_training_seeds":len(SEEDS),
        "rank_selected_without_planted_ground_truth_count":True,
        "no_patient_records_in_outputs":True,
        "results":frame.groupby(
            ["true_oracle_factor_count","method"],dropna=False
        ).agg(mean_auprc=("auprc","mean"),
              mean_optimal_abs_oracle_rho=("observed_mean_abs_spearman_OPTIMISTIC","mean"),
              mean_shuffled_oracle_rho=("shuffled_patient_oracle_mean_abs_spearman_OPTIMISTIC_NULL","mean")).reset_index().astype(
                  object
              ).where(lambda x:pd.notna(x),None).to_dict("records"),
        "limitations":[
            "Synthetic linear/nonlinear simulator is not a hospital",
            "Known oracle used for optimistic Hungarian alignment at evaluation only",
            "Single generator family, three seeds; evidence descriptive",
            "Matching rank to planted K does not prove nonlinear identifiability",
            "Observable source rank may be different from true latent factor count",
        ],
    }
    (a.out_dir/"STATUS.json").write_text(json.dumps(
        status,indent=2,allow_nan=False
    ),encoding="utf-8")
    print(pd.DataFrame(status["results"]).to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    for key in ("config","reference-v6-config","reference-v7-config",
                "reference-v10-config","reference-v11-config",
                "reference-v12-config","out-dir"):
        p.add_argument("--"+key,required=True,type=Path)
    run(p.parse_args())


if __name__=="__main__":
    main()
