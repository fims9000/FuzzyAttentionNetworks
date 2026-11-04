"""V22: compare PCA versus TRAIN-environment source axes *inside* ConceptFAN.

Only measured data and external TRAIN environment IDs are used to propose
source axes. Source truth is loaded after every predeclared native fit and
used solely for evaluation. This is NOT a post-hoc explanation model;
both arms are actual native bottleneck models trained with one shared V10
forecast/denoise/risk objective and identical measured whitened inputs.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,roc_auc_score,brier_score_loss,
)

from .mixed_source_data_v22 import (
    FAMILIES,SEEDS,SIZES,HISTORY,MODEL_CHANNELS,
    make_observable_v22,train_only_whitening,
    build_native_v22_data,load_oracle_v22_validation_for_postfit_audit,
)
from .environmental_unmixing_v22 import (
    fit_train_only_environment_source_directions,
    native_basis_from_separation,separation_passport,
)
from .experiment_native_risk_v18 import _config
from .clinical_multiscale_v10 import fit_multiscale


MODEL_SEEDS=(11,23,37)
ARMS=("pooled_pca_basis","environment_contrast_basis")
TRAIN_EPOCHS=10
BATCH=96


def _rho(x,y):
    r=spearmanr(x,y).statistic
    return float(r) if np.isfinite(r) else 0.


def _oracle_alignment_no_training_feedback(scores,physical,instrument,environment):
    # This function must be called ONLY AFTER every V22 model fit and
    # never supply its alignment/permutation to training or model selection.
    source=np.column_stack([physical,instrument])
    matrix=np.asarray([
        [abs(_rho(scores[:,i],source[:,j])) for j in range(2)]
        for i in range(2)
    ])
    a,b=linear_sum_assignment(-matrix)
    pairs=[float(matrix[i,j]) for i,j in zip(a,b)]
    by_env={}
    for e in (0,1):
        ix=environment==e
        matches=[
            abs(_rho(scores[ix,i],source[ix,j]))
            for i,j in zip(a,b)
        ]
        by_env[f"environment_{e}_oracle_alignment_eval_only"]=float(np.mean(matches))
    return {
        "mean_abs_oracle_source_spearman_eval_only":float(np.mean(pairs)),
        "worst_of_two_oracle_source_abs_spearman_eval_only":float(np.min(pairs)),
        "best_oracle_permutation_eval_only":[int(j) for j in b],
        "frozen_assignment_never_used_in_model":True,
        **by_env,
    }


def _native_readout(model,split):
    model.eval()
    pred=[]
    concept=[]
    effects=[]
    max_error=0.
    with torch.no_grad():
        for start in range(0,len(split.y),128):
            end=min(start+128,len(split.y))
            x=torch.from_numpy(split.x[start:end])
            m=torch.from_numpy(split.mask[start:end])
            out=model(x,m,process_mask=m)
            estimated=out.physical.raw_activation[:,-6:,:]
            support=out.physical.availability[:,-6:,:]
            r=(estimated*support).sum(1)/support.sum(1).clamp_min(1e-8)
            concept.extend(r.cpu().numpy().tolist())
            effects.extend(out.physical.contributions.cpu().numpy().tolist())
            pred.extend(out.probabilities.cpu().numpy().tolist())
            max_error=max(max_error,float(out.decomposition_error()))
    if max_error>2e-5:
        raise AssertionError("Source-bottleneck native logit is not exactly additive")
    return {
        "validation_auprc":float(average_precision_score(split.y,pred)),
        "validation_auroc":float(roc_auc_score(split.y,pred)),
        "validation_brier":float(brier_score_loss(split.y,pred)),
        "max_native_logit_decomposition_error":max_error,
        "average_absolute_native_concept_effect":float(np.abs(effects).mean()),
    },np.asarray(concept),np.asarray(effects)


def run(outdir:Path):
    if outdir.exists():
        raise FileExistsError("V22 evidence must not overwrite a previous completed run")
    torch.set_num_threads(1)
    source_receipts={}
    all_opportunities=[]
    train_fits=[]
    raw_model_vectors={}
    independent_oracle_alignment={}
    for family in FAMILIES:
        observed=make_observable_v22(family)
        whitener=train_only_whitening(observed)
        data=build_native_v22_data(observed,whitener)
        assert data.train.oracle_states is None and data.validation.oracle_states is None
        evidence=fit_train_only_environment_source_directions(
            observed,whiten=whitener,
            seed=522221,permutations=199,bootstraps=48,
        )
        source_receipts[family]=separation_passport(evidence)
        print(family,
              "TRAIN-only source status",evidence.status,
              "gap",round(evidence.contrast_eigen_gap_standardized,4),
              "subspace stability",round(evidence.environment_sensor_subspace_min_cosine,4),
              flush=True)
        original=np.eye(len(MODEL_CHANNELS),2,dtype=np.float64)
        arms={"pooled_pca_basis":original}
        if evidence.status=="ACCEPT_OBSERVATIONAL_ENVIRONMENT_UNMIXING":
            arms["environment_contrast_basis"]=native_basis_from_separation(evidence)
        else:
            arms["environment_contrast_basis"]=None

        config,dual,v10=_config()
        config["training"]["epochs"]=TRAIN_EPOCHS
        config["training"]["batch_size"]=BATCH
        raw_model_vectors[family]={arm:[] for arm in ARMS}
        fit_records=[]
        for seed in MODEL_SEEDS:
            for arm in ARMS:
                basis=arms[arm]
                if basis is None:
                    all_opportunities.append({
                        "family":family,"seed":seed,"arm":arm,
                        "status":evidence.status,
                        "trained":False,
                        "validation_auprc":None,
                        "validation_auroc":None,
                        "validation_brier":None,
                        "max_native_logit_decomposition_error":None,
                        "average_absolute_native_concept_effect":None,
                        "oracle_source_alignment_eval_only":None,
                        "physically_validated_source":False,
                    })
                    continue
                model,receipt=fit_multiscale(
                    data,basis,seed,config,dual,v10,
                    consistency=False,feature_exploration=False,
                    expert_feedback=None,
                    native_head_refit=False,
                )
                statistics,concept,effects=_native_readout(model,data.validation)
                if model.physical.n_slots!=2:
                    raise AssertionError("Changed intrinsic source bottleneck capacity")
                row={
                    "family":family,"seed":seed,"arm":arm,
                    "status":"TRAINED","trained":True,
                    "training_epochs":TRAIN_EPOCHS,
                    "source_separation_status":evidence.status,
                    "concepts_form_the_real_native_risk_path":True,
                    "fixed_training_target_forecast_and_denoising_loss":True,
                    "both_arms_same_train_only_whitened_input":True,
                    "oracle_source_alignment_eval_only":None,
                    "physically_validated_source":False,
                    **statistics,
                }
                all_opportunities.append(row)
                fit_records.append((row,concept))
                raw_model_vectors[family][arm].append(effects)
                print("  ",arm,seed,
                      "AUPRC",round(statistics["validation_auprc"],5),
                      flush=True)

        # No oracle exists in the source/evidence/model input APIs. Only
        # AFTER all eligible native model fits use the synthetic truth to
        # score source identity without fine-tuning/aligning trained models.
        truth=load_oracle_v22_validation_for_postfit_audit(family)
        true_physical=truth["generated_physical_source"][:,-6:].mean(1)
        true_artifact=truth["generated_instrument_source"][:,-6:].mean(1)
        val_env=observed.validation_environment
        z=data.validation.x[:,-6:,:2].mean(1)
        pca_basis=original[:2,:]
        raw_scores={
            "pooled_pca_basis":z@pca_basis,
            "environment_contrast_basis":(
                z@evidence.source_vectors_whitened
                if arms["environment_contrast_basis"] is not None else None
            ),
        }
        for arm,score in raw_scores.items():
            independent_oracle_alignment[f"{family}__{arm}"]=(
                _oracle_alignment_no_training_feedback(
                    score,true_physical,true_artifact,val_env,
                ) if score is not None else None
            )
        for row,scores in fit_records:
            row["oracle_source_alignment_eval_only"]=_oracle_alignment_no_training_feedback(
                scores,true_physical,true_artifact,val_env,
            )
        del truth,true_physical,true_artifact,fit_records

    summary=[]
    for family in FAMILIES:
        for arm in ARMS:
            entries=[
                row for row in all_opportunities
                if row["family"]==family and row["arm"]==arm
            ]
            completed=[x for x in entries if x["status"]=="TRAINED"]
            signed_vectors=raw_model_vectors[family][arm]
            per_source_signed_rank=[]
            for i in range(len(signed_vectors)):
                for j in range(i+1,len(signed_vectors)):
                    for source in (0,1):
                        per_source_signed_rank.append(
                            _rho(signed_vectors[i][:,source],
                                 signed_vectors[j][:,source])
                        )
            summary.append({
                "family":family,"arm":arm,
                "n_registered_opportunities":len(entries),
                "n_completed_native_fits":len(completed),
                "n_train_identifiability_abstentions":len(entries)-len(completed),
                "mean_validation_auprc":(
                    float(np.mean([e["validation_auprc"] for e in completed]))
                    if completed else None
                ),
                "mean_validation_brier":(
                    float(np.mean([e["validation_brier"] for e in completed]))
                    if completed else None
                ),
                "mean_native_concept_oracle_source_alignment_postfit":(
                    float(np.mean([
                        e["oracle_source_alignment_eval_only"]["mean_abs_oracle_source_spearman_eval_only"]
                        for e in completed
                    ])) if completed else None
                ),
                "mean_crossseed_signed_native_contribution_rank_spearman_unaligned":(
                    float(np.mean(per_source_signed_rank))
                    if per_source_signed_rank else None
                ),
                "PCA_or_contrast_raw_linear_source_alignment_eval_only":(
                    independent_oracle_alignment[f"{family}__{arm}"]
                ),
                "clinical_source_identity_proven":False,
            })
    if len(all_opportunities)!=24 or len(summary)!=8 or len(source_receipts)!=4:
        raise AssertionError("Missing a mixed-source method or aborted native model")
    outdir.mkdir(parents=True,exist_ok=False)
    for name,rows in (
        ("ALL_24_REGISTERED_NATIVE_SOURCE_OPPORTUNITIES.csv",all_opportunities),
        ("EIGHT_ARM_SUMMARIES.csv",summary),
    ):
        # CSV is deliberately flat. The detailed matched oracle source
        # alignment dict is persisted in STATUS.json.
        keys=sorted({key for row in rows for key in row})
        with (outdir/name).open("w",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    k:(json.dumps(row[k],sort_keys=True) if isinstance(row.get(k),(dict,list)) else row.get(k))
                    for k in keys
                })
    (outdir/"SOURCE_ENVIRONMENT_PROVENANCE.json").write_text(
        json.dumps(source_receipts,indent=2,allow_nan=False)
    )
    receipt={
        "status":"V22_MIXED_SOURCE_ENVIRONMENT_IDENTIFIABILITY_AND_NATIVE_CONCEPT_COMPARISON",
        "n_registered_native_opportunities":len(all_opportunities),
        "n_trained_native_models":sum(r["status"]=="TRAINED" for r in all_opportunities),
        "n_observational_identifiability_abstentions":sum(
            r["status"]!="TRAINED" for r in all_opportunities
        ),
        "n_source_generator_families":len(FAMILIES),
        "n_model_initialization_seeds":len(MODEL_SEEDS),
        "expected_two_source_mixing_nonorthogonality_cosine":float(
            abs(np.dot(
                np.array([.62,.49,.31,.48,.39])/np.linalg.norm([.62,.49,.31,.48,.39]),
                np.array([.53,.72,-.41,.22,-.18])/np.linalg.norm([.53,.72,-.41,.22,-.18]),
            ))
        ),
        "no_latent_oracle_used_in_train_or_source_proposal":True,
        "validation_truth_loaded_only_after_model_fit":True,
        "synthetic_test_seed_522203_sealed":True,
        "comparisons_have_exactly_same_native_train_task_and_whitened_observations":True,
        "no_raw_feature_to_logit_bypass_or_posthoc_classifier":True,
        "mixing_environment_assumption_can_fail_and_still_spoof_identifiability":True,
        "physical_vs_instrument_meaning_known_to_generator_only":True,
        "not_a_novel_BSS_identification_theorem":True,
        "no_real_physician_or_icu_external_data":True,
        "all_predeclared_opportunities_reported":True,
        "source_evidence":source_receipts,
        "per_fit_or_abstention":all_opportunities,
        "aggregate":summary,
    }
    (outdir/"STATUS.json").write_text(json.dumps(receipt,indent=2,allow_nan=False))
    return receipt


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    run(args.out_dir)
