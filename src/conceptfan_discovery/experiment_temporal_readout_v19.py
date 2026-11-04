"""V19: frozen native temporal-readout comparison, not a clinical benchmark.

All native models use identical V10 multi-scale concept generation, TRAIN-only
observable candidate definition, exact additive native logit, optimization,
loss and seed. Only concept-trajectory *task-summary* changes. This is not an
independent new dataset: V18 validation has been inspected previously.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss

from .candidate_admission_v17 import propose_observable_residual_concept
from .clinical_multiscale_v10 import fit_multiscale
from .experiment_native_risk_v18 import (
    make_data,_config,MODES,MODEL_SEEDS,CHANNELS,
    DISCOVERY_N,TRAIN_N,VAL_N,SEALED_TEST_N,FIRST_EPOCHS,SECOND_EPOCHS,
)
from .temporal_readout_v19 import SHAPE_OPERATORS

ARMS=("legacy","fixed_shape","learned_shape")
EXPECTED_FITS=len(MODES)*len(MODEL_SEEDS)*len(ARMS)


def _train_only_basis(data):
    prior=np.eye(len(CHANNELS),2,dtype=np.float64)
    train_x=data.train.x[:DISCOVERY_N]
    found=propose_observable_residual_concept(
        train_x,np.ones_like(train_x),prior,CHANNELS,
        seed=20281,permutations=99,bootstraps=24,
    )
    if found.permutation_p > .01:
        raise ValueError("ABSTAIN: not a strongly supported TRAIN-only third direction")
    return np.ascontiguousarray(np.column_stack((prior,found.direction))),found


def _score(model,split,candidate,mode):
    model.eval()
    probabilities=[]
    raw=[]
    signed_effect=[]
    aligned=[]
    max_error=0.
    with torch.no_grad():
        for start in range(0,len(split.y),128):
            stop=min(start+128,len(split.y))
            x=torch.from_numpy(split.x[start:stop])
            mask=torch.from_numpy(split.mask[start:stop])
            result=model(x,mask,process_mask=mask)
            max_error=max(max_error,float(result.decomposition_error()))
            probabilities.extend(result.probabilities.cpu().numpy().tolist())
            signed_effect.extend(result.physical.contributions[:,-1].cpu().numpy().tolist())
            activation=result.physical.raw_activation[:,-6:,-1]
            reliability=result.physical.availability[:,-6:,-1]
            estimate=(activation*reliability).sum(1)/reliability.sum(1).clamp_min(1e-8)
            raw.extend(estimate.cpu().numpy().tolist())
            v=x[:,-6:] @ torch.as_tensor(candidate.direction,dtype=x.dtype)
            aligned.extend(v.mean(1).cpu().numpy().tolist())
    if max_error > 2e-5:
        raise AssertionError("V19 exact native additive decomposition failed")
    p=np.asarray(probabilities)
    correlation=spearmanr(raw,aligned).statistic
    metric={
        "validation_auprc":float(average_precision_score(split.y,p)),
        "validation_auroc":float(roc_auc_score(split.y,p)),
        "validation_brier":float(brier_score_loss(split.y,p)),
        "max_additive_logit_error":max_error,
        "mean_abs_new_slot_signed_effect":float(np.mean(np.abs(signed_effect))),
        "native_new_slot_raw_vs_observed_spearman":(
            float(correlation) if np.isfinite(correlation) else None),
    }
    if mode!="legacy":
        pi=model.physical.temporal_readout_weights().detach().cpu().numpy()
        entropy=float(-(pi*np.log(np.maximum(pi,1e-10))).sum(axis=-1).mean())
        metric.update({
            "shape_operator_weight_recent_mean_positive_slot3":float(pi[-1,0]),
            "shape_operator_weight_outside_positive_recent_mean_slot3":float(1-pi[-1,0]),
            "mean_shape_entropy":entropy,
        })
    else:
        metric.update({
            "shape_operator_weight_recent_mean_positive_slot3":1.,
            "shape_operator_weight_outside_positive_recent_mean_slot3":0.,
            "mean_shape_entropy":0.,
        })
    return metric,np.asarray(signed_effect,dtype=np.float64)


def run(directory:Path):
    if directory.exists():
        raise FileExistsError("V19 evidence output must be fresh and immutable")
    torch.set_num_threads(1)
    rows=[];effects={};candidates={}
    for mode in MODES:
        data=make_data(mode)
        if data.train.oracle_states is not None or data.validation.oracle_states is not None:
            raise AssertionError("Never train on hidden physical source oracle")
        if len(data.train.y)!=TRAIN_N or len(data.validation.y)!=VAL_N:
            raise AssertionError("TRAIN/VALIDATION dataset contract changed")
        basis,candidate=_train_only_basis(data)
        candidates[mode]={
            "candidate_id":candidate.candidate_id,
            "train_only_candidate_p":candidate.permutation_p,
            "train_only_candidate_bootstrap_q10":candidate.bootstrap_abs_cosine_q10,
        }
        base,dual,v10=_config()
        base["training"]["epochs"]=FIRST_EPOCHS+SECOND_EPOCHS
        effects[mode]={a:[] for a in ARMS}
        for seed in MODEL_SEEDS:
            for arm in ARMS:
                model,receipt=fit_multiscale(
                    data,basis,seed,base,dual,v10,consistency=False,
                    expert_feedback=None,expert_feedback_weight=0.,
                    feature_exploration=True,temporal_readout_mode=arm,
                )
                metric,per_patient_effect=_score(
                    model,data.validation,candidate,arm,
                )
                if receipt["temporal_readout_mode"]!=arm:
                    raise AssertionError("Requested V19 arm was not trained")
                effects[mode][arm].append(per_patient_effect)
                row={
                    "mode":mode,"initialization_seed":seed,"arm":arm,
                    "status":"TRAINED","native_slots":model.physical.n_slots,
                    "train_epochs":FIRST_EPOCHS+SECOND_EPOCHS,
                    "train_feature_bridge":receipt["expert_operator_gradient_bridge_enabled"],
                    **metric,
                }
                rows.append(row)
                print(mode,seed,arm,"AUPRC",round(metric["validation_auprc"],6),
                      "shape_nonrecent",round(metric["shape_operator_weight_outside_positive_recent_mean_slot3"],4),
                      flush=True)
    summaries=[]
    for mode in MODES:
        for arm in ARMS:
            group=[r for r in rows if r["mode"]==mode and r["arm"]==arm]
            vectors=effects[mode][arm]
            paired=[]
            for i in range(len(vectors)):
                for j in range(i+1,len(vectors)):
                    rho=spearmanr(vectors[i],vectors[j]).statistic
                    if np.isfinite(rho):
                        paired.append(float(rho))
            summaries.append({
                "mode":mode,"arm":arm,
                "n_independent_model_initializations":len(group),
                "mean_validation_auprc":float(np.mean([r["validation_auprc"] for r in group])),
                "mean_validation_auroc":float(np.mean([r["validation_auroc"] for r in group])),
                "mean_validation_brier":float(np.mean([r["validation_brier"] for r in group])),
                "mean_abs_new_slot_effect":float(np.mean([r["mean_abs_new_slot_signed_effect"] for r in group])),
                "mean_native_slot_observable_alignment":float(np.mean([r["native_new_slot_raw_vs_observed_spearman"] for r in group
                  if r["native_new_slot_raw_vs_observed_spearman"] is not None])),
                "mean_nonrecent_shape_weight_slot3":float(np.mean([r["shape_operator_weight_outside_positive_recent_mean_slot3"] for r in group])),
                "mean_same_slot_signed_effect_crossseed_spearman_dependent_pairs":float(np.mean(paired)) if paired else None,
                "not_confirmatory_p_value":True,
            })
    if len(rows)!=EXPECTED_FITS:
        raise AssertionError("Do not hide failed model fits")
    directory.mkdir(parents=True,exist_ok=False)
    for name,data in (("ALL_NATIVE_FITS.csv",rows),("AGGREGATE.csv",summaries)):
        with (directory/name).open("w",newline="") as file:
            w=csv.DictWriter(file,fieldnames=list(data[0]))
            w.writeheader();w.writerows(data)
    manifest={
        "status":"V19_NATIVE_TEMPORAL_POOLING_FIXED_VERSUS_LEARNED",
        "dataset_original_family":"V18_SYNTHETIC_ALREADY_INSPECTED",
        "number_independent_model_fits":len(rows),
        "number_generator_modes":len(MODES),
        "initialization_seeds":list(MODEL_SEEDS),
        "model_arms":list(ARMS),
        "native_slots":3,
        "same_train_data_across_arms":True,
        "train_only_discovery_candidate":True,
        "evaluation_labels_not_used_in_training":True,
        "test_cohort_sealed":SEALED_TEST_N,
        "raw_patient_effects_not_exported":True,
        "all_scenario_and_seed_fits_reported":True,
        "no_clinical_physical_semantics":True,
        "not_a_new_independent_heldout_hospital":True,
        "synthetic_source_provenance":candidates,
        "shape_readout_operators":list(SHAPE_OPERATORS),
        "results":rows,
        "aggregate":summaries,
    }
    (directory/"STATUS.json").write_text(json.dumps(manifest,indent=2,allow_nan=False))
    return manifest


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",required=True,type=Path)
    args=p.parse_args()
    run(args.out_dir)
