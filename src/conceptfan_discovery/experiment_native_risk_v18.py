"""V18: native retraining comparison after *independent* predictive screening.

Deliberately synthetic observable source -> risk test. All arms are trained on
the same TRAIN episodes, same optimizer config and order. No validation or
oracle hidden states enter candidate generation, policy screening or training.
The signed-policy effect is NOT a proof of independent signed-effect stability.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
import yaml

from .data import DiscoveryData,DiscoverySplit
from .clinical_multiscale_v10 import fit_multiscale
from .candidate_admission_v17 import (
    propose_observable_residual_concept,admit_candidate_without_prediction_drift,
)
from .risk_screen_v18 import (
    screen_risk_association,configure_native_risk_orientation,
    begin_native_risk_training,native_observable_grounding_loss,
    risk_evidence_passport,
)
from .model_v2 import masked_observation_view,v2_loss
from .observation_process_concepts_v7 import dual_training_objective
from .multiscale_physiology_v10 import physiological_scale_entropy

DATA_GENERATOR_SEED=20411204
MODEL_SEEDS=(11,23,37)
MODES=("latent_predictive_positive","latent_predictive_negative","latent_risk_null")
ARMS=("continued_k2","unrestricted_k3","screened_k3",
      "screened_grounded_k3","intentionally_wrong_sign_k3")
TRAIN_N=800
VAL_N=200
SEALED_TEST_N=80
DISCOVERY_N=350
SCREEN_N=300
HISTORY=24
HORIZON=6
CHANNELS=tuple(f"sensor_{j}" for j in range(5))
FIRST_EPOCHS=6
SECOND_EPOCHS=6
GROUNDING_COEFFICIENT=1.0
INITIAL_NEW_COEFF=.05
RISK_P_ALPHA=.01
BATCH=64


def make_data(mode):
    if mode not in MODES:
        raise ValueError("Unknown predeclared synthetic mode")
    rng=np.random.default_rng(DATA_GENERATOR_SEED)
    n=TRAIN_N+VAL_N+SEALED_TEST_N
    steps=HISTORY+HORIZON
    z=np.zeros((n,steps,3))
    for i in range(steps):
        shocks=rng.normal(size=(n,3))
        z[:,i]=(shocks if i==0 else .8*z[:,i-1]+.6*shocks)
    x=rng.normal(scale=.1,size=(n,steps,5))
    x[:,:,0]+=.7*z[:,:,0]
    x[:,:,1]+=.7*z[:,:,1]
    x[:,:,2]+=.9*z[:,:,2]
    future=z[:,HISTORY:,2].mean(axis=1)
    if mode=="latent_predictive_positive":
        logit=3.2*future
    elif mode=="latent_predictive_negative":
        logit=-3.2*future
    else:
        logit=np.zeros_like(future)
    p=1/(1+np.exp(-logit))
    y=(rng.random(n)<p).astype(np.float32)
    median=np.median(x[:TRAIN_N,:HISTORY],axis=(0,1))
    iqr=np.maximum(np.quantile(x[:TRAIN_N,:HISTORY],.75,axis=(0,1))
                   -np.quantile(x[:TRAIN_N,:HISTORY],.25,axis=(0,1)),.01)
    def build(a,b):
        ids=np.arange(a,b,dtype=np.int64)
        past=np.clip((x[ids,:HISTORY]-median)/iqr,-6,6).astype(np.float32)
        future_obs=np.clip((x[ids,HISTORY:]-median)/iqr,-6,6).astype(np.float32)
        return DiscoverySplit(
            x=past,mask=np.ones_like(past),
            future=future_obs,future_mask=np.ones_like(future_obs),
            y=y[ids],oracle_states=None,episode_ids=ids,
        )
    return DiscoveryData(
        train=build(0,TRAIN_N),validation=build(TRAIN_N,TRAIN_N+VAL_N),
        receipt={"channel_names":list(CHANNELS),
                 "test_ids_sealed":SEALED_TEST_N,
                 "no_oracle_source_trajectories_exported":True},
        medians=median,scales=iqr,
    )


def _config():
    base=yaml.safe_load(Path("configs/medical/pd_clinical_unknown_neutral_v6.yaml").read_text())
    dual=yaml.safe_load(Path("configs/medical/pd_clinical_dual_process_v7.yaml").read_text())
    v10=yaml.safe_load(Path("configs/medical/pd_clinical_v10_multiscale.yaml").read_text())
    base["training"]["epochs"]=FIRST_EPOCHS
    base["training"]["batch_size"]=BATCH
    v10["architecture"]["scales_hours"]=[1,4,12]
    return base,dual,v10


def _fit_continuation(model,data,seed,base,dual,candidate=None,grounding_weight=0.0):
    t=base["training"]; maskhead=dual["dual_head"]
    optimizer=torch.optim.AdamW(
        model.parameters(),lr=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
    )
    positives=float(data.train.y.sum())
    pos_weight=torch.tensor((len(data.train.y)-positives)/max(positives,1.))
    loader=data.train.loader(BATCH,True,int(t["batch_order_seed"])+710)
    mask_rng=torch.Generator().manual_seed(seed+88021)
    for _ in range(SECOND_EPOCHS):
        model.train()
        for x,original_mask,future,future_mask,y in loader:
            optimizer.zero_grad(set_to_none=True)
            inputs,visible,heldout=masked_observation_view(
                x,original_mask,fraction=float(t["main_mask_fraction"]),
                generator=mask_rng,
            )
            out=model(inputs,visible,process_mask=original_mask)
            physical,parts=v2_loss(
                model.physical,out.physical,y,future,future_mask,
                x,heldout,mode="full",pos_weight=pos_weight,
                loss_weights=t["loss_weights"],
            )
            total,_=dual_training_objective(
                model,out,y,physical,parts["task"],pos_weight=pos_weight,
                mask_coeff_l1=float(maskhead["signed_mask_coefficient_l1"]),
                mask_redundancy_weight=float(maskhead["mask_definition_redundancy"]),
            )
            pen=model.physical.spectral_definition_penalty()
            total=total+float(t["spectral_anchor_weight"])*pen["operator_displacement"]
            total=total+float(t["spectral_axis_independence_weight"])*pen["observable_axis_nonorthogonality"]
            total=total+.002*physiological_scale_entropy(model)
            if grounding_weight:
                if candidate is None:
                    raise ValueError("Cannot supervise nonexistent novel source")
                grounded=native_observable_grounding_loss(
                    out.physical,inputs,visible,candidate,slot=model.physical.n_slots-1,
                    window=6,
                )
                total=total+grounding_weight*grounded
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite native V18 loss")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),float(t["gradient_clip"]))
            optimizer.step()
    model.eval()
    return model


def _evaluate(model,split,candidate=None):
    model.eval()
    predictions=[]; effects=[]; raw=[]; align=[];err=0.
    with torch.no_grad():
        for i in range(0,len(split.y),128):
            x=torch.from_numpy(split.x[i:i+128])
            mask=torch.from_numpy(split.mask[i:i+128])
            out=model(x,mask,process_mask=mask)
            predictions.extend(out.probabilities.cpu().numpy().tolist())
            err=max(err,float(out.decomposition_error()))
            if candidate is not None:
                k=model.physical.n_slots-1
                effects.extend(out.physical.contributions[:,k].cpu().numpy().tolist())
                activation=out.physical.raw_activation[:,-6:,k]
                reliability=out.physical.availability[:,-6:,k]
                measurement=(activation*reliability).sum(1)/reliability.sum(1).clamp_min(1e-8)
                raw.extend(measurement.cpu().numpy().tolist())
                v=x[:,-6:,:]@torch.as_tensor(candidate.direction,dtype=x.dtype)
                align.extend(v.mean(1).cpu().numpy().tolist())
    if err>2e-5:
        raise AssertionError("Exact-native additive logit violated")
    pred=np.asarray(predictions)
    if candidate is not None:
        effect=np.asarray(effects)
        r=float(spearmanr(raw,align).statistic)
        coeff=float(model.physical.decision_weights()[-1].detach())
        return {
            "heldout_auprc":float(average_precision_score(split.y,pred)),
            "heldout_auroc":float(roc_auc_score(split.y,pred)),
            "heldout_brier":float(brier_score_loss(split.y,pred)),
            "max_additive_error":err,
            "mean_abs_new_slot_effect":float(np.mean(abs(effect))),
            "new_slot_native_coefficient":coeff,
            "native_raw_vs_observed_direction_spearman":r if np.isfinite(r) else None,
            "patient_signed_effect_vector":effect,
        }
    return {
        "heldout_auprc":float(average_precision_score(split.y,pred)),
        "heldout_auroc":float(roc_auc_score(split.y,pred)),
        "heldout_brier":float(brier_score_loss(split.y,pred)),
        "max_additive_error":err,
        "mean_abs_new_slot_effect":None,
        "new_slot_native_coefficient":None,
        "native_raw_vs_observed_direction_spearman":None,
        "patient_signed_effect_vector":None,
    }


def run(output:Path):
    if output.exists():
        raise FileExistsError("Never overwrite scientific artifacts")
    torch.set_num_threads(1)
    all_rows=[];diagnostic_vectors={}
    screened_status={}
    for mode in MODES:
        data=make_data(mode)
        basis=np.eye(len(CHANNELS),2)
        discovered=data.train.x[:DISCOVERY_N]
        candidate=propose_observable_residual_concept(
            discovered,np.ones_like(discovered),basis,CHANNELS,
            seed=20281,permutations=99,bootstraps=24,
        )
        cohort=slice(DISCOVERY_N,DISCOVERY_N+SCREEN_N)
        evidence=screen_risk_association(
            candidate,data.train.x[cohort],data.train.mask[cohort],
            data.train.y[cohort],
            screening_episode_ids=data.train.episode_ids[cohort],
            discovery_episode_ids=data.train.episode_ids[:DISCOVERY_N],
            seed=20871,permutations=199,bootstraps=199,
        )
        screened_status[mode]=risk_evidence_passport(evidence)
        print("risk screen",mode,evidence.status,evidence.signed_orientation,
              round(evidence.point_biserial_r,4),flush=True)
        base,dual,v10=_config()
        diagnostic_vectors[mode]={a:[] for a in ARMS}
        for seed in MODEL_SEEDS:
            original,_=fit_multiscale(
                data,basis,seed,base,dual,v10,consistency=False,
                feature_exploration=True,
            )
            original.eval()
            for arm in ARMS:
                if arm.startswith("screened") or arm.startswith("intentionally"):
                    if evidence.status!="ACCEPT_PREDICTIVE_ASSOCIATION":
                        all_rows.append({
                            "mode":mode,"seed":seed,"arm":arm,"status":"ABSTAIN_RISK_SCREEN",
                            "screen_signed_orientation":0,
                            "heldout_auprc":None,"heldout_auroc":None,
                            "heldout_brier":None,"max_additive_error":None,
                            "mean_abs_new_slot_effect":None,
                            "new_slot_native_coefficient":None,
                            "native_raw_vs_observed_direction_spearman":None,
                        })
                        continue
                if arm=="continued_k2":
                    model=copy.deepcopy(original)
                else:
                    model=admit_candidate_without_prediction_drift(
                        original,candidate,expert_decision="approve",
                    )
                    if arm in ("screened_k3","screened_grounded_k3","intentionally_wrong_sign_k3"):
                        from dataclasses import replace
                        policy = evidence if arm!="intentionally_wrong_sign_k3" else replace(
                            evidence,signed_orientation=-evidence.signed_orientation,
                        )
                        configure_native_risk_orientation(model,candidate,policy,slot=2)
                        begin_native_risk_training(model,slot=2,
                                                   initial_coefficient_raw=INITIAL_NEW_COEFF)
                    else:
                        # Matched nonzero risk-head starting coefficient:
                        # no sign constraint, so it can learn either direction.
                        with torch.no_grad():
                            model.physical.decision_raw_weight[-1]=INITIAL_NEW_COEFF
                model=_fit_continuation(
                    model,data,seed,base,dual,
                    candidate=candidate if arm=="screened_grounded_k3" else None,
                    grounding_weight=(GROUNDING_COEFFICIENT if arm=="screened_grounded_k3" else 0.),
                )
                metric=_evaluate(
                    model,data.validation,
                    candidate=None if arm=="continued_k2" else candidate,
                )
                diagnostic_vectors[mode][arm].append(metric.pop("patient_signed_effect_vector"))
                all_rows.append({
                    "mode":mode,"seed":seed,"arm":arm,"status":"TRAINED",
                    "screen_signed_orientation":evidence.signed_orientation,
                    **metric,
                })
                print(mode,seed,arm,round(metric["heldout_auprc"],4),
                      round(metric["mean_abs_new_slot_effect"] or 0,5),
                      flush=True)
    summary=[]
    for mode in MODES:
        for arm in ARMS:
            sub=[x for x in all_rows if x["mode"]==mode and x["arm"]==arm]
            fitted=[x for x in sub if x["status"]=="TRAINED"]
            vectors=[v for v in diagnostic_vectors[mode][arm] if v is not None]
            pairwise=[]
            for i in range(len(vectors)):
                for j in range(i+1,len(vectors)):
                    val=spearmanr(vectors[i],vectors[j]).statistic
                    if np.isfinite(val):pairwise.append(float(val))
            summary.append({
                "mode":mode,"arm":arm,
                "possible_seed_count":len(MODEL_SEEDS),"fits":len(fitted),
                "screen_abstentions":len(sub)-len(fitted),
                "mean_heldout_auprc":float(np.mean([r["heldout_auprc"] for r in fitted])) if fitted else None,
                "mean_heldout_brier":float(np.mean([r["heldout_brier"] for r in fitted])) if fitted else None,
                "mean_new_slot_abs_effect":float(np.mean([r["mean_abs_new_slot_effect"] for r in fitted])) if fitted and arm!="continued_k2" else None,
                "mean_native_vs_observed_alignment":float(np.mean([r["native_raw_vs_observed_direction_spearman"] for r in fitted])) if fitted and arm!="continued_k2" else None,
                "signed_effect_cross_seed_spearman_three_dependent_pairs":float(np.mean(pairwise)) if pairwise else None,
                "sign_fixed_by_constructed_policy_not_independent_validation":arm in (
                    "screened_k3","screened_grounded_k3","intentionally_wrong_sign_k3",
                ),
            })
    output.mkdir(parents=True,exist_ok=False)
    for filename,records in (("ALL_FITS_AND_ABSTENTIONS.csv",all_rows),("SUMMARY.csv",summary)):
        with (output/filename).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]))
            writer.writeheader();writer.writerows(records)
    status={
        "status":"V18_NATIVE_RISK_ORIENTATION_AND_GROUNDING_FULL_ABLATION",
        "dataset_generator_seed":DATA_GENERATOR_SEED,
        "mode_count":len(MODES),"model_initialization_seeds":list(MODEL_SEEDS),
        "predeclared_arm_count":len(ARMS),"all_cases":len(all_rows),
        "discovery_n":DISCOVERY_N,"risk_screen_n":SCREEN_N,
        "train_n":TRAIN_N,"heldout_validation_n":VAL_N,
        "sealed_test_n":SEALED_TEST_N,
        "risk_screen_disjoint_from_source_discovery":True,
        "no_oracle_states_entered_training":True,
        "raw_patient_contribution_vectors_not_exported":True,
        "no_physical_causal_claim":True,
        "signed_orientation_fixed_by_design_not_demonstrated":True,
        "prior_research_validation_not_reused":True,
        "complete_denominator_includes_screen_abstentions":True,
        "risk_screen_passports":screened_status,
        "all_fits_and_abstentions":all_rows,
        "summary":summary,
    }
    (output/"STATUS.json").write_text(json.dumps(status,indent=2,allow_nan=False))
    return status


if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-dir",type=Path,required=True)
    args=ap.parse_args()
    run(args.out_dir)
