"""V17 controlled *post-admission retraining* positive capability experiment.

A newly planted independent persistent factor drives a future outcome.
This is an intentionally favorable synthetic demonstration, not clinical
identification, a medical predictor, or an independent-hospital benchmark.
Three matched paths: capacity-2 continuous, native 2->3 growth, capacity-3
from scratch. Candidate was found from TRAIN observation residuals only.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score
import yaml

from .data import DiscoveryData, DiscoverySplit
from .candidate_admission_v17 import (
    propose_observable_residual_concept,admit_candidate_without_prediction_drift,
)
from .clinical_multiscale_v10 import fit_multiscale
from .model_v2 import v2_loss,masked_observation_view
from .observation_process_concepts_v7 import dual_training_objective
from .multiscale_physiology_v10 import physiological_scale_entropy

GENERATOR_SEED=20311104
SEEDS=(11,23,37)
TRAIN_N=600
VAL_N=200
TEST_SEALED_N=40
HISTORY=24
FUTURE=6
CHANNELS=tuple(f"sensor_{i}" for i in range(5))
FIRST_EPOCHS=6
SECOND_EPOCHS=6
BATCH=64


def _generate_data():
    n=TRAIN_N+VAL_N+TEST_SEALED_N
    steps=HISTORY+FUTURE
    rng=np.random.default_rng(GENERATOR_SEED)
    hidden=np.zeros((n,steps,3))
    for t in range(steps):
        new=rng.normal(size=(n,3))
        hidden[:,t,:]=(new if t==0 else
            .8*hidden[:,t-1,:]+.6*new)
    observed=rng.normal(scale=.10,size=(n,steps,5))
    observed[:,:,0]+=hidden[:,:,0]*.7
    observed[:,:,1]+=hidden[:,:,1]*.7
    observed[:,:,2]+=hidden[:,:,2]*.9
    # Future outcome is driven by the *unmodeled third factor* only.
    risk=2.0*hidden[:,HISTORY:HISTORY+FUTURE,2].mean(axis=1)
    prob=1/(1+np.exp(-risk))
    y=(rng.random(n)<prob).astype(np.float32)
    med=np.median(observed[:TRAIN_N,:HISTORY,:],axis=(0,1))
    q25=np.quantile(observed[:TRAIN_N,:HISTORY,:],.25,axis=(0,1))
    q75=np.quantile(observed[:TRAIN_N,:HISTORY,:],.75,axis=(0,1))
    scale=np.maximum(q75-q25,.01)

    def split(a,b):
        ids=np.arange(a,b,dtype=np.int64)
        if np.unique(y[ids]).size != 2:
            raise ValueError("Need both outcome classes")
        hist=np.clip((observed[ids,:HISTORY]-med)/scale,-6,6).astype(np.float32)
        fut=np.clip((observed[ids,HISTORY:]-med)/scale,-6,6).astype(np.float32)
        return DiscoverySplit(
            x=hist,mask=np.ones_like(hist),future=fut,future_mask=np.ones_like(fut),
            y=y[ids],oracle_states=None,episode_ids=ids,
        )
    data=DiscoveryData(
        train=split(0,TRAIN_N),validation=split(TRAIN_N,TRAIN_N+VAL_N),
        receipt={"channel_names":list(CHANNELS),
                 "train_only_normalization":True,
                 "n_train":TRAIN_N,"n_validation":VAL_N,
                 "n_test_sealed":TEST_SEALED_N,
                 "future_latent_hidden_from_optimizer":True},
        medians=med,scales=scale,
    )
    return data


def _configs():
    v6=yaml.safe_load(Path("configs/medical/pd_clinical_unknown_neutral_v6.yaml").read_text())
    v7=yaml.safe_load(Path("configs/medical/pd_clinical_dual_process_v7.yaml").read_text())
    v10=yaml.safe_load(Path("configs/medical/pd_clinical_v10_multiscale.yaml").read_text())
    v6["training"]["epochs"]=FIRST_EPOCHS
    v6["training"]["batch_size"]=BATCH
    v10["architecture"]["scales_hours"]=[1,4,12]
    return v6,v7,v10


def _continue_fit(model,data,seed,config,v7,*,epochs):
    """Matched native objective, without any raw-input bypass."""
    t=config["training"]; dual=v7["dual_head"]
    optimizer=torch.optim.AdamW(
        model.parameters(),lr=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
    )
    positives=float(data.train.y.sum())
    posweight=torch.tensor((len(data.train.y)-positives)/max(positives,1))
    loader=data.train.loader(BATCH,True,int(t["batch_order_seed"])+710)
    mask_rng=torch.Generator().manual_seed(seed+88021)
    for epoch in range(epochs):
        model.train()
        for x,m,future,future_mask,y in loader:
            optimizer.zero_grad(set_to_none=True)
            masked,visible,withheld=masked_observation_view(
                x,m,fraction=float(t["main_mask_fraction"]),generator=mask_rng)
            out=model(masked,visible,process_mask=m)
            phys,parts=v2_loss(model.physical,out.physical,y,future,future_mask,
                x,withheld,mode="full",pos_weight=posweight,
                loss_weights=t["loss_weights"])
            objective,_=dual_training_objective(
                model,out,y,phys,parts["task"],pos_weight=posweight,
                mask_coeff_l1=float(dual["signed_mask_coefficient_l1"]),
                mask_redundancy_weight=float(dual["mask_definition_redundancy"]))
            source=model.physical.spectral_definition_penalty()
            objective=(objective
                +float(t["spectral_anchor_weight"])*source["operator_displacement"]
                +float(t["spectral_axis_independence_weight"])*source["observable_axis_nonorthogonality"]
                +.002*physiological_scale_entropy(model))
            if not torch.isfinite(objective):
                raise FloatingPointError("Non-finite native growth training loss")
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),float(t["gradient_clip"]))
            optimizer.step()
    model.eval()
    return model


def _score(model,split,slot=None):
    model.eval();pred=[];decomp=0.;magnitude=[]
    with torch.no_grad():
        for i in range(0,len(split.y),128):
            x=torch.from_numpy(split.x[i:i+128])
            m=torch.from_numpy(split.mask[i:i+128])
            out=model(x,m,process_mask=m)
            pred.extend(out.probabilities.numpy().tolist())
            decomp=max(decomp,float(out.decomposition_error()))
            if slot is not None:
                magnitude.extend(out.physical.contributions[:,slot].abs().numpy().tolist())
    return {
        "validation_auprc":float(average_precision_score(split.y,pred)),
        "validation_auroc":float(roc_auc_score(split.y,pred)),
        "max_exact_additive_error":decomp,
        "mean_new_slot_abs_logit_effect":float(np.mean(magnitude)) if magnitude else None,
    }


def run(output:Path):
    if output.exists():
        raise FileExistsError("Refuse to overwrite scientific outcomes")
    torch.set_num_threads(1)
    data=_generate_data()
    config,v7,v10=_configs()
    prior=np.eye(len(CHANNELS),2)
    candidate=propose_observable_residual_concept(
        data.train.x,data.train.mask,prior,CHANNELS,
        seed=20311105,permutations=99,bootstraps=24,
    )
    # This approval is PROGRAMMATIC POSITIVE CONTROL on simulator, not
    # a doctor's judgement or proof of the source's physical identity.
    if candidate.permutation_p > .01:
        raise ValueError("No frozen candidate for controlled growth")
    rows=[]
    for seed in SEEDS:
        initial,receipt=fit_multiscale(
            data,prior,seed,config,v7,v10,consistency=False,
        )
        initial.eval()
        initial_score=_score(initial,data.validation)
        unchanged=copy.deepcopy(initial)
        expanded=admit_candidate_without_prediction_drift(
            initial,candidate,expert_decision="approve")
        before=initial_score
        after=_score(expanded,data.validation,slot=2)
        if abs(after["validation_auprc"]-before["validation_auprc"])>1e-9:
            raise AssertionError("Candidate admission changed prediction before training")
        unchanged=_continue_fit(unchanged,data,seed,config,v7,epochs=SECOND_EPOCHS)
        expanded=_continue_fit(expanded,data,seed,config,v7,epochs=SECOND_EPOCHS)
        scratch_config=copy.deepcopy(config)
        scratch_config["training"]["epochs"]=FIRST_EPOCHS+SECOND_EPOCHS
        scratch,_=fit_multiscale(
            data,np.eye(len(CHANNELS),3),seed,scratch_config,v7,v10,
            consistency=False,
        )
        for arm,model,slot in (
            ("continue_k2",unchanged,None),
            ("add_new_k3",expanded,2),
            ("scratch_k3",scratch,2),
        ):
            scores=_score(model,data.validation,slot=slot)
            if scores["max_exact_additive_error"]>2e-5:
                raise AssertionError("Native bottleneck changed after training")
            rows.append({
                "seed":seed,"arm":arm,
                "initial_pre_growth_auprc":before["validation_auprc"],
                "candidate_p_train":candidate.permutation_p,
                "candidate_q10_train":candidate.bootstrap_abs_cosine_q10,
                **scores,
            })
            print(seed,arm,scores["validation_auprc"],flush=True)
    summary=[]
    for arm in ("continue_k2","add_new_k3","scratch_k3"):
        group=[row for row in rows if row["arm"]==arm]
        summary.append({
            "arm":arm,"n_models":len(group),
            "mean_val_auprc":float(np.mean([r["validation_auprc"] for r in group])),
            "mean_new_slot_logit_abs_effect":(
                float(np.mean([r["mean_new_slot_abs_logit_effect"] for r in group]))
                if arm!="continue_k2" else None),
        })
    output.mkdir(parents=True,exist_ok=False)
    with (output/"ALL_FITS.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (output/"SUMMARY.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    result={
        "status":"V17_POSITIVE_CONTROL_NATIVE_POST_ADMISSION_RETRAINING",
        "n_models":len(rows),
        "new_source_observed_only_train_candidate":candidate.candidate_id,
        "candidate_train_bootstrap_q10":candidate.bootstrap_abs_cosine_q10,
        "model_never_sees_oracle_latent_tracks":True,
        "test_ids_remain_sealed":True,
        "outcome_depends_on_future_hidden_factor_by_generator_design":True,
        "first_stage_epochs":FIRST_EPOCHS,
        "continued_epochs":SECOND_EPOCHS,
        "scratch_epoch_budget":FIRST_EPOCHS+SECOND_EPOCHS,
        "all_models_reported":True,
        "report_is_not_external_or_clinical_validation":True,
        "results":rows,"summary":summary,
    }
    (output/"STATUS.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",required=True,type=Path)
    args=parser.parse_args()
    run(args.out_dir)
