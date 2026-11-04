"""V19c: native concept capacity, not a raw classifier, versus temporal readout.

This is an explicitly FORCED overcapacity scientific control: added K4/K5
observable directions from train-only residual SVD are NOT certified semantic
concepts or approved deployment changes. Model fitting retains the V10
prediction-through-native-concepts contract and full per-arm denominator.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss

from .candidate_admission_v17 import (
    ProposedObservableConcept,propose_observable_residual_concept,
    admit_candidate_without_prediction_drift,
)
from .risk_screen_v18 import screen_risk_association,risk_evidence_passport
from .multiscale_physiology_v10 import MultiScaleDualProcessConceptFAN
from .clinical_multiscale_v10 import fit_multiscale
from .experiment_native_risk_v18 import (
    make_data,_config,_fit_continuation,
    MODES,MODEL_SEEDS,CHANNELS,DISCOVERY_N,SCREEN_N,INITIAL_NEW_COEFF,
)

ARMS=("k3_legacy","k3_learned_shape","k5_legacy","k5_learned_shape")
EXPECTED_FITS=len(MODES)*len(MODEL_SEEDS)*len(ARMS)


def _extra_train_residual_axes(train,known_basis):
    matrix=train.x.reshape(-1,train.x.shape[-1]).astype(np.float64)
    q=np.asarray(known_basis,dtype=np.float64)
    projector=np.eye(q.shape[0])-q@q.T
    residual=(matrix-matrix.mean(axis=0,keepdims=True))@projector
    _,_,vt=np.linalg.svd(residual,full_matrices=False)
    directions=[]
    for axis in vt:
        direction=projector@axis
        for old in directions:
            direction=direction-old*np.dot(old,direction)
        norm=np.linalg.norm(direction)
        if norm>1e-6:
            direction=direction/norm
            pivot=int(np.argmax(np.abs(direction)))
            if direction[pivot]<0:
                direction=-direction
            directions.append(direction)
        if len(directions)==2:
            break
    if len(directions)!=2:
        raise ValueError("Need exactly two train-only residual capacity controls")
    expanded=np.column_stack([q,*directions])
    if not np.allclose(expanded.T@expanded,np.eye(5),atol=1e-4):
        raise ValueError("Uncertified overcapacity vectors fail orthogonality")
    return directions


def _add_unscreened_capacity(k3,directions):
    model=k3
    for i,d in enumerate(directions):
        candidate=ProposedObservableConcept(
            candidate_id=f"forced_training_svd_capacity_axis_{i}",
            direction=d,
            train_episode_count=800,
            positive_lag_strength=0.,
            permutation_p=1.,
            bootstrap_abs_cosine_q10=0.,
            top_signed_sensors=(("TRAIN_ONLY_SVD_NOT_CERTIFIED",1.),),
        )
        model=admit_candidate_without_prediction_drift(
            model,candidate,expert_decision="approve",
        )
    return model


def _as_shape(model):
    source=model.physical.source_predictive_basis.detach().cpu().numpy()
    alternative=MultiScaleDualProcessConceptFAN(
        observable_loadings=source,
        channels=model.physical.channels,
        forecast_horizon=model.physical.horizon,
        n_observation_concepts=model.process.n_concepts,
        scales=model.physical.scales,
        adaptive_feature_gradients=True,
        temporal_readout_mode="learned_shape",
    )
    missing,extra=alternative.load_state_dict(model.state_dict(),strict=False)
    if set(missing)!={"physical.temporal_readout_logits"} or extra:
        raise AssertionError("V19c changed native pretraining state outside shape mix")
    alternative.eval()
    return alternative


def _logits(model,x,mask):
    model.eval()
    with torch.no_grad():
        output=model(x,mask,process_mask=mask)
        return output.logits.detach().cpu().numpy(),output.physical.future_prediction.detach().cpu().numpy()


def _evaluate(model,split,candidate):
    model.eval()
    probabilities=[]
    source_effect=[]
    extras=[]
    raw=[]
    observed=[]
    error=0.
    with torch.no_grad():
        for start in range(0,len(split.y),128):
            stop=min(start+128,len(split.y))
            x=torch.from_numpy(split.x[start:stop])
            mask=torch.from_numpy(split.mask[start:stop])
            out=model(x,mask,process_mask=mask)
            probabilities.extend(out.probabilities.cpu().numpy().tolist())
            source_effect.extend(out.physical.contributions[:,2].cpu().numpy().tolist())
            if model.physical.n_slots>3:
                extras.extend(out.physical.contributions[:,3:].abs().sum(1).cpu().numpy().tolist())
            rel=out.physical.availability[:,-6:,2]
            act=out.physical.raw_activation[:,-6:,2]
            score=(act*rel).sum(1)/rel.sum(1).clamp_min(1e-8)
            raw.extend(score.cpu().numpy().tolist())
            sensor_score=x[:,-6:]@torch.as_tensor(candidate.direction,dtype=x.dtype)
            observed.extend(sensor_score.mean(1).cpu().numpy().tolist())
            error=max(error,float(out.decomposition_error()))
    if error>2e-5:
        raise AssertionError("Not an exact native additive decision")
    prediction=np.asarray(probabilities)
    rho=spearmanr(raw,observed).statistic
    shape_weight=(
        float(1-model.physical.temporal_readout_weights()[2,0].detach())
        if hasattr(model.physical,"temporal_readout_logits") else 0.
    )
    return {
        "heldout_auprc":float(average_precision_score(split.y,prediction)),
        "heldout_auroc":float(roc_auc_score(split.y,prediction)),
        "heldout_brier":float(brier_score_loss(split.y,prediction)),
        "max_native_decomposition_error":error,
        "mean_abs_third_source_direct_effect":float(np.mean(np.abs(source_effect))),
        "mean_abs_unscreened_extra_direct_effect":float(np.mean(extras)) if extras else None,
        "third_source_raw_vs_observed_rho":float(rho) if np.isfinite(rho) else None,
        "third_source_nonrecent_shape_weight":shape_weight,
        "third_source_patient_effect":np.asarray(source_effect,dtype=np.float64),
    }


def run(outdir:Path):
    if outdir.exists():
        raise FileExistsError("Never overwrite V19c scientific evidence")
    torch.set_num_threads(1)
    rows=[]
    recorded={mode:{arm:[] for arm in ARMS} for mode in MODES}
    screening={}
    for mode in MODES:
        data=make_data(mode)
        basis=np.eye(len(CHANNELS),2)
        candidate=propose_observable_residual_concept(
            data.train.x[:DISCOVERY_N],data.train.mask[:DISCOVERY_N],
            basis,CHANNELS,seed=20281,permutations=99,bootstraps=24,
        )
        frame=slice(DISCOVERY_N,DISCOVERY_N+SCREEN_N)
        evidence=screen_risk_association(
            candidate,data.train.x[frame],data.train.mask[frame],data.train.y[frame],
            screening_episode_ids=data.train.episode_ids[frame],
            discovery_episode_ids=data.train.episode_ids[:DISCOVERY_N],
            seed=20871,permutations=199,bootstraps=199,
        )
        screening[mode]=risk_evidence_passport(evidence)
        k3_basis=np.column_stack([basis,candidate.direction])
        residual_axes=_extra_train_residual_axes(data.train,k3_basis)
        config,dual,v10=_config()
        for seed in MODEL_SEEDS:
            original,_=fit_multiscale(
                data,basis,seed,config,dual,v10,
                consistency=False,feature_exploration=True,
            )
            original.eval()
            k3=admit_candidate_without_prediction_drift(
                original,candidate,expert_decision="approve",
            )
            with torch.no_grad():
                k3.physical.decision_raw_weight[2]=(
                    evidence.signed_orientation*INITIAL_NEW_COEFF
                    if evidence.status=="ACCEPT_PREDICTIVE_ASSOCIATION" else 0.
                )
            k5=_add_unscreened_capacity(k3,residual_axes)
            validation_x=torch.from_numpy(data.validation.x[:24])
            validation_m=torch.from_numpy(data.validation.mask[:24])
            old_risk,old_forecast=_logits(k3,validation_x,validation_m)
            extended_risk,extended_forecast=_logits(k5,validation_x,validation_m)
            if (np.max(np.abs(old_risk-extended_risk))>1e-6
                or np.max(np.abs(old_forecast-extended_forecast))>1e-6):
                raise AssertionError("Adding unchecked capacity changed model predictions before retraining")
            copies=[
                ("k3_legacy",k3),("k3_learned_shape",_as_shape(k3)),
                ("k5_legacy",k5),("k5_learned_shape",_as_shape(k5)),
            ]
            for arm,model in copies:
                model=_fit_continuation(
                    model,data,seed,config,dual,
                    candidate=None,grounding_weight=0.,
                )
                metric=_evaluate(model,data.validation,candidate)
                patient_effect=metric.pop("third_source_patient_effect")
                recorded[mode][arm].append(patient_effect)
                rows.append({
                    "mode":mode,"seed":seed,"arm":arm,
                    "n_native_physical_slots":model.physical.n_slots,
                    "forced_extra_slots_not_physically_certified":arm.startswith("k5"),
                    "risk_screen_status":evidence.status,
                    **metric,
                })
                print(mode,seed,arm,"AUPRC",round(metric["heldout_auprc"],6),
                      "extra_abs",round(metric["mean_abs_unscreened_extra_direct_effect"] or 0,5),
                      flush=True)
    summary=[]
    for mode in MODES:
        for arm in ARMS:
            group=[r for r in rows if r["mode"]==mode and r["arm"]==arm]
            vectors=recorded[mode][arm]
            correlations=[]
            for i in range(len(vectors)):
                for j in range(i+1,len(vectors)):
                    rho=spearmanr(vectors[i],vectors[j]).statistic
                    if np.isfinite(rho):correlations.append(float(rho))
            summary.append({
                "mode":mode,"arm":arm,
                "n_fits":len(group),
                "mean_validation_auprc":float(np.mean([r["heldout_auprc"] for r in group])),
                "mean_validation_brier":float(np.mean([r["heldout_brier"] for r in group])),
                "mean_abs_third_source_effect":float(np.mean([r["mean_abs_third_source_direct_effect"] for r in group])),
                "mean_abs_unscreened_extra_effect":float(np.mean([r["mean_abs_unscreened_extra_direct_effect"] for r in group])) if arm.startswith("k5") else None,
                "third_source_signed_effect_rank_rho_dependent_seed_pairs":float(np.mean(correlations)) if correlations else None,
            })
    if len(rows)!=EXPECTED_FITS:
        raise AssertionError("Must report every capacity ablation fit")
    outdir.mkdir(parents=True,exist_ok=False)
    for name,items in (("ALL_36_CAPACITY_FITS.csv",rows),("SUMMARY.csv",summary)):
        with (outdir/name).open("w",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=list(items[0]))
            writer.writeheader();writer.writerows(items)
    manifest={
        "status":"V19C_TRAIN_ONLY_FORCED_CONCEPT_CAPACITY_AND_NATIVE_SHAPE",
        "n_fits":len(rows),"modes":list(MODES),
        "initialization_seeds":list(MODEL_SEEDS),
        "arms":list(ARMS),
        "no_reuse_of_test_labels":True,
        "no_oracle_states_used_for_source_discovery":True,
        "unscreened_k4_k5_are_only_computational_capacity_controls":True,
        "null_mode_forced_training_is_not_approved_risk_interpretation":True,
        "new_source_k3_direction_from_train_only":True,
        "all_scenarios_and_seeds_reported":True,
        "previously_viewed_synthetic_validation":True,
        "no_clinical_causal_semantics":True,
        "no_patient_level_contribution_vectors_exported":True,
        "screening":screening,
        "results":rows,"aggregate":summary,
    }
    (outdir/"STATUS.json").write_text(json.dumps(manifest,indent=2,allow_nan=False))
    return manifest


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",required=True,type=Path)
    arguments=parser.parse_args()
    run(arguments.out_dir)
