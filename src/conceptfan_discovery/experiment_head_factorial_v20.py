"""V20: matched native risk-head optimizer/weighting experiment.

Every arm uses the SAME already trained ConceptFAN model and frozen native
C[t,k]. Optimized variables are only physical.decision_raw_weight and
physical.decision_bias. This is neither a new post-hoc decision model nor a
physical concept identification experiment.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,log_loss

from .candidate_admission_v17 import propose_observable_residual_concept
from .risk_screen_v18 import screen_risk_association,risk_evidence_passport
from .clinical_multiscale_v10 import fit_multiscale
from .experiment_signmatched_temporal_v19b import _new_model
from .experiment_native_risk_v18 import (
    make_data,_config,_fit_continuation,
    CHANNELS,MODES,MODEL_SEEDS,DISCOVERY_N,SCREEN_N,
)
from .native_head_factorial_v20 import fit_native_head_factorial_v20

ARMS=(
    "no_refit",
    "lbfgs_unweighted",
    "lbfgs_weighted",
    "adamw_unweighted",
    "adamw_weighted",
)


def _audit_nonhead(model):
    return {
        name:p.detach().clone()
        for name,p in model.state_dict().items()
        if name not in ("physical.decision_raw_weight","physical.decision_bias")
    }


def _predict(model,split):
    model.eval()
    probabilities=[]
    last_effect=[]
    max_additive=0.
    with torch.no_grad():
        for i in range(0,len(split.y),128):
            x=torch.from_numpy(split.x[i:i+128])
            mask=torch.from_numpy(split.mask[i:i+128])
            output=model(x,mask,process_mask=mask)
            probabilities.extend(output.probabilities.detach().cpu().numpy().tolist())
            last_effect.extend(output.physical.contributions[:,-1].detach().cpu().numpy().tolist())
            max_additive=max(max_additive,float(output.decomposition_error()))
    probs=np.asarray(probabilities,dtype=np.float64)
    if max_additive>2e-5:
        raise AssertionError("Concept-only decision graph no longer decomposes exactly")
    return {
        "validation_auprc":float(average_precision_score(split.y,probs)),
        "validation_auroc":float(roc_auc_score(split.y,probs)),
        "validation_brier":float(brier_score_loss(split.y,probs)),
        "validation_logloss":float(log_loss(split.y,probs,labels=[0,1])),
        "mean_absolute_final_concept_contribution":float(np.mean(np.abs(last_effect))),
        "max_native_additive_logit_error":max_additive,
    }


def run(out_dir:Path):
    if out_dir.exists():
        raise FileExistsError("Do not overwrite executed scientific evidence")
    torch.set_num_threads(1)
    evidence=[]
    screens={}
    for mode in MODES:
        data=make_data(mode)
        assert data.train.oracle_states is None and data.validation.oracle_states is None
        reference=np.eye(len(CHANNELS),2)
        candidate=propose_observable_residual_concept(
            data.train.x[:DISCOVERY_N],data.train.mask[:DISCOVERY_N],
            reference,CHANNELS,seed=20281,permutations=99,bootstraps=24,
        )
        screen=slice(DISCOVERY_N,DISCOVERY_N+SCREEN_N)
        risk=screen_risk_association(
            candidate,data.train.x[screen],data.train.mask[screen],
            data.train.y[screen],
            screening_episode_ids=data.train.episode_ids[screen],
            discovery_episode_ids=data.train.episode_ids[:DISCOVERY_N],
            seed=20871,permutations=199,bootstraps=199,
        )
        screens[mode]=risk_evidence_passport(risk)
        orientation=(risk.signed_orientation
                     if risk.status=="ACCEPT_PREDICTIVE_ASSOCIATION" else 0)
        conf,dual,v10=_config()
        for seed in MODEL_SEEDS:
            k2,_=fit_multiscale(
                data,reference,seed,conf,dual,v10,
                consistency=False,feature_exploration=True,
            )
            k2.eval()
            # Deliberately continue the proven original native legacy branch.
            k3=_new_model(k2,candidate,"legacy_signinit",orientation)
            k3=_fit_continuation(
                k3,data,seed,conf,dual,
                candidate=None,grounding_weight=0.,
            )
            baseline=_predict(k3,data.validation)
            original_head=torch.cat([
                k3.physical.decision_raw_weight.detach().flatten(),
                k3.physical.decision_bias.detach().reshape(1),
            ])
            original_other=_audit_nonhead(k3)
            for arm in ARMS:
                model=copy.deepcopy(k3)
                receipt=None
                if arm!="no_refit":
                    optimizer,class_weight=arm.split("_")
                    receipt=fit_native_head_factorial_v20(
                        model,data.train,
                        declared_train_episode_ids=data.train.episode_ids,
                        method=optimizer,
                        pos_weighted=(class_weight=="weighted"),
                        effective_beta_l2=.001,
                        lbfgs_max_iter=120,
                        adamw_steps=300,
                        adamw_lr=.03,
                    )
                for name,old in original_other.items():
                    if not torch.equal(model.state_dict()[name],old):
                        raise AssertionError("Native risk refit mutated other parameter: "+name)
                metrics=_predict(model,data.validation)
                beta=model.physical.decision_weights().detach().cpu().numpy()
                row={
                    "mode":mode,"seed":seed,"arm":arm,
                    "risk_screen_status":risk.status,
                    "risk_orient_from_train":orientation,
                    "null_source_forced_diagnostic_only":mode=="latent_risk_null",
                    "original_head_parameters_equal_at_arm_start":True,
                    "initial_native_head_raw_l2":float(torch.linalg.vector_norm(original_head)),
                    "train_positive_rate":float(data.train.y.mean()),
                    "train_label_pos_weight":float(
                        (len(data.train.y)-data.train.y.sum())/max(data.train.y.sum(),1.)),
                    "fitted_train_objective":receipt.final_objective if receipt else None,
                    "initial_train_objective":receipt.initial_objective if receipt else None,
                    "train_weighted_bce_after":receipt.final_weighted_bce if receipt else None,
                    "train_unweighted_bce_after":receipt.final_unweighted_bce if receipt else None,
                    "training_method":receipt.method if receipt else "original_multitask",
                    "training_uses_pos_weight":receipt.pos_weighted if receipt else True,
                    "n_concept_slots":model.physical.n_slots,
                    "mean_absolute_effective_native_coefficient":float(np.mean(np.abs(beta))),
                    "coefficient_near_tanh_saturation_fraction":float(np.mean(np.abs(beta)>2.7)),
                    "concepts_forecasts_observation_process_unchanged":True,
                    **metrics,
                    "delta_vs_original_validation_auprc":metrics["validation_auprc"]-baseline["validation_auprc"],
                }
                evidence.append(row)
                print(mode,seed,arm,
                      "AUPRC",round(row["validation_auprc"],6),
                      "Brier",round(row["validation_brier"],6),flush=True)
    aggregate=[]
    for mode in MODES:
        for arm in ARMS:
            rows=[r for r in evidence if r["mode"]==mode and r["arm"]==arm]
            aggregate.append({
                "mode":mode,"arm":arm,
                "number_model_initializations":len(rows),
                "mean_validation_auprc":float(np.mean([r["validation_auprc"] for r in rows])),
                "mean_validation_auroc":float(np.mean([r["validation_auroc"] for r in rows])),
                "mean_validation_brier":float(np.mean([r["validation_brier"] for r in rows])),
                "mean_delta_auprc_vs_multitask":float(np.mean([r["delta_vs_original_validation_auprc"] for r in rows])),
                "mean_coefficient_saturation_fraction":float(np.mean([r["coefficient_near_tanh_saturation_fraction"] for r in rows])),
            })
    if len(evidence)!=45 or len(aggregate)!=15:
        raise AssertionError("Unexpected missing optimizer/class-weight controls")
    out_dir.mkdir(parents=True,exist_ok=False)
    for name,rows in (("ALL_45_HEAD_CONTROLS.csv",evidence),("SUMMARY.csv",aggregate)):
        with (out_dir/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    receipt={
        "status":"V20_MATCHED_NATIVE_HEAD_OBJECTIVE_AND_OPTIMIZER_FACTORIAL",
        "n_train_only_native_head_evaluations":len(evidence),
        "n_distinct_pretrained_models":9,
        "n_extra_optimizer_refits":len(evidence)-9,
        "arms":list(ARMS),"modes":list(MODES),
        "model_initialization_seeds":list(MODEL_SEEDS),
        "same_concept_trajectories_and_init_per_arm":True,
        "only_original_native_decision_coefficients_updated":True,
        "no_new_posthoc_inference_model":True,
        "no_validation_labels_ever_read_by_optimizer":True,
        "test_and_external_real_data_sealed":True,
        "previously_viewed_synthetic_validation":True,
        "no_medical_causal_identification":True,
        "negative_null_risk_is_only_forced_diagnostic":True,
        "all_predeclared_cases_reported":True,
        "screens":screens,"records":evidence,"aggregate":aggregate,
    }
    (out_dir/"STATUS.json").write_text(json.dumps(receipt,indent=2,allow_nan=False))
    return receipt


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",required=True,type=Path)
    args=p.parse_args()
    run(args.out_dir)
