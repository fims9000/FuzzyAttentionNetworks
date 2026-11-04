"""V20e: extra full joint epochs against the actual native-head-only refit.

All arms start from precisely one shared 6-epoch K2 + 6-epoch K3
ConceptFAN model. The full-joint arms retain all old reconstruction,
forecasting and signed native task losses. The head-only arm edits solely
the original native decision coefficients. No raw-input bypass.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .candidate_admission_v17 import propose_observable_residual_concept
from .risk_screen_v18 import screen_risk_association,risk_evidence_passport
from .clinical_multiscale_v10 import fit_multiscale
from .experiment_signmatched_temporal_v19b import _new_model
from .experiment_native_risk_v18 import (
    make_data,_config,_fit_continuation,
    CHANNELS,MODES,MODEL_SEEDS,DISCOVERY_N,SCREEN_N,
)
from .experiment_head_factorial_v20 import _predict
from .native_head_factorial_v20 import fit_native_head_factorial_v20


ARMS=("baseline_joint_6plus6","extra_joint_6","extra_joint_12","frozen_head_adamw300")


def _native_concept_summaries(model,split):
    model.eval()
    result=[]
    with torch.no_grad():
        for i in range(0,len(split.y),128):
            x=torch.from_numpy(split.x[i:i+128])
            m=torch.from_numpy(split.mask[i:i+128])
            output=model(x,m,process_mask=m)
            result.append(output.physical.concept_summary.cpu().numpy())
    return np.concatenate(result,axis=0)


def _continue_original_joint(model,data,seed,base,dual,n_blocks):
    current=model
    for block in range(1,n_blocks+1):
        config=copy.deepcopy(base)
        # Each block uses the SAME V10 multitask loss/optimizer/hyperparams,
        # but a distinct frozen-order/mask seed to avoid replaying the exact
        # same synthetic dropout schedule. Optimizer restarts are disclosed.
        config["training"]["batch_order_seed"]=(
            int(base["training"]["batch_order_seed"])+10000*block
        )
        current=_fit_continuation(
            current,data,seed+111*block,config,dual,
            candidate=None,grounding_weight=0.,
        )
    return current


def run(directory:Path):
    if directory.exists():
        raise FileExistsError("Never overwrite V20e prior scientific outcomes")
    torch.set_num_threads(1)
    outcomes=[]
    screens={}
    for mode in MODES:
        data=make_data(mode)
        prior=np.eye(len(CHANNELS),2)
        candidate=propose_observable_residual_concept(
            data.train.x[:DISCOVERY_N],data.train.mask[:DISCOVERY_N],
            prior,CHANNELS,seed=20281,permutations=99,bootstraps=24,
        )
        gate_slice=slice(DISCOVERY_N,DISCOVERY_N+SCREEN_N)
        evidence=screen_risk_association(
            candidate,data.train.x[gate_slice],data.train.mask[gate_slice],
            data.train.y[gate_slice],
            screening_episode_ids=data.train.episode_ids[gate_slice],
            discovery_episode_ids=data.train.episode_ids[:DISCOVERY_N],
            seed=20871,permutations=199,bootstraps=199,
        )
        screens[mode]=risk_evidence_passport(evidence)
        orientation=evidence.signed_orientation if evidence.status=="ACCEPT_PREDICTIVE_ASSOCIATION" else 0
        config,dual,v10=_config()
        for seed in MODEL_SEEDS:
            k2,_=fit_multiscale(
                data,prior,seed,config,dual,v10,
                consistency=False,feature_exploration=True,
            )
            k2.eval()
            baseline=_new_model(k2,candidate,"legacy_signinit",orientation)
            baseline=_fit_continuation(
                baseline,data,seed,config,dual,
                candidate=None,grounding_weight=0.,
            )
            base_summary=_native_concept_summaries(baseline,data.validation)
            base_other={
                name:t.detach().clone()
                for name,t in baseline.state_dict().items()
                if name not in ("physical.decision_raw_weight","physical.decision_bias")
            }
            for arm in ARMS:
                model=copy.deepcopy(baseline)
                if arm=="extra_joint_6":
                    model=_continue_original_joint(model,data,seed,config,dual,n_blocks=1)
                elif arm=="extra_joint_12":
                    model=_continue_original_joint(model,data,seed,config,dual,n_blocks=2)
                elif arm=="frozen_head_adamw300":
                    fit_native_head_factorial_v20(
                        model,data.train,
                        declared_train_episode_ids=data.train.episode_ids,
                        method="adamw",pos_weighted=False,
                        effective_beta_l2=.001,
                        adamw_steps=300,adamw_lr=.03,
                    )
                    for key,value in base_other.items():
                        if not torch.equal(model.state_dict()[key],value):
                            raise AssertionError("Frozen native HEAD refit changed original concepts")
                elif arm!="baseline_joint_6plus6":
                    raise ValueError("Unknown predeclared V20e arm")
                val=_predict(model,data.validation)
                train=_predict(model,data.train)
                shape=_native_concept_summaries(model,data.validation)
                drift=float(np.mean(np.abs(shape-base_summary)))
                if arm=="frozen_head_adamw300" and drift!=0.:
                    raise AssertionError("Head-only refit must not alter native concept summaries")
                outcomes.append({
                    "mode":mode,"seed":seed,"arm":arm,
                    "initial_screen_status":evidence.status,
                    "train_signed_source_orientation":orientation,
                    "initial_fit_epochs_k2":6,
                    "initial_fit_epochs_k3":6,
                    "additional_joint_epochs":(6 if arm=="extra_joint_6" else 12 if arm=="extra_joint_12" else 0),
                    "native_head_only_adamw_steps":300 if arm=="frozen_head_adamw300" else 0,
                    "optimizer_restart_per_joint_six_epoch_block":arm.startswith("extra_joint"),
                    "new_concept_oracle_used":False,
                    "validation_auprc":val["validation_auprc"],
                    "validation_auroc":val["validation_auroc"],
                    "validation_brier":val["validation_brier"],
                    "train_auprc":train["validation_auprc"],
                    "train_brier":train["validation_brier"],
                    "mean_abs_new_slot_logit_contribution":val["mean_absolute_final_concept_contribution"],
                    "mean_native_concept_summary_drift_from_common_baseline":drift,
                    "max_exact_additive_error":val["max_native_additive_logit_error"],
                })
                print(mode,seed,arm,"validation AUPRC",round(val["validation_auprc"],6),
                      "concept drift",round(drift,6),flush=True)
    summaries=[]
    for mode in MODES:
        for arm in ARMS:
            group=[r for r in outcomes if r["mode"]==mode and r["arm"]==arm]
            summaries.append({
                "mode":mode,"arm":arm,"n_independent_initializations":len(group),
                "mean_validation_auprc":float(np.mean([r["validation_auprc"] for r in group])),
                "mean_validation_brier":float(np.mean([r["validation_brier"] for r in group])),
                "mean_train_auprc":float(np.mean([r["train_auprc"] for r in group])),
                "mean_native_concept_summary_drift":float(np.mean([r["mean_native_concept_summary_drift_from_common_baseline"] for r in group])),
                "not_exactly_matched_compute_or_task_objective":True,
            })
    if len(outcomes)!=36 or len(summaries)!=12:
        raise AssertionError("V20e full joint-epoch and head-only denominator incomplete")
    directory.mkdir(parents=True,exist_ok=False)
    for filename,records in (("ALL_36_JOINT_VERSUS_HEAD_FITS.csv",outcomes),("SUMMARY.csv",summaries)):
        with (directory/filename).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]))
            w.writeheader();w.writerows(records)
    receipt={
        "status":"V20E_EXTRA_JOINT_EPOCHS_VERSUS_HEAD_ONLY_NATIVE_OPTIMIZATION",
        "complete_native_evaluations":len(outcomes),
        "distinct_original_k2_to_k3_models":9,
        "all_predeclared_modes_reported":True,
        "no_test_unsealing":True,
        "concept_head_frozen_for_head_only":True,
        "original_combined_loss_in_joint_arms":True,
        "extra_joint_arms_restart_optimizer_between_blocks":True,
        "not_compute_matched_due_different_objectives":True,
        "null_outcome_forced_fit_is_negative_control":True,
        "no_clinical_physical_source_identification":True,
        "data_already_inspected_in_previous_synthetic_validation":True,
        "screens":screens,"rows":outcomes,"summary":summaries,
    }
    (directory/"STATUS.json").write_text(json.dumps(receipt,indent=2,allow_nan=False))
    return receipt


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    run(args.out_dir)
