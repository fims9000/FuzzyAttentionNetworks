"""V19e: TRAIN-only actual native risk parameter refit, no post-hoc head.

Compares the already trained native additive risk head against a second
TRAIN-only optimization phase on the SAME native concept scores, keeping
all concept formation and auxiliary forecast/denoising paths unchanged.
The 200-sample V18 validation cohort is previously inspected; results
are not confirmatory scientific or clinical generalization claims.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .candidate_admission_v17 import propose_observable_residual_concept
from .risk_screen_v18 import screen_risk_association,risk_evidence_passport
from .clinical_multiscale_v10 import fit_multiscale
from .experiment_native_risk_v18 import (
    make_data,_config,_fit_continuation,
    CHANNELS,MODES,MODEL_SEEDS,DISCOVERY_N,SCREEN_N,
)
from .experiment_signmatched_temporal_v19b import _new_model
from .experiment_native_probe_v19d import (
    _features_and_native_predictions,_metrics,
)
from .native_head_refit_v19e import fit_train_only_native_risk_head

ARMS=("legacy_signinit","learned_shape_signinit")
N_FITS=len(MODES)*len(MODEL_SEEDS)*len(ARMS)


def _fixed_nonhead_model(model):
    return {
        name:t.detach().clone()
        for name,t in model.state_dict().items()
        if name not in ("physical.decision_raw_weight","physical.decision_bias")
    }


def _assert_other_params_unchanged(model,saved):
    for key,pre in saved.items():
        current=model.state_dict()[key]
        if not torch.equal(current,pre):
            raise AssertionError(f"Head refit changed nondecision native tensor {key}")


def run(outdir:Path):
    if outdir.exists():
        raise FileExistsError("Never overwrite executed V19e evidence")
    torch.set_num_threads(1)
    rows=[]
    passports={}
    for mode in MODES:
        data=make_data(mode)
        prior=np.eye(len(CHANNELS),2)
        candidate=propose_observable_residual_concept(
            data.train.x[:DISCOVERY_N],data.train.mask[:DISCOVERY_N],
            prior,CHANNELS,seed=20281,permutations=99,bootstraps=24,
        )
        part=slice(DISCOVERY_N,DISCOVERY_N+SCREEN_N)
        evidence=screen_risk_association(
            candidate,data.train.x[part],data.train.mask[part],
            data.train.y[part],
            screening_episode_ids=data.train.episode_ids[part],
            discovery_episode_ids=data.train.episode_ids[:DISCOVERY_N],
            seed=20871,permutations=199,bootstraps=199,
        )
        passports[mode]=risk_evidence_passport(evidence)
        config,dual,v10=_config()
        for seed in MODEL_SEEDS:
            original,_=fit_multiscale(
                data,prior,seed,config,dual,v10,
                consistency=False,feature_exploration=True,
            )
            original.eval()
            for arm in ARMS:
                orientation=(
                    evidence.signed_orientation
                    if evidence.status=="ACCEPT_PREDICTIVE_ASSOCIATION" else 0
                )
                model=_new_model(original,candidate,arm,orientation)
                model=_fit_continuation(
                    model,data,seed,config,dual,
                    candidate=None,grounding_weight=0.,
                )
                before_train_concepts,before_train_preds,train_error=_features_and_native_predictions(
                    model,data.train,
                )
                before_val_concepts,before_val_preds,old_error=_features_and_native_predictions(
                    model,data.validation,
                )
                frozen=_fixed_nonhead_model(model)
                fitted=fit_train_only_native_risk_head(
                    model,data.train,
                    declared_train_episode_ids=data.train.episode_ids,
                    weight_l2=.001,max_iter=120,
                )
                _assert_other_params_unchanged(model,frozen)
                after_train_concepts,after_train_preds,train_error_new=_features_and_native_predictions(
                    model,data.train,
                )
                after_val_concepts,after_val_preds,new_error=_features_and_native_predictions(
                    model,data.validation,
                )
                if (not np.array_equal(before_train_concepts,after_train_concepts) or
                    not np.array_equal(before_val_concepts,after_val_concepts)):
                    raise AssertionError("Native head refit silently changed concept representations")
                if max(train_error,old_error,train_error_new,new_error)>2e-5:
                    raise AssertionError("Native exact additive explanation broken")
                previous=_metrics(data.validation.y,before_val_preds)
                updated=_metrics(data.validation.y,after_val_preds)
                row={
                    "mode":mode,"seed":seed,"arm":arm,
                    "status":"NATIVE_HEAD_REFIT_EXECUTED",
                    "screen_status":evidence.status,
                    "initial_new_slot_sign_from_train_only":orientation,
                    "n_train":fitted.n_train_episodes,
                    "n_native_physical_concepts":fitted.native_slots,
                    "initial_train_objective":fitted.original_objective,
                    "final_train_objective":fitted.final_objective,
                    "initial_train_bce":fitted.original_train_bce,
                    "final_train_bce":fitted.final_train_bce,
                    "native_before_val_auprc":previous["auprc"],
                    "native_after_val_auprc":updated["auprc"],
                    "after_minus_before_auprc":updated["auprc"]-previous["auprc"],
                    "native_before_val_auroc":previous["auroc"],
                    "native_after_val_auroc":updated["auroc"],
                    "native_before_val_brier":previous["brier"],
                    "native_after_val_brier":updated["brier"],
                    "unchanged_native_concept_trajectories":True,
                    "unchanged_process_head_and_forecast_decoders":True,
                    "max_after_additive_error":new_error,
                    "native_third_slot_risk_beta_after":float(model.physical.decision_weights()[-1].detach()),
                }
                rows.append(row)
                print(mode,seed,arm,"before",round(previous["auprc"],5),
                      "after",round(updated["auprc"],5),flush=True)
    summary=[]
    for mode in MODES:
        for arm in ARMS:
            selected=[r for r in rows if r["mode"]==mode and r["arm"]==arm]
            summary.append({
                "mode":mode,"arm":arm,"n_seeds":len(selected),
                "mean_native_auprc_before":float(np.mean([r["native_before_val_auprc"] for r in selected])),
                "mean_native_auprc_after":float(np.mean([r["native_after_val_auprc"] for r in selected])),
                "mean_native_auprc_gain":float(np.mean([r["after_minus_before_auprc"] for r in selected])),
                "mean_native_brier_before":float(np.mean([r["native_before_val_brier"] for r in selected])),
                "mean_native_brier_after":float(np.mean([r["native_after_val_brier"] for r in selected])),
                "train_objective_never_increased":all(
                    r["final_train_objective"]<=r["initial_train_objective"]+1e-6
                    for r in selected
                ),
            })
    if len(rows)!=N_FITS:
        raise AssertionError("V19e full denominator must include all experiments")
    outdir.mkdir(parents=True,exist_ok=False)
    for filename,items in (("ALL_18_NATIVE_HEAD_REFITS.csv",rows),("SUMMARY.csv",summary)):
        with (outdir/filename).open("w",newline="") as file:
            w=csv.DictWriter(file,fieldnames=list(items[0]))
            w.writeheader();w.writerows(items)
    receipt={
        "status":"V19E_TRAIN_ONLY_REFIT_EXISTING_NATIVE_RISK_HEAD",
        "n_native_pretrained_models":len(rows),
        "n_direct_native_head_refits":len(rows),
        "no_external_risk_student_head_at_inference":True,
        "no_raw_input_to_logit_shortcut":True,
        "physical_concept_parameters_frozen_during_refit":True,
        "mask_process_head_and_forecast_unchanged":True,
        "source_proposal_risk_screen_train_disjoint":True,
        "test_cohort_sealed":80,
        "already_inspected_validation":True,
        "null_risk_forced_fit_only_diagnostic":True,
        "no_clinical_semantic_identification":True,
        "complete_denominator_preserved":True,
        "risk_screens":passports,
        "all_fits":rows,
        "summary":summary,
    }
    (outdir/"STATUS.json").write_text(json.dumps(receipt,indent=2,allow_nan=False))
    return receipt


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",required=True,type=Path)
    args=p.parse_args()
    run(args.out_dir)
