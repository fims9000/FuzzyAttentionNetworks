"""V18b: controlled test of the risk-policy *initialization confound*.

The V18 negative-risk improvement may reflect merely starting a novel risk
coefficient on the correct side of zero, rather than restricting its future
sign. This protocol holds initial *effective* signed coefficients identical
between a signed constraint and an unrestricted native risk readout.
It never uses validation labels to choose initializations or gates.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from .candidate_admission_v17 import (
    propose_observable_residual_concept,admit_candidate_without_prediction_drift,
)
from .risk_screen_v18 import (
    screen_risk_association,configure_native_risk_orientation,
    begin_native_risk_training,risk_evidence_passport,
)
from .experiment_native_risk_v18 import (
    make_data,_config,_fit_continuation,_evaluate,
    CHANNELS,DISCOVERY_N,SCREEN_N,MODEL_SEEDS,INITIAL_NEW_COEFF,
)
from .clinical_multiscale_v10 import fit_multiscale

MODES=("latent_predictive_positive","latent_predictive_negative","latent_risk_null")
ARMS=("unrestricted_correct_signed_start",
      "restricted_correct_signed_start",
      "unrestricted_opposite_signed_start")


def _pretraining_score(model,x,mask):
    model.eval()
    with torch.no_grad():
        result=model(x,mask,process_mask=mask)
        return result.logits.detach().cpu().numpy()


def run(outdir:Path):
    if outdir.exists():
        raise FileExistsError("Never overwrite executed scientific evidence")
    torch.set_num_threads(1)
    records=[]
    pairs={mode:{arm:[] for arm in ARMS} for mode in MODES}
    screen_receipts={}
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
        screen_receipts[mode]=risk_evidence_passport(evidence)
        for seed in MODEL_SEEDS:
            if evidence.status!="ACCEPT_PREDICTIVE_ASSOCIATION":
                for arm in ARMS:
                    records.append({
                        "mode":mode,"seed":seed,"arm":arm,
                        "status":"ABSTAIN_INDEPENDENT_TRAIN_SCREEN",
                        "screen_orientation":0,
                        "initial_native_coefficient":None,
                        "max_logit_diff_from_correct_start":None,
                        "heldout_auprc":None,"heldout_auroc":None,
                        "heldout_brier":None,"max_additive_error":None,
                        "mean_abs_new_slot_effect":None,
                        "new_slot_native_coefficient":None,
                        "native_raw_vs_observed_direction_spearman":None,
                    })
                continue
            base,dual,v10=_config()
            original,_=fit_multiscale(
                data,basis,seed,base,dual,v10,
                consistency=False,feature_exploration=True,
            )
            original.eval()
            created=[]
            for arm in ARMS:
                expanded=admit_candidate_without_prediction_drift(
                    original,candidate,expert_decision="approve",
                )
                if arm=="restricted_correct_signed_start":
                    configure_native_risk_orientation(
                        expanded,candidate,evidence,slot=2,
                    )
                    begin_native_risk_training(
                        expanded,slot=2,initial_coefficient_raw=INITIAL_NEW_COEFF,
                    )
                else:
                    sign=(evidence.signed_orientation if
                          arm=="unrestricted_correct_signed_start" else
                          -evidence.signed_orientation)
                    with torch.no_grad():
                        expanded.physical.decision_raw_weight[-1]=float(sign)*INITIAL_NEW_COEFF
                expanded.eval()
                created.append((arm,expanded))
            first=[m for a,m in created if a=="unrestricted_correct_signed_start"][0]
            signed=[m for a,m in created if a=="restricted_correct_signed_start"][0]
            x=torch.from_numpy(data.validation.x[:32])
            mask=torch.from_numpy(data.validation.mask[:32])
            before=_pretraining_score(first,x,mask)
            after=_pretraining_score(signed,x,mask)
            match=float(np.max(np.abs(before-after)))
            if match>1e-6:
                raise AssertionError("Unrestricted and constrained signed initial logits mismatch")

            for arm,expanded in created:
                init_coeff=float(expanded.physical.decision_weights()[-1].detach())
                model=_fit_continuation(
                    expanded,data,seed,base,dual,
                    candidate=None,grounding_weight=0,
                )
                scored=_evaluate(model,data.validation,candidate)
                vector=scored.pop("patient_signed_effect_vector")
                pairs[mode][arm].append(vector)
                records.append({
                    "mode":mode,"seed":seed,"arm":arm,"status":"TRAINED",
                    "screen_orientation":evidence.signed_orientation,
                    "initial_native_coefficient":init_coeff,
                    "max_logit_diff_from_correct_start":match if arm!="unrestricted_opposite_signed_start" else None,
                    **scored,
                })
                print(mode,seed,arm,round(scored["heldout_auprc"],6),
                      round(scored["new_slot_native_coefficient"],6),flush=True)
    summary=[]
    for mode in MODES:
        for arm in ARMS:
            subset=[r for r in records if r["mode"]==mode and r["arm"]==arm]
            complete=[r for r in subset if r["status"]=="TRAINED"]
            trajectories=pairs[mode][arm]
            corr=[]
            for i in range(len(trajectories)):
                for j in range(i+1,len(trajectories)):
                    value=spearmanr(trajectories[i],trajectories[j]).statistic
                    if np.isfinite(value):corr.append(float(value))
            summary.append({
                "mode":mode,"arm":arm,
                "opportunities":len(subset),
                "trained":len(complete),
                "abstentions":len(subset)-len(complete),
                "mean_heldout_auprc":float(np.mean([r["heldout_auprc"] for r in complete])) if complete else None,
                "mean_abs_new_slot_effect":float(np.mean([r["mean_abs_new_slot_effect"] for r in complete])) if complete else None,
                "mean_contribution_crossseed_spearman":float(np.mean(corr)) if corr else None,
            })
    outdir.mkdir(parents=True,exist_ok=False)
    for name,items in (("ALL_INIT_MATCHED_FITS_AND_REFUSALS.csv",records),("SUMMARY.csv",summary)):
        with (outdir/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)
    receipt={
        "status":"V18B_SAME_EFFECTIVE_SIGN_INITIALIZATION_CONTROL",
        "all_opportunities":len(records),
        "trained":sum(r["status"]=="TRAINED" for r in records),
        "abstentions":sum(r["status"]!="TRAINED" for r in records),
        "discovery_screen_train_disjoint":True,
        "no_patient_specific_results_exported":True,
        "initial_effective_logits_exactly_matched_between_primary_arms":True,
        "sign_constraint_is_reparameterization_until_boundary_crossing":True,
        "clinical_causal_identity_not_proven":True,
        "accepted_PIERE_unchanged":True,
        "risk_screens":screen_receipts,
        "all_cases":records,
        "summary":summary,
    }
    (outdir/"STATUS.json").write_text(json.dumps(receipt,indent=2,allow_nan=False))
    return receipt


if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-dir",type=Path,required=True)
    args=ap.parse_args()
    run(args.out_dir)
