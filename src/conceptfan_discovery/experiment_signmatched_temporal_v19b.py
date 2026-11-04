"""V19b: exact signed-initialization control of learned temporal readout.

A controlled follow-up to V19's negative result. Only native K3 ConceptFAN
prediction paths are used. A physically unverified observed source direction
is proposed on one TRAIN subset, risk orientation is screened on a disjoint
TRAIN subset, then all arms share the same prefit native K2 parameters.
No synthetic oracle or validation labels are used to initialize the readout.
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
    screen_risk_association,risk_evidence_passport,
)
from .multiscale_physiology_v10 import MultiScaleDualProcessConceptFAN
from .clinical_multiscale_v10 import fit_multiscale
from .experiment_native_risk_v18 import (
    make_data,_config,_fit_continuation,_evaluate,
    MODES,MODEL_SEEDS,CHANNELS,DISCOVERY_N,SCREEN_N,INITIAL_NEW_COEFF,
)

ARMS=("legacy_signinit","fixed_shape_signinit","learned_shape_signinit")
EXPECTED_OPPORTUNITIES=len(MODES)*len(MODEL_SEEDS)*len(ARMS)


def _new_model(parent,candidate,mode,orientation):
    base=admit_candidate_without_prediction_drift(
        parent,candidate,expert_decision="approve",
    )
    with torch.no_grad():
        base.physical.decision_raw_weight[-1]=orientation*INITIAL_NEW_COEFF
    if mode=="legacy_signinit":
        base.eval()
        return base
    shape_mode={
        "fixed_shape_signinit":"fixed_shape",
        "learned_shape_signinit":"learned_shape",
    }[mode]
    source=base.physical.source_predictive_basis.detach().cpu().numpy()
    child=MultiScaleDualProcessConceptFAN(
        observable_loadings=source,
        channels=base.physical.channels,
        forecast_horizon=base.physical.horizon,
        n_observation_concepts=base.process.n_concepts,
        scales=base.physical.scales,
        adaptive_feature_gradients=True,
        temporal_readout_mode=shape_mode,
    )
    missing,extra=child.load_state_dict(base.state_dict(),strict=False)
    if set(missing)!={"physical.temporal_readout_logits"} or extra:
        raise AssertionError("V19 temporal readout changed pre-trained physical/state paths")
    child.eval()
    return child


def run(directory:Path):
    if directory.exists():
        raise FileExistsError("Do not overwrite executed V19b evidence")
    torch.set_num_threads(1)
    data_rows=[]
    vectors={mode:{arm:[] for arm in ARMS} for mode in MODES}
    passports={}
    for mode in MODES:
        data=make_data(mode)
        prior=np.eye(len(CHANNELS),2)
        candidate=propose_observable_residual_concept(
            data.train.x[:DISCOVERY_N],data.train.mask[:DISCOVERY_N],
            prior,CHANNELS,seed=20281,permutations=99,bootstraps=24,
        )
        section=slice(DISCOVERY_N,DISCOVERY_N+SCREEN_N)
        evidence=screen_risk_association(
            candidate,data.train.x[section],data.train.mask[section],
            data.train.y[section],
            screening_episode_ids=data.train.episode_ids[section],
            discovery_episode_ids=data.train.episode_ids[:DISCOVERY_N],
            seed=20871,permutations=199,bootstraps=199,
        )
        passports[mode]=risk_evidence_passport(evidence)
        print(mode,evidence.status,"TRAIN-only-sign",evidence.signed_orientation,flush=True)
        base_config,dual,v10=_config()
        for seed in MODEL_SEEDS:
            if evidence.status!="ACCEPT_PREDICTIVE_ASSOCIATION":
                for arm in ARMS:
                    data_rows.append({
                        "mode":mode,"seed":seed,"arm":arm,
                        "status":"ABSTAIN_TRAIN_ONLY_OUTCOME_SCREEN",
                        "screen_orientation":0,
                        "fixed_vs_learned_pretrain_logits_max_abs_error":None,
                        "heldout_auprc":None,"heldout_auroc":None,
                        "heldout_brier":None,"max_additive_error":None,
                        "mean_abs_new_slot_effect":None,
                        "new_slot_native_coefficient":None,
                        "native_raw_vs_observed_direction_spearman":None,
                        "nonrecent_shape_weight_slot3":None,
                    })
                continue
            original,_=fit_multiscale(
                data,prior,seed,base_config,dual,v10,
                consistency=False,feature_exploration=True,
            )
            original.eval()
            prepared=[(arm,_new_model(original,candidate,arm,evidence.signed_orientation)) for arm in ARMS]
            fixed=[m for a,m in prepared if a=="fixed_shape_signinit"][0]
            learned=[m for a,m in prepared if a=="learned_shape_signinit"][0]
            x=torch.from_numpy(data.validation.x[:24])
            m=torch.from_numpy(data.validation.mask[:24])
            with torch.no_grad():
                first=fixed(x,m,process_mask=m).logits
                second=learned(x,m,process_mask=m).logits
            equal_error=float((first-second).abs().max())
            if equal_error > 1e-6:
                raise AssertionError("Frozen and trainable V19 summaries differ before fitting")
            for arm,model in prepared:
                model=_fit_continuation(
                    model,data,seed,base_config,dual,
                    candidate=None,grounding_weight=0.,
                )
                metrics=_evaluate(model,data.validation,candidate)
                patient_effects=metrics.pop("patient_signed_effect_vector")
                vectors[mode][arm].append(patient_effects)
                novelty=(
                    float(1-model.physical.temporal_readout_weights()[-1,0].detach())
                    if arm!="legacy_signinit" else 0.
                )
                data_rows.append({
                    "mode":mode,"seed":seed,"arm":arm,"status":"TRAINED",
                    "screen_orientation":evidence.signed_orientation,
                    "fixed_vs_learned_pretrain_logits_max_abs_error":equal_error,
                    **metrics,
                    "nonrecent_shape_weight_slot3":novelty,
                })
                print(mode,seed,arm,round(metrics["heldout_auprc"],6),
                      "shape_more_than_recent",round(novelty,4),flush=True)
    summary=[]
    for mode in MODES:
        for arm in ARMS:
            sub=[r for r in data_rows if r["mode"]==mode and r["arm"]==arm]
            complete=[r for r in sub if r["status"]=="TRAINED"]
            effects=vectors[mode][arm]
            correlations=[]
            for i in range(len(effects)):
                for j in range(i+1,len(effects)):
                    rho=spearmanr(effects[i],effects[j]).statistic
                    if np.isfinite(rho):
                        correlations.append(float(rho))
            summary.append({
                "mode":mode,"arm":arm,"opportunities":len(sub),
                "trained":len(complete),
                "abstentions":len(sub)-len(complete),
                "mean_heldout_auprc":float(np.mean([r["heldout_auprc"] for r in complete])) if complete else None,
                "mean_heldout_brier":float(np.mean([r["heldout_brier"] for r in complete])) if complete else None,
                "mean_abs_new_slot_effect":float(np.mean([r["mean_abs_new_slot_effect"] for r in complete])) if complete else None,
                "mean_nonrecent_shape_weight_slot3":float(np.mean([r["nonrecent_shape_weight_slot3"] for r in complete])) if complete else None,
                "signed_contribution_crossseed_spearman_3_dependent_pairs":float(np.mean(correlations)) if correlations else None,
            })
    if len(data_rows)!=EXPECTED_OPPORTUNITIES:
        raise AssertionError("V19b denominator mismatch")
    directory.mkdir(parents=True,exist_ok=False)
    for name,records in (
        ("ALL_SIGNMATCHED_FITS_AND_REFUSALS.csv",data_rows),
        ("SUMMARY.csv",summary),
    ):
        with (directory/name).open("w",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=list(records[0]))
            writer.writeheader();writer.writerows(records)
    receipt={
        "status":"V19B_TRAIN_RISK_SIGN_MATCHED_TEMPORAL_READOUT_CONTROL",
        "opportunities":len(data_rows),
        "trained":sum(r["status"]=="TRAINED" for r in data_rows),
        "abstentions":sum(r["status"]!="TRAINED" for r in data_rows),
        "source_and_risk_screen_train_disjoint":True,
        "same_initial_native_risk_sign_all_arms":True,
        "fixed_and_learned_shape_readouts_identical_before_training":True,
        "test_cohort_sealed":80,
        "already_inspected_synthetic_validation":True,
        "no_clinical_semantics_proven":True,
        "no_oracle_latent_labels_in_training":True,
        "no_individual_patient_effects_exported":True,
        "all_opportunities_reported":True,
        "screen_receipts":passports,
        "per_seed":data_rows,
        "summary":summary,
    }
    (directory/"STATUS.json").write_text(json.dumps(receipt,indent=2,allow_nan=False))
    return receipt


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",required=True,type=Path)
    arguments=parser.parse_args()
    run(arguments.out_dir)
