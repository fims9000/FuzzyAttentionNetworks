"""V21b: isolate TRAIN-only risk-sign initialization and actual signed stability.

No hard sign restrictions, no post-hoc patient-effect sign realignment. A
concept may have a consistent *instrument-artifact* risk effect and still
be semantically wrong. Synthetic oracle used only for postfit audit.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from .provenance_data_v21 import (
    FAMILIES,CHANNELS,DISCOVERY_N,SCREEN_N,CORROBORATION_N,
    make_v21_provenance_cohort,load_oracle_source_truth_for_postfit_audit,
)
from .candidate_admission_v17 import propose_observable_residual_concept
from .risk_screen_v18 import screen_risk_association,risk_evidence_passport
from .measurement_corroboration_v21 import (
    corroborate_candidate_on_disjoint_measured_reference,provenance_passport,
)
from .clinical_multiscale_v10 import fit_multiscale
from .experiment_native_risk_v18 import _config
from .experiment_provenance_v21 import (
    MODEL_SEEDS,EPOCHS,BATCH,_evaluate_without_oracle,
)

ARMS=("arbitrary_native_start","train_screened_signed_start")
INITIAL_RAW_EFFECT=.05
OPPORTUNITIES=len(FAMILIES)*len(MODEL_SEEDS)*len(ARMS)


def _safe_spearman(a,b):
    stat=spearmanr(a,b).statistic
    return float(stat) if np.isfinite(stat) else None


def run(outdir:Path):
    if outdir.exists():
        raise FileExistsError("Never overwrite V21b signed-effect evidence")
    torch.set_num_threads(1)
    records=[]
    vectors={family:{arm:[] for arm in ARMS} for family in FAMILIES}
    screens={}
    for family in FAMILIES:
        cohort=make_v21_provenance_cohort(family)
        data=cohort.discovery
        if data.train.oracle_states is not None or data.validation.oracle_states is not None:
            raise AssertionError("Physical oracle state accidentally exposed to native model")
        prior=np.eye(len(CHANNELS),2)
        source=slice(0,DISCOVERY_N)
        risk_part=slice(DISCOVERY_N,DISCOVERY_N+SCREEN_N)
        check=slice(DISCOVERY_N+SCREEN_N,DISCOVERY_N+SCREEN_N+CORROBORATION_N)
        candidate=propose_observable_residual_concept(
            data.train.x[source],data.train.mask[source],
            prior,CHANNELS,seed=211131,permutations=199,bootstraps=32,
        )
        risk=screen_risk_association(
            candidate,data.train.x[risk_part],data.train.mask[risk_part],
            data.train.y[risk_part],
            screening_episode_ids=data.train.episode_ids[risk_part],
            discovery_episode_ids=data.train.episode_ids[source],
            seed=211132,permutations=199,bootstraps=199,
        )
        corroboration=corroborate_candidate_on_disjoint_measured_reference(
            candidate,
            data.train.x[check],cohort.independent_train_reference[check],
            provenance_episode_ids=data.train.episode_ids[check],
            discovery_episode_ids=data.train.episode_ids[source],
            risk_screen_episode_ids=data.train.episode_ids[risk_part],
            model_channel_devices=cohort.model_sensor_devices,
            reference_device_id=cohort.reference_device_id,
            seed=211133,permutations=199,bootstraps=199,
        )
        accepted=risk.status=="ACCEPT_PREDICTIVE_ASSOCIATION"
        screens[family]={
            "risk":risk_evidence_passport(risk),
            "measurement":provenance_passport(corroboration),
        }
        basis=np.ascontiguousarray(np.column_stack([prior,candidate.direction]))
        config,dual,v10=_config()
        config["training"]["epochs"]=EPOCHS
        config["training"]["batch_size"]=BATCH
        source_results=[]
        for seed in MODEL_SEEDS:
            for arm in ARMS:
                if arm=="train_screened_signed_start" and not accepted:
                    records.append({
                        "family":family,"seed":seed,"arm":arm,
                        "status":"ABSTAIN_NO_TRAIN_SCREENED_PREDICTIVE_ORIENTATION",
                        "risk_screen_status":risk.status,
                        "measurement_corroboration_status":corroboration.status,
                        "initial_native_raw_weight_slot3":None,
                        "trained_native_decision_weight_slot3":None,
                        "trained_decision_weight_sign_matches_train_screen":None,
                        "validation_auprc":None,
                        "validation_auroc":None,
                        "validation_brier":None,
                        "max_native_logit_additive_error":None,
                        "mean_abs_native_new_slot_contribution":None,
                        "oracle_physical_alignment_abs_spearman_eval_only":None,
                        "oracle_instrument_alignment_abs_spearman_eval_only":None,
                        "physiological_concept_certified":False,
                    })
                    continue
                override=({2:float(risk.signed_orientation)*INITIAL_RAW_EFFECT}
                          if arm=="train_screened_signed_start" else None)
                model,receipt=fit_multiscale(
                    data,basis,seed,config,dual,v10,
                    consistency=False,expert_feedback=None,
                    feature_exploration=True,
                    temporal_readout_mode="legacy",native_head_refit=False,
                    initial_native_raw_decision_weights=override,
                )
                if arm=="train_screened_signed_start" and not (
                    receipt["train_initial_native_raw_decision_weight_overrides"] ==
                    {"2":float(risk.signed_orientation)*INITIAL_RAW_EFFECT}
                ):
                    raise AssertionError("Initial signed native effect not established before optimizer")
                evals,activation,signed=_evaluate_without_oracle(
                    model,data.validation,cohort.independent_validation_reference,
                    orientation=(
                        corroboration.reference_orientation_from_train_only
                        if corroboration.status=="ACCEPT_MEASUREMENT_CORROBORATION" else 1
                    ),
                )
                vectors[family][arm].append(signed)
                beta=float(model.physical.decision_weights()[2].detach())
                row={
                    "family":family,"seed":seed,"arm":arm,
                    "status":"TRAINED",
                    "risk_screen_status":risk.status,
                    "measurement_corroboration_status":corroboration.status,
                    "initial_native_raw_weight_slot3":(
                        override[2] if override is not None else None
                    ),
                    "trained_native_decision_weight_slot3":beta,
                    "trained_decision_weight_sign_matches_train_screen":(
                        bool(np.sign(beta)==risk.signed_orientation)
                        if accepted else None
                    ),
                    **evals,
                    "oracle_physical_alignment_abs_spearman_eval_only":None,
                    "oracle_instrument_alignment_abs_spearman_eval_only":None,
                    "physiological_concept_certified":False,
                }
                source_results.append((row,activation))
                records.append(row)
                print(family,seed,arm,
                      "native_beta",round(beta,5),
                      "AUPRC",round(evals["validation_auprc"],5),flush=True)
        # Only after *all* fitted models for this family, evaluate true hidden
        # physical and instrument source correlation, never used to train.
        truth=load_oracle_source_truth_for_postfit_audit(family)
        physical=truth["synthetic_physical_new"][:,-6:].mean(1)
        artifact=truth["synthetic_instrument_artifact"][:,-6:].mean(1)
        for row,activation in source_results:
            phys=_safe_spearman(activation,physical)
            art=_safe_spearman(activation,artifact)
            row["oracle_physical_alignment_abs_spearman_eval_only"]=(abs(phys) if phys is not None else None)
            row["oracle_instrument_alignment_abs_spearman_eval_only"]=(abs(art) if art is not None else None)
        del truth,physical,artifact,source_results
    summary=[]
    for family in FAMILIES:
        for arm in ARMS:
            opportunities=[r for r in records if r["family"]==family and r["arm"]==arm]
            completed=[r for r in opportunities if r["status"]=="TRAINED"]
            effect=vectors[family][arm]
            pair_rho=[]
            for i in range(len(effect)):
                for j in range(i+1,len(effect)):
                    rho=_safe_spearman(effect[i],effect[j])
                    if rho is not None:pair_rho.append(rho)
            summary.append({
                "family":family,"arm":arm,
                "opportunities":len(opportunities),
                "trained":len(completed),
                "abstentions":len(opportunities)-len(completed),
                "mean_validation_auprc":float(np.mean([r["validation_auprc"] for r in completed])) if completed else None,
                "mean_patient_signed_effect_crossseed_spearman":float(np.mean(pair_rho)) if pair_rho else None,
                "mean_abs_physical_source_alignment_eval_only":float(np.mean([r["oracle_physical_alignment_abs_spearman_eval_only"] for r in completed])) if completed else None,
                "mean_abs_artifact_alignment_eval_only":float(np.mean([r["oracle_instrument_alignment_abs_spearman_eval_only"] for r in completed])) if completed else None,
                "signed_stability_is_not_a_semantic_or_causal_proof":True,
            })
    if len(records)!=OPPORTUNITIES or len(summary)!=len(FAMILIES)*len(ARMS):
        raise AssertionError("V21b must retain full signed-stability opportunity denominator")
    outdir.mkdir(parents=True,exist_ok=False)
    for name,rows in (("ALL_30_SIGNED_START_OPPORTUNITIES.csv",records),
                      ("SUMMARY.csv",summary)):
        with (outdir/name).open("w",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    receipt={
        "status":"V21B_TRAIN_SCREENED_NATIVE_SIGN_INITIALIZATION_STABILITY_CONTROL",
        "registered_opportunities":len(records),
        "n_trained":sum(r["status"]=="TRAINED" for r in records),
        "n_train_screen_abstentions":sum(r["status"]!="TRAINED" for r in records),
        "native_risk_sign_initial_condition_only_not_hard_constraint":True,
        "identical_model_and_training_config_per_seed":True,
        "no_raw_input_to_risk_bypass":True,
        "risk_initialization_derived_from_disjoint_train_only_labels":True,
        "latent_oracle_eval_only_after_training":True,
        "test_stays_sealed":True,
        "common_mode_source_still_instrument_artifact":True,
        "no_real_human_review_or_clinical_semantics":True,
        "all_opportunities_recorded":True,
        "screens":screens,
        "all_cases":records,"aggregate":summary,
    }
    (outdir/"STATUS.json").write_text(json.dumps(receipt,indent=2,allow_nan=False))
    return receipt


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    run(args.out_dir)
