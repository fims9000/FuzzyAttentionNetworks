"""V21b: test TRAIN-screened native risk sign initialization against arbitrary starts.

A third directly measured native concept can be highly aligned with a stable
data-generating direction yet its *signed patient-level risk contribution*
can reverse between independently seeded neural training runs. V21b
compares identical K3 native initial networks with only one native raw risk
parameter changed BEFORE the original end-to-end fit.

All source discovery, risk screens and measured-reference tests use disjoint
TRAIN cohorts. The oracle true source is never available to training. The
TEST split is sealed. This is a negative-control diagnosis, NOT a physical
source certificate or a post-hoc explanation aligner.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss

from .provenance_data_v21 import (
    FAMILIES,CHANNELS,DISCOVERY_N,SCREEN_N,CORROBORATION_N,
    make_v21_provenance_cohort,load_oracle_source_truth_for_postfit_audit,
)
from .candidate_admission_v17 import (
    propose_observable_residual_concept,candidate_passport,
)
from .risk_screen_v18 import screen_risk_association,risk_evidence_passport
from .measurement_corroboration_v21 import (
    corroborate_candidate_on_disjoint_measured_reference,provenance_passport,
)
from .clinical_multiscale_v10 import fit_multiscale
from .experiment_native_risk_v18 import _config

MODEL_SEEDS=(11,23,37)
ARMS=("arbitrary_native_start","train_screened_signed_start")
EPOCHS=10
BATCH_SIZE=96
SIGNED_INITIAL_RAW_MAGNITUDE=.05


def _rho(a,b):
    r=spearmanr(np.asarray(a),np.asarray(b)).statistic
    return float(r) if np.isfinite(r) else None


def _eval(model,split):
    model.eval()
    pred=[]
    raw=[]
    signed=[]
    max_error=0.
    with torch.no_grad():
        for i in range(0,len(split.y),128):
            stop=min(i+128,len(split.y))
            x=torch.from_numpy(split.x[i:stop])
            mask=torch.from_numpy(split.mask[i:stop])
            out=model(x,mask,process_mask=mask)
            activation=out.physical.raw_activation[:,-6:,2]
            availability=out.physical.availability[:,-6:,2]
            estimate=(activation*availability).sum(1)/availability.sum(1).clamp_min(1e-8)
            raw.extend(estimate.detach().cpu().numpy().tolist())
            signed.extend(out.physical.contributions[:,2].detach().cpu().numpy().tolist())
            pred.extend(out.probabilities.detach().cpu().numpy().tolist())
            max_error=max(max_error,float(out.decomposition_error()))
    if max_error>2e-5:
        raise AssertionError("Native additive decision contract violated")
    y=np.asarray(split.y)
    return {
        "validation_auprc":float(average_precision_score(y,pred)),
        "validation_auroc":float(roc_auc_score(y,pred)),
        "validation_brier":float(brier_score_loss(y,pred)),
        "max_additive_logit_decomposition_error":max_error,
        "mean_absolute_direct_signed_contribution":float(np.abs(signed).mean()),
        "native_third_effective_risk_coefficient":float(model.physical.decision_weights()[2].detach()),
        "physical_slot_risk_sign_policy_fixed":int(model.physical.signed_risk_policy[2].detach())!=0,
    },np.asarray(raw),np.asarray(signed)


def run(out_dir:Path):
    if out_dir.exists():
        raise FileExistsError("Refuse to overwrite predeclared V21b evidence")
    torch.set_num_threads(1)
    results=[]
    screenings=[]
    statuses=[]
    pair_data={}
    per_mode_truth={}
    for family in FAMILIES:
        cohort=make_v21_provenance_cohort(family)
        data=cohort.discovery
        if data.train.oracle_states is not None or data.validation.oracle_states is not None:
            raise AssertionError("Ground truth cannot enter native fitting")
        old_basis=np.eye(len(CHANNELS),2,dtype=np.float64)
        propose=slice(0,DISCOVERY_N)
        risk_part=slice(DISCOVERY_N,DISCOVERY_N+SCREEN_N)
        reference_part=slice(DISCOVERY_N+SCREEN_N,
                             DISCOVERY_N+SCREEN_N+CORROBORATION_N)
        candidate=propose_observable_residual_concept(
            data.train.x[propose],data.train.mask[propose],
            old_basis,CHANNELS,seed=211131,permutations=199,bootstraps=32,
        )
        evidence=screen_risk_association(
            candidate,data.train.x[risk_part],data.train.mask[risk_part],
            data.train.y[risk_part],
            screening_episode_ids=data.train.episode_ids[risk_part],
            discovery_episode_ids=data.train.episode_ids[propose],
            seed=211132,permutations=199,bootstraps=199,
        )
        measurement=corroborate_candidate_on_disjoint_measured_reference(
            candidate,
            data.train.x[reference_part],
            cohort.independent_train_reference[reference_part],
            provenance_episode_ids=data.train.episode_ids[reference_part],
            discovery_episode_ids=data.train.episode_ids[propose],
            risk_screen_episode_ids=data.train.episode_ids[risk_part],
            model_channel_devices=cohort.model_sensor_devices,
            reference_device_id=cohort.reference_device_id,
            seed=211133,permutations=199,bootstraps=199,
        )
        screenings.append({
            "family":family,
            "candidate":candidate_passport(candidate),
            "risk":risk_evidence_passport(evidence),
            "reference":provenance_passport(measurement),
        })
        basis=np.ascontiguousarray(np.column_stack([old_basis,candidate.direction]))
        config,dual,v10=_config()
        config["training"]["epochs"]=EPOCHS
        config["training"]["batch_size"]=BATCH_SIZE
        pair_data[family]={a:[] for a in ARMS}
        model_audits=[]
        for seed in MODEL_SEEDS:
            for arm in ARMS:
                if arm=="train_screened_signed_start" and evidence.status!="ACCEPT_PREDICTIVE_ASSOCIATION":
                    statuses.append({
                        "family":family,"seed":seed,"arm":arm,
                        "status":"ABSTAIN_NO_INDEPENDENT_TRAIN_RISK_SIGN",
                        "risk_screen_status":evidence.status,
                        "model_was_trained":False,
                    })
                    continue
                initial=(
                    None if arm=="arbitrary_native_start"
                    else {2:SIGNED_INITIAL_RAW_MAGNITUDE*int(evidence.signed_orientation)}
                )
                model,receipt=fit_multiscale(
                    data,basis,seed,config,dual,v10,
                    consistency=False,expert_feedback=None,
                    feature_exploration=True,
                    temporal_readout_mode="legacy",
                    native_head_refit=False,
                    initial_native_raw_decision_weights=initial,
                )
                metric,raw,signed=_eval(model,data.validation)
                if metric["physical_slot_risk_sign_policy_fixed"]:
                    raise AssertionError("V21b must not hard-constrain fitted native coefficient signs")
                if initial is None:
                    if receipt["train_initial_native_raw_decision_weight_overrides"] is not None:
                        raise AssertionError("Original V21 default risk initialization changed")
                elif receipt["train_initial_native_raw_decision_weight_overrides"]!={"2":initial[2]}:
                    raise AssertionError("TRAIN-screened initializer not included in training receipt")
                row={
                    "family":family,"seed":seed,"arm":arm,
                    "status":"TRAINED",
                    "risk_screen_status":evidence.status,
                    "reference_corroboration_status":measurement.status,
                    "native_initial_raw_risk_source":(
                        "unchanged_v10_initialization" if initial is None
                        else "independently_train_screened_risk_orientation"),
                    "initial_native_raw_risk_slot2_override":(
                        initial[2] if initial is not None else None),
                    "n_training_epochs":EPOCHS,
                    "n_training_episodes":len(data.train.y),
                    "noncausal_risk_sign_from_train_only":True,
                    "native_concept_truth_oracle_used_in_training":False,
                    "oracle_physical_alignment_abs_spearman_eval_only":None,
                    "oracle_instrument_alignment_abs_spearman_eval_only":None,
                    **metric,
                }
                results.append(row)
                statuses.append({
                    "family":family,"seed":seed,"arm":arm,
                    "status":"TRAINED",
                    "risk_screen_status":evidence.status,
                    "model_was_trained":True,
                })
                pair_data[family][arm].append((seed,raw,signed))
                model_audits.append((row,raw))
                print(family,seed,arm,"val AUPRC",round(metric["validation_auprc"],5),
                      "effective source beta",round(metric["native_third_effective_risk_coefficient"],5),
                      flush=True)
        # True source/audit happens only AFTER all native training for family.
        truth=load_oracle_source_truth_for_postfit_audit(family,split="validation")
        physical=truth["synthetic_physical_new"][:,-6:].mean(1)
        artifact=truth["synthetic_instrument_artifact"][:,-6:].mean(1)
        for row,raw in model_audits:
            row["oracle_physical_alignment_abs_spearman_eval_only"]=abs(_rho(raw,physical))
            row["oracle_instrument_alignment_abs_spearman_eval_only"]=abs(_rho(raw,artifact))
        del truth,physical,artifact,model_audits

    aggregate=[]
    for family in FAMILIES:
        for arm in ARMS:
            selected=[r for r in results if r["family"]==family and r["arm"]==arm]
            pair=pair_data[family][arm]
            rank_pairs=[]
            for i in range(len(pair)):
                for j in range(i+1,len(pair)):
                    rho=_rho(pair[i][2],pair[j][2])
                    if rho is not None:rank_pairs.append(rho)
            aggregate.append({
                "family":family,"arm":arm,
                "n_fitted_models":len(selected),
                "n_abstained_opportunities":sum(
                    s["status"].startswith("ABSTAIN")
                    for s in statuses if s["family"]==family and s["arm"]==arm
                ),
                "mean_validation_auprc":float(np.mean([r["validation_auprc"] for r in selected])) if selected else None,
                "mean_validation_brier":float(np.mean([r["validation_brier"] for r in selected])) if selected else None,
                "mean_abs_native_risk_effect":float(np.mean([r["mean_absolute_direct_signed_contribution"] for r in selected])) if selected else None,
                "mean_signed_native_risk_coeff":float(np.mean([r["native_third_effective_risk_coefficient"] for r in selected])) if selected else None,
                "all_native_risk_coeff_signs":(
                    [int(np.sign(r["native_third_effective_risk_coefficient"])) for r in selected] if selected else []
                ),
                "unadjusted_patient_specific_signed_effect_pairwise_spearman_mean":(
                    float(np.mean(rank_pairs)) if rank_pairs else None
                ),
                "unadjusted_patient_specific_signed_effect_pairwise_spearman_all_pairs":rank_pairs,
                "mean_oracle_physical_alignment_abs_spearman_eval_only":(
                    float(np.mean([r["oracle_physical_alignment_abs_spearman_eval_only"] for r in selected]))
                    if selected else None
                ),
                "mean_oracle_instrument_alignment_abs_spearman_eval_only":(
                    float(np.mean([r["oracle_instrument_alignment_abs_spearman_eval_only"] for r in selected]))
                    if selected else None
                ),
                "physical_causal_identity_proven":False,
            })
    if (len(statuses)!=30 or len(results)!=24
            or sum(s["status"].startswith("ABSTAIN") for s in statuses)!=6
            or len(aggregate)!=10 or len(screenings)!=5):
        raise AssertionError("V21b must retain all 30 opportunities, 24 fits and 6 refusals")
    out_dir.mkdir(parents=True,exist_ok=False)
    for filename,records in (
        ("ALL_30_REGISTERED_OPPORTUNITIES.csv",statuses),
        ("ALL_24_NATIVE_TRAINED_FITS.csv",results),
        ("AGGREGATE.csv",aggregate),
    ):
        with (out_dir/filename).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]))
            writer.writeheader();writer.writerows(records)
    (out_dir/"SOURCE_EVIDENCE.json").write_text(json.dumps(screenings,indent=2,allow_nan=False))
    receipt={
        "status":"V21B_TRAIN_ONLY_NATIVE_RISK_SIGN_INITIALIZATION_ABLATION",
        "n_registered_opportunities":len(statuses),
        "n_fitted_native_models":len(results),
        "n_screen_abstentions":sum(s["status"].startswith("ABSTAIN") for s in statuses),
        "risk_source_orientation_from_disjoint_train_only":True,
        "no_permanent_native_risk_coefficient_sign_constraint":True,
        "no_source_identity_oracle_in_any_training":True,
        "truth_loaded_after_family_training_only":True,
        "not_fresh_validation_cohort":True,
        "original_V21_nonreview_arm_unchanged_by_default":True,
        "no_real_clinician_feedback":True,
        "test_episodes_sealed":240,
        "unadjusted_signed_patient_effects_not_posthoc_oracle_aligned":True,
        "physical_or_causal_identity_claimed":False,
        "all_predeclared_negative_cases_reported":True,
        "registered_opportunities":statuses,
        "source_evidence":screenings,
        "per_fit":results,"aggregate":aggregate,
    }
    (out_dir/"STATUS.json").write_text(json.dumps(receipt,indent=2,allow_nan=False))
    return receipt


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",type=Path,required=True)
    options=p.parse_args()
    run(options.out_dir)
