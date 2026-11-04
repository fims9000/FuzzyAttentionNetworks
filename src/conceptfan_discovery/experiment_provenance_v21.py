"""V21: executed ConceptFAN provenance and simulated-reviewer falsification.

TRAIN-only observable source, disjoint TRAIN-only risk and measured-device
corroboration. Native K3 concept models are trained with/without simulated
reference yes/no labels via the ORIGINAL V16 loss, not a post-hoc probe.

Synthetic oracle physical/instrument states are loaded only AFTER every model
fit for the corresponding family and are used solely for an explicit audit.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,roc_auc_score,brier_score_loss,
)

from .provenance_data_v21 import (
    FAMILIES,CHANNELS,DISCOVERY_N,SCREEN_N,CORROBORATION_N,
    make_v21_provenance_cohort,
    load_oracle_source_truth_for_postfit_audit,
)
from .measurement_corroboration_v21 import (
    corroborate_candidate_on_disjoint_measured_reference,
    provenance_passport,
)
from .candidate_admission_v17 import (
    propose_observable_residual_concept,candidate_passport,
)
from .risk_screen_v18 import screen_risk_association,risk_evidence_passport
from .expert_feedback_v16 import (
    ExpertReview,prepare_feedback,safe_feedback_manifest,
)
from .clinical_multiscale_v10 import fit_multiscale
from .experiment_native_risk_v18 import _config


ARMS=("no_feedback","measured_reference_feedback","intentionally_inverted_feedback")
MODEL_SEEDS=(11,23,37)
EPOCHS=10
BATCH=96
INTERVAL_START=18
THRESHOLD=.25
PER_CLASS=40
FEEDBACK_WEIGHT=1.0


def _select_reviews(cohort,corroboration,*,arm:str):
    if arm=="no_feedback":
        return None,{"review_label_origin":"none","train_review_count":0}
    if arm not in ARMS:
        raise ValueError("Unregistered native feedback experiment")
    train=cohort.discovery.train
    anchor=cohort.independent_train_reference
    aligned=(corroboration.reference_orientation_from_train_only
             if corroboration.status=="ACCEPT_MEASUREMENT_CORROBORATION" else 1)
    measured=anchor[:DISCOVERY_N,INTERVAL_START:].mean(axis=1)*aligned
    eligible=np.abs(measured)>=THRESHOLD
    classes=(measured>0).astype(np.int8)
    random=np.random.default_rng(211109)
    chosen=[]
    for target in (0,1):
        indices=np.flatnonzero(eligible & (classes==target))
        if len(indices)<PER_CLASS:
            raise RuntimeError("Synthetic reference lacks 40 blinded examples of each class")
        subs=random.choice(indices,PER_CLASS,replace=False)
        chosen.extend((int(index),int(target)) for index in subs)
    chosen.sort()
    # Simulated reviewer knows only the independently measured reference,
    # not synthetic generator latent truth or future risk labels.
    use_yes_no=(
        arm=="intentionally_inverted_feedback"
        or corroboration.status=="ACCEPT_MEASUREMENT_CORROBORATION"
    )
    reviews=[
        ExpertReview(
            episode_id=int(train.episode_ids[i]),
            slot=2,start=INTERVAL_START,end=train.x.shape[1],
            decision=(
                ("yes" if (target ^ (arm=="intentionally_inverted_feedback")) else "no")
                if use_yes_no else "unsure"
            ),
            confidence=1.,
            reviewer_id="synthetic_reference_rule_not_a_physician",
        )
        for i,target in chosen
    ]
    prepared=prepare_feedback(
        reviews,train.episode_ids,n_slots=3,n_steps=train.x.shape[1],
    )
    if use_yes_no and len(prepared.accepted)!=2*PER_CLASS:
        raise AssertionError("V21 active simulated reviewer supervision missing")
    if not use_yes_no and prepared.accepted:
        raise AssertionError("An uncorroborated reference must not supervise a native physiological concept")
    return prepared,{
        "review_label_origin":"measured_external_reference_not_oracle",
        "train_review_count":len(reviews),
        "review_effective_binary_supervision":len(prepared.accepted),
        "review_is_real_doctor":False,
        "unknown_is_negative":False,
        "inverted_review_is_deliberately_wrong":arm=="intentionally_inverted_feedback",
        "audit":safe_feedback_manifest(prepared),
    }


def _evaluate_without_oracle(model,split,measured_reference,orientation):
    model.eval()
    pred=[]
    raw=[]
    contribution=[]
    error=0.
    with torch.no_grad():
        for start in range(0,len(split.y),128):
            stop=min(start+128,len(split.y))
            x=torch.from_numpy(split.x[start:stop])
            m=torch.from_numpy(split.mask[start:stop])
            out=model(x,m,process_mask=m)
            act=out.physical.raw_activation[:,-6:,2]
            rel=out.physical.availability[:,-6:,2]
            estimate=(act*rel).sum(1)/rel.sum(1).clamp_min(1e-8)
            pred.extend(out.probabilities.cpu().numpy().tolist())
            raw.extend(estimate.cpu().numpy().tolist())
            contribution.extend(out.physical.contributions[:,2].cpu().numpy().tolist())
            error=max(error,float(out.decomposition_error()))
    if error>2e-5:
        raise AssertionError("ConceptFAN decision is not an exact native concept bottleneck")
    p=np.asarray(pred,dtype=np.float64)
    activation=np.asarray(raw,dtype=np.float64)
    signed=np.asarray(contribution,dtype=np.float64)
    reference=np.asarray(measured_reference[:,-6:].mean(1)*orientation,dtype=np.float64)
    valid=np.abs(reference)>=THRESHOLD
    binary=(reference>0).astype(int)
    if valid.sum()<60 or np.unique(binary[valid]).size!=2:
        raise ValueError("Synthetic reviewer reference validation has insufficient support")
    auc=float(roc_auc_score(binary[valid],activation[valid]))
    ap=float(average_precision_score(binary[valid],activation[valid]))
    return {
        "validation_auprc":float(average_precision_score(split.y,p)),
        "validation_auroc":float(roc_auc_score(split.y,p)),
        "validation_brier":float(brier_score_loss(split.y,p)),
        "reference_measured_proxy_auroc":auc,
        "reference_measured_proxy_auprc":ap,
        "validation_reference_reviewable_episodes":int(valid.sum()),
        "max_native_logit_additive_error":error,
        "mean_abs_native_new_slot_contribution":float(np.abs(signed).mean()),
    },activation,signed


def _rho(a,b):
    result=spearmanr(a,b).statistic
    return float(result) if np.isfinite(result) else None


def run(outdir:Path):
    if outdir.exists():
        raise FileExistsError("Never overwrite V21 executed evidence")
    torch.set_num_threads(1)
    all_fits=[]
    source_evidence=[]
    cross_seed_vectors={}
    for family in FAMILIES:
        cohort=make_v21_provenance_cohort(family)
        data=cohort.discovery
        if data.train.oracle_states is not None or data.validation.oracle_states is not None:
            raise AssertionError("Oracle latent states must be inaccessible in ConceptFAN fitting")
        prior=np.eye(len(CHANNELS),2,dtype=np.float64)
        discovery=slice(0,DISCOVERY_N)
        risk_section=slice(DISCOVERY_N,DISCOVERY_N+SCREEN_N)
        gate_section=slice(DISCOVERY_N+SCREEN_N,DISCOVERY_N+SCREEN_N+CORROBORATION_N)
        candidate=propose_observable_residual_concept(
            data.train.x[discovery],data.train.mask[discovery],
            prior,CHANNELS,seed=211131,
            permutations=199,bootstraps=32,
        )
        risk=screen_risk_association(
            candidate,
            data.train.x[risk_section],data.train.mask[risk_section],
            data.train.y[risk_section],
            screening_episode_ids=data.train.episode_ids[risk_section],
            discovery_episode_ids=data.train.episode_ids[discovery],
            seed=211132,permutations=199,bootstraps=199,
        )
        provenance=corroborate_candidate_on_disjoint_measured_reference(
            candidate,
            data.train.x[gate_section],
            cohort.independent_train_reference[gate_section],
            provenance_episode_ids=data.train.episode_ids[gate_section],
            discovery_episode_ids=data.train.episode_ids[discovery],
            risk_screen_episode_ids=data.train.episode_ids[risk_section],
            model_channel_devices=cohort.model_sensor_devices,
            reference_device_id=cohort.reference_device_id,
            seed=211133,permutations=199,bootstraps=199,
        )
        print(family,
              "risk",risk.status,
              "replication",provenance.status,
              "r",round(provenance.observed_signed_pearson_r,5),
              flush=True)
        source_evidence.append({
            "family":family,
            "candidate":candidate_passport(candidate),
            "risk":risk_evidence_passport(risk),
            "reference":provenance_passport(provenance),
            "source_recognition_is_not_physical_certification":True,
        })
        basis=np.ascontiguousarray(np.column_stack([prior,candidate.direction]))
        original,dual,v10=_config()
        original["training"]["epochs"]=EPOCHS
        original["training"]["batch_size"]=BATCH
        # Grounded positive/negative measurement feedback is enabled only
        # if the independent device reference passed a TRAIN-only gate.
        arm_outputs=[]
        cross_seed_vectors[family]={arm:[] for arm in ARMS}
        for arm in ARMS:
            reviews,review_meta=_select_reviews(cohort,provenance,arm=arm)
            for seed in MODEL_SEEDS:
                model,receipt=fit_multiscale(
                    data,basis,seed,original,dual,v10,
                    consistency=False,
                    expert_feedback=reviews,
                    expert_feedback_weight=(
                        FEEDBACK_WEIGHT if reviews is not None else 0.,
                    ),
                    feature_exploration=True,
                    temporal_readout_mode="legacy",
                    native_head_refit=False,
                )
                stats,act,signed=_evaluate_without_oracle(
                    model,data.validation,
                    cohort.independent_validation_reference,
                    orientation=(
                        provenance.reference_orientation_from_train_only
                        if provenance.status=="ACCEPT_MEASUREMENT_CORROBORATION" else 1
                    ),
                )
                cross_seed_vectors[family][arm].append(signed)
                arm_outputs.append((arm,seed,model,act))
                all_fits.append({
                    "family":family,"arm":arm,"seed":seed,
                    "risk_screen_status":risk.status,
                    "source_measurement_provenance_status":provenance.status,
                    "source_risk_signed_orientation":risk.signed_orientation,
                    "source_reference_signed_orientation":provenance.reference_orientation_from_train_only,
                    "native_slots":model.physical.n_slots,
                    "training_epochs":EPOCHS,
                    "n_simulated_train_reviews":review_meta["train_review_count"],
                    "n_active_native_binary_review_intervals":review_meta.get(
                        "review_effective_binary_supervision",0),
                    "is_real_physician_feedback":False,
                    "native_review_loss_batches":receipt["expert_feedback_supervised_batches"],
                    "mean_native_review_loss":receipt["mean_train_expert_feedback_loss"],
                    "oracle_physical_alignment_abs_spearman_eval_only":None,
                    "oracle_instrument_alignment_abs_spearman_eval_only":None,
                    "physical_source_certified":False,
                    **stats,
                })
                print("  ",arm,seed,
                      "AUPRC",round(stats["validation_auprc"],5),
                      "referenceAUROC",round(stats["reference_measured_proxy_auroc"],5),
                      flush=True)
        # Load sealed truth ONLY AFTER all 9 native training runs for family
        # have completed and no model fit or selection remains for this family.
        truth=load_oracle_source_truth_for_postfit_audit(family,split="validation")
        physical=truth["synthetic_physical_new"][:,-6:].mean(1)
        instrument=truth["synthetic_instrument_artifact"][:,-6:].mean(1)
        audited={(arm,seed):act for arm,seed,_,act in arm_outputs}
        for row in all_fits:
            if row["family"]==family:
                scores=audited[(row["arm"],row["seed"])]
                row["oracle_physical_alignment_abs_spearman_eval_only"]=abs(_rho(scores,physical))
                row["oracle_instrument_alignment_abs_spearman_eval_only"]=abs(_rho(scores,instrument))
        # Oracle never returns to model, reward, or candidate selection.
        del truth,physical,instrument,arm_outputs
    aggregate=[]
    for family in FAMILIES:
        for arm in ARMS:
            rows=[r for r in all_fits if r["family"]==family and r["arm"]==arm]
            vectors=cross_seed_vectors[family][arm]
            correlations=[]
            for i in range(len(vectors)):
                for j in range(i+1,len(vectors)):
                    r=_rho(vectors[i],vectors[j])
                    if r is not None:correlations.append(r)
            aggregate.append({
                "family":family,"arm":arm,"n_model_initialization_seeds":len(rows),
                "mean_validation_outcome_auprc":float(np.mean([r["validation_auprc"] for r in rows])),
                "mean_measured_reference_auroc":float(np.mean([r["reference_measured_proxy_auroc"] for r in rows])),
                "mean_oracle_physical_alignment_abs_spearman_eval_only":float(np.mean([r["oracle_physical_alignment_abs_spearman_eval_only"] for r in rows])),
                "mean_oracle_instrument_alignment_abs_spearman_eval_only":float(np.mean([r["oracle_instrument_alignment_abs_spearman_eval_only"] for r in rows])),
                "mean_signed_patient_effect_rank_crossseed_spearman":float(np.mean(correlations)) if correlations else None,
                "clinical_mechanism_or_causal_identity_established":False,
            })
    if len(all_fits)!=45 or len(aggregate)!=15 or len(source_evidence)!=5:
        raise AssertionError("Missing V21 candidate, reviewer, or negative scenario")
    outdir.mkdir(parents=True,exist_ok=False)
    for name,rows in (
        ("ALL_45_NATIVE_FEEDBACK_FITS.csv",all_fits),
        ("AGGREGATE.csv",aggregate),
    ):
        with (outdir/name).open("w",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    (outdir/"FIVE_SOURCE_PROVENANCE_PASSPORTS.json").write_text(
        json.dumps(source_evidence,indent=2,allow_nan=False)
    )
    manifest={
        "status":"V21_SOURCE_VERSUS_ARTIFACT_AND_NATIVE_FEEDBACK_FALSIFICATION",
        "five_independent_generator_conditions":len(FAMILIES),
        "n_native_full_model_fits":len(all_fits),
        "n_model_initialization_seeds":len(MODEL_SEEDS),
        "n_feedback_arms":len(ARMS),
        "source_proposal_uses_train_sensor_measurements_only":True,
        "risk_and_measurement_references_from_disjoint_train_subsets":True,
        "independent_device_reference_never_input_to_native_model":True,
        "native_train_oracle_states_excluded":True,
        "oracle_truth_loaded_after_all_fits_in_each_condition":True,
        "oracle_never_in_training_or_model_selection":True,
        "real_doctor_labels_provided":False,
        "common_mode_artifact_can_imitate_crossdevice_measurement":True,
        "no_physical_or_causal_concept_certification":True,
        "all_fits_and_failed_semantic_controls_reported":True,
        "synthetic_test_episodes_sealed":240,
        "per_fit":all_fits,
        "source_passports":source_evidence,
        "aggregate":aggregate,
    }
    (outdir/"STATUS.json").write_text(json.dumps(manifest,indent=2,allow_nan=False))
    return manifest


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",required=True,type=Path)
    args=p.parse_args()
    run(args.out_dir)
