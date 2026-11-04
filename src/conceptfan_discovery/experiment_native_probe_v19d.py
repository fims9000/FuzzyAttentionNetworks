"""V19d: post-training *diagnostic* logistic probe on frozen native concepts.

A train-only linear readout refit on ConceptFAN's own native K3 summaries
isolates whether predictive information already survived concept formation.
This does NOT create an improved concept model or post-hoc explanation.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss

from .candidate_admission_v17 import propose_observable_residual_concept
from .risk_screen_v18 import screen_risk_association,risk_evidence_passport
from .clinical_multiscale_v10 import fit_multiscale
from .experiment_signmatched_temporal_v19b import _new_model
from .experiment_native_risk_v18 import (
    make_data,_config,_fit_continuation,
    CHANNELS,MODES,MODEL_SEEDS,DISCOVERY_N,SCREEN_N,
)

ARMS=("legacy_signinit","learned_shape_signinit")
TOTAL_FITS=len(MODES)*len(MODEL_SEEDS)*len(ARMS)


def _features_and_native_predictions(model,split):
    model.eval()
    summaries=[]
    probs=[]
    decomposition=0.
    with torch.no_grad():
        for start in range(0,len(split.y),128):
            end=min(start+128,len(split.y))
            x=torch.from_numpy(split.x[start:end])
            m=torch.from_numpy(split.mask[start:end])
            out=model(x,m,process_mask=m)
            summaries.append(out.physical.concept_summary.cpu().numpy())
            probs.extend(out.probabilities.cpu().numpy().tolist())
            decomposition=max(decomposition,float(out.decomposition_error()))
    return np.concatenate(summaries,axis=0),np.asarray(probs),decomposition


def _metrics(y,probabilities):
    return {
        "auprc":float(average_precision_score(y,probabilities)),
        "auroc":float(roc_auc_score(y,probabilities)),
        "brier":float(brier_score_loss(y,probabilities)),
    }


def run(outdir:Path):
    if outdir.exists():
        raise FileExistsError("V19d cannot overwrite previously executed results")
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
        cohort=slice(DISCOVERY_N,DISCOVERY_N+SCREEN_N)
        evidence=screen_risk_association(
            candidate,data.train.x[cohort],data.train.mask[cohort],
            data.train.y[cohort],
            screening_episode_ids=data.train.episode_ids[cohort],
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
                # A null risk screen normally requires ABSTAIN for clinical
                # interpretation; here zero-init native task fitting is
                # forced solely as the no-signal diagnostic.
                orientation=(
                    evidence.signed_orientation
                    if evidence.status=="ACCEPT_PREDICTIVE_ASSOCIATION" else 0
                )
                model=_new_model(original,candidate,arm,orientation)
                model=_fit_continuation(
                    model,data,seed,config,dual,
                    candidate=None,grounding_weight=0.,
                )
                train_concepts,train_probs,train_error=_features_and_native_predictions(
                    model,data.train,
                )
                validation_concepts,validation_probs,val_error=_features_and_native_predictions(
                    model,data.validation,
                )
                if train_concepts.shape!=(800,3) or validation_concepts.shape!=(200,3):
                    raise AssertionError("Probe must receive exactly K3 native scalar summaries")
                if max(train_error,val_error)>2e-5:
                    raise AssertionError("Native task decision lost exact concept decomposition")
                # Separate diagnostic refit: ONLY K3 ConceptFAN concepts enter
                # this probe; no raw values, missingness or future observations.
                probe=Pipeline([
                    ("scale",StandardScaler()),
                    ("head",LogisticRegression(
                        C=1.,max_iter=1000,solver="lbfgs",random_state=701,
                    )),
                ])
                probe.fit(train_concepts,data.train.y)
                prob_probe_train=probe.predict_proba(train_concepts)[:,1]
                prob_probe_val=probe.predict_proba(validation_concepts)[:,1]
                train_native=_metrics(data.train.y,train_probs)
                train_probe=_metrics(data.train.y,prob_probe_train)
                val_native=_metrics(data.validation.y,validation_probs)
                val_probe=_metrics(data.validation.y,prob_probe_val)
                recorded={
                    "mode":mode,"seed":seed,"arm":arm,
                    "status":"TRAINED",
                    "screen_status":evidence.status,
                    "train_only_sign":orientation,
                    "n_native_concept_features":train_concepts.shape[1],
                    "min_train_concept_std":float(np.std(train_concepts,axis=0).min()),
                    "max_train_concept_std":float(np.std(train_concepts,axis=0).max()),
                    "native_train_auprc":train_native["auprc"],
                    "probe_train_auprc":train_probe["auprc"],
                    "native_val_auprc":val_native["auprc"],
                    "probe_val_auprc":val_probe["auprc"],
                    "probe_minus_native_val_auprc":val_probe["auprc"]-val_native["auprc"],
                    "native_val_auroc":val_native["auroc"],
                    "probe_val_auroc":val_probe["auroc"],
                    "native_val_brier":val_native["brier"],
                    "probe_val_brier":val_probe["brier"],
                    "probe_weight_l2":float(np.linalg.norm(probe.named_steps["head"].coef_)),
                    "max_additive_native_error":max(train_error,val_error),
                    "test_ids_sealed":True,
                }
                rows.append(recorded)
                print(mode,seed,arm,"native",round(val_native["auprc"],5),
                      "probe",round(val_probe["auprc"],5),flush=True)
    summary=[]
    for mode in MODES:
        for arm in ARMS:
            chosen=[r for r in rows if r["mode"]==mode and r["arm"]==arm]
            summary.append({
                "mode":mode,"arm":arm,
                "n_seeds":len(chosen),
                "mean_native_val_auprc":float(np.mean([r["native_val_auprc"] for r in chosen])),
                "mean_probe_val_auprc":float(np.mean([r["probe_val_auprc"] for r in chosen])),
                "mean_probe_minus_native_auprc":float(np.mean([r["probe_minus_native_val_auprc"] for r in chosen])),
                "mean_native_val_brier":float(np.mean([r["native_val_brier"] for r in chosen])),
                "mean_probe_val_brier":float(np.mean([r["probe_val_brier"] for r in chosen])),
                "probe_never_sees_raw_input":True,
            })
    if len(rows)!=TOTAL_FITS:
        raise AssertionError("All neural fits and probes must be retained")
    outdir.mkdir(parents=True,exist_ok=False)
    for name,items in (("ALL_PROBES_AND_NATIVE_FITS.csv",rows),("SUMMARY.csv",summary)):
        with (outdir/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(items[0]))
            writer.writeheader();writer.writerows(items)
    receipt={
        "status":"V19D_K3_NATIVE_CONCEPT_SUFFICIENCY_LOGISTIC_PROBE",
        "n_native_fits":len(rows),"n_logistic_probes":len(rows),
        "same_frozen_concepts_for_native_and_probe":True,
        "probe_input_is_only_frozen_native_concept_summary":True,
        "probe_labels_are_train_only":True,
        "no_raw_sensor_features_in_probe":True,
        "validation_not_used_for_training_or_tuning":True,
        "previously_inspected_validation":True,
        "sealed_test_episodes":80,
        "forced_null_fit_is_diagnostic_not_approved_risk":True,
        "no_semantic_or_causal_clinical_claim":True,
        "complete_denominator_all_modes":True,
        "screens":passports,
        "results":rows,"summary":summary,
    }
    (outdir/"STATUS.json").write_text(json.dumps(receipt,indent=2,allow_nan=False))
    return receipt


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",required=True,type=Path)
    args=p.parse_args()
    run(args.out_dir)
