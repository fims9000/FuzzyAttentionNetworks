"""V22e: actual external predictive controls versus native concept models.

Compare scikit-learn logistic and HGB predictors on the same synthetic
outcome against the V22d 600+200 independently TRAIN-screened native K2
ConceptFAN variants. Entirely separate observed history feature encoding,
no oracle physiological states or environment metadata passed to predictors.

All 18 concept models and eight external predictors are FIT before the
previously unused synthetic TEST seed 522323 is even generated. The
original 24 native model opportunities include six identifiability ABSTAINS.
A strong external predictor winning is a meaningful negative result.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import csv
import hashlib
import json
import pickle

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,roc_auc_score,brier_score_loss,log_loss,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import DiscoverySplit
from .mixed_source_data_v22 import (
    FAMILIES,HISTORY,HORIZON,MODEL_CHANNELS,
    _simulate,make_observable_v22,
    train_only_whitening,build_native_v22_data,
)
from .environmental_unmixing_v22 import (
    fit_train_only_environment_source_directions,
    native_basis_from_separation,separation_passport,
)
from .native_source_risk_screen_v22b import (
    screen_v22_native_sources_on_disjoint_train_labels,
    risk_plan_passport,
)
from .experiment_risk_screened_sources_v22b import (
    CORE_TRAIN_N,INDEPENDENT_RISK_SCREEN_N,
    TRAIN_EPOCHS,BATCH_SIZE,MODEL_SEEDS,
)
from .experiment_unseen_test_v22c import _fingerprint
from .experiment_mixed_source_v22 import (
    _config,_oracle_alignment_no_training_feedback,
)
from .clinical_multiscale_v10 import fit_multiscale


FRESH_PREDICTIVE_TEST_SEED=522323
FRESH_PREDICTIVE_TEST_N=300
METHODS=(
    "native_pca_600fit_200screen",
    "native_environment_600fit_200screen",
    "logistic_30_temporal_summaries",
    "hgb_30_temporal_summaries",
)


def observed_temporal_summaries(raw_measurements:np.ndarray)->np.ndarray:
    """Only observed 24-hour five-sensor histories, never outcomes or oracle."""
    raw=np.asarray(raw_measurements,dtype=np.float64)
    if (raw.ndim!=3 or raw.shape[1]<HISTORY or raw.shape[-1]!=5
            or not np.isfinite(raw).all()):
        raise ValueError("Need finite raw [patients,24+,5] observed sensor histories")
    history=raw[:,:HISTORY,:]
    last6=history[:,-6:].mean(axis=1)
    previous6=history[:,-12:-6].mean(axis=1)
    whole24=history.mean(axis=1)
    last=history[:,-1,:]
    diff=last6-previous6
    std12=history[:,-12:].std(axis=1)
    matrix=np.concatenate(
        [last6,previous6,whole24,last,diff,std12],axis=1,
    )
    if matrix.shape!=(len(raw),30):
        raise AssertionError("External predictor feature contract changed")
    return matrix.astype(np.float64)


def _external_estimators():
    return {
        "logistic_30_temporal_summaries":make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.,max_iter=2000,solver="lbfgs",class_weight=None,
            ),
        ),
        "hgb_30_temporal_summaries":HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=.05,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=.1,
            random_state=11,
        ),
    }


def _predictive_metrics(y,p):
    y=np.asarray(y,dtype=np.float64)
    p=np.asarray(p,dtype=np.float64)
    if (y.shape!=p.shape or len(y)<50 or not np.isfinite(p).all()
            or np.min(p)<0 or np.max(p)>1):
        raise AssertionError("Unusable TEST risk predictions")
    return {
        "test_auprc":float(average_precision_score(y,p)),
        "test_auroc":float(roc_auc_score(y,p)),
        "test_brier":float(brier_score_loss(y,p)),
        "test_logloss":float(log_loss(y,np.clip(p,1e-9,1-1e-9),labels=[0,1])),
    }


def _native_test_score(model,split,physical,instrument,env):
    model.eval()
    predictions=[]
    sources=[]
    max_error=0.
    with torch.no_grad():
        for start in range(0,len(split.y),128):
            stop=min(start+128,len(split.y))
            x=torch.from_numpy(split.x[start:stop])
            mask=torch.from_numpy(split.mask[start:stop])
            result=model(x,mask,process_mask=mask)
            raw=result.physical.raw_activation[:,-6:,:]
            reliability=result.physical.availability[:,-6:,:]
            concept=(raw*reliability).sum(1)/reliability.sum(1).clamp_min(1e-8)
            sources.extend(concept.cpu().numpy().tolist())
            predictions.extend(result.probabilities.cpu().numpy().tolist())
            max_error=max(max_error,float(result.decomposition_error()))
    if max_error>2e-5:
        raise AssertionError("External comparison must preserve exact native concept logit")
    source_scores=np.asarray(sources,dtype=np.float64)
    audit=_oracle_alignment_no_training_feedback(
        source_scores,physical,instrument,env,
    )
    return {
        **_predictive_metrics(split.y,np.asarray(predictions)),
        "max_native_logit_decomposition_error":max_error,
        "native_physical_vs_instrument_concept_alignment_postfit_only":audit,
    }


def _pickle_fingerprint(estimator)->str:
    return hashlib.sha256(pickle.dumps(estimator,protocol=4)).hexdigest()


def run(outdir:Path):
    if outdir.exists():
        raise FileExistsError("Never replace V22e registered external baseline evidence")
    torch.set_num_threads(1)
    registrations=[]
    source_status={}
    risk_status={}
    retained={}
    n_native=0
    n_abstained=0
    n_external=0
    for family in FAMILIES:
        full=make_observable_v22(family)
        core=replace(
            full,
            train_x_full=full.train_x_full[:CORE_TRAIN_N].copy(),
            train_environment=full.train_environment[:CORE_TRAIN_N].copy(),
            train_labels=full.train_labels[:CORE_TRAIN_N].copy(),
            train_episode_ids=full.train_episode_ids[:CORE_TRAIN_N].copy(),
        )
        w=train_only_whitening(core)
        sep=fit_train_only_environment_source_directions(
            core,whiten=w,seed=522221,permutations=199,bootstraps=48,
        )
        source_status[family]=separation_passport(sep)
        data=build_native_v22_data(core,w)
        history=w.transform(full.train_x_full[CORE_TRAIN_N:])[:,:HISTORY]
        screen_ids=full.train_episode_ids[CORE_TRAIN_N:]
        if (len(full.train_labels)!=800 or
                len(data.train.y)!=600 or len(screen_ids)!=200 or
                set(map(int,data.train.episode_ids))&set(map(int,screen_ids))):
            raise AssertionError("External comparison source/risk TRAIN populations overlapped")
        basis_by_method={
            "native_pca_600fit_200screen":
                np.eye(len(MODEL_CHANNELS),2,dtype=np.float64),
            "native_environment_600fit_200screen":
                (native_basis_from_separation(sep)
                 if sep.status=="ACCEPT_OBSERVATIONAL_ENVIRONMENT_UNMIXING"
                 else None),
        }
        config,dual,v10=_config()
        config["training"]["epochs"]=TRAIN_EPOCHS
        config["training"]["batch_size"]=BATCH_SIZE
        retained[family]={
            "whitener":w,
            "native":[],
            "external":[],
        }
        for method,basis in basis_by_method.items():
            risk=(
                screen_v22_native_sources_on_disjoint_train_labels(
                    history,np.ones_like(history),full.train_labels[CORE_TRAIN_N:],
                    basis,
                    original_source_train_episode_ids=data.train.episode_ids,
                    screen_episode_ids=screen_ids,
                    seed=522231,permutations=199,bootstraps=199,
                )
                if basis is not None else None
            )
            risk_status[f"{family}__{method}"]=(
                risk_plan_passport(risk) if risk is not None else None
            )
            for seed in MODEL_SEEDS:
                if basis is None:
                    registrations.append({
                        "family":family,"method":method,"seed":seed,
                        "status":sep.status,"trained":False,
                        "test_evaluated":False,
                        "total_train_label_budget":800,
                        "test_auprc":None,"test_auroc":None,
                        "test_brier":None,"test_logloss":None,
                        "source_alignment_only_for_native":None,
                        "real_clinical_semantics_certified":False,
                    })
                    n_abstained+=1
                    continue
                signed=risk.active_native_raw_decision_initialization
                blocked=risk.structurally_descriptive_only_native_slots
                model,receipt=fit_multiscale(
                    data,basis,seed,config,dual,v10,
                    consistency=False,feature_exploration=False,
                    expert_feedback=None,native_head_refit=False,
                    initial_native_raw_decision_weights=(
                        dict(signed) if signed else None
                    ),
                    initial_native_descriptive_only_slots=(
                        tuple(blocked) if blocked else None
                    ),
                )
                model.eval()
                for slot in blocked:
                    if (int(model.physical.signed_risk_policy[slot])!=2
                            or float(model.physical.decision_weights()[slot])!=0):
                        raise AssertionError("A descriptive source leaked into prediction")
                checksum=_fingerprint(model)
                retained[family]["native"].append(
                    (method,seed,model,checksum)
                )
                registrations.append({
                    "family":family,"method":method,"seed":seed,
                    "status":"TRAINED","trained":True,
                    "test_evaluated":False,
                    "total_train_label_budget":800,
                    "trained_native_model_fit_episodes":600,
                    "trained_task_risk_screen_episodes":200,
                    "source_basis_and_whitener_fit_episodes":600,
                    "descriptive_only_source_slots":list(blocked),
                    "source_risk_signed_initial_values":{
                        str(k):float(v) for k,v in signed.items()
                    },
                    "native_model_sha256_before_test":checksum,
                    "test_auprc":None,"test_auroc":None,
                    "test_brier":None,"test_logloss":None,
                    "source_alignment_only_for_native":None,
                    "real_clinical_semantics_certified":False,
                })
                n_native+=1
                print("NATIVE_FIT_BEFORE_TEST",family,method,seed,flush=True)

        # External predictive models share the FULL 800 original observed
        # five-sensor TRAIN histories and outcomes; no hidden factor oracle
        # or environment annotations are ever features.
        observed_features=observed_temporal_summaries(full.train_x_full)
        for method,baseline in _external_estimators().items():
            baseline.fit(observed_features,full.train_labels.astype(int))
            checksum=_pickle_fingerprint(baseline)
            retained[family]["external"].append(
                (method,baseline,checksum)
            )
            registrations.append({
                "family":family,"method":method,"seed":None,
                "status":"TRAINED","trained":True,
                "test_evaluated":False,
                "total_train_label_budget":800,
                "model_input":"30_predeclared_observed_temporal_summary_features",
                "source_alignment_only_for_native":None,
                "predictor_sha256_before_test":checksum,
                "test_auprc":None,"test_auroc":None,
                "test_brier":None,"test_logloss":None,
                "real_clinical_semantics_certified":False,
            })
            n_external+=1
            print("EXTERNAL_FIT_BEFORE_TEST",family,method,flush=True)

    # IMPORTANT barrier: no TEST generator seed 522323 has been instantiated
    # above this line. 18 native models and eight external predictors have
    # finished fitting; six BSS identifiability cases have abstained.
    if (n_native!=18 or n_external!=8 or n_abstained!=6
            or len(registrations)!=32):
        raise AssertionError(
            "Cannot unseal V22e TEST until all external and native fits finish"
        )
    test_generated_only_after_all_models_fitted=True

    for family in FAMILIES:
        raw,environment,y,latent=_simulate(
            family,FRESH_PREDICTIVE_TEST_SEED,FRESH_PREDICTIVE_TEST_N,
        )
        w=retained[family]["whitener"]
        x=w.transform(raw)
        mask=np.ones_like(x,dtype=np.float32)
        split=DiscoverySplit(
            x=x[:,:HISTORY],mask=mask[:,:HISTORY],
            future=x[:,HISTORY:HISTORY+HORIZON],
            future_mask=mask[:,HISTORY:HISTORY+HORIZON],
            y=y.astype(np.float32),
            oracle_states=None,
            episode_ids=np.arange(70000,70000+FRESH_PREDICTIVE_TEST_N,dtype=np.int64),
        )
        physical=latent[:,HISTORY-6:HISTORY,0].mean(1)
        instrument=latent[:,HISTORY-6:HISTORY,1].mean(1)
        def target_row(method,seed):
            matches=[r for r in registrations if
                     r["family"]==family and r["method"]==method and r["seed"]==seed]
            if len(matches)!=1 or matches[0]["test_evaluated"]:
                raise AssertionError("Every registered TEST method must score exactly once")
            return matches[0]
        for method,seed,model,checksum in retained[family]["native"]:
            if _fingerprint(model)!=checksum:
                raise AssertionError("Native model changed before test")
            score=_native_test_score(
                model,split,physical,instrument,environment,
            )
            row=target_row(method,seed)
            row.update({
                **{k:score[k] for k in (
                    "test_auprc","test_auroc","test_brier","test_logloss",
                )},
                "max_native_logit_decomposition_error":score[
                    "max_native_logit_decomposition_error"
                ],
                "source_alignment_only_for_native":score[
                    "native_physical_vs_instrument_concept_alignment_postfit_only"
                ],
                "native_model_sha256_after_test":_fingerprint(model),
                "test_evaluated":True,
            })
            if row["native_model_sha256_after_test"]!=checksum:
                raise AssertionError("Native test inference mutated model")
        external_features=observed_temporal_summaries(raw)
        for method,model,checksum in retained[family]["external"]:
            if _pickle_fingerprint(model)!=checksum:
                raise AssertionError("External predictive model mutated before TEST")
            score=_predictive_metrics(
                y,model.predict_proba(external_features)[:,1],
            )
            row=target_row(method,None)
            row.update({
                **score,
                "predictor_sha256_after_test":_pickle_fingerprint(model),
                "test_evaluated":True,
            })
            if row["predictor_sha256_after_test"]!=checksum:
                raise AssertionError("External test inference mutated model")
        del raw,environment,y,latent,split,physical,instrument,external_features

    aggregate=[]
    for family in FAMILIES:
        for method in METHODS:
            rows=[r for r in registrations
                  if r["family"]==family and r["method"]==method]
            scored=[r for r in rows if r["test_evaluated"]]
            aggregate.append({
                "family":family,"method":method,
                "n_registered":len(rows),
                "n_fitted_and_test_evaluated":len(scored),
                "n_source_identifiability_abstentions":len(rows)-len(scored),
                "mean_test_auprc":(
                    float(np.mean([r["test_auprc"] for r in scored]))
                    if scored else None
                ),
                "mean_test_auroc":(
                    float(np.mean([r["test_auroc"] for r in scored]))
                    if scored else None
                ),
                "mean_test_brier":(
                    float(np.mean([r["test_brier"] for r in scored]))
                    if scored else None
                ),
                "mean_test_logloss":(
                    float(np.mean([r["test_logloss"] for r in scored]))
                    if scored else None
                ),
                "mean_postfit_intrinsic_source_alignment":(
                    float(np.mean([
                        r["source_alignment_only_for_native"][
                            "mean_abs_oracle_source_spearman_eval_only"
                        ] for r in scored
                    ]))
                    if method.startswith("native") and scored else None
                ),
                "physiological_mechanism_certified":False,
            })
    if len(aggregate)!=16 or sum(r["test_evaluated"] for r in registrations)!=26:
        raise AssertionError("Missing external predictive TEST control or neural source")
    outdir.mkdir(parents=True,exist_ok=False)
    for name,rows in (
        ("ALL_32_NATIVE_AND_EXTERNAL_BENCHMARK_CASES.csv",registrations),
        ("ALL_16_PREDICTOR_FAMILY_SUMMARIES.csv",aggregate),
    ):
        keys=sorted({k for r in rows for k in r})
        with (outdir/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    k:json.dumps(row.get(k),sort_keys=True)
                    if isinstance(row.get(k),(dict,list)) else row.get(k)
                    for k in keys
                })
    receipt={
        "status":"V22E_FROZEN_EXTERNAL_PREDICTION_BASELINES_VS_NATIVE_CONCEPTFAN",
        "n_predeclared_native_opportunities":24,
        "n_native_fits":n_native,
        "n_native_identifiability_abstentions":n_abstained,
        "n_external_training_fits":n_external,
        "n_total_registered_opportunities":len(registrations),
        "n_total_test_model_evaluations":sum(r["test_evaluated"] for r in registrations),
        "train_generator_seed":522201,
        "train_labels_available_per_family":800,
        "fresh_test_seed":FRESH_PREDICTIVE_TEST_SEED,
        "fresh_test_patients_per_family":FRESH_PREDICTIVE_TEST_N,
        "all_native_and_external_fitted_before_any_fresh_test":test_generated_only_after_all_models_fitted,
        "no_test_hyperparameter_selection_or_optimizer_updates":True,
        "same_raw_input_history_and_test_outcomes_per_family":True,
        "no_latent_oracle_or_environment_in_external_features":True,
        "known_source_truth_loaded_only_during_postfit_evaluation":True,
        "native_bottleneck_is_train_native_not_posthoc_probe":True,
        "external_predictors_do_not_certify_or_satisfy_concept_fidelity":True,
        "not_an_official_concept_bottleneck_baseline":True,
        "all_negative_and_nonidentifiable_cases_retained":True,
        "no_physiological_causation_or_real_clinical_labels":True,
        "per_case":registrations,
        "summary":aggregate,
        "TRAIN_source_proposal":source_status,
        "TRAIN_source_risk_screens":risk_status,
    }
    (outdir/"SOURCE_PROPOSALS.json").write_text(
        json.dumps(source_status,indent=2,allow_nan=False)
    )
    (outdir/"SOURCE_RISK_SCREENS.json").write_text(
        json.dumps(risk_status,indent=2,allow_nan=False)
    )
    (outdir/"STATUS.json").write_text(
        json.dumps(receipt,indent=2,allow_nan=False)
    )
    return receipt


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",required=True,type=Path)
    args=p.parse_args()
    run(args.out_dir)
