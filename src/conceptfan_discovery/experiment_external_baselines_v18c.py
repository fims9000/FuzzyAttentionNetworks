"""V18c: deliberately simple *external predictive* baselines.

V18 evidence is a positive control for dynamic concept creation, not evidence
that ConceptFAN beats established classifiers. Freeze simple, reproducible
baselines on exactly the same TRAIN/heldout VALIDATION split and inputs.
No test outcomes, latent oracle or concept-factor identity enter training.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,roc_auc_score,brier_score_loss,log_loss,
)

from .experiment_native_risk_v18 import (
    make_data, MODES, DATA_GENERATOR_SEED,
    TRAIN_N, VAL_N, SEALED_TEST_N,
)

BASELINES=(
    "constant_prevalence",
    "logistic_recent_observable",
    "logistic_full_past",
    "hist_gradient_boosting_recent",
)
HISTORY_STEPS=24
N_SENSORS=5


def input_features(x):
    if x.ndim!=3 or x.shape[1:]!=(HISTORY_STEPS,N_SENSORS):
        raise ValueError("Expected historical TRAIN/VALIDATION records only")
    recent=x[:,-6:,:].mean(axis=1)
    old=x[:,:6,:].mean(axis=1)
    last=x[:,-1,:]
    recent_std=x[:,-6:,:].std(axis=1)
    lowdim=np.concatenate([recent,last,recent-old,recent_std],axis=1)
    return lowdim,x.reshape(len(x),-1)


def _estimator(name):
    if name=="constant_prevalence":
        return DummyClassifier(strategy="prior",random_state=3)
    if name=="logistic_recent_observable":
        return Pipeline([
            ("scale",StandardScaler()),
            ("risk",LogisticRegression(C=1.0,solver="lbfgs",max_iter=1000)),
        ])
    if name=="logistic_full_past":
        return Pipeline([
            ("scale",StandardScaler()),
            ("risk",LogisticRegression(C=1.0,solver="lbfgs",max_iter=1000)),
        ])
    if name=="hist_gradient_boosting_recent":
        return HistGradientBoostingClassifier(
            max_iter=150,max_leaf_nodes=15,min_samples_leaf=20,
            learning_rate=.05,random_state=3,
        )
    raise ValueError("Unknown frozen predictive baseline")


def run(outdir:Path):
    if outdir.exists():
        raise FileExistsError("Do not overwrite executed benchmark")
    rows=[]
    for mode in MODES:
        data=make_data(mode)
        if len(data.train.y)!=TRAIN_N or len(data.validation.y)!=VAL_N:
            raise AssertionError("Disjoint V18 synthetic split changed")
        if data.train.oracle_states is not None or data.validation.oracle_states is not None:
            raise AssertionError("Baseline must not see hidden factor oracle")
        train_low,train_full=input_features(data.train.x)
        valid_low,valid_full=input_features(data.validation.x)
        for name in BASELINES:
            model=_estimator(name)
            train=train_full if name=="logistic_full_past" else train_low
            valid=valid_full if name=="logistic_full_past" else valid_low
            model.fit(train,data.train.y)
            p=model.predict_proba(valid)[:,1]
            if len(p)!=len(data.validation.y) or not np.isfinite(p).all():
                raise AssertionError("Invalid baseline prediction")
            metric={
                "mode":mode,"model":name,
                "n_train":len(data.train.y),"n_holdout_validation":len(data.validation.y),
                "n_sealed_test_episodes":SEALED_TEST_N,
                "heldout_positive_rate":float(data.validation.y.mean()),
                "heldout_auprc":float(average_precision_score(data.validation.y,p)),
                "heldout_auroc":float(roc_auc_score(data.validation.y,p)),
                "heldout_brier":float(brier_score_loss(data.validation.y,p)),
                "heldout_logloss":float(log_loss(data.validation.y,p,labels=[0,1])),
                "validation_used_for_training_or_tuning":False,
                "hidden_factor_oracle_used":False,
            }
            rows.append(metric)
            print(mode,name,"AUPRC",round(metric["heldout_auprc"],6),flush=True)
    outdir.mkdir(parents=True,exist_ok=False)
    with (outdir/"BASELINES_ALL.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    status={
        "status":"V18C_SAME_SPLIT_EXTERNAL_PREDICTIVE_BASELINES",
        "n_fits":len(rows),"n_modes":len(MODES),
        "n_baselines":len(BASELINES),
        "generator_seed":DATA_GENERATOR_SEED,
        "no_hyperparameters_selected_on_validation":True,
        "baseline_is_not_a_concept_explanation_architecture":True,
        "hospital_generalization_not_established":True,
        "med_circuitbench_not_reused":True,
        "test_ids_sealed":True,
        "all_per_mode_results":rows,
    }
    (outdir/"STATUS.json").write_text(json.dumps(status,indent=2,allow_nan=False))
    return status


if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-dir",required=True,type=Path)
    args=ap.parse_args()
    run(args.out_dir)
