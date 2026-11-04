"""V20b: evaluate V20 train-native refit on genuinely different synthetic dynamics.

This exercise tests only native risk optimization robustness. Known observed
sensor coordinate slots (K3) are NOT claimed to be automatically discovered
physiological factors. TEST labels are not used to choose any method.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,log_loss

from .distinct_synthetic_v20b import (
    make_distinct_generator_data,FAMILIES,CHANNELS,SIZES,SEEDS,
)
from .experiment_native_risk_v18 import _config
from .clinical_multiscale_v10 import fit_multiscale
from .native_head_factorial_v20 import fit_native_head_factorial_v20


SEEDS_MODEL=(11,23,37)
ARMS=("no_refit","lbfgs_unweighted","lbfgs_weighted")


def _predict(model,split):
    model.eval()
    probs=[]
    signed_effect=[]
    maximum=0.
    with torch.no_grad():
        for start in range(0,len(split.y),128):
            last=min(start+128,len(split.y))
            x=torch.from_numpy(split.x[start:last])
            mask=torch.from_numpy(split.mask[start:last])
            out=model(x,mask,process_mask=mask)
            probs.extend(out.probabilities.cpu().numpy().tolist())
            signed_effect.extend(out.physical.contributions[:,-1].cpu().numpy().tolist())
            maximum=max(maximum,float(out.decomposition_error()))
    p=np.asarray(probs,dtype=np.float64)
    if maximum>2e-5:
        raise AssertionError("Concept risk bypass broke exact additive evidence")
    return {
        "auprc":float(average_precision_score(split.y,p)),
        "auroc":float(roc_auc_score(split.y,p)),
        "brier":float(brier_score_loss(split.y,p)),
        "logloss":float(log_loss(split.y,p,labels=[0,1])),
        "mean_abs_slot3_logit_contribution":float(np.mean(np.abs(signed_effect))),
        "max_native_additive_error":maximum,
    }


def _measured_history_features(split):
    x=np.asarray(split.x,dtype=np.float64)
    m=np.asarray(split.mask,dtype=np.float64)
    def recent_mean(window):
        return x[:,window,:].sum(1)/np.maximum(m[:,window,:].sum(1),1.)
    recent=recent_mean(slice(-6,None))
    early=recent_mean(slice(0,6))
    last=x[:,-1,:]
    last_available=m[:,-1,:]
    block=x[:,-6:,:]
    observed=m[:,-6:,:]
    centered=(block-recent[:,None,:])*observed
    std=np.sqrt((centered**2).sum(1)/np.maximum(observed.sum(1),1.))
    coverage=m.mean(1)
    # Observable past only; missingness can be informative but never
    # impersonates a named physiological mechanism here.
    return np.column_stack([recent,recent-early,last,last_available,std,coverage])


def _external_fixed_logistic(data):
    model=Pipeline([
        ("scale",StandardScaler()),
        ("risk",LogisticRegression(C=1.,solver="lbfgs",max_iter=1000)),
    ])
    model.fit(_measured_history_features(data.discovery.train),data.discovery.train.y)
    output={}
    for label,split in (
        ("validation",data.discovery.validation),
        ("test",data.test),
    ):
        p=model.predict_proba(_measured_history_features(split))[:,1]
        output.update({
            label+"_auprc":float(average_precision_score(split.y,p)),
            label+"_auroc":float(roc_auc_score(split.y,p)),
            label+"_brier":float(brier_score_loss(split.y,p)),
            label+"_logloss":float(log_loss(split.y,p,labels=[0,1])),
        })
    return output


def _not_head(model):
    return {
        k:v.detach().clone()
        for k,v in model.state_dict().items()
        if k not in ("physical.decision_raw_weight","physical.decision_bias")
    }


def run(out_dir:Path):
    if out_dir.exists():
        raise FileExistsError("Never overwrite separate-family stress results")
    torch.set_num_threads(1)
    rows=[]
    external=[]
    family_receipts=[]
    for family in FAMILIES:
        family_data=make_distinct_generator_data(family)
        data=family_data.discovery
        assert data.train.oracle_states is None
        assert data.validation.oracle_states is None and family_data.test.oracle_states is None
        if len(data.train.y)!=1000 or len(data.validation.y)!=300 or len(family_data.test.y)!=300:
            raise AssertionError("V20b independent data split sizes changed")
        family_receipts.append({
            "family":family,
            "train_positive_rate":float(data.train.y.mean()),
            "validation_positive_rate":float(data.validation.y.mean()),
            "test_positive_rate":float(family_data.test.y.mean()),
            "train_observed_fraction":float(data.train.mask.mean()),
            "validation_observed_fraction":float(data.validation.mask.mean()),
            "test_observed_fraction":float(family_data.test.mask.mean()),
            "fixed_predeclared_measurement_directions_not_physical_labels":True,
        })
        logistic=_external_fixed_logistic(family_data)
        external.append({"family":family,"model":"raw_observable_recent_logistic",**logistic})
        print("family",family,"standard predictor test AUPRC",
              round(logistic["test_auprc"],5),flush=True)
        conf,dual,v10=_config()
        conf["training"]["epochs"]=12
        conf["training"]["batch_size"]=64
        basis=np.eye(len(CHANNELS),3)
        # First train ALL predeclared native models. Then score both new
        # validation and synthetic TEST with no intermediate model selection.
        trained=[]
        for seed in SEEDS_MODEL:
            original,training_receipt=fit_multiscale(
                data,basis,seed,conf,dual,v10,
                consistency=False,feature_exploration=True,
            )
            original.eval()
            snapshot=_not_head(original)
            for arm in ARMS:
                model=copy.deepcopy(original)
                fitter=None
                if arm!="no_refit":
                    fitter=fit_native_head_factorial_v20(
                        model,data.train,
                        declared_train_episode_ids=data.train.episode_ids,
                        method="lbfgs",
                        pos_weighted=(arm=="lbfgs_weighted"),
                        effective_beta_l2=.001,
                        lbfgs_max_iter=120,
                    )
                for key,value in snapshot.items():
                    if not torch.equal(model.state_dict()[key],value):
                        raise AssertionError("Separate-family refit changed a physical or care parameter")
                trained.append((seed,arm,model,fitter))

        for seed,arm,model,fitter in trained:
            val=_predict(model,data.validation)
            test=_predict(model,family_data.test)
            rows.append({
                "family":family,"seed":seed,"arm":arm,
                "train_positive_rate":float(data.train.y.mean()),
                "val_positive_rate":float(data.validation.y.mean()),
                "test_positive_rate":float(family_data.test.y.mean()),
                "train_observation_rate":float(data.train.mask.mean()),
                "val_observation_rate":float(data.validation.mask.mean()),
                "test_observation_rate":float(family_data.test.mask.mean()),
                "train_refit_objective_final":fitter.final_objective if fitter else None,
                "native_concept_count":model.physical.n_slots,
                "native_weights_and_bias_only_refitted":fitter is not None,
                "same_measurement_concept_generation_across_arms":True,
                "validation_auprc":val["auprc"],
                "validation_auroc":val["auroc"],
                "validation_brier":val["brier"],
                "validation_logloss":val["logloss"],
                "test_auprc":test["auprc"],
                "test_auroc":test["auroc"],
                "test_brier":test["brier"],
                "test_logloss":test["logloss"],
                "test_mean_abs_slot3_contribution":test["mean_abs_slot3_logit_contribution"],
                "max_exact_additive_error":max(val["max_native_additive_error"],test["max_native_additive_error"]),
            })
            print(family,seed,arm,"TEST AUPRC",round(test["auprc"],6),
                  "TEST Brier",round(test["brier"],6),flush=True)
    aggregates=[]
    for family in FAMILIES:
        for arm in ARMS:
            values=[r for r in rows if r["family"]==family and r["arm"]==arm]
            aggregates.append({
                "family":family,"arm":arm,"n_initialization_seeds":len(values),
                "mean_val_auprc":float(np.mean([r["validation_auprc"] for r in values])),
                "mean_test_auprc":float(np.mean([r["test_auprc"] for r in values])),
                "mean_test_auroc":float(np.mean([r["test_auroc"] for r in values])),
                "mean_test_brier":float(np.mean([r["test_brier"] for r in values])),
                "mean_test_nonzero_slot_effect":float(np.mean([r["test_mean_abs_slot3_contribution"] for r in values])),
                "test_scored_after_all_predeclared_models_trained":True,
            })
    if len(rows)!=36 or len(aggregates)!=12 or len(external)!=4:
        raise AssertionError("Incomplete full-denominator V20b independent-family benchmark")
    out_dir.mkdir(parents=True,exist_ok=False)
    for file_name,records in (
        ("ALL_36_NATIVE_TEST_RESULTS.csv",rows),
        ("SUMMARY.csv",aggregates),
        ("FOUR_RAW_FEATURE_PREDICTORS.csv",external),
        ("SPLIT_AND_OBSERVATION_RECEIPTS.csv",family_receipts),
    ):
        with (out_dir/file_name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]))
            writer.writeheader();writer.writerows(records)
    status={
        "status":"V20B_NEW_NONLINEAR_REGIME_MASK_AND_NULL_GENERATOR_FAMILIES",
        "n_native_test_evaluations":len(rows),
        "n_jointly_trained_native_models":12,
        "n_additional_native_head_refits":24,
        "n_raw_feature_external_logistic_controls":len(external),
        "n_generator_families":len(FAMILIES),
        "independent_seeds_per_train_validation_test":SEEDS,
        "fixed_train_val_test_sizes":SIZES,
        "only_train_normalization_and_refit":True,
        "all_test_arms_fixed_before_test_labels_read":True,
        "no_unobserved_latent_states_in_training_or_inference":True,
        "fixed_sensor_axes_do_not_prove_semantic_source_discovery":True,
        "missingness_process_care_branch_remains_separate":True,
        "no_clinical_physio_causation_identified":True,
        "all_36_outcomes_reported":True,
        "full_rows":rows,"aggregate":aggregates,
        "external_predictive_controls":external,
        "family_provenance":family_receipts,
    }
    (out_dir/"STATUS.json").write_text(json.dumps(status,indent=2,allow_nan=False))
    return status


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",required=True,type=Path)
    args=p.parse_args()
    run(args.out_dir)
