#!/usr/bin/env python3
"""V65 real data architecture repair: absolute+dynamic multiquantile fuzzy sources.

TEST has already been viewed in V63/V64, so even successful results here are
POSTHOC EXPLORATORY and do not independently establish physical source identity.
"""
from __future__ import annotations

import argparse,csv,json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

from conceptfan_discovery.train_innovation_ica_bank_v63 import TrainOnlyInnovationBank
from conceptfan_discovery.dualstream_multifuzzy_bank_v65 import TrainOnlyDualStreamMultiFuzzyBank
from conceptfan_discovery.native_temporal_fuzzy_v56 import NativeSelfSupervisedConceptFAN
from scripts.experiment_real_dynamic_gas_ica_v63 import (
    build_real_windows,train_only_sensor_scaling,reg_metrics,
    DATASET_FILES,SECOND_COLS,source_sha256
)

METHODS=(
    "v65_static_dualstream_36",
    "v65_native_dualstream_36",
    "v64_original_ica_absolute_linear_6",
    "v64_original_ica_delta_linear_6",
    "v63_original_ica_fuzzy_all_3",
    "raw_16_sensor_ridge_32",
    "raw_16_sensor_hgb_32",
    "constant_train_mean",
)


def extract_all_concept_features(x,m,ica,dual,native):
    """All risk heads see only TRAIN-derived concept summaries or raw comparators."""
    stat=dual.static_concepts(x,m)
    trained=native._transform(x,m)["trajectories"]
    if stat.shape!=trained.shape or stat.shape[-1]!=dual.k:
        raise AssertionError("Static and learned fuzzy concept dimensions differ")
    def both_time_scales(c):
        return np.concatenate([c.mean(axis=1),c[:,90:120].mean(axis=1)],axis=-1)
    abs_level=np.einsum("ntd,dk->ntk",x,ica.loading,optimize=True)
    delta,_=ica.transform(x,m)
    fuzzy=ica.static_fuzzy(x,m)
    raw=np.concatenate([x.mean(axis=1),x[:,90:120].mean(axis=1)],axis=-1)
    all_features={
        "v65_static_dualstream_36":both_time_scales(stat),
        "v65_native_dualstream_36":both_time_scales(trained),
        "v64_original_ica_absolute_linear_6":both_time_scales(abs_level),
        "v64_original_ica_delta_linear_6":both_time_scales(delta),
        "v63_original_ica_fuzzy_all_3":fuzzy.mean(axis=1),
        "raw_16_sensor_ridge_32":raw,
        "raw_16_sensor_hgb_32":raw,
    }
    if len(all_features)!=7 or any(not np.isfinite(v).all()
                                  for v in all_features.values()):
        raise AssertionError("Invalid set of registered source feature stages")
    return all_features


def _fit_predict_with_receipt(method,xtrain,ytrain,xtest):
    if method=="raw_16_sensor_hgb_32":
        model=HistGradientBoostingRegressor(
            max_iter=100,max_leaf_nodes=15,
            l2_regularization=1.,random_state=0)
        model.fit(xtrain,ytrain)
        return model.predict(xtest),{"model":"HGB","source_head_additive":False}
    if method.startswith("v65_"):
        # All coordinates are inspectable fuzzy states and reference only their
        # final two temporal summaries, no raw source input bypass.
        # The raw Ridge comparator uses StandardScaler. Give bounded
        # sigmoid memberships the SAME TRAIN-only column standardization
        # before Ridge(alpha=1), then algebraically express the head in the
        # original inspectable fuzzy coordinates for exact signed attribution.
        train0=xtrain-.5
        mu=train0.mean(axis=0)
        scale=np.std(train0,axis=0)
        scale=np.where(scale<1e-8,1.,scale)
        model=Ridge(alpha=1.)
        model.fit((train0-mu)/scale,ytrain)
        beta=model.coef_/scale
        bias=float(model.intercept_-np.dot(beta,mu))
        parts=(xtest-.5)*beta[None,:]
        pred=bias+parts.sum(axis=1)
        normalized_prediction=model.predict(((xtest-.5)-mu)/scale)
        error=float(np.max(np.abs(pred-normalized_prediction)))
        if error>1e-9:
            raise AssertionError("Scaled native concept head lost exact additive fidelity")
        return pred,{
            "model":"TRAIN_STANDARDIZED_CONCEPT_ONLY_SIGNED_ADDITIVE_RIDGE",
            "n_fuzzy_membership_state_time_summaries":xtest.shape[1],
            "max_additive_error":error,
            "original_fuzzy_coordinate_intercept":bias,
            "signed_coefficients":beta.tolist(),
            "train_only_standardization":True,
            "risk_raw_sensor_bypass":False
        }
    model=make_pipeline(StandardScaler(),Ridge(alpha=1.))
    model.fit(xtrain,ytrain)
    return model.predict(xtest),{"model":"Ridge","source_head_additive":False}


def run(path:Path,out:Path):
    if out.exists():
        raise FileExistsError("Refuse to overwrite registered V65 evidence")
    runs,original=build_real_windows(path)
    x,m,ref,session,groups,normalizer=train_only_sensor_scaling(runs)
    ica=TrainOnlyInnovationBank(
        basis="ica",seed=20261913,explained_variance_target=.95,
        max_k=8,block_bootstraps=20
    ).fit_unlabeled(x["TRAIN"],m["TRAIN"],train_halfhour_blocks=groups)
    dual=TrainOnlyDualStreamMultiFuzzyBank(ica).fit_unlabeled(
        x["TRAIN"],m["TRAIN"])
    native=NativeSelfSupervisedConceptFAN(
        dual,neural_seed=11,epochs=8).fit_source_unlabeled(
        x["TRAIN"],m["TRAIN"])
    source_sha=native.frozen_network_sha
    if dual.k!=ica.k*6 or dual.k*2!=2*3*2*ica.k:
        raise AssertionError("Fuzzy states are not six per observed ICA axis")
    features={
        cohort:extract_all_concept_features(
            x[cohort],m[cohort],ica,dual,native)
        for cohort in ("TRAIN","VALID","TEST")}
    rows=[];head_receipts=[];flags={}
    for rec_id,record_name in enumerate(DATASET_FILES):
        for idx,gas in ((0,SECOND_COLS[rec_id]),(1,"ethylene")):
            target=record_name+":"+gas
            train_idx=np.flatnonzero(session["TRAIN"]==rec_id)
            y_train=ref["TRAIN"][train_idx,idx]
            if np.std(y_train)<1e-6:
                raise ValueError("Unvarying TRAIN reference target")
            for cohort in ("VALID","TEST"):
                ev=np.flatnonzero(session[cohort]==rec_id)
                y=ref[cohort][ev,idx]
                for method in METHODS:
                    if method=="constant_train_mean":
                        pred=np.full(len(ev),float(y_train.mean()))
                        receipt={"model":"constant TRAIN mean","source_head_additive":True}
                    else:
                        pred,receipt=_fit_predict_with_receipt(
                            method,
                            features["TRAIN"][method][train_idx],
                            y_train,
                            features[cohort][method][ev])
                    rows.append({
                        "gas_target":target,"cohort":cohort,
                        "method":method,
                        "n_temporal_correlated_windows":len(ev),
                        "n_independent_physical_sessions_per_gas":1,
                        "n_original_statistical_ICA_axes":ica.k,
                        "n_observed_fuzzy_memberships":dual.k,
                        "same_TEST_previously_inspected_V63_V64":True,
                        **reg_metrics(y,pred,float(np.std(y_train)))})
                    if cohort=="VALID":
                        head_receipts.append({
                            "target":target,"method":method,
                            "TRAIN_label_count":len(train_idx),
                            "head_receipt":receipt})
            val_native=next(z["R2"] for z in rows if
                z["gas_target"]==target and z["cohort"]=="VALID"
                and z["method"]=="v65_native_dualstream_36")
            val_raw=next(z["R2"] for z in rows if
                z["gas_target"]==target and z["cohort"]=="VALID"
                and z["method"]=="raw_16_sensor_ridge_32")
            flags[target]={
                "incomplete_concept_bottleneck_by_frozen_VALID_rule":
                    bool(val_raw>val_native+.05),
                "raw_ridge_VALID_R2":val_raw,
                "native_dualstream_VALID_R2":val_native,
                "VALID_rule_margin":.05,
                "rule_does_not_retrain_source_or_head":True,
            }
    if native.frozen_network_sha!=source_sha:
        raise AssertionError("Supervised gas labels modified frozen native source")
    receipt={
        "status":"V65_EXPLORATORY_DUALSTREAM_MULTIFUZZY_EXECUTED",
        "preregistered_after_V63_V64_test_inspection":True,
        "new_independent_confirmatory_test":False,
        "source_dataset":"UCI322 real dynamic gas mixtures, TWO sessions",
        "original_zip_sha256":source_sha256(path),
        "real_independent_physical_recordings":2,
        "locked_chronological_split_and_embargo":original,
        "n_temporally_correlated_windows":{
            k:len(v) for k,v in runs.items()},
        "train_only_sensor_normalization":normalizer,
        "original_autonomously_detected_statistical_K":ica.k,
        "n_dual_absolute_and_innovation_streams":2*ica.k,
        "n_fuzzy_membership_states":dual.k,
        "n_visible_concept_only_signed_head_contributions":2*dual.k,
        "source_bank":dual.public_passport(),
        "frozen_native_source_sha":source_sha,
        "self_supervised_unlabeled_training_receipts":native.loss_receipt,
        "source_and_encoder_never_received_gas_setpoints":True,
        "gas_target_only_train_risk_heads_and_external_audit":True,
        "frozen_VALID_incompleteness_flags":flags,
        "all_original_raw_and_concept_baseline_scores":rows,
        "train_only_task_head_receipts":head_receipts,
        "physical_concept_semantics_proven":False,
        "controller_setpoints_are_not_independent_analyzer_traces":True,
        "protocol":"docs/medical/CONCEPTFAN_V65_FROZEN_DUAL_STREAM_MULTIFUZZY_20261113.md",
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"STATUS.json").write_text(json.dumps(
        receipt,ensure_ascii=False,indent=2,allow_nan=False)+"\n",
        encoding="utf8")
    with (out/"METRICS.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({
        "event":"V65_TWO_STREAM_EXECUTED",
        "k_statistical_axes":ica.k,
        "n_fuzzy_states":dual.k,
        "valid_incompleteness":flags,
        "test_rows":[{k:v for k,v in z.items()
                      if k in ("gas_target","method","R2","MAE_ppm")}
                     for z in rows if z["cohort"]=="TEST"]
    },ensure_ascii=False,allow_nan=False),flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--uci-zip",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args()
    run(args.uci_zip,args.out_dir)


if __name__=="__main__":
    main()
