#!/usr/bin/env python3
"""V70 official unseen 58-TRIAL acetone/ethanol mixture source experiment.

Original UCI308 raw 928 sensor rows = 58 real independent trials x 16
physical sensors x 7500 REAL time samples, not 928 or millions of trials.
'ace_conc' and 'eth_conc' are nominal water-DILUTION dose vol.% references,
not gas-phase ppm or independent analyzer gas concentration trajectories.
"""
from __future__ import annotations

import argparse,csv,gzip,io,json
from collections import Counter
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile

import numpy as np
from scipy.stats import rankdata,spearmanr
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

from conceptfan_discovery.trialwise_dualstream_fuzzy_v70 import TrialOnlyDualStreamBank

BATCHES=("day-1-morning","day-2-morning","day-2-afternoon",
         "day-3-morning","day-4-afternoon")
EXPECTED_BATCH_COUNTS=(19,10,10,11,8)
SENSOR_COUNT=16
N_RAW_SAMPLES_PER_SENSOR=7500
DOWNSAMPLED_TIME_BINS=60
RAW_PER_BIN=125
EXPECTED_ORIGINAL_EXPERIMENTS=58
ANALYTES=("acetone","ethanol")
METHODS=(
    "V70_STATIC_DUALSTREAM_MULTIFUZZY",
    "V70_LINEAR_ABSOLUTE_PLUS_CAUSAL_DELTA",
    "ORIGINAL_RAW16_RIDGE",
    "ORIGINAL_RAW16_HGB",
    "CONSTANT_TRAIN_DOSE",
)


def original_source_sha256(path):
    digest=sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_uci308_genuine_independent_trials(path:Path):
    """All official feature rows are metadata, never extra physical trials."""
    with ZipFile(path) as z:
        if set(z.namelist())!={"features.csv","rawdata.csv.gz"}:
            raise ValueError("Unexpected official UCI308 archive member set")
        with z.open("features.csv") as f:
            reader=csv.DictReader(io.TextIOWrapper(f,encoding="utf8"))
            metadata=list(reader)
        if len(metadata)!=EXPECTED_ORIGINAL_EXPERIMENTS:
            raise ValueError("Physical trial count is not the official 58")
        if not {"exp","batch","ace_conc","eth_conc"}.issubset(metadata[0]):
            raise ValueError("Reference metadata is absent")
        lookup={}
        for r in metadata:
            key=(r["batch"],r["exp"])
            if key in lookup:
                raise ValueError("Duplicated metadata physical experiment ID")
            if key[0] not in BATCHES:
                raise ValueError("Unexpected chronological acquisition batch")
            ace=float(r["ace_conc"])
            eth=float(r["eth_conc"])
            if any(not np.isfinite(q) or q<0 or q>1 for q in (ace,eth)):
                raise ValueError("Official nominal water-dilution fraction unexpected")
            lookup[key]=(ace,eth)
        if (tuple(sum(k[0]==name for k in lookup) for name in BATCHES)
                !=EXPECTED_BATCH_COUNTS):
            raise ValueError("The precommitted five real acquisition batches changed")
        observed={}
        raw_rows=0
        expected=[f"dR_t{i}" for i in range(1,N_RAW_SAMPLES_PER_SENSOR+1)]
        with z.open("rawdata.csv.gz") as stream:
            with gzip.GzipFile(fileobj=stream) as dec:
                reader=csv.DictReader(io.TextIOWrapper(dec,encoding="utf8"))
                head=reader.fieldnames or []
                if (head[:9]!=["sensor","sample","exp","batch","ace_conc",
                               "eth_conc","gas","lab","col"] or
                    head[9:]!=expected):
                    raise ValueError("Original raw 7500 sample and physical sensor schema changed")
                for record in reader:
                    raw_rows+=1
                    key=(record["batch"],record["exp"])
                    if key not in lookup:
                        raise ValueError("Sensor row has no experiment metadata")
                    sensor=int(record["sensor"])
                    if not 1<=sensor<=SENSOR_COUNT:
                        raise ValueError("Physical sensor ID outside 1..16")
                    if (not np.isclose(float(record["ace_conc"]),lookup[key][0])
                            or not np.isclose(float(record["eth_conc"]),lookup[key][1])):
                        raise ValueError("Reference gas preparation doses do not match sensor row")
                    id=(key,sensor)
                    if id in observed:
                        raise ValueError("Same trial and sensor was duplicated")
                    readings=np.asarray([record[n] for n in expected],
                                        dtype=np.float64)
                    if (len(readings)!=N_RAW_SAMPLES_PER_SENSOR
                            or not np.isfinite(readings).all()):
                        raise ValueError("Raw real sensor timeline malformed or incomplete")
                    measured_bins=np.median(
                        readings.reshape(DOWNSAMPLED_TIME_BINS,RAW_PER_BIN),
                        axis=1)
                    observed[id]=measured_bins
    if (raw_rows!=58*16 or len(observed)!=58*16 or
        set(k for k,_ in observed)!=set(lookup)):
        raise ValueError("Original real 58 trials x 16 sensors not completely observed")
    keys=sorted(lookup,key=lambda key:(BATCHES.index(key[0]),int(key[1])))
    x=np.empty((58,60,16),dtype=np.float64)
    labels=np.empty((58,2),dtype=np.float64)
    batch=np.empty(58,dtype=np.int64)
    for i,key in enumerate(keys):
        x[i]=np.stack([observed[key,s] for s in range(1,17)],axis=1)
        labels[i]=lookup[key]
        batch[i]=1+BATCHES.index(key[0])
    groups={
        "TRAIN":np.flatnonzero(batch<=3),
        "VALID":np.flatnonzero(batch==4),
        "TEST":np.flatnonzero(batch==5),
    }
    if tuple(len(groups[k]) for k in ("TRAIN","VALID","TEST"))!=(39,11,8):
        raise ValueError("Precommitted independent trial split broken")
    if len(set(np.concatenate(list(groups.values()))))!=58:
        raise ValueError("Trial appears in two chronological groups")
    return x,labels,batch,groups,{
        "independent_physical_trials":58,
        "one_original_real_trial_is_one_statistical_unit":True,
        "physical_sensor_channels_per_trial":16,
        "original_measured_time_points_per_sensor":7500,
        "time_bin_seconds":5,
        "measured_time_points_per_time_bin":125,
        "source_time_bins_5_seconds_per_trial":60,
        "raw_data_sensor_row_count":raw_rows,
        "physical_measured_trial_counts_by_ordered_acquisition_batch":
            dict(zip(BATCHES,EXPECTED_BATCH_COUNTS)),
        "source_features_file_is_metadata_of_same_trials_not_new_trials":True,
        "frozen_trial_split":{"TRAIN":list(BATCHES[:3]),
                              "VALID":[BATCHES[3]],"TEST":[BATCHES[4]]},
        "reference_target_units":"nominal acetone/ethanol vol.percent dilution in water, NOT gas-phase ppm",
        "only_5_actual_acquisition_sessions":True,
    }


def signed_concept_ridge(train_features,train_reference,eval_features):
    train=np.asarray(train_features,dtype=np.float64)
    test=np.asarray(eval_features,dtype=np.float64)
    scaler=StandardScaler().fit(train-.5)
    model=Ridge(alpha=1.)
    model.fit(scaler.transform(train-.5),train_reference)
    beta=model.coef_/scaler.scale_
    bias=float(model.intercept_-(beta@scaler.mean_))
    parts=(test-.5)*beta[None,:]
    prediction=bias+parts.sum(axis=1)
    original=model.predict(scaler.transform(test-.5))
    err=float(np.max(np.abs(original-prediction)))
    if err>1e-9:
        raise AssertionError("V70 exact signed source-only Ridge reconstruction failed")
    return prediction,{
        "head_model":"Ridge(alpha=1),TRAIN-only state scaling",
        "source_membership_original_units_bias":bias,
        "signed_original_fuzzy_coordinates":beta.tolist(),
        "n_training_original_real_trials":len(train),
        "n_concept_only_terms":len(beta),
        "max_source_only_signed_additive_prediction_error":err,
        "no_original_sensor_to_risk_bypass":True,
        "supervised_gas_preparation_doses_used_only_in_head":True,
    }


def physical_trial_metrics(y,estimate,TRAIN_std):
    y=np.asarray(y,float);p=np.asarray(estimate,float)
    if not len(y)>=2 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Required original independent trial measurements missing")
    if TRAIN_std<=1e-9:
        raise ValueError("Nominal preparation dose reference constant on TRAIN")
    if np.std(y)>1e-9:
        r2=float(r2_score(y,p))
    else:
        r2=None
    if np.std(y)>1e-9 and np.std(p)>1e-9:
        rho=float(spearmanr(y,p).statistic)
        if not np.isfinite(rho):rho=None
    else:
        rho=None
    return {
        "n_independent_experiments":len(y),
        "nominal_water_dilution_volpercent_MAE":float(mean_absolute_error(y,p)),
        "nominal_water_dilution_volpercent_RMSE":float(np.sqrt(mean_squared_error(y,p))),
        "normalized_MAE_relative_TRAIN_target_std":float(mean_absolute_error(y,p)/TRAIN_std),
        "R2":r2,
        "spearman_descriptive":rho,
        "sample_size_too_small_for_stable_error_claim":len(y)<20,
    }


def partial_spearman(a,b,c):
    a=np.asarray(a,float);b=np.asarray(b,float);c=np.asarray(c,float)
    if len(a)<6 or min(np.std(a),np.std(b),np.std(c))<1e-12:
        return None
    mat=np.column_stack([np.ones(len(a)),rankdata(c)])
    def resid(x):
        v=rankdata(x)
        return v-mat@np.linalg.lstsq(mat,v,rcond=None)[0]
    u=resid(a);v=resid(b)
    if np.std(u)<1e-9 or np.std(v)<1e-9:
        return None
    r=float(np.corrcoef(u,v)[0,1])
    return r if np.isfinite(r) else None


def physical_semantics_audit(source,x,reference,groups,seed=20261970):
    """Reference used AFTER source freeze to name axes; small n is descriptive."""
    channels=source.source_channels(x)
    # Use original anonymous absolute axes and real trial-level late-time mean.
    zs=channels[:,-15:,:source.K].mean(axis=1)
    associations={}
    for cohort in ("VALID","TEST"):
        ix=groups[cohort]
        vals=zs[ix];dose=reference[ix]
        associations[cohort]=[
            {"axis":int(k),
             analyte:{
                 "partial_spearman_controlling_other_preparation_dose":
                     partial_spearman(vals[:,k],dose[:,j],dose[:,1-j]),
                 "n_original_physical_experiments":len(ix),
                 "unique_gas_chemical_identity_established":False,
             }
             for j,analyte in enumerate(ANALYTES)}
            for k in range(source.K)]
    if source.K<2:
        alignment={"status":"ABSTAIN_K_LT_TWO_NOT_CHEMICAL_IDENTITY"}
    else:
        valid=associations["VALID"]
        matrix=np.asarray([[
            abs(valid[k][analyte]["partial_spearman_controlling_other_preparation_dose"]
                or 0.)
            for k in range(source.K)] for analyte in ANALYTES])
        rows,cols=linear_sum_assignment(-matrix)
        mapping={analyte:int(cols[list(rows).index(j)])
                 for j,analyte in enumerate(ANALYTES)}
        alignment={
            "status":"VALID_ASSISTED_NAMING_NOT_AUTONOMOUS_PHYSICAL_IDENTIFICATION",
            "VALID_frozen_one_to_one_axis_mapping":mapping,
            "TEST_no_rematch_partial_rho":{
                analyte:associations["TEST"][mapping[analyte]][analyte][
                    "partial_spearman_controlling_other_preparation_dose"]
                for analyte in ANALYTES},
            "n_independent_VALID_trials":len(groups["VALID"]),
            "n_independent_TEST_trials":len(groups["TEST"]),
            "independent_chemical_analyzer_gas_measurement_available":False,
            "physical_source_identity_proven":False,
        }
    rng=np.random.default_rng(seed)
    perm=reference[groups["TEST"]][rng.permutation(len(groups["TEST"]))]
    associations["TEST_shuffled_nominal_dose_control"]=[
        {"axis":int(k),
         **{analyte:partial_spearman(
             zs[groups["TEST"],k],perm[:,j],perm[:,1-j])
            for j,analyte in enumerate(ANALYTES)}}
        for k in range(source.K)]
    return {"all_source_axis_trial_level_associations":associations,
            "VALID_only_postfit_axis_matching":alignment,
            "all_gas_names_used_only_after_unsupervised_source_fit":True}


def run(path:Path,out:Path):
    if out.exists():
        raise FileExistsError("Do not overwrite independent V70 source results")
    x,targets,batches,groups,schema=load_uci308_genuine_independent_trials(path)
    train=groups["TRAIN"]
    source=TrialOnlyDualStreamBank().fit_unlabeled(x[train])
    fixed_sha=source.sha
    features={
        "V70_STATIC_DUALSTREAM_MULTIFUZZY":source.fuzzy_features(x),
        "V70_LINEAR_ABSOLUTE_PLUS_CAUSAL_DELTA":source.linear_features(x),
        "ORIGINAL_RAW16_RIDGE":source.raw_features(x),
        "ORIGINAL_RAW16_HGB":source.raw_features(x),
    }
    results=[]
    concept_heads=[]
    for j,analyte in enumerate(ANALYTES):
        y_train=targets[train,j]
        if np.std(y_train)<=1e-9:
            raise ValueError("No observed independent variation in TRAIN water preparation levels")
        for cohort in ("VALID","TEST"):
            index=groups[cohort]
            y=targets[index,j]
            for method in METHODS:
                if method=="CONSTANT_TRAIN_DOSE":
                    pred=np.full(len(index),float(np.mean(y_train)))
                elif method=="V70_STATIC_DUALSTREAM_MULTIFUZZY":
                    pred,head=signed_concept_ridge(
                        features[method][train],y_train,features[method][index])
                    if cohort=="VALID":concept_heads.append({
                        "reference":analyte,"head":head})
                elif method=="ORIGINAL_RAW16_HGB":
                    model=HistGradientBoostingRegressor(
                        max_iter=100,max_leaf_nodes=15,
                        l2_regularization=1.,random_state=0)
                    model.fit(features[method][train],y_train)
                    pred=model.predict(features[method][index])
                else:
                    model=make_pipeline(StandardScaler(),Ridge(alpha=1.))
                    model.fit(features[method][train],y_train)
                    pred=model.predict(features[method][index])
                results.append({
                    "target_preparation_dose":analyte,
                    "cohort":cohort,
                    "model":method,
                    "n_independent_physical_trials":len(index),
                    "gas_concentration_unit_not_ppm":"nominal water dilution vol.%",
                    "physical_concept_source_K":source.K,
                    "source_fit_used_no_gas_outcomes":True,
                    **physical_trial_metrics(y,pred,float(np.std(y_train)))})
    if fixed_sha!=source.sha:
        raise AssertionError("Risk-head dose labels modified frozen anonymous source")
    audit=physical_semantics_audit(source,x,targets,groups)
    receipt={
        "status":"V70_NEW_REAL_ACETONE_ETHANOL_TWO_ANALYTE_MIXTURE_TRIALS_EXECUTED",
        "official_dataset":"UCI308 Gas Sensor Array Under Flow Modulation",
        "official_doi":"10.24432/C5BG7G",
        "original_official_zip_sha256":original_source_sha256(path),
        "dataset_family_never_tested_in_previous_ConceptFAN_V53_V69":True,
        "dataset_physical_trial_structure":schema,
        "real_original_independent_trial_split":{
            name:len(inds) for name,inds in groups.items()},
        "source_bank":source.public_passport(),
        "source_frozen_before_ANY_reference_head":True,
        "source_fit_used_NO_preparation_doses_or_gas_labels":True,
        "n_independent_real_physical_TEST_trials":len(groups["TEST"]),
        "all_predeclared_regression_metrics":results,
        "exact_additive_signed_concept_head_receipts":concept_heads,
        "frozen_source_postfit_chemical_semantic_audit":audit,
        "real_physical_source_identity_proven":False,
        "water_dilution_percent_is_not_airborne_gas_concentration":True,
        "not_V65_recurrent_neural_evaluation":True,
        "scientific_limit":"Single apparatus, five sessions and only 8 independently heldout TEST experiments",
        "original_accepted_IOI_PIERE_manuscripts_unchanged":True,
        "protocol":"docs/medical/CONCEPTFAN_V70_FROZEN_UNSEEN_TWO_GAS_MIXTURE_PROTOCOL_20261113.md",
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"STATUS.json").write_text(json.dumps(
        receipt,indent=2,ensure_ascii=False,allow_nan=False)+"\n",
        encoding="utf8")
    with (out/"METRICS.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader();writer.writerows(results)
    print(json.dumps({
        "event":"V70_INDEPENDENT_REAL_MIXTURE_TRIALS_DONE",
        "real_trial_counts":{name:len(v) for name,v in groups.items()},
        "K_statistical":source.K,
        "test":[{k:row[k] for k in ("target_preparation_dose","model","R2",
                                    "nominal_water_dilution_volpercent_MAE")}
                for row in results if row["cohort"]=="TEST"],
        "physical_semantic_matching":audit["VALID_only_postfit_axis_matching"],
    },ensure_ascii=False,allow_nan=False),flush=True)
    return receipt


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--uci-zip",required=True,type=Path)
    p.add_argument("--out-dir",required=True,type=Path)
    a=p.parse_args()
    run(a.uci_zip,a.out_dir)


if __name__=="__main__":
    main()
