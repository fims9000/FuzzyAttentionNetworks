#!/usr/bin/env python3
"""V61: real independent gas-plume interventions, run-level Concepts vs controls.

No gas metadata, file paths, ambient weather or GC-MS mean concentration are
inputs to sensor source fitting or the self-supervised temporal concept encoder.
Each whole 300-second experimental recording is ONE independent test unit.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conceptfan_discovery.native_temporal_fuzzy_v56 import (
    NativeSelfSupervisedConceptFAN,TrainOnlySourceBank
)
from conceptfan_discovery.robust_observable_concepts_v54 import (
    SpectralObservableConceptFAN,fit_observable_spectral_definitions
)
from scripts.experiment_autonomous_temporal_v53 import metric,raw_features

DATASET_ID=309
SOURCE_SEED=20261731
SPLIT_SEED=20261701
NETWORK_SEED=11
LEVELS=("N","L","M","H")
GAS_PPM={
    "ethylene":{"N":0.,"L":31.,"M":46.,"H":96.},
    "co":{"N":0.,"L":270.,"M":397.,"H":460.},
    "methane":{"N":0.,"L":51.,"M":115.,"H":131.},
}
RUN_PATTERN=re.compile(
    r"(?P<run>\d{3})_Et_(?P<eth>[nLMH])_(?P<second>CO|Me)_(?P<second_level>[nLMH])$",
    flags=re.IGNORECASE
)
BIN_SECONDS=5
N_BINS=60
N_SENSORS=8
TRAIN_RUNS=120


def source_sha256(path:Path):
    h=sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda:file.read(1048576),b""):
            h.update(chunk)
    return h.hexdigest()


def parse_run_name(filename:str):
    name=Path(filename).name
    m=RUN_PATTERN.fullmatch(name)
    if m is None:
        raise ValueError("Not an official gas source configuration file: "+name)
    run=int(m.group("run"))
    eth=m.group("eth").upper()
    secondary=m.group("second").upper()
    level=m.group("second_level").upper()
    other="co" if secondary=="CO" else "methane"
    result={
        "run_index":run,"ethyl_level":eth,
        "other_type":other,"other_level":level,
        "gas_config":f"{eth}-{other}-{level}",
        "ppm":{
            "ethylene":GAS_PPM["ethylene"][eth],
            "co":GAS_PPM["co"][level] if other=="co" else 0.,
            "methane":GAS_PPM["methane"][level] if other=="methane" else 0.,
        },
        "targets":{
            "ethylene_present":int(eth!="N"),
            "co_present":int(other=="co" and level!="N"),
            "methane_present":int(other=="methane" and level!="N"),
        }
    }
    return result


def time_median_bins(contents:bytes):
    """One run -> sixty 5-second sensor bins. No interpolation or new labels."""
    matrix=np.loadtxt(io.BytesIO(contents),delimiter=",",dtype=np.float64,ndmin=2)
    if matrix.ndim!=2 or matrix.shape[1]!=11 or len(matrix)<1000:
        raise ValueError("Expected 11-column downsampled official sensor timeline")
    time=matrix[:,0]
    x=matrix[:,3:11]
    if not np.isfinite(matrix).all() or (np.diff(time)<0).any():
        raise ValueError("Nonfinite or nonchronological sensor timeline")
    intervals=np.floor(time/BIN_SECONDS).astype(int)
    response=np.empty((N_BINS,N_SENSORS),dtype=np.float64)
    sizes=[]
    for bin_id in range(N_BINS):
        seen=(intervals==bin_id)
        count=int(seen.sum())
        if count<5:
            raise ValueError("Insufficient measured samples in one 5s bin: "+str(bin_id))
        response[bin_id]=np.median(x[seen],axis=0)
        sizes.append(count)
    if np.any(~np.isfinite(response)):
        raise ValueError("Invalid physical sensor response after binning")
    return response,{"n_time_rows":len(matrix),"min_sensor_samples_per_bin":min(sizes)}


def load_official_runs(archive:Path):
    """Choose downsampled 100ms copies, never duplicate raw 20ms files."""
    if not archive.is_file():
        raise FileNotFoundError(archive)
    with zipfile.ZipFile(archive) as z:
        names=sorted(v for v in z.namelist()
                     if v.startswith("dataset_twosources_downsampled/")
                     and not v.endswith("/"))
        if len(names)!=180:
            raise ValueError("Expected 180 official independent downsampled runs")
        records=[];values=[];counts=[]
        for member in names:
            meta=parse_run_name(member)
            matrix,receipt=time_median_bins(z.read(member))
            records.append(meta)
            values.append(matrix)
            counts.append(receipt)
    if len(set(meta["run_index"] for meta in records))!=180:
        raise ValueError("Duplicate real experimental run IDs")
    config_count=Counter(meta["gas_config"] for meta in records)
    if len(config_count)!=30 or set(config_count.values())!={6}:
        raise ValueError("Must have 30 configurations, exactly six repetitions each")
    return np.stack(values),records,{
        "n_independent_runs":180,
        "n_distinct_predeclared_conditions":30,
        "repetitions_per_condition":6,
        "n_seconds_per_run":300,
        "sampling_seconds_per_bin":5,
        "signal_tensor_shape":[180,60,8],
        "raw_time_rows_min":min(v["n_time_rows"] for v in counts),
        "smallest_5sec_bin_original_sample_count":min(
            v["min_sensor_samples_per_bin"] for v in counts),
        "no_raw_20ms_duplicate_read":True,
    }


def stratify_whole_runs(records,*,seed:int=SPLIT_SEED):
    grouped=defaultdict(list)
    for idx,meta in enumerate(records):
        grouped[meta["gas_config"]].append(idx)
    if len(grouped)!=30 or any(len(v)!=6 for v in grouped.values()):
        raise ValueError("Six independent repetitions per each of 30 gas configurations needed")
    rng=np.random.default_rng(seed)
    groups={"TRAIN":[],"VALID":[],"TEST":[]}
    for key in sorted(grouped):
        shuffled=rng.permutation(grouped[key]).tolist()
        groups["TRAIN"].extend(shuffled[:4])
        groups["VALID"].append(shuffled[4])
        groups["TEST"].append(shuffled[5])
    result={name:np.asarray(sorted(indices),dtype=int)
            for name,indices in groups.items()}
    if list(map(len,result.values()))!=[120,30,30]:
        raise AssertionError("Unexpected 4/1/1 run-level split")
    if len(set(np.concatenate(list(result.values()))))!=180:
        raise AssertionError("Recording appears in two groups")
    return result


def train_sensor_normalizer(raw,train_indices):
    reference=raw[train_indices]
    median=np.median(reference,axis=(0,1))
    lo=np.quantile(reference,.25,axis=(0,1))
    hi=np.quantile(reference,.75,axis=(0,1))
    scale=hi-lo
    if np.any(scale<=1e-6) or not np.isfinite(scale).all():
        raise ValueError("Sensor not variable or not observable in TRAIN")
    normalized=np.clip((raw-median[None,None,:])/scale,-8,8)
    return normalized.astype(np.float32),np.ones_like(raw,dtype=bool),{
        "medians":median.tolist(),"iqr":scale.tolist()}


def full_run_spectral_bank(train_x,train_mask):
    """120 independent runs establish source rank. Do NOT split runs into folds."""
    definition=fit_observable_spectral_definitions(
        train_x,train_mask,seed=SOURCE_SEED,
        permutations=60,bootstraps=20)
    bank=TrainOnlySourceBank(seed=SOURCE_SEED,permutations=60,bootstraps=20)
    bank.route="PREDICTIVE_SUBSPACE" if definition.rank else "OBSERVATION_ABSTAIN"
    bank.source=definition
    bank.k=definition.rank
    bank.tau=np.asarray(definition.thresholds,dtype=float)
    bank.width=np.asarray(definition.widths,dtype=float)
    bank.sha=sha256(json.dumps(
        definition.as_dict(),sort_keys=True,allow_nan=False).encode()).hexdigest()
    bank.train_receipt={
        "source_route":bank.route,
        "number_of_discovered_source_slots":bank.k,
        "source_sha256":bank.sha,
        "n_train_unlabeled":len(train_x),
        "independent_units":"WHOLE_WIND_TUNNEL_RUN",
        "spectral_only_no_unjustified_patient_fold_claim":True,
        "positive_lag_rank_bootstrap_runs":20,
        "outcome_labels_and_known_gas_levels_in_source_fit":False,
        "manual_source_number_or_K2_provided":False,
        "physical_source_identification_proven":False,
    }
    return bank


def raw_exposure_features(x,mask):
    # Registered source exposure window begins at second60 (index12),
    # finishes second240 (index48). This comparator uses known timing,
    # but no actual gas concentration/labels as a feature.
    def mean(start,end):
        observed=mask[:,start:end]
        values=np.where(observed,x[:,start:end],0.)
        return values.sum(axis=1)/np.maximum(1,observed.sum(axis=1))
    base=mean(0,12)
    active=mean(12,48)
    recovery=mean(48,60)
    whole=mean(0,60)
    return np.column_stack((whole,base,active,recovery,active-base,
                            mask.mean(axis=1)))


def run_level_spearman(x,y):
    mask=np.isfinite(x)&np.isfinite(y)
    if int(mask.sum())<10 or np.std(x[mask])<1e-9 or np.std(y[mask])<1e-9:
        return {"rho":None,"n":int(mask.sum())}
    value=float(spearmanr(x[mask],y[mask]).statistic)
    return {"rho":value if np.isfinite(value) else None,"n":int(mask.sum())}


def semantic_audit(concepts,raw,records,indices,seed):
    # Real source exposures have only run-level, *mean* concentration control.
    # There is NO concentration ground truth trajectory inside a run.
    c=(concepts[:,12:48].mean(axis=1)-concepts[:,:12].mean(axis=1))[indices]
    sensor=(raw[:,12:48].mean(axis=1)-raw[:,:12].mean(axis=1))[indices]
    ctrl=np.asarray([[m["ppm"][name] for name in ("ethylene","co","methane")]
                     for m in records],dtype=float)[indices]
    rng=np.random.default_rng(seed)
    shuffled=ctrl[rng.permutation(len(ctrl))]
    names=("ethylene","co","methane")
    def compare(features,reference):
        return [{"axis":int(k),**{
            name:run_level_spearman(features[:,k],reference[:,j])
            for j,name in enumerate(names)}}
            for k in range(features.shape[1])]
    full=compare(c,ctrl)
    null=compare(c,shuffled)
    sensor_full=compare(sensor,ctrl)
    def max_abs(rows,name):
        vals=[abs(row[name]["rho"]) for row in rows
              if row[name]["rho"] is not None]
        return max(vals) if vals else None
    return {
        "concept_exposure_minus_clean_air_by_gas":full,
        "shuffled_condition_negative_control":null,
        "raw_sensor_exposure_minus_baseline_by_gas":sensor_full,
        "max_abs_by_gas":{
            name:{"concept":max_abs(full,name),
                  "shuffled":max_abs(null,name),
                  "raw":max_abs(sensor_full,name)}
            for name in names},
        "n_independent_whole_test_runs":len(indices),
        "reference_levels_are_run_mean_controls_not_true_time_trajectories":True,
        "semantic_source_identity_proven":False,
    }


def train_methods(train_x,train_m,train_y,eval_x,eval_m,bank,native):
    static=SpectralObservableConceptFAN(bank.source).fit_risk_head(
        train_x,train_m,train_y)
    native.fit_risk_head(train_x,train_m,train_y)
    out_static=static.predict(eval_x,eval_m)
    out_native=native.predict(eval_x,eval_m)
    if (out_static["max_additive_error"]>1e-12
            or out_native["max_additive_logit_error"]>1e-12):
        raise AssertionError("Model has direct, unexplained raw-to-risk bypass")
    ftrain=raw_features(train_x,train_m);feval=raw_features(eval_x,eval_m)
    ftrain_exp=raw_exposure_features(train_x,train_m)
    feval_exp=raw_exposure_features(eval_x,eval_m)
    result={
        "v55_static_spectral":out_static["probability"],
        "v56_native_spectral":out_native["probability"],
    }
    models={
        "raw_logistic":(make_pipeline(StandardScaler(),
            LogisticRegression(C=1.,max_iter=1200)),ftrain,feval),
        "raw_pca_logistic":(make_pipeline(StandardScaler(),
            PCA(n_components=max(1,min(bank.k,len(train_x)-1,ftrain.shape[-1]))),
            LogisticRegression(C=1.,max_iter=1200)),ftrain,feval),
        "raw_exposure_logistic":(make_pipeline(StandardScaler(),
            LogisticRegression(C=1.,max_iter=1200)),ftrain_exp,feval_exp),
        "raw_exposure_hgb":(HistGradientBoostingClassifier(
            max_iter=100,max_leaf_nodes=15,l2_regularization=1.,
            random_state=0),ftrain_exp,feval_exp),
    }
    for name,(model,xtr,xte) in models.items():
        model.fit(xtr,train_y)
        result[name]=model.predict_proba(xte)[:,1]
    return result


def run(archive:Path,out:Path):
    if out.exists():
        raise FileExistsError("Do not replace previously computed V61 evidence")
    raw,records,schema=load_official_runs(archive)
    splits=stratify_whole_runs(records)
    x,m,normalizer=train_sensor_normalizer(raw,splits["TRAIN"])
    bank=full_run_spectral_bank(x[splits["TRAIN"]],m[splits["TRAIN"]])
    # 3 temporal crops per TRAIN recording are only dependent augmentation.
    # All 120 original physical recordings remain independent split units.
    temporal_train=x[splits["TRAIN"]].reshape(-1,20,N_SENSORS)
    temporal_mask=m[splits["TRAIN"]].reshape(-1,20,N_SENSORS)
    if len(temporal_train)!=360:
        raise AssertionError("Exactly three unlabeled augmentation windows per TRAIN run")
    native=NativeSelfSupervisedConceptFAN(
        bank,neural_seed=NETWORK_SEED,epochs=8).fit_source_unlabeled(
            temporal_train,temporal_mask)
    network_sha=native.frozen_network_sha
    frozen_concepts=native._transform(x,m)["trajectories"]
    audits={
        name:semantic_audit(frozen_concepts,raw,records,splits[name],
                            seed=20261760+(name=="TEST"))
        for name in ("VALID","TEST")}
    results=[];label_support={}
    for label in ("ethylene_present","co_present","methane_present"):
        y=np.asarray([meta["targets"][label] for meta in records],dtype=int)
        if set(np.unique(y[splits["TRAIN"]]))!={0,1}:
            raise ValueError("Missing train class in preregistered physical gas target")
        label_support[label]={
            name:{"n_whole_runs":len(indices),
                  "positive":int(y[indices].sum()),
                  "prevalence":float(y[indices].mean())}
            for name,indices in splits.items()}
        for cohort in ("VALID","TEST"):
            predictions=train_methods(
                x[splits["TRAIN"]],m[splits["TRAIN"]],y[splits["TRAIN"]],
                x[splits[cohort]],m[splits[cohort]],bank,native)
            if native.frozen_network_sha!=network_sha:
                raise AssertionError("Supervised physical gas label changed source representation")
            for method,p in predictions.items():
                results.append({
                    "gas_physical_control":label,
                    "cohort":cohort,
                    "method":method,
                    "n_whole_run_independent_cases":len(splits[cohort]),
                    "positive_prevalence":float(y[splits[cohort]].mean()),
                    "selected_observable_K":bank.k,
                    **metric(y[splits[cohort]],p)})
    evidence={
        "status":"V61_REAL_INDEPENDENT_GAS_PLUMES_EXPLORATORY_COMPLETE",
        "dataset":"UCI gas sensor array exposed to turbulent gas mixtures id=309",
        "dataset_doi":"10.24432/C5JS5P",
        "original_official_zip_sha256":source_sha256(archive),
        "whole_run_schema":schema,
        "split_whole_recording_counts":{k:len(v) for k,v in splits.items()},
        "each_configuration_train_valid_test_repetitions":[4,1,1],
        "dependent_20step_train_augmentation_count":len(temporal_train),
        "dependent_augmentations_not_claimed_independent":True,
        "input_only_8_physical_gas_sensors":True,
        "source_fitting_never_receives_gas_names_or_setpoints":True,
        "source_fit_independently_observed_run_count":len(splits["TRAIN"]),
        "source_bank":bank.public_passport(),
        "native_frozen_sha256":network_sha,
        "self_supervised_train_losses":native.loss_receipt,
        "train_only_normalization":normalizer,
        "physical_target_counts":label_support,
        "whole_run_valid_and_test_metrics":results,
        "independent_physical_setpoint_associations":audits,
        "gas_concentration_time_trajectory_labels_available":False,
        "real_individual_device_fault_ground_truth_available":False,
        "unique_physical_concept_identification_claimed":False,
        "preregistered_protocol":"docs/medical/CONCEPTFAN_V61_FROZEN_REAL_GAS_INTERVENTION_PROTOCOL_20261113.md",
        "output_contains_raw_timeline_or_file_metadata":False,
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"STATUS.json").write_text(
        json.dumps(evidence,ensure_ascii=False,allow_nan=False,indent=2)+"\n")
    import csv
    with (out/"METRICS.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader();writer.writerows(results)
    print(json.dumps({
        "event":"V61_REAL_CONTROL_COMPLETE","K":bank.k,
        "n_test_whole_runs":len(splits["TEST"]),
        "test_results":[r for r in results if r["cohort"]=="TEST"],
        "test_semantic":audits["TEST"]["max_abs_by_gas"],
    },allow_nan=False),flush=True)
    return evidence


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--uci-zip",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args()
    run(args.uci_zip,args.out_dir)


if __name__=="__main__":
    main()
