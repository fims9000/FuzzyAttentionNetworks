#!/usr/bin/env python3
"""V60: real UCI Air Quality held-out certified analyzer audit.

Only the five PT08 low-cost sensor responses are source/classification inputs.
Analyzer CO/NOx values are used for post-source TRAIN risk labels and held-out
semantic association checks; not for self-supervised concept training.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conceptfan_discovery.robust_observable_concepts_v54 import (
    RobustGroupConceptFAN, SpectralObservableConceptFAN
)
from conceptfan_discovery.native_temporal_fuzzy_v56 import (
    NativeSelfSupervisedConceptFAN,TrainOnlySourceBank
)
from scripts.experiment_autonomous_temporal_v53 import metric,raw_features

SOURCE_COLUMNS=(
    "PT08.S1(CO)","PT08.S2(NMHC)","PT08.S3(NOx)",
    "PT08.S4(NO2)","PT08.S5(O3)")
REFERENCE_COLUMNS=("CO(GT)","NOx(GT)")
INPUT_STEPS=12
BLOCK_STEPS=13
SOURCE_SEED=20261601
NETWORK_SEED=11


def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):
            h.update(block)
    return h.hexdigest()


def read_official_csv(archive:Path)->pd.DataFrame:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    with zipfile.ZipFile(archive) as z:
        names=[name for name in z.namelist()
               if Path(name).name.lower()=="airqualityuci.csv"]
        if len(names)!=1:
            raise ValueError("Expected exactly one official AirQualityUCI.csv")
        raw=z.read(names[0])
    frame=pd.read_csv(io.BytesIO(raw),sep=";",decimal=",",
                      encoding="latin-1",low_memory=False)
    required=["Date","Time",*SOURCE_COLUMNS,*REFERENCE_COLUMNS]
    if any(name not in frame.columns for name in required):
        raise ValueError("Official CSV missing a preregistered sensor/reference column")
    time=frame["Time"].astype(str).str.replace(".",":",regex=False)
    stamps=pd.to_datetime(
        frame["Date"].astype(str)+" "+time,
        format="%d/%m/%Y %H:%M:%S",errors="coerce")
    good=stamps.notna()
    frame=frame.loc[good,required].copy().reset_index(drop=True)
    stamps=pd.DatetimeIndex(stamps.loc[good].to_numpy())
    if len(frame)<5000 or not stamps.is_monotonic_increasing or stamps.has_duplicates:
        raise ValueError("Incomplete, nonchronological or duplicate UCI hourly timeline")
    for name in (*SOURCE_COLUMNS,*REFERENCE_COLUMNS):
        value=pd.to_numeric(frame[name],errors="coerce")
        frame[name]=value.mask(value<=-200).astype(float)
    frame.insert(0,"timestamp",stamps)
    return frame


def build_disjoint_blocks(frame:pd.DataFrame,*,history:int=INPUT_STEPS):
    if history!=12 or any(name not in frame.columns
                          for name in (*SOURCE_COLUMNS,*REFERENCE_COLUMNS,"timestamp")):
        raise ValueError("Only the registered 12-hour source history is allowed")
    data=frame.reset_index(drop=True)
    times=pd.DatetimeIndex(data["timestamp"])
    sensor=[];history_ref=[];future_ref=[];days=[];discarded=0
    for start in range(0,len(data)-BLOCK_STEPS+1,BLOCK_STEPS):
        idx=np.arange(start,start+BLOCK_STEPS)
        # Pandas 3 may store DatetimeIndex at microsecond rather than
        # nanosecond precision; compare timedeltas, never raw asi8 units.
        if not np.all((times[idx[1:]]-times[idx[:-1]])==pd.Timedelta(hours=1)):
            discarded+=1
            continue
        sensor.append(data.loc[idx[:-1],list(SOURCE_COLUMNS)].to_numpy(dtype=float))
        history_ref.append(data.loc[idx[:-1],list(REFERENCE_COLUMNS)].to_numpy(dtype=float))
        future_ref.append(data.loc[idx[-1],list(REFERENCE_COLUMNS)].to_numpy(dtype=float))
        days.append(str(times[idx[0]].date()))
    if not sensor:
        raise ValueError("No nonoverlapping contiguous source windows")
    x=np.stack(sensor);ref=np.stack(history_ref);future=np.stack(future_ref)
    n=len(x);tr=int(.60*n);val=int(.80*n)
    splits={"TRAIN":np.arange(0,tr),"VALID":np.arange(tr,val),
            "TEST":np.arange(val,n)}
    if min(len(v) for v in splits.values())<50 or len(splits["TRAIN"])<300:
        raise ValueError("Insufficient chronological episodes for V56")
    return {"raw_x":x,"reference_history":ref,"reference_future":future,
            "day_start":np.asarray(days),"splits":splits,
            "n_discarded_noncontiguous_windows":discarded,
            "n_official_nonempty_hours":len(data),
            "n_ignored_trailing_hours":len(data)%BLOCK_STEPS}


def train_only_normalize(raw:np.ndarray,train_index:np.ndarray):
    if raw.ndim!=3 or raw.shape[-1]!=len(SOURCE_COLUMNS):
        raise ValueError("Exactly five PT08 response channels permitted")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        med=np.nanmedian(raw[train_index],axis=(0,1))
        low=np.nanquantile(raw[train_index],.25,axis=(0,1))
        high=np.nanquantile(raw[train_index],.75,axis=(0,1))
    scale=high-low
    if not np.isfinite(med).all() or not np.isfinite(scale).all() or np.any(scale<1e-6):
        raise ValueError("Train-only sensor medians/IQR unsupported")
    mask=np.isfinite(raw)
    x=np.where(mask,np.clip((np.where(mask,raw,med)-med)/scale,-8,8),0.)
    return x.astype(np.float32),mask,{"medians":med.tolist(),"iqr":scale.tolist()}


def split_labels(future:np.ndarray,train_index:np.ndarray,j:int):
    observed_train=future[train_index,j]
    observed=np.isfinite(observed_train)
    if observed.sum()<100:
        raise ValueError("Fewer than 100 valid TRAIN certified-analyzer references")
    threshold=float(np.median(observed_train[observed]))
    seen=np.isfinite(future[:,j])
    y=(future[:,j]>threshold).astype(np.int64)
    if len(np.unique(y[train_index][seen[train_index]]))!=2:
        raise ValueError("Invalid TRAIN class balance after reference cutoff")
    return y,seen,threshold


def _masked_episode_mean(history):
    m=np.isfinite(history);n=m.sum(axis=1)
    return np.divide(np.where(m,history,0.).sum(axis=1),n,
                     out=np.full(n.shape,np.nan,dtype=float),where=n>0)


def _spearman(a,b):
    m=np.isfinite(a)&np.isfinite(b)
    if int(m.sum())<30 or np.std(a[m])<1e-9 or np.std(b[m])<1e-9:
        return {"rho":None,"n":int(m.sum())}
    rho=float(spearmanr(a[m],b[m]).statistic)
    return {"rho":rho if np.isfinite(rho) else None,"n":int(m.sum())}


def semantic_association(concepts,raw,reference_history,indices,
                         *,permutation_seed=20261691):
    if concepts.ndim!=2 or len(concepts)!=len(raw) or len(concepts)!=len(reference_history):
        raise ValueError("Unaligned held-out semantic comparison")
    c=concepts[indices]
    sensor=_masked_episode_mean(raw)[indices]
    ref=_masked_episode_mean(reference_history)[indices]
    rng=np.random.default_rng(permutation_seed)
    shuffled=ref[rng.permutation(len(ref))]
    def cross(features,references):
        return [{"feature_index":int(k),
                 **{name:_spearman(features[:,k],references[:,j])
                    for j,name in enumerate(REFERENCE_COLUMNS)}}
                for k in range(features.shape[-1])]
    full=cross(c,ref);permuted=cross(c,shuffled);raw_associations=cross(sensor,ref)
    def max_abs(rows,name):
        values=[abs(z[name]["rho"]) for z in rows if z[name]["rho"] is not None]
        return max(values) if values else None
    return {
        "all_concept_reference_ranks":full,
        "all_permuted_reference_negative_control_ranks":permuted,
        "all_raw_sensors_reference_ranks":raw_associations,
        "max_abs_spearman_by_reference":{
            name:{"concept":max_abs(full,name),
                  "permuted_concept":max_abs(permuted,name),
                  "raw_sensor":max_abs(raw_associations,name)}
            for name in REFERENCE_COLUMNS},
        "semantic_identity_confirmed":False,
        "frozen_no_reference_outcome_training_of_sources":True}


def _fit_target_heads(xtr,mtr,ytr,xev,mev,source,native):
    if source.route=="MEASURED_GROUPS":
        static=RobustGroupConceptFAN(source.source)
    else:
        static=SpectralObservableConceptFAN(source.source)
    # Both arms use the IDENTICAL all-TRAIN unlabeled source vocabulary;
    # label availability cannot change source definitions.
    static.fit_risk_head(xtr,mtr,ytr)
    native.fit_risk_head(xtr,mtr,ytr)
    pred_static=static.predict(xev,mev)
    pred_native=native.predict(xev,mev)
    if (pred_native["max_additive_logit_error"]>1e-12 or
        pred_static.get("logit_additive_error",pred_static.get("max_additive_error",1.))>1e-12):
        raise AssertionError("Exact source-only output decomposition lost")
    ftr=raw_features(xtr,mtr);fev=raw_features(xev,mev)
    models={
        "raw_logistic":make_pipeline(StandardScaler(),
                   LogisticRegression(C=1.,max_iter=1500)),
        "raw_pca_logistic":make_pipeline(StandardScaler(),
                   PCA(n_components=max(1,min(source.k,ftr.shape[-1],len(xtr)-1))),
                   LogisticRegression(C=1.,max_iter=1500)),
        "raw_hist_gradient_boosting":HistGradientBoostingClassifier(
                   max_iter=100,max_leaf_nodes=15,
                   l2_regularization=1.,random_state=0)}
    probabilities={"v55_static_concepts":pred_static["probability"],
                   "v56_native_concepts":pred_native["probability"]}
    for method,model in models.items():
        model.fit(ftr,ytr)
        probabilities[method]=model.predict_proba(fev)[:,1]
    return probabilities


def run(archive:Path,out:Path):
    if out.exists():
        raise FileExistsError("Research receipts must not be overwritten")
    frame=read_official_csv(archive)
    windows=build_disjoint_blocks(frame)
    splits=windows["splits"]
    x,m,normalization=train_only_normalize(windows["raw_x"],splits["TRAIN"])
    bank=TrainOnlySourceBank(seed=SOURCE_SEED,permutations=60,bootstraps=20)
    bank.fit_unlabeled(x[splits["TRAIN"]],m[splits["TRAIN"]])
    native=NativeSelfSupervisedConceptFAN(
        bank,neural_seed=NETWORK_SEED,epochs=8).fit_source_unlabeled(
            x[splits["TRAIN"]],m[splits["TRAIN"]])
    source_hash=native.frozen_network_sha
    # Analysis uses the frozen unsupervised concept states BEFORE task labels.
    all_concepts=native._transform(x,m)["summaries"]
    association={
        name:semantic_association(all_concepts,windows["raw_x"],
               windows["reference_history"],splits[name],
               permutation_seed=20261691+(name=="TEST"))
        for name in ("VALID","TEST")}
    rows=[];task_receipts={}
    for j,reference_name in enumerate(REFERENCE_COLUMNS):
        labels,available,threshold=split_labels(
            windows["reference_future"],splits["TRAIN"],j)
        itr=splits["TRAIN"][available[splits["TRAIN"]]]
        task_receipts[reference_name]={
            "threshold_train_median":threshold,
            "train_supervised_count":len(itr),
            "target_coverage":{
                name:{"n_total":len(idx),"n_with_analyzer":int(available[idx].sum()),
                      "class_prevalence":float(labels[idx][available[idx]].mean())
                        if available[idx].any() else None}
                for name,idx in splits.items()}}
        for cohort in ("VALID","TEST"):
            ie=splits[cohort][available[splits[cohort]]]
            if len(ie)<30:
                raise ValueError("Insufficient independent analyzer records")
            probabilities=_fit_target_heads(
                x[itr],m[itr],labels[itr],x[ie],m[ie],bank,native)
            if native.frozen_network_sha!=source_hash:
                raise AssertionError("Target outcome modified unsupervised source")
            for method,p in probabilities.items():
                rows.append({
                    "target_future_analyzer":reference_name,
                    "cohort":cohort,"method":method,
                    "n_reference_labeled":len(ie),
                    "positive_fraction":float(labels[ie].mean()),
                    "source_route":bank.route,"source_slots_K":bank.k,
                    **metric(labels[ie],p)})
    result={
        "status":"REAL_AIRQUALITY_V60_EXPLORATORY_COMPLETE",
        "data_reference":"UCI Air Quality dataset id 360, DOI 10.24432/C59K5F",
        "official_url":"https://archive.ics.uci.edu/dataset/360/air+quality",
        "official_zip_sha256":sha256_file(archive),
        "n_original_nonempty_hours":windows["n_official_nonempty_hours"],
        "n_disjoint_contiguous_13h_windows":len(x),
        "n_discarded_noncontiguous_windows":windows["n_discarded_noncontiguous_windows"],
        "n_unused_trailing_rows":windows["n_ignored_trailing_hours"],
        "chronological_split_counts":{name:len(idx) for name,idx in splits.items()},
        "chronological_date_ranges":{
            name:{"first":windows["day_start"][idx[0]],
                  "last":windows["day_start"][idx[-1]]}
            for name,idx in splits.items()},
        "input_only_5_mox_sensors":list(SOURCE_COLUMNS),
        "analyzers_never_passed_to_concept_encoder":True,
        "train_only_normalization":normalization,
        "train_only_source":bank.public_passport(),
        "native_network_sha256":source_hash,
        "unlabeled_native_losses":native.loss_receipt,
        "target_label_train_receipts":task_receipts,
        "chronological_predictions_metrics":rows,
        "external_reference_association_audit":association,
        "physical_source_semantics_identified":False,
        "recorded_independent_sensor_fault_ground_truth":False,
        "single_station_only":True,
        "no_test_tuning":True,
        "raw_time_series_in_artifact":False,
        "frozen_protocol":"docs/medical/CONCEPTFAN_V60_FROZEN_REAL_AIRQUALITY_PROTOCOL_20261113.md"}
    out.mkdir(parents=True,exist_ok=False)
    (out/"STATUS.json").write_text(json.dumps(
        result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    import csv
    with (out/"METRICS.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({
        "event":"V60_REAL_AIRQUALITY_COMPLETE",
        "status":result["status"],"windows":len(x),
        "K":bank.k,"route":bank.route,
        "test":list(z for z in rows if z["cohort"]=="TEST"),
        "test_concept_association":association["TEST"]["max_abs_spearman_by_reference"]
    },ensure_ascii=False,allow_nan=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--uci-zip",required=True,type=Path)
    p.add_argument("--out-dir",required=True,type=Path)
    args=p.parse_args()
    run(args.uci_zip,args.out_dir)


if __name__=="__main__":
    main()
