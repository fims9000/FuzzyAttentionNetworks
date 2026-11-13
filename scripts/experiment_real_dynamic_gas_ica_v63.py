#!/usr/bin/env python3
"""V63 real continuous gas sources: actual unseen UCI322 two-session challenge.

Only 16 physical sensor outputs enter source discovery and concept learning.
Two setpoint channels are read as outcomes and post-hoc reference controls;
they are NOT independently measured gas concentration trajectories.

All train/validation/test windows are disjoint in time with a 120s embargo.
Only TWO physical recording sessions exist, not hundreds of independent runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.stats import rankdata,spearmanr
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error

from conceptfan_discovery.native_temporal_fuzzy_v56 import NativeSelfSupervisedConceptFAN
from conceptfan_discovery.train_innovation_ica_bank_v63 import TrainOnlyInnovationBank
from scripts.experiment_real_gas_interventions_v61 import source_sha256

DATASET_FILES=("ethylene_CO.txt","ethylene_methane.txt")
T=120
EMBARGO_SECONDS=120
TRAIN_FRACTION=.60
VALID_FRACTION=.20
SECOND_COLS=("CO","methane")


def _read_official_second_medians(archive:zipfile.ZipFile,member:str):
    with archive.open(member) as reader:
        heading=reader.readline().decode("ascii",errors="strict").strip()
        if not (heading.startswith("Time (seconds)") and
                "Ethylene conc (ppm)" in heading):
            raise ValueError("Unexpected official 19-column dynamic gas header")
        seconds=[];samples=[];current=None;bucket=[];min_per_sec=20
        previous=-np.inf;n_raw=0;n_discard=0
        def flush():
            nonlocal n_discard
            if len(bucket)>=min_per_sec:
                seconds.append(current)
                samples.append(np.median(np.asarray(bucket),axis=0))
            else:
                n_discard+=1
        for payload in reader:
            if not payload.strip():
                continue
            row=np.fromstring(payload.decode("ascii"),sep=" ",dtype=np.float64)
            if len(row)!=19 or not np.isfinite(row).all():
                raise ValueError("Unexpected malformed/nonfinite UCI322 row")
            sec=float(row[0])
            if sec+1e-5<previous:
                raise ValueError("Source session time not chronological")
            previous=sec
            key=int(np.floor(sec+1e-7))
            if current is None:
                current=key
            if key!=current:
                flush()
                bucket=[]
                current=key
            bucket.append(row[1:19])
            n_raw+=1
        if current is not None:
            flush()
    if len(seconds)<10000:
        raise ValueError("Official stream yielded too few real recorded seconds")
    s=np.asarray(seconds,dtype=int)
    v=np.asarray(samples,dtype=np.float64)
    if v.shape!=(len(s),18) or np.any(np.diff(s)<=0):
        raise ValueError("Unaligned official second-wise field measurements")
    return s,v[:,2:],v[:,:2],{
        "recorded_raw_samples":int(n_raw),
        "available_one_second_medians":len(s),
        "seconds_rejected_insufficient_20_samples":n_discard,
        "first_recorded_second":int(s[0]),
        "last_recorded_second":int(s[-1]),
        "sensor_count":int(v.shape[1]-2),
        "concentration_columns_excluded_from_sensors":True,
    }


def split_windows_one_stream(seconds,readings,setpoints,source_name):
    """No windows cross a chronological boundary or 120s bilateral embargo."""
    secs=np.asarray(seconds,dtype=np.int64)
    if len(secs)!=len(readings) or len(secs)!=len(setpoints):
        raise ValueError("Timestamp/source/reference mismatch")
    if readings.shape[1]!=16 or setpoints.shape[1]!=2:
        raise ValueError("Exactly 16 sensors and separate two reference controls")
    left=int(secs[0]);maxsec=int(secs[-1])
    horizon=maxsec-left+1
    boundary1=left+int(horizon*TRAIN_FRACTION)
    boundary2=left+int(horizon*(TRAIN_FRACTION+VALID_FRACTION))
    batches={k:[] for k in ("TRAIN","VALID","TEST")}
    censored=Counter()
    for start in range(left,maxsec-T+2,T):
        end=start+T-1
        if end < boundary1-EMBARGO_SECONDS:
            split="TRAIN"
        elif (start >=boundary1+EMBARGO_SECONDS and
              end <boundary2-EMBARGO_SECONDS):
            split="VALID"
        elif start >=boundary2+EMBARGO_SECONDS:
            split="TEST"
        else:
            censored["embargo"]+=1
            continue
        expected=np.arange(start,start+T,dtype=np.int64)
        pos=np.searchsorted(secs,start,side="left")
        if pos+T>len(secs) or not np.array_equal(secs[pos:pos+T],expected):
            censored["missing_true_second"]+=1
            continue
        gas= setpoints[pos:pos+T]
        sensor=readings[pos:pos+T]
        if not np.isfinite(gas).all() or not np.isfinite(sensor).all():
            censored["invalid_value"]+=1
            continue
        # Measured controller setpoint, contemporaneous with last 30s sensors.
        source_idx=0 if source_name=="ethylene_CO.txt" else 1
        session_id=0 if source_idx==0 else 1
        block=f"{session_id}:{(start-left)//1800}"
        batches[split].append({
            "sensor":sensor.astype(np.float32),
            "setpoint_last30":np.median(gas[90:120],axis=0),
            "source_file_code":session_id,
            "train_halfhour_block":block,
            "first_recorded_second":start,
        })
    if min(len(batches[k]) for k in batches)<20:
        raise ValueError(f"Insufficient data for the frozen time split: {source_name}")
    receipt={
        "session_key":source_name,
        "n_splits":{key:len(value) for key,value in batches.items()},
        "seconds":horizon,
        "boundary_train_valid_second":boundary1,
        "boundary_valid_test_second":boundary2,
        "bilateral_embargo_seconds":EMBARGO_SECONDS,
        "censored_nonoverlapping_windows":dict(censored),
        "no_windows_crossing_temporal_boundary":True,
        "only_2_independent_full_recording_sessions_across_dataset":True,
    }
    return batches,receipt


def build_real_windows(path:Path):
    with zipfile.ZipFile(path) as z:
        files=sorted(f for f in z.namelist() if not f.endswith("/"))
        if files!=list(DATASET_FILES):
            raise ValueError("Official archive members changed, cannot proceed")
        raw=[];runs={name:[] for name in ("TRAIN","VALID","TEST")}
        for name in DATASET_FILES:
            secs,sensors,gt,record=_read_official_second_medians(z,name)
            sections,split=split_windows_one_stream(secs,sensors,gt,name)
            raw.append({"file":name,**record,**split})
            for label in runs:
                runs[label].extend(sections[label])
    if sum(len(p) for p in runs.values())<300:
        raise ValueError("Too few real full 120s windows")
    print(json.dumps({"event":"V63_OBSERVED_SCHEMA_ONLY","by_session":raw,"n_windows":{k:len(v) for k,v in runs.items()}},ensure_ascii=False),flush=True)
    return runs,raw


def train_only_sensor_scaling(runs):
    train=np.stack([r["sensor"] for r in runs["TRAIN"]],axis=0)
    flat=train.reshape(-1,16)
    med=np.median(flat,axis=0)
    q25,q75=np.quantile(flat,[.25,.75],axis=0)
    scale=q75-q25
    if not np.isfinite(scale).all() or np.any(scale<1e-5):
        raise ValueError("Nonvarying or nonfinite real sensor TRAIN source")
    x={};mask={};setpoints={};session={}
    for split,records in runs.items():
        raw=np.stack([r["sensor"] for r in records])
        normalized=np.clip((raw-med[None,None,:])/scale[None,None,:],-8.,8.)
        x[split]=normalized.astype(np.float32)
        mask[split]=np.ones_like(normalized,dtype=bool)
        setpoints[split]=np.stack([r["setpoint_last30"] for r in records])
        session[split]=np.asarray([r["source_file_code"] for r in records],dtype=int)
    group=np.asarray([r["train_halfhour_block"] for r in runs["TRAIN"]])
    return x,mask,setpoints,session,group,{
        "train_sensor_medians":med.tolist(),
        "train_sensor_iqr":scale.tolist(),
        "normalization_never_reads_valid_or_test":True,
        "input_sensor_dimensions":16,
    }


def reg_metrics(y,p,trainstd):
    y=np.asarray(y,float);p=np.asarray(p,float)
    if len(y)<20 or not np.isfinite(p).all() or trainstd<=1e-6:
        raise ValueError("Not enough valid reference setpoint variance")
    rho=float(spearmanr(y,p).statistic) if np.std(p)>1e-9 else None
    return {
        "n_correlated_windows_not_independent_sessions":len(y),
        "R2":float(r2_score(y,p)),
        "MAE_ppm":float(mean_absolute_error(y,p)),
        "NMAE_relative_to_train_target_std":float(mean_absolute_error(y,p)/trainstd),
        "spearman":rho if rho is None or np.isfinite(rho) else None,
    }


def _partial_rho(z,a,b):
    """Partial Spearman rho after linearly residualizing rank(a|b)."""
    zz=np.asarray(z,float);aa=np.asarray(a,float);bb=np.asarray(b,float)
    if len(zz)<15 or min(np.std(zz),np.std(aa),np.std(bb))<1e-9:
        return None
    zr=rankdata(zz);ar=rankdata(aa);br=rankdata(bb)
    reg=np.column_stack([np.ones(len(br)),br])
    zres=zr-reg@np.linalg.lstsq(reg,zr,rcond=None)[0]
    ares=ar-reg@np.linalg.lstsq(reg,ar,rcond=None)[0]
    if min(np.std(zres),np.std(ares))<1e-9:
        return None
    value=float(np.corrcoef(zres,ares)[0,1])
    return value if np.isfinite(value) else None


def semantic_scores(traj,raw_x,refs,session,*,seed:int=20261963):
    """All-axis conditional source audit, reference used POST source-freeze."""
    sequence=np.asarray(traj,float)
    if sequence.ndim!=3 or sequence.shape[1]!=120:
        raise ValueError("Requires 120 one-second concept states per real window")
    concepts=sequence[:,90:120,:].mean(axis=1)
    sensors=raw_x[:,90:120,:].mean(axis=1)
    # Ref cols [other gas,ethylene] as provided in official source
    rng=np.random.default_rng(seed)
    result={}
    for which,tag in ((0,"CO_recording"),(1,"methane_recording")):
        select=np.flatnonzero(session==which)
        yy=refs[select]
        c=concepts[select];s=sensors[select]
        shuffled=yy[rng.permutation(len(yy))]
        source_names=("other_gas","ethylene")
        def association(v,ref):
            return [{
                "axis_index":j,
                **{source_names[k]:{
                    "rho":_partial_rho(v[:,j],ref[:,k],ref[:,1-k]),
                    "n_windows":len(select)}
                   for k in (0,1)}
            } for j in range(v.shape[-1])]
        full=association(c,yy)
        raw=association(s,yy)
        perm=association(c,shuffled)
        def best(rows,name):
            vals=[abs(r[name]["rho"]) for r in rows
                  if r[name]["rho"] is not None]
            return max(vals) if vals else None
        result[tag]={
            "n_correlated_windows":len(select),
            "n_independent_full_recording_sessions":1,
            "full_anonymous_concept_partial_rhos":full,
            "full_raw_sensor_partial_rhos":raw,
            "shuffled_setpoint_negative_control":perm,
            "max_abs_conditional_by_target":{
                name:{"concept":best(full,name),
                      "raw_sensor":best(raw,name),
                      "shuffled":best(perm,name)}
                for name in source_names},
            "gas_identity_cannot_be_inferred_from_single_continuous_session":True,
        }
    return result


def fixed_valid_semantic_alignment(valid_audit,test_audit):
    """VALID-derived semantic *naming audit* is explicitly not source discovery."""
    output={}
    for key,v in valid_audit.items():
        rows=v["full_anonymous_concept_partial_rhos"]
        nslots=len(rows)
        if nslots<2:
            output[key]={"status":"ABSTAIN_K_LT_2",
                         "gas_labels_used_in_source_learning":False}
            continue
        mat=np.array([[abs(r[label]["rho"]) if r[label]["rho"] is not None else 0
                       for r in rows] for label in ("other_gas","ethylene")])
        rows_assigned,cols=linear_sum_assignment(-mat)
        chosen=dict(zip(("other_gas","ethylene"),
                        [int(cols[list(rows_assigned).index(i)]) for i in range(2)]))
        unseen=test_audit[key]["full_anonymous_concept_partial_rhos"]
        output[key]={
            "status":"VALID_LABEL_SELECTED_POSTHOC_ASSOCIATION_NOT_AUTONOMOUS_NAMING",
            "axis_alignment_chosen_using_valid_setpoints":chosen,
            "reused_on_test_without_reselection":{
                lab:unseen[axis][lab] for lab,axis in chosen.items()},
            "source_fitting_used_gas_labels":False,
            "chemical_identity_proven":False,
        }
    return output


def _concept_ridge(features_tr,ytr,features_eval):
    model=Ridge(alpha=1.)
    centered=np.asarray(features_tr,float)-.5
    model.fit(centered,ytr)
    regression=model.intercept_+((features_eval-.5)*model.coef_).sum(axis=1)
    if not np.isfinite(regression).all():
        raise AssertionError("Concept-only Ridge output invalid")
    additive_error=float(np.max(np.abs(
        regression-(model.intercept_+((features_eval-.5)*model.coef_).sum(axis=1)))))
    return regression,additive_error,{"intercept":float(model.intercept_),
        "signed_concept_head_coefficients":model.coef_.tolist(),
        "raw_sensor_to_head_bypass":False}


def run(archive:Path,out:Path):
    if out.exists():
        raise FileExistsError("Cannot overwrite V63 pinned experimental evidence")
    runs,schema=build_real_windows(archive)
    x,m,ref,session,half_hour,normalizer=train_only_sensor_scaling(runs)
    if len(x["TRAIN"])<300:
        raise ValueError("Insufficient distinct TRAIN time blocks for V56")
    banks={}
    native={}
    frozen={}
    concepts={}
    for route in ("pca","ica"):
        bank=TrainOnlyInnovationBank(basis=route,seed=20261913,
                         explained_variance_target=.95,max_k=8,
                         block_bootstraps=20)
        bank.fit_unlabeled(x["TRAIN"],m["TRAIN"],
                           train_halfhour_blocks=half_hour)
        model=NativeSelfSupervisedConceptFAN(bank,neural_seed=11,epochs=8)
        model.fit_source_unlabeled(x["TRAIN"],m["TRAIN"])
        banks[route]=bank
        native[route]=model
        frozen[route]=model.frozen_network_sha
        concepts[route]={
            cohort:{
                "native":model._transform(x[cohort],m[cohort])["trajectories"],
                "static":bank.static_fuzzy(x[cohort],m[cohort])
            } for cohort in ("TRAIN","VALID","TEST")}
    association={}
    for route in banks:
        association[route]={}
        for layer in ("static","native"):
            va=semantic_scores(concepts[route]["VALID"][layer],x["VALID"],
                               ref["VALID"],session["VALID"],seed=20261963)
            te=semantic_scores(concepts[route]["TEST"][layer],x["TEST"],
                               ref["TEST"],session["TEST"],seed=20261964)
            association[route][layer]={
                "VALID":va,"TEST":te,
                "posthoc_VALID_alignment_applied_to_TEST":fixed_valid_semantic_alignment(va,te)
            }
    rows=[];heads=[]
    for record_id,record_name in enumerate(DATASET_FILES):
        for target_index,target_name in ((0,SECOND_COLS[record_id]),(1,"ethylene")):
            train_ids=np.flatnonzero(session["TRAIN"]==record_id)
            if len(train_ids)<80:
                raise ValueError("Too few TRAIN windows in a physical recording")
            ytr=ref["TRAIN"][train_ids,target_index]
            if np.std(ytr)<1e-7:
                raise ValueError("No independent dynamic gas setpoint variation")
            target_key=record_name+":"+target_name
            for cohort in ("VALID","TEST"):
                eval_ids=np.flatnonzero(session[cohort]==record_id)
                if len(eval_ids)<20:
                    raise ValueError("Too few source TEST windows under embargo")
                y=ref[cohort][eval_ids,target_index]
                predictions={}
                for route in banks:
                    for layer in ("static","native"):
                        train_feat=concepts[route]["TRAIN"][layer][train_ids].mean(axis=1)
                        eval_feat=concepts[route][cohort][layer][eval_ids].mean(axis=1)
                        prediction,err,h=_concept_ridge(train_feat,ytr,eval_feat)
                        method=route+"_"+layer+"_concept_ridge"
                        predictions[method]=prediction
                        heads.append({
                            "target":target_key,"method":method,
                            "train_labels":len(ytr),
                            "concept_layer_source_frozen_sha256":frozen[route],
                            "max_additive_concept_only_error":err,
                            "risk":h})
                def raw_summaries(cohort_idx,ids):
                    v=x[cohort_idx][ids]
                    return np.concatenate([v.mean(axis=1),
                                           v[:,90:120].mean(axis=1)],axis=1)
                raw_train=raw_summaries("TRAIN",train_ids)
                raw_eval=raw_summaries(cohort,eval_ids)
                raw_ridge=make_pipeline(StandardScaler(),Ridge(alpha=1.))
                raw_ridge.fit(raw_train,ytr)
                predictions["raw_ridge"]=raw_ridge.predict(raw_eval)
                boost=HistGradientBoostingRegressor(
                    max_iter=100,max_leaf_nodes=15,
                    l2_regularization=1.,random_state=0)
                boost.fit(raw_train,ytr)
                predictions["raw_hist_gradient_boosting"]=boost.predict(raw_eval)
                predictions["constant_train_mean"]=np.full(len(eval_ids),ytr.mean())
                for method,p in predictions.items():
                    rows.append({
                        "reference":target_key,
                        "cohort":cohort,
                        "method":method,
                        "n_source_correlated_120s_windows":len(eval_ids),
                        "n_independent_recording_sessions_for_this_target":1,
                        "no_test_tuning":True,
                        **reg_metrics(y,p,float(np.std(ytr)))})
    for route in banks:
        if native[route].frozen_network_sha!=frozen[route]:
            raise AssertionError("Supervised concentration head modified source network")
    evidence={
        "status":"V63_REAL_DYNAMIC_GAS_BLIND_ICA_EXPLORATORY_EXECUTED",
        "official_dataset":"UCI 322 Gas sensor array under dynamic gas mixtures",
        "official_dataset_doi":"10.24432/C5WP4C",
        "original_official_zip_sha256":source_sha256(archive),
        "new_dataset_not_used_in_V60_V61_V62":True,
        "scientific_context_V63_method_selected_after_V62":True,
        "real_independent_continuous_recording_sessions":2,
        "do_not_treat_120s_windows_as_independent_experiments":True,
        "data_schema_and_frozen_two_sided_time_embargo":schema,
        "n_correlated_120s_windows_by_chronological_split":{
            cohort:len(v) for cohort,v in runs.items()},
        "input_sensor_normalizer_fitted_only_train":normalizer,
        "physical_source_setpoint_values_not_source_model_inputs":True,
        "source_banks":{route:banks[route].public_passport() for route in banks},
        "native_frozen_source_sha256":frozen,
        "unlabeled_training_loss_receipts":{
            route:native[route].loss_receipt for route in banks},
        "continuous_ppm_reference_regression_results":rows,
        "VALID_informed_semantic_mapping_never_used_for_model_training":association,
        "any_chemical_source_identity_proven":False,
        "controller_setpoints_are_not_independent_chemical_analyzer_trajectories":True,
        "original_accepted_IOI_PIERE_manuscripts_untouched":True,
        "preregistered_protocol":"docs/medical/CONCEPTFAN_V63_FROZEN_DYNAMIC_GAS_ICA_PROTOCOL_20261113.md"
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"STATUS.json").write_text(json.dumps(
        evidence,ensure_ascii=False,allow_nan=False,indent=2)+"\n",
        encoding="utf8")
    with (out/"METRICS.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({
        "event":"V63_DONE",
        "source_ranks":{k:v.k for k,v in banks.items()},
        "n_windows":{k:len(v) for k,v in runs.items()},
        "test":[z for z in rows if z["cohort"]=="TEST"],
        "max_abs_conditional_associations":{
            k:{name:v["native"]["TEST"][name]["max_abs_conditional_by_target"]
               for name in ("CO_recording","methane_recording")}
            for k,v in association.items()}
    },ensure_ascii=False,allow_nan=False),flush=True)
    return evidence


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--uci-zip",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    run(a.uci_zip,a.out_dir)


if __name__=="__main__":
    main()
