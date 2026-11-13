#!/usr/bin/env python3
"""V59: PhysioNet 2012 TRAIN-only availability of candidate independent anchors.

This is a measurements/mask audit, NOT a clinical outcome model or proof that
two named sensors are separately calibrated or have independent fault domains.
No patient record IDs, time-series rows, or patient-level scores are published.

Uses the original project's stratified 2400/600/400/600 split only so even
measurement-availability feasibility does not inspect sealed test timelines.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

N_PATIENTS=4000
TRAIN=2400
HOURS=36
BIN_SIZE=4
FOLD_SEED=20261513
SPLIT_SEED=20260717
MIN_PATIENTS_WITH_SIX_PAIRS=100
MIN_CELLS_PER_FOLD=250
CHANNELS=("MAP","NIMAP","SysABP","NISysABP","DiasABP","NIDiasABP")
PAIRS=(("MAP","NIMAP"),("SysABP","NISysABP"),
       ("DiasABP","NIDiasABP"))
INDEX={name:i for i,name in enumerate(CHANNELS)}


def sha256_file(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_original_split(manifest:pd.DataFrame,*,seed:int=SPLIT_SEED):
    """Exact independent reimplementation of original create_split.

    The original package __init__ eagerly imports PyTorch; this harmless
    aggregate availability audit must not need to load the neural stack.
    Splitting is copied verbatim from conceptfan_realdata.data.create_split.
    """
    frame=manifest.rename(columns={"record_id":"RecordID",
                                   "icu_type":"ICUType"}).copy()
    frame["stratum"]=frame["In-hospital_death"].astype(str)+"_"+frame["ICUType"].astype(str)
    ids=frame.index.to_numpy()
    tvc,test=train_test_split(
        ids,test_size=600,random_state=seed,
        stratify=frame.loc[ids,"stratum"])
    tv,cal=train_test_split(
        tvc,test_size=400,random_state=seed+1,
        stratify=frame.loc[tvc,"stratum"])
    train,val=train_test_split(
        tv,test_size=600,random_state=seed+2,
        stratify=frame.loc[tv,"stratum"])
    mapping={int(index):name for name,group in
             (("train",train),("validation",val),
              ("calibration",cal),("test",test))
             for index in group}
    frame["split"]=[mapping[int(idx)] for idx in frame.index]
    expected={"train":2400,"validation":600,"calibration":400,"test":600}
    counts=frame["split"].value_counts().to_dict()
    if counts!=expected:
        raise AssertionError(f"Original frozen split mismatch: {counts}")
    return frame[["RecordID","split"]],counts


def parse_rows(payload:bytes):
    text=payload.decode("utf-8-sig")
    stream=csv.reader(io.StringIO(text))
    header=next(stream,None)
    if header!=["Time","Parameter","Value"]:
        raise ValueError("Invalid official PhysioNet per-record header")
    for row in stream:
        if len(row)!=3:
            raise ValueError("Malformed row in public patient record")
        yield row


def _number(value):
    try:
        v=float(value)
    except ValueError:
        return None
    return v if np.isfinite(v) and v!=-1 else None


def static_info(payload:bytes,expected_record_id:int):
    record_id=None
    icu_type=-1
    icu_seen=False
    # Stop at the two admission descriptors. In particular DO NOT iterate
    # through held-out episodes' time-series readings merely to make a split.
    for _,parameter,value in parse_rows(payload):
        if parameter=="RecordID":
            z=_number(value)
            if z is not None:
                record_id=int(z)
        elif parameter=="ICUType":
            icu_seen=True
            z=_number(value)
            if z is not None:
                icu_type=int(z)
        if record_id is not None and icu_seen:
            break
    if record_id!=expected_record_id or not icu_seen:
        raise ValueError("Original static RecordID does not match archive filename")
    return icu_type


def observed_hours(payload:bytes,hours:int=HOURS):
    slots={}
    for timestamp,parameter,value in parse_rows(payload):
        if parameter not in INDEX:
            continue
        hour,minute=(int(v) for v in timestamp.split(":"))
        if hour<0 or hour>=hours or minute<0 or minute>=60:
            continue
        number=_number(value)
        if number is not None:
            slots.setdefault((hour,INDEX[parameter]),[]).append(number)
    values=np.full((hours,len(CHANNELS)),np.nan,dtype=np.float64)
    mask=np.zeros((hours,len(CHANNELS)),dtype=bool)
    for (hour,j),list_values in slots.items():
        values[hour,j]=float(np.median(list_values))
        mask[hour,j]=True
    return values,mask


def aggregate_bin(values,mask,bin_size:int):
    if values.ndim!=3 or values.shape!=mask.shape or values.shape[1]%bin_size:
        raise ValueError("Expected aligned observed hourly values with divisible window")
    n,t,d=values.shape
    blocks=np.where(mask,values,np.nan).reshape(n,t//bin_size,bin_size,d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",category=RuntimeWarning)
        summarized=np.nanmedian(blocks,axis=2)
    availability=mask.reshape(n,t//bin_size,bin_size,d).any(axis=2)
    return np.where(availability,summarized,np.nan),availability


def audit_one_pair(values,mask,first:str,second:str,
                   *,seed:int=FOLD_SEED):
    if values.shape!=mask.shape or values.ndim!=3:
        raise ValueError("Paired observations require aligned [patients,time,channels]")
    if first not in INDEX or second not in INDEX:
        raise ValueError("Unregistered measurement family")
    i,j=INDEX[first],INDEX[second]
    observed_i=mask[:,:,i]
    observed_j=mask[:,:,j]
    both=observed_i & observed_j
    pairs_per_person=both.sum(axis=1)
    patient_count=len(both)
    rng=np.random.default_rng(seed)
    folds=np.array_split(rng.permutation(patient_count),3)
    fold_counts=tuple(int(both[ids].sum()) for ids in folds)
    def pearson(a,b):
        if len(a)>=3 and np.std(a)>1e-8 and np.std(b)>1e-8:
            return float(np.corrcoef(a,b)[0,1])
        return None
    a=values[:,:,i][both]
    b=values[:,:,j][both]
    r=pearson(a,b)
    fold_corr=[]
    for ids in folds:
        seen=both[ids]
        fold_corr.append(pearson(values[ids,:,i][seen],values[ids,:,j][seen]))
    n_six=int((pairs_per_person>=6).sum())
    passed=(n_six>=MIN_PATIENTS_WITH_SIX_PAIRS
            and min(fold_counts)>=MIN_CELLS_PER_FOLD)
    # Diagnostic added after V59 overlap run, but gate is the exact original
    # PREEXISTING V57 train-fold abs(r)>=.65 condition. Do not tune it.
    all_folds=all(v is not None for v in fold_corr)
    corr_agreement=(all_folds and
        min(abs(v) for v in fold_corr)>=.65 and
        abs(sum(np.sign(v) for v in fold_corr))==3)
    v57_alignment_eligible=bool(passed and corr_agreement)
    return {
        "pair":f"{first}/{second}",
        "n_train_patients":int(patient_count),
        "both_sensors_present_somewhere_in_history":int(
            (observed_i.any(axis=1)&observed_j.any(axis=1)).sum()),
        "aligned_joint_time_bins":int(both.sum()),
        "n_patients_at_least_one_aligned_bin":int((pairs_per_person>=1).sum()),
        "n_patients_at_least_six_aligned_bins":n_six,
        "percentage_patients_at_least_six_aligned_bins":
            float(n_six/patient_count*100),
        "aligned_pair_cells_in_three_disjoint_train_folds":list(fold_counts),
        "descriptive_observed_only_pearson":r,
        "three_disjoint_train_fold_pearson":fold_corr,
        "passes_original_v57_signed_corr_min_abs_065":bool(corr_agreement),
        "meets_v57_synthetic_thresholds_for_pair_coverage":bool(passed),
        "v57_source_alignment_possible_on_measured_association":v57_alignment_eligible,
        "measurement_device_fault_independence_proven":False,
        "clinical_concept_semantics_proven":False,
    }


def run(archive_path:Path,outcomes_path:Path,output:Path):
    if output.exists():
        raise FileExistsError("Never overwrite pinned V59 aggregate evidence")
    if not archive_path.is_file() or not outcomes_path.is_file():
        raise FileNotFoundError("Official PhysioNet Set A and Outcomes-a required")
    outcomes=pd.read_csv(outcomes_path)
    if (len(outcomes)!=N_PATIENTS
        or "RecordID" not in outcomes
        or "In-hospital_death" not in outcomes
        or outcomes["RecordID"].duplicated().any()):
        raise ValueError("The official Outcomes-a file must contain 4000 unique records")
    outcomes_by_id=outcomes.set_index("RecordID")["In-hospital_death"]
    with zipfile.ZipFile(archive_path) as z:
        # Do not pre-extract/checksum every sealed patient timeline; reading
        # the ZIP index does not consume any time-series. TRAIN records alone
        # are parsed as observational trajectories below.
        names={int(Path(name).stem):name
               for name in z.namelist() if name.lower().endswith(".txt")}
        if len(names)!=N_PATIENTS or set(names)!=set(outcomes_by_id.index.astype(int)):
            raise ValueError("Official Set A record identity mismatch")
        metas=[]
        for record_id,name in sorted(names.items()):
            icu=static_info(z.read(name),record_id)
            metas.append({
                "record_id":record_id,
                "icu_type":icu,
                "In-hospital_death":int(outcomes_by_id.loc[record_id])
            })
        split,split_summary=frozen_original_split(pd.DataFrame(metas),seed=SPLIT_SEED)
        train_ids=sorted(split.loc[split["split"].eq("train"),"RecordID"].astype(int))
        if len(train_ids)!=TRAIN:
            raise AssertionError("Frozen original TRAIN split changed")
        # Only TRAIN patients' time-series measurements are extracted.
        x=np.full((len(train_ids),HOURS,len(CHANNELS)),np.nan,dtype=np.float64)
        m=np.zeros_like(x,dtype=bool)
        for n,record_id in enumerate(train_ids):
            current,mask=observed_hours(z.read(names[record_id]))
            x[n]=current
            m[n]=mask
    if not np.isfinite(x[m]).all():
        raise AssertionError("Nonfinite observed source measurement")
    receipt={
        "dataset":"Official PhysioNet/Computing in Cardiology Challenge 2012 Set A",
        "official_url":"https://physionet.org/content/challenge-2012/1.0.0/",
        "set_a_zip_sha256":sha256_file(archive_path),
        "outcomes_a_sha256":sha256_file(outcomes_path),
        "frozen_split_seed":SPLIT_SEED,
        "split_sizes":{str(k):int(v) for k,v in split_summary.items()},
        "train_count":len(train_ids),
        "train_id_manifest_sha256":hashlib.sha256(
            ",".join(map(str,train_ids)).encode("ascii")).hexdigest(),
        "only_original_train_time_series_extracted":True,
        "outcomes_used_only_for_original_stratified_split_membership":True,
        "patient_level_raw_data_exposed_in_artifact":False,
        "input_hours":HOURS,
        "same_hour_measurement_alignment_no_filling":True,
        "candidate_pairs_predeclared":[list(x) for x in PAIRS],
    }
    reports=[]
    for bin_size,values,mask in (
        (1,x,m),(BIN_SIZE,*aggregate_bin(x,m,BIN_SIZE))
    ):
        for first,second in PAIRS:
            z=audit_one_pair(values,mask,first,second)
            z["hour_per_time_bin"]=bin_size
            z["max_available_time_bins_per_patient"]=HOURS//bin_size
            reports.append(z)
    passed=[z for z in reports if z["meets_v57_synthetic_thresholds_for_pair_coverage"]]
    linked=[z for z in reports if z["v57_source_alignment_possible_on_measured_association"]]
    result={
        "status":("MEASUREMENT_OVERLAP_FEASIBLE_UNVERIFIED_PROVENANCE"
                 if passed else "ABSTAIN_CLINICAL_ANCHOR_UNOBSERVABLE"),
        "post_support_diagnostic_v57_correlation_gate":(
            "PRESPECIFIED_IN_V57_BUT_ADDED_TO_V59_AFTER_INITIAL_SUPPORT_INSPECTION"),
        "v57_real_observational_source_link_status":(
            "ALIGNMENT_POSSIBLE_BUT_INDEPENDENT_PROVENANCE_UNVERIFIED"
            if linked else
            "ABSTAIN_NO_PAIR_MEETS_UNCHANGED_V57_CORRELATION_AND_SUPPORT_GATES"),
        "n_pairs_resolutions_meeting_v57_source_link":len(linked),
        "frozen_preinspection_protocol":"docs/medical/CONCEPTFAN_V59_FROZEN_CLINICAL_REFERENCE_FEASIBILITY_20261113.md",
        "receipt":receipt,
        "reference_overlap_tests":reports,
        "n_feasible_sensor_resolutions":len(passed),
        "reference_independent_calibration_documented":False,
        "instrument_fault_event_ground_truth_available":False,
        "physical_concept_identity_validated":False,
        "clinical_predictive_model_trained_or_tested":False,
        "sealed_original_validation_calibration_and_test_timelines_read":False,
        "published_artifacts_are_aggregate_only":True,
    }
    output.mkdir(parents=True,exist_ok=False)
    (output/"STATUS.json").write_text(json.dumps(
        result,indent=2,allow_nan=False,ensure_ascii=False)+"\n",encoding="utf8")
    print(json.dumps({
        "event":"V59_REAL_OBSERVABILITY_DONE","status":result["status"],
        "n_train":len(train_ids),"counts":[
            {"pair":r["pair"],"bin":r["hour_per_time_bin"],
             "co_observed_episodes_ge6":r["n_patients_at_least_six_aligned_bins"],
             "pass":r["meets_v57_synthetic_thresholds_for_pair_coverage"]}
            for r in reports]},
        ensure_ascii=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--set-a-zip",type=Path,required=True)
    p.add_argument("--outcomes",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    run(args.set_a_zip,args.outcomes,args.out_dir)

if __name__=="__main__":
    main()
