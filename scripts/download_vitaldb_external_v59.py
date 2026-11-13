#!/usr/bin/env python3
"""V59: download a PREDECLARED independent VitalDB external-site TEST cohort.

Public VitalDB API: https://api.vitaldb.net/cases, /trks, /{tid}.
Seoul National University Hospital, CC BY 4.0. Only numeric tracks are used.
- Solar8000/PLETH_HR -> source PULSE
- Solar8000/RR_CO2 -> source RESP (capnography, true site modality change)
- Solar8000/PLETH_SPO2 -> source SPO2
- Solar8000/HR -> same-time outcome/reference, NEVER an input feature

Enrollment by public inventory, independent subject IDs and data availability
ONLY; no label values, prediction outcomes or train-model scores.
Raw deidentified clinical vital signs and subject IDs remain CI-local.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import time
from urllib.error import HTTPError,URLError
from urllib.request import Request,urlopen

import numpy as np

BASE="https://api.vitaldb.net"
SUBJECT_SELECTION_SEED=20265913
CANDIDATE_SUBJECT_LIMIT=80
MAX_ACCEPTED_CASES=40
MIN_ACCEPTED_CASES=30
WINDOW_LENGTH_SEC=480
REQUIRED_TRACKS=(
    "Solar8000/PLETH_HR","Solar8000/RR_CO2",
    "Solar8000/PLETH_SPO2","Solar8000/HR"
)
CHANNEL_BOUNDS={
    "Solar8000/PLETH_HR":(25.,250.),
    "Solar8000/RR_CO2":(4.,80.),
    "Solar8000/PLETH_SPO2":(45.,100.),
    "Solar8000/HR":(25.,250.),
}


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def _fetch(url,tries=5):
    error=None
    for i in range(tries):
        try:
            r=Request(url,headers={"User-Agent":"ConceptFAN-V59-external-evaluation/1.0",
                                   "Accept-Encoding":"gzip"})
            with urlopen(r,timeout=55) as response:
                raw=response.read()
                if raw[:2]==b"\x1f\x8b":
                    raw=gzip.decompress(raw)
                return raw
        except (HTTPError,URLError,TimeoutError,OSError) as exc:
            error=exc
            if i+1<tries:time.sleep(min(2**i,8))
    raise RuntimeError(f"VitalDB public API unavailable: {url}: {error}")


def _rows(blob):
    result=list(csv.DictReader(io.StringIO(blob.decode("utf-8-sig"))))
    if not result:raise ValueError("Public VitalDB CSV was empty")
    return result


def _inventory():
    cases_blob=_fetch(BASE+"/cases")
    tracks_blob=_fetch(BASE+"/trks")
    cases={}
    for row in _rows(cases_blob):
        case=str(row.get("caseid","")).strip()
        subject=str(row.get("subjectid","")).strip()
        if case and subject and subject.lower()!="nan":
            if case in cases and cases[case]!=subject:
                raise ValueError("Inconsistent subjectid for same VitalDB case")
            cases[case]=subject
    if len(cases)<3000:raise RuntimeError("Clinical inventory incomplete")
    bycase=defaultdict(dict)
    for row in _rows(tracks_blob):
        case=str(row.get("caseid","")).strip()
        tname=row.get("tname","").strip()
        tid=row.get("tid","").strip()
        if case in cases and tname in REQUIRED_TRACKS and tid:
            if tname in bycase[case] and bycase[case][tname]!=tid:
                # Strict avoid selecting an arbitrarily duplicated track.
                bycase[case][tname]=None
            else:
                bycase[case][tname]=tid
    complete={case:tracks for case,tracks in bycase.items()
              if all(tracks.get(name) for name in REQUIRED_TRACKS)}
    groups=defaultdict(list)
    for case in complete:
        groups[cases[case]].append(case)
    if len(groups)<CANDIDATE_SUBJECT_LIMIT:
        raise RuntimeError("Insufficient VitalDB external patients with four signals")
    # One case per person, picked by public case ordinal, BEFORE downloading
    # a single vital sign value or reading target HR.
    subject_choice={subject:sorted(ids,key=lambda x:int(x))[0]
                    for subject,ids in groups.items()}
    subjects=sorted(subject_choice)
    rng=np.random.default_rng(SUBJECT_SELECTION_SEED)
    perm=rng.permutation(len(subjects))
    picked=[subject_choice[subjects[i]] for i in perm[:CANDIDATE_SUBJECT_LIMIT]]
    return cases,complete,picked,sha256(cases_blob),sha256(tracks_blob),len(groups)


def _numeric_track(raw,tname):
    rows=_rows(raw)
    # API numeric samples have Time, Value, missing points omitted.
    t=[];v=[]
    for row in rows:
        normalized={k.strip().lower():x for k,x in row.items()}
        if not("time" in normalized and "value" in normalized):
            raise ValueError("Non-numeric track format (Time,Value expected)")
        try:
            tt=float(normalized["time"])
            val=float(normalized["value"])
        except (TypeError,ValueError):
            continue
        lo,hi=CHANNEL_BOUNDS[tname]
        if np.isfinite(tt) and np.isfinite(val) and lo<=val<=hi:
            t.append(tt);v.append(val)
    if len(t)<300:
        raise ValueError(f"Insufficient valid {tname} numeric track observations")
    t=np.asarray(t,float);v=np.asarray(v,float)
    if np.any(np.diff(t)<0):
        raise ValueError("Nonmonotonic source timestamps")
    # Same-timestamp repeats are possible; choose last observed sample.
    keep=np.r_[np.diff(t)>0,True]
    return t[keep],v[keep]


def _align(t,v,grid):
    ix=np.searchsorted(t,grid,side="right")-1
    valid=ix>=0
    latest=t[np.maximum(ix,0)]
    valid &= (grid-latest>=0)&(grid-latest<=3.)
    result=np.full(len(grid),np.nan,dtype=float)
    result[valid]=v[ix[valid]]
    return result


def _earliest_complete_window(tracks):
    """Eligibility from presence, not values, Y, model or predicted errors."""
    starts=np.array([tt[0] for tt,_ in tracks.values()])
    ends=np.array([tt[-1] for tt,_ in tracks.values()])
    early=int(np.ceil(starts.max()))
    late=int(np.floor(ends.min()))-WINDOW_LENGTH_SEC+1
    if early>late:return None
    # Inspect first five minutes of possible overlap and accept earliest
    # eligible 480s interval; no "best" target/score selection.
    for origin in range(early,min(late+1,early+301),30):
        grid=origin+np.arange(WINDOW_LENGTH_SEC)
        aligned=np.column_stack([
            _align(*tracks[tname],grid) for tname in REQUIRED_TRACKS])
        if np.all(np.isfinite(aligned).mean(axis=0)>=.70):
            return origin,aligned
    return None


def download_vitaldb(out:Path):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    patients,track_index,candidate_cases,case_sha,tracks_sha,n_subjects=_inventory()
    results={}
    def one(case):
        resolved={};digests={}
        for name in REQUIRED_TRACKS:
            raw=_fetch(BASE+"/"+track_index[case][name])
            digests[name]=sha256(raw)
            resolved[name]=_numeric_track(raw,name)
        choice=_earliest_complete_window(resolved)
        if choice is None:
            raise ValueError("No 480-second window with predeclared 70% coverage")
        origin,arr=choice
        return case,origin,arr,digests
    # Bound concurrency and memory, even though all enrollments were already
    # frozen purely on public track metadata.
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures={executor.submit(one,case):case for case in candidate_cases}
        for future in as_completed(futures):
            case=futures[future]
            try:
                c,origin,values,hashes=future.result()
            except Exception as exc:
                results[case]={"accepted":False,"reason":type(exc).__name__}
                continue
            path=out/f"vitaldb_case_{case}.npz"
            # Clinical values and subject IDs stay runner-local; DO NOT add
            # these NPZ files to GitHub artifact paths.
            np.savez_compressed(path,signals=values.astype(np.float32),
                                origin_sec=origin)
            results[case]={
                "accepted":True,"selected_origin_sec":origin,
                "valid_fraction":[float(x) for x in np.isfinite(values).mean(axis=0)],
                "track_sha256":hashes,
            }
    accepted=[case for case in candidate_cases if results.get(case,{}).get("accepted")]
    selected=accepted[:MAX_ACCEPTED_CASES]
    if len(selected)<MIN_ACCEPTED_CASES:
        raise RuntimeError(f"External VitalDB cohort below 30 genuine cases: {len(selected)}")
    # Delete the numeric files for nonselected cases (they were never used
    # for model fitting, enrollment, tuning or evaluation).
    for case in set(accepted)-set(selected):
        (out/f"vitaldb_case_{case}.npz").unlink(missing_ok=True)
    if len({patients[c] for c in selected})!=len(selected):
        raise AssertionError("External cases are not from unique real patients")
    receipt={
        "source":"VitalDB open dataset, Seoul National University Hospital",
        "source_API":BASE,
        "license":"CC-BY-4.0 with dataset attribution",
        "source_cases_csv_sha256":case_sha,
        "source_tracks_csv_sha256":tracks_sha,
        "source_candidates_with_4_tracks_before_values":n_subjects,
        "candidate_selection_seed":SUBJECT_SELECTION_SEED,
        "predeclared_case_candidates":CANDIDATE_SUBJECT_LIMIT,
        "accepted_before_40_case_cap":len(accepted),
        "accepted_external_distinct_patients":len(selected),
        "selected_public_case_ordinals":list(map(int,selected)),
        "availability_rejection_count":sum(not item["accepted"] for item in results.values()),
        "min_coverage_predeclared":.70,
        "window_seconds":WINDOW_LENGTH_SEC,
        "site_different_from_BIDMC":True,
        "clinical_HR_not_an_input":True,
        "raw_subject_ids_exported":False,
        "raw_patient_timeseries_uploaded":False,
        # Public source CASE ids are ordinals, not hospital patient identifiers.
        "selected_metadata":{str(c):results[c] for c in selected},
    }
    (out/"vitaldb_download_receipt.json").write_text(
        json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "status":"READY_EXTERNAL_SITE_NOT_USED_IN_TRAIN",
        "cases":len(selected),"candidates":len(candidate_cases),
        "other_eligible_unique_subjects":n_subjects},indent=2))
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    download_vitaldb(args.out)
