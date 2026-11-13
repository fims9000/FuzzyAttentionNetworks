#!/usr/bin/env python3
"""V62 download only original UCI HAR participant-disjoint raw inertial windows.

Original source: UCI HAR Using Smartphones, doi:10.24432/C54S4K.
Official 30 subjects: 21 train (7352 windows), 9 heldout test (2947).
Full raw 128x6 accelerometer+gyroscope arrays remain runner-local;
GitHub artifacts contain only aggregate provenance/hash receipts.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import time
from urllib.error import URLError,HTTPError
from urllib.request import Request,urlopen
from zipfile import ZipFile

import numpy as np

URLS=(
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
)
N={"train":7352,"test":2947}
CHANNELS=("body_acc_x","body_acc_y","body_acc_z",
          "body_gyro_x","body_gyro_y","body_gyro_z")


def digest(v):
    return sha256(v).hexdigest()


def _fetch():
    failures=[]
    for url in URLS:
        for attempt in range(3):
            try:
                request=Request(url,headers={
                    "User-Agent":"ConceptFAN-V62-public-uci-research/1.0"})
                with urlopen(request,timeout=90) as result:
                    raw=result.read()
                if len(raw)<10_000_000 or raw[:2]!=b"PK":
                    raise ValueError("UCI download was not official dataset zip")
                return url,raw
            except (HTTPError,URLError,TimeoutError,OSError,ValueError) as exc:
                failures.append(f"{url}: {type(exc).__name__}")
                if attempt<2:time.sleep(2**attempt)
    raise RuntimeError("Cannot retrieve original UCI HAR official archive: "+
                       ";".join(failures))


def _selected_zip(raw):
    outer=ZipFile(io.BytesIO(raw))
    contents=[x for x in outer.namelist() if not x.endswith("/")]
    if any(x.endswith("train/y_train.txt") for x in contents):
        return outer,raw
    nested=[x for x in contents
            if x.lower().endswith("uci har dataset.zip")]
    if len(nested)!=1:
        raise RuntimeError("UCI HAR archive nesting/layout not recognized")
    inside=outer.read(nested[0])
    inner=ZipFile(io.BytesIO(inside))
    return inner,inside


def _entry(z,path):
    possible=[p for p in z.namelist() if p.endswith("/"+path) or p==path]
    if len(possible)!=1:
        raise RuntimeError("Missing or ambiguous official UCI HAR source "+path)
    return z.read(possible[0])


def read(split,z):
    ids=_entry(z,f"{split}/subject_{split}.txt")
    targets=_entry(z,f"{split}/y_{split}.txt")
    people=np.loadtxt(io.BytesIO(ids),dtype=np.int32).reshape(-1)
    y=np.loadtxt(io.BytesIO(targets),dtype=np.int32).reshape(-1)
    if (len(y)!=N[split] or len(people)!=N[split] or
        np.any((y<1)|(y>6))):
        raise RuntimeError("Official HAR sample counts or label IDs changed")
    stack=[];checksums={
        "subject_SHA256":digest(ids),"activity_labels_SHA256":digest(targets)}
    for name in CHANNELS:
        key=f"{split}/Inertial Signals/{name}_{split}.txt"
        raw=_entry(z,key)
        data=np.loadtxt(io.BytesIO(raw),dtype=np.float32)
        if data.shape!=(N[split],128) or not np.isfinite(data).all():
            raise RuntimeError("Official UCI HAR raw 128-point inertial windows corrupted")
        stack.append(data)
        checksums[name+"_SHA256"]=digest(raw)
    signal=np.stack(stack,axis=-1)
    if signal.shape!=(N[split],128,6):
        raise AssertionError("HAR original paired acceleration/gyroscope unavailable")
    return signal,people,y,checksums


def download(out:Path):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    url,archive=_fetch()
    zipfile,inner=_selected_zip(archive)
    if zipfile.testzip() is not None:
        raise RuntimeError("Original UCI HAR ZIP CRC error")
    source={}
    groups={}
    for split in ("train","test"):
        x,subject,y,hashes=read(split,zipfile)
        groups[split]=set(subject.tolist())
        np.savez_compressed(out/f"private_{split}.npz",
                            axes=x,subject=subject,y=y)
        source[split]={
            "original_windows":len(y),
            "participant_count":len(groups[split]),
            "classes_present":sorted(set(y.tolist())),
            "source_file_sha256":hashes,
        }
    if (len(groups["train"])!=21 or len(groups["test"])!=9 or
        groups["train"]&groups["test"]):
        raise RuntimeError("Original official HAR person split is not patient-disjoint")
    receipt={
        "dataset":"UCI HAR Using Smartphones, DOI 10.24432/C54S4K",
        "official_source_url":url,
        "official_outer_zip_sha256":digest(archive),
        "source_interior_zip_sha256":digest(inner),
        "license":"CC BY 4.0, attribution required",
        "original_raw_time_samples_per_window":128,
        "overlap_per_original_source":".50, original 2.56s windows 50Hz",
        "two_physical_sensor_types":["BODY_ACCELEROMETER","BODY_GYROSCOPE"],
        "source_names":list(CHANNELS),
        "official_participant_test_disjoint":True,
        "original_official_participant_count":30,
        "split_provenance":source,
        "uses_precomputed_561_features":False,
        "raw_signals_in_uploaded_public_artifact":False,
        "per_participant_activity_risk_in_public_artifact":False,
    }
    (out/"uci_har_official_source_receipt.json").write_text(
        json.dumps(receipt,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
        "status":"AUTHENTIC_OFFICIAL_UCI_HAR_SOURCE",
        "official_train_people":len(groups["train"]),
        "unseen_official_test_people":len(groups["test"]),
        "windows_total":sum(v["original_windows"] for v in source.values()),
        "sha256":receipt["official_outer_zip_sha256"]},indent=2))
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    download(args.out)
