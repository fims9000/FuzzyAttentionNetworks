"""V58: strict real BIDMC numerics loader with patient-level isolation.

Only PPG-derived monitor PULSE, impedance-derived monitor RESP and SpO2
enter FAN. ECG-derived monitor HR is withheld from all X and unsupervised
candidate formation; it is used as a simultaneous threshold label and
post-inference comparison, NOT an independent second physical assay.
No patient IDs, vital signs or windows are written to public artifacts.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import numpy as np

EXPECTED_RECORDS=53
TIME_STEPS=16
WINDOW_STRIDE=8
GROUP_SEED=20265813
THRESHOLD_BPM=100.
N_FEATURES=10
FEATURE_NAMES=(
    "PPG_MONITOR_PULSE","IMPEDANCE_MONITOR_RESP","PPG_MONITOR_SPO2",
    "PULSE_MINUS_RESP",
    "PULSE_FIRST_DIFFERENCE","RESP_FIRST_DIFFERENCE",
    "SPO2_FIRST_DIFFERENCE",
    "PULSE_TRAILING_3S_STD","RESP_TRAILING_3S_STD",
    "SPO2_TRAILING_3S_STD",
)


@dataclass(frozen=True)
class NormalizationV58:
    mean:tuple[float,...]
    scale:tuple[float,...]
    train_count:int

    def manifest(self):
        return {"features":FEATURE_NAMES,
                "mean":self.mean,"scale":self.scale,
                "train_observed_pixels":self.train_count,
                "fit_only_on_train":True}


def _read_patient_group(text):
    m=re.search(r"(?im)^MIMIC II matched wdb ID:\s*(\S+)\s*$",text)
    if m is None:
        raise ValueError("Cannot determine patient identity, refusing record-level leakage")
    return m.group(1)


def _numbers(path:Path):
    with path.open("r",encoding="utf8-sig",newline="") as file:
        reader=csv.DictReader(file)
        if not {"Time [s]","HR","PULSE","RESP","SpO2"}.issubset(
            {x.strip() for x in (reader.fieldnames or [])}):
            raise ValueError(f"Expected BIDMC numeric channel headers missing in {path.name}")
        t=[];rows=[]
        for row in reader:
            col={k.strip():v for k,v in row.items()}
            def parse(name):
                try:return float(col[name])
                except (ValueError,TypeError):return float("nan")
            t.append(parse("Time [s]"))
            rows.append([parse(c) for c in ("PULSE","RESP","SpO2","HR")])
    time=np.asarray(t,dtype=np.float64)
    data=np.asarray(rows,dtype=np.float64)
    if (len(time)<450 or not np.isfinite(time).all() or
        (np.diff(time)<=0).any() or
        np.any(np.isinf(data))):
        raise ValueError(f"Bad BIDMC numeric time or infinite measurements in {path.name}")
    if not np.allclose(np.diff(time),1.,atol=1e-4):
        raise ValueError(f"1Hz monitor rate assumed, but not present: {path.name}")
    return data


def causal_monitor_features(raw_three):
    """10 measured/derived channels; no ECG, no deliberately delayed copies.

    All within-second derivatives and trailing volatility use present/past
    samples only; unavailable inputs have zero placeholder + false mask.
    """
    raw=np.asarray(raw_three,dtype=np.float64)
    if raw.ndim!=2 or raw.shape[1]!=3 or np.isinf(raw).any():
        raise ValueError("Expected raw numeric PULSE/RESP/SpO2 [seconds,3]")
    length=len(raw)
    mask=np.isfinite(raw)
    out=np.zeros((length,N_FEATURES),np.float64)
    valid=np.zeros((length,N_FEATURES),bool)
    out[:,:3]=np.where(mask,raw,0.)
    valid[:,:3]=mask
    both=mask[:,0]&mask[:,1]
    out[:,3]=np.where(both,out[:,0]-out[:,1],0.)
    valid[:,3]=both
    for j in range(3):
        good=mask[1:,j]&mask[:-1,j]
        out[1:,4+j]=np.where(good,out[1:,j]-out[:-1,j],0.)
        valid[1:,4+j]=good
        # Explicit trailing window; no centered or future smoothing.
        for t in range(2,length):
            if mask[t-2:t+1,j].all():
                out[t,7+j]=raw[t-2:t+1,j].std(ddof=0)
                valid[t,7+j]=True
    if not np.isfinite(out).all():
        raise AssertionError("Non-finite masked feature placeholder")
    return out,valid


def group_allocations(group_to_records):
    """Allocate whole MIMIC-II patient IDs, never individual overlapping windows."""
    patients=sorted(group_to_records)
    n=len(patients)
    if n<32:raise RuntimeError("Too few independently identifiable patients")
    rng=np.random.default_rng(GROUP_SEED)
    perm=rng.permutation(n)
    ntrain=int(np.floor(.60*n))
    nvalid=max(6,int(np.round(.14*n)))
    ntest=n-ntrain-2*nvalid
    if ntest<6 or ntrain<18:
        raise RuntimeError("Insufficient independent patient groups for frozen split")
    ordered=[patients[i] for i in perm]
    split={
        "train":set(ordered[:ntrain]),
        "valid1":set(ordered[ntrain:ntrain+nvalid]),
        "valid2":set(ordered[ntrain+nvalid:ntrain+2*nvalid]),
        "test":set(ordered[ntrain+2*nvalid:]),
    }
    groups=list(split.values())
    if any(a&b for i,a in enumerate(groups) for b in groups[i+1:]):
        raise AssertionError("Same real patient leaked across splits")
    return split


def fit_train_normalization(values,mask):
    x=np.asarray(values,dtype=np.float64)
    m=np.asarray(mask,bool)
    if x.shape!=m.shape or x.shape[2]!=N_FEATURES:
        raise ValueError("Incompatible input numeric train mask")
    counts=m.sum(axis=(0,1))
    if np.any(counts<100):
        raise RuntimeError("Insufficient observed TRAIN cells for feature")
    mean=np.asarray([(x[:,:,j][m[:,:,j]]).mean()
                     for j in range(N_FEATURES)])
    std=np.asarray([max((x[:,:,j][m[:,:,j]]).std(),.1)
                    for j in range(N_FEATURES)])
    return NormalizationV58(tuple(map(float,mean)),
                            tuple(map(float,std)),int(m.sum()))


def apply_train_normalization(x,m,normalization):
    means=np.asarray(normalization.mean,dtype=np.float64)
    scales=np.asarray(normalization.scale,dtype=np.float64)
    if (x.shape!=m.shape or x.shape[-1]!=len(means) or
        np.any(scales<=0)):
        raise ValueError("Expected common frozen train channel normalization")
    scaled=(np.where(m,x,means)-means)/scales
    # No imputation as observed zero. Zero placeholder + observed mask.
    return np.clip(np.where(m,scaled,0.),-8.,8.).astype(np.float32)


def load_bidmc_realworld_v58(root:Path):
    root=Path(root)
    sections=[]
    patient_to_records=defaultdict(list)
    for rec in range(1,EXPECTED_RECORDS+1):
        prefix=f"bidmc_{rec:02d}"
        path=root/f"{prefix}_Numerics.csv"
        fixed=root/f"{prefix}_Fix.txt"
        if not path.is_file() or not fixed.is_file():
            raise FileNotFoundError(f"Missing original BIDMC {prefix}")
        patient=_read_patient_group(fixed.read_text(encoding="utf8-sig"))
        data=_numbers(path)
        three=data[:,:3]
        hr=data[:,3]
        features,available=causal_monitor_features(three)
        patient_to_records[patient].append(rec)
        for start in range(0,len(hr)-TIME_STEPS+1,WINDOW_STRIDE):
            end=start+TIME_STEPS
            label_hr=hr[end-4:end]
            good=np.isfinite(label_hr)&(label_hr>=25)&(label_hr<=250)
            if int(good.sum())<3:continue
            bpm=float(label_hr[good].mean())
            # Keep direct PPG PULSE baseline measured during same
            # four output seconds (no ECG in any feature).
            pulse=three[end-4:end,0]
            valid_pulse=np.isfinite(pulse)
            pulse_bpm=float(pulse[valid_pulse].mean()) if valid_pulse.sum()>=3 else None
            sections.append({
                "record":rec,"patient":patient,
                "start_sec":start,
                "raw_values":features[start:end].copy(),
                "available":available[start:end].copy(),
                "y":int(bpm>=THRESHOLD_BPM),
                "hr_reference":bpm,
                "pulse_bpm":pulse_bpm,
            })
    if not sections:raise RuntimeError("No real observed windows")
    split=group_allocations(patient_to_records)
    bysplit={}
    for label,groups in split.items():
        rows=[r for r in sections if r["patient"] in groups]
        if len(rows)<100:
            raise RuntimeError(f"Too few windows in isolated {label}")
        raw=np.asarray([r["raw_values"] for r in rows],dtype=np.float64)
        mask=np.asarray([r["available"] for r in rows],dtype=bool)
        bysplit[label]={
            "raw":raw,"mask":mask,
            "y":np.asarray([r["y"] for r in rows],dtype=np.uint8),
            "hr":np.asarray([r["hr_reference"] for r in rows],dtype=float),
            "pulse":np.asarray([float("nan") if r["pulse_bpm"] is None
                                else r["pulse_bpm"] for r in rows]),
            # A local temporary grouping key for patient-cluster bootstrap.
            # NEVER export key or individual row.
            "patient":np.asarray([r["patient"] for r in rows]),
            "record":np.asarray([r["record"] for r in rows]),
        }
    norm=fit_train_normalization(
        bysplit["train"]["raw"],bysplit["train"]["mask"])
    for row in bysplit.values():
        row["x"]=apply_train_normalization(
            row["raw"],row["mask"],norm)
        if not np.isfinite(row["x"]).all():
            raise AssertionError("Invalid normalized observed sensor")
    out_manifest={
        "dataset":"PhysioNet BIDMC v1.0.0, numerics and Fix only",
        "source_doi":"10.13026/C2208R",
        "records":EXPECTED_RECORDS,
        "identified_patient_groups":len(patient_to_records),
        "split_seed":GROUP_SEED,
        "windows_seconds":TIME_STEPS,"stride_seconds":WINDOW_STRIDE,
        "outcome":"contemporaneous ECG-derived monitor HR >= 100 bpm",
        "HR_used_as_input":False,
        "HR_is_both_supervised_target_and_numeric_evaluation_reference":True,
        "no_independent_external_assay_claim":True,
        "identical_patient_in_multiple_splits":False,
        "source_monitor_input_channels":["PULSE","RESP","SpO2"],
        "normalization":norm.manifest(),
        "splits":{
            name:{
                "patient_group_count":len(groups),
                "public_BIDMC_record_ordinals":sorted({
                    rec for p in groups for rec in patient_to_records[p]}),
                "windows":len(bysplit[name]["y"]),
                "positive_prevalence":float(bysplit[name]["y"].mean()),
                "natural_observation_fraction":
                    float(bysplit[name]["mask"].mean()),
            } for name,groups in split.items()
        },
        "patient_identifier_values_exported":False,
        "any_patient_time_series_exported":False,
        "raw_vitals_exported":False,
    }
    return bysplit,norm,out_manifest
