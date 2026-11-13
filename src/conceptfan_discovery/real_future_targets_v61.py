"""V61 real tasks with hard outcome-channel exclusion BEFORE preprocessing.

Two preregistered tasks:
  HR+60s: binary future ECG monitor HR >=100 bpm; input PULSE,RESP,SpO2.
  RESP+60s: binary future monitor RESP >=20 bpm; input PULSE,SpO2 ONLY.
Target 4-second window starts 60 seconds *after* 16-second input ends.
No supervised label, target-channel values, future time sample or group-specific
medical state is allowed into source feature compilation.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict,dataclass
import json
from pathlib import Path

import numpy as np

from .bidmc_realworld_v58 import (
    EXPECTED_RECORDS,TIME_STEPS,WINDOW_STRIDE,
    _read_patient_group,_numbers,group_allocations,
)

TASKS=("future_hr_60s","future_resp_60s")
TARGETS={
    "future_hr_60s":{"outcome_channel":3,"threshold":100.,
                      "allowed_channels":(0,1,2),"target_name":"ECG_MONITOR_HR"},
    "future_resp_60s":{"outcome_channel":1,"threshold":20.,
                        "allowed_channels":(0,2),"target_name":"RESP_MONITOR"},
}
HORIZON=60
FUTURE_WIDTH=4


@dataclass(frozen=True)
class AllowedSensorScalerV61:
    median:tuple[float,...]
    iqr:tuple[float,...]
    allowed_channels:tuple[int,...]
    fitted_on_train_only:bool=True
    excluded_target_channel_was_read_for_scaling:bool=False

    def passport(self):
        return asdict(self)


def _rows(real_four,case,patient,task):
    if task not in TARGETS:raise ValueError("Unknown frozen future monitor outcome")
    values=np.asarray(real_four,float)
    if values.ndim!=2 or values.shape[1]!=4:
        raise ValueError("Expected PULSE RESP SPO2 HR monitored numeric readings")
    spec=TARGETS[task]
    allowed=spec["allowed_channels"]
    outcome=values[:,spec["outcome_channel"]]
    original=np.asarray(values[:,:3],float)
    out=[]
    max_start=len(values)-TIME_STEPS-HORIZON-FUTURE_WIDTH+1
    for start in range(0,max_start,WINDOW_STRIDE):
        end=start+TIME_STEPS
        a=end+HORIZON
        target=outcome[a:a+FUTURE_WIDTH]
        valid=np.isfinite(target)&(target>=(
            25. if task=="future_hr_60s" else 4.))&(
            target<=(250. if task=="future_hr_60s" else 80.))
        if int(valid.sum())<3:continue
        # This is a *future threshold on a monitor*, not a diagnosis.
        label=float(target[valid].mean())>=spec["threshold"]
        inx=original[start:end].copy()
        mask=np.isfinite(inx)
        for channel in range(3):
            if channel not in allowed:
                mask[:,channel]=False
                inx[:,channel]=0.
        inx=np.where(mask,inx,0.)
        current_pulse=original[end-4:end,0]
        pulse_valid=np.isfinite(current_pulse)
        pulse=(float(current_pulse[pulse_valid].mean())
               if int(pulse_valid.sum())>=3 else np.nan)
        out.append({
            "case":case,"patient":patient,"start":start,
            "input_end_exclusive":end,"target_first":a,
            "raw":inx,"mask":mask,"y":int(label),
            "pulse":pulse,"future_value":float(target[valid].mean()),
        })
    return out


def _fit_scaler(train,allowed):
    raw=np.asarray(train["raw"],float)
    mask=np.asarray(train["mask"],bool)
    median=np.zeros(3);iqr=np.ones(3)
    for j in allowed:
        samples=raw[:,:,j][mask[:,:,j]]
        if len(samples)<100:raise RuntimeError("TRAIN observed allowed raw sensor insufficient")
        lo,mid,hi=np.quantile(samples,[.25,.5,.75])
        median[j]=float(mid)
        iqr[j]=float(max(hi-lo,.1))
    return AllowedSensorScalerV61(
        tuple(map(float,median)),tuple(map(float,iqr)),tuple(allowed))


def transform_allowed_v61(raw,mask,scaler):
    x=np.asarray(raw,float)
    m=np.asarray(mask,bool).copy()
    if x.ndim!=3 or x.shape[1:]!=(16,3) or m.shape!=x.shape:
        raise ValueError("Expected raw windows [N,16,3] aligned mask")
    allowed=set(scaler.allowed_channels)
    for j in range(3):
        if j not in allowed:m[:,:,j]=False
    med=np.asarray(scaler.median)
    div=np.asarray(scaler.iqr)
    if np.any(div<=0):raise ValueError("Bad TRAIN-only numeric scaling")
    z=np.where(m,np.clip((x-med[None,None,:])/div[None,None,:],
                         -8.,8.),0.).astype(np.float32)
    if not np.isfinite(z).all():
        raise AssertionError("Nonfinite normalized real monitor")
    return z,m


def _partition(rows,labels):
    obj=[r for r in rows if r["patient"] in labels]
    if len(obj)<70:raise RuntimeError("Insufficient real future monitor episodes")
    return {
        "raw":np.stack([r["raw"] for r in obj]),
        "mask":np.stack([r["mask"] for r in obj]),
        "y":np.asarray([r["y"] for r in obj],np.uint8),
        "patient":np.asarray([r["patient"] for r in obj]),
        "record":np.asarray([r["case"] for r in obj]),
        "pulse":np.asarray([r["pulse"] for r in obj],float),
        "future_value":np.asarray([r["future_value"] for r in obj],float),
    }


def load_bidmc_future_v61(source:Path,task):
    if task not in TARGETS:raise ValueError("Unknown frozen outcome")
    root=Path(source)
    people=defaultdict(list);rows=[]
    for rec in range(1,EXPECTED_RECORDS+1):
        pre=f"bidmc_{rec:02d}"
        person=_read_patient_group(
            (root/(pre+"_Fix.txt")).read_text(encoding="utf-8-sig"))
        values=_numbers(root/(pre+"_Numerics.csv"))
        people[person].append(rec)
        rows.extend(_rows(values,rec,person,task))
    groups=group_allocations(people)
    data={key:_partition(rows,patients) for key,patients in groups.items()}
    allowed=TARGETS[task]["allowed_channels"]
    norm=_fit_scaler(data["train"],allowed)
    for d in data.values():
        d["x"],d["mask"]=transform_allowed_v61(
            d["raw"],d["mask"],norm)
    if (len(set().union(*groups.values()))!=len(people)
        or any( (d["mask"][:,:,1].any())
               for d in data.values() if task=="future_resp_60s")):
        raise AssertionError("Patient or excluded RESP target leakage")
    receipt={
        "task":task,"future_horizon_seconds":HORIZON,
        "outcome_last4_mean_threshold":TARGETS[task]["threshold"],
        "target_name":TARGETS[task]["target_name"],
        "allowed_source_channels":[("PULSE","RESP","SPO2")[i] for i in allowed],
        "target_channel_absent_from_ALL_inputs":
            task!="future_resp_60s" or 1 not in allowed,
        "target_samples_never_overlap_input":True,
        "trained_on_real_BIDMC_patients_not_windows":True,
        "scaler":norm.passport(),
        "patient_identifiers_in_public_receipt":False,
        "split_seed_unchanged_from_V58":20265813,
        "splits":{
            key:{"patients":len(groups[key]),
                 "windows":len(data[key]["y"]),
                 "positive_fraction":float(np.mean(data[key]["y"])),
                 "public_BIDMC_record_ordinals":
                     sorted({rec for p in groups[key] for rec in people[p]})}
            for key in data
        },
    }
    return data,norm,receipt


def load_vitaldb_future_v61(source:Path,task,scaler):
    if task not in TARGETS:raise ValueError("Unknown frozen external outcome")
    root=Path(source)
    receipt=json.loads((root/"vitaldb_download_receipt.json").read_text())
    cases=receipt["selected_public_case_ordinals"]
    if not(30<=len(cases)<=40) or len(set(cases))!=len(cases):
        raise RuntimeError("External source enrollment deviated from frozen 40")
    rows=[]
    for case in cases:
        with np.load(root/f"vitaldb_case_{case}.npz",allow_pickle=False) as z:
            signal=np.asarray(z["signals"],float)
        if signal.shape!=(480,4):
            raise RuntimeError("External numerical record contract drift")
        rows.extend(_rows(signal,case,int(case),task))
    if len(rows)<700:
        raise RuntimeError("Too few patient-isolated external future windows")
    data=_partition(rows,set(cases))
    data["x"],data["mask"]=transform_allowed_v61(
        data["raw"],data["mask"],scaler)
    if (len(np.unique(data["patient"]))!=len(cases)
        or (task=="future_resp_60s" and
            data["mask"][:,:,1].any())):
        raise AssertionError("Unseen hospital outcome-channel/patient isolation failed")
    return data,receipt


def stress_v61(data,scaler,scenario,*,seed=20266199):
    if scenario not in ("natural","MCAR20","pulse_missing","pulse_plus_25_bpm"):
        raise ValueError("Unregistered V61 measured-sensor stress")
    x=data["raw"].copy()
    m=data["mask"].copy()
    if scenario=="MCAR20":
        rng=np.random.default_rng(seed)
        m &= rng.random(m.shape)>=.20
    elif scenario=="pulse_missing":
        m[:,:,0]=False
    elif scenario=="pulse_plus_25_bpm":
        x[:,:,0]=np.where(m[:,:,0],x[:,:,0]+25.,0.)
    z,m=transform_allowed_v61(x,m,scaler)
    return z,m
