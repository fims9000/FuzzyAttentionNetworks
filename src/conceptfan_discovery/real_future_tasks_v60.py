"""V60: real hospital chronology for *non-contemporaneous* +60s HR task.

The new target has never been a ConceptFAN input at any timestamp:
  input t0 ... t0+15; target t0+16+60 ... t0+16+63.
MIMIC-BIDMC records and VitalDB patient groups remain strictly disjoint;
train normalization uses BIDMC TRAIN input X only.
This module does not endorse future clinical prediction from bedside numerics.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from .bidmc_realworld_v58 import (
    EXPECTED_RECORDS,TIME_STEPS,WINDOW_STRIDE,
    THRESHOLD_BPM,
    _read_patient_group,_numbers,causal_monitor_features,
    group_allocations,fit_train_normalization,
    apply_train_normalization,load_bidmc_realworld_v58,
)

FUTURE_HORIZON=60
TARGET_LENGTH=4
TARGET_OFFSET=TIME_STEPS+FUTURE_HORIZON


def _window_rows(raw_three,hr,*,case,patient,task):
    if task not in ("simultaneous","future_60s"):
        raise ValueError("V60 target mode not registered")
    feat,mask=causal_monitor_features(raw_three)
    rows=[]
    shift=(FUTURE_HORIZON if task=="future_60s" else -TARGET_LENGTH)
    for start in range(0,len(hr)-TIME_STEPS-max(0,FUTURE_HORIZON+TARGET_LENGTH)
                       +1,WINDOW_STRIDE):
        end=start+TIME_STEPS
        target_start=(end+FUTURE_HORIZON if task=="future_60s"
                      else end-TARGET_LENGTH)
        target=hr[target_start:target_start+TARGET_LENGTH]
        good=np.isfinite(target)&(target>=25)&(target<=250)
        if int(good.sum())<3:continue
        bpm=float(target[good].mean())
        pulse=raw_three[end-4:end,0]
        avail=np.isfinite(pulse)
        current_pulse=float(pulse[avail].mean()) if avail.sum()>=3 else np.nan
        rows.append({
            "patient":patient,"case":case,"start":start,
            "raw":feat[start:end].copy(),
            "mask":mask[start:end].copy(),
            "y":int(bpm>=THRESHOLD_BPM),
            "hr":bpm,"pulse":current_pulse,
            "target_begin":target_start,
            "input_end_exclusive":end,
        })
    return rows


def load_bidmc_task_v60(root:Path,task):
    if task=="simultaneous":
        data,norm,manifest=load_bidmc_realworld_v58(root)
        return data,norm,{**manifest,"task":"simultaneous",
              "future_HR_as_input":False,
              "target_begins_after_input":False,
              "outcome_horizon_seconds":0}
    if task!="future_60s":
        raise ValueError("Only exact preregistered tasks")
    root=Path(root)
    by_patient=defaultdict(list);windows=[]
    for i in range(1,EXPECTED_RECORDS+1):
        prefix=f"bidmc_{i:02d}"
        patient=_read_patient_group(
            (root/(prefix+"_Fix.txt")).read_text(encoding="utf-8-sig"))
        data=_numbers(root/(prefix+"_Numerics.csv"))
        rows=_window_rows(data[:,:3],data[:,3],
                          case=i,patient=patient,task=task)
        windows.extend(rows);by_patient[patient].append(i)
    allocation=group_allocations(by_patient)
    out={}
    for name,groups in allocation.items():
        rows=[w for w in windows if w["patient"] in groups]
        if len(rows)<90:raise RuntimeError("Too few future windows "+name)
        assert all(r["target_begin"]>=r["input_end_exclusive"]+FUTURE_HORIZON
                   for r in rows)
        out[name]={
            "raw":np.stack([r["raw"] for r in rows]),
            "mask":np.stack([r["mask"] for r in rows]),
            "y":np.array([r["y"] for r in rows],np.uint8),
            "hr":np.array([r["hr"] for r in rows],float),
            "pulse":np.array([r["pulse"] for r in rows],float),
            "patient":np.array([r["patient"] for r in rows]),
            "record":np.array([r["case"] for r in rows]),
        }
    norm=fit_train_normalization(
        out["train"]["raw"],out["train"]["mask"])
    for partition in out.values():
        partition["x"]=apply_train_normalization(
            partition["raw"],partition["mask"],norm)
    manifest={
        "dataset":"PhysioNet BIDMC v1.0.0, official numeric and Fix files",
        "task":"future_60s",
        "outcome":"ECG monitor HR>=100 bpm 60s after input window ends",
        "outcome_horizon_seconds":FUTURE_HORIZON,
        "target_length_seconds":TARGET_LENGTH,
        "future_HR_as_input":False,
        "target_begins_after_input":True,
        "source_monitor_input_channels":["PULSE","RESP","SpO2"],
        "train_only_feature_normalization":norm.manifest(),
        "shared_patient_allocation_seed":20265813,
        "split_patient_id_values_exported":False,
        "splits":{
            name:{
                "patient_group_count":len(groups),
                "public_BIDMC_record_ordinals":sorted({
                    i for p in groups for i in by_patient[p]}),
                "windows":len(out[name]["y"]),
                "positive_prevalence":float(np.mean(out[name]["y"])),
            } for name,groups in allocation.items()
        },
    }
    return out,norm,manifest


def load_vitaldb_task_v60(root:Path,norm,task):
    if task=="simultaneous":
        from scripts.experiment_conceptfan_vitaldb_transfer_v59 import (
            load_external_vitaldb,
        )
        return load_external_vitaldb(root,norm)
    if task!="future_60s":raise ValueError("Unsupported future task")
    root=Path(root)
    receipt=json.loads((root/"vitaldb_download_receipt.json").read_text())
    cases=receipt["selected_public_case_ordinals"]
    if not(30<=len(cases)<=40) or len(set(cases))!=len(cases):
        raise RuntimeError("External site enrolment changed")
    windows=[]
    for case in cases:
        with np.load(root/f"vitaldb_case_{case}.npz",allow_pickle=False) as z:
            signals=z["signals"].astype(float)
        if signals.shape!=(480,4):
            raise RuntimeError("External source record interval is not 480 seconds")
        windows.extend(_window_rows(
            signals[:,:3],signals[:,3],case=case,
            patient=int(case),task="future_60s"))
    if len(windows)<700:
        raise RuntimeError("Too few external future windows")
    raw=np.stack([r["raw"] for r in windows])
    mask=np.stack([r["mask"] for r in windows])
    output={
        "raw":raw,"mask":mask,
        "x":apply_train_normalization(raw,mask,norm),
        "y":np.array([r["y"] for r in windows],np.uint8),
        "hr":np.array([r["hr"] for r in windows],float),
        "pulse":np.array([r["pulse"] for r in windows],float),
        # Public VitalDB case ordinal doubles as patient-group key:
        # downloader confirmed exactly one case per distinct subject.
        "patient":np.array([r["patient"] for r in windows]),
    }
    if len(set(output["patient"]))!=len(cases):
        raise AssertionError("VitalDB unique patient-group contract broken")
    return output,receipt
