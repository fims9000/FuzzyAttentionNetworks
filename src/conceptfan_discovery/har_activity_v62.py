"""V62 real UCI HAR sensor modalities and participant-isolated tasks.

The UCI dataset's original overlapping 128-point windows are NOT independent
trials. We retain official 21-person development / 9-person final TEST split.
One "anchor" body-accelerometer vector magnitude vs two gyroscope measurements
tests new conditional statistical information from a genuinely different
physical sensor type. No physiological/causal identity is inferred.
"""
from __future__ import annotations

from dataclasses import asdict,replace
import json
from pathlib import Path

import numpy as np

from .program_compiler_v60 import ProgramBankV60,MeasuredProgramV60
from .real_future_targets_v61 import (
    AllowedSensorScalerV61,_fit_scaler,transform_allowed_v61,
)

SEED=20266213
TASKS={
    "STAIRS_UP_VS_DOWN":{"labels":(2,3),"positive":2},
    "STAND_VS_SIT":{"labels":(4,5),"positive":5},
}
SENSOR_NAMES=("BODY_ACCELEROMETER_NORM",
              "BODY_GYROSCOPE_NORM",
              "BODY_GYROSCOPE_SIGNED_Y")
STRESSES=("natural","MCAR20","gyro_missing",
          "accel_missing","gyro_gain_1p5")


def collapse_raw_axes_v62(raw_axes):
    axes=np.asarray(raw_axes,np.float64)
    if axes.ndim!=3 or axes.shape[1:]!=(128,6) or not np.isfinite(axes).all():
        raise ValueError("Expected original six-axis UCI raw inertial [N,128,6]")
    anchor=np.sqrt(np.sum(axes[:,:,:3]**2,axis=2))
    gyro=np.sqrt(np.sum(axes[:,:,3:6]**2,axis=2))
    gyro_y=axes[:,:,4]
    raw=np.stack((anchor,gyro,gyro_y),axis=-1)
    return raw.reshape(len(raw),16,8,3).mean(axis=2)


def _section(source,mask):
    with np.load(source,allow_pickle=False) as z:
        axes=z["axes"].astype(np.float32)
        ids=z["subject"].astype(np.int32)
        label=z["y"].astype(np.int32)
    take=np.isin(label,mask["labels"])
    if int(take.sum())<200:
        raise RuntimeError("Predeclared UCI activity task too small")
    raw_axes=axes[take];p=ids[take];out=(label[take]==mask["positive"]).astype(np.uint8)
    if not len(np.unique(p))>=4:
        raise RuntimeError("Insufficient physically independent HAR people")
    return {"axes":raw_axes,"patient":p,"y":out}


def _masked_partition(section,people):
    # np.isin(array, set) treats a Python set as ONE object; convert
    # genuine subject-id membership explicitly before masking windows.
    ids=np.isin(section["patient"],np.asarray(sorted(people)))
    if not ids.any():raise RuntimeError("No official HAR participant split")
    axes=section["axes"][ids]
    raw=collapse_raw_axes_v62(axes)
    mask=np.ones_like(raw,dtype=bool)
    return {"axes":axes,"raw":raw,"mask":mask,
            "y":section["y"][ids],
            "patient":section["patient"][ids],
            "anchor":raw[:,-4:,0].mean(axis=1)}


def load_har_v62(data:Path,task):
    if task not in TASKS:raise ValueError("V62 task not preregistered")
    data=Path(data)
    # No attempt to open private_test.npz until after every TRAIN and VALID
    # model/gate is frozen. The official 21/9 group split is public metadata.
    raw_train=_section(data/"private_train.npz",TASKS[task])
    # Subject partition MUST be fixed from official 21 person list before
    # task-specific labels are examined; data loader uses all original ids.
    with np.load(data/"private_train.npz",allow_pickle=False) as z:
        people=sorted(set(z["subject"].tolist()))
    if len(people)!=21:raise RuntimeError("Official UCI HAR train subjects not 21")
    rng=np.random.default_rng(SEED)
    groups=np.asarray(people)[rng.permutation(len(people))]
    subsets={"train":set(groups[:13]),"valid1":set(groups[13:17]),
             "valid2":set(groups[17:21])}
    sections={key:_masked_partition(raw_train,s)
              for key,s in subsets.items()}
    if any(len(set(sections[k]["patient"]))!=v
           for k,v in (("train",13),("valid1",4),("valid2",4))):
        raise AssertionError("HAR subject participant split is not 13/4/4")
    scaler=_fit_scaler(sections["train"],(0,1,2))
    for section in sections.values():
        section["x"],section["mask"]=transform_allowed_v61(
            section["raw"],section["mask"],scaler)
    receipt={
        "dataset":"UCI HAR using Smartphones DOI 10.24432/C54S4K",
        "binary_task":task,
        "official_source_classes":TASKS[task],
        "seed_dev_participant_split":SEED,
        "official_TEST_participants_not_touched_by_training":True,
        "official_TEST_n_people_public_metadata":9,
        "official_TEST_opened_only_after_ALL_train_VALID_fit":True,
        "original_raw_samples_per_window":128,
        "nonoverlapping_causal_blocks":16,
        "source_names":SENSOR_NAMES,
        "precomputed_561_vector_not_used":True,
        "TRAIN_only_scaler":scaler.passport(),
        "raw_participant_identifiers_in_public_receipt":False,
        "splits":{
            key:{"true_participant_count":len(np.unique(sec["patient"])),
                 "overlapping_windows":len(sec["y"]),
                 "class_prevalence":float(np.mean(sec["y"]))}
            for key,sec in sections.items()
        },
    }
    return sections,scaler,receipt


def load_har_test_v62(data:Path,task,scaler):
    """FIRST read original heldout test AFTER all seed-specific models/gates.

    Split details are checked only after all model definitions are frozen.
    This function must never be called inside source generation or fitting.
    """
    if task not in TASKS:raise ValueError("Unregistered final TEST task")
    root=Path(data)
    # Official identifiers only used for runtime group leakage check.
    with np.load(root/"private_train.npz",allow_pickle=False) as z:
        dev_people=set(z["subject"].tolist())
    raw_test=_section(root/"private_test.npz",TASKS[task])
    held_people=set(raw_test["patient"].tolist())
    if (len(held_people)!=9 or dev_people&held_people):
        raise AssertionError("Official UCI HAR TEST people overlap TRAIN")
    section=_masked_partition(raw_test,held_people)
    section["x"],section["mask"]=transform_allowed_v61(
        section["raw"],section["mask"],scaler)
    return section


def rename_bank_as_har_v62(bank:ProgramBankV60):
    updated=[]
    for c in bank.candidates:
        updated.append(replace(c,
            channel_names=tuple(SENSOR_NAMES[j] for j in c.channels),
            measurement_scope="STATISTICAL_HAR_INERTIAL_PROGRAM"))
    return replace(bank,candidates=tuple(updated),channels=SENSOR_NAMES)


def stress_har_v62(section,scaler,stress,*,seed):
    if stress not in STRESSES:raise ValueError("Unknown registered HAR stress")
    rng=np.random.default_rng(seed)
    original=section["axes"].astype(np.float64).copy()
    n=len(original)
    # Availability is registered at original 8-point blocks; an entire
    # gyroscope device outage also suppresses BOTH of its derived sources.
    available=np.ones((n,16,2),bool) # body_acc / body_gyro
    if stress=="MCAR20":
        available=rng.random((n,16,2))>=.20
    elif stress=="gyro_missing":
        available[:,:,1]=False
    elif stress=="accel_missing":
        available[:,:,0]=False
    elif stress=="gyro_gain_1p5":
        original[:,:,3:]*=1.5
    for k,(a,b) in enumerate(((0,3),(3,6))):
        mask8=np.repeat(available[:,:,k],8,axis=1)
        original[:,:,a:b]=np.where(mask8[:,:,None],
                                   original[:,:,a:b],0.)
    raw=collapse_raw_axes_v62(original)
    mask=np.ones_like(raw,dtype=bool)
    mask[:,:,0]=available[:,:,0]
    mask[:,:,1]=available[:,:,1]
    mask[:,:,2]=available[:,:,1]
    x,mask=transform_allowed_v61(raw,mask,scaler)
    return x,mask,original,available
