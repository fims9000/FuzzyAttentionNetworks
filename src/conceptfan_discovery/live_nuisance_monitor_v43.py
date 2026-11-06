"""V43 real, separately observed nuisance monitor: TRAIN-only alarm & live gate.

This module NEVER receives true latent source S, unknown mixing H,
source energy R/E, clinical Y, or TEST labels. The detector obtains actual
independent measurements of U_on and U_off, assumes their gain and physical
meaning are externally validated and tests whether the *observed* two
backgrounds are matched. A healthy alarm only certifies this narrow
condition; it is NOT a certificate of healthy ON/OFF switching, sensor
polarity, no source leakage, or medical concept identity.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
from pathlib import Path
import json

import numpy as np

MIN_OVERLAP=8
TRAIN_QUANTILE=.99
VALID_MAX_FALSE_ALARM=.03
VALID_MIN_COVERAGE=.90


@dataclass(frozen=True)
class LiveNuisanceMonitorV43:
    train_threshold_RMS:float
    train_joint_n:int
    valid1_joint_n:int
    valid2_joint_n:int
    valid1_false_alarm_fraction:float
    valid2_false_alarm_fraction:float
    valid1_sufficient_fraction:float
    valid2_sufficient_fraction:float
    passed_both_VALIDs:bool
    reason:str
    source_background_monitor_is_new_external_instrument:bool=True
    no_Y_R_E_S_H_in_monitor_fitting:bool=True
    no_current_sensor_isolation_or_truth_certification:bool=True

    def audit(self):
        return asdict(self)


def monitor_measurements_v43(control_on,control_off,mask_on,mask_off):
    a=np.asarray(control_on)
    b=np.asarray(control_off)
    ma=np.asarray(mask_on)
    mb=np.asarray(mask_off)
    if (a.ndim!=2 or a.shape[1]!=16 or b.shape!=a.shape
        or ma.shape!=a.shape or mb.shape!=a.shape
        or ma.dtype!=np.bool_ or mb.dtype!=np.bool_
        or not np.issubdtype(a.dtype,np.number)
        or not np.issubdtype(b.dtype,np.number)
        or not np.isfinite(a[ma]).all() or
        not np.isfinite(b[mb]).all()):
        raise ValueError(
            "Strict actually observed nuisance monitor [N,16] with BOOL masks required")
    both=ma&mb
    n=both.sum(axis=1).astype(int)
    # np.where avoids reading hidden NaN payload into arithmetic,
    # although NaN->0 paths are explicitly disallowed as measurements.
    x=np.where(both,np.asarray(a,dtype=np.float64),0.)
    y=np.where(both,np.asarray(b,dtype=np.float64),0.)
    rms=np.sqrt(np.sum((x-y)**2,axis=1)/np.maximum(n,1))
    return rms,n,n>=MIN_OVERLAP


def fit_live_nuisance_monitor_v43(
    train,valid1,valid2,
)->LiveNuisanceMonitorV43:
    """3 disjoint already-observed tuple(ON,OFF,on_mask,off_mask) cohorts."""
    if len(train[0])!=1000 or len(valid1[0])!=400 or len(valid2[0])!=400:
        raise ValueError("Frozen TRAIN1000/VALID400/VALID400 monitor budget")
    dtr,ntr,used_tr=monitor_measurements_v43(*train)
    d1,n1,used1=monitor_measurements_v43(*valid1)
    d2,n2,used2=monitor_measurements_v43(*valid2)
    if used_tr.mean()<.90 or used_tr.sum()<900:
        raise ValueError("Unreliable new independently observed training monitor")
    threshold=float(np.quantile(dtr[used_tr],TRAIN_QUANTILE))
    if not np.isfinite(threshold) or threshold<=0:
        raise ValueError("Invalid TRAIN calibration threshold on physical nuisance")
    false1=float(np.mean(d1[used1]>threshold)) if used1.any() else 1.
    false2=float(np.mean(d2[used2]>threshold)) if used2.any() else 1.
    accepted=(used1.mean()>=VALID_MIN_COVERAGE and
              used2.mean()>=VALID_MIN_COVERAGE and
              false1<=VALID_MAX_FALSE_ALARM and
              false2<=VALID_MAX_FALSE_ALARM)
    return LiveNuisanceMonitorV43(
        threshold,int(used_tr.sum()),int(used1.sum()),int(used2.sum()),
        false1,false2,float(used1.mean()),float(used2.mean()),accepted,
        ("TRAIN_AND_BOTH_VALID_INDEPENDENT_PHYSICAL_NUISANCE_MATCH"
         if accepted else "NO_HISTORICAL_MONITOR_REPRODUCIBILITY"),
    )


def evaluate_live_nuisance_monitor_v43(
    monitor:LiveNuisanceMonitorV43,
    control_on,control_off,mask_on,mask_off,
):
    rms,counts,measured=monitor_measurements_v43(
        control_on,control_off,mask_on,mask_off)
    detected=measured & (rms>monitor.train_threshold_RMS)
    # Vectorized fail-closed run; do not turn a scalar decision into a
    # source estimate when monitoring is unavailable.
    accepted=(
        measured&(rms<=monitor.train_threshold_RMS)
        if monitor.passed_both_VALIDs else np.zeros(len(rms),dtype=bool))
    reason=np.where(
        ~monitor.passed_both_VALIDs,
        "NO_VALIDATED_NUISANCE_MONITOR",
        np.where(~measured,
                 "REFUSE_MONITOR_UNAVAILABLE",
                 np.where(detected,
                          "REFUSE_LIVE_ON_OFF_BACKGROUND_MISMATCH",
                          "MEASURED_MONITOR_MATCHED_BACKGROUND"))
    )
    return {
        "accepted_matching_background":accepted,
        "current_MONITOR_RMS":rms,
        "current_MONITOR_joint_timesteps":counts,
        "alarm_TRUE_observed_background_difference":detected,
        "measured_monitor_sufficient":measured,
        "decision_reason":reason,
        "external_instrument_physical_meaning_ASSUMED":True,
        "current_sensor_source_isolation_proved":False,
        "monitor_has_no_direct_access_to_target_physical_E":True,
    }


def gate_physical_reference_v43(
    history:dict,live:dict,
):
    old=np.asarray(history["physical_reference_allowed"],dtype=bool)
    accepts=np.asarray(live["accepted_matching_background"],dtype=bool)
    if old.shape!=accepts.shape:
        raise ValueError("History vs present instrument cohort differs")
    new=old&accepts
    old_energy=np.asarray(
        history["physical_reference_estimated_energy"],dtype=np.float64)
    old_lower=np.asarray(
        history["physical_reference_interval_lower"],dtype=np.float64)
    old_upper=np.asarray(
        history["physical_reference_interval_upper"],dtype=np.float64)
    reason=np.asarray(history["physical_reference_reason"],dtype="U100").copy()
    reason[old & ~accepts]=np.asarray(live["decision_reason"])[old & ~accepts]
    return {
        **history,
        "physical_reference_allowed":new,
        "physical_reference_estimated_energy":np.where(
            new,old_energy,np.nan),
        "physical_reference_interval_lower":np.where(
            new,old_lower,np.nan),
        "physical_reference_interval_upper":np.where(
            new,old_upper,np.nan),
        "live_monitor_RMS":np.asarray(live["current_MONITOR_RMS"]).copy(),
        "live_monitor_accepted":accepts.copy(),
        "live_monitor_joint_timesteps":np.asarray(
            live["current_MONITOR_joint_timesteps"]).copy(),
        "physical_reference_reason":reason,
        "extra_hardware_assumed_not_self_certifying":True,
        "source_leak_or_sensor_flip_may_pass_monitor":True,
        "no_current_reference_E_or_R_input":True,
    }


def save_live_monitor_v43(path:Path,model:LiveNuisanceMonitorV43):
    path=Path(path)
    if path.exists():
        raise FileExistsError("Never overwrite TRAIN-calibrated monitor")
    payload={
        "schema":"conceptfan-v43-independent-nuisance-monitor-v1",
        "monitor":model.audit(),
        "external_physical_instrument_ASSUMED":True,
        "no_patient_level_readings_or_hidden_truth_stored":True,
    }
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(
        payload,indent=2,ensure_ascii=False,allow_nan=False))


def load_live_monitor_v43(path:Path)->LiveNuisanceMonitorV43:
    payload=json.loads(Path(path).read_text())
    if (payload.get("schema")!="conceptfan-v43-independent-nuisance-monitor-v1"
        or payload.get("external_physical_instrument_ASSUMED") is not True
        or payload.get("no_patient_level_readings_or_hidden_truth_stored") is not True):
        raise ValueError("Invalid saved current physical control model")
    data=payload["monitor"]
    obj=LiveNuisanceMonitorV43(**data)
    if (not np.isfinite(obj.train_threshold_RMS)
        or obj.train_threshold_RMS<=0
        or obj.train_joint_n<900 or
        obj.valid1_joint_n<360 or obj.valid2_joint_n<360
        or not obj.source_background_monitor_is_new_external_instrument
        or not obj.no_Y_R_E_S_H_in_monitor_fitting):
        raise ValueError("Invalid monitor threshold or forged instrument receipt")
    return obj
