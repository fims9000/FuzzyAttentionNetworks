"""V38 external reference-response audit and negative-control sensor admission.

Neither is inferable from the patient observations X alone. The sensor10
wiring and known actuator targets are EXTERNAL knowledge, simulator-supplied
here. This is neither unsupervised biological concept discovery nor
instrumental-variable inference for a clinical treatment effect.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np

from .threeview_centered_v37 import threeview_feature_v37,ThreeViewFitV37


@dataclass(frozen=True)
class SentinelFitV38:
    threshold_train_only:float
    train_monitor_episodes:int
    min_observed_times:int=10
    independently_wired_instrument_artifact_monitor_assumed:bool=True
    physical_source_identity_certified:bool=False


@dataclass(frozen=True)
class ProbeGateV38:
    passed:bool
    target_mean_effect_half:tuple[float,float]
    nuisance_mean_effect_half:tuple[float,float]
    target_bootstrap_lower_half:tuple[float,float]
    target_vs_nuisance_half:tuple[float,float]
    target_supported_pairs_half:tuple[int,int]
    nuisance_supported_pairs_half:tuple[int,int]
    independent_actuator_calibration_episodes_per_type:int
    abs_sentinel_command_response_correlation:float
    known_actuator_mapping_externally_supplied:bool=True
    latent_H_or_S_supplied_to_calibrator:bool=False
    biological_causal_semantics_proven:bool=False

    def audit(self):
        return asdict(self)


def _validate(x,m):
    v=np.asarray(x,dtype=np.float64)
    mask=np.asarray(m,dtype=bool)
    if (v.ndim!=3 or v.shape!=mask.shape or v.shape[1:]!=(16,12)
        or len(v)==0 or not np.isfinite(v[mask]).all()):
        raise ValueError("Invalid monitored source measurements")
    return v,mask


def observed_sentinel_v38(x,m):
    """Per-episode centered variance of actually OBSERVED channel10 samples."""
    v,mask=_validate(x,m)
    present=mask[:,:,10]
    n=present.sum(axis=1)
    data=np.where(present,v[:,:,10],0.)
    mu=data.sum(axis=1)/np.maximum(n,1)
    var=np.where(present,(data-mu[:,None])**2,0.).sum(axis=1)/np.maximum(n,1)
    return {"centered_variance":var,"observed_times":n}


def fit_sentinel_train_only_v38(train_x,train_mask)->SentinelFitV38:
    stat=observed_sentinel_v38(train_x,train_mask)
    sufficient=stat["observed_times"]>=10
    if int(sufficient.sum())<180:
        raise ValueError("ABSTAIN: negative-control sensor not measured on TRAIN")
    q=float(np.quantile(stat["centered_variance"][sufficient],.995))
    if not np.isfinite(q):
        raise ValueError("ABSTAIN: no finite TRAIN sentinel calibration")
    return SentinelFitV38(max(.08,q),int(sufficient.sum()))


def negative_control_runtime_v38(sentinel:SentinelFitV38,x,m):
    stat=observed_sentinel_v38(x,m)
    good=(stat["observed_times"]>=sentinel.min_observed_times)&(
        stat["centered_variance"]<=sentinel.threshold_train_only)
    return {
        "eligible":good,
        "sentinel_centered_variance":stat["centered_variance"],
        "sentinel_observed_timesteps":stat["observed_times"],
        "train_only_variance_threshold":sentinel.threshold_train_only,
        "external_sentinel_wiring_assumed":True,
        "artifact_can_bypass_control":True,
    }


def _paired_calibration_effect(fit:ThreeViewFitV37,record:dict):
    if set(record)!={
        "off","on","mask","known_actuator_command",
        "physical_source_identity_external_assumption",
    }:
        raise ValueError("Probe data must contain ONLY observed paired readouts and actuator command")
    off=np.asarray(record["off"],dtype=np.float32)
    on=np.asarray(record["on"],dtype=np.float32)
    mask=np.asarray(record["mask"],dtype=bool)
    command=np.asarray(record["known_actuator_command"],dtype=np.float64)
    if (off.shape!=on.shape or off.shape!=mask.shape or
        off.ndim!=3 or off.shape[1:]!=(16,12)
        or command.shape!=off.shape[:2] or len(off)<180
        or not record["physical_source_identity_external_assumption"]):
        raise ValueError("Insufficient independently randomized paired reference data")
    measured_off=threeview_feature_v37(fit,off,mask)
    measured_on=threeview_feature_v37(fit,on,mask)
    supported=(measured_off["support"]>0)&(measured_on["support"]>0)
    effect=measured_on["q"]-measured_off["q"]
    return effect,supported,command,off,on,mask


def _probe_sentinel_correlation(record):
    delta=np.asarray(record["on"],dtype=np.float64)[:,:,10]-(
        np.asarray(record["off"],dtype=np.float64)[:,:,10])
    command=np.asarray(record["known_actuator_command"],dtype=np.float64)
    mask=np.asarray(record["mask"],dtype=bool)[:,:,10]
    x=command[mask]
    y=delta[mask]
    if len(x)<500 or float(y.std())<.005:
        raise ValueError("ABSTAIN: controlled probe negative-control readout invalid")
    return abs(float(np.corrcoef(x,y)[0,1]))


def fit_probe_intervention_v38(
    source_fit:ThreeViewFitV37,
    calibration:dict,
    *,
    seed:int=20263881,
    bootstrap_reps:int=199,
)->ProbeGateV38:
    if bootstrap_reps!=199 or set(calibration)!={"target","nuisance"}:
        raise ValueError("Only frozen registered target and negative-control probes")
    target=_paired_calibration_effect(source_fit,calibration["target"])
    nuisance=_paired_calibration_effect(source_fit,calibration["nuisance"])
    te,ts,command,*_=target
    ne,ns,nuisance_command,*_=nuisance
    if len(te)!=len(ne) or len(te) not in (180,240):
        raise ValueError("Registered paired probes must be disjoint and balanced")
    if not np.allclose(np.abs(command),1) or not np.allclose(
        np.abs(nuisance_command),1):
        raise ValueError("Actuator command not a registered randomized square wave")
    sentinel_corr=max(_probe_sentinel_correlation(calibration[k])
                      for k in ("target","nuisance"))
    rng=np.random.default_rng(seed)
    target_means=[];nuisance_means=[];lowers=[];gaps=[]
    ns_target=[];ns_nuisance=[]
    for s in (slice(0,len(te)//2),slice(len(te)//2,None)):
        a=te[s][ts[s]]
        b=ne[s][ns[s]]
        if len(a)<35 or len(b)<35:
            raise ValueError("ABSTAIN: independent probe pairs not measured")
        ma,mb=float(a.mean()),float(b.mean())
        draw=rng.integers(0,len(a),size=(bootstrap_reps,len(a)))
        lower=float(np.quantile(a[draw].mean(axis=1),.025))
        target_means.append(ma);nuisance_means.append(mb)
        lowers.append(lower);gaps.append(ma-mb)
        ns_target.append(len(a));ns_nuisance.append(len(b))
    approved=(all(v>.03 for v in target_means)
              and all(v>.02 for v in gaps)
              and all(v>0. for v in lowers)
              and sentinel_corr<.15)
    return ProbeGateV38(
        approved,
        tuple(target_means),tuple(nuisance_means),
        tuple(lowers),tuple(gaps),
        tuple(ns_target),tuple(ns_nuisance),
        len(te),sentinel_corr,
    )


def source_with_independent_evidence_v38(
    source_fit:ThreeViewFitV37,
    sentinel_fit:SentinelFitV38,
    probe_gate:ProbeGateV38,
    x,m,
    *,
    require_probe:bool=True,
):
    measured=threeview_feature_v37(source_fit,x,m)
    monitor=negative_control_runtime_v38(sentinel_fit,x,m)
    approved=(measured["support"]>0)&monitor["eligible"]
    if require_probe:
        approved &= bool(probe_gate.passed)
    return {
        **measured,
        "q":np.where(approved,measured["q"],.5),
        "support":approved.astype(np.float64),
        "original_threeview_source_support":measured["support"],
        "external_probe_gate_passed":bool(probe_gate.passed),
        "independent_probe_required":bool(require_probe),
        "instrument_control":monitor,
        "source_physically_causal_or_biologically_named":False,
        "actuator_reference_is_external_supervision":True,
        "false_source_possible_if_artifact_bypasses_monitor":True,
    }
