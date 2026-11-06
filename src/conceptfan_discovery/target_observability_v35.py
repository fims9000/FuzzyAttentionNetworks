"""V35 calibrated *target-specific* concept observability.

Known inverse-problem fact, NOT a new identifiability theorem:

  y_M = A_M s + epsilon, target a=l.T @ s
  a is identifiable iff l in rowspace(A_M).
  If identifiable, Var[l.T @ s_hat] = l.T @ (A.T R^-1 A)^+ @ l
  under known A,R and independent Gaussian noise.

Rank(A_M)==number_of_sources is unnecessarily strong for recovering only a.
The pseudoinverse may yield a plausible number for an unidentifiable a;
NEVER call that number an identified source. A posterior imputation (Flow /
diffusion) cannot recover this missing information without a prior.

This module is conditional on independently calibrated device response and
KNOWN nonzero wiring support. Its Gaussian intervals are plug-in intervals,
NOT distribution-free guarantees; drift outside external calibration breaks
the claimed coordinate orientation. Measurements are never supervised by Y.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np
from scipy.special import expit


SENSORS=(4,5,6)
PHI=np.r_[np.ones(8),-np.ones(8)].astype(np.float64)
# This topology is EXTERNALLY VERIFIED PHANTOM HARDWARE METADATA, not
# recovered merely by inspecting correlations or outcome labels.
SUPPORT=np.array([[True,True],[False,True],[True,False]],dtype=bool)
TARGET=np.array([1.,0.],dtype=np.float64)
METHODS=(
    "unsafe_mask_fraction",
    "calibrated_full_rank",
    "calibrated_target_rowspace",
    "calibrated_target_residual",
)


@dataclass(frozen=True)
class ReferenceCalibration:
    response: np.ndarray    # [3,2] measured response to two known inputs
    noise_sd: np.ndarray    # [3] measured calibration residual noise
    verified_routing_support: np.ndarray  # [3,2] from device specification
    reference_count: int
    reference_sources_known: bool
    external_device_reference_used: bool
    calibration_sha256: str

    def __post_init__(self):
        h=np.asarray(self.response)
        s=np.asarray(self.noise_sd)
        support=np.asarray(self.verified_routing_support)
        if (h.shape!=(3,2) or s.shape!=(3,) or support.shape!=(3,2)
            or h.dtype.kind not in "fc" or s.dtype.kind not in "fc"
            or support.dtype!=bool or not np.isfinite(h).all()
            or not np.isfinite(s).all() or (s<=0).any()
            or self.reference_count<100 or not self.reference_sources_known
            or not self.external_device_reference_used
            or np.any(np.abs(h[~support])>1e-12)):
            raise ValueError("Cannot identify a measured source without explicit reference calibration")


def fit_external_reference(
    reference_source_inputs:np.ndarray,
    reference_device_outputs:np.ndarray,
    *,
    physically_verified_routing_support:np.ndarray,
    independent_reference:bool,
)->ReferenceCalibration:
    """Fit H from OUTCOME-BLIND known phantom excitations, not true patient a.

    Routing zeros MUST originate from external device specification, not
    statistical thresholding to make a rank-deficient case look identifiable.
    """
    u=np.asarray(reference_source_inputs,dtype=np.float64)
    x=np.asarray(reference_device_outputs,dtype=np.float64)
    support=np.asarray(physically_verified_routing_support)
    if (u.ndim!=2 or u.shape[1]!=2 or len(u)<100
        or x.shape!=(len(u),3) or support.shape!=(3,2)
        or support.dtype!=bool or not independent_reference
        or not np.isfinite(u).all() or not np.isfinite(x).all()
        or np.linalg.matrix_rank(u)<2
        or (support.sum(axis=1)==0).any()):
        raise ValueError("Independent two-source excitation and verified hardware routing required")
    h=np.zeros((3,2),dtype=np.float64)
    residual=np.empty_like(x)
    for j in range(3):
        active=support[j]
        h[j,active]=np.linalg.lstsq(u[:,active],x[:,j],rcond=None)[0]
        residual[:,j]=x[:,j]-u@h[j]
    sd=np.maximum(.01,np.sqrt(np.mean(residual**2,axis=0)))
    digest=sha256(
        h.astype(np.float64).tobytes()+
        sd.astype(np.float64).tobytes()+
        support.astype(np.uint8).tobytes()).hexdigest()
    return ReferenceCalibration(
        response=h,noise_sd=sd,verified_routing_support=support.copy(),
        reference_count=len(u),reference_sources_known=True,
        external_device_reference_used=True,calibration_sha256=digest,
    )


def make_independent_phantom(
    *,
    seed:int=20263500,
    n:int=900,
    # simulator-only true H is only in the DEVICE PHANTOM generator, NEVER
    # supplied to fit_external_reference or the TEST estimator.
    simulator_H:np.ndarray|None=None,
)->tuple[np.ndarray,np.ndarray]:
    if n<100:
        raise ValueError("Need independent reference excitation")
    h=(np.array([[1.,1.],[0.,1.],[1.,0.]])
       if simulator_H is None else np.asarray(simulator_H,dtype=float))
    if h.shape!=(3,2):
        raise ValueError("Invalid private simulator source mixing matrix")
    rng=np.random.default_rng(seed)
    u=rng.uniform(-1.65,1.65,size=(n,2))
    x=u@h.T+rng.normal(0,.32,size=(n,3))
    return u,x


def _target_pseudoinverse(
    observed_readings:np.ndarray,
    observed_design:np.ndarray,
    noise_sd:np.ndarray,
    *,
    target:np.ndarray=TARGET,
)->dict:
    """SVD of WHITENED design; nullspace test is structural, before estimate."""
    d=np.asarray(observed_design,dtype=np.float64)
    y=np.asarray(observed_readings,dtype=np.float64).reshape(-1)
    sd=np.asarray(noise_sd,dtype=np.float64).reshape(-1)
    ell=np.asarray(target,dtype=np.float64).reshape(-1)
    if (d.ndim!=2 or len(y)!=len(d) or len(sd)!=len(y)
        or d.shape[1]!=len(ell) or len(ell)<1
        or not np.isfinite(d).all() or not np.isfinite(y).all()
        or not np.isfinite(sd).all() or (sd<=0).any()
        or not np.isfinite(ell).all() or not np.linalg.norm(ell)):
        raise ValueError("Invalid calibrated target inverse problem")
    if len(y)==0:
        return {
            "estimated_target":0.,"target_se":float("inf"),
            "rank":0,"target_in_rowspace":False,
            "nullspace_target_fraction":1.,
            "normalized_residual":float("inf"),
            "n_rows":0,
        }
    a=d/sd[:,None]
    b=y/sd
    uu,ss,vt=np.linalg.svd(a,full_matrices=True)
    tol=max(a.shape)*np.finfo(float).eps*max(ss[0],1.)
    rank=int((ss>tol).sum())
    invisible=vt[rank:,:]@ell
    projection_error=float(np.linalg.norm(invisible)/np.linalg.norm(ell))
    identifiable=projection_error<=1e-8
    co=np.zeros(len(ell),dtype=np.float64)
    if rank:
        co=vt[:rank].T@((uu[:,:rank].T@b)/ss[:rank])
    estimated=float(ell@co)
    # This variance is defined FOR THE IDENTIFIABLE TARGET. For a target
    # outside rowspace the returned inf is deliberate (no pseudo-CI).
    variance=(
        float(np.sum((vt[:rank]@ell/ss[:rank])**2))
        if identifiable and rank else float("inf")
    )
    residual=float(np.sqrt(np.mean((b-a@co)**2)))
    return {
        "estimated_target":estimated if np.isfinite(estimated) else 0.,
        "target_se":float(np.sqrt(variance)),
        "rank":rank,
        "target_in_rowspace":bool(identifiable),
        "nullspace_target_fraction":projection_error,
        "normalized_residual":residual,
        "n_rows":len(y),
    }


def estimate_calibrated_source(
    x:np.ndarray,
    mask:np.ndarray,
    calibration:ReferenceCalibration,
    *,
    se_limit:float=.15,
    residual_limit:float|None=None,
)->dict[str,np.ndarray]:
    """Per-subject inference of calibrated *specific* target source a1.

    Mask percentages alone never establish identifiability. Full missing
    group gives exact neutral estimate/support0. All outcome supervision
    and pseudo-physical labels are absent.
    """
    v=np.asarray(x,dtype=np.float64)
    m=np.asarray(mask)
    if (v.ndim!=3 or v.shape[1:]!=(16,12) or v.shape!=m.shape
        or m.dtype!=bool or not np.isfinite(v[m]).all()
        or not 0<se_limit<=1
        or (residual_limit is not None and not 0<residual_limit<=10)):
        raise ValueError("Source estimator needs 16 measured times, 12 sensors, calibrated mask")
    n=len(v)
    output={k:[] for k in (
        "estimated_target","target_se","rank","target_in_rowspace",
        "nullspace_target_fraction","normalized_residual","n_rows",
        "measured_fraction",
    )}
    for i in range(n):
        rows=np.argwhere(m[i,:,4:7])
        if not len(rows):
            r=_target_pseudoinverse(
                np.empty(0),np.empty((0,2)),np.empty(0))
        else:
            time=rows[:,0]
            channel=rows[:,1]
            design=PHI[time,None]*calibration.response[channel]
            values=v[i,time,channel+4]
            noise=calibration.noise_sd[channel]
            r=_target_pseudoinverse(values,design,noise)
        r["measured_fraction"]=len(rows)/48.
        for key in output:
            output[key].append(r[key])
    ret={k:np.asarray(val) for k,val in output.items()}
    frac=ret["measured_fraction"]
    row=ret["target_in_rowspace"]
    accuracy=ret["target_se"]<=se_limit
    enough=ret["n_rows"]>0
    ret["unsafe_mask_fraction"]=(frac>=1/3)&enough
    ret["calibrated_full_rank"]=(ret["rank"]==2)&accuracy&enough
    ret["calibrated_target_rowspace"]=row&accuracy&enough
    ret["calibrated_target_residual"]=(
        ret["calibrated_target_rowspace"]&
        (True if residual_limit is None else
         ret["normalized_residual"]<=residual_limit))
    ret["external_calibration_sha256"]=calibration.calibration_sha256
    ret["uncertainty_from_calibration_not_type3"]=True
    ret["plug_in_se_not_distribution_free_ci"]=True
    return ret


def measured_target_membership(
    result:dict[str,np.ndarray],
    method:str,
    *,
    train_tau:float,
    train_width:float,
)->tuple[np.ndarray,np.ndarray]:
    if method not in METHODS:
        raise ValueError("Unknown registered calibrated source policy")
    if not np.isfinite(train_tau) or not np.isfinite(train_width) or train_width<=0:
        raise ValueError("TRAIN concept definition is required")
    used=np.asarray(result[method],dtype=bool)
    amp=np.asarray(result["estimated_target"],dtype=float)
    q=np.where(used,expit((amp-train_tau)/train_width),.5)
    return q,used.astype(np.float64)


def linear_logit_sensitivity_radius(
    gamma:float,
    target_se:np.ndarray,
    *,
    train_width:float,
    eligible:np.ndarray,
)->np.ndarray:
    """Conditional Lipschitz radius, NOT calibrated event probability CI."""
    se=np.asarray(target_se,dtype=float)
    valid=np.asarray(eligible,dtype=bool)
    if se.shape!=valid.shape or train_width<=0 or not np.isfinite(gamma):
        raise ValueError("Misaligned source uncertainty")
    return np.where(valid,abs(gamma)*1.96*se/(4*train_width),0.)


def nullspace_counterexample()->dict:
    """Same physical measurement from two different latent concept worlds."""
    H=np.array([[1.,1.]])
    a=np.array([+.9,-.9])
    b=np.array([-.9,+.9])
    assert np.array_equal(H@a,H@b)
    assert not np.isclose(a[0],b[0])
    return {
        "sensor4_readout_world_A":float(H@a),
        "sensor4_readout_world_B":float(H@b),
        "target_a1_world_A":float(a[0]),
        "target_a1_world_B":float(b[0]),
        "both_worlds_exactly_observationally_indistinguishable":True,
        "calibrated_mixture_does_not_identify_a1":True,
    }
