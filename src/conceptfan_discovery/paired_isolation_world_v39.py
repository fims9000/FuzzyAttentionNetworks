"""V39 paired ON/OFF source isolation simulator with source-aligned artifact.

This is a SYNTHETIC experimental device, not a clinically feasible
procedure. A known physical switch disables the source pathway in OFF
while a contemporaneous common measurement artifact remains.

The generator-only matrices H and source traces MUST NOT be passed to
the blind source algorithm or outcome-fitting stages. They exist only
for fault construction and post-TEST evaluation of source energy.
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit

from conceptfan_discovery.phase_pulse_benchmark_v31 import phase_pulse_world

DEVICE_SEEDS=(20261317,20261319)
VIEWS=((4,5),(6,7),(8,9))
STRESSES=("complete","MCAR20","MCAR40",
          "independent_block8","synchronized_block8",
          "missing_reference","missing_view_C")
CASES=(
    ("healthy","energy","none","none"),
    ("TEST_aligned_artifact_4x","energy","none","shared4"),
    ("TEST_aligned_artifact_8x","energy","none","shared8"),
    ("TEST_OFF_artifact_drift","energy","none","off_drift"),
    ("TEST_source_isolation_leak_50pct","energy","none","leak"),
    ("TEST_reference_missing","energy","none","no_ref"),
    ("TEST_A_pol_flip_on","energy","none","flip_on_A"),
    ("TEST_offset_plus2_both","energy","none","offset"),
    ("K2_complete_dictionary","complete","none","none"),
    ("VALID2_known_OFF_drift","energy","off_drift","off_drift"),
)


def _random_H(rng):
    a,_=np.linalg.qr(rng.normal(size=(2,2)))
    b,_=np.linalg.qr(rng.normal(size=(2,2)))
    return a@np.diag(rng.uniform(.70,1.35,size=2))@b.T


def _ar_process(rng,n,rho):
    z=np.empty((n,16))
    z[:,0]=rng.standard_normal(n)
    for t in range(1,16):
        z[:,t]=rho*z[:,t-1]+np.sqrt(1-rho*rho)*rng.standard_normal(n)
    return z


def generate_paired_isolation_world_v39(
    *,device_seed:int,n_train:int=1000,n_valid:int=800,n_test:int=600,
):
    if device_seed not in DEVICE_SEEDS or n_valid%2 or min(n_train,n_valid,n_test)<100:
        raise ValueError("Unregistered V39 independent device/split")
    base=phase_pulse_world(
        seed=device_seed+331,n_train=n_train,
        n_valid=n_valid,n_test=n_test)
    rng=np.random.default_rng(device_seed+52063)
    n=len(base["x"])
    slow=_ar_process(rng,n,.77)
    fast=_ar_process(rng,n,-.27)
    episode_gain=rng.uniform(.65,1.65,size=(n,1))
    slow*=episode_gain
    fast*=episode_gain
    latent=np.stack((slow,fast),axis=-1)
    U=.35*_ar_process(rng,n,.77)
    H=tuple(_random_H(rng) for _ in VIEWS)

    on=base["x"].copy()
    off=base["x"].copy()
    for (i,j),mix in zip(VIEWS,H):
        common=U[:,:,None]*mix[:,0][None,None,:]
        on[:,:,i:j+1]=(latent@mix.T+common+
              rng.normal(0,.19,size=(n,16,2))).astype(np.float32)
        off[:,:,i:j+1]=(common+
              rng.normal(0,.19,size=(n,16,2))).astype(np.float32)
    # Channel10 is NOT wired to U; even V38 sentinel cannot observe U.
    neutral=rng.normal(0,.08,size=(n,16))
    on[:,:,10]=neutral
    off[:,:,10]=neutral
    on[:,:,11]=off[:,:,11]=rng.normal(size=(n,16))
    known=base["true_known_CONCEPTS_AUDIT_ONLY"].mean(axis=1)
    base_logit=3.7*(known[:,0]-.5)-3.6*(known[:,1]-.5)-.10
    centered=slow-slow.mean(axis=1,keepdims=True)
    energy=np.mean(centered**2,axis=1)
    coin=rng.random(n)
    y={
        "energy":(coin<expit(base_logit+2.2*(energy-.65))).astype(np.int64),
        "complete":(coin<expit(base_logit)).astype(np.int64),
    }
    return {
        "on":on,"off":off,"outcomes":y,
        "train":base["train"],"valid1":base["valid1"],
        "valid2":base["valid2"],"test":base["test"],
        "hidden_mixing_AUDIT_ONLY":[m.tolist() for m in H],
        "hidden_S_AUDIT_ONLY":latent,
        "true_source_fluctuation_power_AUDIT_ONLY":energy,
        "constant_shared_artifact_U_is_invisible_to_sentinel10":True,
        "physical_ON_OFF_source_path_known_to_simulator_only":True,
        "unknown_H_not_passed_to_source_estimator":True,
    }


def apply_paired_fault_v39(
    on,off,*,mode:str,seed:int,
    hidden_H_SIMULATOR_ONLY,
    hidden_S_SIMULATOR_ONLY,
):
    on=np.asarray(on,dtype=np.float32)
    off=np.asarray(off,dtype=np.float32)
    if on.shape!=off.shape or on.ndim!=3 or on.shape[1:]!=(16,12):
        raise ValueError("Registered paired 16x12 inputs only")
    a,b=on.copy(),off.copy()
    rng=np.random.default_rng(seed)
    if mode=="none":
        return a,b
    if mode=="no_ref":
        # The missingness mask, NOT fake reference values, is the abstention.
        return a,b
    if mode=="offset":
        a[:,:,4:10]+=2.
        b[:,:,4:10]+=2.
    elif mode=="flip_on_A":
        a[:,:,4:6]*=-1
    elif mode in ("shared4","shared8","off_drift"):
        u=_ar_process(rng,len(a),.77)
        amplitude={"shared4":1.4,"shared8":2.8,"off_drift":1.4}[mode]
        u*=amplitude
        for (lo,hi),mix in zip(VIEWS,hidden_H_SIMULATOR_ONLY):
            shared=(u[:,:,None]*np.asarray(mix)[:,0][None,None,:]).astype(np.float32)
            if mode=="off_drift":
                b[:,:,lo:hi+1]+=shared
            else:
                a[:,:,lo:hi+1]+=shared
                b[:,:,lo:hi+1]+=shared
    elif mode=="leak":
        z=np.asarray(hidden_S_SIMULATOR_ONLY)
        if z.shape!=(len(a),16,2):
            raise ValueError("Generator-only source trace mismatch")
        for (lo,hi),mix in zip(VIEWS,hidden_H_SIMULATOR_ONLY):
            leaked=.5*z[:,:,0,None]*np.asarray(mix)[:,0][None,None,:]
            b[:,:,lo:hi+1]+=leaked.astype(np.float32)
    else:
        raise ValueError("Unregistered ON/OFF fault")
    np.testing.assert_array_equal(a[:,:,:4],on[:,:,:4])
    np.testing.assert_array_equal(b[:,:,:4],off[:,:,:4])
    return a,b


def masks_v39(on,off,mode,*,on_seed:int,off_seed:int):
    on=np.asarray(on);off=np.asarray(off)
    if on.shape!=off.shape or on.ndim!=3 or on.shape[1:]!=(16,12) or mode not in STRESSES:
        raise ValueError("Registered paired 16x12/missingness only")
    n=len(on)
    rng=np.random.default_rng(on_seed)
    mo=np.ones(on.shape,dtype=bool)
    mo[:,:,:4]=np.random.default_rng(on_seed+71).random((n,16,4))>=.20
    # Reference instrument itself samples 95% of each unknown channel.
    mr=np.ones(off.shape,dtype=bool)
    mr[:,:,4:10]=np.random.default_rng(off_seed).random((n,16,6))>=.05
    if mode in ("MCAR20","MCAR40"):
        frac=.2 if mode=="MCAR20" else .4
        mo[:,:,4:10]=rng.random((n,16,6))>=frac
    elif mode=="independent_block8":
        starts=rng.integers(0,9,size=(n,6))
        for i in range(n):
            for j in range(6):
                mo[i,starts[i,j]:starts[i,j]+8,4+j]=False
    elif mode=="synchronized_block8":
        starts=rng.integers(0,9,size=n)
        for i,j in enumerate(starts):
            mo[i,j:j+8,4:10]=False
    elif mode=="missing_reference":
        mr[:,:,4:10]=False
    elif mode=="missing_view_C":
        mo[:,:,8:10]=False
    return mo,mr


def observed_paired_difference_v39(on,off,on_mask,off_mask):
    """Mask intersection; NO inferred hidden values or source latent oracle."""
    on=np.asarray(on,dtype=np.float32)
    off=np.asarray(off,dtype=np.float32)
    mo=np.asarray(on_mask,dtype=bool)
    mr=np.asarray(off_mask,dtype=bool)
    if on.ndim!=3 or on.shape[1:]!=(16,12) or off.shape!=on.shape or (
        mo.shape!=on.shape or mr.shape!=on.shape):
        raise ValueError("Expected paired [N,16,12] observations and masks")
    if not np.isfinite(on[mo]).all() or not np.isfinite(off[mr]).all():
        raise ValueError("Observed readings cannot have NaN/Inf")
    available=mo&mr
    difference=np.where(available,on,0.)-np.where(available,off,0.)
    difference[~available]=0.
    return difference.astype(np.float32),available
