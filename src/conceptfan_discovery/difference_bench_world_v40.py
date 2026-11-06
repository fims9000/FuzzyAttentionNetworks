"""V40 new devices and nonlinear pre-readout sensor family, TRAIN/TEST separated.

Research simulator, NOT a clinically implementable intervention. All exact
mixing H, slow/fast latent S, source-aligned artifact U, and readout errors
remain known to simulator ONLY for drawing observed on/off measurements and
external post-hoc audits; NONE are input to source fitting/risk baselines.
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit

from conceptfan_discovery.phase_pulse_benchmark_v31 import phase_pulse_world
from conceptfan_discovery.paired_isolation_world_v39 import (
    _random_H,_ar_process,masks_v39,observed_paired_difference_v39,
)

REGISTERED_DEVICES={
    20261407:"linear_AR",
    20261409:"linear_AR",
    20261421:"nonlinear_switching",
    20261423:"nonlinear_switching",
}
VIEWS=((4,5),(6,7),(8,9))
STRESSES=("complete","MCAR20","MCAR40",
          "synchronized_block8","missing_reference")
CASES=(
    ("healthy_energy","energy","none","none"),
    ("common_U8_matched_ON_OFF","energy","none","common8"),
    ("common_U4_matched_ON_OFF","energy","none","common4"),
    ("OFF_only_U4_drift","energy","none","off_drift"),
    ("OFF_source_leak_50pct","energy","none","leak"),
    ("OFF_reference_unavailable","energy","none","no_reference"),
    ("constant_offset_both_2","energy","none","offset"),
    ("K2_complete_no_extra_source","complete","none","none"),
    ("VALID2_known_OFF_drift","energy","off_drift","off_drift"),
)


def _source_laws(rng,n,family):
    if family=="linear_AR":
        slow=_ar_process(rng,n,.77)
        fast=_ar_process(rng,n,-.27)
    elif family=="nonlinear_switching":
        slow=np.empty((n,16),dtype=np.float64)
        fast=np.empty_like(slow)
        slow[:,0]=rng.normal(size=n)
        fast[:,0]=rng.normal(size=n)
        # A genuine nonlinear, non-Gaussian temporal process. Per-episode
        # volatility differs, with occasional transient impulses.
        gain=rng.uniform(.70,1.25,size=n)
        for t in range(1,16):
            inn=rng.standard_t(df=5,size=n)/np.sqrt(5./3.)
            impulse=(rng.random(n)<.045)*rng.normal(0,1.35,size=n)
            slow[:,t]=(
                .63*slow[:,t-1]+.24*np.tanh(1.6*slow[:,t-1])
                +.39*gain*inn+impulse
            )
            fast[:,t]=(
                -.29*fast[:,t-1]+.70*gain*
                rng.standard_t(df=6,size=n)/np.sqrt(6./4.)
            )
        # Switch the volatility of selected subjects mid-episode, rather
        # than a stationary Gaussian source fitted in V36/V37/V38/V39.
        flag=(rng.random(n)<.45)
        slow[flag,8:]*=1.30
        fast[flag,8:]*=.80
    else:
        raise ValueError("Unregistered V40 nonlinear dynamics family")
    episode_gain=rng.uniform(.65,1.55,size=(n,1))
    return np.stack((slow*episode_gain,fast*episode_gain),axis=-1)


def _nonlinear_transfer(z,family):
    if family=="linear_AR":
        return z
    if family=="nonlinear_switching":
        # Nonlinearity BEFORE source/background separation: g(a+b)-g(a)
        # != b. Its effect increases with unknown common artifact amplitude.
        return z+.035*z*np.abs(z)
    raise ValueError("Unregistered instrument transfer family")


def generate_device_v40(
    *,device_seed:int,n_train=1000,n_valid=800,n_test=600,
):
    if (device_seed not in REGISTERED_DEVICES or min(n_train,n_valid,n_test)<100
        or n_valid%2):
        raise ValueError("Only preregistered V40 devices with disjoint splits")
    family=REGISTERED_DEVICES[device_seed]
    base=phase_pulse_world(
        seed=device_seed+331,n_train=n_train,
        n_valid=n_valid,n_test=n_test)
    rng=np.random.default_rng(device_seed+52063)
    n=len(base["x"])
    latent=_source_laws(rng,n,family)
    artifact=.35*_ar_process(rng,n,.77)
    mixing=tuple(_random_H(rng) for _ in VIEWS)
    e_on=rng.normal(0,.19,size=(n,16,6))
    e_off=rng.normal(0,.19,size=(n,16,6))
    original=base["x"].copy()
    original[:,:,10]=rng.normal(0,.08,size=(n,16))
    original[:,:,11]=rng.normal(size=(n,16))
    true_slow=latent[:,:,0]
    energy=np.mean((true_slow-true_slow.mean(axis=1,keepdims=True))**2,axis=1)
    known=base["true_known_CONCEPTS_AUDIT_ONLY"].mean(axis=1)
    z=3.7*(known[:,0]-.5)-3.6*(known[:,1]-.5)-.10
    coin=rng.random(n)
    y={
        "energy":(coin<expit(z+2.2*(energy-.65))).astype(np.int64),
        "complete":(coin<expit(z)).astype(np.int64),
    }
    simulator={
        "family":family,
        "base":original,
        "mixing":mixing,
        "hidden_source":latent,
        "hidden_artifact":artifact,
        "on_readout_error":e_on,
        "off_readout_error":e_off,
    }
    return {
        "simulator_AUDIT_ONLY":simulator,
        "family":family,
        "device_seed":device_seed,
        "train":base["train"],"valid1":base["valid1"],
        "valid2":base["valid2"],"test":base["test"],
        "outcomes":y,
        "true_slow_source_energy_POSTFIT_AUDIT_ONLY":energy,
        "mixing_not_exposed_to_concept_learner":True,
        "new_source_dynamics_family":family,
        "OFF_is_physical_source_isolation_EXPERIMENTAL_ASSUMPTION":True,
        "instrument_transfer_is_pre_readout_nonlinear":
            family=="nonlinear_switching",
    }


def render_observed_v40(world,indices,*,fault:str,seed:int):
    """SIMULATOR-ONLY physically specified intervention, NOT an inference API.

    The estimator sees only returned on/off sensor tensors and masks. Raw
    simulator H/S/U data are never passed to fit_centered_threeview_v37.
    """
    sim=world["simulator_AUDIT_ONLY"]
    family=sim["family"]
    indices=np.asarray(indices,dtype=int)
    n=len(indices)
    source=sim["hidden_source"][indices]
    base_U=sim["hidden_artifact"][indices]
    on=sim["base"][indices].copy()
    off=sim["base"][indices].copy()
    rng=np.random.default_rng(seed)
    if fault in ("common8","common4","off_drift"):
        extra=_ar_process(rng,n,.77)
        extra*={"common8":2.8,"common4":1.4,"off_drift":1.4}[fault]
    elif fault in ("none","offset","leak","no_reference"):
        extra=np.zeros_like(base_U)
    else:
        raise ValueError("Unregistered V40 physical/device fault")
    for k,(_,_) in enumerate(VIEWS):
        idx=4+2*k
        H=sim["mixing"][k]
        # Family uses externally specified instrument transfer, including
        # actual nonlinear mixing prior to any available measurement.
        u_on=base_U+(
            extra if fault in ("common8","common4") else 0)
        u_off=base_U+(
            extra if fault in ("common8","common4","off_drift") else 0)
        source_on=source@H.T
        leak=(.5*source[:,:,0,None]*H[:,0][None,None,:]
              if fault=="leak" else 0.)
        on_pre=source_on+u_on[:,:,None]*H[:,0][None,None,:]
        off_pre=leak+u_off[:,:,None]*H[:,0][None,None,:]
        on[:,:,idx:idx+2]=(
            _nonlinear_transfer(on_pre,family)+
            sim["on_readout_error"][indices,:,2*k:2*k+2]
        ).astype(np.float32)
        off[:,:,idx:idx+2]=(
            _nonlinear_transfer(off_pre,family)+
            sim["off_readout_error"][indices,:,2*k:2*k+2]
        ).astype(np.float32)
    if fault=="offset":
        on[:,:,4:10]+=2.
        off[:,:,4:10]+=2.
    np.testing.assert_array_equal(on[:,:,:4],off[:,:,:4])
    return on,off


def masks_v40(on,off,stress,*,device_seed:int,mask_seed:int,
              known_TEST_first4:np.ndarray|None=None):
    if stress not in STRESSES:
        raise ValueError("Unregistered V40 missingness stress")
    mo,mr=masks_v39(
        on,off,stress,
        on_seed=mask_seed+device_seed%101,
        off_seed=mask_seed+991+device_seed%101)
    if known_TEST_first4 is not None:
        arr=np.asarray(known_TEST_first4,dtype=bool)
        if arr.shape!=mo[:,:,:4].shape:
            raise ValueError("Shared same-patient original K2 mask mismatch")
        mo[:,:,:4]=arr
    return mo,mr


def measured_difference_v40(on,off,m_on,m_off):
    # Exact preexisting audited mask AND, no hidden observation,
    # no synthetic source reconstruction or latent oracle.
    return observed_paired_difference_v39(on,off,m_on,m_off)
