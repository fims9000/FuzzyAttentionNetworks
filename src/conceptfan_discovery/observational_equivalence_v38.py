"""Constructive observational non-identifiability with an unmonitored common artifact.

This module is a MATHEMATICAL COUNTEREXAMPLE, not an extra V38 TEST
experiment, and deliberately never enters the architecture training or
threshold tuning. A source-aligned common additive instrument error can
produce exactly the same observed views and independent sentinel readings
as a genuine perturbation of the latent source, while the *physical source*
energy differs. No function f(X, sentinel, prior calibration) can distinguish
the worlds without fresh information about current device errors.

This simple equality is not a novel identifiability theorem; it makes the
operational scope of the V38 external negative-control assumption precise.
"""
from __future__ import annotations
import numpy as np


def source_vs_unmonitored_aligned_error(*,seed:int=20263870,n:int=240):
    if n<50:
        raise ValueError("At least 50 constructively compared episodes")
    rng=np.random.default_rng(seed)
    slow=np.zeros((n,16))
    fast=np.zeros((n,16))
    slow[:,0]=rng.normal(size=n)
    fast[:,0]=rng.normal(size=n)
    for t in range(1,16):
        slow[:,t]=.77*slow[:,t-1]+np.sqrt(1-.77**2)*rng.normal(size=n)
        fast[:,t]=-.27*fast[:,t-1]+np.sqrt(1-.27**2)*rng.normal(size=n)
    s=np.stack((slow,fast),axis=-1)
    delta=1.8*rng.choice([-1.,1.],size=(n,1))*(
        np.r_[np.ones(8),-np.ones(8)])[None,:]
    matrices=(
        np.array([[1.14,.51],[.22,1.18]]),
        np.array([[.67,-.74],[1.19,.23]]),
        np.array([[.93,.19],[-.39,.81]]),
    )
    noise=[rng.normal(0,.21,size=(n,16,2)) for _ in matrices]
    physical=np.zeros((n,16,12),dtype=np.float64)
    artifact=np.zeros_like(physical)
    common_known=rng.normal(size=(n,16,4))
    common_sentinel=rng.normal(0,.08,size=(n,16))
    physical[:,:,:4]=artifact[:,:,:4]=common_known
    physical[:,:,10]=artifact[:,:,10]=common_sentinel
    physical[:,:,11]=artifact[:,:,11]=rng.normal(size=(n,16))
    for j,H in enumerate(matrices):
        lo=4+j*2
        # World 1: the actual source S0 is changed, independent device error.
        p=(s+np.stack((delta,np.zeros_like(delta)),axis=-1))@H.T+noise[j]
        # World 2: source S0 is unchanged; device error is changed in
        # exactly the corresponding source-loading direction H[:,0]*delta.
        e=s@H.T+noise[j]+delta[:,:,None]*H[:,0][None,None,:]
        physical[:,:,lo:lo+2]=p
        artifact[:,:,lo:lo+2]=e
    actual_physical_energy=((slow+delta-(slow+delta).mean(axis=1)[:,None])**2).mean(axis=1)
    actual_noise_world_energy=((slow-slow.mean(axis=1)[:,None])**2).mean(axis=1)
    return {
        "physical_world_observed_X":physical,
        "artifact_world_observed_X":artifact,
        "physical_source_energy_with_perturbation":actual_physical_energy,
        "source_energy_without_perturbation":actual_noise_world_energy,
        "sentinel_cannot_see_aligned_artifact":True,
        "calibration_from_previous_time_can_be_identical":True,
        "known_external_actuator_is_not_run_at_TEST":True,
        "use_of_true_H_is_SIMULATOR_COUNTEREXAMPLE_ONLY":True,
    }
