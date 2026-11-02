"""Counterexamples: predictive observation rank is not the number of physical causes.

These are POSITIVE tests of method honesty, not model failures: even with
infinite data, a latent process that never influences measured channels is
not identifiable; temporally persistent sensor artifacts can appear as
additional dynamical modes.
"""
from __future__ import annotations

import numpy as np

from conceptfan_discovery.lagged_rank import fit_lagged_predictive_rank


def _observations_with_two_predictable_modes(seed:int=29):
    rng=np.random.default_rng(seed)
    n,t,d=260,32,8
    mixing,_=np.linalg.qr(rng.normal(size=(d,2)))
    z=np.empty((n,t,2))
    z[:,0]=rng.normal(size=(n,2))
    for hour in range(1,t):
        z[:,hour]=.72*z[:,hour-1]+rng.normal(scale=.22,size=(n,2))
    observed=z@mixing.T+rng.normal(scale=.05,size=(n,t,d))
    return rng, observed, np.ones_like(observed), z


def test_three_fully_unobserved_dynamic_factors_cannot_be_discovered():
    rng,x,mask,z=_observations_with_two_predictable_modes()
    # A physically distinct system has *five* latent processes:
    # observed Z1,Z2 and unobserved U1,U2,U3. U is dynamically active
    # but independent and B_U=0, so X and Y distributions stay exactly
    # the same. No observable-data algorithm can recover U.
    u=np.empty((len(x),x.shape[1],3))
    u[:,0]=rng.normal(size=(len(x),3))
    for t in range(1,x.shape[1]):
        u[:,t]=.8*u[:,t-1]+rng.normal(scale=.2,size=(len(x),3))
    true_physical_state=np.concatenate([z,u],axis=2)
    assert true_physical_state.shape[-1]==5
    measured=fit_lagged_predictive_rank(
        x,mask,permutations=55,patient_bootstraps=10,seed=13,
    )
    assert measured.rank==2  # observable rank, NOT physical K=5


def test_temporally_persistent_sensor_artifact_changes_observable_rank():
    rng,x,mask,z=_observations_with_two_predictable_modes(33)
    artifact=np.empty((len(x),x.shape[1]))
    artifact[:,0]=rng.normal(scale=.9,size=len(x))
    for t in range(1,x.shape[1]):
        artifact[:,t]=.83*artifact[:,t-1]+rng.normal(scale=.15,size=len(x))
    # A persistent artifact in an otherwise independent sensor has
    # its own predictive covariance mode. It is observable, but its
    # inclusion as a "physiological concept" would be false.
    contaminated=x.copy()
    contaminated[:,:,7]+=artifact
    initial=fit_lagged_predictive_rank(
        x,mask,permutations=55,patient_bootstraps=10,seed=19,
    )
    modified=fit_lagged_predictive_rank(
        contaminated,mask,permutations=55,patient_bootstraps=10,seed=19,
    )
    assert initial.rank==2
    assert modified.rank==3
    assert z.shape[-1]==2  # no new physiological latent process
