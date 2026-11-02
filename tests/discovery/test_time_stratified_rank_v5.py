"""Time-conditional missingness rank contract and falsification controls."""
from __future__ import annotations

import numpy as np
import pytest

from conceptfan_discovery.time_stratified_rank_v5 import (
    NotEstimableRank, fit_time_stratified_rank,
)


def _observable_linear_process(
    k:int, *,
    n:int=260,t:int=30,d:int=6,seed:int=991,
    missingness:str="mcar",
)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    rng=np.random.default_rng(int(seed)+int(k))
    loading,_=np.linalg.qr(rng.normal(size=(d,k)))
    z=np.zeros((n,t,k),dtype=float)
    z[:,0]=rng.normal(size=(n,k))
    for hour in range(1,t):
        z[:,hour]=.73*z[:,hour-1]+rng.normal(scale=.26,size=(n,k))
    x=z@loading.T+rng.normal(scale=.055,size=(n,t,d))
    if missingness=="mcar":
        p=np.full((1,t,d),.72)
    elif missingness=="time":
        hour=np.arange(t)[None,:,None]
        channel=np.arange(d)[None,None,:]
        p=np.clip(.62+.17*np.sin(hour*.38+channel*.7),.32,.87)
    elif missingness=="state":
        # MNAR selection intentionally depends on unobserved-for-training
        # magnitude. Not a recoverability guarantee for any rank method.
        p=np.clip(.53-.22*np.tanh(np.abs(x)),.18,.75)
    elif missingness=="all":
        p=np.ones((1,t,d))
    else:
        raise ValueError(missingness)
    mask=(rng.random(size=(n,t,d))<p).astype(float)
    return x*mask,mask,z


@pytest.mark.parametrize("k",[2,4])
def test_detects_linear_observable_rank_under_time_independent_mcar(k):
    x,m,_=_observable_linear_process(k,missingness="mcar")
    rank=fit_time_stratified_rank(
        x,m,permutations=55,min_pairs_per_hour=12,min_time_strata=3,seed=31
    )
    assert rank.rank==k
    assert rank.source_direction.shape==(6,6)
    np.testing.assert_allclose(rank.source_direction.T@rank.source_direction,
                               np.eye(6),atol=1e-9)
    assert rank.summary()["no_human_or_synthetic_concept_labels"] is True


def test_deterministic_hour_specific_observation_policy_preserves_rank():
    x,m,_=_observable_linear_process(2,missingness="time")
    a=fit_time_stratified_rank(
        x,m,permutations=55,min_pairs_per_hour=12,min_time_strata=3,seed=31,
    )
    b=fit_time_stratified_rank(
        x.copy(),m.copy(),permutations=55,min_pairs_per_hour=12,
        min_time_strata=3,seed=31,
    )
    assert a.rank==2
    assert b.rank==a.rank
    np.testing.assert_array_equal(a.singular_values,b.singular_values)


def test_missing_values_cannot_leak_into_within_hour_centering():
    x,m,_=_observable_linear_process(2,missingness="time")
    after=x.copy()
    after[m==0]=1e10
    a=fit_time_stratified_rank(
        x,m,permutations=30,min_pairs_per_hour=12,min_time_strata=3,seed=117,
    )
    b=fit_time_stratified_rank(
        after,m,permutations=30,min_pairs_per_hour=12,min_time_strata=3,seed=117,
    )
    assert a.rank==b.rank
    np.testing.assert_allclose(a.singular_values,b.singular_values,atol=1e-11)


def test_common_hourly_measurement_policy_shift_is_not_a_patient_latent_factor():
    rng=np.random.default_rng(116)
    n,t,d=260,30,6
    trend=np.linspace(-2,2,t)[None,:,None]
    channel_scale=np.arange(1,d+1)[None,None,:]*.12
    noise=rng.normal(scale=.15,size=(n,t,d))
    x=noise+trend*channel_scale
    m=np.ones_like(x)
    out=fit_time_stratified_rank(
        x,m,permutations=55,min_pairs_per_hour=12,min_time_strata=3,seed=31,
    )
    same_patients_without_common_trend=fit_time_stratified_rank(
        noise,m,permutations=55,min_pairs_per_hour=12,min_time_strata=3,seed=31,
    )
    # A single 5%-level permutation test can produce a false discovery even
    # under pure white noise. The exact, falsifiable property is that a
    # deterministic common time-policy drift cannot ALTER the spectrum.
    assert out.rank==same_patients_without_common_trend.rank
    np.testing.assert_allclose(
        out.singular_values,same_patients_without_common_trend.singular_values,
        atol=1e-13,rtol=1e-10,
    )


def test_rank_abstains_when_matched_hourly_observations_are_missing():
    x,m,_=_observable_linear_process(2,missingness="all")
    m[:,:,:]=0.
    m[:5,:,:]=1.
    with pytest.raises(NotEstimableRank,match="Positive-lag channels lack"):
        fit_time_stratified_rank(
            x*m,m,permutations=30,min_pairs_per_hour=12,min_time_strata=3,
        )


def test_invalid_missingness_and_lag_configuration_rejected():
    x,m,_=_observable_linear_process(2,missingness="mcar")
    with pytest.raises(ValueError):
        fit_time_stratified_rank(x,m,lags=(1,1))
    with pytest.raises(ValueError):
        fit_time_stratified_rank(x,m,permutations=5)
    with pytest.raises(ValueError):
        fit_time_stratified_rank(x,np.clip(m+.3,0,1))
