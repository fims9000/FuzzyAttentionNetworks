"""Sparse ICU lag observability must be reported, not hidden by imputation."""
from __future__ import annotations
import numpy as np
import pytest

from conceptfan_discovery.clinical_observability import (
    aggregate_hours,joint_pair_counts,support_selected_channels,
    expected_null_joint_pair_counts,support_selected_channels_across_subsamples,
)


def test_masked_hourly_aggregation_never_uses_missing_values():
    rng=np.random.default_rng(31)
    x=rng.normal(size=(100,24,3))
    mask=(rng.random(size=x.shape)>.2).astype(float)
    a,b=aggregate_hours(x,mask,4)
    copy=x.copy()
    copy[mask==0]=1000000.
    c,d=aggregate_hours(copy,mask,4)
    np.testing.assert_allclose(a,c,atol=1e-14)
    np.testing.assert_array_equal(b,d)
    assert a.shape==(100,6,3)
    assert not np.isnan(a).any()


def test_missing_clinical_measurement_is_not_fake_zero_covariance():
    rng=np.random.default_rng(32)
    mask=np.ones((100,24,3))
    # Entire third channel is never observed; it has no dynamic evidence.
    mask[:,:,2]=0.
    support=support_selected_channels(mask,["heart","lactate","unobserved"])
    assert support["all_selected_joint_lag_pairs_estimable"] is True
    assert support["selected_channels"]==["heart","lactate"]
    assert support["excluded_channels"]==["unobserved"]
    assert support["selected_pair_fraction_above_minimum"]==1.


def test_disjoint_observation_patterns_fail_joint_lag_support():
    mask=np.zeros((100,24,2),dtype=float)
    mask[:50,:,0]=1.
    mask[50:,:,1]=1.
    support=support_selected_channels(mask,["a","b"])
    assert support["channels_selected"]==1
    assert support["all_selected_joint_lag_pairs_estimable"] is False


def test_time_bins_reject_invalid_shapes():
    with pytest.raises(ValueError):
        aggregate_hours(np.ones((10,12,2)),np.ones((10,12,2)),6)
    with pytest.raises(ValueError):
        joint_pair_counts(np.zeros((10,2,2)),(1,2,3))
    with pytest.raises(ValueError):
        support_selected_channels(np.ones((10,24,2)),["only_one_name"])


def test_null_expected_pair_support_matches_monte_carlo_mean():
    rng=np.random.default_rng(18)
    mask=(rng.random(size=(80,10,3)) < .32).astype(float)
    predicted=expected_null_joint_pair_counts(mask,lags=(1,2))
    observed=[]
    for _ in range(100):
        swap=rng.permutation(80)
        count=np.stack([
            mask[:,:-lag,:].reshape(-1,3).T @
            mask[swap,lag:,:].reshape(-1,3)
            for lag in (1,2)
        ])
        observed.append(count)
    np.testing.assert_allclose(np.mean(observed,axis=0),predicted,rtol=.09,atol=2)


def test_null_support_guard_rejects_patient_aligned_but_unshuffleable_sources():
    # Each channel has sufficient within-patient lag pairs (same 10
    # patients), yet independently permuted future patients have very
    # few expected pairs. Do not pretend patient-shuffled null is valid.
    mask=np.zeros((200,15,2))
    mask[:10,:,0]=1.
    mask[:10,:,1]=1.
    observed=joint_pair_counts(mask,(1,2,3))
    assert observed.min()>=20
    predicted=expected_null_joint_pair_counts(mask,(1,2,3))
    assert predicted.min()<50
    result=support_selected_channels_across_subsamples(
        (mask,mask),["a","b"],min_pairs=20,min_expected_null_pairs=50,
    )
    assert result["selected_n"]==0
    assert result["all_subsamples_joints_adequate"] is False


def test_null_support_guard_retains_dense_measurements():
    mask=np.ones((150,12,3))
    result=support_selected_channels_across_subsamples(
        (mask,mask),["a","b","c"],min_pairs=20,min_expected_null_pairs=50,
    )
    assert result["selected_n"]==3
    assert result["all_subsamples_joints_adequate"]
    assert result["minimum_expected_shuffled_pair_count_within_selected"]>=50
