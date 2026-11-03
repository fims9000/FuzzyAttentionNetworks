"""Predictive-state simulation positive/negative controls, no oracle fitting."""
from __future__ import annotations

import numpy as np
import pytest

from conceptfan_discovery.identifiability_stress_v11 import (
    simulate_linear_dynamics,
)
from conceptfan_discovery.lagged_rank import _center,_matrix_from_centered


@pytest.mark.parametrize("k",[3,5,7])
@pytest.mark.parametrize("regime",["distinct","degenerate","time_shuffled","iid_temporal_null"])
def test_generator_respects_train_history_and_latent_count(k,regime):
    x,m,basis,receipt=simulate_linear_dynamics(
        k=k,regime=regime,seed=730+k,n_patients=180,
    )
    assert x.shape==(180,36,10)
    assert m.shape==x.shape
    assert basis.shape==(10,k)
    assert receipt["n_true_factors_oracle_evaluation_only"]==k
    assert receipt["oracle_latent_traces_never_used_for_fit"] is True
    assert np.isfinite(x).all()
    assert np.array_equal((m==1),m.astype(bool))
    assert (x[m==0]==0).all()


def test_temporal_shuffle_reduces_positive_lag_predictability():
    x,m,_,_=simulate_linear_dynamics(
        k=3,regime="distinct",seed=830,n_patients=450,
        observed_fraction=.96,
    )
    centered=_center(x,m)
    signal=np.linalg.svd(_matrix_from_centered(centered,m,(1,2,3)),
                         compute_uv=False)
    rng=np.random.default_rng(319)
    shuffled=x.copy()
    masks=m.copy()
    for i in range(len(x)):
        idx=rng.permutation(x.shape[1])
        shuffled[i],masks[i]=x[i,idx],m[i,idx]
    null=np.linalg.svd(
        _matrix_from_centered(_center(shuffled,masks),masks,(1,2,3)),
        compute_uv=False,
    )
    # Within-patient shuffling destroys ordered transitions but preserves
    # finite-window patient means, so it need NOT halve the operator norm.
    assert signal[0]>null[0]
    # Fully independent episode-to-episode future pairing additionally
    # removes patient-specific temporal links while retaining each
    # patient's own marginal serial structure.
    other=np.linalg.svd(
        _matrix_from_centered(centered,m,(1,2,3),
                              future_permutation=rng.permutation(len(x))),
        compute_uv=False,
    )
    assert signal[0]>2*other[0]


def test_distinct_dynamics_and_degeneracy_use_exact_planted_decay_regimes():
    _,_,_,first=simulate_linear_dynamics(
        k=5,regime="distinct",seed=31,n_patients=180,
    )
    _,_,_,second=simulate_linear_dynamics(
        k=5,regime="degenerate",seed=31,n_patients=180,
    )
    d0,d1=first["rate_values_used_only_in_generator"],second["rate_values_used_only_in_generator"]
    assert d0[0]!=d0[1]
    assert d1[0]==d1[1]


@pytest.mark.parametrize("bad",[(2,"distinct"),(8,"distinct"),(3,"unknown")])
def test_invalid_oracle_generators_rejected(bad):
    with pytest.raises(ValueError):
        simulate_linear_dynamics(k=bad[0],regime=bad[1],seed=1)


def test_strict_iid_time_null_retains_cross_sensor_structure_but_not_predictable_lags():
    x,m,emission,_=simulate_linear_dynamics(
        k=3,regime="iid_temporal_null",seed=47,n_patients=450,
        observed_fraction=.96,
    )
    y,mm,other_emission,_=simulate_linear_dynamics(
        k=3,regime="distinct",seed=47,n_patients=450,
        observed_fraction=.96,
    )
    # Exactly the same mixing matrix makes it a legitimate control,
    # rather than reducing signal by hiding channels or changing sensors.
    np.testing.assert_allclose(emission,other_emission)
    iid=np.linalg.svd(_matrix_from_centered(_center(x,m),m,(1,2,3)),
                      compute_uv=False)[0]
    source=np.linalg.svd(_matrix_from_centered(_center(y,mm),mm,(1,2,3)),
                         compute_uv=False)[0]
    assert source>2*iid
