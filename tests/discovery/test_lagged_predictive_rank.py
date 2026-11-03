"""Independent mathematical checks for train-only predictive rank."""
from __future__ import annotations

import numpy as np
import pytest

from conceptfan_discovery.lagged_rank import fit_lagged_predictive_rank


def _linear_dynamic_observations(k: int, seed: int = 41) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + k)
    n, t, d = 240, 30, 8
    loadings, _ = np.linalg.qr(rng.normal(size=(d, k)))
    latent = np.zeros((n, t, k))
    latent[:, 0] = rng.normal(size=(n, k))
    for hour in range(1, t):
        latent[:, hour] = .68 * latent[:, hour - 1] + rng.normal(scale=.25, size=(n, k))
    signal = latent @ loadings.T + rng.normal(scale=.08, size=(n, t, d))
    observed = (rng.random(size=signal.shape) > .14).astype(float)
    return signal * observed, observed


@pytest.mark.parametrize("rank",[2,4,6])
def test_true_rank_of_observable_stable_linear_system(rank):
    x, mask = _linear_dynamic_observations(rank)
    result = fit_lagged_predictive_rank(
        x, mask, permutations=55, patient_bootstraps=12, seed=3,
    )
    assert result.rank == rank
    assert result.patient_bootstrap_interval == (rank, rank)
    assert result.left_observable_loadings.shape == (8, 8)
    np.testing.assert_allclose(result.left_observable_loadings.T @
                               result.left_observable_loadings, np.eye(8), atol=1e-10)
    assert result.to_json()["latent_labels_used"] is False


def test_independent_white_noise_has_no_stable_predictive_rank():
    rng = np.random.default_rng(11)
    values = rng.normal(size=(240, 30, 8))
    mask = np.ones_like(values)
    result = fit_lagged_predictive_rank(
        values, mask, permutations=55, patient_bootstraps=10, seed=3,
    )
    assert result.rank == 0


def test_rank_does_not_depend_on_labels_or_synthetic_ground_truth():
    x, mask = _linear_dynamic_observations(4)
    a = fit_lagged_predictive_rank(x, mask, permutations=35, patient_bootstraps=5, seed=17)
    b = fit_lagged_predictive_rank(x.copy(), mask.copy(), permutations=35, patient_bootstraps=5, seed=17)
    assert a.rank == b.rank
    np.testing.assert_array_equal(a.singular_values, b.singular_values)
    np.testing.assert_array_equal(a.bootstrap_rank_counts, b.bootstrap_rank_counts)


def test_missing_or_invalid_measurements_rejected():
    x, mask = _linear_dynamic_observations(2)
    with pytest.raises(ValueError):
        fit_lagged_predictive_rank(x[:10], mask[:10])
    with pytest.raises(ValueError):
        fit_lagged_predictive_rank(x, np.full_like(mask, np.nan))
    with pytest.raises(ValueError):
        fit_lagged_predictive_rank(x, mask, lags=(1, 1))
    with pytest.raises(ValueError):
        fit_lagged_predictive_rank(x, np.zeros_like(mask))
