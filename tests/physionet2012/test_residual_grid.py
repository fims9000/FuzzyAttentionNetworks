"""Sanity tests for the two-way conditional residual bootstrap."""
import numpy as np
import pytest

from conceptfan_realdata.residual_grid import hierarchical_delta_ci


def test_zero_gain_for_identical_predictions_and_deterministic():
    rng = np.random.default_rng(12)
    y = np.array([0, 1] * 20)
    baseline = np.clip(rng.uniform(size=40), .01, .99)
    copies = np.stack([baseline] * 4)
    a = hierarchical_delta_ci(y, baseline, copies, repetitions=30, seed=4)
    b = hierarchical_delta_ci(y, baseline, copies, repetitions=30, seed=4)
    assert a == b
    np.testing.assert_allclose(a["delta_auprc_ci95"], [0, 0], atol=1e-13)
    np.testing.assert_allclose(a["delta_nll_ci95"], [0, 0], atol=1e-13)


def test_predictive_signal_detectable():
    rng = np.random.default_rng(34)
    y = np.array([0, 1] * 30)
    baseline = np.full(60, .5)
    models = np.stack([np.where(y == 1, .9, .1) +
                       rng.normal(scale=.015, size=60) for _ in range(3)]).clip(.01, .99)
    result = hierarchical_delta_ci(y, baseline, models, repetitions=30)
    assert result["delta_auprc_mean"] > .45
    assert result["delta_auprc_ci95"][0] > 0
    assert result["delta_nll_mean"] < 0


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        hierarchical_delta_ci(np.array([0, 1] * 5), np.ones(10), np.ones((3, 10)))
    with pytest.raises(ValueError):
        hierarchical_delta_ci(np.ones(10, dtype=int), np.full(10, .5), np.full((3, 10), .5))
