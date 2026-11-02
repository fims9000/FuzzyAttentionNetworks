"""Regression tests for CIs under dependent model pairs."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from conceptfan_realdata.model_pair_bootstrap import model_pair_bootstrap_ci

PAIRS = list(itertools.combinations(("m0", "m1", "m2", "m3"), 2))


def test_constant_pair_and_episode_score_stays_constant():
    for shape in [(len(PAIRS),), (len(PAIRS), 19)]:
        ci = model_pair_bootstrap_ci(np.full(shape, 0.17), PAIRS, 150, np.random.default_rng(3))
        np.testing.assert_allclose(ci, [0.17, 0.17], atol=1e-14)


def test_deterministic_and_nontrivial_variation():
    values = np.array([0.0, 0.2, 0.4, 0.7, 0.9, 1.0])
    a = model_pair_bootstrap_ci(values, PAIRS, 900, np.random.default_rng(123))
    b = model_pair_bootstrap_ci(values, PAIRS, 900, np.random.default_rng(123))
    assert a == b
    assert 0 <= a[0] < a[1] <= 1


def test_missing_pairs_or_duplicates_are_rejected():
    with pytest.raises(ValueError, match="all distinct"):
        model_pair_bootstrap_ci(np.arange(5), PAIRS[:-1], 100, np.random.default_rng(1))
    with pytest.raises(ValueError, match="duplicate model pair"):
        model_pair_bootstrap_ci(np.arange(6), PAIRS[:-1] + [("m1", "m0")], 100, np.random.default_rng(1))


def test_two_models_do_not_gain_fake_self_agreement():
    ci = model_pair_bootstrap_ci(np.array([0.12]), [("a", "b")], 100, np.random.default_rng(17))
    np.testing.assert_allclose(ci, [0.12, 0.12], atol=1e-14)


def test_episode_resampling_is_seeded():
    values = np.vstack([np.arange(20) / 20 + j / 20 for j in range(6)])
    a = model_pair_bootstrap_ci(values, PAIRS, 300, np.random.default_rng(41))
    b = model_pair_bootstrap_ci(values, PAIRS, 300, np.random.default_rng(41))
    assert a == b
    assert a[0] < a[1]
