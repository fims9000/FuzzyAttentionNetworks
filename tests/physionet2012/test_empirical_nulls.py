"""Regression controls for empirical rank/sign chance agreement."""
from __future__ import annotations

import numpy as np
import pytest

from conceptfan_realdata.stability_nulls import agreement_nulls


def test_uniform_independent_models_have_chance_baseline():
    rng = np.random.default_rng(41)
    x = rng.normal(size=(8, 350, 5))
    result = agreement_nulls(x, repetitions=130, seed=19)
    assert result["model_pairs"] == 28
    assert abs(result["top1"]["marginal_frequency_null"] - 0.2) < 0.035
    assert abs(result["sign"]["marginal_frequency_null"] - 0.5) < 0.035
    assert abs(result["top1"]["permutation_null_mean"] -
               result["top1"]["marginal_frequency_null"]) < 0.02
    assert abs(result["sign"]["permutation_null_mean"] -
               result["sign"]["marginal_frequency_null"]) < 0.02


def test_biased_frequencies_are_preserved_not_forced_to_uniform():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(5, 90, 5))
    x[:, :, 0] = 100.0
    result = agreement_nulls(x, repetitions=100)
    assert result["top1"]["observed"] == 1.0
    assert result["top1"]["marginal_frequency_null"] == 1.0
    assert result["top1"]["permutation_null_mean"] == 1.0


def test_shared_episode_alignment_is_detectable_and_deterministic():
    rng = np.random.default_rng(11)
    base = rng.normal(size=(140, 5))
    x = np.stack([base] * 5)
    a = agreement_nulls(x, repetitions=100, seed=42)
    b = agreement_nulls(x, repetitions=100, seed=42)
    assert a == b
    assert a["top1"]["observed"] == 1.0
    assert a["sign"]["observed"] == 1.0
    assert a["top1"]["observed_minus_null_mean"] > 0.5
    assert a["sign"]["observed_minus_null_mean"] > 0.3


def test_null_inputs_checked():
    with pytest.raises(ValueError):
        agreement_nulls(np.ones((2, 100)))
    with pytest.raises(ValueError):
        agreement_nulls(np.full((3, 10, 5), np.nan))
    with pytest.raises(ValueError):
        agreement_nulls(np.ones((3, 10, 5)), repetitions=1)
