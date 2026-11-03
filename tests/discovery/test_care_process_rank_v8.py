"""CARE mask process rank must emerge from TRAIN masks, not a manually set 3."""
from __future__ import annotations

import numpy as np
import pytest

from conceptfan_discovery.care_process_rank_v8 import (
    fit_care_policy_rank,_positive_negative_template,
)


def _correlated_measurement_process(n=260,t=24,d=5,seed=31):
    rng=np.random.default_rng(seed)
    patient_intensity=rng.normal(size=(n,1,1))
    time_effect=.45*np.sin(np.arange(t)[None,:,None]*.27)
    channel_effect=np.arange(d)[None,None,:]*.17
    logits=.3+patient_intensity+time_effect-channel_effect
    p=1./(1.+np.exp(-logits))
    return (rng.random((n,t,d))<p).astype(np.float32)


def test_train_only_mask_rank_is_reproducible_and_anchors_are_simplex():
    m=_correlated_measurement_process()
    first=fit_care_policy_rank(m,permutations=40,max_slots=5,seed=21)
    second=fit_care_policy_rank(m.copy(),permutations=40,max_slots=5,seed=21)
    assert 1<=first.chosen_k<=5
    assert first.chosen_k==second.chosen_k
    np.testing.assert_array_equal(first.raw_eigenvalues,second.raw_eigenvalues)
    np.testing.assert_array_equal(first.positive_negative_simplex_anchors,
                                  second.positive_negative_simplex_anchors)
    assert first.positive_negative_simplex_anchors.shape==(first.chosen_k,2*4*5)
    np.testing.assert_allclose(
        first.positive_negative_simplex_anchors.sum(axis=1),
        np.ones(first.chosen_k),atol=1e-6,
    )
    assert first.receipt()["clinical_outcomes_used_for_k"] is False
    assert first.receipt()["physiological_measurement_values_used_for_k"] is False


def test_constant_care_policy_abstains_instead_of_fabricating_mechanisms():
    with pytest.raises(ValueError,match="too few nonconstant"):
        fit_care_policy_rank(np.ones((160,24,4),dtype=np.float32),permutations=30)
    with pytest.raises(ValueError):
        fit_care_policy_rank(np.zeros((160,24,4),dtype=np.float32),permutations=30)


def test_zero_variance_operators_do_not_create_fake_factors():
    m=_correlated_measurement_process()
    m[:,:,4]=0
    evidence=fit_care_policy_rank(m,permutations=30,seed=27)
    assert np.isfinite(evidence.raw_eigenvalues).all()
    assert evidence.chosen_k<=8
    assert np.all(evidence.source_operator_scale>=0)


def test_manually_inconsistent_signed_basis_rejected():
    with pytest.raises(ValueError):
        _positive_negative_template(np.ones((8,3)))
    x=np.eye(8)[:,:3]
    out=_positive_negative_template(x)
    assert out.shape==(3,16)
    np.testing.assert_allclose(out.sum(axis=1),np.ones(3))


def test_invalid_null_protocol_and_nonbinary_mask_rejected():
    m=_correlated_measurement_process()
    with pytest.raises(ValueError):
        fit_care_policy_rank(m,permutations=10)
    with pytest.raises(ValueError):
        fit_care_policy_rank(m,max_slots=0)
    with pytest.raises(ValueError):
        fit_care_policy_rank(m*.5)
