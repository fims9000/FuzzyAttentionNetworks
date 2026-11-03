"""Prevent tautologically fixed sign distributions from masquerading as explanation stability."""
from __future__ import annotations

import numpy as np

from conceptfan_discovery.evaluation import DiscoveryEvidence, compare_independent_discoveries


def _evidence(contributions: np.ndarray) -> DiscoveryEvidence:
    n,k=contributions.shape
    rng=np.random.default_rng(13)
    time_variation=rng.normal(scale=.1,size=(n,8,k))
    trajectory=np.clip(.5+time_variation,0,1)
    weights=np.eye(k,64)
    return DiscoveryEvidence(
        metrics={"n_active_and_nonconstant_slots":k},
        probabilities=np.full(n,.5),
        contributions=contributions,
        trajectories=trajectory,
        raw_activation=trajectory,
        reliability=np.ones_like(trajectory),
        weights=weights,
        gates=np.ones(k),
        boundary=np.zeros(k),
        width=np.ones(k),
        decision_weights=np.ones(k),
    )


def test_identical_patient_specific_signs_have_excess_over_chance():
    rng=np.random.default_rng(18)
    effects=rng.normal(size=(300,4))
    result=compare_independent_discoveries(_evidence(effects),_evidence(effects))
    assert result["paired_contribution_sign_agreement"]==1.0
    assert result["sign_frequency_preserving_marginal_null"]<.55
    assert result["case_specific_excess_sign_agreement"]>.45


def test_constant_shared_sign_pattern_has_zero_excess_over_chance():
    scores=np.repeat(np.array([[1.,-2.,3.,-.4]]),200,axis=0)
    result=compare_independent_discoveries(_evidence(scores),_evidence(scores))
    assert result["paired_contribution_sign_agreement"]==1.0
    assert result["sign_frequency_preserving_marginal_null"]==1.0
    assert abs(result["case_specific_excess_sign_agreement"])<1e-12
    assert abs(result["excess_over_unmatched_episode_null_spearman"])<1e-12
