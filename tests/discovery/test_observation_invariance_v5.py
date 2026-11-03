"""Check observation-policy stress and fail-closed concept-number selection."""
from __future__ import annotations
from types import SimpleNamespace

import numpy as np
import pytest

from conceptfan_discovery.observation_regimes_v5 import (
    apply_measurement_environment,
    evaluate_rank_invariance,
    principal_subspace_overlap,
)


def _fit(k:int,basis:np.ndarray|None=None):
    if basis is None:
        basis=np.eye(6)
    return SimpleNamespace(rank=k,source_direction=basis)


def test_time_channel_missingness_is_predeclared_reproducible_and_no_value_leakage():
    rng=np.random.default_rng(93)
    x=rng.normal(size=(100,24,6)).astype("float32")
    m=(rng.random(size=x.shape)>.13).astype("float32")
    x*=m
    a,ma,receipt=apply_measurement_environment(
        x,m,environment="time_channel_mar",seed=11,
    )
    b,mb,_=apply_measurement_environment(
        x,m,environment="time_channel_mar",seed=11,
    )
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(ma,mb)
    assert np.all(ma<=m)
    assert np.all(a[ma==0]==0)
    assert receipt["selection_depends_on_current_observed_value"] is False
    assert receipt["independent_research_cohort"] is False


def test_diagnostic_mnar_is_not_misreported_as_missing_at_random():
    rng=np.random.default_rng(18)
    x=rng.normal(size=(100,24,6)).astype("float32")
    m=np.ones_like(x)
    _,_,rec=apply_measurement_environment(
        x,m,environment="observed_value_mnar",seed=3,
    )
    assert rec["selection_depends_on_current_observed_value"] is True
    assert "MNAR" in rec["observation_process_assumptions"]


def test_invariant_k_requires_all_measurement_policies_and_independent_halves():
    reference=_fit(3)
    all_environment={
        env:{"whole":reference,"halves":(reference,reference)}
        for env in ("original","mcar_thin","time_channel_mar")
    }
    gate=evaluate_rank_invariance(all_environment,method="equal_time_v5")
    assert gate.status=="MEASUREMENT_INVARIANT_OBSERVABLE_RANK"
    assert gate.consensus_k==3
    assert gate.reasons==()
    assert len(gate.environment_ranks)==3


def test_abstains_on_missing_joint_support_or_inconsistent_rank():
    original=_fit(3)
    other=_fit(2)
    result=evaluate_rank_invariance({
        "base":{"whole":original,"halves":(original,original)},
        "low_observability":{"whole":None,"halves":(None,None)},
        "policy_change":{"whole":other,"halves":(other,other)},
    },method="equal_time_v5")
    assert result.status=="ABSTAIN_UNIDENTIFIABLE"
    assert result.consensus_k is None
    assert "noninvariant_rank_across_observation_environments" in result.reasons
    assert any("insufficient_joint" in x for x in result.reasons)


def test_abstains_when_same_rank_has_rotated_semantic_measurement_space():
    a=np.eye(6)
    b=np.eye(6)[:,[3,4,5,0,1,2]]
    assert principal_subspace_overlap(a,b,3,3)==pytest.approx(0.)
    result=evaluate_rank_invariance({
        "original":{"whole":_fit(3,a),"halves":(_fit(3,a),_fit(3,b))},
    },method="time_observed")
    assert result.consensus_k is None
    assert "original:weak_split_half_subspace_overlap" in result.reasons


def test_bad_observation_policy_does_not_get_fallback():
    x=np.zeros((10,8,3),dtype="float32")
    m=np.ones_like(x)
    with pytest.raises(ValueError):
        apply_measurement_environment(x,m,environment="pretend_MNAR_fixed",seed=1)
    with pytest.raises(ValueError):
        evaluate_rank_invariance({},method="none")
    with pytest.raises(ValueError):
        evaluate_rank_invariance({
            "base":{"whole":_fit(3),"halves":(_fit(3),_fit(3))}
        },method="none",min_squared_principal_cosine=.1)
