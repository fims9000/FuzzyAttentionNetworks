"""V20c: accept only predictive and demonstrably improved native refits."""
import numpy as np
import pytest

from conceptfan_discovery.native_head_refit_gate_v20c import (
    decide_head_refit_from_fit_disjoint_train_gate,
)
from conceptfan_discovery.experiment_refit_gate_v20c import _make_data
from conceptfan_discovery.distinct_synthetic_v20b import FAMILIES


def test_independent_gate_accepts_large_predictive_train_only_improvement():
    rng=np.random.default_rng(305018)
    latent=rng.normal(size=450)
    y=(rng.random(450)<1/(1+np.exp(-3*latent))).astype(float)
    old=np.full(len(y),.5)
    new=1/(1+np.exp(-2*latent))
    ev=decide_head_refit_from_fit_disjoint_train_gate(
        y,old,new,permutations=199,bootstraps=199,
    )
    assert ev.status=="ACCEPT_TRAIN_ONLY_NATIVE_REFIT"
    assert ev.candidate_risk_point_biserial_r>0
    assert ev.candidate_risk_one_sided_permutation_p<=.01
    assert ev.gain_bootstrap_ci90[0]>0


def test_zero_benefit_and_random_outcome_are_rejected():
    rng=np.random.default_rng(305019)
    n=400
    y=rng.integers(0,2,size=n).astype(float)
    old=np.full(n,.5)
    same=decide_head_refit_from_fit_disjoint_train_gate(
        y,old,old,permutations=199,bootstraps=199,
    )
    assert same.status=="KEEP_ORIGINAL_ABSTAIN_UNCERTAIN_GAIN"
    assert same.mean_gate_logloss_gain==0
    irrelevant=1/(1+np.exp(-rng.normal(size=n)))
    ev=decide_head_refit_from_fit_disjoint_train_gate(
        y,old,irrelevant,permutations=199,bootstraps=199,
    )
    assert ev.status=="KEEP_ORIGINAL_ABSTAIN_UNCERTAIN_GAIN"


def test_malformed_gate_data_and_thresholds_are_refused():
    y=np.tile([0.,1.],100)
    p=np.full(200,.5)
    with pytest.raises(ValueError,match="ABSTAIN"):
        decide_head_refit_from_fit_disjoint_train_gate(
            y[:-1],p,p,
        )
    with pytest.raises(ValueError,match="ABSTAIN"):
        decide_head_refit_from_fit_disjoint_train_gate(
            np.ones(200),p,p,
        )
    with pytest.raises(ValueError,match="Unregistered"):
        decide_head_refit_from_fit_disjoint_train_gate(
            y,p,p,minimum_absolute_logloss_gain=-.1,
        )


@pytest.mark.parametrize("family",FAMILIES)
def test_fit_disjoint_train_gate_and_test_are_independent(family):
    data,test=_make_data(family)
    train=data.train
    gate=data.validation
    assert len(train.y)==800
    assert len(gate.y)==len(test.y)==300
    assert set(train.episode_ids).isdisjoint(set(gate.episode_ids))
    assert set(train.episode_ids).isdisjoint(set(test.episode_ids))
    assert set(gate.episode_ids).isdisjoint(set(test.episode_ids))
    assert train.oracle_states is None
    assert gate.oracle_states is None
    assert test.oracle_states is None
    assert data.receipt["all_normalization_from_core_train_only"]
    assert np.array_equal(train.x[train.mask==0],np.zeros_like(train.x[train.mask==0]))
    assert np.isfinite(train.x).all()
