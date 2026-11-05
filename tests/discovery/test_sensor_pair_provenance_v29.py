from __future__ import annotations

import numpy as np
import pytest

from conceptfan_discovery.sensor_pair_provenance_v29 import (
    fit_unlabelled_pair_guard,pair_diagnostics,pair_is_credible,
)


def _paired_sensor_experiment(seed=20261109,n=1000):
    rng=np.random.default_rng(seed)
    latent=rng.standard_normal((n,16))
    for t in range(1,16):
        latent[:,t]=.7*latent[:,t-1]+np.sqrt(1-.7**2)*latent[:,t]
    observed=np.zeros((n,16,6),dtype=np.float64)
    observed[:,:,4]=latent+.28*rng.standard_normal((n,16))
    observed[:,:,5]=latent+.28*rng.standard_normal((n,16))
    mask=rng.random(observed.shape)>.2
    return observed,mask


def test_train_only_pair_policy_and_healthy_support():
    x,m=_paired_sensor_experiment()
    rule=fit_unlabelled_pair_guard(x,m)
    accepted=pair_is_credible(rule,x,m)
    assert rule.outcome_labels_used is False
    assert rule.does_not_certify_common_mode_faults is True
    assert rule.train_subjects==1000 and rule.valid_train_subjects>900
    assert 0<=rule.min_pearson_corr<1
    assert rule.max_mean_abs_disagreement>0
    assert accepted["integrity_passed"].mean()>.90


def test_one_readout_sign_or_offset_change_is_detectable():
    x,m=_paired_sensor_experiment()
    rule=fit_unlabelled_pair_guard(x[:700],m[:700])
    clean=pair_is_credible(rule,x[700:],m[700:])
    one_flipped=x[700:].copy()
    one_flipped[:,:,5]*=-1
    sign=pair_is_credible(rule,one_flipped,m[700:])
    one_biased=x[700:].copy()
    one_biased[:,:,5]+=2.0
    biased=pair_is_credible(rule,one_biased,m[700:])
    assert clean["integrity_passed"].mean()>.85
    assert sign["integrity_passed"].mean()<.15
    assert biased["integrity_passed"].mean()<.15


def test_common_mode_both_sensor_flip_is_indistinguishable_to_pair_guard():
    x,m=_paired_sensor_experiment()
    rule=fit_unlabelled_pair_guard(x[:700],m[:700])
    intact=pair_is_credible(rule,x[700:],m[700:])
    both=x[700:].copy()
    both[:,:,4:6]*=-1
    inverted=pair_is_credible(rule,both,m[700:])
    np.testing.assert_array_equal(intact["integrity_passed"],
                                  inverted["integrity_passed"])
    # This is a required NEGATIVE control: pair-only monitor cannot certify
    # global orientation, physical factor identity, or causal correctness.


def test_unobserved_nan_payloads_never_enter_pair_monitor():
    x,m=_paired_sensor_experiment(n=150)
    prior=pair_diagnostics(x,m)
    hidden=x.copy()
    hidden[~m]=float("nan")
    after=pair_diagnostics(hidden,m)
    for a,b in zip(prior,after):
        np.testing.assert_allclose(a,b,equal_nan=True,atol=0,rtol=0)


def test_all_missing_or_insufficient_joint_measures_force_refusal():
    x,m=_paired_sensor_experiment()
    rule=fit_unlabelled_pair_guard(x,m)
    m[:,:,:]=0
    result=pair_is_credible(rule,x,m)
    assert not result["integrity_passed"].any()
    assert not result["enough_joint_history"].any()
    assert np.isnan(result["corr"]).all()


def test_malformed_observation_mask_is_rejected():
    x,m=_paired_sensor_experiment(n=150)
    bad=m.astype(float)
    bad[0,0,4]=0.6
    with pytest.raises(ValueError):
        pair_diagnostics(x,bad)
