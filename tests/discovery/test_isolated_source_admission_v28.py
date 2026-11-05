from __future__ import annotations

import numpy as np
from scipy.special import expit

from conceptfan_discovery.isolated_source_admission_v28 import (
    SOURCE_PROVENANCE,
    candidate_logit,
    isolated_source_prediction,
    measured_candidate,
    train_source_coefficient,
    two_environment_admission,
)


def _synthetic_train_and_validation(seed=20261108):
    rng=np.random.default_rng(seed)
    z_train=rng.uniform(-1.2,1.2,size=850)
    train_c=expit(z_train)
    train_y=rng.binomial(1,expit(7*(train_c-.5)))
    gamma=train_source_coefficient(np.zeros(len(z_train)),train_c,train_y)
    envs=[]
    for i in range(2):
        z=rng.uniform(-1.2,1.2,size=600)
        c=expit(z)
        y=rng.binomial(1,expit(7*(c-.5)))
        envs.append((f"environment{i+1}",y,np.zeros(len(y)),c,np.ones(len(y))))
    return gamma,envs


def test_measured_candidate_ignores_missing_payload_and_is_neutral_when_absent():
    rng=np.random.default_rng(4)
    x=rng.standard_normal((7,16,6))
    m=rng.integers(0,2,size=x.shape,dtype=np.int8)
    c1,p1=measured_candidate(x,m)
    x2=x.copy()
    x2[m==0]=float("nan")
    c2,p2=measured_candidate(x2,m)
    np.testing.assert_allclose(c1,c2,atol=0,rtol=0)
    np.testing.assert_allclose(p1,p2,atol=0,rtol=0)
    m[:,:,:]=0
    unknown,coverage=measured_candidate(x2,m)
    np.testing.assert_array_equal(unknown,np.full(7,.5))
    np.testing.assert_array_equal(coverage,np.zeros(7))
    assert SOURCE_PROVENANCE["causal_physical_identity_certified"] is False


def test_extra_source_is_exact_additive_and_original_head_does_not_change():
    b=np.array([.2,-.3,.9,-1.2])
    c=np.array([.6,.7,.45,.2])
    coverage=np.array([1.,0.,1.,1.])
    old=b.copy()
    out=isolated_source_prediction(
        b,c,coverage,coefficient=2.2,globally_admitted=True,
        minimum_subject_support=.5)
    np.testing.assert_array_equal(b,old)
    np.testing.assert_array_equal(out["base_logit"],old)
    np.testing.assert_array_equal(out["gated_logit"][[1]],old[[1]])
    np.testing.assert_allclose(out["gated_logit"],
                               b+out["source_signed_contribution"],rtol=0,atol=1e-14)
    rejected=isolated_source_prediction(
        b,c,coverage,coefficient=2.2,globally_admitted=False)
    np.testing.assert_array_equal(rejected["gated_logit"],old)
    assert not rejected["used_source"].any()


def test_independent_stable_positive_source_passes_both_validation_environments():
    gamma,env=_synthetic_train_and_validation()
    assert gamma>0
    result=two_environment_admission(gamma,environments=env,bootstrap_replicates=199)
    assert result.admitted
    assert len(result.environments)==2
    assert all(e.passed for e in result.environments)
    assert result.only_known_environment_predictive_validity


def test_known_sensor_sign_reversal_is_rejected():
    gamma,env=_synthetic_train_and_validation()
    env2=list(env[1])
    env2[3]=1.0-env2[3]
    decision=two_environment_admission(
        gamma,environments=(env[0],tuple(env2)),bootstrap_replicates=199)
    assert not decision.admitted
    assert decision.environments[0].passed
    assert not decision.environments[1].passed


def test_unseen_shift_does_not_change_prior_admission_and_harms_unchecked_source():
    gamma,env=_synthetic_train_and_validation()
    gate1=two_environment_admission(gamma,environments=env,bootstrap_replicates=199)
    gate2=two_environment_admission(gamma,environments=env,bootstrap_replicates=199)
    assert gate1.as_dict()==gate2.as_dict() and gate1.admitted
    rng=np.random.default_rng(20261108)
    c=expit(rng.uniform(-1.2,1.2,size=600))
    y=rng.binomial(1,expit(7*(c-.5)))
    reversed_candidate=1.-c
    old=np.zeros(len(y))
    out=isolated_source_prediction(
        old,reversed_candidate,np.ones(len(y)),coefficient=gamma,
        globally_admitted=gate1.admitted)
    p0=expit(old)
    p1=expit(out["gated_logit"])
    nll0=-(y*np.log(p0)+(1-y)*np.log(1-p0)).mean()
    nll1=-(y*np.log(p1)+(1-y)*np.log(1-p1)).mean()
    assert nll1>nll0, "Future unseen device shift should falsify static admission"


def test_invalid_sensor_availability_is_rejected():
    x=np.ones((3,16,6))
    m=np.ones_like(x)
    m[0,1,5]=.3
    try:
        measured_candidate(x,m)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected a binary-mask contract failure")
