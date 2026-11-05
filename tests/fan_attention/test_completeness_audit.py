from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from fan.concept.completeness_audit import audit_oracle_concept_coverage


def fixture(omitted: bool, seed=18, n=4200):
    rng=np.random.default_rng(seed)
    true=rng.normal(size=(n,3))
    unknown=rng.normal(size=(n,1))
    pred=true+0.8*rng.normal(size=true.shape)
    logit=1.2*true[:,0]-1.3*true[:,1]+.8*true[:,2]
    if omitted:
        logit=logit+1.7*unknown[:,0]
    y=rng.binomial(1,expit(logit))
    return true,pred,unknown,y,np.arange(2800),np.arange(2800,n)


def test_separates_known_concept_estimation_error_from_omitted_signal():
    c,p,u,y,tr,te=fixture(False)
    a=audit_oracle_concept_coverage(labels=y,predicted_concepts=p,
        true_concepts=c,residual_candidate=u,train_indices=tr,test_indices=te,
        bootstrap_replicates=299)
    assert a.concept_estimation_gain>.02
    assert a.concept_estimation_gain_ci95[0]>0
    assert a.oracle_residual_gain < .02
    assert not a.residual_gain_positive_ci


def test_omitted_true_factor_is_detected_conditionally_on_gold_concepts():
    c,p,u,y,tr,te=fixture(True)
    a=audit_oracle_concept_coverage(labels=y,predicted_concepts=p,
        true_concepts=c,residual_candidate=u,train_indices=tr,test_indices=te,
        bootstrap_replicates=299)
    assert a.oracle_residual_gain > .06
    assert a.oracle_residual_gain_ci95[0] > 0
    assert a.residual_gain_positive_ci
    assert a.concept_estimation_gain_ci95[0] > 0


def test_test_leakage_prevented_by_disjoint_split_guard():
    c,p,u,y,tr,te=fixture(False)
    te=te.copy()
    te[0]=tr[1]
    with pytest.raises(ValueError,match="Disjoint"):
        audit_oracle_concept_coverage(labels=y,predicted_concepts=p,
            true_concepts=c,residual_candidate=u,train_indices=tr,test_indices=te)


def test_invalid_oracle_inputs_fail_closed():
    c,p,u,y,tr,te=fixture(False)
    bad=c.copy()
    bad[2,1]=np.nan
    with pytest.raises(ValueError,match="finite"):
        audit_oracle_concept_coverage(labels=y,predicted_concepts=p,
            true_concepts=bad,residual_candidate=u,train_indices=tr,test_indices=te)
