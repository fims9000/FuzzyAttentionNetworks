"""Prespecified environment-transport and patient-bootstrap acceptance controls."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from conceptfan_discovery.cross_environment_transport import (
    CASES,changed_sensor_environment,paired_test_ap_difference,case_manifest,
)
from conceptfan_discovery.auto_measured_concept_benchmark import (
    extended_nonlinear_world,masked_histories,
)
from conceptfan_discovery.auto_measured_concept_proposals import (
    form_train_measured_groups,group_provenance_fingerprint,
)
from conceptfan_discovery.auto_measured_concept_admission import (
    evaluate_two_independent_environments,
)
from conceptfan_discovery.isolated_source_admission_v28 import (
    train_source_coefficient,
)


@pytest.mark.parametrize("regime",("none","benign_drift","flip_source","flip_artifact"))
def test_shift_changes_only_unknown_sensors_and_retains_known_k2_sensor_firewall(regime):
    d=extended_nonlinear_world(seed=20261114,train_n=300,valid_n=300,test_n=300)
    original=d["x"][d["test"]]
    shifted=changed_sensor_environment(original,regime)
    np.testing.assert_array_equal(original[:,:,:4],shifted[:,:,:4])
    if regime=="none":
        np.testing.assert_array_equal(original,shifted)
    elif regime=="flip_source":
        np.testing.assert_array_equal(shifted[:,:,4:7],-original[:,:,4:7])
        np.testing.assert_array_equal(shifted[:,:,9:11],original[:,:,9:11])
    elif regime=="flip_artifact":
        np.testing.assert_array_equal(shifted[:,:,9:11],-original[:,:,9:11])
        np.testing.assert_array_equal(shifted[:,:,4:7],original[:,:,4:7])
    else:
        assert not np.array_equal(shifted[:,:,4:],original[:,:,4:])
    assert len(case_manifest())==5
    assert len(set(z.name for z in CASES))==5


def test_two_cold_validation_environment_shift_can_reject_a_predictive_source():
    rng=np.random.default_rng(20261006)
    s=rng.normal(size=1700)
    qtr=expit(s)
    ytrain=rng.binomial(1,expit(7.0*(qtr-.5)))
    gamma=train_source_coefficient(np.zeros(1700),qtr,ytrain)
    assert gamma>0
    val=[]
    for env in range(2):
        q=expit(rng.normal(size=600))
        y=rng.binomial(1,expit(7.*(q-.5)))
        val.append((f"VALID{env+1}",y,np.zeros(len(y)),q,np.ones(len(y))))
    clean=evaluate_two_independent_environments(
        "measured_source",gamma,val[0],val[1],
        train_label_count=1700,seed=100)
    inverted=list(val[1])
    inverted[3]=1.0-inverted[3]
    shifted=evaluate_two_independent_environments(
        "measured_source",gamma,val[0],tuple(inverted),
        train_label_count=1700,seed=100)
    assert clean.approved
    assert shifted.environment_receipts[0].passed
    assert not shifted.environment_receipts[1].passed
    assert shifted.approved is False
    assert shifted.gamma_train_only==clean.gamma_train_only
    assert clean.n_total_valid_labels==shifted.n_total_valid_labels==1200


def test_train_only_source_formations_cannot_see_later_environment_shifts():
    d=extended_nonlinear_world(seed=20261114,train_n=420,valid_n=300,test_n=300)
    tr=d["train"]
    m=masked_histories(d["x"][tr],"MCAR20",seed=20268201)
    s=form_train_measured_groups(d["x"][tr],m,known_sensor_indices=(0,1,2,3),
                                  seed=20268204)
    assert {frozenset(p.sensor_indices) for p in s.candidates}>={
        frozenset((4,5,6)),frozenset((7,8)),frozenset((9,10))}
    assert not s.outcomes_used_for_grouping
    v2=d["x"][d["valid_env2"]].copy()
    v2=changed_sensor_environment(v2,"flip_source")
    assert group_provenance_fingerprint(s)==group_provenance_fingerprint(
        form_train_measured_groups(d["x"][tr],m,
                                  known_sensor_indices=(0,1,2,3),seed=20268204))
    assert not np.array_equal(v2,d["x"][d["valid_env2"]])


def test_paired_patient_bootstrap_detects_gain_and_is_deterministic():
    rng=np.random.default_rng(20261114)
    y=rng.integers(0,2,size=600)
    q=.15+.7*y+.08*rng.normal(size=len(y))
    good=np.stack([np.clip(q+.015*rng.normal(size=len(y)),.001,.999)
                   for _ in range(3)])
    bad=np.stack([rng.uniform(.2,.8,size=len(y)) for _ in range(3)])
    a=paired_test_ap_difference(y,good,bad,seed=7,resamples=199)
    b=paired_test_ap_difference(y,good,bad,seed=7,resamples=199)
    assert a==b
    assert a["difference_AUPRC_a_minus_b"]>.20
    assert a["paired_TEST_subject_bootstrap_95pct_ci"][0]>0
    assert a["n_nn_fits_shared_subjects"]==3
    zero=paired_test_ap_difference(y,good,good,seed=7,resamples=199)
    assert zero["difference_AUPRC_a_minus_b"]==0.0
    np.testing.assert_array_equal(
        zero["paired_TEST_subject_bootstrap_95pct_ci"],[0.,0.])


def test_new_measurement_regime_names_refuse_undeclared_faults():
    x=np.ones((101,16,12),dtype=np.float32)
    with pytest.raises(ValueError):
        changed_sensor_environment(x,"device_undeclared")
    with pytest.raises(ValueError):
        paired_test_ap_difference(
            np.zeros(101,dtype=np.int64),
            np.zeros((3,101)),np.ones((3,101)),seed=1,resamples=199)
