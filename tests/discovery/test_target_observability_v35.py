"""V35 falsifiers for source-specific observability, uncertainty and calibration."""
from __future__ import annotations

import numpy as np
import pytest

from conceptfan_discovery.target_observability_v35 import (
    PHI,SUPPORT,TARGET,
    make_independent_phantom,fit_external_reference,
    estimate_calibrated_source,measured_target_membership,
    linear_logit_sensitivity_radius,
    nullspace_counterexample,_target_pseudoinverse,
    source_concept_or_abstain_without_calibration,
)


def _cal():
    u,x=make_independent_phantom(seed=20263500,n=900)
    return fit_external_reference(
        u,x,physically_verified_routing_support=SUPPORT,
        independent_reference=True)


def _world(n=300,seed=19):
    rng=np.random.default_rng(seed)
    s=rng.uniform(-1.3,1.3,size=(n,2))
    h=np.array([[1.,1.],[0.,1.],[1.,0.]])
    x=np.zeros((n,16,12),dtype=np.float64)
    x[:,:,4:7]=(PHI[None,:,None]*
                 (s@h.T)[:,None,:]+
                 rng.normal(0,.32,size=(n,16,3)))
    return x,s


def _single_sensor(x,j):
    m=np.zeros_like(x,dtype=bool)
    m[:,:,j]=True
    return m


def test_known_reference_calibration_recovers_topology_and_calibration_error():
    calibration=_cal()
    htrue=np.array([[1.,1.],[0.,1.],[1.,0.]])
    np.testing.assert_allclose(calibration.response,htrue,atol=.06)
    np.testing.assert_array_equal(
        calibration.response[~SUPPORT],np.zeros((~SUPPORT).sum()))
    assert np.all((calibration.noise_sd>.27)&(calibration.noise_sd<.37))
    assert len(calibration.calibration_sha256)==64
    u,x=make_independent_phantom()
    with pytest.raises(ValueError,match="Independent"):
        fit_external_reference(
            u,x,physically_verified_routing_support=SUPPORT,
            independent_reference=False)
    # If routing is not known, an independent excitation does not certify
    # an exact zero crosstalk; no target-only rank1 guarantee is possible.
    untrusted=np.ones((3,2),dtype=bool)
    alt=fit_external_reference(
        u,x,physically_verified_routing_support=untrusted,
        independent_reference=True)
    assert alt.response[2,1]!=0
    # Independent reference excitation with finite noisy samples does not
    # CERTIFY an exact structural zero. Without external wiring metadata,
    # sensor6 has estimated nonzero nuisance cross-talk: rank1 does not
    # uniquely identify a1 under the strict rowspace definition.
    sample,_=_world(n=140)
    observed=_single_sensor(sample,6)
    unverified=estimate_calibrated_source(sample,observed,alt)
    assert not unverified["target_in_rowspace"].any()
    assert not unverified["calibrated_target_rowspace"].any()


def test_mixture_not_identifiable_even_with_all_16_measured_timepoints():
    x,_=_world()
    r=estimate_calibrated_source(
        x,_single_sensor(x,4),_cal())
    assert np.all(r["unsafe_mask_fraction"])
    assert not np.any(r["target_in_rowspace"])
    assert not np.any(r["calibrated_target_rowspace"])
    assert not np.any(r["calibrated_full_rank"])
    assert np.isinf(r["target_se"]).all()
    assert np.all(r["rank"]==1)
    q,s=measured_target_membership(
        r,"calibrated_target_rowspace",
        train_tau=0.,train_width=1.)
    np.testing.assert_array_equal(q,np.full(len(x),.5))
    np.testing.assert_array_equal(s,np.zeros(len(x)))


def test_nuisance_only_unidentifiable_but_target_only_identifiable_rank1():
    x,truth=_world()
    r_n=estimate_calibrated_source(x,_single_sensor(x,5),_cal())
    assert not r_n["target_in_rowspace"].any()
    r_t=estimate_calibrated_source(x,_single_sensor(x,6),_cal())
    assert r_t["target_in_rowspace"].all()
    assert r_t["calibrated_target_rowspace"].all()
    assert not r_t["calibrated_full_rank"].any()
    assert (r_t["rank"]==1).all()
    assert np.abs(r_t["estimated_target"]-truth[:,0]).mean()<.12
    assert np.all(r_t["target_se"]<.15)


def test_two_indirect_sensors_identify_target_without_direct_sensor():
    x,truth=_world()
    m=np.zeros_like(x,dtype=bool)
    m[:,:,4:6]=True
    r=estimate_calibrated_source(x,m,_cal())
    assert np.all(r["rank"]==2)
    assert r["target_in_rowspace"].all()
    assert r["calibrated_target_rowspace"].all()
    assert np.mean(abs(r["estimated_target"]-truth[:,0]))<.18


def test_no_source_returns_exactly_neutral_without_pseudoinverse_fabrication():
    x,_=_world(n=140)
    mask=np.zeros_like(x,dtype=bool)
    r=estimate_calibrated_source(x,mask,_cal())
    assert np.all(r["rank"]==0)
    assert not r["target_in_rowspace"].any()
    assert np.isinf(r["target_se"]).all()
    assert not r["calibrated_target_rowspace"].any()
    q,use=measured_target_membership(r,"calibrated_target_rowspace",
                                     train_tau=0.,train_width=1.)
    np.testing.assert_array_equal(q,np.full(len(x),.5))
    np.testing.assert_array_equal(use,np.zeros(len(x)))
    radius=linear_logit_sensitivity_radius(
        2.5,r["target_se"],train_width=.8,
        eligible=r["calibrated_target_rowspace"])
    np.testing.assert_array_equal(radius,np.zeros(len(x)))


def test_pair_of_latent_worlds_cannot_be_disambiguated_from_mixture_only():
    worlds=nullspace_counterexample()
    assert worlds["both_worlds_exactly_observationally_indistinguishable"]
    assert worlds["sensor4_readout_world_A"]==worlds["sensor4_readout_world_B"]
    assert worlds["target_a1_world_A"]!=worlds["target_a1_world_B"]
    single=_target_pseudoinverse(
        np.array([0.]),np.array([[1.,1.]]),np.array([.3]))
    assert single["rank"]==1
    assert not single["target_in_rowspace"]
    assert single["nullspace_target_fraction"]>0.5
    assert np.isinf(single["target_se"])


def test_hidden_payload_nan_does_not_change_observed_inverse_problem():
    x,_=_world(n=150)
    rng=np.random.default_rng(45)
    m=rng.random(x.shape)>.35
    first=estimate_calibrated_source(x,m,_cal())
    modified=x.copy()
    modified[~m]=float("nan")
    second=estimate_calibrated_source(modified,m,_cal())
    for key in ("estimated_target","target_se","rank",
                "target_in_rowspace","nullspace_target_fraction",
                "normalized_residual","calibrated_target_rowspace"):
        np.testing.assert_array_equal(first[key],second[key])


def test_gls_standard_error_and_conditional_gaussian_95pct_coverage():
    x,truth=_world(n=1200)
    m=np.ones_like(x,dtype=bool)
    r=estimate_calibrated_source(x,m,_cal())
    assert r["calibrated_target_rowspace"].all()
    assert np.abs(r["estimated_target"]-truth[:,0]).mean()<.1
    pct=np.mean(np.abs(r["estimated_target"]-truth[:,0])<=
                1.96*r["target_se"])
    # Plug-in H and noise measured on separate synthetic calibration, so
    # small departures are expected; this is NOT a distribution-free claim.
    assert .90<pct<.98, pct


def test_common_mode_flip_undetectable_by_linear_residual_alone():
    n=300
    x,_=_world(n=n)
    m=np.zeros_like(x,dtype=bool)
    m[:,:,4:7]=True
    reference=estimate_calibrated_source(x,m,_cal())
    shifted=x.copy()
    shifted[:,:,4:7]*=-1
    flipped=estimate_calibrated_source(shifted,m,_cal())
    np.testing.assert_allclose(flipped["estimated_target"],
                               -reference["estimated_target"],atol=1e-12)
    np.testing.assert_allclose(flipped["normalized_residual"],
                               reference["normalized_residual"],atol=1e-12)
    assert flipped["target_in_rowspace"].all()


def test_source_sensitivity_bound_is_algebraic_not_probability_CI():
    se=np.array([.10,.03,np.inf])
    eligible=np.array([True,True,False])
    r=linear_logit_sensitivity_radius(
        gamma=-2.,target_se=se,train_width=.5,eligible=eligible)
    np.testing.assert_allclose(
        r[:2],2*1.96*se[:2]/(4*.5))
    assert r[2]==0
    with pytest.raises(ValueError):
        linear_logit_sensitivity_radius(3.,se,train_width=0,eligible=eligible)


def test_no_independent_calibration_fails_closed_even_if_all_sensors_present():
    x,_=_world(n=120)
    m=np.ones_like(x,dtype=bool)
    r=source_concept_or_abstain_without_calibration(
        x,m,None,method="calibrated_target_rowspace",
        train_tau=0.,train_width=1.)
    assert r["reason"]=="ABSTAIN_MISSING_INDEPENDENT_EXTERNAL_REFERENCE"
    np.testing.assert_array_equal(r["q"],np.full(len(x),.5))
    np.testing.assert_array_equal(r["support"],np.zeros(len(x)))
    assert not r["used_calibrated_source"].any()
    assert np.isinf(r["target_se"]).all()
    assert not r["calibration_verified"]
    with pytest.raises(ValueError,match="External"):
        estimate_calibrated_source(x,m,None)
    approved=source_concept_or_abstain_without_calibration(
        x,m,_cal(),method="calibrated_target_rowspace",
        train_tau=0.,train_width=1.)
    assert approved["used_calibrated_source"].all()
    assert approved["measurement_operator_known"]
    assert not approved["physical_semantics_certified"]
