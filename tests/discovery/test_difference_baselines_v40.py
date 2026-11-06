"""V40: pre-transfer nonlinearity and equal-label raw differences falsifiers."""
from __future__ import annotations
import numpy as np
import pytest

from conceptfan_discovery.difference_bench_world_v40 import (
    REGISTERED_DEVICES,generate_device_v40,render_observed_v40,
    masks_v40,measured_difference_v40,
)
from conceptfan_discovery.matched_raw_baselines_v40 import (
    measured_temporal_quadratic_features_v40,fit_registered_raw_v40,
)


def _world(seed):
    return generate_device_v40(
        device_seed=seed,n_train=360,n_valid=300,n_test=240)


def test_four_new_device_seeds_without_V39_reuse_and_two_temporal_laws():
    assert set(REGISTERED_DEVICES)=={
        20261407,20261409,20261421,20261423}
    assert sorted(set(REGISTERED_DEVICES.values()))==[
        "linear_AR","nonlinear_switching"]
    for seed in REGISTERED_DEVICES:
        w=_world(seed)
        assert len(w["train"])==360 and len(w["test"])==240
        assert not np.intersect1d(w["train"],w["test"]).size
        assert w["mixing_not_exposed_to_concept_learner"]
        assert w["OFF_is_physical_source_isolation_EXPERIMENTAL_ASSUMPTION"]


def test_linear_matched_source_aligned_common_noise_exactly_cancels():
    world=_world(20261407)
    ix=world["test"]
    on,off=render_observed_v40(
        world,ix,fault="none",seed=119)
    matched_on,matched_off=render_observed_v40(
        world,ix,fault="common8",seed=119)
    ma,mb=masks_v40(on,off,"MCAR20",device_seed=20261407,
                    mask_seed=818)
    d0,m0=measured_difference_v40(on,off,ma,mb)
    d1,m1=measured_difference_v40(
        matched_on,matched_off,ma,mb)
    np.testing.assert_array_equal(m0,m1)
    np.testing.assert_allclose(d0,d1,atol=5e-6,rtol=0)
    # Strictly the old ON-only is corrupted by the aligned artifact.
    assert np.mean(abs(matched_on[:,:,4:10]-on[:,:,4:10]))>.3


def test_nonlinear_pretransfer_means_matched_artifact_no_longer_cancels():
    world=_world(20261421)
    ix=world["test"]
    on,off=render_observed_v40(world,ix,fault="none",seed=119)
    a,b=render_observed_v40(world,ix,fault="common8",seed=119)
    ma,mb=masks_v40(on,off,"complete",device_seed=20261421,
                    mask_seed=818)
    original,mask=measured_difference_v40(on,off,ma,mb)
    affected,mask2=measured_difference_v40(a,b,ma,mb)
    np.testing.assert_array_equal(mask,mask2)
    assert np.mean(abs(affected[:,:,4:10]-original[:,:,4:10]))>.05


def test_off_only_drift_and_source_leak_are_observable_failures():
    for seed in (20261407,20261421):
        w=_world(seed)
        ix=w["test"]
        original_on,original_off=render_observed_v40(
            w,ix,fault="none",seed=111)
        a,b=render_observed_v40(w,ix,fault="off_drift",seed=111)
        _,leaked=render_observed_v40(w,ix,fault="leak",seed=111)
        assert np.array_equal(a,original_on)
        assert np.mean(abs(b[:,:,4:10]-original_off[:,:,4:10]))>.1
        assert np.mean(abs(leaked[:,:,4:10]-original_off[:,:,4:10]))>.05


def test_missing_reference_does_not_supply_fabricated_difference():
    w=_world(20261407)
    on,off=render_observed_v40(w,w["test"],fault="none",seed=91)
    mo,mr=masks_v40(on,off,"missing_reference",
                    device_seed=20261407,mask_seed=91)
    d,mask=measured_difference_v40(on,off,mo,mr)
    assert not mask[:,:,4:10].any()
    assert not d[:,:,4:10].any()
    f=measured_temporal_quadratic_features_v40(
        on,mo,d,mask,use_reference=True)
    assert np.isfinite(f).all()


def test_raw_baselines_never_see_hidden_payload_or_latent_mixing():
    w=_world(20261407)
    on,off=render_observed_v40(
        w,w["train"],fault="none",seed=191)
    mo,mr=masks_v40(on,off,"MCAR20",device_seed=20261407,
                    mask_seed=91)
    d,mask=measured_difference_v40(on,off,mo,mr)
    a=measured_temporal_quadratic_features_v40(
        on,mo,d,mask,use_reference=True)
    on_hidden=on.copy()
    off_hidden=off.copy()
    on_hidden[~mo]=np.nan
    off_hidden[~mr]=np.nan
    d_hidden,mask_hidden=measured_difference_v40(
        on_hidden,off_hidden,mo,mr)
    b=measured_temporal_quadratic_features_v40(
        on_hidden,mo,d_hidden,mask_hidden,use_reference=True)
    np.testing.assert_array_equal(a,b)
    assert a.shape[0]==len(w["train"]) and a.shape[1]>100
    # It has access to the same original K2 ON sensor samples and actually
    # paired 4:9 difference, but not generator H/S/source truth.
    assert np.isfinite(a).all()
    incomplete=mo.copy();incomplete[:,:,4:10]=False
    d_bad,md_bad=measured_difference_v40(
        on,off,incomplete,mr)
    c=measured_temporal_quadratic_features_v40(
        on,incomplete,d_bad,md_bad,use_reference=True)
    assert np.mean(abs(a-c))>.01


def test_registered_raw_method_trains_only_with_frozen_label_count():
    w=_world(20261407)
    on,off=render_observed_v40(
        w,w["train"],fault="none",seed=191)
    mo,mr=masks_v40(on,off,"MCAR20",device_seed=20261407,
                    mask_seed=91)
    d,md=measured_difference_v40(on,off,mo,mr)
    X=measured_temporal_quadratic_features_v40(
        on,mo,d,md,use_reference=True)
    y=w["outcomes"]["energy"][w["train"]]
    logistic=fit_registered_raw_v40("RAW_DIFF_LOGIT_TRAIN1000",X,y)
    assert logistic.predict_proba(X[:10]).shape==(10,2)
    with pytest.raises(ValueError,match="exactly registered"):
        fit_registered_raw_v40(
            "RAW_DIFF_LOGIT_TRAIN1000",X[:300],y[:300])
    with pytest.raises(ValueError,match="Unregistered"):
        fit_registered_raw_v40("ORACLE_HIDDEN_SOURCE",X,y)


def test_known_test_mask_constant_across_source_stress_worlds():
    w=_world(20261407)
    on,off=render_observed_v40(
        w,w["test"],fault="none",seed=111)
    known=np.random.default_rng(15).random((len(on),16,4))>=.2
    for stress in ("MCAR20","MCAR40","missing_reference"):
        mo,_=masks_v40(
            on,off,stress,device_seed=20261407,
            mask_seed=111,known_TEST_first4=known)
        np.testing.assert_array_equal(mo[:,:,:4],known)
