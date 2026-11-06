"""V37 measurable-centered three-view source and negative controls.

No Y / latent S / matrices H are ever supplied to source fit.
Tests enforce batch-vs-single inference and exact refusal, avoiding V36 bug.
"""
from __future__ import annotations
import numpy as np
import pytest

from conceptfan_discovery.threeview_world_v37 import (
    generate_threeview_world_v37,apply_device_world_v37,view_masks_v37,
)
from conceptfan_discovery.threeview_centered_v37 import (
    fit_centered_threeview_v37,threeview_feature_v37,pair_feature_v37,
)


def _fitted(seed=20261217):
    d=generate_threeview_world_v37(
        mix_seed=seed,n_train=440,n_valid=300,n_test=240)
    ix=d["train"]
    mask=view_masks_v37(d["x"][ix],"MCAR20",seed=27,known_seed=72)
    fit=fit_centered_threeview_v37(
        d["x"][ix],mask,independent_view_noise_documented=True,
        seed=20263701)
    return d,fit


def test_two_new_mixes_have_different_hidden_H_but_source_fit_does_not_access_them():
    d1,f1=_fitted(20261217)
    d2,f2=_fitted(20261219)
    assert d1["synthetic_mixes_AUDIT_ONLY"]!=d2["synthetic_mixes_AUDIT_ONLY"]
    assert not f1.true_physical_semantics_certified
    assert f1.audit()["latent_sources_and_true_mixing_not_used"]
    assert all(not p.original_mixing_supplied for p in f1.pairs.values())
    assert all(p.spectral_gap>=.25 for p in f1.pairs.values())
    assert all(p.bootstrap_abs_cos_q10>=.8 for p in f2.pairs.values())


def test_constant_per_episode_offsets_do_not_change_source_or_admission():
    d,fit=_fitted()
    x=d["x"][d["test"]]
    m=view_masks_v37(x,"MCAR20",seed=77,known_seed=18)
    original=threeview_feature_v37(fit,x,m)
    for intervention in ("offset_A","offset_all"):
        changed=apply_device_world_v37(x,intervention,seed=912)
        result=threeview_feature_v37(fit,changed,m)
        np.testing.assert_array_equal(
            result["support"],original["support"])
        np.testing.assert_allclose(
            result["q"],original["q"],rtol=0,atol=3e-6)
        np.testing.assert_allclose(
            result["pairwise_max_membership_discrepancy"],
            original["pairwise_max_membership_discrepancy"],rtol=0,atol=3e-6)


def test_all_three_views_missing_or_one_view_missing_must_abstain():
    d,fit=_fitted()
    x=d["x"][d["test"]]
    for stress in ("missing_A","missing_C","no_views"):
        mask=view_masks_v37(x,stress,seed=62,known_seed=54)
        result=threeview_feature_v37(fit,x,mask)
        assert not result["support"].any()
        np.testing.assert_array_equal(result["q"],np.full(len(x),.5))


def test_hidden_payload_never_changes_inference_and_single_episode_supported():
    d,fit=_fitted()
    x=d["x"][d["test"]]
    mask=view_masks_v37(x,"MCAR20",seed=88,known_seed=31)
    original=threeview_feature_v37(fit,x,mask)
    nan_hidden=x.copy()
    nan_hidden[~mask]=np.nan
    changed=threeview_feature_v37(fit,nan_hidden,mask)
    np.testing.assert_array_equal(changed["q"],original["q"])
    alone=threeview_feature_v37(fit,nan_hidden[:1],mask[:1])
    np.testing.assert_array_equal(alone["q"],original["q"][:1])
    with pytest.raises(ValueError):
        threeview_feature_v37(fit,nan_hidden[:0],mask[:0])


def test_unknown_single_view_sign_flip_is_not_threeview_gauge_invariant():
    d,fit=_fitted()
    x=d["x"][d["test"]]
    mask=view_masks_v37(x,"complete",seed=99,known_seed=11)
    a=threeview_feature_v37(fit,x,mask)
    b=threeview_feature_v37(
        fit,apply_device_world_v37(x,"flip_A",seed=17),mask)
    assert b["support"].mean()<a["support"].mean()
    assert not b["not_physically_certified"] is False


def test_common_mode_artifact_can_fool_all_three_views_and_is_not_certified():
    d,fit=_fitted()
    x=d["x"][d["test"]]
    m=view_masks_v37(x,"MCAR20",seed=51,known_seed=71)
    b=threeview_feature_v37(
        fit,apply_device_world_v37(x,"shared_AB",seed=52),m)
    all_bad=threeview_feature_v37(
        fit,apply_device_world_v37(x,"shared_ABC",seed=52),m)
    assert np.isfinite(b["q"]).all()
    assert np.isfinite(all_bad["q"]).all()
    assert b["not_physically_certified"]
    assert all_bad["not_physically_certified"]
    # Do not claim all-view shared artifacts can always be detected.


def test_no_independent_view_provenance_is_hard_refusal():
    d,_=_fitted()
    x=d["x"][d["train"]]
    m=view_masks_v37(x,"MCAR20",seed=27,known_seed=72)
    with pytest.raises(ValueError,match="independent"):
        fit_centered_threeview_v37(
            x,m,independent_view_noise_documented=False)


def test_train_source_gamma_cannot_be_identified_from_equal_spectral_lags():
    d=generate_threeview_world_v37(
        mix_seed=20261217,n_train=440,n_valid=300,n_test=240,
        tied_spectral_lags=True)
    x=d["x"][d["train"]]
    m=view_masks_v37(x,"MCAR20",seed=27,known_seed=72)
    with pytest.raises(ValueError,match="ABSTAIN"):
        fit_centered_threeview_v37(
            x,m,independent_view_noise_documented=True)
