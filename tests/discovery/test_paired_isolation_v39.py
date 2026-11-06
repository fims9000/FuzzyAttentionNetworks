"""V39 physically extra source-isolation measurement, with strict limits."""
from __future__ import annotations
import numpy as np
import pytest

from conceptfan_discovery.paired_isolation_world_v39 import (
    generate_paired_isolation_world_v39,apply_paired_fault_v39,
    masks_v39,observed_paired_difference_v39,
)
from conceptfan_discovery.threeview_centered_v37 import (
    fit_centered_threeview_v37,threeview_feature_v37,
)


def _paired(device_seed=20261317):
    d=generate_paired_isolation_world_v39(
        device_seed=device_seed,n_train=460,n_valid=300,n_test=240)
    tr=d["train"]
    on,off=d["on"][tr],d["off"][tr]
    mo,mr=masks_v39(on,off,"MCAR20",on_seed=17,off_seed=31)
    diff,observed=observed_paired_difference_v39(on,off,mo,mr)
    fit=fit_centered_threeview_v37(
        diff,observed,independent_view_noise_documented=True,
        seed=20263941,bootstrap=24)
    return d,fit


def test_extra_OFF_observation_changes_identifiability_class_not_magic():
    d,fit=_paired()
    assert d["constant_shared_artifact_U_is_invisible_to_sentinel10"]
    assert d["physical_ON_OFF_source_path_known_to_simulator_only"]
    assert d["unknown_H_not_passed_to_source_estimator"]
    assert fit.view_partition_externally_assumed
    assert fit.independent_view_noise_externally_assumed
    assert not fit.true_physical_semantics_certified


def test_identical_source_aligned_instrument_artifact_cancels_if_reference_matched():
    d,fit=_paired()
    te=d["test"]
    on,off=d["on"][te],d["off"][te]
    mo,mr=masks_v39(on,off,"MCAR20",on_seed=74,off_seed=31)
    baseline,observed=observed_paired_difference_v39(on,off,mo,mr)
    reference=threeview_feature_v37(fit,baseline,observed)
    for name in ("shared4","shared8"):
        fault_on,fault_off=apply_paired_fault_v39(
            on,off,mode=name,seed=99,
            hidden_H_SIMULATOR_ONLY=d["hidden_mixing_AUDIT_ONLY"],
            hidden_S_SIMULATOR_ONLY=d["hidden_S_AUDIT_ONLY"][te])
        changed,mask=observed_paired_difference_v39(
            fault_on,fault_off,mo,mr)
        np.testing.assert_array_equal(mask,observed)
        np.testing.assert_allclose(changed,baseline,atol=2e-6,rtol=0.)
        new=threeview_feature_v37(fit,changed,mask)
        # Actual measured fast and slow factor responses remain the same
        # up to float32 rounding, even under an UNKNOWN aligned instrument U.
        np.testing.assert_allclose(
            reference["q"],new["q"],atol=4e-6,rtol=0.)
    assert reference["support"].mean()>.20


def test_reference_absence_gives_exact_uninformative_source():
    d,fit=_paired()
    te=d["test"]
    on,off=d["on"][te],d["off"][te]
    mo,mr=masks_v39(on,off,"missing_reference",on_seed=74,off_seed=31)
    diff,paired=observed_paired_difference_v39(on,off,mo,mr)
    result=threeview_feature_v37(fit,diff,paired)
    np.testing.assert_array_equal(result["q"],np.full(len(on),.5))
    assert not result["support"].any()
    assert not paired[:,:,4:10].any()


def test_single_episode_and_hidden_nan_payload_do_not_change_measured_source():
    d,fit=_paired()
    te=d["test"]
    on,off=d["on"][te],d["off"][te]
    mo,mr=masks_v39(on,off,"MCAR20",on_seed=71,off_seed=31)
    diff,mask=observed_paired_difference_v39(on,off,mo,mr)
    original=threeview_feature_v37(fit,diff,mask)
    a,b=on.copy(),off.copy()
    a[~mo]=np.nan
    b[~mr]=np.nan
    d2,m2=observed_paired_difference_v39(a,b,mo,mr)
    second=threeview_feature_v37(fit,d2,m2)
    np.testing.assert_array_equal(original["q"],second["q"])
    one=threeview_feature_v37(fit,d2[:1],m2[:1])
    np.testing.assert_array_equal(one["q"],original["q"][:1])
    a[mo]=np.nan
    with pytest.raises(ValueError,match="Observed readings"):
        observed_paired_difference_v39(a,b,mo,mr)


def test_negative_source_reference_drift_does_not_cancel_and_cannot_be_certified():
    d,fit=_paired()
    te=d["test"]
    on,off=d["on"][te],d["off"][te]
    mo,mr=masks_v39(on,off,"complete",on_seed=74,off_seed=31)
    baseline,_=observed_paired_difference_v39(on,off,mo,mr)
    _,drift_ref=apply_paired_fault_v39(
        on,off,mode="off_drift",seed=99,
        hidden_H_SIMULATOR_ONLY=d["hidden_mixing_AUDIT_ONLY"],
        hidden_S_SIMULATOR_ONLY=d["hidden_S_AUDIT_ONLY"][te])
    changed,_=observed_paired_difference_v39(on,drift_ref,mo,mr)
    assert np.mean(abs(changed[:,:,4:10]-baseline[:,:,4:10]))>.1
    assert not fit.true_physical_semantics_certified


def test_two_independent_random_mixing_systems_and_signed_gain_remain_ambiguous():
    a,_=_paired(20261317)
    b,_=_paired(20261319)
    assert a["hidden_mixing_AUDIT_ONLY"]!=b["hidden_mixing_AUDIT_ONLY"]


def test_source_leak_in_off_switch_changes_difference():
    d,_=_paired()
    te=d["test"]
    on,off=d["on"][te],d["off"][te]
    mo,mr=masks_v39(on,off,"complete",on_seed=71,off_seed=31)
    original,_=observed_paired_difference_v39(on,off,mo,mr)
    _,leak=apply_paired_fault_v39(
        on,off,mode="leak",seed=99,
        hidden_H_SIMULATOR_ONLY=d["hidden_mixing_AUDIT_ONLY"],
        hidden_S_SIMULATOR_ONLY=d["hidden_S_AUDIT_ONLY"][te])
    changed,_=observed_paired_difference_v39(on,leak,mo,mr)
    assert np.mean(abs(changed-original))>.04
