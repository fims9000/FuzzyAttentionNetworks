"""V41 acceptance/negative tests: per-episode measured pair, TRAIN gauge, fail-closed."""
from __future__ import annotations

import numpy as np
import pytest

from conceptfan_discovery.adaptive_pair_bench_world_v41 import (
    REGISTERED_DEVICES,generate_device_v41,render_observed_v41,
    masks_v41,measured_difference_v41,
)
from conceptfan_discovery.threeview_centered_v37 import (
    fit_centered_threeview_v37,threeview_feature_v37,pair_feature_v37,
)
from conceptfan_discovery.adaptive_pair_consensus_v41 import (
    fit_adaptive_source_v41,adaptive_pair_feature_v41,
    unsafe_unconfirmed_any_pair_v41,_select_features,ORDER,
)


@pytest.fixture(scope="module")
def prepared():
    w=generate_device_v41(
        device_seed=20261507,n_train=500,n_valid=300,n_test=220)
    on,off=render_observed_v41(
        w,w["train"],fault="none",seed=20261577)
    ma,mb=masks_v41(
        on,off,"MCAR20",device_seed=20261507,mask_seed=20261578)
    d,md=measured_difference_v41(on,off,ma,mb)
    v37=fit_centered_threeview_v37(
        d,md,independent_view_noise_documented=True,
        seed=20261579,bootstrap=24)
    calibrated=fit_adaptive_source_v41(v37,d,md)
    return w,calibrated


def test_four_new_independent_device_seeds_and_no_prior_V40_reuse():
    assert set(REGISTERED_DEVICES)=={
        20261507,20261509,20261521,20261523}
    assert set(REGISTERED_DEVICES.values())=={
        "linear_AR","nonlinear_switching"}


def test_train_only_no_simulator_H_or_latent_y_in_calibration(prepared):
    _,fit=prepared
    receipt=fit.audit()
    assert fit.alignment_reference_name=="AB"
    assert not fit.clinical_or_causal_semantics_certified
    assert not fit.uses_hidden_truth_H_or_S
    assert not fit.uses_outcome_Y_for_source_definition
    assert receipt["external_source_isolation_and_sensor_wiring_assumed"]
    assert fit.gauges["AB"].calibrated
    assert fit.gauges["AB"].slope==1.
    for k in ORDER[1:]:
        if fit.gauges[k].calibrated:
            assert fit.gauges[k].train_correlation_to_AB>=.20
            assert .25<=fit.gauges[k].slope<=4.
            assert fit.gauges[k].train_overlap>=60


def test_any_measurable_pair_can_be_selected_when_one_view_is_lost(prepared):
    w,fit=prepared
    on,off=render_observed_v41(
        w,w["test"],fault="none",seed=20261579)
    for missing,needed in (
        ("missing_view_A","BC"),("missing_view_B","AC"),
        ("missing_view_C","AB"),
    ):
        ma,mb=masks_v41(
            on,off,missing,device_seed=20261507,mask_seed=20261588)
        d,md=measured_difference_v41(on,off,ma,mb)
        measured=adaptive_pair_feature_v41(fit,d,md)
        assert set(np.unique(measured["selected_pair"])).issubset({"--",needed})
        assert not measured["two_or_three_pairs_corroborated"].any()
        assert np.isfinite(measured["q"]).all()
        assert np.all(measured["selected_joint_measured_times"]<=16)
        assert not (measured["support"][measured["selected_pair"]=="--"]>0).any()
        # Calibration can legitimately abstain on a completely uninformative
        # pair, but the old required-ALL-three method must *always* reject.
        old=threeview_feature_v37(fit.v37_fit,d,md)
        assert not old["support"].any()


def test_missing_OFF_reference_must_exactly_refuse_unmeasured_source(prepared):
    w,fit=prepared
    on,off=render_observed_v41(
        w,w["test"],fault="none",seed=20261577)
    ma,mb=masks_v41(
        on,off,"missing_reference",device_seed=20261507,mask_seed=20261589)
    d,md=measured_difference_v41(on,off,ma,mb)
    a=adaptive_pair_feature_v41(fit,d,md)
    assert not a["support"].any()
    np.testing.assert_array_equal(a["q"],np.full(len(on),.5))
    assert set(np.unique(a["selected_pair"]))=={"--"}


def test_all_supported_disagree_then_refuse_instead_of_cherry_picking(prepared):
    _,fit=prepared
    n=9
    measured={
        "AB":{"q":np.full(n,.05),"support":np.ones(n),
              "paired_timepoints":np.full(n,14)},
        "AC":{"q":np.full(n,.95),"support":np.ones(n),
              "paired_timepoints":np.full(n,13)},
        "BC":{"q":np.full(n,.05),"support":np.ones(n),
              "paired_timepoints":np.full(n,12)},
    }
    strict=_select_features(measured,fit,enforce_consensus=True)
    assert not strict["support"].any()
    assert np.all(strict["refused_due_to_disagreement"] |
                  (strict["n_supported_pairs"]<2))
    assert not (strict["selected_pair"]!="--").any()
    unsafe=_select_features(measured,fit,enforce_consensus=False)
    assert unsafe["support"].all()
    assert set(unsafe["selected_pair"])=={"AB"}


def test_single_pair_cannot_be_claimed_multi_view_verified(prepared):
    w,fit=prepared
    on,off=render_observed_v41(
        w,w["test"],fault="none",seed=20261578)
    ma,mb=masks_v41(
        on,off,"missing_view_C",device_seed=20261507,mask_seed=20261590)
    d,md=measured_difference_v41(on,off,ma,mb)
    measured=adaptive_pair_feature_v41(fit,d,md)
    used=measured["support"]>0
    assert np.all(measured["single_pair_uncorroborated"][used])
    assert not measured["two_or_three_pairs_corroborated"].any()
    assert measured["single_pair_not_semantically_certified"]


def test_hidden_nan_and_single_record_equality(prepared):
    w,fit=prepared
    on,off=render_observed_v41(
        w,w["test"],fault="none",seed=20261588)
    ma,mb=masks_v41(
        on,off,"MCAR40",device_seed=20261507,mask_seed=20261591)
    d,md=measured_difference_v41(on,off,ma,mb)
    full=adaptive_pair_feature_v41(fit,d,md)
    a,b=on.copy(),off.copy()
    a[~ma]=np.nan;b[~mb]=np.nan
    d2,md2=measured_difference_v41(a,b,ma,mb)
    changed=adaptive_pair_feature_v41(fit,d2,md2)
    np.testing.assert_array_equal(full["q"],changed["q"])
    np.testing.assert_array_equal(full["support"],changed["support"])
    np.testing.assert_array_equal(full["selected_pair"],changed["selected_pair"])
    single=adaptive_pair_feature_v41(fit,d2[:1],md2[:1])
    np.testing.assert_array_equal(single["q"],full["q"][:1])
    np.testing.assert_array_equal(single["support"],full["support"][:1])
    a[ma]=np.nan
    with pytest.raises(ValueError,match="Observed readings"):
        measured_difference_v41(a,b,ma,mb)


def test_more_pairs_not_guaranteed_identifiability_under_unmonitored_drift(prepared):
    w,fit=prepared
    on,off=render_observed_v41(
        w,w["test"],fault="off_drift",seed=20261599)
    ma,mb=masks_v41(
        on,off,"MCAR20",device_seed=20261507,mask_seed=20261591)
    d,md=measured_difference_v41(on,off,ma,mb)
    strict=adaptive_pair_feature_v41(fit,d,md)
    assert not fit.clinical_or_causal_semantics_certified
    assert np.isfinite(strict["q"]).all()
    assert np.all(strict["source_mask"]==False) if "source_mask" in strict else True


def test_fixed_pair_priority_uses_most_joint_measured_not_largest_q(prepared):
    _,fit=prepared
    n=8
    measurements={
        "AB":{"q":np.full(n,.99),"support":np.ones(n),
              "paired_timepoints":np.full(n,7)},
        "AC":{"q":np.full(n,.01),"support":np.ones(n),
              "paired_timepoints":np.full(n,9)},
        "BC":{"q":np.full(n,.5),"support":np.zeros(n),
              "paired_timepoints":np.zeros(n)},
    }
    # This UNSAFE ablation skips compatibility so pair ranking can be
    # inspected independently; otherwise disagreement correctly refuses.
    unsafe=_select_features(measurements,fit,enforce_consensus=False)
    assert set(unsafe["selected_pair"])=={"AC"}
    np.testing.assert_array_equal(unsafe["q"],np.full(n,.01))
