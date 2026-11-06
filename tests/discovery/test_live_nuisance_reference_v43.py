"""V43 fault detection, false alarm, raw physical-R fairness and masking."""
from __future__ import annotations
import json
import numpy as np
import pytest

from conceptfan_discovery.live_nuisance_world_v43 import (
    REGISTERED_DEVICES,generate_device_v43,render_observed_v43,
    masks_v43,measured_current_nuisance_monitor_v43,
)
from conceptfan_discovery.live_nuisance_monitor_v43 import (
    fit_live_nuisance_monitor_v43,monitor_measurements_v43,
    evaluate_live_nuisance_monitor_v43,gate_physical_reference_v43,
    save_live_monitor_v43,load_live_monitor_v43,
)
from conceptfan_discovery.physical_reference_raw_baselines_v43 import (
    features_measured_reference_v43,fit_raw_reference_v43,
    predict_raw_reference_v43,
)


@pytest.fixture(scope="module")
def prepared():
    w=generate_device_v43(
        device_seed=20261707,n_train=1000,n_valid=800,n_test=200)
    def observed(ids,fault="none",monitor_fault="honest",seed=901):
        return measured_current_nuisance_monitor_v43(
            w,ids,physical_fault=fault,
            monitor_fault=monitor_fault,same_seed_as_ON_OFF=seed)
    fit=fit_live_nuisance_monitor_v43(
        observed(w["train"],seed=301),
        observed(w["valid1"],seed=302),
        observed(w["valid2"],seed=303))
    return w,fit,observed


def test_new_independent_device_seeds():
    assert set(REGISTERED_DEVICES)=={
        20261707,20261709,20261721,20261723}


def test_healthy_frozen_monitor_has_train_and_two_valid_receipts(prepared):
    w,model,observed=prepared
    assert model.train_joint_n==1000
    assert model.valid1_joint_n==400 and model.valid2_joint_n==400
    assert model.passed_both_VALIDs
    assert 0<model.train_threshold_RMS<.3
    assert model.valid1_false_alarm_fraction<=.03
    assert model.valid2_false_alarm_fraction<=.03
    assert model.source_background_monitor_is_new_external_instrument
    assert model.no_Y_R_E_S_H_in_monitor_fitting
    assert model.no_current_sensor_isolation_or_truth_certification


def test_live_detects_future_only_OFF_drift_not_shared_background(prepared):
    w,monitor,observed=prepared
    ix=w["test"]
    healthy=evaluate_live_nuisance_monitor_v43(
        monitor,*observed(ix,"none",seed=1001))
    drift=evaluate_live_nuisance_monitor_v43(
        monitor,*observed(ix,"off_drift",seed=1001))
    common=evaluate_live_nuisance_monitor_v43(
        monitor,*observed(ix,"common8",seed=1001))
    assert healthy["accepted_matching_background"].mean()>.90
    assert drift["accepted_matching_background"].mean()<.10
    np.testing.assert_array_equal(
        healthy["current_MONITOR_RMS"],common["current_MONITOR_RMS"])
    assert drift["alarm_TRUE_observed_background_difference"].mean()>.90


def test_blind_external_monitor_does_NOT_detect_actual_OFF_drift(prepared):
    w,monitor,observed=prepared
    ix=w["test"]
    honest=evaluate_live_nuisance_monitor_v43(
        monitor,*observed(ix,"off_drift","honest",seed=1012))
    blind=evaluate_live_nuisance_monitor_v43(
        monitor,*observed(ix,"off_drift","blind_drift",seed=1012))
    healthy=evaluate_live_nuisance_monitor_v43(
        monitor,*observed(ix,"none","honest",seed=1012))
    assert honest["accepted_matching_background"].mean()<.10
    assert blind["accepted_matching_background"].mean()>.90
    np.testing.assert_array_equal(
        blind["current_MONITOR_RMS"],healthy["current_MONITOR_RMS"])


def test_false_monitor_alarm_refuses_healthy_physics(prepared):
    w,monitor,observed=prepared
    ix=w["test"]
    honest=evaluate_live_nuisance_monitor_v43(
        monitor,*observed(ix,"none","honest",seed=1030))
    false=evaluate_live_nuisance_monitor_v43(
        monitor,*observed(ix,"none","false_off_alarm",seed=1030))
    assert honest["accepted_matching_background"].mean()>.90
    assert false["accepted_matching_background"].mean()<.10


def test_missing_nuisance_monitor_causes_frozen_source_refusal(prepared):
    w,monitor,observed=prepared
    ix=w["test"]
    missing=evaluate_live_nuisance_monitor_v43(
        monitor,*observed(ix,"none","missing",seed=1015))
    assert not missing["accepted_matching_background"].any()
    assert not missing["measured_monitor_sufficient"].any()
    assert set(missing["decision_reason"])=={"REFUSE_MONITOR_UNAVAILABLE"}
    fake_prior={
        "physical_reference_allowed":np.ones(len(ix),dtype=bool),
        "physical_reference_estimated_energy":np.full(len(ix),.7),
        "physical_reference_interval_lower":np.full(len(ix),.4),
        "physical_reference_interval_upper":np.full(len(ix),1.),
        "physical_reference_reason":np.repeat("V42_ACCEPTED",len(ix)),
    }
    gated=gate_physical_reference_v43(fake_prior,missing)
    assert not gated["physical_reference_allowed"].any()
    assert np.isnan(gated["physical_reference_estimated_energy"]).all()


def test_corrupted_VALID_monitor_refuses_all_deployment_claims(prepared):
    w,model,observed=prepared
    bad=fit_live_nuisance_monitor_v43(
        observed(w["train"],seed=301),
        observed(w["valid1"],seed=302),
        observed(w["valid2"],"none","false_off_alarm",seed=303))
    assert not bad.passed_both_VALIDs
    later=evaluate_live_nuisance_monitor_v43(
        bad,*observed(w["test"],seed=401))
    assert not later["accepted_matching_background"].any()


def test_monitor_mask_nan_is_ignored_only_when_explicitly_missing(prepared):
    w,m,observed=prepared
    a,b,ma,mb=observed(w["test"],seed=500)
    ma[:20,:5]=False;mb[:20,:7]=False
    d1=monitor_measurements_v43(a,b,ma,mb)
    a[~ma]=np.nan;b[~mb]=np.nan
    d2=monitor_measurements_v43(a,b,ma,mb)
    for first,second in zip(d1,d2):
        np.testing.assert_array_equal(first,second)
    a[ma]=np.nan
    with pytest.raises(ValueError,match="Strict actually observed"):
        monitor_measurements_v43(a,b,ma,mb)


def test_monitor_state_JSON_reopens_without_patient_readings(prepared,tmp_path):
    w,m,observed=prepared
    state=tmp_path/"calibrated_live_nuisance.json"
    save_live_monitor_v43(state,m)
    restored=load_live_monitor_v43(state)
    assert restored.audit()==m.audit()
    original=evaluate_live_nuisance_monitor_v43(
        m,*observed(w["test"],seed=900))
    rebuilt=evaluate_live_nuisance_monitor_v43(
        restored,*observed(w["test"],seed=900))
    for key in ("accepted_matching_background",
                "current_MONITOR_RMS","decision_reason"):
        np.testing.assert_array_equal(original[key],rebuilt[key])
    assert "hidden_source" not in state.read_text()
    assert "true_physical" not in state.read_text()
    with pytest.raises(FileExistsError):
        save_live_monitor_v43(state,m)


def test_equal_R300_raw_baseline_and_mask_firewall(prepared):
    w,monitor,_=prepared
    ix=w["train"][:300]
    x,z=render_observed_v43(w,ix,fault="none",seed=77)
    ma,mb=masks_v43(
        x,z,"MCAR20",device_seed=20261707,mask_seed=81)
    f,s=features_measured_reference_v43(x,z,ma,mb)
    x[~ma]=np.nan;z[~mb]=np.nan
    fhidden,shidden=features_measured_reference_v43(
        x,z,ma,mb)
    np.testing.assert_array_equal(f,fhidden)
    np.testing.assert_array_equal(s,shidden)
    R=w["independent_reference_readout_CALIBRATION_ONLY"][ix]
    for method in ("RAW_RIDGE_R300","RAW_HGB_R300"):
        model=fit_raw_reference_v43(method,f,R)
        p=predict_raw_reference_v43(model,f[:7])
        assert p.shape==(7,) and np.isfinite(p).all()
        assert np.all(p>=0)
    with pytest.raises(ValueError,match="EXACT"):
        fit_raw_reference_v43("RAW_HGB_R700",f,R)
