"""V38 external probe + independent wired sentinel, no clinical claims."""
from __future__ import annotations

import numpy as np
import pytest

from conceptfan_discovery.exogenous_probe_world_v38 import (
    generate_exogenous_world_v38,apply_artifact_v38,mask_world_v38,
)
from conceptfan_discovery.threeview_centered_v37 import (
    fit_centered_threeview_v37,threeview_feature_v37,
)
from conceptfan_discovery.exogenous_probe_gate_v38 import (
    fit_sentinel_train_only_v38,negative_control_runtime_v38,
    fit_probe_intervention_v38,source_with_independent_evidence_v38,
)


def _fixture(seed=20261231):
    d=generate_exogenous_world_v38(
        device_seed=seed,n_train=460,n_valid=300,n_test=240,
        per_actuator=180)
    x=d["x"]
    train=d["train"]
    m=mask_world_v38(x[train],"MCAR20",seed=20263801,known_seed=20263881)
    s=fit_centered_threeview_v37(
        x[train],m,independent_view_noise_documented=True,
        seed=20263803,bootstrap=24)
    sentinel=fit_sentinel_train_only_v38(x[train],m)
    probe=fit_probe_intervention_v38(
        s,d["independent_calibration_onoff"],seed=20263810)
    return d,s,sentinel,probe


def test_independent_probe_has_no_access_to_hidden_H_or_task_labels():
    d,s,monitor,probe=_fixture()
    cal=d["independent_calibration_onoff"]
    assert set(cal)=={"target","nuisance"}
    for k,c in cal.items():
        assert set(c)=={
            "off","on","mask","known_actuator_command",
            "physical_source_identity_external_assumption",
        }
        assert c["off"].shape==c["on"].shape==c["mask"].shape
        assert c["known_actuator_command"].shape==(180,16)
    assert probe.independent_actuator_calibration_episodes_per_type==180
    assert not probe.latent_H_or_S_supplied_to_calibrator
    assert not probe.biological_causal_semantics_proven
    assert 0<=probe.abs_sentinel_command_response_correlation<.15
    assert monitor.independently_wired_instrument_artifact_monitor_assumed


def test_monitored_shared_artifact_detected_but_bypassed_monitor_is_not():
    d,s,mon,probe=_fixture()
    x=d["x"][d["test"]]
    m=mask_world_v38(x,"complete",seed=20263887,known_seed=11)
    healthy=negative_control_runtime_v38(mon,x,m)
    observed=negative_control_runtime_v38(
        mon,apply_artifact_v38(x,"shared_ABC_monitored",seed=808),m)
    blind=negative_control_runtime_v38(
        mon,apply_artifact_v38(x,"shared_ABC_unmonitored",seed=808),m)
    assert healthy["eligible"].mean()>.95
    assert observed["eligible"].mean()<.15
    np.testing.assert_array_equal(healthy["eligible"],blind["eligible"])
    assert np.isfinite(blind["sentinel_centered_variance"]).all()


def test_missing_sentinel_or_view_means_no_new_concept_even_when_probe_works():
    d,s,mon,probe=_fixture()
    x=d["x"][d["test"]]
    for stress in ("missing_sentinel","missing_C","missing_all_views"):
        m=mask_world_v38(x,stress,seed=20263887,known_seed=13)
        gated=source_with_independent_evidence_v38(s,mon,probe,x,m)
        assert not gated["support"].any()
        np.testing.assert_array_equal(gated["q"],np.full(len(x),.5))


def test_hidden_nan_and_single_episode_identical_to_batch_predictions():
    d,s,mon,probe=_fixture()
    x=d["x"][d["test"]]
    m=mask_world_v38(x,"MCAR20",seed=20263887,known_seed=13)
    whole=source_with_independent_evidence_v38(s,mon,probe,x,m)
    hidden=x.copy()
    hidden[~m]=np.nan
    changed=source_with_independent_evidence_v38(s,mon,probe,hidden,m)
    np.testing.assert_array_equal(whole["q"],changed["q"])
    np.testing.assert_array_equal(whole["support"],changed["support"])
    one=source_with_independent_evidence_v38(s,mon,probe,hidden[:1],m[:1])
    np.testing.assert_array_equal(one["q"],whole["q"][:1])
    np.testing.assert_array_equal(one["support"],whole["support"][:1])


def test_probe_leaked_to_negative_control_is_detected_and_rejected():
    d,s,mon,probe=_fixture()
    orig=d["independent_calibration_onoff"]
    leak={}
    for k,record in orig.items():
        leak[k]={j:np.array(v,copy=True) if isinstance(v,np.ndarray)
                 else v for j,v in record.items()}
    leak["target"]["on"][:,:,10]+=(
        1.8*leak["target"]["known_actuator_command"])
    contaminated=fit_probe_intervention_v38(s,leak,seed=20263810)
    assert contaminated.abs_sentinel_command_response_correlation>.8
    assert not contaminated.passed


def test_constant_sensor_bias_does_not_change_centered_source_or_negative_control():
    d,s,mon,probe=_fixture()
    x=d["x"][d["test"]]
    m=mask_world_v38(x,"MCAR20",seed=20263887,known_seed=13)
    clean=source_with_independent_evidence_v38(s,mon,probe,x,m)
    fault=source_with_independent_evidence_v38(
        s,mon,probe,apply_artifact_v38(x,"offset_all",seed=12),m)
    np.testing.assert_array_equal(clean["support"],fault["support"])
    np.testing.assert_allclose(clean["q"],fault["q"],atol=5e-6,rtol=0)


def test_independent_sentinel_false_alarm_has_coverage_cost():
    d,s,mon,probe=_fixture()
    x=d["x"][d["test"]]
    m=mask_world_v38(x,"complete",seed=20263887,known_seed=13)
    clean=source_with_independent_evidence_v38(s,mon,probe,x,m)
    fault=source_with_independent_evidence_v38(
        s,mon,probe,apply_artifact_v38(x,"sentinel_only",seed=999),m)
    assert not fault["support"].any()
    assert clean["instrument_control"]["eligible"].mean()>.95


def test_sentinel_provenance_is_not_inferable_from_X_without_external_wiring():
    d,s,mon,probe=_fixture()
    assert not mon.physical_source_identity_certified
    assert not s.true_physical_semantics_certified
    assert probe.known_actuator_mapping_externally_supplied



def test_theoretically_indistinguishable_source_vs_unmonitored_aligned_artifact():
    """Falsifier: the same observed X can have different actual source energy.

    The original mixing H is used ONLY by the simulator creating TWO
    structural worlds; no new algorithm, risk head, calibration, or TEST
    selection gets access to it.
    """
    from conceptfan_discovery.observational_equivalence_v38 import (
        source_vs_unmonitored_aligned_error,
    )
    worlds=source_vs_unmonitored_aligned_error(seed=20263870,n=240)
    a=worlds["physical_world_observed_X"]
    b=worlds["artifact_world_observed_X"]
    np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)
    power_gap=(
        worlds["physical_source_energy_with_perturbation"]-
        worlds["source_energy_without_perturbation"])
    assert float(np.median(power_gap))>1.0
    assert worlds["sentinel_cannot_see_aligned_artifact"]
    assert worlds["calibration_from_previous_time_can_be_identical"]
    assert worlds["known_external_actuator_is_not_run_at_TEST"]
    # Apply the real calibrated ConceptFAN measurement path to both
    # worlds; it cannot distinguish identical observed patient data.
    _,source,monitor,probe=_fixture()
    m=mask_world_v38(a,"MCAR20",seed=77,known_seed=17)
    pred1=source_with_independent_evidence_v38(source,monitor,probe,a,m)
    pred2=source_with_independent_evidence_v38(source,monitor,probe,b,m)
    np.testing.assert_allclose(pred1["q"],pred2["q"],atol=1e-12,rtol=0)
    np.testing.assert_array_equal(pred1["support"],pred2["support"])
    assert not pred1["source_physically_causal_or_biologically_named"]
