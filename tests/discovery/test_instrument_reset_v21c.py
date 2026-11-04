"""V21c: independent paired repeat distinguishes resettable artifacts, not all."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from conceptfan_discovery.provenance_data_v21 import CHANNELS
from conceptfan_discovery.candidate_admission_v17 import ProposedObservableConcept
from conceptfan_discovery.instrument_reset_data_v21c import (
    RESET_FAMILIES,make_v21c_repeat_acquisitions,
    CALIBRATION_SEED,INDEPENDENT_RESET_SEED,
)
from conceptfan_discovery.instrument_repeatability_v21c import (
    assess_candidate_after_independent_acquisition_reset,
    paired_repeatability_passport,
)


def _candidate(slot):
    return ProposedObservableConcept(
        candidate_id=f"v21c_observed_sensor_{slot}",
        direction=np.eye(len(CHANNELS))[:,slot],
        train_episode_count=350,
        positive_lag_strength=.25,
        permutation_p=.005,
        bootstrap_abs_cosine_q10=.99,
        top_signed_sensors=((CHANNELS[slot],1.),),
    )


def _screen(scenario,*,candidate=None):
    data=make_v21c_repeat_acquisitions(scenario)
    truth_slot=2 if scenario.startswith("physical_") else 3
    return assess_candidate_after_independent_acquisition_reset(
        candidate or _candidate(truth_slot),data,
        discovery_episode_ids=range(0,350),
        outcome_screen_episode_ids=range(350,650),
        reference_screen_episode_ids=range(650,900),
        permutations=99,bootstraps=99,seed=211144,
    )


@pytest.mark.parametrize("scenario",RESET_FAMILIES)
def test_acquisition_data_reproducible_disjoint_and_does_not_expose_oracle(scenario):
    one=make_v21c_repeat_acquisitions(scenario)
    two=make_v21c_repeat_acquisitions(scenario)
    assert one.first_model_visible_measurements.shape==(240,24,4)
    assert one.repeated_model_visible_measurements.shape==(240,24,4)
    assert one.first_external_reference_measurement.shape==(240,24)
    assert one.repeated_external_reference_measurement.shape==(240,24)
    assert np.array_equal(one.first_model_visible_measurements,two.first_model_visible_measurements)
    assert np.array_equal(one.repeated_model_visible_measurements,two.repeated_model_visible_measurements)
    assert len(set(one.calibration_episode_ids))==240
    assert min(one.calibration_episode_ids)>=40000
    assert not hasattr(one,"hidden_physical_source")
    assert not hasattr(one,"hidden_instrument_artifact")
    assert not hasattr(one,"y")
    assert one.no_generator_latent_states_or_outcomes_exposed
    assert one.controlled_instrument_reset_declared
    assert one.same_physical_system_held_fixed_in_simulator
    assert one.patient_histories_from_train_validation_and_test_reused is False
    assert not np.array_equal(
        one.first_model_visible_measurements,
        one.repeated_model_visible_measurements,
    )
    assert np.isfinite(one.first_model_visible_measurements).all()
    assert np.isfinite(one.repeated_model_visible_measurements).all()


@pytest.mark.parametrize("scenario",["physical_predictive","physical_nontask"])
def test_real_physical_signal_survives_independent_instrument_reset(scenario):
    report=_screen(scenario)
    assert report.status=="ACCEPT_ACQUISITION_REPEATABILITY"
    assert report.first_repeat_pearson_r>.9
    assert report.bootstrap_ci95[0]>.5
    assert report.permutation_p_two_sided<=.01
    assert report.n_independent_calibration_episodes==240
    assert not report.physical_causal_identity_established
    assert not report.persistent_commonmode_interference_ruled_out


@pytest.mark.parametrize("scenario",[
    "artifact_predictive","artifact_nontask","artifact_common_mode_reference",
])
def test_resettable_instrument_artifact_breaks_cross_acquisition_repeatability(scenario):
    report=_screen(scenario)
    assert report.status=="ABSTAIN_UNSTABLE_UNDER_INSTRUMENT_RESET"
    assert report.first_repeat_pearson_r<.65
    assert not paired_repeatability_passport(report)["clinical_physical_identity_certified"]


def test_persistent_common_mode_interference_fools_reset_as_expected():
    report=_screen("artifact_environmental_persistent")
    assert report.status=="ACCEPT_ACQUISITION_REPEATABILITY"
    assert report.first_repeat_pearson_r>.9
    assert report.before_reference_signed_r>.9
    assert report.after_reference_signed_r>.9
    assert not report.physical_causal_identity_established
    assert not report.persistent_commonmode_interference_ruled_out


def test_provenance_disjointness_and_declared_reset_are_mandatory():
    pair=make_v21c_repeat_acquisitions("physical_predictive")
    common={
        "candidate":_candidate(2),
        "pairs":pair,
        "discovery_episode_ids":range(0,350),
        "outcome_screen_episode_ids":range(350,650),
        "reference_screen_episode_ids":range(650,900),
        "permutations":99,"bootstraps":99,
    }
    with pytest.raises(ValueError,match="disjoint"):
        assess_candidate_after_independent_acquisition_reset(
            **{**common,"discovery_episode_ids":[int(pair.calibration_episode_ids[0])]},
        )
    with pytest.raises(ValueError,match="reset"):
        assess_candidate_after_independent_acquisition_reset(
            **{**common,"pairs":replace(pair,controlled_instrument_reset_declared=False)},
        )
    with pytest.raises(ValueError,match="paired acquisition"):
        assess_candidate_after_independent_acquisition_reset(
            **{**common,"pairs":replace(
                pair,repeat_recorded_mask=np.zeros_like(pair.repeat_recorded_mask))},
        )
    with pytest.raises(ValueError,match="threshold"):
        assess_candidate_after_independent_acquisition_reset(
            **{**common,"minimum_r":-.5},
        )


def test_reviewer_reference_appears_valid_within_both_acquisitions_for_common_mode():
    first=_screen("artifact_common_mode_reference")
    assert first.before_reference_signed_r>.9
    assert first.after_reference_signed_r>.9
    assert first.status=="ABSTAIN_UNSTABLE_UNDER_INSTRUMENT_RESET"
    second=_screen("artifact_environmental_persistent")
    assert second.before_reference_signed_r>.9
    assert second.after_reference_signed_r>.9
    assert second.status=="ACCEPT_ACQUISITION_REPEATABILITY"
