"""Independent TRAIN pilot must choose sensor dictionary before any rank result."""
from __future__ import annotations
import numpy as np
import pytest

from conceptfan_discovery.pilot_temporal_sensors_v5 import (
    choose_common_temporal_sensors,time_pair_support,
)


def test_100_percent_observed_sensors_are_all_eligible():
    mask=np.ones((140,36,4))
    selected=choose_common_temporal_sensors(
        mask,["a","b","c","d"],lags=(1,2),
    )
    assert selected["status"]=="PILOT_SENSORS_ELIGIBLE"
    assert selected["selected_indices"]==[0,1,2,3]
    assert selected["min_eligible_strata_within_selected"]==34
    assert selected["all_selected_sensor_pairs_adequate_in_pilot"]
    assert selected["outcomes_or_measurement_values_used"] is False


def test_rare_channel_excluded_before_rank_without_corrupting_other_sensors():
    rng=np.random.default_rng(110)
    mask=np.ones((160,36,4))
    mask[:,:,3]=(rng.random((160,36))<.1).astype(float)
    selected=choose_common_temporal_sensors(
        mask,["a","b","c","rare"],lags=(1,2),
    )
    assert selected["selected_names"]==["a","b","c"]
    assert selected["excluded_names"]==["rare"]
    assert selected["all_selected_sensor_pairs_adequate_in_pilot"]


def test_disjoint_patient_observation_panels_do_not_look_jointly_observed():
    m=np.zeros((200,36,2))
    m[:100,:,0]=1.
    m[100:,:,1]=1.
    result=choose_common_temporal_sensors(
        m,["group1_only","group2_only"],lags=(1,2),
    )
    assert result["status"]=="ABSTAIN_INSUFFICIENT_PILOT_TEMPORAL_SUPPORT"
    assert len(result["selected_indices"])==1


def test_known_additional_thinning_requires_more_pilot_joint_coverage():
    rng=np.random.default_rng(17)
    m=(rng.random((180,36,4))<.53).astype(float)
    lenient=choose_common_temporal_sensors(
        m,["a","b","c","d"],worst_additional_observation_retention=1.0,
    )
    strict=choose_common_temporal_sensors(
        m,["a","b","c","d"],worst_additional_observation_retention=.4,
    )
    assert len(strict["selected_indices"])<=len(lenient["selected_indices"])
    assert strict["pilot_min_original_pair_count"]>lenient["pilot_min_original_pair_count"]


def test_invalid_pilot_channel_data_fail_without_fabricated_k():
    mask=np.ones((100,36,3))
    with pytest.raises(ValueError):
        choose_common_temporal_sensors(mask,["a","b"])
    with pytest.raises(ValueError):
        choose_common_temporal_sensors(mask,["a","b","c"],worst_additional_observation_retention=.02)
    with pytest.raises(ValueError):
        time_pair_support(mask,lags=(1,1))
