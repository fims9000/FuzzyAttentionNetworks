"""V33 falsification tests, no simulated TEST outcomes used for model fitting."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from conceptfan_discovery.synchronized_source_outages_v33 import (
    STRESS_V33,SIMULATOR_PHYSICAL_SENSOR_GROUP,
    synchronized_outage_mask,phase_corroborated_template,phase_concept,
    observed_payload_digest,
)
from conceptfan_discovery.concept_intervention_audit_v33 import (
    known_concept_intervention_audit,generator_conditional_probability,
    observational_indistinguishability_falsifier,
)
from conceptfan_discovery.auto_measured_concept_proposals import (
    form_train_measured_groups,MeasuredGroup,
)
from conceptfan_discovery.masked_temporal_template import (
    fit_masked_rank1_template,template_concept,
)
from conceptfan_discovery.masked_template_benchmark import (
    own_draw,
)
from conceptfan_discovery.phase_pulse_benchmark_v31 import (
    measured_mask,
)


def _trained_group(n=450):
    d=own_draw(n_train=n,n_valid=300,n_test=240)
    tr=d["train"]
    m=measured_mask(d["x"][tr],"MCAR20",seed=20271301)
    groups=form_train_measured_groups(
        d["x"][tr],m,
        known_sensor_indices=(0,1,2,3),seed=20271304,
    )
    physical=next(g for g in groups.candidates
                  if set(g.sensor_indices)=={4,5,6})
    return d,physical,fit_masked_rank1_template(
        physical,d["x"][tr],m)


@pytest.mark.parametrize("stress",STRESS_V33)
def test_synced_masks_preserve_shape_and_original_k2_observation_channel(stress):
    d=own_draw(n_train=300,n_valid=300,n_test=240)
    x=d["x"][d["test"]]
    m=synchronized_outage_mask(x,stress,seed=20271900)
    assert m.shape==x.shape and m.dtype==bool
    assert m[:,:,:4].any()
    other=measured_mask(x,"MCAR20",seed=20271900)
    if stress.startswith("synchronized_"):
        np.testing.assert_array_equal(m[:,:,:4],other[:,:,:4])
        np.testing.assert_array_equal(m[:,:,9:11],other[:,:,9:11])
    if stress=="synchronized_full16":
        assert not m[:,:,4:7].any()
    if stress=="synchronized_early8":
        assert not m[:,:8,4:7].any()
    if stress=="synchronized_late8":
        assert not m[:,8:,4:7].any()
    if stress=="synchronized_block8":
        assert np.all((~m[:,:,4:7]).any(axis=2).sum(axis=1)>=8)


def test_complete_group_outage_returns_exact_neutral_concept_both_policies():
    d,g,template=_trained_group()
    x=d["x"][d["test"]]
    m=synchronized_outage_mask(x,"synchronized_full16",seed=10)
    q,s=template_concept(template,g,x,m)
    q2,s2=phase_concept(template,g,x,m)
    np.testing.assert_array_equal(q,np.full(len(q),.5))
    np.testing.assert_array_equal(s,np.zeros(len(s)))
    np.testing.assert_array_equal(q2,np.full(len(q2),.5))
    np.testing.assert_array_equal(s2,np.zeros(len(s2)))


@pytest.mark.parametrize("stress",("synchronized_early8","synchronized_late8"))
def test_phase_guard_rejects_one_hidden_half_even_when_wls_extrapolates(stress):
    d,g,template=_trained_group()
    x=d["x"][d["test"]]
    m=synchronized_outage_mask(x,stress,seed=22)
    z=phase_corroborated_template(template,g,x,m)
    np.testing.assert_array_equal(
        z["q"],np.full(len(x),.5))
    np.testing.assert_array_equal(
        z["source_support"],np.zeros(len(x)))
    assert not z["phase_corroboration_passed"].any()
    # WLS may reconstruct the hidden half, but that requires a stationary
    # waveform assumption which is not independently verified by the data.
    rawq,raws=template_concept(template,g,x,m)
    assert np.isfinite(rawq).all()
    assert np.isfinite(raws).all()


def test_phase_guard_is_invariant_to_all_hidden_payload_changes():
    d,g,template=_trained_group()
    x=d["x"][d["test"]]
    m=synchronized_outage_mask(x,"independent_block8",seed=29)
    a=phase_corroborated_template(template,g,x,m)
    withheld=x.copy()
    withheld[~m]=float("nan")
    b=phase_corroborated_template(template,g,withheld,m)
    for field in ("q","source_support","measured_time_positions_early",
                  "measured_time_positions_late",
                  "phase_corroboration_passed"):
        np.testing.assert_array_equal(a[field],b[field])
    assert observed_payload_digest(x,m)==observed_payload_digest(withheld,m)


def test_identical_masked_histories_cannot_identify_opposite_hidden_laws():
    pair=observational_indistinguishability_falsifier()
    a=pair["world_a_measured_history"]
    b=pair["world_b_measured_history"]
    m=pair["observation_mask"]
    assert pair["world_a_true_event_probability"]>.95
    assert pair["world_b_true_event_probability"]<.05
    assert observed_payload_digest(a,m)==observed_payload_digest(b,m)
    d,g,template=_trained_group()
    for estimate in (
        lambda x:template_concept(template,g,x,m)[0],
        lambda x:phase_concept(template,g,x,m)[0],
    ):
        np.testing.assert_array_equal(estimate(a),estimate(b))


def test_intervention_exact_decomposition_and_single_coordinate_noninterference():
    q=np.array([[.15,.70],[.80,.30],[.53,.62]],dtype=np.float64)
    true=np.array([[.80,.70],[.80,.85],[.53,.62]],dtype=np.float64)
    w=np.array([2.0,-3.0])
    extra=np.array([+.11,-.43,+.19])
    rows,scores=known_concept_intervention_audit(
        q,true,original_head_bias=-.25,
        original_head_weights=w,
        additional_source_contribution=extra,
        observed_test_outcomes=np.array([1,0,1]),
        true_conditional_event_probability=np.array([.86,.12,.62]),
    )
    z=(-.25+q@w+extra)
    np.testing.assert_array_equal(scores["none"],z)
    np.testing.assert_allclose(
        scores["concept1_ONLY"]-z,w[0]*(true[:,0]-q[:,0]),
        atol=1e-15,rtol=0)
    np.testing.assert_allclose(
        scores["concept2_ONLY"]-z,w[1]*(true[:,1]-q[:,1]),
        atol=1e-15,rtol=0)
    np.testing.assert_allclose(
        scores["both_known_concepts"]-z,
        (true-q)@w,
        atol=1e-15,rtol=0)
    assert all(x.max_exact_intervention_reconstruction_error<2e-12
               for x in rows)
    assert not any(x.causal_intervention_certified for x in rows)


def test_oracle_known_coordinate_correction_can_increase_expected_risk():
    q=np.tile([.20,.5],(200,1))
    true=np.tile([.80,.5],(200,1))
    y=np.ones(200,dtype=int)
    rows,_=known_concept_intervention_audit(
        q,true,original_head_bias=.0,
        original_head_weights=np.array([-4.,0.]),
        additional_source_contribution=np.zeros(200),
        observed_test_outcomes=y,
        true_conditional_event_probability=np.full(200,.90),
    )
    first=next(x for x in rows if x.intervention=="concept1_ONLY")
    assert first.delta_expected_nll>0
    assert first.per_episode_expected_harm_rate==1.0
    assert first.mean_corrected_concept_absolute_error_before>.5
    assert first.delta_observed_nll>0


def test_oracle_conditional_risk_matches_generator_definition():
    d=own_draw(n_train=300,n_valid=300,n_test=240)
    z=d["true_known_CONCEPTS_AUDIT_ONLY"]
    p0=generator_conditional_probability(
        z,outcome_law="complete",
        physical_amplitude=d["amplitude_CONCEPTS_AUDIT_ONLY"]["physical"],
        artifact_amplitude=d["amplitude_CONCEPTS_AUDIT_ONLY"]["artifact"],
    )
    trueq=np.asarray(z,dtype=np.float64).mean(axis=1)
    np.testing.assert_allclose(
        p0,expit(3.7*(trueq[:,0]-.5)-3.6*(trueq[:,1]-.5)-.10))
    p1=generator_conditional_probability(
        z,outcome_law="physical",
        physical_amplitude=d["amplitude_CONCEPTS_AUDIT_ONLY"]["physical"],
        artifact_amplitude=d["amplitude_CONCEPTS_AUDIT_ONLY"]["artifact"],
    )
    assert not np.array_equal(p0,p1)
    with pytest.raises(ValueError):
        generator_conditional_probability(
            z,outcome_law="nonexistent",
            physical_amplitude=d["amplitude_CONCEPTS_AUDIT_ONLY"]["physical"],
            artifact_amplitude=d["amplitude_CONCEPTS_AUDIT_ONLY"]["artifact"],
        )


def test_invalid_sync_stress_and_policy_threshold_fail_closed():
    x=np.zeros((200,16,12),dtype=np.float32)
    with pytest.raises(ValueError):
        synchronized_outage_mask(x,"invented",seed=1)
    d,g,template=_trained_group()
    with pytest.raises(ValueError):
        phase_corroborated_template(
            template,g,d["x"][d["test"]],
            np.ones_like(d["x"][d["test"]],dtype=bool),
            minimum_observed_positions_per_half=1)
