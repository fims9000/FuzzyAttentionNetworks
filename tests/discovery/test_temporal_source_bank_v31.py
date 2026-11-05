"""V31: temporal ordering, masked source integrity and honest gate baselines."""
from __future__ import annotations

import numpy as np
import pytest

from conceptfan_discovery.phase_pulse_benchmark_v31 import (
    SEED,CASES,STRESS,phase_pulse_world,
    device_environment,measured_mask,temporal_sensor_features,
)
from conceptfan_discovery.temporal_source_candidate_bank_v31 import (
    candidate_bank,measured_temporal_activation,
)
from conceptfan_discovery.auto_measured_concept_proposals import (
    MeasuredGroup,form_train_measured_groups,
)


def _identity_source():
    return MeasuredGroup(
        candidate_id="known_test_group",
        sensor_indices=(4,5,6),sensor_signs=(1,1,1),
        train_means=(0.,0.,0.),
        train_stds=(1.,1.,1.),
        min_fold_abs_corr=.9,
        train_episodes=1500,
    )


def test_two_equal_mean_opposite_phase_sequences_have_different_temporal_concept():
    x=np.zeros((2,16,12),dtype=np.float32)
    phase=np.r_[np.ones(8),-np.ones(8)]
    x[0,:,4:7]=phase[:,None]
    x[1,:,4:7]=-phase[:,None]
    m=np.ones_like(x,dtype=bool)
    group=_identity_source()
    static,s_static=measured_temporal_activation(group,x,m,statistic="static_mean")
    phase_q,s_phase=measured_temporal_activation(group,x,m,statistic="half_contrast")
    np.testing.assert_allclose(static,[.5,.5],atol=1e-14,rtol=0)
    assert phase_q[0]>.70 and phase_q[1]<.30
    np.testing.assert_allclose(s_phase,[1.,1.])
    # This is the precise counterexample to the original source mean pool:
    # same full mean and availability, different temporal risk evidence.


def test_temporal_pulse_generator_is_not_ar_and_labels_use_pulse_order():
    d=phase_pulse_world(seed=SEED,n_train=400,n_valid=300,n_test=300)
    assert d["generator_family"]=="non_autoregressive_signed_two_phase_pulse_contrast"
    assert d["pulse_has_zero_true_full_history_mean"]
    assert d["x"].shape==(1000,16,12)
    y=d["outcomes"]
    assert not np.array_equal(y["physical"],y["complete"])
    assert not np.array_equal(y["artifact"],y["physical"])
    assert np.isfinite(d["x"]).all()
    # The two source amplitudes originate from separate random variables,
    # not a true latent/oracle feature available to the predictor.


def test_label_free_masked_correlations_find_multiple_pulse_readout_groups():
    d=phase_pulse_world(seed=SEED,n_train=450,n_valid=300,n_test=300)
    tr=d["train"]
    m=measured_mask(d["x"][tr],"MCAR20",seed=20269301)
    groups=form_train_measured_groups(
        d["x"][tr],m,known_sensor_indices=(0,1,2,3),
        seed=20269304,min_pair_abs_corr=.60,
        min_joint_cells_per_fold=300,max_candidates=5)
    found={frozenset(g.sensor_indices) for g in groups.candidates}
    assert {frozenset((4,5,6)),frozenset((7,8)),frozenset((9,10))}<=found
    assert groups.outcomes_used_for_grouping is False
    assert groups.excluded_known_sensors==(0,1,2,3)
    bank=candidate_bank(groups.candidates)
    assert len(bank)==2*len(groups.candidates)
    assert all(not z.candidate_definition_uses_outcomes for z in bank)
    assert all(not z.concept_source_physical_identity_verified for z in bank)


def test_temporal_candidate_never_reads_unobserved_payloads():
    d=phase_pulse_world(seed=SEED,n_train=400,n_valid=300,n_test=300)
    tr,te=d["train"],d["test"]
    mtr=measured_mask(d["x"][tr],"MCAR20",seed=20269301)
    source=form_train_measured_groups(
        d["x"][tr],mtr,known_sensor_indices=(0,1,2,3)
    ).candidates[0]
    x=d["x"][te].copy()
    m=measured_mask(x,"MCAR40",seed=20269902)
    original,coverage=measured_temporal_activation(
        source,x,m,statistic="half_contrast")
    x[~m]=float("nan")
    same,coverage2=measured_temporal_activation(
        source,x,m,statistic="half_contrast")
    np.testing.assert_array_equal(original,same)
    np.testing.assert_array_equal(coverage,coverage2)
    m[:,:,:]=False
    neutral,zero=measured_temporal_activation(
        source,x,m,statistic="half_contrast")
    np.testing.assert_array_equal(neutral,np.full(len(neutral),.5))
    np.testing.assert_array_equal(zero,np.zeros(len(zero)))


@pytest.mark.parametrize("regime",("none","benign","flip_physical","flip_artifact"))
def test_sensor_shifts_keep_original_k2_inputs_exactly_unchanged(regime):
    d=phase_pulse_world(seed=SEED,n_train=300,n_valid=300,n_test=300)
    x=d["x"][d["test"]]
    shifted=device_environment(x,regime)
    np.testing.assert_array_equal(shifted[:,:,:4],x[:,:,:4])
    assert len(CASES)==5 and len(STRESS)==5


def test_strong_raw_temporal_baseline_gets_halfwave_information():
    d=phase_pulse_world(seed=SEED,n_train=300,n_valid=300,n_test=300)
    x=d["x"][d["test"]]
    m=measured_mask(x,"MCAR20",seed=20269901)
    f=temporal_sensor_features(x,m)
    assert f.shape==(len(x),12*6)
    assert not np.array_equal(f[:,12:24],f[:,36:48])
    assert np.isfinite(f).all()


def test_temporal_source_must_have_both_halves_measured():
    group=_identity_source()
    x=np.zeros((4,16,12),dtype=np.float32)
    x[:,0:8,4:7]=1.5
    x[:,8:16,4:7]=-1.5
    m=np.ones_like(x,dtype=bool)
    m[:,:8,4:7]=False
    q,s=measured_temporal_activation(
        group,x,m,statistic="half_contrast")
    np.testing.assert_array_equal(q,np.full(4,.5))
    np.testing.assert_array_equal(s,np.zeros(4))


def test_nonbinary_observation_mask_refused():
    x=np.zeros((4,16,12),dtype=np.float32)
    m=np.ones_like(x)
    m[0,0,4]=.42
    with pytest.raises(ValueError):
        measured_temporal_activation(_identity_source(),x,m,
                                     statistic="half_contrast")
