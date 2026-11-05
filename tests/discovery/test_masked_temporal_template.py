"""V32 test contracts: outcome-blind template, missingness, and known failures."""
from __future__ import annotations

import numpy as np
import pytest

from conceptfan_discovery.auto_measured_concept_proposals import (
    MeasuredGroup,form_train_measured_groups,
)
from conceptfan_discovery.masked_temporal_template import (
    fit_masked_rank1_template,project_masked_template,template_concept,
)
from conceptfan_discovery.masked_template_benchmark import (
    SEED,CASES,own_draw,test_device_observations as simulate_device_observations,
)
from conceptfan_discovery.phase_pulse_benchmark_v31 import (
    measured_mask,
)
from conceptfan_discovery.temporal_source_candidate_bank_v31 import (
    measured_temporal_activation,
)


def _known(n=700,seed=20261118):
    rng=np.random.default_rng(seed)
    a=rng.uniform(-1.5,1.5,n)
    phase=np.r_[np.ones(8),-np.ones(8)]
    x=np.zeros((n,16,12),dtype=np.float32)
    for j in (4,5,6):
        x[:,:,j]=(a[:,None]*phase[None,:]+
                    rng.normal(0,.29,size=(n,16))).astype(np.float32)
    m=rng.random(x.shape)>=.2
    g=MeasuredGroup(candidate_id="source_train_only",
       sensor_indices=(4,5,6),sensor_signs=(1,1,1),
       train_means=(0.,0.,0.),train_stds=(1.,1.,1.),
       min_fold_abs_corr=.85,train_episodes=n)
    return x,m,g,a,phase


def test_frozen_template_recovers_pulse_order_without_outcome_or_generator_amplitude():
    x,m,g,a,phase=_known()
    profile=fit_masked_rank1_template(g,x,m)
    assert profile.template_used_target_outcomes is False
    assert profile.template_used_hidden_generator_factors is False
    assert profile.semantic_physical_identity_certified is False
    assert profile.number_train_subjects==700
    assert profile.train_residual_variance>0
    assert profile.shrinkage_ridge>0
    phi=np.array(profile.phi)
    assert np.corrcoef(phi,phase)[0,1]>.97
    score=project_masked_template(profile,g,x,m)
    assert np.corrcoef(score['amplitude_estimate'],a)[0,1]>.92
    assert np.isfinite(score['amplitude_standard_error']).all()
    assert score["source_temporal_template_sha256"]


def test_projection_uses_observed_cells_only_and_falls_back_for_none():
    x,m,g,_,_=_known(n=650)
    profile=fit_masked_rank1_template(g,x[:500],m[:500])
    q,s=template_concept(profile,g,x[500:],m[500:])
    hidden=x[500:].copy()
    hidden[~m[500:]]=np.nan
    q2,s2=template_concept(profile,g,hidden,m[500:])
    np.testing.assert_array_equal(q,q2)
    np.testing.assert_array_equal(s,s2)
    missing=m[500:].copy()
    missing[:,:,4:7]=False
    q0,s0=template_concept(profile,g,hidden,missing)
    np.testing.assert_array_equal(q0,np.full(len(q0),.5))
    np.testing.assert_array_equal(s0,np.zeros(len(s0)))


def test_template_handles_one_missing_half_where_half_contrast_must_reject():
    x,m,g,_,_= _known(n=650)
    profile=fit_masked_rank1_template(g,x[:500],m[:500])
    damaged=m[500:].copy()
    # Exactly ONE half has all of its channel observations, and the
    # other half is absent. Preserve the half with >=50% of the
    # TRAIN-learned squared template energy; this is the prespecified
    # >=.50 information contract, not TEST-threshold tuning.
    phi=np.asarray(profile.phi)
    if np.sum(phi[:8]**2)>=np.sum(phi[8:]**2):
        damaged[:,:8,4:7]=True
        damaged[:,8:,4:7]=False
    else:
        damaged[:,:8,4:7]=False
        damaged[:,8:,4:7]=True
    source=x[500:]
    q_static,s_static=measured_temporal_activation(
        g,source,damaged,statistic="static_mean")
    q_half,s_half=measured_temporal_activation(
        g,source,damaged,statistic="half_contrast")
    q_template,s_template=template_concept(
        profile,g,source,damaged)
    assert not np.any(s_half>=.5)
    np.testing.assert_array_equal(q_half,np.full(len(q_half),.5))
    assert (s_template>=.5).mean()>.95
    assert (np.abs(q_template-.5)>.02).mean()>.40
    assert np.isfinite(q_static).all()


def test_train_only_group_unknown_mapping_and_new_draw():
    d=own_draw(n_train=450,n_valid=300,n_test=250)
    tr=d["train"]
    masks=measured_mask(d["x"][tr],"MCAR20",seed=20270301)
    g=form_train_measured_groups(
        d["x"][tr],masks,known_sensor_indices=(0,1,2,3),
        seed=20270304,
    )
    profiles=[fit_masked_rank1_template(x,d["x"][tr],masks)
              for x in g.candidates]
    assert len(profiles)>=2
    assert all(p.template_used_target_outcomes is False for p in profiles)
    assert d["pulse_has_zero_true_full_history_mean"]
    assert len(CASES)==6


def test_hidden_amplitude_defines_only_test_observation_deformation():
    d=own_draw(n_train=350,n_valid=300,n_test=250)
    x=d["x"][d["test"]]
    amp=d["amplitude_CONCEPTS_AUDIT_ONLY"]["physical"][d["test"]]
    shifted=simulate_device_observations(
        x,"change_pulse_shape",
        oracle_pulse_amplitude_for_GENERATOR_ONLY=amp)
    np.testing.assert_array_equal(x[:,:,:4],shifted[:,:,:4])
    assert not np.array_equal(x[:,:,4:7],shifted[:,:,4:7])
    np.testing.assert_array_equal(x[:,:,9:11],shifted[:,:,9:11])
    old=np.mean(x[:,:,4:7],axis=1)
    new=np.mean(shifted[:,:,4:7],axis=1)
    np.testing.assert_allclose(old,new,atol=1e-6,rtol=0)


def test_inconsistent_source_template_ids_raise_before_TEST():
    x,m,g,_,_=_known(n=650)
    p=fit_masked_rank1_template(g,x[:500],m[:500])
    wrong=MeasuredGroup(candidate_id="WRONG",sensor_indices=(4,5,6),
       sensor_signs=(1,1,1),train_means=(0.,0.,0.),
       train_stds=(1.,1.,1.),min_fold_abs_corr=.8,train_episodes=500)
    with pytest.raises(ValueError):
        project_masked_template(p,wrong,x[500:],m[500:])


def test_missing_source_does_not_become_observed_time_zero():
    x,m,g,_,_=_known(n=650)
    p=fit_masked_rank1_template(g,x[:500],m[:500])
    zero_mask=m[500:].copy()
    zero_mask[:,:,4:7]=False
    r=project_masked_template(p,g,x[500:],zero_mask)
    np.testing.assert_array_equal(r["measured_source_times"],np.zeros(len(r["q"])))
    assert not r["sufficient_observations"].any()
    assert r["no_hidden_source_or_outcome_labels_used"]
