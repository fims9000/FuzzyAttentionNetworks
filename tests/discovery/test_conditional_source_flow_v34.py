"""V34 flow matching trained on masked measured source proxy, never oracle."""
from __future__ import annotations

import numpy as np
import pytest

from conceptfan_discovery.auto_measured_concept_proposals import (
    form_train_measured_groups,
)
from conceptfan_discovery.masked_temporal_template import (
    fit_masked_rank1_template,
)
from conceptfan_discovery.phase_pulse_benchmark_v31 import (
    phase_pulse_world,measured_mask,
)
from conceptfan_discovery.conditional_source_flow_v34 import (
    train_conditional_measured_source_flow,flow_amplitude_samples,
    observed_flow_context,
)


def _source(seed=20261126):
    d=phase_pulse_world(seed=seed,n_train=500,n_valid=300,n_test=240)
    tr=d["train"]
    m=measured_mask(d["x"][tr],"MCAR20",seed=20263401)
    proposed=form_train_measured_groups(
        d["x"][tr],m,known_sensor_indices=(0,1,2,3),
        seed=20263404)
    group=next(g for g in proposed.candidates if set(g.sensor_indices)=={4,5,6})
    templ=fit_masked_rank1_template(group,d["x"][tr],m)
    return d,m,group,templ


def test_flow_uses_only_observed_source_measurements_and_has_train_only_calibration():
    d,m,group,template=_source()
    flow=train_conditional_measured_source_flow(
        template,group,d["x"][d["train"]],m,seed=20263401,steps=20)
    assert flow.n_fit_samples>150
    assert flow.n_calibration_samples>30
    assert not flow.training_used_outcomes
    assert not flow.training_used_latent_source_truth
    assert not flow.final_flow_is_a_bayesian_posterior
    assert flow.heldout_train_width_multiplier>0
    assert 0<=flow.heldout_train_nominal_80pct_coverage<=1
    x=d["x"][d["test"]]
    mask=measured_mask(x,"MCAR40",seed=20263903)
    p=flow_amplitude_samples(flow,template,group,x,mask,seed=54,n_samples=16)
    assert p["empirical_samples"].shape==(len(x),16)
    assert p["amplitude_interval_80pct_proxy_calibrated"].shape==(len(x),2)
    assert np.all(p["amplitude_sd"]>=0)
    assert p["samples_are_generated_not_observed"]
    assert np.all(np.isfinite(p["amplitude_mean"]))


def test_conditional_flow_never_reads_hidden_payload_or_uses_missing_as_evidence():
    d,m,group,template=_source()
    flow=train_conditional_measured_source_flow(
        template,group,d["x"][d["train"]],m,seed=20263401,steps=20)
    x=d["x"][d["test"]]
    mask=measured_mask(x,"MCAR20",seed=25)
    original=flow_amplitude_samples(flow,template,group,x,mask,seed=333,n_samples=16)
    hidden=x.copy()
    hidden[~mask]=float("nan")
    heldout=flow_amplitude_samples(flow,template,group,hidden,mask,seed=333,n_samples=16)
    np.testing.assert_array_equal(original["amplitude_mean"],heldout["amplitude_mean"])
    np.testing.assert_array_equal(original["empirical_samples"],heldout["empirical_samples"])
    no_group=mask.copy()
    no_group[:,:,group.sensor_indices]=False
    missing=flow_amplitude_samples(flow,template,group,hidden,no_group,
                                   seed=333,n_samples=16)
    assert not missing["eligible"].any()
    np.testing.assert_array_equal(missing["amplitude_mean"],np.zeros(len(x)))
    np.testing.assert_array_equal(missing["amplitude_sd"],np.zeros(len(x)))
    assert np.isfinite(missing["empirical_samples"]).all()
    # Samples may exist as CONDITIONAL PRIOR draws, but MUST NOT be used
    # as evidence or passed into the final risk head without measurements.


def test_flow_condition_is_source_measurement_only_not_target_labels():
    d,m,group,template=_source()
    c,info=observed_flow_context(template,group,d["x"][d["train"]],m)
    assert c.shape==(500,4)
    assert np.all(info>=0)
    assert not np.array_equal(c[:,0],d["outcomes"]["physical"][d["train"]])
    with pytest.raises(ValueError):
        train_conditional_measured_source_flow(
            template,group,d["x"][d["train"]],m,steps=4)
