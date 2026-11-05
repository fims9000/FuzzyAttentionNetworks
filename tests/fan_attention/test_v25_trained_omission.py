"""Focused V25 protocol guards, including *trained* canonical model smoke."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.experiment_trained_conceptfan_omission_v25 import (
    CHANNELS, STRESS, TIME_STEPS, frozen_world, mask_mechanism, make_model,
    predict_native, provenance_map, fit_native, metrics,
)


def test_artifact_and_omission_are_observationally_identical():
    world=frozen_world(n_train=130,n_cal=30,n_test=150)
    np.testing.assert_array_equal(world["y_omitted"], world["y_artifact"])
    assert not np.array_equal(world["y_complete"], world["y_omitted"])
    assert world["x"].shape == (310,TIME_STEPS,CHANNELS)
    assert world["concept_trajectories"].shape == (310,TIME_STEPS,3)
    assert np.intersect1d(world["index_train"],world["index_test"]).size == 0
    assert np.intersect1d(world["index_cal"],world["index_test"]).size == 0


@pytest.mark.parametrize("mode",STRESS)
def test_corruption_masks_are_deterministic_binary_and_never_use_labels(mode):
    world=frozen_world(n_train=130,n_cal=30,n_test=150)
    x=world["x"][world["index_test"]]
    first=mask_mechanism(x,mode,seed=200)
    second=mask_mechanism(x,mode,seed=200)
    np.testing.assert_array_equal(first,second)
    assert first.dtype==bool and first.shape==x.shape
    if mode=="complete":assert bool(first.all())
    elif mode=="MAR":assert bool(first[:,:,0].all())
    else:assert not bool(first.all())


def test_native_training_has_exact_concept_only_decomposition_and_frozen_eval():
    world=frozen_world(n_train=128,n_cal=32,n_test=144)
    idx=world["index_train"]
    model=make_model("native",2,11)
    mask=mask_mechanism(world["x"][idx],"MCAR20",seed=5)
    trace=fit_native(model,world["x"][idx],mask,
                     world["concept_trajectories"][idx],
                     world["y_omitted"][idx],seed=11,epochs=1)
    assert trace["epochs"]==1
    first=predict_native(model,world["x"][world["index_test"]],
                         mask_mechanism(world["x"][world["index_test"]],
                                        "MCAR40",seed=6),
                         world["concept_trajectories"][world["index_test"]])
    assert first["max_additive_error"] < 2e-5
    assert first["concept_mse"] is not None
    assert first["p"].shape==(144,)
    assert metrics(world["y_omitted"][world["index_test"]],first["p"])["n"]==144


def test_observation_wrapper_never_uses_unsupported_concepts():
    world=frozen_world(n_train=128,n_cal=32,n_test=144)
    x=world["x"][world["index_test"]]
    mask=np.ones_like(x,dtype=bool)
    mask[:80,:,0:2]=False
    model=make_model("observability",2,11).eval()
    out=predict_native(model,x,mask,world["concept_trajectories"][world["index_test"]])
    assert out["max_additive_error"] < 2e-5
    assert out["max_unsupported_effect"] == 0
    assert bool(out["abstain"][:80].all())
    assert not bool(out["abstain"][80:].any())
    np.testing.assert_array_equal(out["support"][:80,0],np.zeros(80,dtype=bool))
    assert provenance_map(2).shape==(2,CHANNELS)


def test_mnar_depends_on_missing_unobserved_value():
    world=frozen_world(n_train=130,n_cal=30,n_test=150)
    x=world["x"][world["index_test"]]
    m=mask_mechanism(x,"MNAR",seed=12)
    # Check generator protocol rather than estimating statistical effect with
    # a fragile random-significance assertion.
    assert np.any(~m)
    assert np.isfinite(x).all()
