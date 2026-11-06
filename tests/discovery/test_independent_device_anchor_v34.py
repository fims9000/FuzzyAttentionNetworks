"""V34 independent device phantom, never self-anchored from source drift."""
from __future__ import annotations

import numpy as np
import pytest

from conceptfan_discovery.independent_device_anchor_v34 import (
    fit_two_reference_regimes,synthetic_known_reference_training_only,
)
from conceptfan_discovery.adaptive_higher_order_concepts_v34 import (
    fit_measured_definition,select_order_from_train,
)


def test_two_real_registered_synthetic_reference_regimes_produce_third_order_boundary():
    anchor=synthetic_known_reference_training_only(seed=20263420)
    assert anchor.reference_calibration_verified_by_simulator
    assert not anchor.calibration_is_physical_concept_identity
    assert anchor.n_per_regime==(240,240)
    assert 0.14<anchor.half_difference_boundary_uncertainty<0.22
    rng=np.random.default_rng(17)
    a=rng.normal(0,1,size=200)
    d=fit_measured_definition(
        "measured_source",(4,5,6),a,
        np.full(200,.12),
        independently_calibrated_boundary_spread=(
            anchor.half_difference_boundary_uncertainty),
        certified_environments=2,
        trusted_external_anchor=True,
    )
    assert select_order_from_train(d)=="restricted_interval_type3"


def test_unknown_train_source_distribution_shift_is_not_external_device_calibration():
    rng=np.random.default_rng(7)
    a=rng.normal(size=180)
    observed_spread=np.full(180,.12)
    with pytest.raises(ValueError,match="Do not invent"):
        fit_measured_definition(
            "g",(4,5),a,observed_spread,
            independently_calibrated_boundary_spread=.5,
            certified_environments=2,
            trusted_external_anchor=False,
        )
    with pytest.raises(ValueError,match="Independent known-reference"):
        fit_two_reference_regimes(
            np.ones(150),np.ones(150)*1.25,known_reference_value=0.,
            regime_names=("unlabelledA","unlabelledB"),
            independently_verified=False,
        )


def test_device_anchor_does_not_use_any_hidden_labeled_dataset():
    a=synthetic_known_reference_training_only(seed=20263420)
    b=synthetic_known_reference_training_only(seed=20263420)
    assert a==b
    assert a.reference_input_level==0.
    assert a.independent_regime_identifiers!=("train","test")
