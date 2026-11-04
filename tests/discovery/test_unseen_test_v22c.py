"""V22c: test sealed TEST interface without opening predeclared TEST 522203."""
from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest
import torch

from conceptfan_discovery.mixed_source_data_v22 import (
    make_observable_v22,train_only_whitening,build_native_v22_data,
)
from conceptfan_discovery.experiment_unseen_test_v22c import (
    _fingerprint,_test_split_without_oracle,
)
import conceptfan_discovery.experiment_unseen_test_v22c as v22c
from conceptfan_discovery.multiscale_physiology_v10 import MultiScaleDualProcessConceptFAN
from conceptfan_discovery.experiment_mixed_source_v22 import _native_readout


def test_test_split_helper_does_not_surface_oracle_and_uses_train_whitener(monkeypatch):
    # DO NOT instantiate reserved TEST seed 522203 in a test. Use unrelated
    # synthetic seed to exercise the exact TEST serialization path.
    monkeypatch.setattr(v22c,"FRESH_TEST_SEED",522299)
    observed=make_observable_v22("mixed_distinct_variance")
    core=replace(
        observed,
        train_x_full=observed.train_x_full[:600].copy(),
        train_labels=observed.train_labels[:600].copy(),
        train_episode_ids=observed.train_episode_ids[:600].copy(),
        train_environment=observed.train_environment[:600].copy(),
    )
    w=train_only_whitening(core)
    before=w.center.copy(),w.linear_map.copy()
    split,env,latent=v22c._test_split_without_oracle(
        "mixed_distinct_variance",w,
    )
    assert split.x.shape==(300,24,5)
    assert split.future.shape==(300,6,5)
    assert env.shape==(300,)
    assert latent.shape==(300,24,2)  # separate audit return ONLY after fit
    assert split.oracle_states is None
    assert not hasattr(split,"latent")
    assert not set(map(int,split.episode_ids))&set(map(int,core.train_episode_ids))
    assert np.array_equal(w.center,before[0])
    assert np.array_equal(w.linear_map,before[1])
    assert np.all(split.x[:,:,2:]==0.)
    assert len(set(split.episode_ids))==len(split.y)


def test_native_test_scoring_changes_no_weights_and_explanation_stays_additive():
    observed=make_observable_v22("mixed_distinct_variance")
    whitener=train_only_whitening(observed)
    data=build_native_v22_data(observed,whitener)
    torch.manual_seed(1207)
    model=MultiScaleDualProcessConceptFAN(
        observable_loadings=np.eye(5,2),
        channels=tuple(data.receipt["channel_names"]),
        forecast_horizon=6,
        n_observation_concepts=3,
        scales=(1,4,12),
        adaptive_feature_gradients=False,
    )
    old_hash=_fingerprint(model)
    score,concept,effects=_native_readout(model,data.validation)
    assert score["max_native_logit_decomposition_error"]<1e-6
    assert len(concept)==len(data.validation.y)
    assert len(effects)==len(data.validation.y)
    assert _fingerprint(model)==old_hash


def test_sealed_test_protocol_requires_full_fit_barrier_before_instantiation():
    source=inspect.getsource(v22c.run)
    barrier=source.index("if (n_trained!=36")
    creation=source.index("test,environment,truth=_test_split_without_oracle")
    assert barrier<creation
    assert "test_unsealed_after_full_training=True" in source
    assert "if _fingerprint(model)!=training_sha:" in source
    assert "if row[\"native_model_sha256_after_test\"]!=training_sha:" in source
    assert v22c.FRESH_TEST_SEED==522203
    assert v22c.CORE_TRAIN_N==600
    assert v22c.INDEPENDENT_RISK_SCREEN_N==200
    assert v22c.EXPECTED_OPPORTUNITIES==48


def test_frozen_arms_never_include_an_external_posthoc_classifier():
    assert v22c.RISK_ARMS==(
        "arbitrary_native_start","independently_screened_risk"
    )
    assert v22c.SOURCE_METHODS==(
        "pooled_pca_basis","environment_contrast_basis"
    )
    assert v22c.TRAIN_EPOCHS==10
    assert v22c.MODEL_SEEDS==(11,23,37)
    assert v22c.FRESH_TEST_N==300
