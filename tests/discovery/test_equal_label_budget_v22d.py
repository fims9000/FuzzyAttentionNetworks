"""V22d: paired label budget, source invariance and sealed TEST isolation."""
from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest
import torch

from conceptfan_discovery.mixed_source_data_v22 import (
    make_observable_v22,train_only_whitening,build_native_v22_data,
)
import conceptfan_discovery.experiment_equal_label_budget_v22d as v22d
from conceptfan_discovery.experiment_mixed_source_v22 import _native_readout
from conceptfan_discovery.experiment_unseen_test_v22c import _fingerprint
from conceptfan_discovery.multiscale_physiology_v10 import MultiScaleDualProcessConceptFAN


def _same_source_two_budget_datasets(family="mixed_distinct_variance"):
    original=make_observable_v22(family)
    core=replace(
        original,
        train_x_full=original.train_x_full[:600].copy(),
        train_labels=original.train_labels[:600].copy(),
        train_environment=original.train_environment[:600].copy(),
        train_episode_ids=original.train_episode_ids[:600].copy(),
    )
    whitener=train_only_whitening(core)
    data600=build_native_v22_data(core,whitener)
    data800=build_native_v22_data(original,whitener)
    return whitener,data600,data800


def test_equal_800_labels_spent_as_joint_or_screen_without_changing_source_axes():
    whitener,smaller,larger=_same_source_two_budget_datasets()
    assert whitener.trained_episodes==600
    assert len(smaller.train.y)==600
    assert len(larger.train.y)==800
    assert np.array_equal(smaller.train.x,larger.train.x[:600])
    assert np.array_equal(smaller.train.y,larger.train.y[:600])
    assert np.array_equal(smaller.train.episode_ids,larger.train.episode_ids[:600])
    assert not set(smaller.train.episode_ids)&set(larger.train.episode_ids[600:])
    assert smaller.train.oracle_states is None
    assert larger.train.oracle_states is None
    assert smaller.validation.oracle_states is None
    assert larger.validation.oracle_states is None
    assert np.array_equal(smaller.medians,larger.medians)
    assert np.array_equal(smaller.scales,larger.scales)


def test_new_test_helper_excludes_oracle_and_uses_unchanged_train_normalizer(monkeypatch):
    # This unit test must NOT open the reserved unused V22d TEST seed 522313.
    monkeypatch.setattr(v22d,"FRESH_HOLDOUT_SEED",522314)
    whitener,smaller,larger=_same_source_two_budget_datasets()
    before=(whitener.center.copy(),whitener.linear_map.copy())
    test,environment,hidden=v22d._generate_test_after_all_training(
        "mixed_distinct_variance",whitener,
    )
    assert len(test.y)==300
    assert test.x.shape==(300,24,5)
    assert test.future.shape==(300,6,5)
    assert environment.shape==(300,)
    assert hidden.shape==(300,24,2)
    assert test.oracle_states is None
    assert not hasattr(test,"latent")
    assert not set(test.episode_ids)&set(larger.train.episode_ids)
    assert np.array_equal(whitener.center,before[0])
    assert np.array_equal(whitener.linear_map,before[1])
    assert np.isfinite(test.x).all()


def test_frozen_test_cannot_be_opened_until_48_opportunity_barrier():
    source=inspect.getsource(v22d.run)
    before=source.index('if len(records)!=48 or n_fitted!=36 or n_abstained!=12:')
    after=source.index('test,environment,hidden=_generate_test_after_all_training')
    assert before<after
    assert "if _fingerprint(model)!=sha:" in source
    assert "if row[\"native_model_sha256_after_test\"]!=sha:" in source
    assert v22d.FRESH_HOLDOUT_SEED==522313
    assert v22d.FRESH_HOLDOUT_N==300
    assert v22d.BUDGET_ARMS==(
        "joint_800_labels","screened_600_fit_200_gate"
    )
    assert v22d.MODEL_SEEDS==(11,23,37)
    assert v22d.CORE_TRAIN_N==600
    assert v22d.INDEPENDENT_RISK_SCREEN_N==200


def test_native_new_test_evaluation_keeps_model_frozen():
    whitener,smaller,larger=_same_source_two_budget_datasets()
    torch.manual_seed(313)
    model=MultiScaleDualProcessConceptFAN(
        observable_loadings=np.eye(5,2),
        channels=tuple(smaller.receipt["channel_names"]),
        forecast_horizon=6,n_observation_concepts=3,
        scales=(1,4,12),adaptive_feature_gradients=False,
    )
    sha=_fingerprint(model)
    stats,concept,effects=_native_readout(model,smaller.validation)
    assert stats["max_native_logit_decomposition_error"]<1e-6
    assert len(concept)==len(smaller.validation.y)
    assert len(effects)==len(smaller.validation.y)
    assert _fingerprint(model)==sha
