"""V22e: deterministic true external baselines and no heldout/oracle leakage."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from conceptfan_discovery.mixed_source_data_v22 import make_observable_v22
import conceptfan_discovery.experiment_external_predictors_v22e as ext


def test_exact_thirty_temporal_features_never_see_future_or_hidden_source():
    observed=make_observable_v22("mixed_distinct_variance")
    features=ext.observed_temporal_summaries(observed.train_x_full)
    assert features.shape==(800,30)
    raw=observed.train_x_full.copy()
    raw[:,24:,:]+=100000.
    np.testing.assert_allclose(
        features,ext.observed_temporal_summaries(raw),atol=0,rtol=0,
    )
    np.testing.assert_allclose(
        features[:,:5],
        observed.train_x_full[:,18:24,:].mean(axis=1),atol=1e-6,
    )
    np.testing.assert_allclose(
        features[:,5:10],
        observed.train_x_full[:,12:18,:].mean(axis=1),atol=1e-6,
    )
    assert not np.isnan(features).any()
    with pytest.raises(ValueError,match="finite"):
        ext.observed_temporal_summaries(np.full((90,24,5),float("nan")))


def test_external_models_are_standard_sklearn_fixed_hyperparameter_predictors():
    models=ext._external_estimators()
    assert set(models)=={
        "logistic_30_temporal_summaries","hgb_30_temporal_summaries",
    }
    assert models["logistic_30_temporal_summaries"].named_steps[
        "logisticregression"
    ].C==1.
    assert models["hgb_30_temporal_summaries"].max_iter==200
    assert models["hgb_30_temporal_summaries"].max_leaf_nodes==15
    assert models["hgb_30_temporal_summaries"].random_state==11


def test_external_scikit_learn_predictors_work_on_train_labels_and_fingerprint_unchanged():
    observed=make_observable_v22("mixed_distinct_variance")
    x=ext.observed_temporal_summaries(observed.train_x_full)
    y=observed.train_labels.astype(int)
    for name,model in ext._external_estimators().items():
        model.fit(x,y)
        fingerprint=ext._pickle_fingerprint(model)
        p=model.predict_proba(x[:100])[:,1]
        assert len(p)==100
        assert np.isfinite(p).all()
        assert np.all((p>=0)&(p<=1))
        assert ext._pickle_fingerprint(model)==fingerprint
        metrics=ext._predictive_metrics(y[:100],p)
        assert 0<=metrics["test_auprc"]<=1
        assert 0<=metrics["test_auroc"]<=1
        assert metrics["test_brier"]>=0
        assert metrics["test_logloss"]>=0


def test_predeclared_unused_test_seed_cannot_be_evaluated_before_all_models_trained():
    text=inspect.getsource(ext.run)
    barrier=text.index("if (n_native!=18 or n_external!=8 or n_abstained!=6")
    unseal=text.index("raw,environment,y,latent=_simulate")
    assert barrier<unseal
    assert 'if _fingerprint(model)!=checksum:' in text
    assert 'if row["native_model_sha256_after_test"]!=checksum:' in text
    assert ext.FRESH_PREDICTIVE_TEST_SEED==522323
    assert ext.FRESH_PREDICTIVE_TEST_N==300
    assert ext.CORE_TRAIN_N==600
    assert ext.INDEPENDENT_RISK_SCREEN_N==200
    assert ext.TRAIN_EPOCHS==10
    assert ext.MODEL_SEEDS==(11,23,37)


def test_no_arbitrary_model_extra_oracle_feature_or_environment_in_baselines():
    assert len(ext.METHODS)==4
    assert ext.METHODS[:2]==(
        "native_pca_600fit_200screen",
        "native_environment_600fit_200screen",
    )
    assert ext.METHODS[2:]==(
        "logistic_30_temporal_summaries",
        "hgb_30_temporal_summaries",
    )
    # No environmental ID or hidden state is appended to the 30 raw
    # temporal features; TEST future information is not used either.
    assert ext.observed_temporal_summaries(
        np.zeros((1,24,5),dtype=np.float32)
    ).shape==(1,30)
