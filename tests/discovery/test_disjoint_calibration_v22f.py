"""V22f: disjoint probability recalibration does not create or alter a concept."""
from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score,roc_auc_score

from conceptfan_discovery.train_only_native_calibration_v22f import (
    fit_train_only_native_affine_calibration,
    native_calibration_passport,
)
import conceptfan_discovery.experiment_disjoint_calibration_v22f as v22f
from conceptfan_discovery.mixed_source_data_v22 import (
    make_observable_v22,train_only_whitening,
)
from conceptfan_discovery.multiscale_physiology_v10 import MultiScaleDualProcessConceptFAN
from conceptfan_discovery.experiment_unseen_test_v22c import _fingerprint


def _calibration_example():
    rng=np.random.default_rng(123)
    z=rng.normal(size=150)
    y=(rng.random(150)<expit(2.5*z-.4)).astype(np.float64)
    return z,y


def _fit_calibrator(z=None,y=None):
    a,b=_calibration_example()
    return fit_train_only_native_affine_calibration(
        a if z is None else z,
        b if y is None else y,
        independent_calibration_episode_ids=np.arange(450,600),
        native_model_fit_episode_ids=np.arange(0,450),
        source_risk_screen_episode_ids=np.arange(600,800),
    )


def test_positive_native_affine_calibration_has_exactly_two_train_fit_parameters():
    z,y=_calibration_example()
    cal=_fit_calibrator(z,y)
    assert cal.optimizer_success
    assert .05<=cal.slope<=50.
    assert -6<=cal.intercept<=6
    assert cal.n_independent_train_calibration_cases==150
    assert np.isfinite(cal.initial_train_bce)
    assert np.isfinite(cal.fitted_train_bce)
    assert cal.fitted_train_bce<cal.initial_train_bce
    assert not cal.clinically_meaningful_concepts_certified
    assert not cal.new_concept_head_created
    assert cal.uses_original_native_logit_only
    passport=native_calibration_passport(cal)
    assert passport["all_source_and_observation_concept_contributions_common_positive_slope"]
    assert passport["native_direct_contributions_remain_exactly_additive"]
    assert not passport["no_concept_source_identity_or_disease_certified"] is False
    raw_prob=expit(z)
    adjusted=cal.probabilities(z)
    assert np.isfinite(adjusted).all()
    assert np.all((adjusted>0)&(adjusted<1))
    assert average_precision_score(y,raw_prob)==pytest.approx(
        average_precision_score(y,adjusted),abs=1e-11,
    )
    assert roc_auc_score(y,raw_prob)==pytest.approx(
        roc_auc_score(y,adjusted),abs=1e-11,
    )


@pytest.mark.parametrize("invalid",[
    "reuse_calibration_as_native_fit",
    "reuse_calibration_as_risk_screen",
    "nonbinary_outcome","one_class",
    "duplicate_calibration_ids","constant_native_logits",
    "nonfinite_native_logits",
])
def test_calibration_fails_closed_on_leakage_or_invalid_outcomes(invalid):
    z,y=_calibration_example()
    args={
        "native_logits":z,
        "independent_train_outcomes":y,
        "independent_calibration_episode_ids":np.arange(450,600),
        "native_model_fit_episode_ids":np.arange(0,450),
        "source_risk_screen_episode_ids":np.arange(600,800),
    }
    if invalid=="reuse_calibration_as_native_fit":
        args["native_model_fit_episode_ids"]=np.arange(450,600)
    elif invalid=="reuse_calibration_as_risk_screen":
        args["source_risk_screen_episode_ids"]=np.arange(450,650)
    elif invalid=="nonbinary_outcome":
        args["independent_train_outcomes"]=np.full(150,.5)
    elif invalid=="one_class":
        args["independent_train_outcomes"]=np.ones(150)
    elif invalid=="duplicate_calibration_ids":
        args["independent_calibration_episode_ids"]=np.zeros(150,dtype=np.int64)
    elif invalid=="constant_native_logits":
        args["native_logits"]=np.zeros(150)
    elif invalid=="nonfinite_native_logits":
        args["native_logits"]=np.full(150,np.nan)
    with pytest.raises(ValueError,match="ABSTAIN"):
        fit_train_only_native_affine_calibration(**args)


def test_scaled_physical_and_observation_contributions_stay_exactly_additive():
    observed=make_observable_v22("mixed_distinct_variance")
    train=replace(
        observed,
        train_x_full=observed.train_x_full[:600].copy(),
        train_environment=observed.train_environment[:600].copy(),
        train_labels=observed.train_labels[:600].copy(),
        train_episode_ids=observed.train_episode_ids[:600].copy(),
    )
    w=train_only_whitening(train)
    torch.manual_seed(137)
    model=MultiScaleDualProcessConceptFAN(
        observable_loadings=np.eye(5,2),
        channels=v22f.MODEL_CHANNELS,
        forecast_horizon=6,
        n_observation_concepts=3,
        scales=(1,4,12),
        adaptive_feature_gradients=False,
    ).eval()
    checksum=_fingerprint(model)
    x=torch.from_numpy(w.transform(observed.validation_x_full[:8])[:,:24])
    mask=torch.ones_like(x)
    cal=_fit_calibrator()
    with torch.no_grad():
        out=model(x,mask,process_mask=mask)
        logit,bias,physical,process=cal.decomposed_native_contributions(out)
    assert float(out.decomposition_error())<1e-6
    torch.testing.assert_close(
        logit,
        cal.slope*out.logits+cal.intercept,
        rtol=1e-6,atol=1e-6,
    )
    torch.testing.assert_close(
        logit,
        bias+physical.sum(-1)+process.sum(-1),
        rtol=1e-6,atol=1e-6,
    )
    torch.testing.assert_close(
        physical,
        cal.slope*out.physical.contributions,
        rtol=0,atol=0,
    )
    torch.testing.assert_close(
        process,
        cal.slope*out.observation_contributions,
        rtol=0,atol=0,
    )
    assert _fingerprint(model)==checksum


def test_test_seed_522333_not_generated_until_all_18_models_and_calibration_fit():
    src=inspect.getsource(v22f.run)
    barrier=src.index('if (len(cases)!=52 or n_models!=18 or n_calibrators!=18')
    reveal=src.index('raw,env,y,hidden=_simulate')
    assert reveal>barrier
    assert v22f.FROZEN_NEW_TEST_SEED==522333
    assert v22f.FROZEN_NEW_TEST_N==300
    assert v22f.SOURCE_TRAIN_HISTORIES==600
    assert v22f.NATIVE_FIT_LABELS==450
    assert v22f.CALIBRATION_LABELS==150
    assert v22f.SOURCE_RISK_LABELS==200
    assert v22f.MODEL_SEEDS==(11,23,37)
    assert v22f.TRAIN_EPOCHS==10


def test_calibration_scoring_does_not_rewrite_native_model_or_change_observed_rank():
    observed=make_observable_v22("mixed_distinct_variance")
    train=replace(
        observed,train_x_full=observed.train_x_full[:600].copy(),
        train_environment=observed.train_environment[:600].copy(),
        train_labels=observed.train_labels[:600].copy(),
        train_episode_ids=observed.train_episode_ids[:600].copy(),
    )
    w=train_only_whitening(train)
    split=v22f._make_split(
        observed.validation_x_full,
        observed.validation_labels,
        observed.validation_episode_ids,w,
    )
    torch.manual_seed(432)
    model=MultiScaleDualProcessConceptFAN(
        observable_loadings=np.eye(5,2),
        channels=v22f.MODEL_CHANNELS,
        forecast_horizon=6,n_observation_concepts=3,
        scales=(1,4,12),
        adaptive_feature_gradients=False,
    )
    checksum=_fingerprint(model)
    before,original,_,native_error,_=v22f._native_logits(model,split)
    cal=_fit_calibrator()
    after,calibrated,_,_,cal_error=v22f._native_logits(model,split,cal)
    np.testing.assert_allclose(before,after,atol=0,rtol=0)
    assert native_error<1e-6
    assert cal_error<1e-6
    assert _fingerprint(model)==checksum
    assert average_precision_score(split.y,original)==pytest.approx(
        average_precision_score(split.y,calibrated),abs=1e-10,
    )
    assert roc_auc_score(split.y,original)==pytest.approx(
        roc_auc_score(split.y,calibrated),abs=1e-10,
    )
