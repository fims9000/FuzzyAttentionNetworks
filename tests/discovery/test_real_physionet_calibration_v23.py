"""V23: real PhysioNet protocol safety; no bundled patient records in tests."""
from __future__ import annotations

import inspect
import numpy as np
import pytest
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score,roc_auc_score

from conceptfan_discovery.data import DiscoveryData,DiscoverySplit
from conceptfan_discovery.multiscale_physiology_v10 import MultiScaleDualProcessConceptFAN
from conceptfan_discovery.experiment_unseen_test_v22c import _fingerprint
from conceptfan_discovery.experiment_real_physionet_calibration_v23 import (
    split_original_train_for_disjoint_calibration,
    fit_reserved_400_train_affine,
    collect_native_real_logits,
    EXPECTED_CHANNELS,EXPECTED_PHYSICAL_K,
    EXPECTED_PUBLIC_SET_A_SHA256,EXPECTED_PUBLIC_OUTCOMES_A_SHA256,
    MODEL_SEEDS,
)
import conceptfan_discovery.experiment_real_physionet_calibration_v23 as v23


def _fake_original_real_training(n=2400,channels=12):
    rng=np.random.default_rng(23)
    n_steps=36
    history=rng.normal(size=(n,n_steps,channels)).astype(np.float32)
    missing=(rng.random(size=history.shape)>.32).astype(np.float32)
    history*=missing
    future=rng.normal(size=(n,12,channels)).astype(np.float32)
    future_mask=np.ones_like(future,dtype=np.float32)
    labels=np.zeros(n,dtype=np.float32)
    labels[:int(n*.15)]=1.
    rng.shuffle(labels)
    train=DiscoverySplit(
        x=history,mask=missing,future=future,future_mask=future_mask,
        y=labels,oracle_states=None,
        episode_ids=np.arange(10000,10000+n,dtype=np.int64),
    )
    val=DiscoverySplit(
        x=train.x[:600].copy(),mask=train.mask[:600].copy(),
        future=train.future[:600].copy(),
        future_mask=train.future_mask[:600].copy(),
        y=train.y[:600].copy(),oracle_states=None,
        episode_ids=np.arange(20000,20600,dtype=np.int64),
    )
    return DiscoveryData(
        train=train,validation=val,
        receipt={"channel_names":list(EXPECTED_CHANNELS)},
        medians=np.zeros(channels),scales=np.ones(channels),
    )


def test_exact_stratified_train_split_and_unique_episode_id_hashes():
    data=_fake_original_real_training()
    model_data,cal,audit=split_original_train_for_disjoint_calibration(data)
    second,cal2,audit2=split_original_train_for_disjoint_calibration(data)
    assert len(model_data.train.y)==2000
    assert len(cal.y)==400
    assert len(model_data.validation.y)==600
    assert np.array_equal(model_data.train.episode_ids,second.train.episode_ids)
    assert np.array_equal(cal.episode_ids,cal2.episode_ids)
    assert set(model_data.train.episode_ids).isdisjoint(cal.episode_ids)
    assert set(model_data.train.episode_ids)|set(cal.episode_ids)==set(data.train.episode_ids)
    assert model_data.train.oracle_states is None
    assert cal.oracle_states is None
    assert model_data.receipt["old_original_calibration_and_test_still_sealed"]
    assert audit==audit2
    assert len(audit["train_id_sha256"])==64
    assert len(audit["calibration_id_sha256"])==64
    assert abs(model_data.train.y.mean()-cal.y.mean())<.03
    assert audit["same_original_2400_patient_train_label_budget"]


def test_reject_duplicate_patient_ids_or_latent_oracle_in_real_calibration_training():
    data=_fake_original_real_training()
    data.train.episode_ids[1]=data.train.episode_ids[0]
    with pytest.raises(ValueError,match="Invalid original"):
        split_original_train_for_disjoint_calibration(data)
    other=_fake_original_real_training()
    other.train.oracle_states=np.zeros((2400,36,3),dtype=np.float32)
    with pytest.raises(ValueError,match="Invalid original"):
        split_original_train_for_disjoint_calibration(other)


def test_fit_reserved_real_train_calibrator_preserves_ranking_and_additivity():
    rng=np.random.default_rng(44)
    logits=rng.normal(size=400)
    y=(rng.random(400)<expit(2.3*logits-.8)).astype(np.float32)
    data=_fake_original_real_training()
    model_data,cal,_=split_original_train_for_disjoint_calibration(data)
    fitted=fit_reserved_400_train_affine(
        logits,y,
        calibration_episode_ids=cal.episode_ids,
        native_gradient_episode_ids=model_data.train.episode_ids,
    )
    assert fitted.optimizer_success
    assert fitted.n_independent_train_calibration_cases==400
    assert .05<=fitted.slope<=50.
    assert -6<=fitted.intercept<=6.
    assert not fitted.clinically_meaningful_concepts_certified
    assert fitted.fitted_train_bce<fitted.initial_train_bce
    original=expit(logits)
    calibrated=fitted.probabilities(logits)
    assert average_precision_score(y,original)==pytest.approx(
        average_precision_score(y,calibrated),abs=1e-10,
    )
    assert roc_auc_score(y,original)==pytest.approx(
        roc_auc_score(y,calibrated),abs=1e-10,
    )

    torch.manual_seed(23)
    model=MultiScaleDualProcessConceptFAN(
        observable_loadings=np.eye(12,8),
        channels=EXPECTED_CHANNELS,
        forecast_horizon=12,n_observation_concepts=3,
        scales=(1,4,12,24),
    ).eval()
    checksum=_fingerprint(model)
    a,b,old_error,_=collect_native_real_logits(model,cal)
    c,d,_,new_error=collect_native_real_logits(model,cal,fitted)
    np.testing.assert_array_equal(a,c)
    assert old_error<1e-6
    assert new_error<1e-6
    assert _fingerprint(model)==checksum


def test_real_calibrator_rejects_shared_ids_or_invalid_outcomes():
    rng=np.random.default_rng(55)
    logits=rng.normal(size=400)
    y=(rng.random(400)<expit(logits)).astype(np.float32)
    kw={
        "model_logits":logits,
        "calibration_targets":y,
        "calibration_episode_ids":np.arange(400),
        "native_gradient_episode_ids":np.arange(400,2400),
    }
    with pytest.raises(ValueError,match="ABSTAIN"):
        fit_reserved_400_train_affine(**{
            **kw,"native_gradient_episode_ids":np.arange(200,2200),
        })
    with pytest.raises(ValueError,match="ABSTAIN"):
        fit_reserved_400_train_affine(**{
            **kw,"calibration_targets":np.ones(400),
        })
    with pytest.raises(ValueError,match="ABSTAIN"):
        fit_reserved_400_train_affine(**{
            **kw,"model_logits":np.full(400,np.nan),
        })
    with pytest.raises(ValueError,match="ABSTAIN"):
        fit_reserved_400_train_affine(**{
            **kw,"calibration_episode_ids":np.full(400,9),
        })


def test_frozen_real_public_source_and_validation_contract():
    assert EXPECTED_PHYSICAL_K==8
    assert len(EXPECTED_CHANNELS)==12
    assert EXPECTED_CHANNELS==(
        "DiasABP","HR","MAP","NIDiasABP","NIMAP","NISysABP",
        "SysABP","Temp","Urine","Weight","RespRate","GCS",
    )
    assert MODEL_SEEDS==(11,23,37)
    assert EXPECTED_PUBLIC_SET_A_SHA256=="1433638192fda622e8bb11a4559b8af2dbb1c69f0e1c3b5b606cc12aabd0a4c5"
    assert EXPECTED_PUBLIC_OUTCOMES_A_SHA256=="2613ea60ccda29f87571a7d6b09ad130858a8ccd66325bd073365925026883c2"
    source=inspect.getsource(v23.run)
    assert "_load_and_match_v10_patient_cohort" in source
    assert "original_v10_status" in source
    assert "calibration_logits,_,cal_decomp_error,_=collect_native_real_logits" in source
    assert "original_calibration_and_test_sealed" in source
    assert "physical_causal_source_identified" in source
    assert "if _fingerprint(model)!=checksum:" in source
    assert "fit_tabular_baseline(" in source
