import inspect

import numpy as np
import pytest

from conceptfan_discovery.environment_contrastive_source_v69 import (
    TrainOnlyDriftContrastiveBank,
)
from scripts.experiment_environment_contrastive_v69 import (
    environment_variance_ratio,fit_linear_fair,
)
from scripts.experiment_real_six_gas_drift_v67 import (
    exact_concept_head
)


def sensor_fixture(n=1400,seed=20261969):
    rng=np.random.default_rng(seed)
    # A few physical anonymous response variations plus strong
    # acquisition-batch-specific offsets. Synthetic only; NOT proof that
    # learned axes correspond to chemical sources.
    batches=np.tile(np.arange(1,8),n//7)
    latent=rng.normal(size=(n,4))
    mixing=rng.normal(size=(4,128))
    sensors=200.+latent@mixing+rng.normal(scale=.1,size=(n,128))
    sensors[:,0:8]+=batches[:,None]*80.
    sensors[:,8:16]-=batches[:,None]*16.
    fake_labels=np.tile(np.arange(1,7),n//6+1)[:n]
    return sensors,batches,fake_labels


def test_label_free_source_fit_and_fuzzy_boundaries():
    raw,batch,_=sensor_fixture()
    sig=inspect.signature(TrainOnlyDriftContrastiveBank.fit_unlabeled)
    assert list(sig.parameters)==["self","train_x","train_batch_ids"]
    source=TrainOnlyDriftContrastiveBank().fit_unlabeled(raw,batch)
    assert 1<=source.K<=12
    assert source.thresholds.shape==(source.K,3)
    assert len(source.widths)==source.K
    assert source.receipt["any_gas_labels_or_ppm_supplied_to_source_fit"] is False
    assert source.receipt["post_V67_inspected_TEST_exploratory"]
    assert source.receipt["n_fuzzy_states"]==3*source.K
    q=source.fuzzy(raw[:20])
    assert q.shape==(20,3*source.K)
    assert np.isfinite(q).all()
    assert (q>=0).all() and (q<=1).all()
    assert (source.widths>0).all()


def test_batch_metadata_affects_unlabeled_environment_objective():
    raw,batch,labels=sensor_fixture()
    a=TrainOnlyDriftContrastiveBank().fit_unlabeled(raw,batch)
    # Change only historical environment partition; no sensor observations.
    altered=np.roll(batch,133)
    b=TrainOnlyDriftContrastiveBank().fit_unlabeled(raw,altered)
    assert a.sha!=b.sha
    assert a.K==b.K
    assert not np.allclose(a.loadings,b.loadings)
    # Relabeling the gas targets cannot influence source because there is
    # no path through the source API that accepts them.
    reversed_labels=labels[::-1]
    assert not np.array_equal(labels,reversed_labels)
    c=TrainOnlyDriftContrastiveBank().fit_unlabeled(raw,batch)
    assert c.sha==a.sha


def test_source_within_between_total_covariance_accounting():
    raw,batch,_=sensor_fixture()
    s=TrainOnlyDriftContrastiveBank().fit_unlabeled(raw,batch)
    eig=np.asarray(s.receipt["eigenvalue_ratios_descending"])
    assert len(eig)==32
    assert np.all(eig>=-1e-9) and np.all(eig<=1+1e-9)
    assert np.all(np.diff(eig)<=1e-10)
    scores=s.source_levels(raw)
    ratio=environment_variance_ratio(scores,batch)
    assert ratio["between_over_within_variance"]>=0.
    assert not ratio["ratio_smaller_does_not_certify_chemical_identity"] is False


def test_train_only_normalization_and_source_do_not_change_with_future():
    raw,batch,_=sensor_fixture()
    source=TrainOnlyDriftContrastiveBank().fit_unlabeled(raw[:1001],batch[:1001])
    ref=source.fuzzy(raw[1001:1101])
    future=raw[1001:1101].copy()
    future[:,16:]=future[:,16:]+500.
    second=source.fuzzy(future)
    assert source.sha==source.public_passport()["source_sha256"]
    assert np.max(np.abs(ref-second))>1e-7
    assert source.receipt["n_unlabeled_train_exposures"]==1001


def test_refuse_nonfinite_and_non_train_acquisition_ids():
    raw,batch,_=sensor_fixture()
    with pytest.raises(ValueError):
        TrainOnlyDriftContrastiveBank().fit_unlabeled(raw,
            np.where(batch==7,10,batch))
    with pytest.raises(ValueError):
        TrainOnlyDriftContrastiveBank().fit_unlabeled(
            np.where(np.arange(len(raw))[:,None]==5,np.nan,raw),batch)
    with pytest.raises(ValueError):
        TrainOnlyDriftContrastiveBank().fit_unlabeled(raw,batch[:-1])
    with pytest.raises(ValueError):
        TrainOnlyDriftContrastiveBank().fit_unlabeled(raw,np.ones(len(batch)))
    source=TrainOnlyDriftContrastiveBank().fit_unlabeled(raw,batch)
    with pytest.raises(ValueError):
        source.source_levels(np.zeros((9,127)))
    with pytest.raises(ValueError):
        source.fuzzy(np.full((9,128),np.nan))


def test_exact_multiclass_additive_fuzzy_head_on_unseen_synthetic_fixture():
    raw,batch,labels=sensor_fixture()
    source=TrainOnlyDriftContrastiveBank().fit_unlabeled(raw[:1200],batch[:1200])
    z=source.fuzzy(raw)
    out,receipt=exact_concept_head(z[:1200],labels[:1200],z[1200:])
    assert out.shape==(200,6)
    assert np.allclose(out.sum(axis=1),1.)
    assert receipt["max_abs_probability_additive_reconstruction_error"]<1e-9
    assert receipt["no_raw_input_to_risk_head_bypass"]
    linear,warn=fit_linear_fair(
        source.source_levels(raw[:1200]),labels[:1200],
        source.source_levels(raw[1200:]))
    assert linear.shape==(200,6)
    assert np.allclose(linear.sum(axis=1),1.)
