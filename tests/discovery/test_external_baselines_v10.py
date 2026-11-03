"""Clinical V10 versus non-ConceptFAN external baseline correctness checks."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from conceptfan_discovery.data import DiscoverySplit
from conceptfan_discovery.external_baselines_v10 import (
    GRUDStyleClassifier, TimeSeriesTransformerClassifier,
    _validate_view, engineer_clinical_history,
    fit_neural_baseline, fit_tabular_baseline,
    predict_neural, predict_tabular,
    time_since_previous_observation,
)
from conceptfan_discovery.clinical_v10_external_benchmark import (
    EXTERNAL_METHODS, V10_METHOD, _paired_bootstrap, _model_score,
)


def _cohort(n: int = 108, t: int = 36, c: int = 5):
    rng=np.random.default_rng(971)
    x=rng.normal(size=(n,t,c)).astype(np.float32)
    mask=(rng.random(size=x.shape)>.45).astype(np.float32)
    # The values at missing locations may be arbitrary; all model
    # representations must ignore them.
    x=x*mask
    source=mask.copy()
    labels=np.asarray([0,1]*(n//2),dtype=np.float32)
    return x,mask,source,labels


def _split():
    x,m,_,y=_cohort()
    return DiscoverySplit(
        x=x,mask=m,future=np.zeros((len(y),6,x.shape[-1]),dtype=np.float32),
        future_mask=np.ones((len(y),6,x.shape[-1]),dtype=np.float32),
        y=y,oracle_states=None,episode_ids=np.arange(len(y),dtype=np.int64),
    )


def test_no_conceptfan_versions_are_used_as_external_baselines():
    assert EXTERNAL_METHODS==(
        "logistic_history","hist_gradient_boosting",
        "gru_d_style","transformer_timeseries",
    )
    assert V10_METHOD=="conceptfan_v10_multiscale_consistency"
    assert not any("v7" in m or "v8" in m or "v9" in m for m in EXTERNAL_METHODS)


def test_36h_history_features_never_use_missing_values():
    x,m,s,_=_cohort()
    first=engineer_clinical_history(x,m,s)
    assert first.shape==(len(x),21*x.shape[-1])
    corrupted=x.copy()
    corrupted[m==0]=10000.
    after=engineer_clinical_history(corrupted,m,s)
    np.testing.assert_allclose(first,after,rtol=0,atol=0)
    assert np.isfinite(after).all()


def test_observed_sampling_schedule_is_distinct_from_artificial_thinning():
    x,m,s,_=_cohort()
    retained=m.copy()
    retained[:,0::2]=0.
    short=engineer_clinical_history(x*retained,retained,s)
    original=engineer_clinical_history(x,m,s)
    c=x.shape[-1]
    # Last six sensor groups are based on original recorded mask, so all
    # models may see the same documented observation-process information.
    np.testing.assert_allclose(short[:,-6*c:],original[:,-6*c:])
    assert not np.allclose(short[:,:-6*c],original[:,:-6*c])
    with pytest.raises(ValueError):
        _validate_view(x,np.ones_like(m),m)


def test_previous_observation_delta_is_causal():
    mask=torch.tensor([[[1.],[0.],[0.],[1.],[0.]]])
    delta=time_since_previous_observation(mask)
    torch.testing.assert_close(
        delta.squeeze(),torch.tensor([0.,1.,2.,3.,1.])
    )


@pytest.mark.parametrize("klass",["gru_d_style","transformer_timeseries"])
def test_neural_baseline_uses_true_masks_and_is_differentiable(klass):
    x,m,s,_=_cohort(n=10)
    if klass=="gru_d_style":
        model=GRUDStyleClassifier(x.shape[-1],hidden_size=16,
                                  train_mean=np.zeros(x.shape[-1],np.float32))
    else:
        model=TimeSeriesTransformerClassifier(x.shape[-1],history_hours=x.shape[1],
                                              d_model=32,n_heads=4,n_layers=1)
    model.eval()
    input_tensor=torch.from_numpy(x)
    mask=torch.from_numpy(m)
    source=torch.from_numpy(s)
    with torch.no_grad():
        initial=model(input_tensor,mask,source)
        missing=input_tensor.clone()
        missing[mask==0]=100000.
        disguised=model(missing,mask,source)
    torch.testing.assert_close(initial,disguised,atol=1e-5,rtol=1e-5)
    assert initial.shape==(len(x),)
    model.train()
    loss=model(input_tensor,mask,source).square().mean()
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


@pytest.mark.parametrize("method",["logistic_history","hist_gradient_boosting"])
def test_real_classifiers_fit_and_score_same_patient_arrangement(method):
    x,m,s,y=_cohort()
    model=fit_tabular_baseline(method,x,m,s,y,
                               augmentation_fraction=.2,seed=73)
    pred=predict_tabular(model,x,m,s)
    assert pred.shape==(len(y),)
    assert np.isfinite(pred).all()
    assert ((pred>=0)&(pred<=1)).all()


def test_short_neural_training_preserves_prediction_and_preset_classes():
    torch.set_num_threads(1)
    split=_split()
    model,record=fit_neural_baseline(
        "gru_d_style",split,seed=11,epochs=1,batch_size=32,
    )
    p=predict_neural(model,split.x,split.mask,split.mask)
    assert p.shape==split.y.shape
    assert record["validation_used_for_optimization"] is False
    assert record["parameters"]>0


def test_exact_paired_patient_bootstrap_is_zero_for_identical_predictions():
    x,m,s,y=_cohort(n=100)
    p=np.linspace(.01,.99,len(y))
    a={11:p.copy(),23:p.copy(),37:p.copy()}
    b={0:p.copy()}
    result=_paired_bootstrap(y,a,b,repetitions=120,seed=871)
    assert abs(result["delta_auprc_v10_minus_external"])<1e-10
    assert all(abs(v)<1e-10 for v in result[
        "descriptive_patient_and_training_run_bootstrap_ci95"
    ])
    assert result["confirmatory_p_value"] is None


def test_score_rejects_mismatched_or_invalid_predictions():
    y=np.asarray([0.,1.,0.,1.])
    with pytest.raises(ValueError):
        _model_score(y,np.asarray([.1,.3,.5]))
    with pytest.raises(ValueError):
        _model_score(y,np.asarray([.1,1.2,.5,.8]))
    ok=_model_score(y,np.asarray([.1,.3,.5,.8]))
    assert ok["n_positive"]==2
    assert ok["prevalence"]==.5
