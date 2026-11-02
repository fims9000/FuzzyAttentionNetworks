"""Refuse unobserved concept effects and verify genuine train-time coupling."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from conceptfan_discovery.unknown_neutral_v6 import (
    UnknownNeutralSpectralConceptFAN,cross_view_concept_consistency,
)
from conceptfan_discovery.spectral_concepts_v3 import SpectralConceptFAN
from conceptfan_discovery.model_v2 import masked_observation_view


def _model(k:int=3):
    basis=np.eye(8,dtype=np.float64)[:,:k]
    return UnknownNeutralSpectralConceptFAN(
        observable_loadings=basis,
        channels=tuple(f"observed_{i+1}" for i in range(8)),
        forecast_horizon=4,
        freeze_definitions=False,
    )


def test_no_physiological_measurement_means_neutral_unknown_concept_not_false_risk():
    model=_model()
    model.eval()
    dummy=torch.randn(7,12,8)
    observed=torch.zeros_like(dummy)
    with torch.no_grad():
        out=model(dummy,observed)
    torch.testing.assert_close(
        out.concept_trajectories,
        (out.gates*.5).view(1,1,-1).expand_as(out.concept_trajectories),
        atol=1e-6,rtol=0,
    )
    torch.testing.assert_close(out.availability,torch.zeros_like(out.availability))
    torch.testing.assert_close(out.contributions,torch.zeros_like(out.contributions),atol=1e-6,rtol=0)
    torch.testing.assert_close(out.logits,model.decision_bias.expand_as(out.logits),atol=1e-6,rtol=0)
    assert float(out.decomposition_error())<1e-6


def test_original_multiplicative_model_exhibits_missingness_as_task_effect():
    original=SpectralConceptFAN(
        observable_loadings=np.eye(8)[:,:3],
        channels=tuple(f"observed_{i+1}" for i in range(8)),
        forecast_horizon=4,freeze_definitions=True,
    )
    original.eval()
    with torch.no_grad():
        original.decision_raw_weight[:] = torch.tensor([.6,-.8,.4])
        out=original(torch.randn(3,8,8),torch.zeros(3,8,8))
    assert not torch.allclose(out.contributions,torch.zeros_like(out.contributions))
    assert float(out.decomposition_error())<1e-6


def test_jointly_reliable_views_consistency_zero_for_identical_observations():
    model=_model()
    model.eval()
    x=torch.randn(6,11,8)
    mask=torch.ones_like(x)
    a=model(x,mask)
    b=model(x.clone(),mask.clone())
    loss,coverage=cross_view_concept_consistency(a,b)
    assert float(loss.detach())==0.
    assert float(coverage)>0.
    assert float(coverage)<=1.


def test_masked_view_consistency_is_finite_and_backpropagates_to_thresholds():
    model=_model()
    model.train()
    x=torch.randn(6,11,8)
    mask=torch.ones_like(x)
    first,ma,_=masked_observation_view(
        x,mask,fraction=.25,generator=torch.Generator().manual_seed(17)
    )
    second,mb,_=masked_observation_view(
        x,mask,fraction=.40,generator=torch.Generator().manual_seed(19)
    )
    original=model(first,ma)
    changed=model(second,mb)
    loss,reliability=cross_view_concept_consistency(original,changed)
    assert torch.isfinite(loss)
    assert loss>=0
    assert reliability>0
    loss.backward()
    assert model.threshold_logits.grad is not None
    assert torch.isfinite(model.threshold_logits.grad).all()


def test_completely_missing_views_get_zero_consistency_weight_not_false_matching():
    model=_model()
    model.eval()
    x=torch.randn(4,8,8)
    mask=torch.zeros_like(x)
    a=model(x,mask)
    b=model(-x,mask)
    penalty,weight=cross_view_concept_consistency(a,b)
    assert penalty==0
    assert weight==0


def test_bad_coordinate_alignment_and_unsupported_confidence_fail():
    model=_model()
    a=model(torch.randn(4,9,8),torch.ones(4,9,8))
    b=model(torch.randn(5,9,8),torch.ones(5,9,8))
    with pytest.raises(ValueError):
        cross_view_concept_consistency(a,b)
    with pytest.raises(ValueError):
        cross_view_concept_consistency(a,a,minimum_joint_reliability=1.0)
