from __future__ import annotations

import numpy as np
import pytest
import torch

from conceptfan_discovery.multiscale_physiology_v10 import (
    MultiScaleDualProcessConceptFAN,
    MultiScaleUnknownNeutralConceptFAN,
    causal_observation_mean,
    expanded_multiscale_definition,
    physiological_scale_entropy,
)
from conceptfan_discovery.unknown_neutral_v6 import UnknownNeutralSpectralConceptFAN
from conceptfan_discovery.observation_process_concepts_v7 import DualProcessConceptFAN
from conceptfan_discovery.model_v2 import masked_observation_view, v2_loss


def _basis():
    return np.eye(5, 3, dtype=np.float64)


def _names():
    return tuple(f"signal_{i}" for i in range(5))


def _values():
    torch.manual_seed(71)
    x = torch.randn(4, 36, 5)
    mask = (torch.rand(4, 36, 5) > .40).to(dtype=x.dtype)
    return x * mask, mask


def test_multiscale_means_do_not_use_future_or_missing_sentinels():
    x, m = _values()
    changed = x.clone()
    changed[:, 18:] = 999.
    for w in (1, 4, 12, 24):
        old, availability=causal_observation_mean(x, m, w)
        new, _=causal_observation_mean(changed, m, w)
        torch.testing.assert_close(old[:, :18], new[:, :18])
        assert ((availability >= 0) & (availability <= 1)).all()
        sentinel=x.clone()
        sentinel[m == 0]=-1e6
        safe,_=causal_observation_mean(sentinel, m, w)
        torch.testing.assert_close(old,safe)
    with pytest.raises(ValueError):
        causal_observation_mean(x,m,0)


def test_missing_all_measurements_is_unknown_and_not_a_risk_signal():
    physical=MultiScaleUnknownNeutralConceptFAN(
        observable_loadings=_basis(),channels=_names(),forecast_horizon=6
    )
    physical.eval()
    x=torch.randn(3, 36, 5)
    missing=torch.zeros_like(x)
    with torch.no_grad():
        out=physical(x,missing)
    assert float(out.contributions.abs().max())<1e-6
    assert float((out.concept_trajectories-.5).abs().max())<1e-6
    assert torch.isfinite(out.future_prediction).all()


def test_learned_scales_are_nonnegative_normalized_and_differentiable():
    physical=MultiScaleUnknownNeutralConceptFAN(
        observable_loadings=_basis(),channels=_names(),forecast_horizon=6
    )
    physical.train()
    x,m=_values()
    out=physical(x,m)
    pi=physical.scale_weights()
    torch.testing.assert_close(pi.sum(dim=-1),torch.ones(3))
    assert (pi > 0).all()
    assert len(physical.multiscale_passport()["concept_temporal_weights"])==3
    loss=out.logit.square().mean()+out.future_prediction.square().mean()
    loss.backward()
    assert physical.scale_logits.grad is not None
    assert torch.isfinite(physical.scale_logits.grad).all()
    assert float(physical.scale_logits.grad.abs().sum())>0.


def test_two_family_additive_logit_and_original_mask_preserved():
    model=MultiScaleDualProcessConceptFAN(
        observable_loadings=_basis(),channels=_names(),forecast_horizon=6,
        n_observation_concepts=3,
    )
    model.eval()
    x,m=_values()
    corrupted,visible,heldout=masked_observation_view(
        x,m,fraction=.4,generator=torch.Generator().manual_seed(7)
    )
    with torch.no_grad():
        original=model(x,m,process_mask=m)
        changed=model(corrupted,visible,process_mask=m)
    assert float(original.decomposition_error())<1e-6
    assert float(changed.decomposition_error())<1e-6
    torch.testing.assert_close(
        original.observation_contributions,changed.observation_contributions
    )
    assert torch.isfinite(changed.physical.logit).all()
    assert heldout.sum()>0.


def test_v10_vs_v7_operators_and_parameters_are_not_fake_identical():
    original=DualProcessConceptFAN(
        observable_loadings=_basis(),channels=_names(),forecast_horizon=6,
        n_observation_concepts=3,
    )
    multiscale=MultiScaleDualProcessConceptFAN(
        observable_loadings=_basis(),channels=_names(),forecast_horizon=6,
        n_observation_concepts=3,
    )
    baseline_def=expanded_multiscale_definition(original)
    multi_def=expanded_multiscale_definition(multiscale)
    assert baseline_def.shape==(3,4*5*4*2)
    assert multi_def.shape==baseline_def.shape
    assert np.allclose(baseline_def[:, 40:],0)
    assert np.all(multi_def[:, 40:].sum(axis=-1)>0)
    assert isinstance(original.physical,UnknownNeutralSpectralConceptFAN)
    assert not isinstance(original.physical,MultiScaleUnknownNeutralConceptFAN)
    assert torch.isfinite(physiological_scale_entropy(multiscale))


def test_reconstruction_still_only_sees_concept_trajectories():
    model=MultiScaleDualProcessConceptFAN(
        observable_loadings=_basis(),channels=_names(),forecast_horizon=6,
    )
    x,m=_values()
    out=model(x,m,process_mask=m)
    assert out.physical.reconstruction is not None
    assert out.physical.reconstruction.shape==(4,36,5)
    y=torch.tensor([0.,1.,0.,1.])
    future=torch.zeros(4,6,5)
    future_mask=torch.ones_like(future)
    hidden=torch.ones_like(x)
    weights={"future":.5,"dynamics":.2,"mdl":.007,"support":.008,
             "redundancy":.08,"nonconstant":.25,
             "transition_l1":.003,"denoising":1.0}
    loss,terms=v2_loss(
        model.physical,out.physical,y,future,future_mask,x,hidden,
        mode="full",pos_weight=torch.tensor(1.),loss_weights=weights,
    )
    assert torch.isfinite(loss)
    assert float(out.decomposition_error())<1e-6
