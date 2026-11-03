"""Exact conceptual and gradient contracts for automatically spectral-anchored concepts."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from conceptfan_discovery.spectral_concepts_v3 import (
    SpectralConceptFAN, spectral_operator_definitions,
)
from conceptfan_discovery.model_v2 import masked_observation_view, v2_loss


def _basis(k=3):
    return np.eye(8,dtype=np.float64)[:,:k]


def _model(frozen=True):
    return SpectralConceptFAN(
        observable_loadings=_basis(),
        channels=tuple(f"sensor_{i+1}" for i in range(8)),
        forecast_horizon=3,
        freeze_definitions=frozen,
    )


def test_automatically_discovered_basis_has_normalized_inspectable_signed_operators():
    weights=spectral_operator_definitions(_basis(),n_channels=8)
    assert weights.shape==(3,64)
    np.testing.assert_allclose(weights.sum(axis=1).numpy(),np.ones(3))
    for i in range(3):
        assert weights[i,i]==1.
        assert torch.count_nonzero(weights[i])==1


@pytest.mark.parametrize("frozen",[True,False])
def test_native_model_uses_all_detected_slots_and_exact_additive_bottleneck(frozen):
    model=_model(frozen)
    model.eval()
    x=torch.randn(7,10,8)
    mask=(torch.rand_like(x)>.17).float()
    out=model(x*mask,mask)
    assert out.concept_trajectories.shape==(7,10,3)
    assert out.future_prediction.shape==(7,3,8)
    assert out.reconstruction.shape==(7,10,8)
    assert out.contributions.shape==(7,3)
    assert float(out.decomposition_error())<1e-6
    torch.testing.assert_close(out.gates,torch.ones(3))
    assert not model.feature_logits.requires_grad if frozen else model.feature_logits.requires_grad
    loss=out.logits.square().mean()+out.reconstruction.square().mean()
    loss.backward()
    assert model.threshold_logits.grad is not None
    assert model.decision_raw_weight.grad is not None
    assert model.feature_logits.grad is None if frozen else model.feature_logits.grad is not None
    assert model.concept_basis_manifest()["concept_labels_from_humans"] is False


def test_finetuning_penalizes_unexplained_rotation_from_measurable_operator_basis():
    model=_model(False)
    with torch.no_grad():
        starting=model.spectral_definition_penalty()
        assert float(starting["operator_displacement"])<1e-10
        model.feature_logits[0].roll_(1) if False else None
        model.feature_logits[0,0]=-1.
        model.feature_logits[0,1]=1.
        moved=model.spectral_definition_penalty()
        assert float(moved["operator_displacement"])>float(starting["operator_displacement"])+0.1


def test_spectral_dimension_is_evidence_not_manual_placeholder():
    channels=tuple(f"sensor_{i+1}" for i in range(8))
    with pytest.raises(ValueError,match="Detected K=1"):
        SpectralConceptFAN(
            observable_loadings=np.eye(8)[:,:1],
            channels=channels,forecast_horizon=3,freeze_definitions=True,
        )
    with pytest.raises(ValueError,match="orthonormal"):
        spectral_operator_definitions(np.full((8,3),1.),n_channels=8)


def test_no_decoder_bypass_with_forced_off_native_concepts():
    model=_model(True)
    model.eval()
    x=torch.randn(5,12,8)
    m=torch.ones_like(x)
    with torch.no_grad():
        disabled=model(x,m,selected_slots=torch.zeros(3))
    torch.testing.assert_close(disabled.gates,torch.zeros(3))
    torch.testing.assert_close(disabled.logits,model.decision_bias.expand_as(disabled.logits))
    torch.testing.assert_close(
        disabled.reconstruction,
        model.observation_decoder.bias[None,None].expand_as(disabled.reconstruction),
    )


def test_spectral_model_trains_on_masked_observables_not_latent_factor_labels():
    model=_model(True)
    model.train()
    x=torch.randn(5,12,8)
    m=torch.ones_like(x)
    corrupted,visible,withheld=masked_observation_view(x,m,fraction=.20)
    out=model(corrupted,visible)
    y=torch.tensor([0.,1.,0.,1.,0.])
    future=torch.randn(5,3,8)
    weights={
        "future":.6,"dynamics":.2,"mdl":.007,"support":.008,
        "redundancy":.08,"nonconstant":.25,"transition_l1":.003,
        "denoising":1.25,
    }
    loss,_=v2_loss(model,out,y,future,torch.ones_like(future),x,withheld,
                    mode="full",pos_weight=torch.ones(()),loss_weights=weights)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.threshold_logits.grad is not None
    assert model.feature_logits.grad is None
