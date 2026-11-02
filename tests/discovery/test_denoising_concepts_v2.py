from __future__ import annotations

import numpy as np
import pytest
import torch

from conceptfan_discovery.model_v2 import (
    DenoisingPredictiveConceptFAN, masked_observation_view, v2_loss,
)
from conceptfan_discovery.selection_v2 import select_v2_cardinality
from conceptfan_discovery.selection import split_validation_for_cardinality
from conceptfan_discovery.variable_factor_data import generate_variable_factor_data


def test_masked_true_measurements_never_enter_concept_input():
    rng=torch.Generator().manual_seed(23)
    x=torch.randn(3,7,8)
    observed=torch.ones_like(x)
    corrupt,mask,hidden=masked_observation_view(x,observed,fraction=.3,generator=rng)
    assert hidden.sum()>0
    assert ((hidden*corrupt)==0).all()
    assert ((hidden*mask)==0).all()
    assert ((1-hidden)*corrupt == (1-hidden)*x).all()
    with pytest.raises(ValueError):
        masked_observation_view(x,observed,fraction=1.0)


def test_v2_prediction_future_and_present_reconstruction_only_through_concepts():
    model=DenoisingPredictiveConceptFAN(n_slots=6,n_channels=8,forecast_horizon=6)
    model.eval()
    x=torch.randn(5,9,8)
    mask=torch.ones_like(x)
    out=model(x,mask)
    assert out.reconstruction.shape==x.shape
    assert out.future_prediction.shape==(5,6,8)
    assert float(out.decomposition_error().detach())<1e-5
    loss=out.reconstruction.square().mean()
    loss.backward()
    assert model.feature_logits.grad is not None
    assert torch.isfinite(model.feature_logits.grad).all()
    # No direct route to observation decoder exists after disabling all
    # concept slots; decoder can only return its constant bias.
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        out_disabled=model(x,mask,selected_slots=torch.zeros(6))
        assert out_disabled.concept_trajectories.abs().sum()==0
        torch.testing.assert_close(
            out_disabled.reconstruction,
            model.observation_decoder.bias.view(1,1,-1).expand_as(out_disabled.reconstruction),
        )


def test_denoising_loss_is_only_added_for_full_ablation():
    model=DenoisingPredictiveConceptFAN(n_slots=6,n_channels=8,forecast_horizon=3)
    model.eval()
    x=torch.randn(5,9,8)
    m=torch.ones_like(x)
    corr,vis,hidden=masked_observation_view(x,m,fraction=.3)
    out=model(corr,vis)
    target=torch.randn(5,3,8)
    future_mask=torch.ones_like(target)
    labels=torch.tensor([0.,1.,0.,1.,0.])
    weights={"future":.6,"dynamics":.2,"mdl":.007,"support":.008,
             "redundancy":.08,"nonconstant":.25,"transition_l1":.003,"denoising":2.}
    full,parts=v2_loss(model,out,labels,target,future_mask,x,hidden,
                       mode="full",pos_weight=torch.tensor(1.),loss_weights=weights)
    no_denoise,_=v2_loss(model,out,labels,target,future_mask,x,hidden,
                        mode="full_without_denoising",pos_weight=torch.tensor(1.),loss_weights=weights)
    torch.testing.assert_close(full-no_denoise,2.*parts["masked_observation"])


def test_v2_automated_k_selection_respects_masked_reconstruction():
    data=generate_variable_factor_data(
        n_true_factors=3,seed=21,split_seed=31,n_samples=240,
    )
    selection,audit=split_validation_for_cardinality(data.validation)
    model=DenoisingPredictiveConceptFAN(n_slots=4,n_channels=8,forecast_horizon=6)
    model.eval()
    result=select_v2_cardinality(model,selection,audit,mode="full",
                                 max_auprc_drop=1.,max_nll_increase=10.,
                                 max_future_rmse_increase=10.,
                                 max_masked_rmse_increase=.10)
    assert 1<=result.selected_mask.sum()<=4
    assert "masked_reconstruction_rmse" in result.audit_metrics
    assert result.audit_metrics["no_outcome_from_audit_used_for_selection"] is True
    assert not set(result.selection_episode_ids)&set(result.audit_episode_ids)
