"""Train-only canonical dynamic concept head: no latent annotations or shortcuts."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from conceptfan_discovery.spectral_concepts_v3 import SpectralConceptFAN
from conceptfan_discovery.spectral_task_head import (
    fit_canonical_task_head, initialize_head, task_head_alignment_penalty,
)


def _training_data():
    rng=np.random.default_rng(19)
    x=rng.normal(size=(120,15,8)).astype(np.float32)
    m=(rng.random(size=x.shape)>.15).astype(np.float32)
    x*=m
    y=np.asarray([0,1]*60,dtype=np.float32)
    return x,m,y,np.eye(8)[:,:3]


def _model(basis):
    return SpectralConceptFAN(
        observable_loadings=basis,
        channels=tuple(f"sensor_{j+1}" for j in range(8)),
        forecast_horizon=3,
        freeze_definitions=True,
    )


def test_train_only_head_reproducible_without_manual_concept_targets():
    x,m,y,b=_training_data()
    args=dict(
        observable_basis=b,
        channels=tuple(f"sensor_{j+1}" for j in range(8)),
        forecast_horizon=3,
    )
    h1=fit_canonical_task_head(x,m,y,**args)
    h2=fit_canonical_task_head(x,m,y,**args)
    np.testing.assert_array_equal(h1.signed_coefficient,h2.signed_coefficient)
    assert h1.bias==h2.bias
    assert h1.n_training_patients==120
    assert h1.to_json()["human_concept_labels"] is False
    assert h1.to_json()["latent_oracle_labels"] is False


def test_shared_signed_head_is_exact_and_frozen_if_predeclared():
    x,m,y,b=_training_data()
    head=fit_canonical_task_head(
        x,m,y,observable_basis=b,
        channels=tuple(f"sensor_{j+1}" for j in range(8)),
        forecast_horizon=3,
    )
    model=_model(b)
    initialize_head(model,head,freeze_head=True)
    np.testing.assert_allclose(
        model.decision_weights().detach().numpy(),head.signed_coefficient,
        atol=1e-6,
    )
    assert model.decision_raw_weight.requires_grad is False
    assert model.decision_bias.requires_grad is False
    out=model(torch.from_numpy(x[:5]),torch.from_numpy(m[:5]))
    assert out.decomposition_error()<1e-6
    out.logits.sum().backward()
    assert model.threshold_logits.grad is not None


def test_head_prior_penalty_zero_at_train_prior_and_positive_after_shift():
    x,m,y,b=_training_data()
    head=fit_canonical_task_head(
        x,m,y,observable_basis=b,
        channels=tuple(f"sensor_{j+1}" for j in range(8)),
        forecast_horizon=3,
    )
    model=_model(b)
    initialize_head(model,head,freeze_head=False)
    assert abs(float(task_head_alignment_penalty(model,head).detach()))<1e-10
    with torch.no_grad():
        model.decision_raw_weight[0]+=.4
    assert float(task_head_alignment_penalty(model,head).detach())>0.01


def test_bad_training_labels_rejected_instead_of_invented_prior():
    x,m,y,b=_training_data()
    with pytest.raises(ValueError):
        fit_canonical_task_head(
            x,m,np.zeros_like(y),observable_basis=b,
            channels=tuple(f"sensor_{j+1}" for j in range(8)),
            forecast_horizon=3,
        )
