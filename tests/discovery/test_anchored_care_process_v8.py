"""Canonical care process: K from TRAIN observation policy, not human labels."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from conceptfan_discovery.anchored_care_process_v8 import (
    MaskRankAnchoredDualConceptFAN,
    fit_train_only_care_task_prior,
    initialize_task_prior,
    process_head_prior_loss,
)
from conceptfan_discovery.care_process_rank_v8 import fit_care_policy_rank


def _source_data(seed:int=44):
    rng=np.random.default_rng(seed)
    n,t,d=240,24,8
    patient=rng.normal(size=(n,1,1))
    time=.2*np.sin(np.arange(t)[None,:,None]*.3)
    sensor=np.arange(d)[None,None,:]*.1
    probs=1/(1+np.exp(-(patient+time-sensor)))
    mask=(rng.random((n,t,d))<probs).astype(np.float32)
    y=(rng.random(n)<(1/(1+np.exp(-patient.reshape(n))))).astype(np.float32)
    return mask,y


def test_rank_and_weights_selected_only_from_train_observation_mask():
    m,y=_source_data()
    rank=fit_care_policy_rank(m,permutations=30,max_slots=5,seed=11)
    model=MaskRankAnchoredDualConceptFAN(
        physiological_observable_basis=np.eye(8)[:,:3],
        care_rank=rank,channels=tuple(f"sensor_{j}" for j in range(8)),
        forecast_horizon=3,
    )
    assert model.process.n_concepts==rank.chosen_k
    assert model.process.feature_logits.requires_grad is False
    assert float(model.care_operator_definition_error())<1e-6
    assert rank.receipt()["clinical_outcomes_used_for_k"] is False
    out=model(torch.randn(3,12,8),torch.ones(3,12,8))
    assert float(out.decomposition_error())<1e-6
    out.logits.sum().backward()
    assert model.process.signed_beta_raw.grad is not None
    assert model.process.threshold_logits.grad is not None
    assert model.process.feature_logits.grad is None


def test_training_task_prior_reproducible_from_same_train_masks_and_y():
    m,y=_source_data()
    rank=fit_care_policy_rank(m,permutations=30,max_slots=5,seed=11)
    model=MaskRankAnchoredDualConceptFAN(
        physiological_observable_basis=np.eye(8)[:,:3],
        care_rank=rank,channels=tuple(f"sensor_{j}" for j in range(8)),
        forecast_horizon=3,
    )
    first=fit_train_only_care_task_prior(model,m,y)
    second=fit_train_only_care_task_prior(model,m,y)
    np.testing.assert_array_equal(first.beta,second.beta)
    assert first.receipt()["physiological_concept_labels_supplied"] is False
    initialize_task_prior(model,first)
    np.testing.assert_allclose(
        model.process.direct_weights().detach().numpy(),first.beta,
        rtol=1e-6,atol=1e-6,
    )
    assert float(process_head_prior_loss(model,first).detach())<1e-9
    with torch.no_grad():
        model.process.signed_beta_raw[0]+=.3
    assert float(process_head_prior_loss(model,first).detach())>1e-4


def test_wrong_source_dictionary_and_nonbinary_labels_fail():
    m,y=_source_data()
    rank=fit_care_policy_rank(m,permutations=30,max_slots=5,seed=11)
    with pytest.raises(ValueError):
        MaskRankAnchoredDualConceptFAN(
            physiological_observable_basis=np.eye(7)[:,:3],
            care_rank=rank,channels=tuple(f"sensor_{j}" for j in range(7)),
            forecast_horizon=3,
        )
    model=MaskRankAnchoredDualConceptFAN(
        physiological_observable_basis=np.eye(8)[:,:3],
        care_rank=rank,channels=tuple(f"sensor_{j}" for j in range(8)),
        forecast_horizon=3,
    )
    with pytest.raises(ValueError):
        fit_train_only_care_task_prior(model,m,np.zeros_like(y))
