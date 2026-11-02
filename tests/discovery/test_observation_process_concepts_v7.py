"""Direct dual-ontology concept formation: source-traceability, exactness and gradients."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conceptfan_discovery.model_v2 import masked_observation_view,v2_loss
from conceptfan_discovery.observation_process_concepts_v7 import (
    OBSERVATION_OPERATORS,
    DualProcessConceptFAN,
    dual_training_objective,
    observation_process_operator_bank,
)


def _model(k=3,obs=2):
    return DualProcessConceptFAN(
        observable_loadings=np.eye(8,dtype=np.float64)[:,:k],
        channels=tuple(f"channel_{j+1}" for j in range(8)),
        forecast_horizon=4,
        n_observation_concepts=obs,
    )


def test_mask_process_bank_never_needs_measured_values_or_target_labels():
    rng=torch.Generator().manual_seed(17)
    m=(torch.rand((5,18,8),generator=rng)>.35).float()
    x=observation_process_operator_bank(m)
    assert x.shape==(5,8*len(OBSERVATION_OPERATORS))
    rate=m.mean(dim=1)
    torch.testing.assert_close(x[:,:8],rate,atol=0,rtol=0)
    assert torch.isfinite(x).all()
    assert (x>=0).all()
    with pytest.raises(ValueError):
        observation_process_operator_bank(torch.rand(5,18,8))


def test_no_physio_observation_makes_physio_contributions_exact_zero_but_not_mask_process():
    model=_model()
    model.eval()
    x=torch.randn(6,18,8)
    empty=torch.zeros_like(x)
    with torch.no_grad():
        out=model(x,empty)
    torch.testing.assert_close(
        out.physical.contributions,torch.zeros_like(out.physical.contributions),
        atol=1e-6,rtol=0,
    )
    assert float(out.decomposition_error())<1e-6
    torch.testing.assert_close(
        out.logits,
        model.physical.decision_bias+out.observation_contributions.sum(dim=-1),
        atol=1e-6,rtol=0,
    )


def test_observation_head_output_does_not_depend_on_physiological_values_given_mask():
    model=_model()
    model.eval()
    mask=(torch.rand(4,12,8)>.30).float()
    x=torch.randn(4,12,8)
    with torch.no_grad():
        a=model(x*mask,mask)
        b=model((x+20)*mask,mask)
    torch.testing.assert_close(
        a.observation_contributions,b.observation_contributions,
        atol=0,rtol=0,
    )
    assert float(a.decomposition_error())<1e-6


def test_new_observation_concept_is_not_hidden_shortcut_and_gradient_reaches_both_families():
    model=_model()
    model.train()
    x=torch.randn(9,12,8)
    mask=(torch.rand_like(x)>.25).float()
    corrupted,visible,withheld=masked_observation_view(
        x,mask,fraction=.20
    )
    out=model(corrupted,visible)
    y=torch.tensor([0.,1.,0.,1.,0.,1.,0.,0.,1.])
    future=torch.randn(9,4,8)
    obsfuture=torch.ones_like(future)
    weights={
        "future":.6,"dynamics":.2,"mdl":.007,
        "support":.008,"redundancy":.08,"nonconstant":.25,
        "transition_l1":.003,"denoising":1.25,
    }
    base,parts=v2_loss(
        model.physical,out.physical,y,future,obsfuture,
        x,withheld,mode="full",
        pos_weight=torch.tensor(1.),loss_weights=weights,
    )
    loss,receipt=dual_training_objective(
        model,out,y,base,parts["task"],pos_weight=torch.tensor(1.),
    )
    assert receipt["physical_task_replaced_not_doubled"] is True
    assert torch.isfinite(loss)
    assert float(out.decomposition_error())<1e-6
    loss.backward()
    assert model.physical.threshold_logits.grad is not None
    assert model.process.threshold_logits.grad is not None
    assert model.process.feature_logits.grad is not None
    assert model.process.signed_beta_raw.grad is not None
    assert model.physical.decision_raw_weight.grad is not None


def test_independent_exact_model_interventions_do_not_modify_other_family():
    model=_model()
    model.eval()
    x=torch.randn(6,11,8)
    m=(torch.rand_like(x)>.20).float()
    out=model(x*m,m)
    original_observation=out.observation_contributions[:,0]
    new_obs=torch.ones(6)
    changed=model.intervene_on_observation_concept(out,0,new_obs)
    expected=(
        out.logits-original_observation+
        .5*model.process.direct_weights()[0]
    )
    torch.testing.assert_close(changed,expected)
    override=torch.ones(6)*.35
    physical_changed=model.intervene_on_physical_concept(out,0,override)
    phys_logits,_=model.physical.intervene(out.physical,0,override)
    torch.testing.assert_close(
        physical_changed,
        phys_logits+out.observation_contributions.sum(dim=-1)
    )
    with pytest.raises(ValueError):
        model.intervene_on_observation_concept(out,500,new_obs)


def test_every_care_process_concept_has_signed_sourced_operator_passport():
    model=_model()
    passport=model.process.passports(model.physical.channels)
    assert len(passport)==2
    assert all(x["evidence_family"]=="care_observation_policy_not_patient_physiology" for x in passport)
    assert all(x["directly_used_in_training_task_logit"] for x in passport)
    assert all(x["clinical_diagnosis_identity_verified"] is False for x in passport)
    assert all(isinstance(x["signed_direct_logit_coefficient"],float) for x in passport)
