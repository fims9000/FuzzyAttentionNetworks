from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest
import torch
import yaml

from conceptfan_discovery.data import DiscoveryData
from conceptfan_discovery.identifiability_v11 import (
    PredictiveGroup,grouped_logit_contributions,
    torch_rotation_invariant_group_loss,
)
from conceptfan_discovery.model import DiscoveryOutput
from conceptfan_discovery.multiscale_physiology_v10 import MultiScaleDualProcessConceptFAN
from conceptfan_discovery.rotation_consistency_v11 import grouped_rotation_invariant_view_consistency
from conceptfan_discovery.train_identifiability_v11 import fit_identifiability_aware_v11
from conceptfan_discovery.variable_factor_data import generate_variable_factor_data


def _groups():
    return (
        PredictiveGroup(0,2,"GROUP_SUBSPACE_STABLE_AXES_UNRESOLVED",
                        1.,1.,None,(0,1)),
        PredictiveGroup(2,3,"INDIVIDUAL_OBSERVABLE_AXIS_STABLE",
                        1.,1.,1.,(2,)),
    )


def _output(concepts:torch.Tensor,reliability:torch.Tensor)->DiscoveryOutput:
    n,t,k=concepts.shape
    return DiscoveryOutput(
        logits=concepts.new_zeros(n),
        probability=concepts.new_full((n,),.5),
        concept_trajectories=concepts,
        raw_activation=concepts,
        availability=reliability,
        concept_summary=concepts.mean(dim=1),
        contributions=concepts.mean(dim=1),
        future_prediction=concepts.new_zeros((n,1,2)),
        gates=concepts.new_ones(k),
        feature_weights=concepts.new_zeros((k,16)),
        decision_bias=concepts.new_zeros(()),
    )


def test_group_view_consistency_does_not_depend_on_basis_rotation():
    generator=torch.Generator().manual_seed(81)
    c=torch.randn((5,18,3),generator=generator,requires_grad=True)
    availability=torch.ones_like(c)
    angle=.75
    cos,sin=np.cos(angle),np.sin(angle)
    rotation=torch.tensor([[cos,-sin],[sin,cos]],dtype=c.dtype)
    rotated=torch.cat([c[:,:,:2]@rotation,c[:,:,2:]],dim=-1)
    a,b=_output(c,availability),_output(rotated,availability)
    penalty,reliability=grouped_rotation_invariant_view_consistency(a,b,_groups())
    assert penalty.item()<1e-7
    assert reliability.item()==pytest.approx(1.)
    penalty.backward()
    assert c.grad is not None and torch.isfinite(c.grad).all()


def test_group_temporal_loss_catches_changed_temporal_patterns():
    generator=torch.Generator().manual_seed(31)
    c=torch.randn((4,18,3),generator=generator)
    wrong=c[:,torch.arange(17,-1,-1),:].clone()
    a,b=_output(c,torch.ones_like(c)),_output(wrong,torch.ones_like(wrong))
    loss,_=grouped_rotation_invariant_view_consistency(a,b,_groups())
    assert loss.item()>0.0001


def test_all_missing_group_consistency_is_zero_not_fake_stability():
    c=torch.randn(5,16,3)
    a,b=_output(c,torch.zeros_like(c)),_output(c+5,torch.zeros_like(c))
    p,coverage=grouped_rotation_invariant_view_consistency(a,b,_groups())
    assert p.item()==0.
    assert coverage.item()==0.


def test_learned_native_model_honors_group_sum_additive_identity():
    torch.manual_seed(10)
    model=MultiScaleDualProcessConceptFAN(
        observable_loadings=np.eye(8,3,dtype=np.float64),
        channels=tuple(f"sensor_{i}" for i in range(8)),forecast_horizon=6,
    )
    model.eval()
    values=torch.randn(6,15,8)
    mask=(torch.rand(6,15,8)>.3).float()
    with torch.no_grad():
        out=model(values*mask,mask,process_mask=mask)
    totals=grouped_logit_contributions(
        out.physical.contributions.numpy(),_groups()
    )
    np.testing.assert_allclose(totals.sum(axis=1),
                               out.physical.contributions.sum(-1).numpy(),atol=1e-6)
    assert out.decomposition_error().item()<1e-5


def test_rotation_projector_loss_changes_concept_parameters():
    torch.manual_seed(91)
    model=MultiScaleDualProcessConceptFAN(
        observable_loadings=np.eye(8,3,dtype=np.float64),
        channels=tuple(f"sensor_{i}" for i in range(8)),forecast_horizon=6,
    )
    w=model.physical.feature_weights()
    penalty=torch_rotation_invariant_group_loss(
        w,model.physical.source_predictive_basis,_groups()
    )
    assert torch.isfinite(penalty)
    penalty.backward()
    assert model.physical.feature_logits.grad is not None
    assert torch.isfinite(model.physical.feature_logits.grad).all()


@pytest.mark.parametrize("variant",["v11_projector_axis_view","v11_projector_group_view"])
def test_full_one_epoch_synthetic_fit_no_oracle_supervision(variant):
    ds=generate_variable_factor_data(n_true_factors=3,seed=774,split_seed=121,
                                     n_samples=240)
    from types import SimpleNamespace
    cert=SimpleNamespace(groups=_groups(),selected_computational_rank=3)
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    with (root/"configs/medical/pd_clinical_unknown_neutral_v6.yaml").open() as f:
        base=yaml.safe_load(f)
    with (root/"configs/medical/pd_clinical_dual_process_v7.yaml").open() as f:
        v7=yaml.safe_load(f)
    with (root/"configs/medical/pd_clinical_v10_multiscale.yaml").open() as f:
        v10=yaml.safe_load(f)
    with (root/"configs/medical/pd_clinical_v11_identifiable_groups.yaml").open() as f:
        v11=yaml.safe_load(f)
    base["training"]["epochs"]=1
    base["training"]["batch_size"]=80
    model,meta=fit_identifiability_aware_v11(
        ds,np.eye(8,3,dtype=np.float64),cert,11,base,v7,v10,v11,
        variant=variant,
    )
    x=torch.from_numpy(ds.validation.x[:3])
    mask=torch.from_numpy(ds.validation.mask[:3])
    with torch.no_grad():
        out=model(x,mask,process_mask=mask)
    assert out.decomposition_error().item()<1e-5
    assert meta["arbitrary_svd_axis_anchor_disabled"]
    assert meta["source_basis_is_train_only"]
    assert meta["n_training_minibatches"]>=3
