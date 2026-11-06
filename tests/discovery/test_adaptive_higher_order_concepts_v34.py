"""V34 actual adaptive Type-1, IT2 and restricted IT3 source mathematics."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from conceptfan_discovery.adaptive_higher_order_concepts_v34 import (
    FuzzySourceDefinition,fit_measured_definition,select_order_from_train,
    source_membership,AdditiveFuzzySourceSlot,
    membership_grade_alpha_plane,ALPHAS,
)


def calibrated(*,aleatoric=.16,device=.21,anchor=True):
    return FuzzySourceDefinition(
        source_id="observed_source_grp4_5_6",
        sensor_indices=(4,5,6),
        threshold_train=.05,
        width_train=.85,
        train_missingness_spread=aleatoric,
        boundary_spread_anchor=device if anchor else 0.,
        certified_calibration_environments=2 if anchor else 0,
        external_anchor_verified=anchor,
    )


def test_auto_order_from_independent_train_and_anchor_not_from_TEST_data():
    assert select_order_from_train(calibrated(aleatoric=.01,device=0))=="type1"
    assert select_order_from_train(calibrated(aleatoric=.12,device=0))=="interval_type2"
    assert select_order_from_train(calibrated(aleatoric=.12,device=.18))=="restricted_interval_type3"
    assert select_order_from_train(calibrated(aleatoric=.12,device=.18,anchor=False))=="interval_type2"
    with pytest.raises(ValueError,match="One environment"):
        FuzzySourceDefinition("g",(4,5),0,1,.1,.2,1,True)
    with pytest.raises(ValueError,match="Do not invent"):
        fit_measured_definition("g",(4,5),np.arange(100),
                                np.ones(100)*.1,certified_environments=2)


def test_type1_membership_is_one_crisp_grade_all_planes_identical():
    x=torch.tensor([-.8,0.,.5,1.])
    u=torch.tensor([.12,.15,.18,.2])
    info=torch.ones(4)
    result=source_membership(x,u,info,calibrated(),order="type1")
    expected=torch.sigmoid((x-.05)/.85)
    torch.testing.assert_close(result.center,expected)
    torch.testing.assert_close(result.degree_lower,expected)
    torch.testing.assert_close(result.degree_upper,expected)
    torch.testing.assert_close(result.alpha_intervals[:,:,0],expected[:,None].expand(-1,4))
    assert result.selected_order=="type1"
    assert not result.interval_is_a_statistical_confidence_interval


def test_interval_type2_exact_core_and_restricted_it3_nesting():
    x=torch.tensor([-.5,0.,.5,1.])
    uncertainty=torch.full((4,),.17)
    info=torch.ones(4)
    definition=calibrated()
    t2=source_membership(x,uncertainty,info,definition,order="interval_type2")
    t3=source_membership(x,uncertainty,info,definition,order="restricted_interval_type3")
    # T3 alpha=1 must equal the original interval-T2 footprint.
    torch.testing.assert_close(t3.alpha_intervals[:,-1,:],t2.alpha_intervals[:,-1,:])
    assert bool((t3.alpha_intervals[:,0,0]<=t3.alpha_intervals[:,-1,0]).all())
    assert bool((t3.alpha_intervals[:,0,1]>=t3.alpha_intervals[:,-1,1]).all())
    assert bool((t3.alpha_intervals[:,:,0][:,:-1]<=
                 t3.alpha_intervals[:,:,0][:,1:]).all())
    assert bool((t3.alpha_intervals[:,:,1][:,:-1]>=
                 t3.alpha_intervals[:,:,1][:,1:]).all())
    assert bool((t3.degree_lower<=t3.center).all())
    assert bool((t3.center<=t3.degree_upper).all())
    assert torch.any(t3.alpha_intervals[:,0,0]<t2.alpha_intervals[:,0,0])
    assert len(ALPHAS)==4
    degree=membership_grade_alpha_plane(t3.alpha_intervals,t3.center)
    assert degree.shape==(4,)
    assert torch.all(degree>0)


def test_type3_is_refused_when_no_independent_device_anchor():
    defn=calibrated(anchor=False)
    with pytest.raises(ValueError,match="independent anchored"):
        source_membership(torch.tensor([1.]),torch.tensor([.2]),
                          torch.tensor([1.]),defn,order="restricted_interval_type3")
    assert source_membership(torch.tensor([1.]),torch.tensor([.2]),
                             torch.tensor([1.]),defn).selected_order=="interval_type2"


def test_source_risk_is_exactly_additive_and_missing_source_never_injected():
    d=calibrated()
    layer=AdditiveFuzzySourceSlot(d,order="restricted_interval_type3")
    with torch.no_grad():
        layer.gamma.fill_(1.25)
    original=torch.tensor([-.44,.13,-.27,1.])
    amp=torch.tensor([.2,1.,-1.,.5],requires_grad=False)
    sd=torch.tensor([.12,.15,.2,.3])
    info=torch.tensor([1.,.62,.0,.48])
    assert not layer(original,amp,sd,info,approved=False)["source_used"].any()
    out=layer(original,amp,sd,info,approved=True)
    torch.testing.assert_close(out["logit"],original+out["source_contribution"])
    torch.testing.assert_close(out["logit"][2:],original[2:])
    assert out["source_used"].tolist()==[True,True,False,False]
    assert bool((out["source_contribution"]>=out["source_contribution_lower"]-1e-7).all())
    assert bool((out["source_contribution"]<=out["source_contribution_upper"]+1e-7).all())
    assert out["source_membership"].source_id==d.source_id
    assert not out["true_physical_source_id_confirmed"]
    loss=out["logit"].sum()
    loss.backward()
    assert layer.gamma.grad is not None
    assert torch.isfinite(layer.gamma.grad)


def test_membership_fit_is_train_only_and_rejects_wrong_shapes():
    rng=np.random.default_rng(123)
    amplitude=rng.normal(size=500)
    spread=np.abs(rng.normal(.15,.03,size=500))
    d=fit_measured_definition("g",(4,5,6),amplitude,spread)
    assert not d.external_anchor_verified
    assert d.fitted_on_train_only
    assert d.threshold_used_outcome_labels is False
    assert d.verified_physical_semantics is False
    assert abs(d.threshold_train-np.median(amplitude))<1e-8
    with pytest.raises(ValueError):
        fit_measured_definition("g",(4,5),amplitude[:100],spread)
    with pytest.raises(ValueError):
        source_membership(torch.ones(3),torch.ones(2),torch.ones(3),d)
