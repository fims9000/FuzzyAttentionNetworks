"""V12 falsification: no fabricated group identity, no label leakage, gradients.

All fixtures are synthetic; this test module NEVER opens Set B or any
historical calibration/test outcomes.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from conceptfan_discovery.identifiability_v11 import PredictiveCertificate, PredictiveGroup
from conceptfan_discovery.model import DiscoveryOutput
from conceptfan_discovery.multiscale_physiology_v10 import MultiScaleDualProcessConceptFAN
from conceptfan_discovery.signed_effect_v12 import (
    SignedAnchorCertificate, orient_only_resolved_singletons,
    numpy_anchor_targets, torch_anchor_targets,
    fit_train_only_signed_anchors, signed_effect_training_regularizers,
)


def _certificate(basis=None):
    if basis is None:
        basis=np.eye(4)
    return PredictiveCertificate(
        observed_singular_values=np.array([4.,2.,2.,.5]),
        bootstrap_singular_values=np.tile([4.,2.,2.,.5],(32,1)),
        reference_basis=np.asarray(basis,dtype=np.float64),
        groups=(
            PredictiveGroup(0,1,"INDIVIDUAL_OBSERVABLE_AXIS_STABLE",1.,1.,1.,(0,)),
            PredictiveGroup(1,3,"GROUP_SUBSPACE_STABLE_AXES_UNRESOLVED",1.,1.,None,(1,2)),
        ),
        certified_boundaries=(1,),boundary_evidence=tuple(),
        selected_computational_rank=3,
        diagnostics={"from_train_only":True},
    )


def _train(*,labels="signal",seed=34):
    rng=np.random.default_rng(seed)
    n=180
    latent=rng.normal(size=n)
    x=rng.normal(scale=.02,size=(n,36,4)).astype(np.float32)
    x[:,:,0]+=latent[:,None]
    x[:,:,1]+=rng.normal(size=(n,1))
    mask=np.ones_like(x)
    if labels=="signal":
        y=(latent>0).astype(np.float32)
    elif labels=="random":
        y=rng.binomial(1,.5,size=n).astype(np.float32)
    else:
        raise ValueError("unknown test labels")
    return x,mask,y


def test_only_train_certified_singletons_get_orientation():
    base=np.eye(4)
    flipped=base.copy()
    flipped[:,0]*=-1
    a,s=orient_only_resolved_singletons(_certificate(base))
    b,t=orient_only_resolved_singletons(_certificate(flipped))
    np.testing.assert_allclose(a,b)
    assert s.tolist()==[True,False,False]
    assert t.tolist()==[True,False,False]
    # An unresolved group can rotate or flip without receiving axis labels.
    rotated=base.copy()
    phi=.83
    rotated[:,1:3]=base[:,1:3]@np.array([
        [np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]
    ])
    g,m=orient_only_resolved_singletons(_certificate(rotated))
    np.testing.assert_allclose(g[:,1:3],rotated[:,1:3])
    assert not m[1:].any()


def test_train_only_label_sign_selected_when_measurable_and_not_for_group():
    x,mask,y=_train()
    bank=fit_train_only_signed_anchors(
        x,mask,y,_certificate(),bootstraps=64,permutations=199,seed=55,
    )
    assert bank.risk_sign.tolist()==[1,0,0]
    assert bank.receipt["number_of_labeled_future_holdouts_accessed"]==0
    assert bank.receipt["n_train_risk_signs"]==1
    assert bank.selection[0]["train_label_permutation_two_sided_p"]<=.05
    assert bank.selection[1]["reason"]=="ABSTAIN_STRUCTURAL"
    assert bank.to_json()["concepts_are_not_validated_disease_mechanisms"]


def test_random_labels_and_no_signal_abstain():
    x,mask,y=_train(labels="random")
    bank=fit_train_only_signed_anchors(
        x,mask,y,_certificate(),bootstraps=64,permutations=199,seed=61,
        familywise_alpha=.01,
    )
    assert (bank.risk_sign==0).all()


def test_masking_and_empty_history_neutral_for_numpy_and_torch():
    x,m,_=_train()
    b=np.eye(4)[:,:3]
    x[:10,-6:,0]=10000
    m[:10,-6:,0]=0
    target,cover=numpy_anchor_targets(x,m,b)
    t,c=torch_anchor_targets(torch.from_numpy(x),torch.from_numpy(m),
                             torch.from_numpy(b.astype(np.float32)))
    np.testing.assert_allclose(target,t.numpy(),rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(cover,c.numpy(),rtol=1e-6,atol=1e-6)
    assert (target[:10,0]==0).all()
    assert (cover[:10,0]==0).all()
    empty=np.zeros_like(m)
    np.testing.assert_array_equal(numpy_anchor_targets(x,empty,b)[0],0.)
    with pytest.raises(ValueError):
        numpy_anchor_targets(x,m,np.ones((3,3)))


def _out(t:torch.Tensor,beta:torch.Tensor):
    _,_,k=t.shape
    summary=t[:,-6:].mean(dim=1)
    contrib=(summary-.5)*beta
    return DiscoveryOutput(
        logits=contrib.sum(dim=-1),probability=torch.sigmoid(contrib.sum(dim=-1)),
        concept_trajectories=t,raw_activation=t,availability=torch.ones_like(t),
        concept_summary=summary,contributions=contrib,
        future_prediction=t.new_zeros((t.shape[0],3,4)),gates=t.new_ones(k),
        feature_weights=t.new_zeros((k,32)),decision_bias=t.new_zeros(()),
    )


def test_native_concept_and_logit_gradients_are_finite_with_no_hidden_classifier():
    torch.manual_seed(2)
    rng=np.random.default_rng(91)
    d,k=4,3
    x=torch.from_numpy(rng.normal(size=(8,36,d)).astype(np.float32))
    mask=torch.ones_like(x)
    model=MultiScaleDualProcessConceptFAN(
        observable_loadings=np.eye(d)[:,:k],
        channels=("HR","MAP","GCS","Temp"),forecast_horizon=3,
    )
    bank=SignedAnchorCertificate(
        basis=np.eye(d)[:,:k],singleton=np.array([True,False,False]),
        risk_sign=np.array([1,0,0],dtype=np.int8),
        selection=tuple(),receipt={"only_train":True},
    )
    latent=torch.randn((8,36,k),requires_grad=True)
    b_latent=torch.randn((8,36,k),requires_grad=True)
    beta=model.physical.decision_weights()
    a=_out(latent.sigmoid(),beta)
    b=_out(b_latent.sigmoid(),beta)
    objective=signed_effect_training_regularizers(model,a,b,x,mask,bank)
    loss=sum(objective.values())
    loss.backward()
    assert torch.isfinite(loss)
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert b_latent.grad is not None and torch.isfinite(b_latent.grad).all()
    assert model.physical.decision_raw_weight.grad is not None
    assert torch.isfinite(model.physical.decision_raw_weight.grad).all()
    assert not any("teacher" in name for name,_ in model.named_parameters())


def test_unresolved_group_never_gets_artificial_signed_effect():
    x=torch.randn(5,36,4)
    mask=torch.ones_like(x)
    model=MultiScaleDualProcessConceptFAN(
        observable_loadings=np.eye(4)[:,:3],
        channels=("HR","MAP","GCS","Temp"),forecast_horizon=3,
    )
    bank=SignedAnchorCertificate(
        basis=np.eye(4)[:,:3],singleton=np.array([False,False,False]),
        risk_sign=np.zeros(3,dtype=np.int8),selection=tuple(),receipt={},
    )
    t=torch.randn(5,36,3).sigmoid()
    a=_out(t,model.physical.decision_weights())
    other=_out(1-t,model.physical.decision_weights())
    loss=signed_effect_training_regularizers(model,a,other,x,mask,bank)
    assert all(v.item()==pytest.approx(0.) for v in loss.values())
