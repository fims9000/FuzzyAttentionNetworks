from __future__ import annotations
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fan.concept.observability import ObservationAwareConceptFAN, observation_weighted_pool


class FakeEncoder(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.input_proj = nn.Linear(2*n_channels, 8)

    def forward(self, x):
        return torch.tanh(self.input_proj(x))


class FakeTemporal(nn.Module):
    def __init__(self, n_concepts, seq=8):
        super().__init__()
        self.n_concepts = n_concepts
        self.value_weight = nn.Parameter(torch.ones(n_concepts))
        self.time_weight = nn.Parameter(torch.zeros(n_concepts))
        self.bias = nn.Parameter(torch.zeros(n_concepts))
        self.register_buffer("time", torch.linspace(0, 1, seq).view(1, seq, 1))

    def forward(self, trajectories, mode):
        if mode == "static":
            w = torch.zeros_like(trajectories)
            w[:, -1, :] = 1
            return trajectories[:, -1, :], w
        s = trajectories * self.value_weight.view(1,1,-1)
        s = s + self.time[:, :trajectories.shape[1], :] * self.time_weight.view(1,1,-1)
        s = s + self.bias.view(1,1,-1)
        w = torch.softmax(s,dim=1)
        return (w*trajectories).sum(dim=1), w


class FakeDecision(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(k))
        self.bias = nn.Parameter(torch.tensor(-0.3))


class FakeCore(nn.Module):
    oracle = False
    def __init__(self, c=3,k=3, mode="attention"):
        super().__init__()
        self.temporal_mode=mode
        self.encoder=FakeEncoder(c)
        self.projector=nn.Sequential(nn.Linear(8,k), nn.Sigmoid())
        self.temporal_aggregator=FakeTemporal(k)
        self.decision_head=FakeDecision(k)

    def forward_from_summaries(self, h, traj, summaries, temporal_weights, concept_mask=None):
        if concept_mask is None:
            concept_mask = torch.ones_like(summaries,dtype=torch.bool)
        signed = summaries * self.decision_head.weight * concept_mask
        logit = self.decision_head.bias + signed.sum(dim=-1)
        return SimpleNamespace(logit=logit, probability=logit.sigmoid(),
                               signed_decision_contributions=signed,
                               temporal_concept_weights=temporal_weights,
                               concept_summaries=summaries)

    def ordinary_forward(self, features):
        h=self.encoder(features)
        traj=self.projector(h)
        summaries, w=self.temporal_aggregator(traj,self.temporal_mode)
        return self.forward_from_summaries(h,traj,summaries,w)


def test_all_observed_exactly_matches_core_ordinary_path():
    torch.manual_seed(7)
    core=FakeCore()
    wrapper=ObservationAwareConceptFAN(core, torch.eye(3),min_observation_fraction=.2)
    x=torch.randn(4,8,3)
    mask=torch.ones_like(x)
    a=wrapper(x,mask)
    b=core.ordinary_forward(torch.cat([x,mask],dim=-1))
    torch.testing.assert_close(a.logit,b.logit,atol=1e-7,rtol=1e-7)
    torch.testing.assert_close(a.native.concept_summaries,b.concept_summaries)
    assert not bool(a.abstain_from_explanation.any())


def test_missing_values_never_flow_through_imputed_payload():
    torch.manual_seed(7)
    model=ObservationAwareConceptFAN(FakeCore(c=3,k=3),torch.eye(3),min_observation_fraction=.1)
    values=torch.randn(4,8,3)
    mask=torch.ones_like(values)
    mask[:,:,1]=0
    a=model(values,mask)
    values[:,:,1]=123456.0
    b=model(values,mask)
    torch.testing.assert_close(a.logit,b.logit)
    assert bool(a.abstain_from_explanation.all())
    assert bool((a.signed_decision_contributions[:,1]==0).all())
    assert bool((a.signed_decision_contributions[:,0]!=0).any())


def test_every_concept_missing_produces_bias_only_with_abstention():
    model=ObservationAwareConceptFAN(FakeCore(c=3,k=3),torch.eye(3))
    values=torch.full((3,8,3),float("nan"))
    mask=torch.zeros_like(values)
    out=model(values,mask)
    assert bool(out.abstain_from_explanation.all())
    assert bool((out.signed_decision_contributions==0).all())
    torch.testing.assert_close(out.logit,torch.full((3,),model.core.decision_head.bias.detach().item()))
    assert bool(torch.isfinite(out.probability).all())


def test_static_last_observed_is_used_not_last_imputed():
    pool=FakeTemporal(1,5)
    concepts=torch.tensor([[[.1],[.2],[.9],[.4],[.8]]])
    reliability=torch.tensor([[[1.],[1.],[1.],[0.],[0.]]])
    summary,weights=observation_weighted_pool(concepts,reliability,pool,"static")
    assert summary.item()==pytest.approx(.9)
    assert weights[0,2,0].item()==1.0
    assert weights[0,4,0].item()==0.0


def test_partial_reliability_matches_weighted_softmax_and_is_differentiable():
    pool=FakeTemporal(1,3)
    x=torch.tensor([[[.1],[.3],[.8]]],requires_grad=True)
    r=torch.tensor([[[1.],[.5],[0.]]])
    summary,w=observation_weighted_pool(x,r,pool,"attention")
    expected=torch.softmax(torch.tensor([.1,.3])+torch.tensor([1.,.5]).log(),dim=0)
    torch.testing.assert_close(w[0,:2,0],expected)
    assert w[0,2,0].item()==0.0
    summary.sum().backward()
    assert x.grad[0,2,0].item()==0.0
    assert x.grad[0,0,0].abs().item()>0.0


@pytest.mark.parametrize("bad",[-1., 1.2, float('nan')])
def test_invalid_observation_masks_refused(bad):
    model=ObservationAwareConceptFAN(FakeCore(c=3,k=3),torch.eye(3))
    values=torch.zeros(1,8,3)
    mask=torch.ones_like(values)
    mask[0,0,0]=bad
    with pytest.raises(ValueError, match="mask"):
        model(values,mask)


def test_gradient_reaches_native_concept_path():
    torch.manual_seed(8)
    model=ObservationAwareConceptFAN(FakeCore(c=3,k=3),torch.eye(3))
    x=torch.randn(6,8,3)
    mask=torch.ones_like(x)
    out=model(x,mask)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(out.logit,torch.randint(0,2,(6,)).float())
    loss.backward()
    assert model.core.projector[0].weight.grad is not None
    assert model.core.projector[0].weight.grad.abs().sum().item() > 0


def test_invalid_sensor_mapping_rejected():
    with pytest.raises(ValueError, match="at least one"):
        ObservationAwareConceptFAN(FakeCore(),torch.tensor([[1.,0.,0.],[0.,0.,0.],[0.,0.,1.]]))
