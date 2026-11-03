"""Rotation-invariant temporal agreement for *unresolved groups* (V11).

The earlier V10 consistency compares slot k to k in a second thinned view,
a potentially arbitrary coordinate inside repeated singular modes.
This alternative aligns past-response Gram matrices and is invariant to
orthogonal rotations *within* empirically unresolved groups.

Only train-time concept activations are involved, never a post-hoc student.
"""
from __future__ import annotations

import torch

from .identifiability_v11 import PredictiveGroup
from .model import DiscoveryOutput


def grouped_rotation_invariant_view_consistency(
    first:DiscoveryOutput,second:DiscoveryOutput,
    groups:tuple[PredictiveGroup,...], *,
    minimum_joint_reliability:float=.2,
    epsilon:float=1e-7,
)->tuple[torch.Tensor,torch.Tensor]:
    x,y=first.concept_trajectories,second.concept_trajectories
    rx,ry=first.availability,second.availability
    if x.ndim!=3 or x.shape!=y.shape or rx.shape!=x.shape or ry.shape!=x.shape:
        raise ValueError("Paired [B,T,K] concepts and reliability required")
    if not groups or groups[0].first!=0 or groups[-1].stop!=x.shape[2]:
        raise ValueError("Empirical predictive groups must span all K slots")
    if not 0<=minimum_joint_reliability<1 or epsilon<=0:
        raise ValueError("Invalid reliability threshold or normalization epsilon")
    total=x.sum()*0.
    valid=x.new_zeros(())
    for g in groups:
        if not 0<=g.first<g.stop<=x.shape[2]:
            raise ValueError("Out-of-range concept group")
        # [B,T,g], centered within observed history:
        a=x[:,:,g.first:g.stop]
        b=y[:,:,g.first:g.stop]
        joint=torch.minimum(rx[:,:,g.first:g.stop],ry[:,:,g.first:g.stop])
        reliability=joint.mean(dim=-1)
        weight=((reliability-minimum_joint_reliability)/
                (1.-minimum_joint_reliability)).clamp(0.,1.)
        denom=weight.sum(dim=1,keepdim=True).clamp_min(1.)
        av=(a*weight.unsqueeze(-1)).sum(dim=1,keepdim=True)/denom.unsqueeze(-1)
        bv=(b*weight.unsqueeze(-1)).sum(dim=1,keepdim=True)/denom.unsqueeze(-1)
        a=(a-av)*weight.sqrt().unsqueeze(-1)
        b=(b-bv)*weight.sqrt().unsqueeze(-1)
        g1=a@a.transpose(1,2)
        g2=b@b.transpose(1,2)
        # Separate Frobenius normalization avoids scale-only agreement
        # falsely counted as predictive identity; near-zero trajectories
        # are handled by an independent nonconstant-activation loss.
        n1=g1.flatten(1).norm(dim=-1).view(-1,1,1)
        n2=g2.flatten(1).norm(dim=-1).view(-1,1,1)
        g1=g1/(n1+epsilon)
        g2=g2/(n2+epsilon)
        difference=(g1-g2).square().mean(dim=(1,2))
        exists=(weight.sum(dim=1)>1).to(dtype=x.dtype)
        total=total+(difference*exists).sum()/exists.sum().clamp_min(1.)
        valid=valid+exists.mean()
    return total/len(groups),valid/len(groups)
