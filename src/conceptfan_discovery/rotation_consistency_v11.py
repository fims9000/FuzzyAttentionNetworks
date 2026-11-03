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
        # Use the exact Gram identity in group-width (K_g), not time (T):
        # ||AA' - BB'||_F^2
        #     = ||A'A||_F^2 + ||B'B||_F^2 - 2||A'B||_F^2.
        # This avoids a [B,T,T] tensor and keeps within-group rotation
        # invariance, with the SAME normalization as the full Gram formula.
        gram_a=a.transpose(1,2)@a
        gram_b=b.transpose(1,2)@b
        gram_cross=a.transpose(1,2)@b
        # vector_norm defines a finite zero-norm backward subgradient;
        # sqrt(sum(square(.))) would produce inf * 0 -> NaN at exactly
        # constant or fully unobserved concept histories.
        norm_a=torch.linalg.vector_norm(gram_a.flatten(1),dim=1)
        norm_b=torch.linalg.vector_norm(gram_b.flatten(1),dim=1)
        cross=gram_cross.square().sum(dim=(1,2))
        da=norm_a+epsilon
        db=norm_b+epsilon
        difference=(
            norm_a.square()/da.square()
            + norm_b.square()/db.square()
            -2.*cross/(da*db)
        ).clamp_min(0.)/float(a.shape[1]**2)
        exists=(weight.sum(dim=1)>1).to(dtype=x.dtype)
        total=total+(difference*exists).sum()/exists.sum().clamp_min(1.)
        valid=valid+exists.mean()
    return total/len(groups),valid/len(groups)
