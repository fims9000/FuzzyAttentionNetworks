"""V6: explicit UNKNOWN concept state and observation-view invariance.

The previous model multiplies every concept by coverage and then subtracts
0.5*gate from its logit contribution. With NO measurement, C=0, so
contribution=-0.5*beta. Missingness therefore appears as a directional
task concept even when no real physiological state was measured.

The new variant uses neutral concept state C=0.5*gate at zero coverage:

   C[t,k] = gate[k] * (0.5 + reliability[t,k] * (raw[t,k]-0.5))

and adds an optional *train-time*, label-free agreement penalty between
two independently thinned views of the same observed training history.
Predictions, reconstructions and forecasts all remain routed only through
the shared native concept bottleneck. The neutral state means UNKNOWN,
not necessarily clinically healthy or zero risk.

This does not identify missing-not-at-random biology or prove clinical
correctness. It is a falsifiable missingness-robustness architecture.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .operators import causal_operator_bank
from .spectral_concepts_v3 import SpectralConceptFAN
from .model import DiscoveryOutput


class UnknownNeutralSpectralConceptFAN(SpectralConceptFAN):
    """A genuinely distinct training-time architecture, not post-hoc saliency."""

    def _concepts(
        self, values:torch.Tensor, mask:torch.Tensor, gate:torch.Tensor,
    ) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        operators=causal_operator_bank(values,mask,channels=self.channels)
        weights=self.feature_weights()
        reliability=(operators.availability@weights.T).clamp(0.,1.)
        numerator=operators.values@weights.T
        signal=numerator/reliability.clamp_min(.2)
        threshold,width=self.concept_boundaries()
        raw=torch.sigmoid((signal-threshold)/width)
        neutral_coverage=.5+reliability*(raw-.5)
        trajectory=neutral_coverage*gate.view(1,1,-1)
        return trajectory,raw,reliability,weights


def cross_view_concept_consistency(
    source:DiscoveryOutput,
    altered:DiscoveryOutput,
    *,
    minimum_joint_reliability:float=.20,
)->tuple[torch.Tensor,torch.Tensor]:
    """Compare training-native concept state where BOTH views measured enough.

    Exact same model and same patient/time coordinate. No latent oracle
    labels, future data or post-hoc concept extractor are used.
    """
    if not 0.<=minimum_joint_reliability<1.:
        raise ValueError("Supported joint confidence threshold in [0,1)")
    if source.concept_trajectories.shape!=altered.concept_trajectories.shape:
        raise ValueError("View concepts must have identical [B,T,K] coordinates")
    if source.availability.shape!=source.concept_trajectories.shape or (
        altered.availability.shape!=altered.concept_trajectories.shape
    ):
        raise ValueError("Matched concept reliability is required")
    confidence=torch.minimum(
        source.availability,altered.availability
    ).clamp(0.,1.)
    confidence=((confidence-minimum_joint_reliability)/
                (1.-minimum_joint_reliability)).clamp(0.,1.)
    residual=(source.concept_trajectories-altered.concept_trajectories).square()
    penalty=(residual*confidence).sum()/confidence.sum().clamp_min(1.)
    return penalty,confidence.detach().mean()
