"""Research-only observability wrapper for the native additive temporal ConceptFAN.

Sensor provenance is supplied, not learned. This module checks measurement
support; it does not certify completeness, clinical meaning, or causality.
No feature or residual path bypasses the native concept decision head.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class SupportedConceptOutput:
    native: Any
    concept_observation_fraction: torch.Tensor
    supported_concepts: torch.Tensor
    abstain_from_explanation: torch.Tensor

    @property
    def logit(self) -> torch.Tensor:
        return self.native.logit

    @property
    def probability(self) -> torch.Tensor:
        return self.native.probability

    @property
    def signed_decision_contributions(self) -> torch.Tensor:
        return self.native.signed_decision_contributions


def observation_weighted_pool(
    trajectories: torch.Tensor,
    reliability: torch.Tensor,
    aggregator: nn.Module,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool [B,T,K] concepts only at supported times.

    With all reliability weights equal to 1 this matches the existing FAN
    temporal aggregator. All-zero columns return zero summary and weights.
    Reliability is measurement support, not a learned confidence score.
    """
    if trajectories.ndim != 3 or trajectories.shape != reliability.shape:
        raise ValueError("Expected matched concept trajectories and reliability [B,T,K]")
    if not bool(torch.isfinite(trajectories).all() and torch.isfinite(reliability).all()):
        raise ValueError("Non-finite concept trajectory or reliability")
    if bool(((reliability < 0.0) | (reliability > 1.0)).any()):
        raise ValueError("Reliability values must be in [0,1]")
    batch, steps, n_concepts = trajectories.shape
    valid = reliability > 0
    if mode == "static":
        index = torch.arange(steps, device=trajectories.device).view(1, steps, 1)
        last = torch.where(valid, index, -1).amax(dim=1)
        weights = torch.zeros_like(trajectories)
        weights.scatter_(1, last.clamp_min(0).unsqueeze(1),
                         (last >= 0).to(trajectories.dtype).unsqueeze(1))
    elif mode == "attention":
        if aggregator.value_weight.numel() != n_concepts:
            raise ValueError("Temporal aggregator concept count mismatch")
        if aggregator.time.shape[1] < steps:
            raise ValueError("Observation sequence exceeds initialized temporal horizon")
        scores = trajectories * aggregator.value_weight.view(1, 1, -1)
        scores = scores + aggregator.time[:, :steps, :] * aggregator.time_weight.view(1, 1, -1)
        scores = scores + aggregator.bias.view(1, 1, -1)
        # log(r) makes a continuous reliability weight equivalent to a prior
        # multiplicative weight inside the temporal softmax.
        scores = scores + reliability.clamp_min(torch.finfo(scores.dtype).tiny).log()
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1) * valid.to(scores.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
    else:
        raise ValueError("Only attention and static temporal pooling are supported")
    summary = (weights * trajectories).sum(dim=1)
    return summary, weights


class ObservationAwareConceptFAN(nn.Module):
    """Optional, backwards-compatible concept-observability architecture.

    The wrapped `core` is a MultiSetAdditiveTemporalConceptFANModel with input
    width 2*C (zero-filled measurements and binary masks). `sensor_weights`
    is a fixed [K,C] evidence map, specified independently of test labels.
    Unknown concepts contribute exactly zero to the native additive logit.

    The produced probability is NOT certified for use when abstain=True;
    a bias-only prediction is a computational fallback, not a safe decision.
    """

    def __init__(
        self,
        core: nn.Module,
        sensor_weights: torch.Tensor,
        *,
        min_observation_fraction: float = 0.10,
    ) -> None:
        super().__init__()
        if not 0.0 < min_observation_fraction <= 1.0:
            raise ValueError("min_observation_fraction must be in (0,1]")
        if getattr(core, "oracle", False):
            raise ValueError("Oracle concept override is not supported in this wrapper")
        if getattr(core, "temporal_mode", None) not in {"attention", "static"}:
            raise ValueError("Unsupported temporal aggregation mode")
        support = torch.as_tensor(sensor_weights, dtype=torch.float32).detach().clone()
        if support.ndim != 2 or support.numel() == 0 or not bool(torch.isfinite(support).all()):
            raise ValueError("Expected finite [concepts,sensors] evidence weights")
        if bool((support < 0).any()) or bool((support.sum(dim=1) <= 0).any()):
            raise ValueError("Every concept requires at least one nonnegative sensor weight")
        k, c = support.shape
        if int(core.encoder.input_proj.in_features) != 2 * c:
            raise ValueError("Core must accept measurements plus observation masks [B,T,2*C]")
        if int(core.temporal_aggregator.n_concepts) != k:
            raise ValueError("Core concept count and sensor weights must match")
        self.core = core
        self.min_observation_fraction = float(min_observation_fraction)
        self.register_buffer("sensor_weights", support / support.sum(dim=1, keepdim=True))

    def forward(self, values: torch.Tensor, observed_mask: torch.Tensor) -> SupportedConceptOutput:
        if values.ndim != 3 or values.shape != observed_mask.shape:
            raise ValueError("Expected aligned [B,T,C] measurement and mask inputs")
        if values.shape[-1] != self.sensor_weights.shape[1]:
            raise ValueError("Measurement channel count does not match provenance")
        if not bool(torch.isfinite(observed_mask).all()):
            raise ValueError("Observation mask must be finite")
        if not bool(((observed_mask == 0) | (observed_mask == 1)).all()):
            raise ValueError("Observation mask must be binary")
        observed = observed_mask.to(dtype=torch.bool)
        if not bool(torch.isfinite(values[observed]).all()):
            raise ValueError("Observed measurements must be finite")
        # Remove hidden values first. Missing NaN payloads never reach encoder.
        clean_values = torch.where(observed, values, torch.zeros_like(values))
        mask_float = observed.to(dtype=clean_values.dtype)
        features = torch.cat((clean_values, mask_float), dim=-1)
        latent = self.core.encoder(features)
        trajectories = self.core.projector(latent)
        reliability = mask_float @ self.sensor_weights.to(dtype=mask_float.dtype).T
        fraction = reliability.mean(dim=1)
        supported = fraction >= self.min_observation_fraction
        summary, time_weights = observation_weighted_pool(
            trajectories, reliability, self.core.temporal_aggregator, self.core.temporal_mode
        )
        native = self.core.forward_from_summaries(
            latent, trajectories, summary, time_weights, concept_mask=supported
        )
        if not bool(torch.isfinite(native.logit).all()):
            raise ValueError("Non-finite native decision")
        # Preserves the exact additive concept-only decision pathway.
        recovered = self.core.decision_head.bias + native.signed_decision_contributions.sum(dim=-1)
        if not bool(torch.allclose(native.logit, recovered, atol=1e-5, rtol=1e-5)):
            raise AssertionError("Concept-only logit decomposition was broken")
        if not bool(torch.all(native.signed_decision_contributions[~supported] == 0)):
            raise AssertionError("An unsupported concept influenced the prediction")
        return SupportedConceptOutput(
            native=native,
            concept_observation_fraction=fraction,
            supported_concepts=supported,
            abstain_from_explanation=~supported.all(dim=1),
        )
