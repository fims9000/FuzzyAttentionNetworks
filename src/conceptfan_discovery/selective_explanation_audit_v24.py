"""V24 research: support audit for the existing V10/V19 native concept graph.

This is a *non-destructive* risk/explanation wrapper, NOT a new predictor.
It leaves physiological and care-observation contributions exactly as trained.
Insufficient measured support means no justified physiological explanation,
NOT automatic evidence that the physical concept dictionary is incomplete.
No medical diagnosis, causal identity, or intervention validity is certified.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class NativeExplanationSupport:
    native: Any
    physical_reliability: torch.Tensor
    physical_support: torch.Tensor
    native_risk_active: torch.Tensor
    unsupported_risk_concepts: torch.Tensor
    abstain_physiological_explanation: torch.Tensor
    risk_contributions_unchanged: bool = True

    @property
    def logits(self) -> torch.Tensor:
        return self.native.logits

    @property
    def probabilities(self) -> torch.Tensor:
        return self.native.probabilities


def _supported_temporal_window(availability: torch.Tensor, mode: str) -> torch.Tensor:
    """Conservative support for V10 recent pooling or V19 early+recent shape.

    Support here is solely *measured/operator availability*. Not a posterior
    probability of physical validity. The V19 gate requires early and recent
    time windows because its shape bank contains both time regions.
    """
    if availability.ndim != 3 or availability.shape[1] < 4:
        raise ValueError("Expected [batch,time,physical concepts], at least four steps")
    if not torch.isfinite(availability).all() or ((availability < 0) | (availability > 1)).any():
        raise ValueError("Concept measurement reliability must be finite and in [0,1]")
    steps = availability.shape[1]
    if mode == "legacy":
        return availability[:, -min(6, steps):, :].mean(dim=1)
    if mode in ("fixed_shape", "learned_shape"):
        half = min(6, steps // 2)
        early = availability[:, :half, :].mean(dim=1)
        recent = availability[:, -half:, :].mean(dim=1)
        return torch.minimum(early, recent)
    raise ValueError("Unknown native temporal readout mode")


class NativeV10SelectiveExplanationAudit(nn.Module):
    """Wrap native MultiScaleDualProcessConceptFAN without mutating prediction.

    * Physical and observation-process components remain separate.
    * Observation support is read from the native physical concept output,
      not reconstructed from assumed sensor-to-concept loadings.
    * Descriptive-only (policy=2) and zero-weighted physical concepts do not
      cause spurious risk-explanation refusals.
    * If no physical concepts contribute to risk, no physiological
      explanation is claimed, even if care-observation concepts predict.
    * Threshold is a fixed configuration chosen without TEST labels.
    """

    def __init__(self, core: nn.Module, *, min_recent_reliability: float = 0.25,
                 minimum_effective_risk_weight: float = 1e-4) -> None:
        super().__init__()
        if not (0.0 < min_recent_reliability <= 1.0):
            raise ValueError("min_recent_reliability must be in (0,1]")
        if minimum_effective_risk_weight < 0.0:
            raise ValueError("minimum_effective_risk_weight must be nonnegative")
        if not hasattr(core, "physical") or not hasattr(core.physical, "decision_weights"):
            raise TypeError("Expected V10 dual-process core with native physical decision weights")
        if not hasattr(core, "process"):
            raise TypeError("Expected separate native observation-process concept branch")
        self.core = core
        self.min_recent_reliability = float(min_recent_reliability)
        self.minimum_effective_risk_weight = float(minimum_effective_risk_weight)

    def forward(self, values: torch.Tensor, observed_mask: torch.Tensor,
                *, process_mask: torch.Tensor | None = None) -> NativeExplanationSupport:
        if values.ndim != 3 or values.shape != observed_mask.shape:
            raise ValueError("Values and masks must have identical [B,T,C] shapes")
        if not torch.isfinite(observed_mask).all() or not torch.logical_or(
            observed_mask == 0, observed_mask == 1
        ).all():
            raise ValueError("Observation mask must be finite and binary")
        observed = observed_mask.bool()
        if not torch.isfinite(values[observed]).all():
            raise ValueError("Observed measurements must be finite")
        recorded_mask = observed_mask if process_mask is None else process_mask
        if recorded_mask.shape != observed_mask.shape:
            raise ValueError("Original observation schedule must match physical mask shape")
        if not torch.isfinite(recorded_mask).all() or not torch.logical_or(
            recorded_mask == 0, recorded_mask == 1
        ).all() or torch.any(observed_mask > recorded_mask):
            raise ValueError("Physical mask must be a subset of the original binary observation schedule")
        # Always discard hidden NaNs/placeholder payloads before the old native
        # operators, so missing values never act as measurements.
        clean = torch.where(observed, values, torch.zeros_like(values))
        native = self.core(clean, observed_mask, process_mask=recorded_mask)
        availability = native.physical.availability
        if availability.ndim != 3 or availability.shape[:2] != values.shape[:2]:
            raise ValueError("Native physical availability has incompatible dimensions")
        temporal_mode = getattr(self.core, "temporal_readout_mode", "legacy")
        observed_support = _supported_temporal_window(availability, temporal_mode)
        weights = self.core.physical.decision_weights()
        gates = native.physical.gates
        if weights.shape != gates.shape or observed_support.shape[-1] != weights.numel():
            raise ValueError("Native concept weight, gate and support dimensions disagree")
        effective_weights = weights.abs() * gates.abs()
        active = effective_weights > self.minimum_effective_risk_weight
        supported = observed_support >= self.min_recent_reliability
        unsupported = (~supported) & active.unsqueeze(0)
        abstain = unsupported.any(dim=1) | (~active.any()).expand(values.shape[0])
        if not torch.isfinite(native.logits).all():
            raise ValueError("Nonfinite native risk logits")
        # This audit does not get to edit the original logit or contributions.
        if native.decomposition_error().item() > 2e-5:
            raise AssertionError("Native exact-additive logit invariant violated")
        return NativeExplanationSupport(
            native=native,
            physical_reliability=observed_support,
            physical_support=supported,
            native_risk_active=active,
            unsupported_risk_concepts=unsupported,
            abstain_physiological_explanation=abstain,
        )
