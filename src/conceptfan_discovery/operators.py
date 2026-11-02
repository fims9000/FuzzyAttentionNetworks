"""Causal, inspectable feature operators for automated time-series concepts.

Operators have no learned labels and never use the future horizon, outcomes,
or latent generator states. Every activation is traceable to a measured
channel, a mathematical operator, and a signed direction.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


CHANNELS = (
    "temperature", "heart_rate", "mean_arterial_pressure", "lactate",
    "white_blood_cells", "creatinine", "urine_output", "oxygen_saturation",
)
OPERATORS = ("level", "recent_mean4", "first_difference", "deviation_from_recent_mean4")


@dataclass
class OperatorBank:
    values: torch.Tensor
    availability: torch.Tensor
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.values.ndim != 3 or self.availability.shape != self.values.shape:
            raise ValueError("operator values and availability must have [B,T,D] shape")
        if self.values.shape[-1] != len(self.names):
            raise ValueError("operator dictionary and tensor dimension disagree")


def _causal_mean(x: torch.Tensor, mask: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor]:
    if window < 1:
        raise ValueError("window must be positive")
    window = min(window, x.shape[1])
    weighted = torch.cumsum(x * mask, dim=1)
    counts = torch.cumsum(mask, dim=1)
    left = torch.cat([torch.zeros_like(weighted[:, :window]), weighted[:, :-window]], dim=1)
    left_count = torch.cat([torch.zeros_like(counts[:, :window]), counts[:, :-window]], dim=1)
    count = counts - left_count
    return (weighted - left) / count.clamp_min(1.0), (count > 0).to(dtype=x.dtype)


def causal_operator_bank(
    normalized_values: torch.Tensor,
    observed_mask: torch.Tensor,
    *,
    channels: tuple[str, ...] = CHANNELS,
    window: int = 4,
) -> OperatorBank:
    """Return 2 signs x 4 known operations x observed measurement channels.

    + and - are explicit signed operators; sparse nonnegative learned mixtures
    can represent rising or falling deviations without an uninterpretable
    unconstrained projection. No operator sees future measurements.
    """
    x = normalized_values
    m = observed_mask.to(dtype=x.dtype)
    if x.ndim != 3 or m.shape != x.shape or x.shape[-1] != len(channels):
        raise ValueError("Expected values and mask [B,T,n_named_channels]")
    if x.shape[1] < 2 or not torch.isfinite(x).all() or not torch.isfinite(m).all():
        raise ValueError("Need >=2 time steps and finite input tensors")
    if torch.any((m < 0) | (m > 1)):
        raise ValueError("observation mask must lie in [0,1]")

    smooth, smooth_available = _causal_mean(x, m, window)
    prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
    prev_mask = torch.cat([torch.zeros_like(m[:, :1]), m[:, :-1]], dim=1)
    both = m * prev_mask
    operators = (
        (torch.tanh(x / 2.0) * m, m),
        (torch.tanh(smooth / 2.0) * smooth_available, smooth_available),
        (torch.tanh((x - prev) / 2.0) * both, both),
        (torch.tanh((x - smooth) / 2.0) * m, m),
    )
    positive = torch.cat([value for value, _ in operators], dim=-1)
    available = torch.cat([flag for _, flag in operators], dim=-1)
    signed = torch.cat([positive, -positive], dim=-1)
    availability = torch.cat([available, available], dim=-1)
    names = tuple(
        f"{channel}|{operation}|{direction}"
        for direction in ("positive", "negative")
        for operation in OPERATORS
        for channel in channels
    )
    return OperatorBank(signed, availability, names)
