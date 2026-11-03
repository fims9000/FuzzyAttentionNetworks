"""PD-ConceptFAN v2: masked-observation prediction through learned concepts.

Medical/ground-truth concept targets are never supplied. Besides downstream
prediction and 6-hour observation forecast, the same concept bottleneck must
predict randomly hidden *actually observed* channels. This strengthens the
notion of predictive sufficiency without inventing causal identification.
"""
from __future__ import annotations

from dataclasses import replace
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import PredictiveDynamicsConceptFAN, DiscoveryOutput, discovery_loss, masked_future_huber


class DenoisingPredictiveConceptFAN(PredictiveDynamicsConceptFAN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_decoder = nn.Linear(self.n_slots, self.n_channels)
        nn.init.normal_(self.observation_decoder.weight, std=.04)
        nn.init.zeros_(self.observation_decoder.bias)

    def forward(self, values: torch.Tensor, mask: torch.Tensor,
                *, hard_gates: bool = False, selected_slots: torch.Tensor | None = None) -> DiscoveryOutput:
        result = super().forward(values, mask,
                                 hard_gates=hard_gates, selected_slots=selected_slots)
        # No shortcut to the decoder: it sees solely the concept trajectories.
        return replace(result, reconstruction=self.observation_decoder(result.concept_trajectories))


def masked_observation_view(
    values: torch.Tensor, observed_mask: torch.Tensor,
    *,
    fraction: float = .20,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Temporarily hide some observed input cells; predict them only via concepts.

    Random hiding is independent of labels, synthetic planted states and
    validation/test outcomes. Held-out observed values are targets, not inputs.
    """
    if values.shape != observed_mask.shape or values.ndim != 3:
        raise ValueError("Expected matched [batch, time, observed_channels] arrays")
    if not (0 < fraction < 1):
        raise ValueError("Holdout fraction must be in (0,1)")
    eligible = observed_mask > 0
    random = torch.rand(values.shape, generator=generator,
                        device=values.device, dtype=values.dtype)
    hidden = eligible & (random < fraction)
    # Sparse ICU data need a deterministic fallback when no entry is masked.
    # Each patient's first observed measurement is hidden only when no other
    # measurement was randomly held out for that patient.
    has_observed = eligible.flatten(1).any(dim=1)
    missing_hidden = has_observed & ~hidden.flatten(1).any(dim=1)
    if missing_hidden.any():
        flat = hidden.flatten(1)
        first_valid = eligible.flatten(1).float().argmax(dim=1)
        row = torch.arange(values.shape[0], device=values.device)
        flat[row[missing_hidden], first_valid[missing_hidden]] = True
    withheld = hidden.to(dtype=values.dtype)
    visible_mask = observed_mask * (1.0 - withheld)
    x_masked = values * visible_mask
    return x_masked, visible_mask, withheld


def v2_loss(
    model: DenoisingPredictiveConceptFAN, output: DiscoveryOutput,
    labels: torch.Tensor, future: torch.Tensor, future_mask: torch.Tensor,
    observed_target: torch.Tensor, artificially_withheld_mask: torch.Tensor,
    *, mode: str, pos_weight: torch.Tensor, loss_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if mode not in ("task_only", "task_future", "full_without_denoising", "full"):
        raise ValueError("Unknown pre-registered v2 ablation")
    original_mode = "full" if mode in ("full", "full_without_denoising") else mode
    objective, parts = discovery_loss(
        model, output, labels, future, future_mask,
        mode=original_mode, pos_weight=pos_weight, loss_weights=loss_weights,
    )
    if output.reconstruction is None:
        raise ValueError("Denoising model must supply concept-mediated reconstruction")
    observed_loss = masked_future_huber(
        output.reconstruction, observed_target, artificially_withheld_mask,
    )
    if mode == "full":
        objective = objective + float(loss_weights["denoising"]) * observed_loss
    parts["masked_observation"] = observed_loss
    parts["total"] = objective
    return objective, parts
