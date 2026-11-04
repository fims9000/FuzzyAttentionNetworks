"""V19: decision-time shape of *native* concept trajectories.

The V10 decision path reduces each concept trajectory to its final-six-step
mean, discarding information about early-to-recent changes, last value and
variability. V19 optionally replaces ONLY that pooling with a small,
interpretable, differentiable *signed mixture of concept-only statistics*.

No raw measurement, future outcome, oracle factor or parallel classifier is
fed to this readout; the physical concept generator, reconstruction and
forecast still use the same native concept trajectories. The exact additive
logit and concept-level intervention algebra are retained.

This is an architectural hypothesis, not demonstrated superiority.
"""
from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiscale_physiology_v10 import (
    MultiScaleUnknownNeutralConceptFAN,
    FeedbackAdaptiveMultiScaleUnknownNeutralConceptFAN,
)


SHAPE_OPERATORS = (
    "recent_mean6",
    "last_value",
    "recent_minus_early_mean",
    "recent_minus_early_absolute_change",
)


class NativeTemporalReadoutMixin:
    """Expose a signed shape dictionary, not an uninterpretable attention MLP.

    Every summary component is in [0, gate] when C is in [0, gate];
    negative forms are gate - component. The convex mixture remains in
    [0, gate] and UNKNOWN=.5*gate if all measurements are absent.
    """

    def __init__(self, *, temporal_shape_trainable: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.temporal_readout_logits = nn.Parameter(
            torch.full((self.n_slots, len(SHAPE_OPERATORS)*2), -1.5)
        )
        with torch.no_grad():
            self.temporal_readout_logits[:, 0] = 2.5
        self.temporal_readout_logits.requires_grad_(temporal_shape_trainable)
        self.temporal_shape_trainable = bool(temporal_shape_trainable)

    def temporal_readout_weights(self) -> torch.Tensor:
        return F.softmax(self.temporal_readout_logits, dim=-1)

    def temporal_shape_bank(
        self, trajectory: torch.Tensor, gate: torch.Tensor,
    ) -> torch.Tensor:
        if trajectory.ndim != 3 or trajectory.shape[-1] != self.n_slots:
            raise ValueError("Expected native concept trajectories [B,T,K]")
        if trajectory.shape[1] < 4 or gate.shape != (self.n_slots,):
            raise ValueError("At least four observed history steps are required")
        window = min(6, trajectory.shape[1] // 2)
        early, recent = trajectory[:, :window, :], trajectory[:, -window:, :]
        recent_mean = recent.mean(dim=1)
        last = trajectory[:, -1, :]
        early_mean = early.mean(dim=1)
        neutral = .5 * gate.view(1,-1)
        # Differences have bounded support [-gate, gate].
        mean_change = neutral + .5 * (recent_mean - early_mean)
        early_steps = (early[:, 1:, :] - early[:, :-1, :]).abs().mean(dim=1)
        recent_steps = (recent[:, 1:, :] - recent[:, :-1, :]).abs().mean(dim=1)
        variation_change = neutral + .5 * (recent_steps - early_steps)
        base = torch.stack(
            [recent_mean,last,mean_change,variation_change], dim=-1,
        )
        sign_reversed = gate.view(1,self.n_slots,1) - base
        return torch.cat([base,sign_reversed],dim=-1)

    def native_shape_summary(
        self, trajectory: torch.Tensor, gate: torch.Tensor,
    ) -> torch.Tensor:
        bank = self.temporal_shape_bank(trajectory,gate)
        mixture = self.temporal_readout_weights()
        return (bank * mixture.unsqueeze(0)).sum(dim=-1)

    def forward(self, values: torch.Tensor, mask: torch.Tensor,
                *, hard_gates: bool = False,
                selected_slots: torch.Tensor | None = None):
        out = super().forward(
            values,mask,hard_gates=hard_gates,selected_slots=selected_slots,
        )
        summary = self.native_shape_summary(out.concept_trajectories,out.gates)
        contributions = (summary-.5*out.gates.view(1,-1))*self.decision_weights()
        logits = self.decision_bias + contributions.sum(dim=-1)
        return replace(
            out,concept_summary=summary,contributions=contributions,
            logits=logits,probability=torch.sigmoid(logits),
        )

    def temporal_readout_passport(self) -> dict:
        weights=self.temporal_readout_weights().detach().cpu().numpy()
        names=tuple(f"{direction}|{operator}"
                    for direction in ("positive","negative")
                    for operator in SHAPE_OPERATORS)
        return {
            "formula":"bias + sum_k decision_weight[k] * (weighted_native_shape[k] - 0.5 * native_gate[k])",
            "operator_names":list(names),
            "learned_native_shape_mixture":weights.tolist(),
            "trainable":self.temporal_shape_trainable,
            "source":"only existing native concept trajectories; no raw-input bypass",
            "forecast_and_reconstruction_use_unchanged_native_concept_trajectories":True,
            "physiological_semantics_proven":False,
        }


class TemporalReadoutMultiScaleConceptFAN(
    NativeTemporalReadoutMixin, MultiScaleUnknownNeutralConceptFAN,
):
    """Unchanged V10 operator weights; new temporal decision readout."""


class AdaptiveTemporalReadoutMultiScaleConceptFAN(
    NativeTemporalReadoutMixin, FeedbackAdaptiveMultiScaleUnknownNeutralConceptFAN,
):
    """Same native readout with V16 4% differentiable operator bridge."""
