"""Jointly learned observable dynamic concepts with an exact decision bottleneck.

No expert concept targets, post-hoc concept extractor, or hidden classifier
shortcut are used. The temporal operator dictionary is fixed and inspectable.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .operators import CHANNELS, OPERATORS, causal_operator_bank


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Differentiable Euclidean projection onto the probability simplex."""
    z = logits - logits.max(dim=dim, keepdim=True).values
    ordered = torch.sort(z, descending=True, dim=dim).values
    width = z.shape[dim]
    ranks = torch.arange(1, width + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.ndim
    shape[dim] = width
    ranks = ranks.view(*shape)
    shifted_cumsum = ordered.cumsum(dim=dim) - 1.0
    support = ranks * ordered > shifted_cumsum
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    threshold = shifted_cumsum.gather(dim, support_size.long() - 1) / support_size
    return (z - threshold).clamp_min(0.0)


class HardConcreteGates(nn.Module):
    """Louizos-style L0 approximation, with separate hard-gate audit."""
    def __init__(self, n_slots: int, temperature: float = 0.67):
        super().__init__()
        self.logits = nn.Parameter(torch.full((n_slots,), 1.2))
        self.temperature = float(temperature)
        self.lower = -0.1
        self.upper = 1.1

    def deterministic(self) -> torch.Tensor:
        return (torch.sigmoid(self.logits) * (self.upper - self.lower) + self.lower).clamp(0.0, 1.0)

    def expected_l0(self) -> torch.Tensor:
        shift = self.temperature * math.log(-self.lower / self.upper)
        return torch.sigmoid(self.logits - shift)

    def forward(self, hard: bool = False) -> torch.Tensor:
        if hard:
            return (self.deterministic() >= 0.5).to(dtype=self.logits.dtype)
        if not self.training:
            return self.deterministic()
        u = torch.rand_like(self.logits).clamp(1e-5, 1.0 - 1e-5)
        s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + self.logits) / self.temperature)
        return (s * (self.upper - self.lower) + self.lower).clamp(0.0, 1.0)


@dataclass
class DiscoveryOutput:
    logits: torch.Tensor
    probability: torch.Tensor
    concept_trajectories: torch.Tensor
    raw_activation: torch.Tensor
    availability: torch.Tensor
    concept_summary: torch.Tensor
    contributions: torch.Tensor
    future_prediction: torch.Tensor
    gates: torch.Tensor
    feature_weights: torch.Tensor
    decision_bias: torch.Tensor

    def decomposition_error(self) -> torch.Tensor:
        return (self.logits - (self.decision_bias + self.contributions.sum(dim=-1))).abs().max()


class PredictiveDynamicsConceptFAN(nn.Module):
    def __init__(
        self, n_slots: int = 12, n_channels: int = len(CHANNELS),
        forecast_horizon: int = 6, min_width: float = 0.08,
    ):
        super().__init__()
        if n_slots < 2 or n_channels < 2 or forecast_horizon < 1 or min_width <= 0:
            raise ValueError("Invalid concept capacity, observed channels or horizon")
        self.n_slots = int(n_slots)
        self.n_channels = int(n_channels)
        self.horizon = int(forecast_horizon)
        self.min_width = float(min_width)
        self.feature_dim = n_channels * len(OPERATORS) * 2
        self.feature_logits = nn.Parameter(torch.randn(n_slots, self.feature_dim) * 0.35)
        self.threshold_logits = nn.Parameter(torch.zeros(n_slots))
        self.width_logits = nn.Parameter(torch.full((n_slots,), -1.75))
        self.gates = HardConcreteGates(n_slots)
        self.decision_raw_weight = nn.Parameter(torch.randn(n_slots) * 0.15)
        self.decision_bias = nn.Parameter(torch.zeros(()))
        self.transition_raw = nn.Parameter(torch.eye(n_slots) * 0.3)
        self.transition_bias = nn.Parameter(torch.zeros(n_slots))
        self.future_decoder = nn.Linear(n_slots, n_channels)
        nn.init.normal_(self.future_decoder.weight, std=0.08)
        nn.init.zeros_(self.future_decoder.bias)

    def feature_weights(self) -> torch.Tensor:
        return sparsemax(self.feature_logits, dim=-1)

    def concept_boundaries(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tanh(self.threshold_logits), self.min_width + F.softplus(self.width_logits)

    def decision_weights(self) -> torch.Tensor:
        return 3.0 * torch.tanh(self.decision_raw_weight)

    def transition_matrix(self) -> torch.Tensor:
        return 0.6 * torch.tanh(self.transition_raw)

    def _concepts(
        self, values: torch.Tensor, mask: torch.Tensor, gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        operators = causal_operator_bank(values, mask, channels=CHANNELS[:self.n_channels])
        weights = self.feature_weights()
        reliability = operators.availability @ weights.T
        signal = (operators.values @ weights.T) / reliability.clamp_min(0.2)
        boundary, width = self.concept_boundaries()
        raw = torch.sigmoid((signal - boundary) / width)
        concepts = raw * reliability * gate.view(1, 1, -1)
        return concepts, raw, reliability, weights

    def _propagate(self, state: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # Stable, bounded transition with explicit sparse off-diagonal control.
        next_raw = torch.sigmoid(F.linear(state, self.transition_matrix(), self.transition_bias))
        return gate * (0.7 * state + 0.3 * next_raw)

    def forecast_from_concepts(self, final_state: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        trajectory = []
        state = final_state
        for _ in range(self.horizon):
            state = self._propagate(state, gate)
            trajectory.append(self.future_decoder(state))
        return torch.stack(trajectory, dim=1)

    def forward(self, values: torch.Tensor, mask: torch.Tensor, *, hard_gates: bool = False) -> DiscoveryOutput:
        if values.ndim != 3 or values.shape[-1] != self.n_channels:
            raise ValueError("Expected standardized clinical observations [B,T,channels]")
        gate = self.gates(hard=hard_gates)
        concepts, raw, reliability, feature_weights = self._concepts(values, mask, gate)
        summaries = concepts[:, -min(6, concepts.shape[1]):].mean(dim=1)
        contribution = (summaries - 0.5 * gate.view(1, -1)) * self.decision_weights()
        logit = self.decision_bias + contribution.sum(dim=-1)
        forecast = self.forecast_from_concepts(concepts[:, -1, :], gate)
        return DiscoveryOutput(
            logits=logit, probability=torch.sigmoid(logit),
            concept_trajectories=concepts, raw_activation=raw,
            availability=reliability, concept_summary=summaries,
            contributions=contribution, future_prediction=forecast,
            gates=gate, feature_weights=feature_weights,
            decision_bias=self.decision_bias,
        )

    def intervene(self, output: DiscoveryOutput, concept_id: int, summary_override: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact *model-level* direct effect with other concept summaries fixed.

        This is not an intervention on a physiological mechanism.
        """
        if not 0 <= concept_id < self.n_slots or summary_override.shape != output.concept_summary[:, concept_id].shape:
            raise ValueError("Invalid concept index or override shape")
        changed = output.concept_summary.clone()
        changed[:, concept_id] = summary_override
        contributions = (changed - 0.5 * output.gates.view(1, -1)) * self.decision_weights()
        return self.decision_bias + contributions.sum(dim=-1), contributions


def masked_future_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Forecast, target and target mask must agree")
    element = F.smooth_l1_loss(pred, target, reduction="none")
    return (element * mask).sum() / mask.sum().clamp_min(1.0)


def discovery_loss(
    model: PredictiveDynamicsConceptFAN,
    output: DiscoveryOutput,
    labels: torch.Tensor,
    future: torch.Tensor,
    future_mask: torch.Tensor,
    *,
    mode: str,
    pos_weight: torch.Tensor,
    loss_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Predictions AND forecasts have the same concept bottleneck."""
    if mode not in ("task_only", "task_future", "full"):
        raise ValueError("Unknown preregistered ablation")
    task = F.binary_cross_entropy_with_logits(output.logits, labels, pos_weight=pos_weight)
    forecast = masked_future_huber(output.future_prediction, future, future_mask)
    concept = output.concept_trajectories
    transitions = model._propagate(concept[:, :-1], output.gates)
    dynamic = F.smooth_l1_loss(transitions, concept[:, 1:])
    w = output.feature_weights
    support = (torch.sqrt(w + 1e-8).sum(dim=-1) - 1.0).clamp_min(0.0)
    complexity = model.gates.expected_l0().sum()
    # Redundancy controls duplicate measurement definitions, not legitimate
    # physiological correlations among concepts.
    overlap = w @ w.T
    overlap = overlap - torch.diag_embed(overlap.diagonal())
    redundancy = (overlap.square()).mean()
    # A feature slot which outputs a constant across patients and time does
    # not become an "interpretable factor" just because its rank is stable.
    variance = concept.var(dim=(0, 1), unbiased=False)
    nonconstant = (F.relu(0.005 - variance) * model.gates.expected_l0()).mean()
    transition_matrix = model.transition_matrix()
    offdiag = transition_matrix - torch.diag_embed(transition_matrix.diagonal())
    sparse_dynamics = offdiag.abs().mean()

    zero = task.new_zeros(())
    total = task
    if mode in ("task_future", "full"):
        total = total + float(loss_weights["future"]) * forecast
    if mode == "full":
        total = total + float(loss_weights["dynamics"]) * dynamic
        total = total + float(loss_weights["mdl"]) * complexity
        total = total + float(loss_weights["support"]) * support.mean()
        total = total + float(loss_weights["redundancy"]) * redundancy
        total = total + float(loss_weights["nonconstant"]) * nonconstant
        total = total + float(loss_weights["transition_l1"]) * sparse_dynamics
    terms = dict(task=task, future=forecast, dynamics=dynamic,
                 expected_active=complexity, support=support.mean(),
                 redundancy=redundancy, nonconstant=nonconstant,
                 transition_l1=sparse_dynamics, total=total)
    return total, terms
