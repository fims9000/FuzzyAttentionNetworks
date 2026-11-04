"""V10: train-time multi-resolution physiological evidence, V7 care branch preserved.

The model learns which *past* observation scales support each physiological
concept. Every scale uses the same inspectable sparse signed operator
dictionary. All supervision, forecasting and masked denoising run through the
same native concept trajectories. No expert concept targets, raw-value
classifier shortcut, test labels, or medical causal claims.

With no observed history, every scale has zero reliability and the physical
concept stays UNKNOWN (neutral), regardless of its learned threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .operators import _causal_mean, causal_operator_bank
from .unknown_neutral_v6 import UnknownNeutralSpectralConceptFAN
from .observation_process_concepts_v7 import DualProcessConceptFAN


SCALES = (1, 4, 12, 24)


def causal_observation_mean(
    values: torch.Tensor, mask: torch.Tensor, window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average only observed historical values, never future or imputed data."""
    if values.ndim != 3 or values.shape != mask.shape or window < 1:
        raise ValueError("Expected [patients,time,channels] and positive window")
    if not torch.isfinite(values).all() or not torch.isfinite(mask).all():
        raise ValueError("Inputs must be finite")
    if torch.any((mask < 0) | (mask > 1)):
        raise ValueError("Observation indicator must lie in [0,1]")
    # Explicitly discard placeholder values at unobserved times.
    observed_values = values * mask
    if window == 1:
        return observed_values, mask
    result, observed = _causal_mean(observed_values, mask, window)
    return result, observed


class MultiScaleUnknownNeutralConceptFAN(UnknownNeutralSpectralConceptFAN):
    """V7 physiological model with learned, inspectable temporal-scale gates."""

    def __init__(
        self, *,
        observable_loadings: np.ndarray,
        channels: tuple[str, ...],
        forecast_horizon: int,
        freeze_definitions: bool = False,
        scales: tuple[int, ...] = SCALES,
    ):
        super().__init__(
            observable_loadings=observable_loadings,
            channels=channels,
            forecast_horizon=forecast_horizon,
            freeze_definitions=freeze_definitions,
        )
        if not scales or scales[0] != 1 or any(x < 1 for x in scales) or tuple(sorted(set(scales))) != scales:
            raise ValueError("Sorted unique scales must begin with 1")
        self.scales = tuple(int(s) for s in scales)
        self.scale_logits = nn.Parameter(torch.zeros(self.n_slots, len(self.scales)))
        # Strong but not exclusive direct-value prior. The optimizer may
        # transfer mass to 4, 12, or 24-hour contexts.
        with torch.no_grad():
            self.scale_logits[:, 0] = 1.5

    def scale_weights(self) -> torch.Tensor:
        return torch.softmax(self.scale_logits, dim=-1)

    def concept_formula(self) -> str:
        return (
            "gate[k] * sum_window pi[k,window] * (0.5 + "
            "reliability[window,t,k] * "
            "(sigmoid((signed_observed_operator_sum[window,t,k]"
            "-threshold[k])/width[k])-0.5)); "
            "pi=softmax(trainable_scale_logits); no support => UNKNOWN"
        )

    def _concepts(
        self, values: torch.Tensor, mask: torch.Tensor, gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if values.shape != mask.shape or values.ndim != 3:
            raise ValueError("Values and recorded availability must be [B,T,channels]")
        weights = self.feature_weights()
        threshold, width = self.concept_boundaries()
        scale_weights = self.scale_weights()
        concepts, raws, confidences = [], [], []
        for window in self.scales:
            values_window, observed_window = causal_observation_mean(values, mask, window)
            operator = causal_operator_bank(
                values_window, observed_window, channels=self.channels
            )
            reliability = (operator.availability @ weights.T).clamp(0.0, 1.0)
            signed_score = (operator.values @ weights.T) / reliability.clamp_min(.2)
            raw = torch.sigmoid((signed_score - threshold) / width)
            # No spurious positive/negative physiological evidence solely
            # because an entire measurement history is unobserved.
            neutral = .5 + reliability * (raw - .5)
            concepts.append(neutral)
            raws.append(raw)
            confidences.append(reliability)
        stacked = torch.stack(concepts, dim=-1)
        rw = scale_weights.view(1, 1, self.n_slots, len(self.scales))
        result = (stacked * rw).sum(dim=-1) * gate.view(1, 1, -1)
        pooled_raw = (torch.stack(raws, dim=-1) * rw).sum(dim=-1)
        pooled_reliability = (torch.stack(confidences, dim=-1) * rw).sum(dim=-1)
        return result, pooled_raw, pooled_reliability, weights

    def multiscale_passport(self) -> dict:
        with torch.no_grad():
            pi = self.scale_weights().detach().cpu().numpy()
            names = list(self.channels)
            return {
                "model": "train-time multi-scale physiological concepts",
                "temporal_context_windows_hours": list(self.scales),
                "concept_temporal_weights": [
                    {"concept_slot": int(k),
                     "weights": {str(w): float(pi[k, j])
                                 for j, w in enumerate(self.scales)}}
                    for k in range(self.n_slots)
                ],
                "frozen_or_trainable_scale_mixture": bool(self.scale_logits.requires_grad),
                "learned_signed_sensor_operators": self.concept_basis_manifest(),
                "no_real_physio_label_claim": True,
                "zero_observation_is_unknown_neutral": True,
                "causal_temporal_inputs_only": True,
            }


class FeedbackAdaptiveMultiScaleUnknownNeutralConceptFAN(MultiScaleUnknownNeutralConceptFAN):
    """Near-sparse feedback mode: enable gradients from labels to operator mix.

    Plain sparsemax at the exactly-one-hot spectral initialization can have a
    zero Jacobian: expert feedback changes thresholds but not operator
    definitions. A fixed 4% softmax bridge makes every measured operator
    locally trainable while preserving an explicit signed observable formula.
    This is an architecture ablation, not an identifiability guarantee.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("expert_exploration_fraction",torch.tensor(.04))

    def feature_weights(self) -> torch.Tensor:
        base = super().feature_weights()
        epsilon = self.expert_exploration_fraction.to(
            dtype=base.dtype,device=base.device
        )
        return ((1.0-epsilon)*base
                +epsilon*torch.softmax(self.feature_logits,dim=-1))


class MultiScaleDualProcessConceptFAN(DualProcessConceptFAN):
    """Same mask-only V7 observation process; only physical formation changes."""
    def __init__(
        self, *, observable_loadings: np.ndarray, channels: tuple[str, ...],
        forecast_horizon: int, n_observation_concepts: int = 3,
        scales: tuple[int, ...] = SCALES,
        adaptive_feature_gradients: bool = False,
    ):
        super().__init__(
            observable_loadings=observable_loadings, channels=channels,
            forecast_horizon=forecast_horizon,
            n_observation_concepts=n_observation_concepts,
        )
        physical_type = (FeedbackAdaptiveMultiScaleUnknownNeutralConceptFAN
                         if adaptive_feature_gradients
                         else MultiScaleUnknownNeutralConceptFAN)
        self.physical = physical_type(
            observable_loadings=observable_loadings, channels=channels,
            forecast_horizon=forecast_horizon,
            scales=scales,
        )


def physiological_scale_entropy(
    model: MultiScaleDualProcessConceptFAN,
) -> torch.Tensor:
    """Optional train-time pressure for a small readable scale support.

    This is *not* an empirical concept validity guarantee.
    """
    pi = model.physical.scale_weights()
    return -(pi * pi.clamp_min(1e-10).log()).sum(dim=-1).mean()


def expanded_multiscale_definition(
    model: DualProcessConceptFAN,
) -> np.ndarray:
    """Use temporal scale AND signed sensor operators to align concepts.

    A baseline V7 explanation is concentrated at the direct scale (1h).
    Matcher never consults patient outcomes or oracle factor identities.
    """
    w = model.physical.feature_weights().detach().cpu().numpy()
    if isinstance(model.physical, MultiScaleUnknownNeutralConceptFAN):
        pi = model.physical.scale_weights().detach().cpu().numpy()
    else:
        pi = np.zeros((w.shape[0], len(SCALES)), dtype=np.float64)
        pi[:, 0] = 1.
    return (pi[:, :, None] * w[:, None, :]).reshape(w.shape[0], -1)
