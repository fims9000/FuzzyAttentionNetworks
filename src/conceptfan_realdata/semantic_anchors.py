"""Experimental semantically anchored ConceptFAN, without a hidden logit bypass.

All constants (centers, widths, coefficients, reference time weights) are
estimated from TRAIN ONLY and shared across independent encoder trainings.
The purpose is to test identifiability, not guarantee increased stability.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from .models import AdditiveConceptModel, RealModelOutput


@dataclass(frozen=True)
class SemanticAnchor:
    center: np.ndarray
    width: np.ndarray
    beta: np.ndarray
    bias: float
    time_weights: np.ndarray

    def to_dict(self) -> dict:
        return {
            "center": self.center.tolist(),
            "width": self.width.tolist(),
            "beta": self.beta.tolist(),
            "bias": float(self.bias),
            "time_weights": self.time_weights.tolist(),
        }


def fixed_temporal_weights(n_steps: int, recency_decay: float = 0.08) -> np.ndarray:
    if n_steps < 2 or recency_decay < 0:
        raise ValueError("n_steps >= 2 and nonnegative decay required")
    weights = np.exp(-recency_decay * (n_steps - 1 - np.arange(n_steps, dtype=np.float64)))
    return weights / weights.sum()


def anchor_evidence(summaries: np.ndarray, center: np.ndarray, width: np.ndarray) -> np.ndarray:
    z = (np.asarray(summaries) - center) / width
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z)) - 0.5


def fit_semantic_anchor(
    train_concepts: np.ndarray,
    train_labels: np.ndarray,
    *,
    recency_decay: float = 0.08,
    logistic_c: float = 1.0,
    minimum_width: float = 0.05,
) -> SemanticAnchor:
    """Fit an exactly additive frozen head from TRAIN concept labels.

    Only fully observed concept trajectories are accepted in v1. Applying
    this to partially observed PhysioNet proxies without a prespecified
    missingness treatment would otherwise bake imputation leakage into
    the anchor.
    """
    c = np.asarray(train_concepts, dtype=np.float64)
    y = np.asarray(train_labels)
    if c.ndim != 3 or c.shape[0] != len(y) or c.shape[-1] < 2:
        raise ValueError("Expected concept trajectories [train patients,steps,concepts]")
    if not np.isfinite(c).all() or np.any(c < 0) or np.any(c > 1):
        raise ValueError("Concept targets must be finite within [0,1]")
    if c.shape[0] < 20 or np.unique(y).size != 2 or logistic_c <= 0 or minimum_width <= 0:
        raise ValueError(">=20 patients, two classes and positive regularization parameters required")
    weights = fixed_temporal_weights(c.shape[1], recency_decay)
    summary = np.einsum("ntk,t->nk", c, weights)
    center = np.median(summary, axis=0)
    width = np.maximum(np.quantile(summary, .9, axis=0) - np.quantile(summary, .1, axis=0), minimum_width)
    evidence = anchor_evidence(summary, center, width)
    clf = LogisticRegression(C=float(logistic_c), solver="liblinear", max_iter=2000)
    clf.fit(evidence, y)
    return SemanticAnchor(center, width, clf.coef_.reshape(-1).copy(),
                          float(clf.intercept_[0]), weights)


class AnchoredConceptFAN(AdditiveConceptModel):
    """Same neural concept encoder as ConceptFAN; identifiable frozen concept head."""

    def __init__(self, config: dict, input_dim: int, anchor: SemanticAnchor):
        super().__init__(config, input_dim, fuzzy=False)
        k = int(config["model"]["n_concepts"])
        steps = int(config["model"]["sequence_length"])
        if (anchor.center.shape != (k,) or anchor.width.shape != (k,) or
                anchor.beta.shape != (k,) or anchor.time_weights.shape != (steps,)):
            raise ValueError("Anchor and architecture dimensions do not match")
        for name, value in (
            ("anchor_center", anchor.center), ("anchor_width", anchor.width),
            ("anchor_beta", anchor.beta), ("anchor_time", anchor.time_weights),
            ("anchor_bias", np.asarray(anchor.bias)),
        ):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32).clone())
        # Prevent unused learned attention/head/fuzzy parameters from obscuring
        # the actual model capacity and from being updated by the optimizer.
        self.temporal_aggregator.requires_grad_(False)
        self.fuzzy_evidence.requires_grad_(False)
        self.decision_weight.requires_grad_(False)
        self.bias.requires_grad_(False)

    def anchored_evidence(self, trajectories: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if trajectories.ndim != 3 or trajectories.shape[1] != len(self.anchor_time):
            raise ValueError("Expected [batch,steps,concepts] trajectories")
        summary = (trajectories * self.anchor_time.view(1, -1, 1)).sum(dim=1)
        evidence = torch.sigmoid((summary - self.anchor_center) / self.anchor_width) - 0.5
        return summary, evidence

    def forward(self, x: torch.Tensor, concept_mask: torch.Tensor | None = None) -> RealModelOutput:
        latent = self.encoder(x)
        trajectories = self.projector(latent)
        summaries, evidence = self.anchored_evidence(trajectories)
        if concept_mask is not None:
            if concept_mask.shape != evidence.shape:
                raise ValueError("concept_mask must match [batch,concepts]")
            evidence = evidence * concept_mask.to(dtype=evidence.dtype)
        contributions = evidence * self.anchor_beta
        logit = self.anchor_bias + contributions.sum(dim=-1)
        return RealModelOutput(
            logit=logit,
            probability=torch.sigmoid(logit),
            concept_trajectories=trajectories,
            concept_summaries=summaries,
            contributions=contributions,
            bias=self.anchor_bias,
            latent_sequence=latent,
            evidence=evidence,
        )


def semantic_concept_regularizers(
    output: RealModelOutput,
    targets: torch.Tensor,
    mask: torch.Tensor,
    anchor: AnchoredConceptFAN,
) -> dict[str, torch.Tensor]:
    """Masked temporal fidelity, oracle-evidence alignment, cross-axis residual covariance.

    Penalizes *errors* correlated with other target concepts, never the
    naturally correlated physiological concept values themselves.
    """
    p = output.concept_trajectories
    if p is None or p.shape != targets.shape or p.shape != mask.shape:
        raise ValueError("Expected matching [batch,steps,concepts] concept trajectories")
    m = mask.to(dtype=p.dtype)
    if torch.any((m < 0) | (m > 1)):
        raise ValueError("Observation mask must be in [0,1]")
    valid = (m[:, 1:] * m[:, :-1])
    differences = (p[:, 1:] - p[:, :-1]) - (targets[:, 1:] - targets[:, :-1])
    temporal = ((differences.square()) * valid).sum() / valid.sum().clamp_min(1.)
    oracle_summaries = (targets * anchor.anchor_time.view(1, -1, 1)).sum(dim=1)
    oracle_evidence = torch.sigmoid(
        (oracle_summaries - anchor.anchor_center) / anchor.anchor_width
    ) - 0.5
    target_contrib = oracle_evidence * anchor.anchor_beta
    evidence_loss = F.smooth_l1_loss(output.contributions, target_contrib)

    residual = (p - targets) * m
    k = p.shape[-1]
    # Error in concept k should not become a proxy for a different target j.
    # Weighted covariance requires overlap of observed concepts.
    residual_flat = residual.reshape(-1, k)
    target_flat = targets.reshape(-1, k)
    mask_flat = m.reshape(-1, k)
    cross = p.new_zeros(())
    comparisons = 0
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            common = mask_flat[:, i] * mask_flat[:, j]
            size = common.sum()
            if size < 3:
                continue
            er = residual_flat[:, i]
            tj = target_flat[:, j]
            em = (common * er).sum() / size
            tm = (common * tj).sum() / size
            cov = (common * (er - em) * (tj - tm)).sum() / size
            cross = cross + cov.square()
            comparisons += 1
    if comparisons:
        cross = cross / comparisons
    return {"temporal": temporal, "oracle_evidence": evidence_loss, "cross_residual": cross}
