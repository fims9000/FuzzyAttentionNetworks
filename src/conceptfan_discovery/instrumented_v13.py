"""V13, conditional identification by logged randomized perturbations.

This is a *new synthetic-only research protocol*. It must not be interpreted
as observational ICU factor identification: the latter does not contain
randomized, independently targeted interventions with known signatures.

Assume Z[t+1] = f(Z[t]) + diag(g) U[t] + eps[t+1],
       X[t]   = B Z[t] + eta[t],
where U[t] is randomized independently of Z[t], eps and eta, with full-rank
Cov(U), B has full column rank, g[k] > 0, and the action-to-factor mapping is
known. Then Cov(X[t+1], U[t]) Cov(U[t])^-1 = B diag(g).
Only column *directions and orientations*, not absolute latent scale, are
identified if g is unknown. If randomized interventions are not available,
no individual source identity is certified by this method.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class InterventionProtocol:
    """Externally warranted assumptions, not inferred from correlation tests."""
    randomized_assignment: bool
    individually_targeted_actions: bool
    known_positive_action_effects: bool
    observation_complete: bool
    action_names: tuple[str, ...]
    time_alignment: str = "action_t_precedes_measurement_t_plus_1"


@dataclass(frozen=True)
class IdentificationCertificate:
    signature: np.ndarray
    dual: np.ndarray
    response_magnitudes: np.ndarray
    worst_bootstrap_oriented_cosine_q10: np.ndarray
    pretreatment_balance_permutation_p: float
    response_condition: float
    n_independent_episodes: int
    n_action_channels: int
    interpretation: str

    def to_json(self) -> dict[str, Any]:
        return {
            "status": "CONDITIONAL_ACTION_SIGNATURE_IDENTIFIED",
            "n_independent_episodes": self.n_independent_episodes,
            "n_action_channels": self.n_action_channels,
            "pretreatment_balance_permutation_p": self.pretreatment_balance_permutation_p,
            "response_condition": self.response_condition,
            "per_action_bootstrap_cosine_q10": self.worst_bootstrap_oriented_cosine_q10.tolist(),
            "response_magnitude_per_action": self.response_magnitudes.tolist(),
            "sensor_response_signature": self.signature.tolist(),
            "dual_observation_operator": self.dual.tolist(),
            "provenance_required": "documented randomized separately targeted logged actions",
            "interpretation": self.interpretation,
            "latent_scale_identified": False,
            "learned_human_physiology": False,
            "causal_risk_effects_identified": False,
        }


def _cross_covariance(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Cov(x,u), rows are sensor channels, columns are intervention slots."""
    x = np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1])
    u = np.asarray(u, dtype=np.float64).reshape(-1, u.shape[-1])
    if x.shape[0] != u.shape[0] or x.shape[0] < 20:
        raise ValueError("Need matching sensor/action observations")
    xc, uc = x - x.mean(axis=0), u - u.mean(axis=0)
    return xc.T @ uc / len(xc)


def _response_operator(pre: np.ndarray, post: np.ndarray, actions: np.ndarray):
    flat_actions = actions.reshape(-1, actions.shape[-1])
    cov_uu = _cross_covariance(flat_actions, flat_actions)
    eig = np.linalg.eigvalsh(cov_uu)
    if eig.min() < 1e-6 or eig.max() / eig.min() > 100.0:
        raise ValueError("ABSTAIN: intervention assignment Gram matrix rank-deficient")
    post_cov = _cross_covariance(post, actions)
    return np.linalg.solve(cov_uu, post_cov.T).T


def identify_action_anchored_concepts(
    observed_history: np.ndarray,
    action_history: np.ndarray,
    protocol: InterventionProtocol,
    *,
    seed: int = 20261104,
    permutations: int = 199,
    bootstraps: int = 64,
    min_response_singular_ratio: float = .08,
    min_oriented_bootstrap_cosine_q10: float = .8,
    min_pretreatment_balance_p: float = .02,
) -> IdentificationCertificate:
    """Read TRAIN sensor histories and logs ONLY; no outcomes or source oracle.

    A permutation test can reject exogeneity; failure to reject NEVER proves
    that treatment was randomized, which needs protocol provenance.
    Patient episodes are resampled/relabeled as independent units.
    """
    x = np.asarray(observed_history, dtype=np.float64)
    u = np.asarray(action_history, dtype=np.float64)
    if (not protocol.randomized_assignment or
            not protocol.individually_targeted_actions or
            not protocol.known_positive_action_effects or
            not protocol.observation_complete or
            protocol.time_alignment != "action_t_precedes_measurement_t_plus_1"):
        raise ValueError("ABSTAIN: randomized, targeted, positive-gain, complete-measurement protocol is required")
    if (x.ndim != 3 or u.ndim != 3 or x.shape[0] != u.shape[0] or
            x.shape[1] != u.shape[1] + 1 or x.shape[2] < u.shape[2] or
            len(protocol.action_names) != u.shape[2] or
            len(set(protocol.action_names)) != u.shape[2] or
            not np.isfinite(x).all() or not np.isfinite(u).all() or
            len(x) < 100 or u.shape[1] < 3 or
            permutations < 99 or bootstraps < 24):
        raise ValueError("ABSTAIN: invalid complete historical TRAIN arrays or insufficient episodes")
    pre, post = x[:, :-1, :], x[:, 1:, :]
    rng = np.random.default_rng(seed)
    observed_placebo = np.linalg.norm(_cross_covariance(pre, u), ord="fro")
    permuted = np.array([
        np.linalg.norm(_cross_covariance(pre, u[rng.permutation(len(u))]), ord="fro")
        for _ in range(permutations)
    ])
    p = (1 + int(np.count_nonzero(permuted >= observed_placebo))) / (permutations + 1)
    if p < min_pretreatment_balance_p:
        raise ValueError("ABSTAIN: pre-action sensor placebo rejects randomization/exogeneity")
    response = _response_operator(pre, post, u)
    singular = np.linalg.svd(response, compute_uv=False)
    if (singular[-1] <= 1e-8 or singular[-1] / singular[0] < min_response_singular_ratio):
        raise ValueError("ABSTAIN: not enough separately identified response directions")
    lengths = np.linalg.norm(response, axis=0)
    if np.any(lengths < 1e-7):
        raise ValueError("ABSTAIN: at least one intervention has no detectable response")
    normalized = response / lengths
    boot = np.empty((bootstraps, u.shape[2]), dtype=float)
    for b in range(bootstraps):
        idx = rng.integers(len(x), size=len(x))
        rb = _response_operator(pre[idx], post[idx], u[idx])
        boot[b] = ((rb / np.maximum(np.linalg.norm(rb, axis=0), 1e-12)) * normalized).sum(axis=0)
    q10 = np.quantile(boot, .1, axis=0)
    if np.min(q10) < min_oriented_bootstrap_cosine_q10:
        raise ValueError("ABSTAIN: unstable intervention response column orientation")
    return IdentificationCertificate(
        signature=normalized,
        dual=np.linalg.pinv(normalized),
        response_magnitudes=lengths,
        worst_bootstrap_oriented_cosine_q10=q10,
        pretreatment_balance_permutation_p=float(p),
        response_condition=float(singular[0] / singular[-1]),
        n_independent_episodes=len(x),
        n_action_channels=u.shape[2],
        interpretation="action-labelled measured response coordinates, up to positive per-factor scale",
    )


class InstrumentedConceptBottleneck(nn.Module):
    """Native train-time concept head with no sensor-to-logit shortcut.

    The action-identified dual measurement operator is a non-trainable buffer;
    thresholds and widths are trainable, concept evidence directly contributes
    to the task logit. The observable anchors are NOT post-hoc attributions.
    """
    def __init__(self, signature: np.ndarray, *, source_quantiles: np.ndarray | None = None):
        super().__init__()
        b = np.asarray(signature, dtype=np.float64)
        if (b.ndim != 2 or b.shape[1] < 2 or b.shape[0] < b.shape[1] or
                not np.isfinite(b).all() or np.linalg.matrix_rank(b) != b.shape[1]):
            raise ValueError("Identified full-column-rank source signature required")
        self.register_buffer("dual", torch.as_tensor(np.linalg.pinv(b), dtype=torch.float32))
        k = b.shape[1]
        if source_quantiles is None:
            thresholds = np.zeros(k, dtype=np.float32)
        else:
            thresholds = np.asarray(source_quantiles, dtype=np.float32)
            if thresholds.shape != (k,) or not np.isfinite(thresholds).all():
                raise ValueError("Quantiles must be train-only and one per action slot")
        self.threshold = nn.Parameter(torch.from_numpy(thresholds.copy()))
        self.raw_width = nn.Parameter(torch.full((k,), -0.7))
        self.raw_beta = nn.Parameter(torch.zeros(k))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        if history.ndim != 3 or history.shape[-1] != self.dual.shape[-1] or history.shape[1] < 2:
            raise ValueError("Expected complete [episodes,history,sensors] measurements")
        if not torch.isfinite(history).all():
            raise ValueError("Incompletely observed history cannot be treated as measured")
        anchors = history @ self.dual.T
        width = 0.12 + F.softplus(self.raw_width)
        trajectories = torch.tanh((anchors - self.threshold) / width)
        concept = trajectories[:, -min(4, history.shape[1]):].mean(dim=1)
        weights = 3 * torch.tanh(self.raw_beta)
        contributions = concept * weights
        logits = contributions.sum(dim=-1) + self.bias
        return {"logits": logits, "probabilities": torch.sigmoid(logits),
                "trajectories": trajectories, "concepts": concept,
                "contributions": contributions, "anchors": anchors}


def fit_instrumented_bottleneck(
    certificate: IdentificationCertificate,
    observations_train: np.ndarray,
    labels_train: np.ndarray,
    *, seed: int, epochs: int = 70, lr: float = .012,
) -> tuple[InstrumentedConceptBottleneck, dict[str, Any]]:
    """Train outcome ONLY through the identified concept path."""
    x = np.asarray(observations_train, dtype=np.float32)
    y = np.asarray(labels_train, dtype=np.float32)
    if (x.ndim != 3 or x.shape[0] != len(y) or x.shape[-1] != certificate.signature.shape[0]
            or not np.isin(y, [0., 1.]).all() or len(np.unique(y)) < 2 or epochs < 1):
        raise ValueError("Invalid TRAIN history and binary target for concept bottleneck")
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    anchors = x @ certificate.dual.T
    center = np.median(anchors, axis=(0, 1)).astype(np.float32)
    model = InstrumentedConceptBottleneck(certificate.signature, source_quantiles=center)
    with torch.no_grad():
        model.raw_beta.normal_(mean=0., std=.18)
        model.threshold.add_(torch.randn_like(model.threshold) * .06)
        model.raw_width.add_(torch.randn_like(model.raw_width) * .05)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0002)
    t_x, t_y = torch.from_numpy(x), torch.from_numpy(y)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        out = model(t_x)
        loss = F.binary_cross_entropy_with_logits(out["logits"], t_y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite native signed concept learning loss")
        loss.backward()
        opt.step()
    model.eval()
    return model, {
        "trained_only_on_source_history_and_outcome": True,
        "concept_count_from_independent_action_channels": certificate.n_action_channels,
        "train_samples": len(x), "epochs": epochs, "seed": int(seed),
        "task_head_has_no_bypass": True,
        "observational_physiological_identity_claim": False,
    }
