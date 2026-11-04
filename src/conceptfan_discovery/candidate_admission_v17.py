"""V17: propose an *observable* new dynamic concept from TRAIN residuals.

No target labels, synthetic hidden factors, physicians' names or post-hoc model
explanations enter this candidate search. A proposal is NOT certification of a
physical latent cause: persistent sensor noise can produce an indistinguishable
candidate. The admit operation preserves *all* old decision/forecast paths
until explicit re-training gives the extra native concept nonzero coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import numpy as np
import torch

from .multiscale_physiology_v10 import (
    MultiScaleDualProcessConceptFAN,
    FeedbackAdaptiveMultiScaleUnknownNeutralConceptFAN,
)


@dataclass(frozen=True)
class ProposedObservableConcept:
    candidate_id: str
    direction: np.ndarray
    train_episode_count: int
    positive_lag_strength: float
    permutation_p: float
    bootstrap_abs_cosine_q10: float
    top_signed_sensors: tuple[tuple[str, float], ...]
    physical_cause_identified: bool = False
    expert_approved: bool = False


def _symmetric_lag(episodes: np.ndarray) -> np.ndarray:
    n, steps, d = episodes.shape
    before = episodes[:, :-1, :].reshape(-1, d)
    after = episodes[:, 1:, :].reshape(-1, d)
    before = before - before.mean(0)
    after = after - after.mean(0)
    return (after.T @ before + before.T @ after) / (2 * (len(before) - 1))


def propose_observable_residual_concept(
    train_values: np.ndarray,
    train_mask: np.ndarray,
    current_basis: np.ndarray,
    channel_names: tuple[str, ...],
    *,
    seed: int = 20301104,
    permutations: int = 199,
    bootstraps: int = 32,
    alpha: float = .01,
    min_bootstrap_abs_cosine: float = .86,
) -> ProposedObservableConcept:
    """Train-only residual lag; 1+ positives are explicitly only candidate evidence.

    Require full measurements in this initial prototype. Missingness can
    create apparent lagged structure, so irregular ICU histories ABSTAIN
    instead of inferring physiology. Negative null breaks patient-pair
    temporal alignment, unlike misleading within-patient time shuffles.
    """
    x = np.asarray(train_values, dtype=np.float64)
    mask = np.asarray(train_mask, dtype=np.float64)
    basis = np.asarray(current_basis, dtype=np.float64)
    if (x.ndim != 3 or mask.shape != x.shape
        or x.shape[0] < 160 or x.shape[1] < 8
        or x.shape[-1] < 3
        or not np.isfinite(x).all() or not np.isfinite(mask).all()):
        raise ValueError("ABSTAIN: insufficient complete TRAIN episodes")
    n, _, d = x.shape
    if not np.array_equal(mask, np.ones_like(mask)):
        raise ValueError("ABSTAIN: measurement gaps need a separate valid null")
    if (basis.ndim != 2 or basis.shape[0] != d
        or not 1 <= basis.shape[1] < d
        or len(channel_names) != d
        or len(set(channel_names)) != d
        or not np.isfinite(basis).all()
        or not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-4)):
        raise ValueError("Invalid independently selected observable basis")
    if not (permutations >= 99 and bootstraps >= 24
            and 0 < alpha <= .05 and 0 < min_bootstrap_abs_cosine < 1):
        raise ValueError("Invalid predeclared tests")

    projector = np.eye(d) - basis @ basis.T
    residual = x @ projector
    # TRAIN mean only: no peeking at new subjects or hidden state.
    residual = residual - residual.mean(axis=(0,1), keepdims=True)
    lag = _symmetric_lag(residual)
    eigvals, eigvecs = np.linalg.eigh(lag)
    strength = float(eigvals[-1])
    candidate = eigvecs[:, -1]
    candidate = projector @ candidate
    norm = float(np.linalg.norm(candidate))
    if norm < 1e-10 or strength <= 0:
        raise ValueError("ABSTAIN: no positive persistent residual direction")
    candidate = candidate / norm

    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    for j in range(permutations):
        # Re-pair all next-time trajectories with *other* patients, preserving
        # their within-time statistics while destroying cross-time source links.
        perm = rng.permutation(n)
        before = residual[:, :-1].reshape(-1, d)
        after = residual[perm, 1:].reshape(-1, d)
        before = before - before.mean(0)
        after = after - after.mean(0)
        cross = (after.T @ before + before.T @ after) / (2*(len(before)-1))
        null[j] = float(np.linalg.eigvalsh(cross)[-1])
    p = (1 + int(np.count_nonzero(null >= strength))) / (1+permutations)
    if p > alpha:
        raise ValueError("ABSTAIN: residual temporal signal not distinguishable from null")

    abs_cos = []
    for j in range(bootstraps):
        sampled = residual[rng.integers(0, n, size=n)]
        eigenvalues, eigenvectors = np.linalg.eigh(_symmetric_lag(sampled))
        test = projector @ eigenvectors[:, -1]
        test = test / max(np.linalg.norm(test),1e-12)
        abs_cos.append(float(abs(test @ candidate)))
    q10 = float(np.quantile(abs_cos,.1))
    if q10 < min_bootstrap_abs_cosine:
        raise ValueError("ABSTAIN: candidate direction not reproducible on TRAIN")
    pivot = int(np.argmax(np.abs(candidate)))
    candidate = candidate * np.sign(candidate[pivot])
    biggest = sorted(range(d),key=lambda i:abs(candidate[i]),reverse=True)[:3]
    manifest = tuple((channel_names[i],float(candidate[i])) for i in biggest)
    digest = sha256(np.round(candidate,decimals=8).tobytes()).hexdigest()[:16]
    return ProposedObservableConcept(
        candidate_id=f"observable-residual-{digest}",
        direction=candidate.copy(),
        train_episode_count=n,
        positive_lag_strength=strength,
        permutation_p=float(p),
        bootstrap_abs_cosine_q10=q10,
        top_signed_sensors=manifest,
    )


def admit_candidate_without_prediction_drift(
    trained_model: MultiScaleDualProcessConceptFAN,
    candidate: ProposedObservableConcept,
    *,
    expert_decision: str,
) -> MultiScaleDualProcessConceptFAN:
    """Expand native V10 concept capacity: acceptance does NOT assert semantics.

    The newly inserted concept has exactly zero decision, forecast and
    reconstruction coefficients. No old response path changes until an
    explicit subsequent training phase optimizes these newly unlocked terms.
    """
    if expert_decision not in ("approve", "reject", "unsure"):
        raise ValueError("Expert candidate-level response unknown")
    if expert_decision != "approve":
        raise ValueError("ABSTAIN: candidate not approved for experimental insertion")
    if not isinstance(trained_model,MultiScaleDualProcessConceptFAN):
        raise ValueError("Expected a native V10 dual-process concept model")
    old = trained_model.physical
    original_basis = old.source_predictive_basis.detach().cpu().numpy()
    new_direction = np.asarray(candidate.direction,dtype=np.float64)
    d,k = original_basis.shape
    if (new_direction.shape != (d,)
        or not np.isfinite(new_direction).all()
        or abs(np.linalg.norm(new_direction)-1) > 1e-4
        or np.max(np.abs(original_basis.T @ new_direction)) > 1e-4
        or k >= d):
        raise ValueError("ABSTAIN: new direction duplicates or corrupts existing basis")
    new_basis = np.column_stack([original_basis, new_direction])
    adaptive = isinstance(old,FeedbackAdaptiveMultiScaleUnknownNeutralConceptFAN)
    expanded = MultiScaleDualProcessConceptFAN(
        observable_loadings=new_basis,
        channels=old.channels,
        forecast_horizon=old.horizon,
        n_observation_concepts=trained_model.process.n_concepts,
        scales=old.scales,
        adaptive_feature_gradients=adaptive,
    )
    # Same process head and all native sensor/threshold/readout parameter
    # values, NOT a frozen copy detached from prediction.
    expanded.process.load_state_dict(trained_model.process.state_dict(),strict=True)
    a,b=old,expanded.physical
    with torch.no_grad():
        for attr in (
            "feature_logits","threshold_logits","width_logits",
            "scale_logits","decision_raw_weight","transition_bias",
        ):
            getattr(b,attr)[:k].copy_(getattr(a,attr)[:k])
        b.decision_raw_weight[k].zero_()
        b.decision_bias.copy_(a.decision_bias)
        b.transition_raw[:k,:k].copy_(a.transition_raw)
        b.transition_raw[:k,k].zero_()
        b.transition_raw[k,:k].zero_()
        b.future_decoder.weight[:,:k].copy_(a.future_decoder.weight)
        b.future_decoder.weight[:,k].zero_()
        b.future_decoder.bias.copy_(a.future_decoder.bias)
        b.observation_decoder.weight[:,:k].copy_(a.observation_decoder.weight)
        b.observation_decoder.weight[:,k].zero_()
        b.observation_decoder.bias.copy_(a.observation_decoder.bias)
        if adaptive:
            b.expert_exploration_fraction.copy_(a.expert_exploration_fraction)
        # Retain existing V18 signed policy when adding another native slot.
        b.signed_risk_policy[:k].copy_(a.signed_risk_policy[:k])
    expanded.train(trained_model.training)
    return expanded


def candidate_passport(candidate: ProposedObservableConcept) -> dict:
    return {
        "candidate_id":candidate.candidate_id,
        "training_n":candidate.train_episode_count,
        "positive_lag_strength":candidate.positive_lag_strength,
        "permutation_p_train":candidate.permutation_p,
        "bootstrap_abs_cosine_q10_train":candidate.bootstrap_abs_cosine_q10,
        "top_signed_observed_sensor_loadings":list(candidate.top_signed_sensors),
        "approved_by_real_clinician":False,
        "physical_cause_identified":False,
        "concept_is_part_of_native_predictive_graph_only_after_admission":True,
    }
