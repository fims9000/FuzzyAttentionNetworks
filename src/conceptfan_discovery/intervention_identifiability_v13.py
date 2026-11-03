"""V13 research: identifiable *observable intervention response* directions.

Strong scoped assumptions: randomized environments, positive known intervention
labels, stable additive linear sensor mixing, one independently targeted latent
factor per environment, NO downstream propagation, and invariant mean noise.
The experimental provenance must independently verify target exclusivity.
Without that provenance, labels name experimental environments, NOT latent
causal variables. The estimator cannot infer target exclusivity from data.

This is a new train-time concept model with a strict additive logit; it is
not a post-hoc explanation of ConceptFAN V11/V12 or a clinical claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class InterventionCertificate:
    response: np.ndarray  # [observed_channels, noncontrol_environment_count]
    unit_directions: np.ndarray
    intervention_ids: tuple[int, ...]
    bootstrap_q10_direction_cosine: np.ndarray
    min_singular_value: float
    q90_bootstrap_spectral_noise: float
    condition_number: float
    rank_estimate: int
    status: str
    target_exclusivity_verified_externally: bool
    receipt: dict

    def to_json(self) -> dict:
        return {
            "status": self.status,
            "intervention_ids": list(self.intervention_ids),
            "experimental_response_columns": self.response.tolist(),
            "normalized_response_directions": self.unit_directions.tolist(),
            "bootstrap_q10_direction_cosine": self.bootstrap_q10_direction_cosine.tolist(),
            "min_singular_value": self.min_singular_value,
            "bootstrap_q90_spectral_noise": self.q90_bootstrap_spectral_noise,
            "condition_number": self.condition_number,
            "empirical_rank": self.rank_estimate,
            "target_exclusivity_verified_externally": self.target_exclusivity_verified_externally,
            "explanation_scope": (
                "single latent factor observable response, up to intervention scale"
                if self.status == "CERTIFIED_CONDITIONAL_SINGLE_TARGET_RESPONSE"
                else "environment contrast span only; individual factor identity ABSTAINS"
            ),
            **self.receipt,
        }


def _observed_environment_means(
    x: np.ndarray, mask: np.ndarray, environments: np.ndarray,
    environment_ids: tuple[int, ...], *, window: int,
) -> np.ndarray:
    """Observed cell-weighted means only, with per-channel availability."""
    means = []
    for ident in environment_ids:
        selected = environments == ident
        xx = x[selected, -window:, :]
        mm = mask[selected, -window:, :]
        counts = mm.sum(axis=(0, 1))
        if np.any(counts < 1):
            raise ValueError("One intervention environment has an unobserved channel")
        means.append((xx * mm).sum(axis=(0, 1)) / counts)
    return np.stack(means, axis=0)


def estimate_intervention_certificate(
    train_values: np.ndarray,
    train_observed_mask: np.ndarray,
    train_environment: np.ndarray,
    *,
    exclusive_single_target_provenance: bool,
    bootstraps: int = 80,
    seed: int = 20261104,
    history_window: int = 6,
    minimum_patients_per_environment: int = 40,
    min_direction_cosine_q10: float = .8,
    min_singular_to_bootstrap_noise: float = 2.0,
    max_condition: float = 30.,
) -> InterventionCertificate:
    x = np.asarray(train_values, dtype=np.float64)
    m = np.asarray(train_observed_mask, dtype=np.float64)
    env = np.asarray(train_environment)
    if (x.ndim != 3 or x.shape != m.shape or env.shape != (len(x),)
            or not 1 <= history_window <= x.shape[1]
            or not np.isfinite(x).all() or not np.isfinite(m).all()
            or not np.isin(m, [0., 1.]).all()):
        raise ValueError("Observed finite TRAIN [N,T,D] and environment per record required")
    if (not isinstance(exclusive_single_target_provenance, (bool, np.bool_))
            or bootstraps < 16 or minimum_patients_per_environment < 10
            or not 0 < min_direction_cosine_q10 < 1
            or min_singular_to_bootstrap_noise <= 0 or max_condition <= 1):
        raise ValueError("Invalid experimental provenance or certificate controls")
    ids = tuple(sorted(int(i) for i in np.unique(env)))
    if tuple(np.unique(env).tolist()) != ids or len(ids) < 3 or ids[0] != 0:
        raise ValueError("Control=0 and at least two labelled intervention environments required")
    if len(ids)-1 > x.shape[-1] or ids != tuple(range(len(ids))):
        raise ValueError("Intervention labels must be dense 0..J and J<=D")
    sizes = [int(np.count_nonzero(env == i)) for i in ids]
    if min(sizes) < minimum_patients_per_environment:
        raise ValueError("Insufficient independent randomized patients per environment")
    means = _observed_environment_means(x, m, env, ids, window=history_window)
    contrast = (means[1:] - means[0]).T
    norms = np.linalg.norm(contrast, axis=0)
    # Keep finite zero columns, but explicitly abstain; never normalize with NaN.
    directions = contrast / np.maximum(norms[None, :], 1e-12)
    singular_values = np.linalg.svd(contrast, compute_uv=False)
    rank = int(np.linalg.matrix_rank(contrast, tol=1e-9))
    rng = np.random.default_rng(seed)
    per_env = [np.flatnonzero(env == i) for i in ids]
    angles = np.zeros((bootstraps, len(ids)-1), dtype=np.float64)
    perturbation = np.zeros(bootstraps, dtype=np.float64)
    for b in range(bootstraps):
        sample = np.concatenate([rng.choice(rows, len(rows), replace=True) for rows in per_env])
        sample_means = _observed_environment_means(
            x[sample], m[sample], env[sample], ids, window=history_window,
        )
        delta = (sample_means[1:] - sample_means[0]).T
        angles[b] = np.sum(directions * (delta / np.maximum(
            np.linalg.norm(delta, axis=0)[None, :], 1e-12
        )), axis=0)
        perturbation[b] = np.linalg.norm(delta - contrast, ord=2)
    q10 = np.quantile(angles, .1, axis=0)
    noise = float(np.quantile(perturbation, .9))
    sigma_min = float(singular_values[-1])
    condition = float(singular_values[0] / max(sigma_min, 1e-12))
    if min(norms) <= 1e-9 or rank < len(ids)-1:
        empirical_status = "ABSTAIN_RESPONSE_RANK_DEFICIENT"
    elif (sigma_min <= min_singular_to_bootstrap_noise * noise
          or condition > max_condition or min(q10) < min_direction_cosine_q10):
        empirical_status = "ABSTAIN_UNRESOLVED_INTERVENTION_RESPONSES"
    else:
        empirical_status = "RESOLVED_OBSERVABLE_ENVIRONMENT_RESPONSES"
    status = (
        "ABSTAIN_UNVERIFIED_SINGLE_TARGET_PROVENANCE"
        if not bool(exclusive_single_target_provenance)
        else ("CERTIFIED_CONDITIONAL_SINGLE_TARGET_RESPONSE"
              if empirical_status == "RESOLVED_OBSERVABLE_ENVIRONMENT_RESPONSES"
              else empirical_status)
    )
    receipt = {
        "train_patients_per_environment": dict(zip(map(str, ids), sizes)),
        "empirical_response_resolution_independent_of_provenance": empirical_status,
        "n_model_slots_is_number_of_targeted_intervention_labels": len(ids)-1,
        "history_window": history_window,
        "patient_bootstraps": bootstraps,
        "seed": seed,
        "minimum_direction_cosine_q10": min_direction_cosine_q10,
        "min_singular_to_bootstrap_noise": min_singular_to_bootstrap_noise,
        "max_condition": max_condition,
        "source_used": "TRAIN observed histories, observation masks and randomized environment labels only",
        "no_outcome_or_latent_labels_in_contrast_estimation": True,
        "external_exclusivity_is_an_assumption_NOT_testable_from_contrasts": True,
        "train_input_and_environment_sha256": sha256(
            np.ascontiguousarray(x, dtype="<f4").tobytes()
            + np.ascontiguousarray(m, dtype="<f4").tobytes()
            + np.ascontiguousarray(env, dtype="<i4").tobytes()
        ).hexdigest(),
    }
    return InterventionCertificate(
        response=contrast, unit_directions=directions, intervention_ids=ids[1:],
        bootstrap_q10_direction_cosine=q10, min_singular_value=sigma_min,
        q90_bootstrap_spectral_noise=noise, condition_number=condition,
        rank_estimate=rank, status=status,
        target_exclusivity_verified_externally=bool(exclusive_single_target_provenance),
        receipt=receipt,
    )


def past_observed_mean_torch(
    values: torch.Tensor, mask: torch.Tensor, *, window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (values.ndim != 3 or mask.shape != values.shape
            or not 1 <= window <= values.shape[1]
            or not torch.isfinite(values).all() or not torch.isfinite(mask).all()
            or torch.any(mask < 0) or torch.any(mask > 1)):
        raise ValueError("Finite observed past and binary mask required")
    take = mask[:, -window:, :]
    counts = take.sum(dim=1)
    return (values[:, -window:, :] * take).sum(dim=1) / counts.clamp_min(1.), (counts > 0).to(values.dtype)


class InterventionAnchoredConceptModel(nn.Module):
    """Native TRAIN-time concept definitions with exact additive risk logit.

    Weights W are trainable, but regularized by W * D = I, where each D_j is
    an independently-verified randomized intervention response. The learned
    concept is a bounded, visible past-sensor operator; not a post-hoc head.
    """
    def __init__(self, certificate: InterventionCertificate, *,
                 history_window: int = 6, anchor_weight: float = 1.0,
                 definition_weight: float = .1):
        super().__init__()
        if certificate.status != "CERTIFIED_CONDITIONAL_SINGLE_TARGET_RESPONSE":
            raise ValueError("Only verified, resolved single-target effects may define individual concepts")
        d, k = certificate.response.shape
        if k < 2 or d < k or not 0 < anchor_weight or definition_weight < 0:
            raise ValueError("Invalid anchored concept rank/weights")
        self.n_channels, self.n_concepts = d, k
        self.window = int(history_window)
        self.anchor_weight = float(anchor_weight)
        self.definition_weight = float(definition_weight)
        response = torch.tensor(certificate.response.copy(), dtype=torch.float32)
        self.register_buffer("response", response)
        initial = torch.linalg.pinv(response)
        self.register_buffer("original_demixer", initial.clone())
        self.definition = nn.Parameter(initial.clone())
        self.threshold = nn.Parameter(torch.zeros(k))
        self.log_width = nn.Parameter(torch.zeros(k))
        self.risk_weight = nn.Parameter(torch.randn(k) * .04)
        self.risk_bias = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.shape[-1] != self.n_channels:
            raise ValueError("Sensor dimension does not match experimental certificate")
        avg, present = past_observed_mean_torch(x, mask, window=self.window)
        # The exact, auditable operation uses available measured channels only.
        raw = avg @ self.definition.T
        reliability = present @ self.definition.abs().T / self.definition.abs().sum(dim=-1).clamp_min(1e-10)
        concepts = torch.tanh((raw - self.threshold) / F.softplus(self.log_width).clamp_min(.1)) * reliability
        signed = concepts * self.risk_weight.view(1, -1)
        logit = self.risk_bias + signed.sum(dim=-1)
        return {"logits": logit, "concepts": concepts, "contributions": signed,
                "reliability": reliability, "observed_sensor_mean": avg}

    def regularizers(self) -> dict[str, torch.Tensor]:
        anchor = ((self.definition @ self.response
                   - torch.eye(self.n_concepts, device=self.definition.device)) ** 2).mean()
        drift = (self.definition - self.original_demixer).square().mean()
        return {"intervention_jacobian": anchor, "definition_drift": drift}

    def observable_response_equivariance(
        self, values: torch.Tensor, mask: torch.Tensor, *,
        intervention_slot: int, minimum_response: float = .05,
    ) -> torch.Tensor:
        """Training-only probe through native concept forward, not latent do().

        D_j is an OBSERVED train-environment contrast; add it to still-visible
        last-history measurements. Suppress non-target concept changes and
        require a positive target response. This is a *synthetic sensor probe*,
        NOT an identifiable patient causal counterfactual.
        """
        if not 0 <= intervention_slot < self.n_concepts or minimum_response < 0:
            raise ValueError("Invalid train-only observable probe")
        base = self(values,mask)["concepts"]
        shift = self.response[:,intervention_slot]
        altered = values.clone()
        altered[:,-self.window:,:] = (
            altered[:,-self.window:,:]
            + mask[:,-self.window:,:]*shift.view(1,1,-1)
        )
        changed = self(altered,mask)["concepts"] - base
        other = torch.ones(self.n_concepts,device=changed.device,dtype=changed.dtype)
        other[intervention_slot] = 0
        off_target = ((changed*other.view(1,-1))**2).sum(dim=-1).mean()/max(self.n_concepts-1,1)
        under_response = (minimum_response-changed[:,intervention_slot]).clamp_min(0).square().mean()
        return off_target + under_response

    def loss(self, output: dict[str, torch.Tensor], labels: torch.Tensor,
             *, pos_weight: torch.Tensor | None = None) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            output["logits"], labels, pos_weight=pos_weight,
        )
        reg = self.regularizers()
        return bce + self.anchor_weight * reg["intervention_jacobian"] + self.definition_weight * reg["definition_drift"]


def train_intervention_concept_model(
    x: np.ndarray, mask: np.ndarray, labels: np.ndarray,
    certificate: InterventionCertificate, *, seed: int,
    epochs: int = 45, batch_size: int = 128, learning_rate: float = .008,
    anchor_weight: float = 2., definition_weight: float = .2,
    observable_response_weight: float = .8,
) -> tuple[InterventionAnchoredConceptModel, dict]:
    x = np.asarray(x, dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    if x.ndim != 3 or x.shape != m.shape or y.shape != (len(x),) or not np.isin(y,[0.,1.]).all():
        raise ValueError("Native training requires valid TRAIN observations and binary labels")
    if observable_response_weight < 0:
        raise ValueError("Response regularizer strength cannot be negative")
    if certificate.receipt["no_outcome_or_latent_labels_in_contrast_estimation"] is not True:
        raise ValueError("Non training-only response certificate")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = InterventionAnchoredConceptModel(
        certificate, anchor_weight=anchor_weight, definition_weight=definition_weight,
    )
    n_pos = float(y.sum())
    if not 0 < n_pos < len(y):
        raise ValueError("Need positive and negative outcomes in TRAIN")
    weight = torch.tensor((len(y) - n_pos) / n_pos, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.0001)
    g = torch.Generator().manual_seed(seed + 3311)
    xt, mt, yt = torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(y)
    model.train()
    slot_generator = torch.Generator().manual_seed(seed + 7721)
    for epoch in range(epochs):
        for row in torch.randperm(len(x),generator=g).split(batch_size):
            output = model(xt[row], mt[row])
            error = model.loss(output, yt[row],pos_weight=weight)
            if observable_response_weight > 0:
                slot = int(torch.randint(model.n_concepts,(1,),generator=slot_generator))
                error = error + observable_response_weight*model.observable_response_equivariance(
                    xt[row],mt[row],intervention_slot=slot
                )
            if not torch.isfinite(error) or torch.max(torch.abs(
                output["logits"] - model.risk_bias - output["contributions"].sum(dim=-1)
            )) > 1e-5:
                raise FloatingPointError("Nonfinite or nonadditive native concept model")
            optimizer.zero_grad(set_to_none=True)
            error.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        error = model.loss(model(xt,mt),yt,pos_weight=weight)
        jacobian = model.definition @ model.response
        learned_movement = torch.linalg.norm(model.definition-model.original_demixer).item()
    return model, {
        "train_loss":float(error),"definition_movement_from_intervention_initializer":learned_movement,
        "train_jacobian_deviation_fro":float(torch.linalg.norm(jacobian-torch.eye(model.n_concepts))),
        "train_intervention_labels_used_for_source_only":True,
        "train_binary_outcome_optimizes_native_concept_definitions":True,
        "n_train_patients":len(y),"train_steps":epochs * int(np.ceil(len(y)/batch_size)),
        "initialization_seed":seed,
        "train_only_observable_response_equivariance_weight":observable_response_weight,
        "no_holdout_or_latent_truth_used":True,
    }
