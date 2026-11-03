"""V12 train-time observable anchoring of signed concept effects.

Controls the ambiguity revealed by the FIRST completed Set B audit, without
claiming discovery of clinical mechanisms. Source concept count/gaps come from
the V11 train-only predictive-covariance certificate. Only certified singleton
source axes get an orientation; unresolved multi-axis groups have NO individual
labels or imposed signs. All label association selection uses TRAIN only.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import torch
import torch.nn.functional as F

from .identifiability_v11 import PredictiveCertificate
from .model import DiscoveryOutput
from .multiscale_physiology_v10 import MultiScaleDualProcessConceptFAN


@dataclass(frozen=True)
class SignedAnchorCertificate:
    basis: np.ndarray
    singleton: np.ndarray
    risk_sign: np.ndarray
    selection: tuple[dict, ...]
    receipt: dict

    def to_json(self) -> dict:
        return {
            **self.receipt,
            "canonical_oriented_source_basis": self.basis.tolist(),
            "structurally_certified_singletons": np.flatnonzero(self.singleton).tolist(),
            "train_label_supported_risk_sign": self.risk_sign.astype(int).tolist(),
            "per_axis_training_diagnostics": list(self.selection),
            "concepts_are_not_validated_disease_mechanisms": True,
            "no_post_hoc_explainer": True,
        }


def orient_only_resolved_singletons(
    certificate: PredictiveCertificate,
) -> tuple[np.ndarray, np.ndarray]:
    basis = np.asarray(certificate.reference_basis, dtype=np.float64)
    k = int(certificate.selected_computational_rank)
    if basis.ndim != 2 or basis.shape[1] < k or k < 1:
        raise ValueError("Nonempty source predictive basis required")
    basis = basis[:, :k].copy()
    singleton = np.zeros(k, dtype=bool)
    for group in certificate.groups:
        if not 0 <= group.first < group.stop <= k:
            raise ValueError("Source group exceeds computational slots")
        if group.status != "INDIVIDUAL_OBSERVABLE_AXIS_STABLE":
            continue
        if group.stop - group.first != 1:
            raise ValueError("Cannot orient unresolved multi-axis group")
        j = group.first
        pivot = int(np.argmax(np.abs(basis[:, j])))
        if abs(basis[pivot, j]) < 1e-8:
            raise ValueError("Zero resolved spectral direction")
        if basis[pivot, j] < 0:
            basis[:, j] *= -1
        singleton[j] = True
    if (not np.isfinite(basis).all()
            or not np.allclose(basis.T @ basis, np.eye(k), atol=2e-4)):
        raise ValueError("Broken source basis orthonormality")
    return basis, singleton


def numpy_anchor_targets(
    values: np.ndarray, mask: np.ndarray, basis: np.ndarray,
    *, history_steps: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounded past-observable proxy. No future, imputation, or concept labels."""
    x = np.asarray(values, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    b = np.asarray(basis, dtype=np.float64)
    if (x.ndim != 3 or x.shape != m.shape or b.ndim != 2
            or b.shape[0] != x.shape[2]
            or not 1 <= history_steps <= x.shape[1]
            or not np.isfinite(x).all() or not np.isfinite(m).all()
            or not np.isfinite(b).all() or not np.isin(m, [0., 1.]).all()):
        raise ValueError("Strict observed [N,T,D] and [D,K] train basis required")
    present = m[:, -history_steps:].sum(axis=1)
    avg = (x[:, -history_steps:] * m[:, -history_steps:]).sum(axis=1) / np.maximum(present, 1.)
    cover = ((present > 0).astype(np.float64) @ np.abs(b)
             / np.maximum(np.abs(b).sum(axis=0), 1e-10))
    return .5 * np.tanh(avg @ b) * cover, cover


def fit_train_only_signed_anchors(
    train_values: np.ndarray, train_mask: np.ndarray, train_labels: np.ndarray,
    certificate: PredictiveCertificate, *,
    bootstraps: int = 96, permutations: int = 199, seed: int = 20261104,
    familywise_alpha: float = .05, min_absolute_class_difference: float = .02,
    history_steps: int = 6,
) -> SignedAnchorCertificate:
    """Stratified patient bootstrap and label-permutation Bonferroni sign gate."""
    x = np.asarray(train_values, dtype=np.float64)
    m = np.asarray(train_mask, dtype=np.float64)
    y = np.asarray(train_labels, dtype=np.float64)
    if (x.ndim != 3 or m.shape != x.shape or y.shape != (len(x),)
            or len(x) < 80 or not np.isin(y, [0., 1.]).all()
            or min(int((y == 0).sum()), int((y == 1).sum())) < 16):
        raise ValueError("Need >=80 TRAIN patients, >=16 per outcome, valid masks")
    if bootstraps < 32 or permutations < 99 or not 0 < familywise_alpha < .2:
        raise ValueError("Invalid patient resampling / permutation protocol")
    basis, singleton = orient_only_resolved_singletons(certificate)
    target, coverage = numpy_anchor_targets(x, m, basis, history_steps=history_steps)
    rng = np.random.default_rng(seed)
    risk_sign = np.zeros(len(singleton), dtype=np.int8)
    evidence = []
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_eligible = max(int(singleton.sum()), 1)
    for j, valid in enumerate(singleton):
        row = {"slot": j, "source_singleton": bool(valid), "sign": 0,
               "train_class_difference": None,
               "stratified_patient_bootstrap_90pct": None,
               "train_label_permutation_two_sided_p": None,
               "reason": "ABSTAIN_STRUCTURAL"}
        if valid:
            z = target[:, j]
            observed = float(z[pos].mean() - z[neg].mean())
            boots = np.asarray([
                z[rng.choice(pos, len(pos), replace=True)].mean()
                - z[rng.choice(neg, len(neg), replace=True)].mean()
                for _ in range(bootstraps)
            ])
            interval = np.quantile(boots, [.05, .95])
            null = np.empty(permutations, dtype=np.float64)
            for trial in range(permutations):
                permuted = rng.permutation(y)
                null[trial] = (z[permuted == 1].mean()
                               - z[permuted == 0].mean())
            p = (1. + float(np.sum(np.abs(null) >= abs(observed)))) / (permutations + 1.)
            reliable = bool((interval[0] > 0 or interval[1] < 0)
                            and p <= familywise_alpha / n_eligible
                            and abs(observed) >= min_absolute_class_difference)
            risk_sign[j] = int(np.sign(observed)) if reliable else 0
            row.update({
                "sign": int(risk_sign[j]),
                "train_class_difference": observed,
                "stratified_patient_bootstrap_90pct": interval.tolist(),
                "train_label_permutation_two_sided_p": p,
                "mean_observed_source_coverage": float(coverage[:, j].mean()),
                "reason": ("TRAIN_ASSOCIATION_SIGN_CERTIFIED_NONCAUSAL" if reliable
                           else "ABSTAIN_WEAK_OR_UNCERTAIN_TRAIN_ASSOCIATION"),
            })
        evidence.append(row)
    digest = sha256(
        np.ascontiguousarray(x, dtype="<f4").tobytes()
        + np.ascontiguousarray(m, dtype="<f4").tobytes()
        + np.ascontiguousarray(y, dtype="<f4").tobytes()
    ).hexdigest()
    return SignedAnchorCertificate(
        basis=basis, singleton=singleton, risk_sign=risk_sign,
        selection=tuple(evidence),
        receipt={
            "source": "TRAIN_ONLY_OBSERVED_PAST_AND_TRAIN_OUTCOME",
            "n_train_patients": len(y),
            "n_source_singletons": int(singleton.sum()),
            "n_train_risk_signs": int(np.count_nonzero(risk_sign)),
            "number_of_labeled_future_holdouts_accessed": 0,
            "source_window_hours": history_steps,
            "strict_anchors_precede_first_model_fit": True,
            "train_input_and_label_sha256": digest,
            "bootstraps": bootstraps, "permutations": permutations,
            "patient_resampling_unit": "whole training episode",
            "multiple_testing": "Bonferroni across train-only certified singletons",
            "familywise_alpha": familywise_alpha,
            "min_absolute_train_effect": min_absolute_class_difference,
            "seed": seed, "not_population_coverage_or_causality": True,
        },
    )


def torch_anchor_targets(
    values: torch.Tensor, mask: torch.Tensor, basis: torch.Tensor,
    *, history_steps: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (values.ndim != 3 or mask.shape != values.shape or basis.ndim != 2
            or basis.shape[0] != values.shape[-1]
            or not 1 <= history_steps <= values.shape[1]):
        raise ValueError("Invalid observable past tensor or oriented basis")
    if (not torch.isfinite(values).all() or not torch.isfinite(mask).all()
            or torch.any(mask < 0) or torch.any(mask > 1)):
        raise ValueError("Nonfinite or out-of-range recorded observations")
    present = mask[:, -history_steps:].sum(dim=1)
    avg = (values * mask)[:, -history_steps:].sum(dim=1) / present.clamp_min(1.)
    cover = ((present > 0).to(values.dtype) @ basis.abs()
             / basis.abs().sum(dim=0).clamp_min(1e-10))
    return .5 * torch.tanh(avg @ basis) * cover, cover


def signed_effect_training_regularizers(
    model: MultiScaleDualProcessConceptFAN,
    first: DiscoveryOutput, second: DiscoveryOutput,
    observed_values: torch.Tensor, observed_mask: torch.Tensor,
    certificate: SignedAnchorCertificate, *, history_steps: int = 6,
    margin: float = .08,
) -> dict[str, torch.Tensor]:
    """Direct backprop through native concept and logit, never post-hoc."""
    a, b = first, second
    if (a.concept_summary.shape != b.concept_summary.shape
            or a.contributions.shape != b.contributions.shape
            or a.concept_summary.shape != a.contributions.shape
            or observed_values.shape != observed_mask.shape):
        raise ValueError("Matched native explanations and original observed history required")
    k = a.contributions.shape[-1]
    if (len(certificate.singleton) != k
            or certificate.basis.shape != (observed_values.shape[-1], k)
            or len(certificate.risk_sign) != k or margin <= 0):
        raise ValueError("Frozen TRAIN-only source certificate must match capacity")
    device, dtype = observed_values.device, observed_values.dtype
    basis = torch.as_tensor(certificate.basis, dtype=dtype, device=device)
    with torch.no_grad():
        target, coverage = torch_anchor_targets(
            observed_values, observed_mask, basis, history_steps=history_steps,
        )
    mask = torch.as_tensor(certificate.singleton, device=device, dtype=dtype).view(1, k)
    weight = mask * coverage
    denom = weight.sum().clamp_min(1.)
    centered_a = a.concept_summary - .5 * a.gates.view(1, k)
    centered_b = b.concept_summary - .5 * b.gates.view(1, k)
    anchor = (((F.smooth_l1_loss(centered_a, target, reduction="none")
                + F.smooth_l1_loss(centered_b, target, reduction="none"))
               * .5 * weight).sum() / denom)
    effect = ((F.smooth_l1_loss(a.contributions, b.contributions,
                               reduction="none") * weight).sum() / denom)
    sign = torch.as_tensor(certificate.risk_sign, device=device, dtype=dtype)
    trained = (sign != 0).to(dtype)
    beta = model.physical.decision_weights()
    sign_regularizer = (((margin - sign * beta).clamp_min(0.) ** 2)
                        * trained).sum() / trained.sum().clamp_min(1.)
    return {
        "observable_anchor": anchor,
        "signed_effect_view": effect,
        "train_association_risk_sign": sign_regularizer,
    }
