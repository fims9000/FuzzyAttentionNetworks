"""Transparent, non-post-hoc audit for automatically discovered concepts.

Matching between independent runs uses ONLY learned operator definitions, not
validation labels or the oracle's hidden state identities.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

import torch

from .data import DiscoverySplit
from .model import PredictiveDynamicsConceptFAN
from .operators import CHANNELS, OPERATORS


@dataclass
class DiscoveryEvidence:
    metrics: dict
    probabilities: np.ndarray
    contributions: np.ndarray
    trajectories: np.ndarray
    raw_activation: np.ndarray
    reliability: np.ndarray
    weights: np.ndarray
    gates: np.ndarray
    boundary: np.ndarray
    width: np.ndarray
    decision_weights: np.ndarray


def evaluate_discovery(
    model: PredictiveDynamicsConceptFAN,
    split: DiscoverySplit,
    batch_size: int = 128,
    *,
    hard_gates: bool = False,
    selected_slots: torch.Tensor | None = None,
) -> DiscoveryEvidence:
    model.eval()
    probabilities = []
    contributions = []
    trajectories = []
    raw_activation = []
    reliabilities = []
    forecast_sse = 0.0
    forecast_denominator = 0.0
    max_decomposition = 0.0
    selected_gates = None
    with torch.no_grad():
        for start in range(0, len(split.y), batch_size):
            stop = min(start + batch_size, len(split.y))
            values = torch.from_numpy(split.x[start:stop])
            mask = torch.from_numpy(split.mask[start:stop])
            out = model(values, mask, hard_gates=hard_gates, selected_slots=selected_slots)
            selected_gates = out.gates.detach().cpu().numpy()
            if not torch.isfinite(out.logits).all() or not torch.isfinite(out.future_prediction).all():
                raise FloatingPointError("Non-finite prediction in discovery audit")
            probabilities.append(out.probability.numpy())
            contributions.append(out.contributions.numpy())
            trajectories.append(out.concept_trajectories.numpy())
            raw_activation.append(out.raw_activation.numpy())
            reliabilities.append(out.availability.numpy())
            residual = (out.future_prediction.numpy() - split.future[start:stop]) ** 2
            active = split.future_mask[start:stop]
            forecast_sse += float((residual * active).sum())
            forecast_denominator += float(active.sum())
            max_decomposition = max(max_decomposition, float(out.decomposition_error()))
    probability = np.concatenate(probabilities)
    contrib = np.concatenate(contributions)
    concept = np.concatenate(trajectories)
    raw = np.concatenate(raw_activation)
    reliability = np.concatenate(reliabilities)
    with torch.no_grad():
        weights = model.feature_weights().numpy()
        gates = selected_gates.copy()
        boundary, width = model.concept_boundaries()
        boundary, width = boundary.numpy(), width.numpy()
        beta = model.decision_weights().numpy()

    # "Active slot" is an explicit operational threshold, not proof of a
    # canonical number of independent real-world mechanisms.
    variance = concept.var(axis=(0, 1))
    active = (gates >= 0.5) & (variance > 0.0005)
    complexity = float(model.gates.expected_l0().sum().detach())
    magnitude = np.abs(contrib)
    total_magnitude = magnitude.sum(axis=1, keepdims=True)
    proportions = np.divide(magnitude, total_magnitude,
                            out=np.zeros_like(magnitude), where=total_magnitude > 1e-12)
    entropy = -(proportions * np.log(np.clip(proportions, 1e-12, 1.0))).sum(axis=1)
    effective_count = np.where(total_magnitude[:, 0] > 1e-12, np.exp(entropy), 0.0)
    score = {
        "n_examples": int(len(split.y)),
        "auprc": float(average_precision_score(split.y, probability)),
        "auroc": float(roc_auc_score(split.y, probability)),
        "future_observed_rmse": float(np.sqrt(forecast_sse / max(forecast_denominator, 1))),
        "max_additive_error": float(max_decomposition),
        "mean_abs_contributions": float(np.abs(contrib).sum(axis=1).mean()),
        "n_hard_gate_slots": int(np.count_nonzero(gates >= .5)),
        "n_active_and_nonconstant_slots": int(active.sum()),
        "expected_l0": complexity,
        "mean_effective_concept_count": float(effective_count.mean()),
        "mean_concept_variance": float(variance.mean()),
        "mean_measurement_coverage": float(split.mask.mean()),
        "hard_gates_used": bool(hard_gates),
    }
    return DiscoveryEvidence(score, probability, contrib, concept, raw,
                             reliability, weights, gates, boundary, width, beta)


def match_by_operator_definition(a: DiscoveryEvidence, b: DiscoveryEvidence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match learned *definitions* before consulting any case values."""
    a_slots = np.flatnonzero((a.gates >= .5) & (a.trajectories.var(axis=(0, 1)) > .0005))
    b_slots = np.flatnonzero((b.gates >= .5) & (b.trajectories.var(axis=(0, 1)) > .0005))
    if not len(a_slots) or not len(b_slots):
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
    wa, wb = a.weights[a_slots], b.weights[b_slots]
    denominator = np.maximum(np.linalg.norm(wa, axis=1, keepdims=True) *
                             np.linalg.norm(wb, axis=1)[None, :], 1e-12)
    similarity = wa @ wb.T / denominator
    rows, cols = linear_sum_assignment(1.0 - similarity)
    return a_slots[rows], b_slots[cols], similarity[rows, cols]


def _episode_excess_agreement(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float, float]:
    if left.shape != right.shape or left.shape[1] < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    n, k = left.shape
    ra, rb = rankdata(left, axis=1), rankdata(right, axis=1)
    ra = ra - ra.mean(axis=1, keepdims=True)
    rb = rb - rb.mean(axis=1, keepdims=True)
    norm_a = np.linalg.norm(ra, axis=1, keepdims=True)
    norm_b = np.linalg.norm(rb, axis=1, keepdims=True)
    ua = ra / np.maximum(norm_a, 1e-12)
    ub = rb / np.maximum(norm_b, 1e-12)
    observed = float(np.mean(np.sum(ua * ub, axis=1)))
    independent_null = float(ua.mean(axis=0) @ ub.mean(axis=0))
    top_a, top_b = np.argmax(np.abs(left), axis=1), np.argmax(np.abs(right), axis=1)
    top_observed = float((top_a == top_b).mean())
    top_null = float(np.bincount(top_a, minlength=k) @ np.bincount(top_b, minlength=k) / n**2)
    return observed, observed - independent_null, top_observed, top_observed - top_null


def compare_independent_discoveries(a: DiscoveryEvidence, b: DiscoveryEvidence) -> dict:
    slots_a, slots_b, definition_cosines = match_by_operator_definition(a, b)
    active_a = a.metrics["n_active_and_nonconstant_slots"]
    active_b = b.metrics["n_active_and_nonconstant_slots"]
    normalized_definition_agreement = float(definition_cosines.sum() / max(active_a, active_b, 1))
    if len(slots_a) >= 2:
        l = a.contributions[:, slots_a]
        r = b.contributions[:, slots_b]
        rho, rho_excess, top1, top1_excess = _episode_excess_agreement(l, r)
        valid = []
        for k in range(len(slots_a)):
            av = a.trajectories[:, :, slots_a[k]].mean(axis=1)
            bv = b.trajectories[:, :, slots_b[k]].mean(axis=1)
            if av.std() > 1e-8 and bv.std() > 1e-8:
                valid.append(float(np.corrcoef(av, bv)[0, 1]))
        case_response_corr = float(np.mean(valid)) if valid else float("nan")
    else:
        rho = rho_excess = top1 = top1_excess = case_response_corr = float("nan")
    return {
        "matched_slots": int(len(slots_a)),
        "active_slots_a": int(active_a), "active_slots_b": int(active_b),
        "unmatched_slots_total": int(active_a + active_b - 2 * len(slots_a)),
        "definition_cosine_normalized": normalized_definition_agreement,
        "case_mean_response_correlation": case_response_corr,
        "paired_spearman": rho,
        "excess_over_unmatched_episode_null_spearman": rho_excess,
        "paired_top1": top1,
        "excess_over_unmatched_episode_null_top1": top1_excess,
        "matching_basis": "training-defined signed feature weights, no oracle labels",
    }


def oracle_correspondence(e: DiscoveryEvidence, oracle_states: np.ndarray) -> dict:
    """Evaluation-only matching to synthetic generator's hidden states."""
    active = np.flatnonzero((e.gates >= .5) &
                           (e.trajectories.var(axis=(0, 1)) > .0005))
    if not len(active):
        return {"oracle_mean_abs_corr": 0.0, "oracle_matched": 0, "oracle_truth_used_for_training": False}
    truths = np.asarray(oracle_states)
    if truths.shape[:2] != e.trajectories.shape[:2]:
        raise ValueError("Oracle must match validation patients and historic time steps")
    left = e.trajectories[:, :, active].reshape(-1, len(active))
    right = truths.reshape(-1, truths.shape[-1])
    cor = np.zeros((len(active), truths.shape[-1]), dtype=np.float64)
    for k in range(len(active)):
        for j in range(truths.shape[-1]):
            if left[:, k].std() > 1e-8 and right[:, j].std() > 1e-8:
                cor[k, j] = float(np.corrcoef(left[:, k], right[:, j])[0, 1])
    rows, cols = linear_sum_assignment(-np.abs(cor))
    result = {"oracle_mean_abs_corr": float(np.abs(cor[rows, cols]).mean()),
              "oracle_matched": int(len(rows)),
              "oracle_abs_per_planted_state": [
                  {"planted_state": int(j), "discovered_slot": int(active[k]),
                   "abs_corr": float(abs(cor[k, j]))}
                  for k, j in zip(rows, cols)
              ],
              "oracle_truth_used_for_training": False}
    return result


def concept_passports(
    e: DiscoveryEvidence,
    model: PredictiveDynamicsConceptFAN,
    split: DiscoverySplit,
    medians: np.ndarray,
    scales: np.ndarray,
    top_features: int = 5,
) -> list[dict]:
    """Reversible descriptions of precisely which observations form a slot."""
    names = tuple(
        f"{model.channels[j]}|{OPERATORS[i]}|{sign}"
        for sign in ("positive", "negative")
        for i in range(len(OPERATORS)) for j in range(len(model.channels))
    )
    passports = []
    interval_min_len = 2
    for k in range(len(e.gates)):
        top_indices = np.argsort(-e.weights[k])[:top_features]
        top = [
            {"operator": names[j], "nonnegative_weight": float(e.weights[k, j])}
            for j in top_indices if e.weights[k, j] > 1e-6
        ]
        raw_active = ((e.raw_activation[:, :, k] >= .5) &
                      (e.reliability[:, :, k] >= .5) & (e.gates[k] >= .5))
        episodes_with_event = 0
        durations = []
        example = []
        for patient in range(len(raw_active)):
            changes = np.diff(np.r_[False, raw_active[patient], False].astype(np.int8))
            starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
            periods = [[int(s), int(t)] for s, t in zip(starts, ends) if t - s >= interval_min_len]
            if periods:
                episodes_with_event += 1
                durations.extend([end - start for start, end in periods])
                if len(example) < 2:
                    example.append({"episode_id": int(split.episode_ids[patient]),
                                    "activation_intervals_in_hours": periods,
                                    "exact_logit_contribution": float(e.contributions[patient, k])})
        # Converting a weighted multivariate threshold into one clinical
        # variable cutoff would be mathematically false. Only a single
        # dominant level operator admits such an approximate translation.
        raw_threshold = None
        j = int(top_indices[0])
        if e.weights[k, j] > .85 and (j % (len(model.channels) * len(OPERATORS))) < len(model.channels):
            channel = j % len(model.channels)
            direction = 1.0 if j < len(model.channels) * len(OPERATORS) else -1.0
            if abs(e.boundary[k]) < .98:
                raw_threshold = {
                    "channel": model.channels[channel],
                    "approximate_value_in_original_units": float(
                        medians[channel] + scales[channel] * 2.0 *
                        np.arctanh(direction * e.boundary[k])),
                    "condition": "single dominant level operator only; approximate",
                }
        event_ratio = episodes_with_event / max(len(raw_active), 1)
        # Exact prediction-head removal, not a causal treatment effect.
        prob_removed = 1.0 / (1.0 + np.exp(-np.clip(
            np.log(np.clip(e.probabilities, 1e-8, 1-1e-8) /
                   np.clip(1 - e.probabilities, 1e-8, 1-1e-8))
            - e.contributions[:, k], -40, 40)))
        ap_removed = float(average_precision_score(split.y, prob_removed))
        passports.append({
            "slot": int(k), "learned_gate": float(e.gates[k]),
            "active_under_half_gate_threshold": bool(e.gates[k] >= .5),
            "nonconstant_activation": bool(e.trajectories[:, :, k].var() > .0005),
            "mathematical_definition": "sigmoid((sparse_signed_operator_sum - learned_threshold) / learned_width) * coverage * gate",
            "top_signed_operators": top,
            "learned_score_threshold": float(e.boundary[k]),
            "learned_transition_width": float(e.width[k]),
            "physical_single_channel_threshold_when_defined": raw_threshold,
            "mean_reliability": float(e.reliability[:, :, k].mean()),
            "activation_episode_fraction": float(event_ratio),
            "mean_detected_event_duration_hours": float(np.mean(durations)) if durations else 0.,
            "mean_absolute_logit_contribution": float(np.abs(e.contributions[:, k]).mean()),
            "average_precision_if_direct_contribution_zeroed": ap_removed,
            "baseline_average_precision": float(e.metrics["auprc"]),
            "illustrative_validation_episodes": example,
            "is_physically_grounded": False,
            "causal_intervention_claim": False,
        })
    return passports
