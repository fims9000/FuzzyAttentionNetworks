"""Validated concept-cardinality selection without oracle labels or test access.

A backward-elimination path removes native concept slots from BOTH the
classification and forecasting computation graph. The selection half of
validation decides cardinality. An untouched audit half measures outcomes.
This does not create concepts post-hoc: every slot was built and trained
inside the model. Cardinality is a separate, disclosed model-selection step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import StratifiedShuffleSplit

from .data import DiscoverySplit
from .evaluation import DiscoveryEvidence, evaluate_discovery
from .model import PredictiveDynamicsConceptFAN


def subset_split(split: DiscoverySplit, indices: np.ndarray) -> DiscoverySplit:
    idx = np.asarray(indices)
    return DiscoverySplit(
        x=split.x[idx], mask=split.mask[idx],
        future=split.future[idx], future_mask=split.future_mask[idx],
        y=split.y[idx], oracle_states=split.oracle_states[idx],
        episode_ids=split.episode_ids[idx],
    )


def split_validation_for_cardinality(
    original: DiscoverySplit, *, random_seed: int = 20261107,
    selection_fraction: float = 0.5,
) -> tuple[DiscoverySplit, DiscoverySplit]:
    y = original.y.astype(int)
    if not 0.25 <= selection_fraction <= 0.75 or len(y) < 40:
        raise ValueError("Validation needs >=40 patients and reasonably sized halves")
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=selection_fraction, random_state=random_seed
    )
    selection, audit = next(splitter.split(original.x, y))
    if len(set(original.episode_ids[selection]) & set(original.episode_ids[audit])):
        raise AssertionError("Validation cardinality selection and audit must be disjoint")
    return subset_split(original, selection), subset_split(original, audit)


@dataclass
class SelectedConceptModel:
    selected_mask: np.ndarray
    path: list[dict]
    selection_metrics: dict
    audit_metrics: dict
    selection_episode_ids: np.ndarray
    audit_episode_ids: np.ndarray


def _evaluate_subset(
    model: PredictiveDynamicsConceptFAN,
    split: DiscoverySplit,
    enabled: list[int],
) -> tuple[dict, DiscoveryEvidence]:
    mask = torch.zeros(model.n_slots, dtype=torch.float32)
    mask[enabled] = 1.0
    e = evaluate_discovery(model, split, selected_slots=mask)
    nll = float(log_loss(split.y, e.probabilities, labels=[0, 1]))
    return {
        "enabled": list(enabled),
        "k": len(enabled),
        "auprc": float(average_precision_score(split.y, e.probabilities)),
        "nll": nll,
        "future_rmse": float(e.metrics["future_observed_rmse"]),
        "hard_gate_nonconstant": int(e.metrics["n_active_and_nonconstant_slots"]),
        "max_additive_error": float(e.metrics["max_additive_error"]),
    }, e


def select_concept_cardinality(
    model: PredictiveDynamicsConceptFAN,
    selection: DiscoverySplit,
    audit: DiscoverySplit,
    *,
    mode: str,
    auprc_tolerance: float = 0.02,
    nll_tolerance: float = 0.03,
    future_rmse_tolerance: float = 0.035,
    future_selection_weight: float = 0.40,
) -> SelectedConceptModel:
    """Find a small *native* concept subset on an independent validation half.

    These predefined noninferiority tolerances are part of the method.
    Exact global minimum K is not guaranteed: backwards elimination is
    greedy and can miss nonnested subsets. The full trace is retained.
    """
    if mode not in ("task_only", "task_future", "full"):
        raise ValueError("Unsupported objective mode")
    if min(auprc_tolerance, nll_tolerance, future_rmse_tolerance) < 0:
        raise ValueError("Noninferiority tolerances must be nonnegative")
    if len(set(selection.episode_ids.tolist()) & set(audit.episode_ids.tolist())):
        raise ValueError("Selection and audit episode IDs must be disjoint")
    model.eval()
    active = list(range(model.n_slots))
    baseline, _ = _evaluate_subset(model, selection, active)
    trace = [{"step": 0, "removed": None, "selection": baseline}]
    while len(active) > 1:
        candidates = []
        for removed in active:
            enabled = [slot for slot in active if slot != removed]
            metrics, _ = _evaluate_subset(model, selection, enabled)
            acceptable = (
                metrics["auprc"] >= baseline["auprc"] - auprc_tolerance
                and metrics["nll"] <= baseline["nll"] + nll_tolerance
                and (mode == "task_only" or
                     metrics["future_rmse"] <= baseline["future_rmse"] + future_rmse_tolerance)
            )
            if acceptable:
                tradeoff = metrics["nll"] + (
                    future_selection_weight * metrics["future_rmse"] if mode != "task_only" else 0.0
                )
                candidates.append((tradeoff, removed, enabled, metrics))
        if not candidates:
            break
        _, removed, active, chosen = min(candidates, key=lambda v: (v[0], v[1]))
        trace.append({"step": len(trace), "removed": removed, "selection": chosen})
    selection_metrics, _ = _evaluate_subset(model, selection, active)
    audit_metrics, _ = _evaluate_subset(model, audit, active)
    selector = np.zeros(model.n_slots, dtype=np.float32)
    selector[active] = 1.0
    audit_metrics["k_selected"] = len(active)
    audit_metrics["selected_slot_ids"] = list(active)
    audit_metrics["selection_not_used_for_audit"] = True
    audit_metrics["selection_method"] = "greedy native-slot elimination with predeclared noninferiority criteria"
    return SelectedConceptModel(
        selector, trace, selection_metrics, audit_metrics,
        selection.episode_ids.copy(), audit.episode_ids.copy(),
    )
