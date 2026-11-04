"""Training-native expert feedback for ConceptFAN V10.

Expert statements target a single native concept slot and a causal observation
interval. 'unsure' and 'artifact' never become negative physiological labels.
The target is raw_activation from the *actual* V10 physical concept module,
not an independent post-hoc classifier. No clinical identity is implied.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import torch
import torch.nn.functional as F

VALID = frozenset(("yes", "no", "unsure", "artifact"))

@dataclass(frozen=True)
class ExpertReview:
    episode_id: int
    slot: int
    start: int
    end: int
    decision: str
    confidence: float = 1.0
    reviewer_id: str = "anonymous"

@dataclass(frozen=True)
class PreparedFeedback:
    # (episode_id, slot, start, end, target, confidence)
    accepted: tuple[tuple[int, int, int, int, float, float], ...]
    audit: dict[str, int]
    n_slots: int
    n_steps: int


def prepare_feedback(reviews: Iterable[ExpertReview], train_episode_ids: Iterable[int],
                     *, n_slots: int, n_steps: int) -> PreparedFeedback:
    """Validate train membership and quarantine contradictory overlapping reviews.

    Repeated equal opinions are merged, not treated as independent patients.
    A single artifact judgement excludes its interval; uncertainty is not a
    vote against presence. Contradictory overlapping hard judgements exclude
    all affected intervals from supervision.
    """
    train_ids = set(int(i) for i in train_episode_ids)
    if not train_ids or n_slots < 1 or n_steps < 2:
        raise ValueError("Need a real TRAIN split and concept/time coordinates")
    groups = defaultdict(list)
    for r in reviews:
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
               for v in (r.episode_id, r.slot, r.start, r.end)):
            raise ValueError("Review IDs and coordinates must be integers")
        if r.episode_id not in train_ids:
            raise ValueError("ABSTAIN: feedback may not access validation/test IDs")
        if not 0 <= r.slot < n_slots or not 0 <= r.start < r.end <= n_steps:
            raise ValueError("Review interval or concept slot out of bounds")
        if r.decision not in VALID or not np.isfinite(r.confidence) or not 0 <= r.confidence <= 1:
            raise ValueError("Unsupported expert decision or confidence")
        if not isinstance(r.reviewer_id, str) or not r.reviewer_id:
            raise ValueError("Reviewer token is required")
        groups[(int(r.episode_id), int(r.slot), int(r.start), int(r.end))].append(r)
    # Reject conflicts even when they occur across *overlapping* intervals.
    keys = sorted(groups)
    opinions = {key: {r.decision for r in groups[key] if r.decision != "unsure"}
                for key in keys}
    excluded = set()
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a[:2] != b[:2]:
                continue
            if a[2] < b[3] and b[2] < a[3] and opinions[a] and opinions[b] and opinions[a] != opinions[b]:
                excluded.update((a, b))
    accepted = []
    audit = dict(received=sum(len(x) for x in groups.values()), accepted=0,
                 uncertain=0, artifact=0, conflicting=0, zero_confidence=0,
                 merged_duplicates=sum(len(x) - 1 for x in groups.values()))
    for key in keys:
        votes = opinions[key]
        if key in excluded or len(votes) > 1:
            audit["conflicting"] += 1
            continue
        if "artifact" in votes:
            audit["artifact"] += 1
            continue
        if not votes:
            audit["uncertain"] += 1
            continue
        decision = next(iter(votes))
        confidence = float(np.mean([r.confidence for r in groups[key] if r.decision == decision]))
        if confidence <= 0:
            audit["zero_confidence"] += 1
            continue
        accepted.append((*key, float(decision == "yes"), confidence))
    audit["accepted"] = len(accepted)
    return PreparedFeedback(tuple(accepted), audit, n_slots, n_steps)


def native_expert_loss(physical_output, batch_episode_ids: torch.Tensor,
                       prepared: PreparedFeedback, *, min_reliability: float = .1):
    """Differentiable feedback loss on V10's native raw concept activation.

    Unsupported observations are discarded, not labeled negative. Smooth-L1
    bounds loss gradient per annotation but is not a defense against organized
    incorrect experts. Each supervised slot gets equal average weight.
    """
    activation = physical_output.raw_activation
    reliability = physical_output.availability
    if activation.ndim != 3 or activation.shape != reliability.shape:
        raise ValueError("Expected V10 native evidence [batch, time, slots]")
    if activation.shape[1:] != (prepared.n_steps, prepared.n_slots):
        raise ValueError("Prepared feedback does not match native V10 concept dimensions")
    if batch_episode_ids.ndim != 1 or len(batch_episode_ids) != len(activation):
        raise ValueError("One original episode ID required per input row")
    if not 0 <= min_reliability < 1:
        raise ValueError("Invalid reliability threshold")
    if not torch.isfinite(activation).all() or not torch.isfinite(reliability).all():
        raise ValueError("Nonfinite concept evidence")
    # Convex mixtures of reliability scores can exceed 1 by float32 ULP.
    # Reject true contract violations, tolerate only numerical roundoff;
    # each per-interval support is clamped before computing the loss.
    if torch.any(reliability < -1e-5) or torch.any(reliability > 1 + 1e-5):
        raise ValueError("Invalid concept measurement reliability")
    index = {int(v): i for i, v in enumerate(batch_episode_ids.tolist())}
    if len(index) != len(batch_episode_ids):
        raise ValueError("Duplicated IDs in minibatch")
    per_slot = defaultdict(list)
    unsupported = 0
    for episode_id, slot, start, end, target, confidence in prepared.accepted:
        row = index.get(episode_id)
        if row is None:
            continue
        support = reliability[row, start:end, slot].detach().clamp(0, 1)
        strength = support.mean()
        if float(strength) < min_reliability or float(support.sum()) <= 1e-8:
            unsupported += 1
            continue
        prediction = (activation[row, start:end, slot] * support).sum() / support.sum().clamp_min(1e-8)
        per_slot[slot].append(F.smooth_l1_loss(prediction, prediction.new_tensor(target), beta=.25)
                              * confidence * strength)
    slot_losses = [torch.stack(values).mean() for _, values in sorted(per_slot.items())]
    result = torch.stack(slot_losses).mean() if slot_losses else activation.sum() * 0
    return result, {"supervised_slots": len(slot_losses), "unsupported_intervals": unsupported,
                    "feedback_loss": float(result.detach())}


def safe_feedback_manifest(prepared: PreparedFeedback) -> dict:
    """Aggregate-only provenance: never export patient-level reviews to CI."""
    return {"feedback_type": "native_train_only_expert_judgement",
            "clinical_validity_established": False,
            "independent_model_supervision": False,
            "unknown_is_negative": False, "artifacts_are_negative": False,
            "individual_records_exported": False, **prepared.audit}
