"""V18: TRAIN-only, independent risk screening of *observable* concepts.

V17 can propose a stable new sensor trajectory, but cannot distinguish a
physical latent factor from temporally correlated instrument noise. Even
physical meaningfulness would not establish association with the outcome.
This module tests only the *predictive association of an observable pattern*
on a disjoint training cohort. It never claims medical causation, clinical
validity, or post-hoc recovery of black-box features.

The risk orientation is attached to the same actual V10 ConceptFAN decision
coefficient: no classifier bypass, no parallel attribution head.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

import numpy as np
import torch

from .candidate_admission_v17 import ProposedObservableConcept
from .multiscale_physiology_v10 import MultiScaleDualProcessConceptFAN


@dataclass(frozen=True)
class RiskAssociationEvidence:
    candidate_id: str
    status: str
    signed_orientation: int
    n_screen_episodes: int
    point_biserial_r: float
    bootstrap_ci95: tuple[float, float]
    permutation_p_two_sided: float
    two_split_correlations: tuple[float, float]
    minimum_abs_r: float
    criterion: str = "TRAIN_ONLY_OBSERVATIONAL_PREDICTIVE_ASSOCIATION"
    physiological_identity_certified: bool = False
    causal_treatment_effect_certified: bool = False


def _point_biserial(scores: np.ndarray, labels: np.ndarray) -> float:
    s = np.asarray(scores,dtype=np.float64)
    y = np.asarray(labels,dtype=np.float64)
    if len(s)<3 or len(np.unique(y)) != 2:
        return 0.
    a = float(np.std(s))
    b = float(np.std(y))
    if a < 1e-12 or b < 1e-12:
        return 0.
    return float(np.clip(np.mean((s-s.mean())*(y-y.mean()))/(a*b),-1,1))


def screen_risk_association(
    candidate: ProposedObservableConcept,
    screening_observations: np.ndarray,
    screening_mask: np.ndarray,
    screening_labels: np.ndarray,
    *,
    screening_episode_ids: Iterable[int],
    discovery_episode_ids: Iterable[int],
    seed: int = 20661104,
    permutations: int = 199,
    bootstraps: int = 299,
    alpha: float = .01,
    min_abs_r: float = .15,
    window: int = 6,
) -> RiskAssociationEvidence:
    """Cross-cohort screen + two-split sign agreement + bootstrap CI.

    Rules fixed before checking outcomes: p<=.01, 95% CI excludes zero,
    |Pearson point-biserial correlation|>=.15, both disjoint split-halves have
    same nonzero direction as the full cohort. Positive/negative correlations
    are valid; no automatic label 'increases disease risk' follows.
    """
    x = np.asarray(screening_observations,dtype=np.float64)
    mask = np.asarray(screening_mask,dtype=np.float64)
    y = np.asarray(screening_labels,dtype=np.float64)
    ids = [int(i) for i in screening_episode_ids]
    discovery = {int(i) for i in discovery_episode_ids}
    if not isinstance(candidate, ProposedObservableConcept):
        raise ValueError("Expected previously registered V17 observable concept")
    if (x.ndim!=3 or x.shape != mask.shape or x.shape[0]<180
            or len(ids)!=x.shape[0] or y.shape!=(x.shape[0],)
            or not 2<=window<=x.shape[1]
            or candidate.direction.shape!=(x.shape[2],)
            or not np.isfinite(candidate.direction).all()
            or not np.isfinite(x).all() or not np.isfinite(mask).all()
            or not np.isfinite(y).all()):
        raise ValueError("ABSTAIN: invalid independent risk-screen sample")
    if len(set(ids)) != len(ids) or set(ids)&discovery or not discovery:
        raise ValueError("ABSTAIN: discovery and risk-screen cohorts must be disjoint")
    if not np.array_equal(mask,np.ones_like(mask)):
        raise ValueError("ABSTAIN: incomplete sensor history requires another null model")
    if not np.isin(y,[0.,1.]).all() or len(np.unique(y))!=2:
        raise ValueError("ABSTAIN: screening labels must contain both binary outcomes")
    if not 0 < alpha <= .05 or not 0 < min_abs_r < 1 or permutations < 99 or bootstraps < 99:
        raise ValueError("Invalid predeclared statistical settings")

    # Candidate orientation is determined solely from V17 TRAIN measurement
    # direction, not fitted to risk labels. Prediction screen is disjoint.
    score = np.mean(x[:,-window:,:]@candidate.direction,axis=1)
    observed_r = _point_biserial(score,y)
    rng = np.random.default_rng(seed)
    shuffle = rng.permutation(len(y))
    idx_a,idx_b = np.array_split(shuffle,2)
    two_r = (_point_biserial(score[idx_a],y[idx_a]),
             _point_biserial(score[idx_b],y[idx_b]))

    null_abs = np.asarray([
        abs(_point_biserial(score,rng.permutation(y)))
        for _ in range(permutations)
    ])
    p = float((1+np.count_nonzero(null_abs >= abs(observed_r)))/(1+permutations))
    bs = np.asarray([
        _point_biserial(score[b],y[b])
        for b in (rng.integers(0,len(y),size=len(y)) for _ in range(bootstraps))
    ])
    low,high=map(float,np.quantile(bs,[.025,.975]))
    direction = 1 if observed_r>0 else -1 if observed_r<0 else 0
    stable = (direction != 0
        and all(float(np.sign(r))==direction for r in two_r)
        and min(abs(r) for r in two_r) >= .035)
    ci_support = (low>0 if direction==1 else high<0 if direction==-1 else False)
    accepted = bool(stable and ci_support and p <= alpha
                    and abs(observed_r) >= min_abs_r)
    return RiskAssociationEvidence(
        candidate_id=candidate.candidate_id,
        status=("ACCEPT_PREDICTIVE_ASSOCIATION" if accepted else "ABSTAIN_UNSTABLE_OR_WEAK_ASSOCIATION"),
        signed_orientation=(direction if accepted else 0),
        n_screen_episodes=len(y),
        point_biserial_r=observed_r,
        bootstrap_ci95=(low,high),
        permutation_p_two_sided=p,
        two_split_correlations=tuple(map(float,two_r)),
        minimum_abs_r=float(min_abs_r),
    )


def configure_native_risk_orientation(
    expanded_model: MultiScaleDualProcessConceptFAN,
    candidate: ProposedObservableConcept,
    evidence: RiskAssociationEvidence,
    *,
    slot: int,
) -> None:
    """Orient one *existing native* risk coefficient without creating a bypass.

    Keeps the newly admitted coefficient at exact zero. No immediate
    prediction drift; explicit begin_native_risk_training later seeds a
    small learnable coefficient to avoid abs()'s zero-gradient at zero.
    """
    if (evidence.status!="ACCEPT_PREDICTIVE_ASSOCIATION"
        or evidence.signed_orientation not in (-1,1)
        or candidate.candidate_id != evidence.candidate_id):
        raise ValueError("ABSTAIN: no independent predictive evidence for this candidate")
    physical = expanded_model.physical
    if slot != physical.n_slots-1 or not hasattr(physical,"signed_risk_policy"):
        raise ValueError("Policy may be attached only to a newly admitted native slot")
    if abs(float(physical.decision_raw_weight[slot].detach())) > 1e-10:
        raise ValueError("Expected zero-influence slot prior to admission")
    with torch.no_grad():
        physical.signed_risk_policy[slot] = evidence.signed_orientation


def begin_native_risk_training(
    expanded_model: MultiScaleDualProcessConceptFAN,
    *,
    slot: int,
    initial_coefficient_raw: float = .05,
) -> None:
    """Explicit zero-to-trainable transition *after* consent and risk screening."""
    p = expanded_model.physical
    if (slot!=p.n_slots-1 or int(p.signed_risk_policy[slot]) not in (-1,1)
        or not .005 <= initial_coefficient_raw <= .1):
        raise ValueError("Risk training not authorized for this slot")
    with torch.no_grad():
        if abs(float(p.decision_raw_weight[slot])) > 1e-10:
            raise ValueError("Risk training must be started exactly once")
        p.decision_raw_weight[slot] = float(initial_coefficient_raw)


def risk_evidence_passport(evidence: RiskAssociationEvidence) -> dict:
    return {
        "candidate_id":evidence.candidate_id,
        "status":evidence.status,
        "orientation":evidence.signed_orientation,
        "n_train_screen":evidence.n_screen_episodes,
        "point_biserial_r":evidence.point_biserial_r,
        "bootstrap_ci95":list(evidence.bootstrap_ci95),
        "permutation_p_two_sided":evidence.permutation_p_two_sided,
        "split_correlations":list(evidence.two_split_correlations),
        "observational_prediction_only":True,
        "physical_cause_identified":False,
        "clinical_semantic_ground_truth":False,
        "sign_is_fixed_from_train_evidence_NOT_new_cross_seed_discovery":True,
    }


def native_observable_grounding_loss(
    physical_output,
    observed_input: torch.Tensor,
    observed_mask: torch.Tensor,
    candidate: ProposedObservableConcept,
    *,
    slot: int,
    window: int = 6,
) -> torch.Tensor:
    """Optional V18 training constraint that ties the actual native activation
    to a measured signed source direction, NOT to an oracle hidden variable.

    The target is computed from *the same masked history the model sees*.
    This is a bounded monotonic observable score, not a physical truth label.
    """
    a=physical_output.raw_activation
    rel=physical_output.availability
    if (observed_input.ndim != 3 or observed_input.shape != observed_mask.shape
        or a.ndim != 3 or a.shape != rel.shape
        or a.shape[:2] != observed_input.shape[:2]
        or not 0 <= slot < a.shape[2]
        or candidate.direction.shape != (observed_input.shape[2],)
        or not 2 <= window <= observed_input.shape[1]):
        raise ValueError("Native and observable concept grounding shapes disagree")
    if not torch.isfinite(observed_input).all() or not torch.isfinite(observed_mask).all():
        raise ValueError("Nonfinite observed training history")
    if torch.any(observed_mask < 0) or torch.any(observed_mask > 1):
        raise ValueError("Invalid observation support")
    direction=torch.as_tensor(candidate.direction,device=observed_input.device,
                              dtype=observed_input.dtype)
    support=observed_mask[:,-window:,:]@direction.abs()
    projected=(observed_input[:,-window:,:] * observed_mask[:,-window:,:])@direction
    score=projected.sum(dim=1) / support.sum(dim=1).clamp_min(1e-8)
    target=(.5 + .40*torch.tanh(score / 2.0)).detach()
    evidence=rel[:,-window:,slot].detach().clamp(0,1)
    predicted=(a[:,-window:,slot]*evidence).sum(dim=1) / evidence.sum(dim=1).clamp_min(1e-8)
    sufficient=(support.sum(dim=1)>1e-8) & (evidence.sum(dim=1)>1e-8)
    if not bool(sufficient.any()):
        return a.sum()*0.0
    # MSE on predicted temporal evidence, not on per-patient risk logit.
    return ((predicted[sufficient]-target[sufficient])**2).mean()
