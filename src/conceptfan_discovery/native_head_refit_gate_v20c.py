"""V20c: conservative acceptance of train-only native head refit.

The acceptance data are independent of the original native model's TRAIN
core, not merely independent of the head refit. A rejected candidate leaves
the original ConceptFAN untouched. The gate certifies neither physiology
nor causal correctness of a discovered concept.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HeadRefitGateDecision:
    status: str
    n_gate_patients: int
    candidate_risk_point_biserial_r: float
    candidate_risk_one_sided_permutation_p: float
    mean_gate_logloss_gain: float
    gain_bootstrap_ci90: tuple[float,float]
    fixed_min_gain: float
    outcome_association_only_not_physical_identity: bool = True


def decide_head_refit_from_fit_disjoint_train_gate(
    gate_labels: np.ndarray,
    old_probabilities: np.ndarray,
    new_probabilities: np.ndarray,
    *,
    seed: int = 70851,
    permutations: int = 499,
    bootstraps: int = 499,
    minimum_absolute_logloss_gain: float = .005,
) -> HeadRefitGateDecision:
    y=np.asarray(gate_labels,dtype=np.float64)
    old=np.asarray(old_probabilities,dtype=np.float64)
    new=np.asarray(new_probabilities,dtype=np.float64)
    if (y.ndim!=1 or old.shape!=y.shape or new.shape!=y.shape
            or len(y)<100 or not np.isin(y,[0,1]).all()
            or len(np.unique(y))!=2
            or not np.isfinite(y).all() or not np.isfinite(old).all()
            or not np.isfinite(new).all()
            or not ((old>=0)&(old<=1)).all()
            or not ((new>=0)&(new<=1)).all()):
        raise ValueError("ABSTAIN: invalid independent labelled gate cohort")
    if not (permutations>=99 and bootstraps>=99
            and .0<=minimum_absolute_logloss_gain<=.1):
        raise ValueError("Unregistered screening rule settings")
    rng=np.random.default_rng(seed)
    def point_biserial(a,b):
        sd_a=float(a.std())
        sd_b=float(b.std())
        if sd_a<1e-12 or sd_b<1e-12:
            return 0.
        return float(np.mean((a-a.mean())*(b-b.mean()))/(sd_a*sd_b))
    point=point_biserial(new,y)
    shuffled=np.array([
        point_biserial(new,rng.permutation(y))
        for _ in range(int(permutations))
    ])
    p=float((1+int(np.count_nonzero(shuffled>=point)))/(1+permutations))
    def logloss_entries(probs):
        q=np.clip(probs,1e-7,1-1e-7)
        return -y*np.log(q)-(1-y)*np.log1p(-q)
    deltas=logloss_entries(old)-logloss_entries(new)
    gains=np.asarray([
        deltas[rng.integers(0,len(deltas),size=len(deltas))].mean()
        for _ in range(int(bootstraps))
    ])
    low,high=map(float,np.quantile(gains,[.05,.95]))
    mean=float(deltas.mean())
    accepted=(point>0 and p<=.01 and mean>=minimum_absolute_logloss_gain
              and low>0)
    return HeadRefitGateDecision(
        status="ACCEPT_TRAIN_ONLY_NATIVE_REFIT" if accepted
                else "KEEP_ORIGINAL_ABSTAIN_UNCERTAIN_GAIN",
        n_gate_patients=len(y),
        candidate_risk_point_biserial_r=point,
        candidate_risk_one_sided_permutation_p=p,
        mean_gate_logloss_gain=mean,
        gain_bootstrap_ci90=(low,high),
        fixed_min_gain=float(minimum_absolute_logloss_gain),
    )
