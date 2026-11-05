"""Independent-label admission of masked, TRAIN-formed measured concept candidates.

Group proposal and PCA are completely label free. Risk coefficient fitting
sees TRAIN labels; admission uses TWO DISJOINT validation cohorts with paired
patient bootstrap and conservative 1% lower-tail thresholds, no TEST outcomes.
This is *predictive usefulness*, not proof of semantic or physical identity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA

from conceptfan_discovery.isolated_source_admission_v28 import (
    train_source_coefficient,
)
from conceptfan_discovery.auto_measured_concept_proposals import (
    MeasuredGroup, measured_group_concept,
)


@dataclass(frozen=True)
class EnvironmentGate:
    name: str
    n: int
    gain_in_mean_nll: float
    one_sided_99pct_bootstrap_lower: float
    source_eligible_fraction: float
    passed: bool


@dataclass(frozen=True)
class CandidateGate:
    candidate_id: str
    gamma_train_only: float
    approved: bool
    environment_receipts: tuple[EnvironmentGate,...]
    reason: str
    min_two_env_gain: float
    n_total_train_labels: int
    n_total_valid_labels: int
    semantic_or_physical_identity_confirmed: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PCAUnknownProposal:
    sensor_indices: tuple[int,...]
    train_means: tuple[float,...]
    train_stds: tuple[float,...]
    projection_weights: tuple[float,...]
    train_episodes: int
    candidate_id: str = "pca_all_unknown_rank1"
    physical_semantics_identified: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def fixed_unobserved_pca(train_x:np.ndarray,train_mask:np.ndarray,
                          unknown:tuple[int,...])->PCAUnknownProposal:
    """Conventional TRAIN PCA baseline on episode-means, zero-fill at TRAIN means."""
    x=np.asarray(train_x,dtype=np.float64)
    m=np.asarray(train_mask).astype(bool)
    if x.ndim!=3 or x.shape!=m.shape or len(x)<100 or len(unknown)<2:
        raise ValueError("Invalid TRAIN PCA source cohort")
    means,scales=[],[]
    for j in unknown:
        v=x[:,:,j][m[:,:,j]]
        if len(v)<100 or not np.isfinite(v).all():
            raise ValueError("No stable per-channel TRAIN PCA normalization")
        means.append(float(v.mean()))
        scales.append(float(max(v.std(),.10)))
    feats=[]
    for j,center,scale in zip(unknown,means,scales):
        valid=m[:,:,j]
        mean=np.where(valid,x[:,:,j],0.).sum(axis=1)/np.maximum(1,valid.sum(axis=1))
        mean=np.where(valid.any(axis=1),mean,center)
        feats.append((mean-center)/scale)
    matrix=np.column_stack(feats)
    pca=PCA(n_components=1,svd_solver="full")
    pca.fit(matrix)
    direction=pca.components_[0].copy()
    pivot=np.argmax(np.abs(direction))
    if direction[pivot]<0:
        direction=-direction
    return PCAUnknownProposal(
        sensor_indices=tuple(unknown),
        train_means=tuple(means),
        train_stds=tuple(scales),
        projection_weights=tuple(float(v) for v in direction),
        train_episodes=len(x),
    )


def pca_unknown_values(pca:PCAUnknownProposal,x:np.ndarray,m:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    x=np.asarray(x,dtype=np.float64)
    m=np.asarray(m).astype(bool)
    if x.ndim!=3 or x.shape!=m.shape or max(pca.sensor_indices)>=x.shape[-1]:
        raise ValueError("PCA source dimensions incompatible with masked TEST")
    features=[]
    for j,mean,std in zip(pca.sensor_indices,pca.train_means,pca.train_stds):
        available=m[:,:,j]
        values=np.where(available,x[:,:,j],0.)
        valid_count=available.sum(axis=1)
        z=values.sum(axis=1)/np.maximum(1,valid_count)
        z=np.where(valid_count>0,z,mean)
        features.append((z-mean)/std)
    matrix=np.column_stack(features)
    proj=matrix@np.asarray(pca.projection_weights)
    support=m[:,:,pca.sensor_indices].mean(axis=(1,2))
    return 0.5+support*(expit(proj)-0.5),support


def _paired_binary_loss(y:np.ndarray,logit:np.ndarray)->np.ndarray:
    yy=np.asarray(y,dtype=np.float64).reshape(-1)
    z=np.asarray(logit,dtype=np.float64).reshape(-1)
    if len(yy)!=len(z) or not np.isin(yy,[0.,1.]).all() or not np.isfinite(z).all():
        raise ValueError("Invalid binary source-risk evaluation")
    return np.logaddexp(0.,z)-yy*z


def admitted_test_logit(
    base:np.ndarray,q:np.ndarray,support:np.ndarray,
    *,gamma:float,approved:bool,minimum_fraction:float=.50
)->tuple[np.ndarray,np.ndarray]:
    base=np.asarray(base,dtype=np.float64).reshape(-1)
    q=np.asarray(q,dtype=np.float64).reshape(-1)
    support=np.asarray(support,dtype=np.float64).reshape(-1)
    if not(len(base)==len(q)==len(support)) or not (
        np.isfinite(base).all() and np.isfinite(q).all() and
        np.isfinite(support).all() and 0<=minimum_fraction<=1):
        raise ValueError("Invalid source, support or frozen risk")
    used=(support>=minimum_fraction)&bool(approved)
    signed=np.where(used,float(gamma)*(q-0.5),0.0)
    return base+signed,used


def evaluate_two_independent_environments(
    name:str,gamma:float,
    env1:tuple[str,np.ndarray,np.ndarray,np.ndarray,np.ndarray],
    env2:tuple[str,np.ndarray,np.ndarray,np.ndarray,np.ndarray],
    *,
    train_label_count:int,
    min_mean_gain:float=.003,
    min_support_fraction:float=.75,
    min_subject_support:float=.50,
    bootstrap_reps:int=399,
    seed:int=20267210,
)->CandidateGate:
    """Two registered disjoint VALID SUBJECT cohorts; all thresholds frozen.

    Bonferroni-like 99% one-sided bootstrap lower bands per candidate in
    a maximum 5-candidate screen are CONSERVATIVE heuristics, not exact FWER.
    """
    if bootstrap_reps<199 or not 0<min_mean_gain<1 or not 0<min_support_fraction<=1:
        raise ValueError("Unregistered source-gate thresholds")
    parts=[]
    ntotal=0
    for i,(label,y,b,q,s) in enumerate((env1,env2)):
        y=np.asarray(y,dtype=np.float64).reshape(-1)
        b=np.asarray(b,dtype=np.float64).reshape(-1)
        q=np.asarray(q,dtype=np.float64).reshape(-1)
        s=np.asarray(s,dtype=np.float64).reshape(-1)
        if len(y)<100 or not(len(y)==len(b)==len(q)==len(s)):
            raise ValueError("Each VALID environment needs at least 100 aligned episodes")
        ntotal+=len(y)
        z,used=admitted_test_logit(b,q,s,gamma=gamma,approved=True,
                                    minimum_fraction=min_subject_support)
        delta=_paired_binary_loss(y,b)-_paired_binary_loss(y,z)
        rng=np.random.default_rng(seed+i)
        indices=rng.integers(0,len(y),size=(bootstrap_reps,len(y)))
        bs=delta[indices].mean(axis=1)
        lower=float(np.quantile(bs,.01))
        gain=float(delta.mean())
        fraction=float(used.mean())
        passed=(lower>0. and gain>=min_mean_gain and
                fraction>=min_support_fraction)
        parts.append(EnvironmentGate(label,len(y),gain,lower,fraction,passed))
    approved=bool(all(p.passed for p in parts))
    return CandidateGate(
        candidate_id=name,gamma_train_only=float(gamma),approved=approved,
        environment_receipts=tuple(parts),
        reason=("ACCEPT_OBSERVATIONAL_TASK_GAIN_IN_BOTH_VALID_ENVIRONMENTS"
                if approved else "ABSTAIN_NO_REPLICATED_VALID_GAIN"),
        min_two_env_gain=float(min(p.gain_in_mean_nll for p in parts)),
        n_total_train_labels=int(train_label_count),
        n_total_valid_labels=int(ntotal),
    )


def select_at_most_one_candidate(candidates:tuple[CandidateGate,...])->str|None:
    approved=[c for c in candidates if c.approved]
    if not approved:
        return None
    approved.sort(key=lambda c:(-c.min_two_env_gain,c.candidate_id))
    return approved[0].candidate_id
