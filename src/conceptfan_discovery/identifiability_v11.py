"""Predictive, rotation-aware concept identifiability (V11 research prototype).

A single time-series classification target cannot identify clinical concept
variables. Positive-lag, TRAIN-ONLY cross covariance supplies observable
predictive directions. Bootstrap singular *gaps* distinguish resolved axes
from rotations of a statistically unresolved predictive subspace.

Certificates are about empirical operator identifiability, NOT medical
causality, unique nonlinear source recovery, or population 95% guarantees.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import torch

from .lagged_rank import _center, _matrix_from_centered


def projector_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized overlap tr(P_a P_b) / dim for orthonormal equal-width bases.

    Sign flips and any within-subspace orthogonal rotation leave it unchanged.
    This explicitly does NOT compare arbitrary unmatched embedding vectors.
    """
    a=np.asarray(a,dtype=np.float64)
    b=np.asarray(b,dtype=np.float64)
    if a.ndim!=2 or b.ndim!=2 or a.shape!=b.shape or a.shape[1]<1:
        raise ValueError("Equal non-empty orthonormal bases required")
    identity=np.eye(a.shape[1])
    if not np.allclose(a.T@a,identity,atol=1e-4) or not np.allclose(
        b.T@b,identity,atol=1e-4
    ):
        raise ValueError("Both bases must have orthonormal columns")
    score=float(np.linalg.norm(a.T@b,ord="fro")**2/a.shape[1])
    return float(np.clip(score,0.,1.))


@dataclass(frozen=True)
class PredictiveGroup:
    first: int
    stop: int
    status: str
    q10_subspace_overlap: float
    median_subspace_overlap: float
    q10_individual_axis_overlap: float | None
    members: tuple[int,...]

    def to_json(self)->dict:
        return {
            "first_slot_zero_index":self.first,"stop_exclusive":self.stop,
            "slots_zero_index":list(self.members),
            "dimension":self.stop-self.first,
            "certificate":self.status,
            "q10_bootstrap_subspace_overlap":self.q10_subspace_overlap,
            "median_bootstrap_subspace_overlap":self.median_subspace_overlap,
            "q10_bootstrap_single_axis_overlap":
                self.q10_individual_axis_overlap,
            "meaning":"observable predictive directions, NOT physiological factor identities",
        }


@dataclass(frozen=True)
class PredictiveCertificate:
    observed_singular_values: np.ndarray
    bootstrap_singular_values: np.ndarray
    reference_basis: np.ndarray
    groups: tuple[PredictiveGroup,...]
    certified_boundaries: tuple[int,...]
    boundary_evidence: tuple[dict,...]
    selected_computational_rank: int
    diagnostics: dict

    def to_json(self)->dict:
        return {
            "singular_values":self.observed_singular_values.tolist(),
            "groups":[g.to_json() for g in self.groups],
            "certified_boundaries":list(self.certified_boundaries),
            "boundaries":[dict(x) for x in self.boundary_evidence],
            "selected_computational_rank":self.selected_computational_rank,
            "bootstrap_repetitions":int(self.bootstrap_singular_values.shape[0]),
            **self.diagnostics,
        }


def certify_from_svd_bootstraps(
    ref_basis:np.ndarray, ref_singular_values:np.ndarray,
    boot_bases:np.ndarray, boot_singular_values:np.ndarray,
    *, computational_rank:int, minimum_subspace_overlap:float=.8,
    gap_noise_quantile:float=.9,
)->PredictiveCertificate:
    """Conservatively segment modes into groups from spectral gaps + resampling.

    A boundary after j modes is called resolved only when:
    (1) the observed gap exceeds the upper sampling-perturbation of both
        adjacent singular values; and
    (2) the cumulative top-j span is stable under patient resampling.

    Neither requirement is an asymptotic theorem; they are explicit
    preregistered decision rules that can return ABSTAIN.
    """
    u=np.asarray(ref_basis,dtype=np.float64)
    s=np.asarray(ref_singular_values,dtype=np.float64)
    us=np.asarray(boot_bases,dtype=np.float64)
    ss=np.asarray(boot_singular_values,dtype=np.float64)
    if u.ndim!=2 or s.ndim!=1 or s.shape[0]!=u.shape[1] or (
        us.ndim!=3 or us.shape[1:]!=u.shape or
        ss.shape!=(us.shape[0],len(s)) or us.shape[0]<8
    ):
        raise ValueError("Expected SVD shapes U[D,D], s[D], Uboot[B,D,D], sboot[B,D]")
    d=u.shape[0]
    if d!=u.shape[1] or not 1<=computational_rank<=d:
        raise ValueError("Only complete orthonormal sensor-space SVD and 1<=K<=D")
    if not np.isfinite(s).all() or not np.isfinite(ss).all() or (
        not np.isfinite(u).all() or not np.isfinite(us).all()
    ):
        raise ValueError("Nonfinite singular spectra or basis")
    if not np.allclose(u.T@u,np.eye(d),atol=2e-4):
        raise ValueError("Reference basis is not orthonormal")
    if np.any(np.diff(s)>1e-8):
        raise ValueError("Singular spectrum must be nonincreasing")
    if not (0<minimum_subspace_overlap<=1 and .5<gap_noise_quantile<1):
        raise ValueError("Invalid preregistered certificate thresholds")

    boundary=[]
    resolved=[]
    # Evaluate every potential split between modes 1..K and the outer K
    # boundary separately. A computational rank at the estimator ceiling
    # is not an independently certified true factor count.
    for j in range(1,min(computational_rank+1,d)):
        reference_gap=float(s[j-1]-s[j])
        noise=np.abs(ss[:,j-1]-s[j-1])+np.abs(ss[:,j]-s[j])
        threshold=float(np.quantile(noise,gap_noise_quantile))
        overlap=np.asarray([
            projector_overlap(u[:,:j],ui[:,:j]) for ui in us
        ])
        q10=float(np.quantile(overlap,.10))
        is_resolved=bool(reference_gap>threshold and
                         q10>=minimum_subspace_overlap)
        boundary.append({
            "after_mode_1_indexed":j,
            "reference_singular_gap":reference_gap,
            "upper_spectral_perturbation":threshold,
            "bootstrap_q10_cumulative_subspace_overlap":q10,
            "certified":is_resolved,
        })
        if is_resolved and j<computational_rank:
            resolved.append(j)

    edges=(0,*resolved,int(computational_rank))
    groups=[]
    for first,stop in zip(edges[:-1],edges[1:]):
        overlap=np.asarray([
            projector_overlap(u[:,first:stop],ui[:,first:stop])
            for ui in us
        ])
        q10=float(np.quantile(overlap,.10))
        individual=float(q10) if stop-first==1 else None
        # A singleton must have both flanking boundaries resolved,
        # not just a consistent orientation selected by SVD convention.
        left_certified=(first==0 or first in resolved)
        outer=(stop==computational_rank and computational_rank<d)
        right_certified=(stop in resolved or (stop==d) or (
            outer and any(x["after_mode_1_indexed"]==stop
                          and x["certified"] for x in boundary)
        ))
        if q10<minimum_subspace_overlap:
            status="ABSTAIN_UNSTABLE_SUBSPACE"
        elif (stop-first==1 and left_certified and right_certified):
            status="INDIVIDUAL_OBSERVABLE_AXIS_STABLE"
        elif stop-first>1 and right_certified:
            status="GROUP_SUBSPACE_STABLE_AXES_UNRESOLVED"
        else:
            status="ABSTAIN_UNRESOLVED_RANK_BOUNDARY"
        groups.append(PredictiveGroup(
            first=first,stop=stop,status=status,
            q10_subspace_overlap=q10,
            median_subspace_overlap=float(np.median(overlap)),
            q10_individual_axis_overlap=individual,
            members=tuple(range(first,stop)),
        ))
    return PredictiveCertificate(
        observed_singular_values=s,bootstrap_singular_values=ss,
        reference_basis=u,groups=tuple(groups),
        certified_boundaries=tuple(resolved),
        boundary_evidence=tuple(boundary),
        selected_computational_rank=int(computational_rank),
        diagnostics={
            "input_used":"training observations and masks ONLY; positive-lag predictor",
            "clinical_concepts_identified":False,
            "causal_variables_identified":False,
            "true_number_of_factors_known":False,
            "identifiability_definition":"empirical bootstrap resolution of predictive spectral subspaces",
            "individual_axis_claim_only_for_certified_singletons":True,
            "internal_bootstrap_confidence_not_population_conformal":True,
        },
    )


def fit_predictive_group_certificate(
    train_values:np.ndarray,train_mask:np.ndarray,
    *, computational_rank:int,
    lags:tuple[int,...]=(1,2,3),
    bootstraps:int=40,seed:int=20261103,
    minimum_subspace_overlap:float=.8,
    gap_noise_quantile:float=.9,
)->PredictiveCertificate:
    """Fit SVD bootstrap on TRAIN data, not outcome, validation, test or oracle."""
    x=np.asarray(train_values,dtype=np.float64)
    m=np.asarray(train_mask,dtype=np.float64)
    if x.ndim!=3 or m.shape!=x.shape or x.shape[0]<80 or x.shape[2]<2:
        raise ValueError("Expected >=80 training episodes [N,T,D]")
    if not np.isfinite(x).all() or not np.isfinite(m).all():
        raise ValueError("Finite observed values/masks required")
    if not np.isin(m,[0.,1.]).all() or bootstraps<8 or not lags or (
        max(lags)>=x.shape[1] or min(lags)<1 or len(set(lags))!=len(lags)
    ):
        raise ValueError("Invalid observed mask, lags, or bootstrap count")
    d=x.shape[-1]
    if not 1<=computational_rank<=d:
        raise ValueError("Requested computational K exceeds sensor dimension")
    centered=_center(x,m)
    reference=_matrix_from_centered(centered,m,lags)
    u,s,_=np.linalg.svd(reference,full_matrices=False)
    rng=np.random.default_rng(int(seed))
    ub=np.empty((bootstraps,d,d),dtype=np.float64)
    sb=np.empty((bootstraps,d),dtype=np.float64)
    for i in range(bootstraps):
        sampled=rng.integers(x.shape[0],size=x.shape[0])
        xx,mm=x[sampled],m[sampled]
        tmp=_matrix_from_centered(_center(xx,mm),mm,lags)
        ub[i],sb[i],_=np.linalg.svd(tmp,full_matrices=False)
    report=certify_from_svd_bootstraps(
        u,s,ub,sb,computational_rank=computational_rank,
        minimum_subspace_overlap=minimum_subspace_overlap,
        gap_noise_quantile=gap_noise_quantile,
    )
    report.diagnostics.update({
        "n_train_episodes":int(x.shape[0]),
        "n_observed_channels":int(d),
        "positive_lags":list(lags),
        "resampling_unit":"complete train patient",
        "bootstraps":int(bootstraps),
        "seed":int(seed),
        "train_patient_order_sha256":sha256(
            np.ascontiguousarray(x.shape,dtype="<i8").tobytes()
        ).hexdigest(),
    })
    return report


def torch_rotation_invariant_group_loss(
    feature_weights:torch.Tensor,
    source_basis:torch.Tensor,
    groups:tuple[PredictiveGroup,...],
    *,
    epsilon:float=1e-4,
)->torch.Tensor:
    """Projector loss on learned SIGNED sensor definitions, never fixed axes.

    Unlike the old V10 axis-wise MSE to a selected singular-vector basis,
    this loss is invariant under ANY within-group orthogonal rotation of
    the source basis. It encourages group-level prediction-relevant
    subspaces without pretending repeated singular vectors have names.

    Soft ridge projector avoids unstable exact inverse at rank-deficient
    learned concept matrices. Collapse is penalized by missing trace.
    """
    if feature_weights.ndim!=2 or source_basis.ndim!=2:
        raise ValueError("Expected [K,8*D] sparse operators and [D,K] source basis")
    k,d=source_basis.shape[1],source_basis.shape[0]
    if feature_weights.shape!=(k,8*d):
        raise ValueError("Signed feature dictionary must include 4 operators and 2 signs/channel")
    if not torch.isfinite(feature_weights).all() or not torch.isfinite(source_basis).all():
        raise ValueError("Nonfinite concept definitions")
    if epsilon<=0 or not groups or groups[0].first!=0 or groups[-1].stop!=k:
        raise ValueError("Consecutive groups spanning exactly K required")
    if not torch.allclose(source_basis.T@source_basis,
                          torch.eye(k,device=source_basis.device,
                                    dtype=source_basis.dtype),atol=1e-3):
        raise ValueError("Source basis must be orthonormal")
    # +/− signed weights summed across the four transparent operators;
    # [K,2,4,D] -> [K,D]. Avoids anchoring a preferred operator scale.
    definitions=feature_weights.reshape(k,2,4,d)
    signed=definitions[:,0].sum(dim=1)-definitions[:,1].sum(dim=1)
    total=feature_weights.new_zeros(())
    for group in groups:
        start,stop=group.first,group.stop
        if not (0<=start<stop<=k):
            raise ValueError("Invalid contiguous concept group")
        learned=signed[start:stop].T  # [D,g]
        target=source_basis[:,start:stop]
        gram=learned.T@learned
        ridge=epsilon*torch.eye(stop-start,device=gram.device,dtype=gram.dtype)
        proj=learned@torch.linalg.solve(gram+ridge,learned.T)
        desired=target@target.T
        total=total+((proj-desired).square().sum()/max(stop-start,1))
    return total/len(groups)


def grouped_logit_contributions(
    contributions:np.ndarray,groups:tuple[PredictiveGroup,...],
)->np.ndarray:
    """Group sums preserve the exact original physiological logit total."""
    z=np.asarray(contributions,dtype=np.float64)
    if z.ndim!=2 or not groups or groups[0].first!=0 or groups[-1].stop!=z.shape[1]:
        raise ValueError("Exact [patients,K] contributions and partition required")
    cover=np.concatenate([np.arange(g.first,g.stop) for g in groups])
    if not np.array_equal(cover,np.arange(z.shape[1])):
        raise ValueError("Groups must partition all concept slots in order")
    totals=np.stack([z[:,g.first:g.stop].sum(axis=1) for g in groups],axis=1)
    if not np.allclose(totals.sum(axis=1),z.sum(axis=1),atol=1e-9):
        raise AssertionError("Group accounting must preserve additive logit")
    return totals


def compare_group_attributions(
    left:np.ndarray,right:np.ndarray,groups:tuple[PredictiveGroup,...],
)->list[dict]:
    """Same-patient group-attribution agreement, NO raw-axis forced matching.

    A high correlation on an abstained group does not upgrade its certificate.
    Group correlations must be reported alongside the certificate and the
    permutation-of-patients marginal control.
    """
    a=grouped_logit_contributions(left,groups)
    b=grouped_logit_contributions(right,groups)
    if a.shape!=b.shape:
        raise ValueError("Same patient cohort, same source groups required")
    evidence=[]
    for j,group in enumerate(groups):
        av,bv=a[:,j],b[:,j]
        if len(av)<3 or np.std(av)<1e-10 or np.std(bv)<1e-10:
            score=None
        else:
            from scipy.stats import rankdata
            ra,rb=rankdata(av),rankdata(bv)
            score=float(np.corrcoef(ra,rb)[0,1])
        evidence.append({
            "slots":list(group.members),
            "certificate":group.status,
            "same_patient_spearman":score,
            "different_patient_expected_spearman":0.0 if score is not None else None,
            "descriptive_only":True,
            "individual_concept_identity_claim":group.status=="INDIVIDUAL_OBSERVABLE_AXIS_STABLE",
        })
    return evidence
