"""V28: named, measurement-backed candidate slot without mutating prior concepts.

A separately named risk source may be admitted on disjoint, TRAIN-only
validation environments. This is predictive evidence, NOT semantic or causal
certification. The original concept head's bias, weights and coordinates
remain literally unchanged: new logit = old logit + gamma*(candidate - 0.5).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit

from fan.concept.completeness_audit import _paired_bootstrap_ci, _per_sample_logloss


SOURCE_PROVENANCE = {
    "name": "measured_sensor_pair_4_5_candidate",
    "sensor_indices": [4, 5],
    "operator": "0.5 + observed_fraction * (sigmoid(mean_of_available_measurements) - 0.5)",
    "causal_physical_identity_certified": False,
    "predefined_measurement_map": True,
    "uses_only_observed_past_values": True,
    "new_slot_changes_existing_concept_coordinates": False,
}


@dataclass(frozen=True)
class ValidationReceipt:
    environment: str
    n_subjects: int
    mean_paired_logloss_gain: float
    ci95: tuple[float, float]
    supported_fraction: float
    passed: bool


@dataclass(frozen=True)
class AdmissionReceipt:
    admitted: bool
    candidate_coefficient: float
    min_required_gain: float
    min_subject_coverage: float
    environments: tuple[ValidationReceipt, ...]
    reason: str
    only_known_environment_predictive_validity: bool = True

    def as_dict(self) -> dict:
        data=asdict(self)
        data["source_provenance"]=SOURCE_PROVENANCE
        return data


def measured_candidate(
    values: np.ndarray, recorded: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Predeclared scalar concept formed only from actually measured source pair."""
    x=np.asarray(values,dtype=np.float64)
    m=np.asarray(recorded)
    if x.ndim!=3 or x.shape!=m.shape or x.shape[-1]!=6 or x.shape[1]<2:
        raise ValueError("Expected 6-channel measured history and aligned binary mask")
    if not np.logical_or(m==0,m==1).all():
        raise ValueError("Recorded mask must be binary")
    obs=m.astype(bool)
    if not np.isfinite(x[obs]).all():
        raise ValueError("Nonfinite observed values")
    pair=obs[:,:,4:6]
    seen=np.where(pair,x[:,:,4:6],0.0)
    count=pair.sum(axis=(1,2))
    avg=seen.sum(axis=(1,2))/np.maximum(1,count)
    support=count/pair[0].size
    candidate=0.5+support*(expit(avg)-0.5)
    if (not np.isfinite(candidate).all()
            or np.any(candidate<0) or np.any(candidate>1)):
        raise AssertionError("Invalid bounded candidate activation")
    return candidate, support


def train_source_coefficient(
    base_train_logit: np.ndarray, candidate_train: np.ndarray, labels: np.ndarray,
    *, penalty: float=0.002
) -> float:
    """Optimize ONE scalar for the source; original K2 weights never refit."""
    logit=np.asarray(base_train_logit,dtype=np.float64).reshape(-1)
    c=np.asarray(candidate_train,dtype=np.float64).reshape(-1)
    y=np.asarray(labels,dtype=np.float64).reshape(-1)
    if (logit.shape!=c.shape or y.shape!=c.shape or len(y)<100
            or not np.isfinite(logit).all() or not np.isfinite(c).all()
            or not np.isin(y,[0.,1.]).all() or len(np.unique(y))!=2
            or not np.logical_and(c>=0,c<=1).all() or penalty<0):
        raise ValueError("Invalid TRAIN-only candidate risk fit")
    x=c-0.5
    def objective(g:float)->float:
        v=logit+g*x
        return float(np.mean(np.logaddexp(0,v)-y*v)+penalty*g*g)
    fitted=minimize_scalar(objective,method="bounded",bounds=(-12.,12.),
                           options={"xatol":1e-9,"maxiter":160})
    if not fitted.success or not np.isfinite(fitted.x):
        raise RuntimeError("Residual candidate risk-head training failed")
    result=float(fitted.x)
    if objective(result)>objective(0.0)+1e-8:
        raise AssertionError("Extra measured source increased TRAIN objective")
    return result


def candidate_logit(
    original_logit: np.ndarray,candidate: np.ndarray,coefficient: float
) -> tuple[np.ndarray,np.ndarray]:
    """Exactly additive logit using original frozen risk plus one named slot."""
    prior=np.asarray(original_logit,dtype=np.float64).reshape(-1)
    c=np.asarray(candidate,dtype=np.float64).reshape(-1)
    if len(prior)!=len(c) or not np.isfinite(prior).all() or not np.isfinite(c).all():
        raise ValueError("Misaligned additive risk components")
    signed_source=coefficient*(c-0.5)
    return prior+signed_source,signed_source


def two_environment_admission(
    coefficient:float,
    *,
    environments: Sequence[tuple[str,np.ndarray,np.ndarray,np.ndarray,np.ndarray]],
    minimum_gain:float=0.002,
    minimum_supported_fraction:float=0.75,
    per_subject_min_source_support:float=0.50,
    bootstrap_replicates:int=399,
    seed:int=20261108,
)->AdmissionReceipt:
    """Only named VALID environments and VALID labels, no TEST inputs or labels.

    Each environment tuple is (name, y, frozen_base_logit, q_candidate,
    measured_source_support). A rejection is not a proof of completeness.
    """
    if len(environments)!=2 or minimum_gain<0 or not 0<=minimum_supported_fraction<=1:
        raise ValueError("Exactly two independent validation environments required")
    if not (0.<per_subject_min_source_support<=1):
        raise ValueError("Unsupported per-subject sensor availability threshold")
    results=[]
    for i,(name,y,base,c,coverage) in enumerate(environments):
        y=np.asarray(y,dtype=np.float64).reshape(-1)
        base=np.asarray(base,dtype=np.float64).reshape(-1)
        c=np.asarray(c,dtype=np.float64).reshape(-1)
        coverage=np.asarray(coverage,dtype=np.float64).reshape(-1)
        if (len(y)<100 or not(len(y)==len(base)==len(c)==len(coverage))
            or not np.isin(y,[0,1]).all() or len(np.unique(y))<2
            or not np.isfinite(base).all() or not np.isfinite(c).all()
            or not np.isfinite(coverage).all() or not np.logical_and(coverage>=0,coverage<=1).all()):
            raise ValueError("Invalid disjoint validation environment")
        admitted_source=coverage>=per_subject_min_source_support
        proposal,_=candidate_logit(base,c,coefficient)
        # Compare all VALID subjects, with safe baseline fallback when a
        # candidate cannot be measured; no selective-subset cherry-picking.
        proposal=np.where(admitted_source,proposal,base)
        gain=_per_sample_logloss(y,np.clip(expit(base),1e-8,1-1e-8))-(
            _per_sample_logloss(y,np.clip(expit(proposal),1e-8,1-1e-8)))
        ci=_paired_bootstrap_ci(gain,seed=seed+i,replicates=bootstrap_replicates)
        avg=float(np.mean(gain))
        fraction=float(np.mean(admitted_source))
        passed=bool(avg>=minimum_gain and ci[0]>0 and fraction>=minimum_supported_fraction)
        results.append(ValidationReceipt(str(name),int(len(y)),avg,ci,fraction,passed))
    accepted=all(r.passed for r in results)
    reason=("predictive in BOTH supplied validation environments; no semantic certification"
            if accepted else "ABSTAIN: candidate lacks independent predictive support in all supplied environments")
    return AdmissionReceipt(
        accepted,float(coefficient),float(minimum_gain),
        float(minimum_supported_fraction),tuple(results),reason,
    )


def isolated_source_prediction(
    base_logit: np.ndarray,
    candidate: np.ndarray,
    sensor_support: np.ndarray,
    *,
    coefficient:float,
    globally_admitted:bool,
    minimum_subject_support:float=0.5,
) -> dict:
    """Preserve base concept weights and add independently named source only."""
    base=np.asarray(base_logit,dtype=np.float64).reshape(-1)
    src=np.asarray(candidate,dtype=np.float64).reshape(-1)
    support=np.asarray(sensor_support,dtype=np.float64).reshape(-1)
    if not(len(base)==len(src)==len(support)):
        raise ValueError("Source and frozen risk must be patient-aligned")
    eligible=support>=minimum_subject_support
    extended,source_signed=candidate_logit(base,src,coefficient)
    forced=np.where(eligible,extended,base)
    admitted=bool(globally_admitted)
    used=eligible & admitted
    gated=np.where(used,extended,base)
    np.testing.assert_array_equal(gated[~used],base[~used])
    np.testing.assert_allclose(gated,base+np.where(used,source_signed,0.0),
                               rtol=0,atol=5e-15)
    return {"base_logit":base,"forced_logit":forced,"gated_logit":gated,
            "used_source":used,"source_supported":eligible,
            "source_signed_contribution":np.where(used,source_signed,0.0),
            "source_support_fraction":float(np.mean(eligible)),
            "source_usage_fraction":float(np.mean(used)),
            "frozen_base_unchanged":True}
