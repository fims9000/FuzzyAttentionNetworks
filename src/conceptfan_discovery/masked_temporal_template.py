"""TRAIN-only masked rank-one temporal templates for measured source groups.

Classical weighted low-rank WLS / PCA: no outcome labels, oracle source
amplitudes, validation measurements or TEST sensor values fit the basis.
A TEST episode may project only ACTUALLY observed source-time values.
Unmeasured time steps are NEVER treated as observed, and uncertainty
and insufficient support are explicit. Source semantics are observational.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
from hashlib import sha256

import numpy as np
from scipy.special import expit

from conceptfan_discovery.auto_measured_concept_proposals import MeasuredGroup


@dataclass(frozen=True)
class MaskedRank1Template:
    group_id: str
    sensor_indices: tuple[int,...]
    phi: tuple[float,...]
    train_residual_variance: float
    train_amplitude_variance: float
    shrinkage_ridge: float
    number_train_subjects: int
    number_train_observed_source_times: int
    deterministic_first_four_positive: bool
    template_used_target_outcomes: bool = False
    template_used_hidden_generator_factors: bool = False
    semantic_physical_identity_certified: bool = False

    def as_dict(self):
        return asdict(self)


def _source_time_matrix(
    group:MeasuredGroup, values:np.ndarray, mask:np.ndarray
)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    x=np.asarray(values,dtype=np.float64)
    m=np.asarray(mask)
    cols=np.asarray(group.sensor_indices,dtype=np.int64)
    if (x.ndim!=3 or x.shape!=m.shape or x.shape[1]!=16
        or x.shape[2]<=int(cols.max())
        or len(cols)<2 or len(cols)!=len(group.sensor_signs)):
        raise ValueError("Expected 16 historical steps and matching source group")
    if not np.logical_or(m==0,m==1).all():
        raise ValueError("Source availability mask must be binary")
    m=m.astype(bool)
    if not np.isfinite(x[m]).all():
        raise ValueError("Nonfinite OBSERVED readouts")
    xx=x[:,:,cols]
    observed=m[:,:,cols]
    whitened=(
        (xx-np.asarray(group.train_means)[None,None,:])
        /np.asarray(group.train_stds)[None,None,:]
    )*np.asarray(group.sensor_signs)[None,None,:]
    n=observed.sum(axis=-1)
    y=np.where(observed,whitened,0.).sum(axis=-1)/np.maximum(1,n)
    # Weight W=1 for a timepoint with >=1 source measurement; cannot
    # multiply a "missing" 0 as though it were an observed 0 value.
    w=(n>0).astype(np.float64)
    y=np.where(w>0,y,0.)
    return y,w,n


def fit_masked_rank1_template(
    group:MeasuredGroup,train_values:np.ndarray,train_mask:np.ndarray,
    *,iterations:int=5
)->MaskedRank1Template:
    """Fixed unsupervised alternating least-squares schedule."""
    y,w,n=_source_time_matrix(group,train_values,train_mask)
    if len(y)<200 or iterations!=5 or int(w.sum())<300:
        raise ValueError("TRAIN-only template requires sufficient source observations")
    if (w.sum(axis=1)>0).sum()<150:
        raise ValueError("Too few TRAIN episodes with any source")
    filled=np.where(w>0,y,0.)
    _,_,right=np.linalg.svd(filled,full_matrices=False)
    phi=np.asarray(right[0],dtype=np.float64)
    phi*=np.sqrt(16)/np.linalg.norm(phi)
    if phi[:4].mean()<0:
        phi=-phi
    for _ in range(iterations):
        denom=(w*(phi[None,:]**2)).sum(axis=1)
        amp=(w*filled*phi[None,:]).sum(axis=1)/np.maximum(1e-10,denom)
        numerator=(w*filled*amp[:,None]).sum(axis=0)
        den=(w*amp[:,None]**2).sum(axis=0)
        phi=numerator/np.maximum(den,1e-10)
        norm=np.linalg.norm(phi)
        if norm<1e-7:
            raise RuntimeError("Source temporal direction collapsed")
        phi*=np.sqrt(16)/norm
        if phi[:4].mean()<0:
            phi=-phi
    information=(w*(phi[None,:]**2)).sum(axis=1)
    amp=(w*filled*phi[None,:]).sum(axis=1)/np.maximum(1e-9,information)
    residual=(w*((filled-amp[:,None]*phi[None,:])**2))
    variance=float(residual.sum()/max(1,w.sum()-len(phi)-1))
    prior=float(np.var(amp[w.sum(axis=1)>=8],ddof=0))
    if not np.isfinite(variance) or variance<=0 or not np.isfinite(prior) or prior<=0:
        raise RuntimeError("Cannot estimate measurement uncertainty without TRAIN variation")
    ridge=float(max(.01,variance/max(prior,1e-8)))
    return MaskedRank1Template(
        group_id=group.candidate_id,
        sensor_indices=group.sensor_indices,
        phi=tuple(map(float,phi)),
        train_residual_variance=variance,
        train_amplitude_variance=prior,
        shrinkage_ridge=ridge,
        number_train_subjects=len(y),
        number_train_observed_source_times=int(w.sum()),
        deterministic_first_four_positive=True,
    )


def project_masked_template(
    template:MaskedRank1Template,
    group:MeasuredGroup,
    x:np.ndarray,m:np.ndarray,
    *,
    min_information_fraction:float=.50,
)->dict:
    """Per-patient WLS amplitude, shrinkage, observed support and uncertainty."""
    if template.group_id!=group.candidate_id or template.sensor_indices!=group.sensor_indices:
        raise ValueError("Template and group identifiers do not match")
    if not .3<=min_information_fraction<=.95:
        raise ValueError("Unsupported frozen information threshold")
    y,w,n=_source_time_matrix(group,x,m)
    phi=np.asarray(template.phi,dtype=np.float64)
    if phi.shape!=(16,):
        raise ValueError("TRAIN template time dimension differs")
    info=(w*(phi[None,:]**2)).sum(axis=1)
    total=float(np.dot(phi,phi))
    observed_information=info/total
    ridge=template.shrinkage_ridge
    amp=(w*y*phi[None,:]).sum(axis=1)/np.maximum(1e-10,info+ridge)
    reliability=info/np.maximum(1e-10,info+ridge)
    # No zero-filled unobserved features enter the WLS numerator or loss.
    q=.5+reliability*(expit(amp)-.5)
    eligible=observed_information>=min_information_fraction
    q=np.where(eligible,q,.5)
    support=np.where(eligible,np.clip(observed_information,0.,1.),0.)
    stderr=np.sqrt(template.train_residual_variance/
                   np.maximum(info+ridge,1e-10))
    return {
        "q":q,
        "source_support":support,
        "observed_template_information_fraction":np.clip(observed_information,0,1),
        "measured_source_times":w.sum(axis=1),
        "source_channel_measurements":n.sum(axis=1),
        "amplitude_estimate":amp,
        "amplitude_standard_error":stderr,
        "template_reliability":reliability,
        "sufficient_observations":eligible,
        "source_temporal_template_sha256":sha256(
            np.asarray(phi,dtype=np.float64).tobytes()).hexdigest(),
        "no_hidden_source_or_outcome_labels_used":True,
    }


def template_concept(
    template:MaskedRank1Template,
    group:MeasuredGroup,
    x:np.ndarray,m:np.ndarray,
)->tuple[np.ndarray,np.ndarray]:
    z=project_masked_template(template,group,x,m)
    return z["q"],z["source_support"]
