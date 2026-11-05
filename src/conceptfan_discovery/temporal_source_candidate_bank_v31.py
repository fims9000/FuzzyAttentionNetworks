"""A source-group temporal feature bank with explicit missingness support.

The source grouping, orientation, per-channel normalization and temporal
split are determined BEFORE considering target outcomes. This module does
not learn causal/physiological source identities or use hidden simulator
states. A source feature remains interpretable as a MEASURED temporal
statistic with a listed group of sensors and fixed time interval.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass

import numpy as np
from scipy.special import expit

from conceptfan_discovery.auto_measured_concept_proposals import (
    MeasuredGroup,measured_group_concept,
)


@dataclass(frozen=True)
class TemporalCandidateSpec:
    candidate_id: str
    source_group_id: str
    sensor_indices: tuple[int,...]
    statistic: str
    time_intervals: tuple[tuple[int,int],...]
    source_provenance: str = "TRAIN_ONLY_CORROBORATED_MEASURED_SOURCE"
    concept_source_physical_identity_verified: bool = False
    candidate_definition_uses_outcomes: bool = False
    candidate_definition_uses_latent_oracle: bool = False

    def as_dict(self):
        return asdict(self)


def candidate_bank(groups:tuple[MeasuredGroup,...])->tuple[TemporalCandidateSpec,...]:
    entries=[]
    for group in groups:
        for name in ("static_mean","half_contrast"):
            intervals=((0,16),) if name=="static_mean" else ((0,8),(8,16))
            entries.append(TemporalCandidateSpec(
                candidate_id=group.candidate_id+"::"+name,
                source_group_id=group.candidate_id,
                sensor_indices=group.sensor_indices,
                statistic=name,
                time_intervals=intervals,
            ))
    return tuple(entries)


def measured_temporal_activation(
    group:MeasuredGroup,
    values:np.ndarray,
    observed:np.ndarray,
    *,
    statistic:str,
)->tuple[np.ndarray,np.ndarray]:
    """Two strictly named statistics, no masked payload or labels read.

    For half_contrast, each half must have observed source readings;
    per-episode support is the *less available* half. Support below .50
    forces the returned evidence to neutral 0.5 (caller must abstain).
    """
    if statistic=="static_mean":
        return measured_group_concept(group,values,observed)
    if statistic!="half_contrast":
        raise ValueError("Unknown candidate temporal statistic")
    x=np.asarray(values,dtype=np.float64)
    m=np.asarray(observed)
    if (x.ndim!=3 or x.shape!=m.shape or x.shape[1]!=16
            or x.shape[2]<=max(group.sensor_indices)):
        raise ValueError("Expected aligned 16-step, masked multichannel sample")
    if not np.logical_or(m==0,m==1).all():
        raise ValueError("Mask must be binary")
    m=m.astype(bool)
    if not np.isfinite(x[m]).all():
        raise ValueError("Nonfinite observed payload")
    cols=np.asarray(group.sensor_indices)
    means=np.asarray(group.train_means,dtype=np.float64)[None,None,:]
    stds=np.asarray(group.train_stds,dtype=np.float64)[None,None,:]
    signs=np.asarray(group.sensor_signs,dtype=np.float64)[None,None,:]
    half_mean=[]
    half_support=[]
    for left,right in ((0,8),(8,16)):
        values_slice=x[:,left:right,:][:,:,cols]
        mask_slice=m[:,left:right,:][:,:,cols]
        whitened=np.where(
            mask_slice,((values_slice-means)/stds)*signs,0.0)
        n=mask_slice.sum(axis=(1,2))
        half_mean.append(whitened.sum(axis=(1,2))/np.maximum(1,n))
        half_support.append(n/(8*len(cols)))
    support=np.minimum(half_support[0],half_support[1])
    delta=.5*(half_mean[0]-half_mean[1])
    q=.5+support*(expit(delta)-.5)
    q=np.where(support>=.5,q,.5)
    if not (np.isfinite(q).all() and np.isfinite(support).all()
            and (support>=0).all() and (support<=1).all()
            and (q>=0).all() and (q<=1).all()):
        raise AssertionError("Invalid temporal measured concept")
    return q,support
