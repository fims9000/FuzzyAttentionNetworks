"""Train-only missingness adequacy checks before clinical lag-rank analysis.

Never treat absence of time-adjacent measurements as a zero covariance.
Select a disclosed jointly observable subset of sensors *without* labels
or planted concepts; report all channels discarded for insufficient support.
This subset is only an exploratory scope, never a clinical ontology.
"""
from __future__ import annotations

import numpy as np


def aggregate_hours(
    values: np.ndarray, mask: np.ndarray, hours: int,
) -> tuple[np.ndarray,np.ndarray]:
    x=np.asarray(values,dtype=np.float64)
    m=np.asarray(mask,dtype=np.float64)
    if x.ndim!=3 or x.shape!=m.shape or hours<1 or x.shape[1]//hours<4:
        raise ValueError("Expected matched hourly [N,T,D] with at least four bins")
    count=x.shape[1]//hours
    x=x[:,:count*hours].reshape(x.shape[0],count,hours,x.shape[2])
    m=m[:,:count*hours].reshape(m.shape[0],count,hours,m.shape[2])
    valid=m.sum(axis=2)
    averaged=np.divide((x*m).sum(axis=2),valid,
                       out=np.zeros((x.shape[0],count,x.shape[3])),where=valid>0)
    return averaged,(valid>0).astype(np.float64)


def joint_pair_counts(mask:np.ndarray,lags:tuple[int,...]=(1,2,3))->np.ndarray:
    m=np.asarray(mask,dtype=np.float64)
    if m.ndim!=3 or min(lags)<1 or max(lags)>=m.shape[1]:
        raise ValueError("Invalid observation mask or temporal lags")
    return np.stack([
        m[:,:-lag,:].reshape(-1,m.shape[-1]).T@
        m[:,lag:,:].reshape(-1,m.shape[-1])
        for lag in lags
    ])


def support_selected_channels(
    mask:np.ndarray,
    names:list[str],
    *,
    lags:tuple[int,...]=(1,2,3),
    min_pairs:int=20,
)->dict:
    """Greedy deterministic label-free joint-temporal-observability subset.

    Starts with strongest own-lag measured channel; then adds candidate
    channels only when their positive-lag pairs with ALL accepted channels
    are reliably observed in both directions. Does not lower min_pairs.
    """
    m=np.asarray(mask,dtype=np.float64)
    if m.ndim!=3 or len(names)!=m.shape[-1] or min_pairs<1:
        raise ValueError("Mask/channel mapping or min pair count invalid")
    pair=joint_pair_counts(m,lags)
    diag=np.stack([np.diag(x) for x in pair])
    score=diag.min(axis=0)
    # A channel must actually have temporal self pairs in the interval.
    candidates=[int(j) for j in np.argsort(-score,kind="stable") if score[j]>=min_pairs]
    accepted=[]
    for j in candidates:
        if all(
            np.all(pair[:,j,i]>=min_pairs) and np.all(pair[:,i,j]>=min_pairs)
            for i in accepted
        ):
            accepted.append(j)
    if accepted:
        subset=pair[:,accepted,:][:,:,accepted]
        fraction=float((subset>=min_pairs).mean())
        smallest=float(subset.min())
    else:
        fraction=0.
        smallest=0.
    return {
        "selected_indices":accepted,
        "selected_channels":[str(names[j]) for j in accepted],
        "excluded_channels":[str(v) for j,v in enumerate(names) if j not in accepted],
        "channels_initial":len(names),
        "channels_selected":len(accepted),
        "min_pairs":int(min_pairs),
        "smallest_pair_count_within_selected":smallest,
        "selected_pair_fraction_above_minimum":fraction,
        "min_own_lag_pair_count_per_channel":{
            str(name):int(score[j]) for j,name in enumerate(names)
        },
        "all_selected_joint_lag_pairs_estimable":bool(
            len(accepted)>=2 and np.all(pair[:,accepted,:][:,:,accepted]>=min_pairs)
        ),
        "warning":"Selection is based on availability only; informative missingness remains confounding",
    }


def support_selected_channels_across_subsamples(
    masks:tuple[np.ndarray,...],
    names:list[str],
    *,
    lags:tuple[int,...]=(1,2,3),
    min_pairs:int=20,
)->dict:
    """Select channels with joint temporal support in EVERY stated subset.

    Mask-only selection cannot guarantee independent rank inference because
    the rank audit subsets also influenced feasibility. It does, however,
    prevent false numerical covariance estimates for unsupported sensor
    pairs. No patient labels or observed values enter this feature filter.
    """
    if len(masks)<2 or min_pairs<1:
        raise ValueError("Need at least two fixed training subsamples")
    pair_counts=[]
    for mask in masks:
        m=np.asarray(mask,dtype=np.float64)
        if m.ndim!=3 or m.shape[-1]!=len(names):
            raise ValueError("Every mask must use identical ordered channels")
        pair_counts.append(joint_pair_counts(m,lags))
    conservative=np.minimum.reduce(pair_counts)
    per_channel=np.stack([np.diag(a) for a in conservative])
    diagonal_min=per_channel.min(axis=0)
    candidates=[
        int(j) for j in np.argsort(-diagonal_min,kind="stable")
        if diagonal_min[j]>=min_pairs
    ]
    chosen=[]
    for j in candidates:
        if all(
            np.all(conservative[:,j,i]>=min_pairs)
            and np.all(conservative[:,i,j]>=min_pairs)
            for i in chosen
        ):
            chosen.append(j)
    return {
        "selection_mode":"observable_sensors_supported_in_each_disjoint_train_subsample",
        "subsamples":len(masks),
        "selected_indices":chosen,
        "selected_channels":[str(names[j]) for j in chosen],
        "excluded_channels":[str(v) for j,v in enumerate(names) if j not in chosen],
        "selected_n":len(chosen),
        "min_joint_temporal_pairs_in_each_subsample":int(min_pairs),
        "minimum_pair_count_across_subsamples":
            float(conservative[:,chosen,:][:,:,chosen].min()) if chosen else 0.,
        "all_subsamples_joints_adequate":bool(
            len(chosen)>=2 and
            np.all(conservative[:,chosen,:][:,:,chosen]>=min_pairs)
        ),
        "interpretation":"measurement feasibility only; no clinical ontology or causal assertion",
    }
