"""V54: outcome-blind discovery of *different* measured operator families.

Learns a sparse library of lagged signed and even-magnitude concepts from
unassigned channels, with subject-fold support, negative same-time controls,
and TRAIN-only normalization. Concepts are statistical, never physical names.
No risk labels, expert oracle, hidden factors, or future-at-t sensor samples.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import combinations
import json
import numpy as np
from scipy.special import expit

from .lagged_operator_concepts_v53 import (
    _check, discover_lagged_concepts_v53, infer_lagged_concept_v53,
)

OP_SIGNED="SIGNED_LAG"
OP_EVEN="EVEN_ENERGY_LAG"
ALLOWED=(OP_SIGNED,OP_EVEN)


@dataclass(frozen=True)
class MeasuredOperatorConceptV54:
    candidate_id: str
    family: str
    source_channel: int
    readout_channel: int
    lag_steps: int
    readout_sign: int
    source_center: float
    source_scale: float
    readout_center: float
    readout_scale: float
    min_fold_signed_correlation: float
    max_abs_contemporaneous_correlation: float
    train_subjects: int
    measurement_scope: str = "STATISTICAL_SENSOR_ASSOCIATION"
    supports_physical_causal_name: bool = False
    discovery_uses_target_y: bool = False
    uses_hidden_generator_source: bool = False
    uses_future_measurements_at_time_t: bool = False

    def passport(self):
        return asdict(self)


@dataclass(frozen=True)
class OperatorLibraryDiscoveryV54:
    candidates: tuple[MeasuredOperatorConceptV54,...]
    operator_families: tuple[str,...]
    train_subjects: int
    known_channels: tuple[int,...]
    max_candidates: int
    all_proposals_outcome_blind: bool = True
    unknown_sources_not_physically_identified: bool = True

    def manifest(self):
        return asdict(self)


def _transform(x,op):
    if op==OP_SIGNED:return x
    if op==OP_EVEN:return x*x
    raise ValueError("Unknown registered concept operator")


def _pair_corr(x,m,ids,i,j,lag,transform,min_cells):
    if lag==0:
        a=x[ids,:,i];b=x[ids,:,j]
        valid=m[ids,:,i]&m[ids,:,j]
    else:
        a=x[ids,:-lag,i];b=x[ids,lag:,j]
        valid=m[ids,:-lag,i]&m[ids,lag:,j]
    if int(valid.sum())<min_cells:return None
    a=_transform(a[valid],transform)
    b=_transform(b[valid],transform)
    a=a-a.mean();b=b-b.mean()
    norm=np.sqrt(np.dot(a,a)*np.dot(b,b))
    if norm<1.e-10:return None
    return float(np.clip(np.dot(a,b)/norm,-1.,1.))


def _stats(x,m,j,family):
    val=_transform(x[:,:,j][m[:,:,j]],family)
    if not len(val):raise ValueError("Missing TRAIN observations")
    return float(val.mean()),float(max(val.std(),.1))


def discover_operator_library_v54(
    train_values,train_mask,*,known_channels=(0,1,2,3),
    seed=20261113,max_lag=3,max_candidates=3,
    min_abs_corr=.65,min_gain_over_contemporaneous=.30,min_fold_cells=120
):
    """Register three TRAIN-fold stable disjoint sensor pairs.

    Candidate operator families compete on the same TRAIN data. The generator
    does not use target Y. Only disjoint pairs admitted (max 3 candidates);
    signed-only is included for a matched V53-like ablation.
    """
    x,m=_check(train_values,train_mask)
    n,t,d=x.shape
    known=tuple(sorted(known_channels))
    if (n<300 or len(known)!=len(set(known)) or
        any(int(i)!=i or i<0 or i>=d for i in known) or
        not(1<=max_lag<=min(6,t-2)) or not(1<=max_candidates<=8) or
        not(.4<=min_abs_corr<=.95) or not(0<=min_gain_over_contemporaneous<=.8)
        or min_fold_cells<30):
        raise ValueError("Bad TRAIN discovery parameters")
    remaining=tuple(i for i in range(d) if i not in known)
    folds=np.array_split(np.random.default_rng(seed).permutation(n),3)
    possibilities=[]
    for i,j in combinations(remaining,2):
        for family in ALLOWED:
            same=[_pair_corr(x,m,ids,i,j,0,family,min_fold_cells)
                  for ids in folds]
            if any(z is None for z in same):continue
            zero=max(abs(z) for z in same)
            for source,readout in ((i,j),(j,i)):
                for lag in range(1,max_lag+1):
                    c=[_pair_corr(x,m,ids,source,readout,lag,family,min_fold_cells)
                       for ids in folds]
                    if any(z is None for z in c):continue
                    sign=np.sign(c)
                    if not np.all(sign==sign[0]):continue
                    corr=min(abs(z) for z in c)
                    gain=corr-zero
                    if corr<min_abs_corr or gain<min_gain_over_contemporaneous:
                        continue
                    possibilities.append((gain,corr,source,readout,lag,
                                          1 if c[0]>0 else -1,zero,family))
    # Same fold and same operator conditions; prefer SIGNED on exact ties only.
    possibilities.sort(key=lambda p:(-p[0],-p[1],p[2],p[3],p[4],ALLOWED.index(p[7])))
    candidates=[];used=set()
    for gain,corr,i,j,lag,sign,zero,family in possibilities:
        if i in used or j in used:continue
        center1,scale1=_stats(x,m,i,family)
        center2,scale2=_stats(x,m,j,family)
        fingerprint=json.dumps(
            [family,i,j,lag,sign,round(corr,8),round(zero,8)],
            separators=(",",":"))
        cid="measured-"+sha256(fingerprint.encode()).hexdigest()[:14]
        candidates.append(MeasuredOperatorConceptV54(
            candidate_id=cid,family=family,source_channel=i,
            readout_channel=j,lag_steps=lag,readout_sign=sign,
            source_center=center1,source_scale=scale1,
            readout_center=center2,readout_scale=scale2,
            min_fold_signed_correlation=sign*corr,
            max_abs_contemporaneous_correlation=zero,train_subjects=n
        ))
        used.update((i,j))
        if len(candidates)==max_candidates:break
    return OperatorLibraryDiscoveryV54(tuple(candidates),ALLOWED,n,
                                      known,max_candidates)


def infer_operator_concept_v54(candidate,values,mask,*,last_steps=6):
    """Every q[t] uses only observed source[t-lag], readout[t].

    EVEN_ENERGY_LAG has exact parity invariance to sign inversion of both
    selected measured channels, provided no other nuisance transformation.
    This is *not* a physical sensor trust guarantee.
    """
    x,m=_check(values,mask)
    n,t,d=x.shape
    i,j,lag=candidate.source_channel,candidate.readout_channel,candidate.lag_steps
    if (candidate.family not in ALLOWED or i==j or not(0<=i<d and 0<=j<d)
        or not(1<=lag<t) or candidate.readout_sign not in (-1,1)
        or candidate.source_scale<=0 or candidate.readout_scale<=0 or
        not(1<=last_steps<=t) or not np.isfinite([
            candidate.source_center,candidate.source_scale,
            candidate.readout_center,candidate.readout_scale]).all()):
        raise ValueError("Invalid frozen operator passport")
    paired=m[:,:-lag,i]&m[:,lag:,j]
    a=np.where(m[:,:-lag,i],x[:,:-lag,i],0.)
    b=np.where(m[:,lag:,j],x[:,lag:,j],0.)
    at=(_transform(a,candidate.family)-candidate.source_center)/candidate.source_scale
    bt=(_transform(b,candidate.family)-candidate.readout_center)/candidate.readout_scale
    value=expit(.5*(at+candidate.readout_sign*bt))
    traj=np.full((n,t),.5,np.float64)
    traj[:,lag:]=np.where(paired,value,.5)
    avail=np.zeros((n,t),bool)
    avail[:,lag:]=paired
    recent=avail[:,-last_steps:]
    nobs=recent.sum(axis=1)
    mean=(traj[:,-last_steps:]-.5).sum(axis=1)/np.maximum(nobs,1)
    support=nobs/last_steps
    q=np.clip(.5+support*mean,0.,1.)
    return {"trajectory":traj,"available":avail,"summary":q,
            "support":support,"statistical_not_physical":True,
            "operator_family":candidate.family}


def fingerprint_v54(proposal):
    return sha256(json.dumps(proposal.manifest(),sort_keys=True,
              ensure_ascii=True,separators=(",",":")).encode()).hexdigest()
