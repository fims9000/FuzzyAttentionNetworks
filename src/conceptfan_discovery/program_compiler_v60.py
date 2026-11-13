"""ConceptFAN V60: typed, causal, outcome-blind measurement program compiler.

Generalizes frozen V54's lag-only source mining to masked single-sensor
levels, trends, volatility, differential energy, high-events, pair contrasts
and same-time coactivity. All 24 expressions are declared *before* labels.
No encoded HR, outcome Y, hidden simulator source or future measurements.

A program is an observational statistical proxy, NOT a physiological,
causal, medically certified or externally authenticated concept.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
from hashlib import sha256
from itertools import combinations
import json

import numpy as np

UNARY=("LEVEL_LAST4","LEVEL_16","TREND_8",
       "VOLATILITY_8","ENERGY_8","HIGH_EVENT_8")
PAIR=("CONTRAST_4","COACTIVITY_8")
MIN_FOLD_COVERAGE=.50
MIN_FOLD_STD=.025
MIN_FOLD_EVALUABLE=25
MAX_REDUNDANCY=.995
MAX_PROGRAMS=24
WINDOW_LENGTH=16
RAW_INPUT_CHANNELS=("PULSE","RESP","SPO2")


@dataclass(frozen=True)
class MeasuredProgramV60:
    candidate_id:str
    operator:str
    channels:tuple[int,...]
    channel_names:tuple[str,...]
    train_q25:tuple[float,...]
    train_median:tuple[float,...]
    train_q75:tuple[float,...]
    value_center:float
    value_scale:float
    train_episodes:int
    train_subjects:int
    train_episode_coverage:float
    fold_stds:tuple[float,...]
    fold_coverages:tuple[float,...]
    temporal_no_future:bool=True
    discovery_uses_outcome_y:bool=False
    physical_or_causal_identity_proven:bool=False
    external_device_authenticated:bool=False
    expression_scope:str="STATISTICAL_MASKED_MEASUREMENT_PROGRAM"

    def passport(self):
        return asdict(self)


@dataclass(frozen=True)
class ProgramBankV60:
    candidates:tuple[MeasuredProgramV60,...]
    candidate_operators:tuple[str,...]
    channels:tuple[str,...]
    train_subjects:int
    train_episodes:int
    screened_programs:int
    rejected_low_variation_or_coverage:int
    rejected_redundant:int
    max_programs:int
    all_proposals_outcome_blind:bool=True
    trained_on_real_physically_attested_source:bool=False

    def manifest(self):
        return asdict(self)


def _checked_inputs(values,mask):
    x=np.asarray(values,dtype=np.float64)
    m=np.asarray(mask)
    if (x.ndim!=3 or x.shape[1:]!=(WINDOW_LENGTH,3)
        or m.shape!=x.shape
        or not np.logical_or(m==0,m==1).all()
        or not np.isfinite(x[m.astype(bool)]).all()):
        raise ValueError("Expected masked observed [N,16,3], binary mask, finite observed")
    return np.where(m.astype(bool),x,0.),m.astype(bool)


def _nanobs(x,m,begin,end,j):
    v=x[:,begin:end,j]
    a=m[:,begin:end,j]
    count=a.sum(axis=1)
    avg=np.where(a,v,0.).sum(axis=1)/np.maximum(count,1)
    return avg,count/(end-begin),count


def _result(v,support,require=.5):
    v=np.asarray(v,np.float64)
    support=np.asarray(support,np.float64)
    if np.any(~np.isfinite(v)) or np.any(~np.isfinite(support)):
        raise AssertionError("Program emitted invalid finite evidence")
    available=(support>=require)
    return np.where(available,v,0.),np.where(available,support,0.)


def _raw_operator(x,m,operator,channels,medians,q75):
    """Vectorized candidate value and observed support per 16-second episode."""
    i=channels[0]
    if operator=="LEVEL_LAST4":
        v,s,c=_nanobs(x,m,12,16,i)
        return _result(v,s)
    if operator=="LEVEL_16":
        v,s,c=_nanobs(x,m,0,16,i)
        return _result(v,s)
    if operator=="TREND_8":
        prev,s0,c0=_nanobs(x,m,8,12,i)
        last,s1,c1=_nanobs(x,m,12,16,i)
        return _result(last-prev,np.minimum(s0,s1))
    if operator=="VOLATILITY_8":
        v=x[:,8:16,i]
        a=m[:,8:16,i]
        count=a.sum(axis=1)
        avg=np.where(a,v,0.).sum(axis=1)/np.maximum(count,1)
        dif=np.where(a,(v-avg[:,None])**2,0.)
        std=np.sqrt(dif.sum(axis=1)/np.maximum(count,1))
        return _result(std,count/8.)
    if operator=="ENERGY_8":
        v=x[:,8:16,i]
        a=m[:,8:16,i]
        pairs=a[:,1:]&a[:,:-1]
        energy=np.where(pairs,(v[:,1:]-v[:,:-1])**2,0.)
        count=pairs.sum(axis=1)
        return _result(energy.sum(axis=1)/np.maximum(count,1),count/7.)
    if operator=="HIGH_EVENT_8":
        v=x[:,8:16,i]
        a=m[:,8:16,i]
        count=a.sum(axis=1)
        rate=(a&(v>q75[i])).sum(axis=1)/np.maximum(count,1)
        return _result(rate,count/8.)
    if operator=="CONTRAST_4":
        j=channels[1]
        pair=m[:,12:16,i]&m[:,12:16,j]
        v=x[:,12:16,i]-x[:,12:16,j]
        count=pair.sum(axis=1)
        return _result(np.where(pair,v,0.).sum(axis=1)/np.maximum(count,1),
                       count/4.)
    if operator=="COACTIVITY_8":
        j=channels[1]
        pair=m[:,8:16,i]&m[:,8:16,j]
        simultaneous=pair&(x[:,8:16,i]>medians[i])&(
            x[:,8:16,j]>medians[j])
        count=pair.sum(axis=1)
        return _result(simultaneous.sum(axis=1)/np.maximum(count,1),count/8.)
    raise ValueError("Unknown or unfrozen program operator")


def _programs():
    yield from ((op,(i,)) for op in UNARY for i in range(3))
    yield from ((op,(i,j)) for op in PAIR for i,j in combinations(range(3),2))


def _train_sensor_quantiles(x,m):
    q25=[];median=[];q75=[]
    for j in range(3):
        vals=x[:,:,j][m[:,:,j]]
        if len(vals)<100 or not np.isfinite(vals).all():
            raise ValueError("Not enough observed TRAIN raw sensor values")
        a,b,c=np.quantile(vals,[.25,.50,.75])
        q25.append(float(a));median.append(float(b));q75.append(float(c))
    return tuple(q25),tuple(median),tuple(q75)


def _subject_folds(groups,seed):
    unique=np.unique(groups)
    if len(unique)<12:raise ValueError("Need at least 12 real independent TRAIN subjects")
    rng=np.random.default_rng(seed)
    shuffled=unique[rng.permutation(len(unique))]
    return [np.isin(groups,fold) for fold in np.array_split(shuffled,3)]


def compile_measured_programs_v60(values,mask,train_subject_groups,*,seed=20266013,
                                   max_programs=MAX_PROGRAMS):
    """Propose measured programs *without access to labels*, source ID or Y."""
    x,m=_checked_inputs(values,mask)
    groups=np.asarray(train_subject_groups)
    if (groups.shape!=(len(x),) or not (1<=max_programs<=MAX_PROGRAMS)
        or len(x)<300 or any(str(g)=="" for g in groups)):
        raise ValueError("TRAIN episodes need genuine per-patient subject-group codes")
    folds=_subject_folds(groups,seed)
    q25,medians,q75=_train_sensor_quantiles(x,m)
    selected=[]
    normalized=[]
    rejected_variation=0
    rejected_redundant=0
    for op,channels in _programs():
        v,s=_raw_operator(x,m,op,channels,medians,q75)
        valid=s>=.5
        std_by_fold=[];cov_by_fold=[]
        okay=True
        for ix in folds:
            fraction=float(np.mean(valid[ix]))
            count=int(np.sum(valid[ix]))
            dev=float(np.std(v[ix][valid[ix]])) if count>=2 else 0.
            std_by_fold.append(dev);cov_by_fold.append(fraction)
            if (fraction<MIN_FOLD_COVERAGE or
                count<MIN_FOLD_EVALUABLE or dev<MIN_FOLD_STD):
                okay=False
        if not okay:
            rejected_variation+=1
            continue
        observed=v[valid]
        center=float(np.median(observed))
        quartiles=np.quantile(observed,[.25,.75])
        scale=float(max(.1,quartiles[1]-quartiles[0]))
        z=np.where(valid,np.clip((v-center)/scale,-8.,8.),0.)
        is_redundant=False
        for zv,ms in normalized:
            common=valid&ms
            if common.sum()<100:continue
            a=z[common];b=zv[common]
            if (np.std(a)<1e-9 or np.std(b)<1e-9):continue
            if abs(float(np.corrcoef(a,b)[0,1]))>MAX_REDUNDANCY:
                is_redundant=True;break
        if is_redundant:
            rejected_redundant+=1
            continue
        canonical=[op,*channels,round(center,8),round(scale,8),
                   [round(x,8) for x in medians],
                   [round(x,8) for x in q75]]
        cid="program-"+sha256(json.dumps(canonical,separators=(",",":")).encode()
                              ).hexdigest()[:16]
        candidate=MeasuredProgramV60(
            candidate_id=cid,operator=op,channels=channels,
            channel_names=tuple(RAW_INPUT_CHANNELS[j] for j in channels),
            train_q25=q25,train_median=medians,train_q75=q75,
            value_center=center,value_scale=scale,
            train_episodes=len(x),train_subjects=len(np.unique(groups)),
            train_episode_coverage=float(np.mean(valid)),
            fold_stds=tuple(std_by_fold),
            fold_coverages=tuple(cov_by_fold),
        )
        selected.append(candidate);normalized.append((z,valid))
        if len(selected)==max_programs:break
    return ProgramBankV60(
        candidates=tuple(selected),candidate_operators=UNARY+PAIR,
        channels=RAW_INPUT_CHANNELS,
        train_subjects=len(np.unique(groups)),train_episodes=len(x),
        screened_programs=len(UNARY)*3+len(PAIR)*3,
        rejected_low_variation_or_coverage=rejected_variation,
        rejected_redundant=rejected_redundant,
        max_programs=max_programs,
    )


def infer_program_matrix_v60(bank:ProgramBankV60,values,mask):
    """Program input: normalized raw scalar [N,K], support [N,K].

    Statistical program values are based only on the current 16-second
    measured input and frozen TRAIN reference. Missing is exactly neutral
    after the native torch concept head applies its support gate.
    """
    x,m=_checked_inputs(values,mask)
    values=[];support=[]
    for c in bank.candidates:
        v,s=_raw_operator(x,m,c.operator,c.channels,
                          c.train_median,c.train_q75)
        centered=np.where(s>=.5,
                          np.clip((v-c.value_center)/c.value_scale,-8.,8.),
                          0.)
        values.append(centered)
        support.append(s)
    if values:
        z=np.column_stack(values);a=np.column_stack(support)
    else:
        z=np.zeros((len(x),0),np.float64)
        a=np.zeros_like(z)
    if not (np.isfinite(z).all() and np.isfinite(a).all()
            and ((a>=0)&(a<=1)).all()):
        raise AssertionError("Nonfinite compiled measured source")
    return z,a


def fingerprint_program_bank_v60(bank:ProgramBankV60):
    return sha256(json.dumps(bank.manifest(),sort_keys=True,
            ensure_ascii=True,separators=(",",":")).encode()).hexdigest()
