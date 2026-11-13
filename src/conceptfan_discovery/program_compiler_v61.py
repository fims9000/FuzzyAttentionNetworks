"""V61 masked typed compiler with hard target-channel exclusion.

Reuses V60 *frozen* operator definitions and subject-fold criteria. The
allowlist is applied BEFORE reading TRAIN channel statistics, not after
looking at excluded-target distributions or labels. Disallowed raw sensor
cells may contain arbitrary private data without influencing the output.
"""
from __future__ import annotations

from hashlib import sha256
import json
import numpy as np

from .program_compiler_v60 import (
    _checked_inputs,_subject_folds,_raw_operator,_programs,
    MeasuredProgramV60,ProgramBankV60,UNARY,PAIR,
    RAW_INPUT_CHANNELS,MIN_FOLD_COVERAGE,MIN_FOLD_EVALUABLE,
    MIN_FOLD_STD,MAX_REDUNDANCY,MAX_PROGRAMS,
)


def compile_allowed_programs_v61(values,mask,train_subject_groups,
                                  *,allowed_channels=(0,1,2),
                                  seed=20266113,max_programs=MAX_PROGRAMS):
    x,m=_checked_inputs(values,mask)
    allowed=tuple(sorted(set(allowed_channels)))
    if not allowed or 0 not in allowed or any(
        c not in (0,1,2) for c in allowed):
        raise ValueError("PULSE baseline and explicit raw target-channel allowlist required")
    if (len(x)<300 or not(1<=max_programs<=MAX_PROGRAMS)):
        raise ValueError("TRAIN cohort and max programs invalid")
    groups=np.asarray(train_subject_groups)
    if groups.shape!=(len(x),):raise ValueError("Real TRAIN patient groups required")
    # Disallowed channels must not affect any program value, TRAIN quantile,
    # fold correlation, ranking, or redundancy: force NaN mask before fitting.
    permitted=np.zeros(3,bool);permitted[list(allowed)]=True
    m=m&permitted[None,None,:]
    x=np.where(m,x,0.)
    folds=_subject_folds(groups,seed)
    medians=np.zeros(3);q25=np.zeros(3);q75=np.zeros(3)
    for j in allowed:
        samples=x[:,:,j][m[:,:,j]]
        if len(samples)<100 or not np.isfinite(samples).all():
            raise ValueError("Insufficient masked TRAIN data in allowed raw sensor")
        q25[j],medians[j],q75[j]=np.quantile(samples,[.25,.5,.75])
    proposals=[];normal=[];refused=0;redundant=0;attempts=0
    for operator,channels in _programs():
        if any(k not in allowed for k in channels):continue
        attempts+=1
        value,support=_raw_operator(x,m,operator,channels,medians,q75)
        good=support>=.5
        fold_std=[];fold_cov=[]
        passed=True
        for ids in folds:
            n=int(np.count_nonzero(good[ids]))
            cov=float(good[ids].mean())
            sd=float(value[ids][good[ids]].std()) if n>=2 else 0.
            fold_std.append(sd);fold_cov.append(cov)
            if n<MIN_FOLD_EVALUABLE or cov<MIN_FOLD_COVERAGE or sd<MIN_FOLD_STD:
                passed=False
        if not passed:
            refused+=1;continue
        obs=value[good]
        center=float(np.median(obs))
        lo,hi=np.quantile(obs,[.25,.75])
        scale=float(max(.1,hi-lo))
        standardized=np.where(good,np.clip((value-center)/scale,-8,8),0.)
        dup=False
        for prev,pmask in normal:
            both=pmask&good
            if both.sum()<100:continue
            a=standardized[both];b=prev[both]
            if min(a.std(),b.std())<1e-9:continue
            if abs(float(np.corrcoef(a,b)[0,1]))>MAX_REDUNDANCY:
                dup=True;break
        if dup:
            redundant+=1;continue
        stable=json.dumps([operator,list(channels),round(center,8),
                           round(scale,8),list(map(float,medians)),
                           list(map(float,q75))],separators=(",",":"))
        candidate_id="conditional-"+sha256(stable.encode()).hexdigest()[:16]
        proposals.append(MeasuredProgramV60(
            candidate_id=candidate_id,operator=operator,
            channels=channels,
            channel_names=tuple(RAW_INPUT_CHANNELS[c] for c in channels),
            train_q25=tuple(map(float,q25)),
            train_median=tuple(map(float,medians)),
            train_q75=tuple(map(float,q75)),
            value_center=center,value_scale=scale,
            train_episodes=len(x),train_subjects=len(np.unique(groups)),
            train_episode_coverage=float(good.mean()),
            fold_stds=tuple(fold_std),fold_coverages=tuple(fold_cov),
        ))
        normal.append((standardized,good))
        if len(proposals)==max_programs:break
    return ProgramBankV60(
        candidates=tuple(proposals),candidate_operators=UNARY+PAIR,
        channels=RAW_INPUT_CHANNELS,
        train_subjects=len(np.unique(groups)),train_episodes=len(x),
        screened_programs=attempts,
        rejected_low_variation_or_coverage=refused,
        rejected_redundant=redundant,max_programs=max_programs,
    )


def select_conditioning_and_novel_programs_v61(bank,allowed_channels):
    allowed=set(allowed_channels)
    pulse=[i for i,c in enumerate(bank.candidates)
           if c.channels==(0,) and c.operator in UNARY]
    novel=[i for i,c in enumerate(bank.candidates)
           if 0 not in c.channels and set(c.channels).issubset(allowed)]
    if not pulse:
        raise ValueError("PULSE-only measured baseline absent: cannot assert conditional novelty")
    if len(set(pulse)&set(novel)):
        raise AssertionError("Conditional new source also reads dominant monitor")
    if any(0 in bank.candidates[j].channels for j in novel):
        raise AssertionError("PULSE contamination of claimed independent source")
    return tuple(pulse),tuple(novel)
