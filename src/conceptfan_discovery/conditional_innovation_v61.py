"""V61 TRAIN-only conditional-innovation concept residualizer and null tests.

Ridge removes the part of an observed RESP/SpO2 program linearly predicted
from *six PULSE-only program histories and their masks*. A residual may carry
incremental statistical information but is NOT a new physiological cause.

No task labels, future inputs, TEST outcomes or site measurements are used
to fit the residualizer. No source evidence => bitwise zero contribution.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
import numpy as np
from sklearn.linear_model import Ridge

from .program_compiler_v60 import _subject_folds

RIDGE_ALPHA=25.
MIN_SOURCE_EPISODES=100
MIN_SOURCE_FOLD_STD=.15
MIN_SOURCE_FOLD_CELLS=25
MIN_SOURCE_FOLD_COVERAGE=.5
SCALE_FLOOR=.25
PULSE_CONTEXT_MIN=.5
N_NULL_PERMUTATIONS=49
NULL_MARGIN=.003


@dataclass(frozen=True)
class ConditionalSourceV61:
    raw_source_indices:tuple[int,...]
    raw_pulse_indices:tuple[int,...]
    coefficients:tuple[tuple[float,...],...]
    intercepts:tuple[float,...]
    train_medians:tuple[float,...]
    train_iqrs:tuple[float,...]
    train_residual_r2:tuple[float,...]
    train_fold_std:tuple[tuple[float,...],...]
    train_n_subjects:int
    train_n_windows:int
    ridge_alpha:float=RIDGE_ALPHA
    fit_used_y:bool=False
    source_reads_pulse_directly:bool=False
    source_is_causal_physical:bool=False

    def passport(self):
        return asdict(self)


def _group_weights(groups):
    labels=np.asarray(groups)
    people,inv,cnt=np.unique(labels,return_inverse=True,return_counts=True)
    if len(people)<12:raise ValueError("Need real subject groups for TRAIN residualizer")
    wt=1./cnt[inv]
    return wt/wt.mean()


def _conditional_context(z,s,pulse_indices,raw_mask):
    zz=np.asarray(z,float)
    ss=np.asarray(s,float)
    mm=np.asarray(raw_mask,bool)
    if (zz.shape!=ss.shape or mm.ndim!=3 or
        mm.shape[:2]!=(len(zz),16) or mm.shape[2]!=3):
        raise ValueError("Mask/context shape incompatible")
    pulse_context_fraction=mm[:,:,0].mean(axis=1)
    idx=np.asarray(pulse_indices,dtype=int)
    if not len(idx):raise ValueError("Missing PULSE baseline program")
    context=np.concatenate((zz[:,idx],ss[:,idx]),axis=1)
    present=pulse_context_fraction>=PULSE_CONTEXT_MIN
    return context,present


def fit_conditional_sources_v61(z,s,raw_mask,train_groups,
                                *,pulse_indices,source_indices,seed=20266119):
    values=np.asarray(z,float)
    sup=np.asarray(s,float)
    if values.ndim!=2 or values.shape!=sup.shape:
        raise ValueError("TRAIN program matrix required")
    groups=np.asarray(train_groups)
    if groups.shape!=(len(values),):raise ValueError("TRAIN patient groups required")
    context,pulse_available=_conditional_context(
        values,sup,pulse_indices,raw_mask)
    weights=_group_weights(groups)
    folds=_subject_folds(groups,seed)
    keep=[];coefs=[];intercepts=[];med=[];iqr=[];r2=[];fold_stats=[]
    for j in source_indices:
        observed=(sup[:,j]>=.5)&pulse_available
        if int(observed.sum())<MIN_SOURCE_EPISODES:continue
        # Source-target of this regression = unlabeled measured expression,
        # never physiological HR / RESP outcome.
        ridge=Ridge(alpha=RIDGE_ALPHA,fit_intercept=True)
        ridge.fit(context[observed],values[observed,j],
                  sample_weight=weights[observed])
        predicted=ridge.predict(context)
        residual=values[:,j]-predicted
        qlo,qmid,qhi=np.quantile(residual[observed],[.25,.5,.75])
        width=float(max(qhi-qlo,SCALE_FLOOR))
        normal=np.clip((residual-qmid)/width,-8,8)
        checks=[];passes=True
        for ids in folds:
            available=observed&ids
            n=int(available.sum())
            coverage=n/max(int(ids.sum()),1)
            dispersion=float(normal[available].std()) if n>=2 else 0.
            checks.append(dispersion)
            if (n<MIN_SOURCE_FOLD_CELLS or
                coverage<MIN_SOURCE_FOLD_COVERAGE or
                dispersion<MIN_SOURCE_FOLD_STD):
                passes=False
        if not passes:continue
        y=values[observed,j]
        residual_ss=float(np.mean((y-predicted[observed])**2))
        target_ss=float(np.var(y))
        score=1.-residual_ss/max(target_ss,1e-10)
        keep.append(int(j));coefs.append(tuple(map(float,ridge.coef_)))
        intercepts.append(float(ridge.intercept_))
        med.append(float(qmid));iqr.append(width)
        r2.append(score);fold_stats.append(tuple(checks))
    if not keep:return None
    return ConditionalSourceV61(
        raw_source_indices=tuple(keep),
        raw_pulse_indices=tuple(map(int,pulse_indices)),
        coefficients=tuple(coefs),intercepts=tuple(intercepts),
        train_medians=tuple(med),train_iqrs=tuple(iqr),
        train_residual_r2=tuple(r2),
        train_fold_std=tuple(fold_stats),
        train_n_subjects=len(np.unique(groups)),train_n_windows=len(values))


def infer_conditional_sources_v61(fit,z,s,raw_mask):
    zz=np.asarray(z,float);sup=np.asarray(s,float)
    if fit is None:
        return np.zeros((len(zz),0)),np.zeros((len(zz),0))
    context,pulse_available=_conditional_context(
        zz,sup,fit.raw_pulse_indices,raw_mask)
    result=[];observed_support=[]
    for j,coef,intercept,med,scale in zip(
        fit.raw_source_indices,fit.coefficients,fit.intercepts,
        fit.train_medians,fit.train_iqrs,
    ):
        prediction=context@np.asarray(coef,float)+intercept
        raw=zz[:,j]-prediction
        available=pulse_available&(sup[:,j]>=.5)
        value=np.where(available,
                       np.clip((raw-med)/scale,-8,8),0.)
        support=np.where(available,sup[:,j],0.)
        result.append(value);observed_support.append(support)
    if not result:raise AssertionError("Source contract empty but non-null")
    v=np.column_stack(result);q=np.column_stack(observed_support)
    if not(np.isfinite(v).all() and np.isfinite(q).all()):
        raise AssertionError("Nonfinite masked conditional novelty")
    if np.any((q==0)&(v!=0)):
        raise AssertionError("Unsupported new source emitted non-neutral value")
    return v,q


def patient_stratified_null_v61(residual,support,pulse_baseline,
                               patient_groups,*,seed):
    """Shuffle entire novel-source rows, *within* patient+PULSE-score strata.

    Controls for patient identity and pulse intensity while breaking exact
    within-window co-temporal novel source/outcome alignment. It is not an
    exchangeability proof under overlapping/auto-correlated windows.
    """
    val=np.asarray(residual,float)
    sup=np.asarray(support,float)
    pulse=np.asarray(pulse_baseline,float).reshape(-1)
    ids=np.asarray(patient_groups)
    if (val.ndim!=2 or val.shape!=sup.shape or
        len(val)!=len(pulse) or len(ids)!=len(val)):
        raise ValueError("Paired patient/pulse/source null input required")
    rng=np.random.default_rng(seed)
    moved=np.arange(len(val))
    for p in np.unique(ids):
        where=np.flatnonzero(ids==p)
        if len(where)<8:continue
        thresholds=np.quantile(pulse[where],[.25,.5,.75])
        strata=np.searchsorted(thresholds,pulse[where],side="right")
        for bin_id in range(4):
            pos=where[strata==bin_id]
            if len(pos)>=2:
                # Entire non-PULSE vector moves together; PULSE remains fixed.
                moved[pos]=rng.permutation(pos)
    fraction=float(np.mean(moved!=np.arange(len(val))))
    return val[moved],sup[moved],fraction


def conditional_shuffle_gate_v61(fit,values,support,base,y,groups,
                                 *,observed_proposed,seed):
    """49 predeclared within-patient/PULSE-stratum negative permutations.

    No TRAIN/VALID label is read to fit residualization; VALID Y appears
    here ONLY to evaluate frozen candidate evidence, like an admission gate.
    """
    from .program_fan_v60 import (
        infer_program_head_v60,group_macro_nll_v60,
    )
    yy=np.asarray(y)
    bb=np.asarray(base,float)
    origin=group_macro_nll_v60(yy,bb,groups)
    observed=float(origin-group_macro_nll_v60(
        yy,observed_proposed,groups))
    shuffled=[];swap=[]
    for k in range(N_NULL_PERMUTATIONS):
        v,s,f=patient_stratified_null_v61(
            values,support,base,groups,seed=seed+97*k)
        forecast=infer_program_head_v60(fit,v,s,bb)["logits"]
        gain=origin-group_macro_nll_v60(yy,forecast,groups)
        shuffled.append(float(gain));swap.append(float(f))
    threshold=float(np.quantile(shuffled,.95))
    min_shift=float(min(swap))
    passed=(observed>threshold+NULL_MARGIN and min_shift>=.25)
    return {
        "conditional_source_subject_stratified_null_repeats":N_NULL_PERMUTATIONS,
        "observed_patient_macro_NLL_gain":observed,
        "null_gain_95th_percentile":threshold,
        "observed_minus_null95":observed-threshold,
        "minimum_row_reassignment_fraction":min_shift,
        "passed":bool(passed),
        "null_preserves_patient_and_pulse_quartile":True,
        "null_broken_joint_nonpulse_window_linkage":True,
        "autocorrelated_window_exchangeability_not_proven":True,
    }
