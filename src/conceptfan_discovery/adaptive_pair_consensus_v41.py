"""V41 measured-only adaptive independent view pair, with TRAIN gauge & consensus.

No labels Y, latent simulator H/S, clinical semantic annotation or generated
measurement values enter fit_adaptive_source_v41. Classical V37 measured
centered cross-view slow-direction energy, now choose amongst AB/AC/BC
using **actual** ON&OFF sensor support, not the patient's predicted outcome.

A single accepted pair is NOT independently validated on this individual
episode. The physical view independence and source-path ON/OFF isolation
are EXTERNAL assumptions of the synthetic experiment; score agreement
does not certify physiological source identity or absence of common noise.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass

import numpy as np
from scipy.special import expit,logit

from .threeview_centered_v37 import (
    PAIRS,ThreeViewFitV37,pair_feature_v37,
)

ORDER=("AB","AC","BC")
LOW=1e-5
MIN_OVERLAP=60
MIN_LINEAR_CORR=.20
SLOPE_RANGE=(.25,4.)
DISAGREEMENT_RANGE=(.12,.38)
PAIR_TIME_MIN=4


@dataclass(frozen=True)
class PairGaugeV41:
    name:str
    slope:float
    intercept:float
    train_overlap:int
    train_correlation_to_AB:float
    calibrated:bool
    reason:str

    def audit(self):
        return asdict(self)


@dataclass(frozen=True)
class AdaptivePairFitV41:
    gauges:dict[str,PairGaugeV41]
    train_pairwise_threshold:dict[str,float|None]
    train_pairwise_overlap:dict[str,int]
    v37_fit:ThreeViewFitV37
    alignment_reference_name:str="AB"
    external_view_wiring_assumed:bool=True
    external_source_isolation_assumed:bool=True
    clinical_or_causal_semantics_certified:bool=False
    uses_hidden_truth_H_or_S:bool=False
    uses_outcome_Y_for_source_definition:bool=False

    def audit(self):
        return {
            "gauges":{k:v.audit() for k,v in self.gauges.items()},
            "train_disagreement_thresholds":self.train_pairwise_threshold,
            "train_pairwise_overlaps":self.train_pairwise_overlap,
            "alignment_reference":self.alignment_reference_name,
            "v37_masked_measured_fit":self.v37_fit.audit(),
            "source_training_consumes_outcome_Y":False,
            "source_training_consumes_true_latent_H_S":False,
            "external_source_isolation_and_sensor_wiring_assumed":True,
            "real_world_physically_named_source_certified":False,
            "single_source_pair_requires_realworld_corroboration":True,
        }


def _natural_pair_features(fit:ThreeViewFitV37,x,m):
    if set(fit.pairs)!=set(ORDER):
        raise ValueError("Expected all three independently fitted TRAIN sensor views")
    return {k:pair_feature_v37(fit.pairs[k],x,m) for k in ORDER}


def _gauge_transform(q,gauge:PairGaugeV41):
    if not gauge.calibrated:
        raise ValueError("Do not infer source using uncalibrated pair")
    raw=np.asarray(q,dtype=np.float64)
    if not np.isfinite(raw).all() or np.min(raw)<0. or np.max(raw)>1.:
        raise ValueError("Measured slow-source confidence outside [0,1]")
    return expit(gauge.slope*logit(np.clip(raw,LOW,1-LOW))+gauge.intercept)


def _regress_positive_pair(reference,source,supported,*,name):
    overlap=int(supported.sum())
    if overlap<MIN_OVERLAP:
        return PairGaugeV41(
            name,0.,0.,overlap,0.,False,"ABSTAIN_TRAIN_GAUGE_OVERLAP_TOO_SMALL")
    a=logit(np.clip(np.asarray(source)[supported],LOW,1-LOW))
    b=logit(np.clip(np.asarray(reference)[supported],LOW,1-LOW))
    x=a-a.mean()
    y=b-b.mean()
    sd1=np.sqrt(float(np.mean(x*x)))
    sd2=np.sqrt(float(np.mean(y*y)))
    if sd1<1e-6 or sd2<1e-6:
        return PairGaugeV41(
            name,0.,0.,overlap,0.,False,"ABSTAIN_NEAR_CONSTANT_PAIR_ENERGY")
    corr=float(np.mean(x*y)/(sd1*sd2))
    slope=float(np.mean(x*y)/(np.mean(x*x)+1e-6))
    intercept=float(b.mean()-slope*a.mean())
    valid=(np.isfinite(corr) and np.isfinite(slope) and np.isfinite(intercept)
           and corr>=MIN_LINEAR_CORR
           and SLOPE_RANGE[0]<=slope<=SLOPE_RANGE[1])
    return PairGaugeV41(
        name,slope,intercept,overlap,corr,valid,
        ("TRAIN_ALIGNED_POSITIVE_MONOTONE_WITH_AB"
         if valid else "ABSTAIN_UNSUPPORTED_BLIND_PAIR_GAUGE"))


def fit_adaptive_source_v41(
    v37:ThreeViewFitV37,train_difference,train_observed_mask,
)->AdaptivePairFitV41:
    """No Y, V1/V2 labels, TEST, mixing H or latent S arguments."""
    pair=_natural_pair_features(
        v37,train_difference,train_observed_mask)
    n=len(pair["AB"]["q"])
    if n<300 or v37.true_physical_semantics_certified:
        raise ValueError("Invalid registered TRAIN-only source calibration")
    gauges={
        "AB":PairGaugeV41(
            "AB",1.,0.,int((pair["AB"]["support"]>0).sum()),1.,
            True,"CANONICAL_ARBITRARY_AB_ENERGY_GAUGE"),
    }
    for name in ORDER[1:]:
        both=(pair["AB"]["support"]>0)&(pair[name]["support"]>0)
        gauges[name]=_regress_positive_pair(
            pair["AB"]["q"],pair[name]["q"],both,name=name)
    thresholds={}
    overlaps={}
    for i,a in enumerate(ORDER):
        for b in ORDER[i+1:]:
            key=a+"|"+b
            valid=gauges[a].calibrated and gauges[b].calibrated
            if valid:
                both=(pair[a]["support"]>0)&(pair[b]["support"]>0)
                overlaps[key]=int(both.sum())
                if overlaps[key]>=MIN_OVERLAP:
                    qa=_gauge_transform(pair[a]["q"],gauges[a])[both]
                    qb=_gauge_transform(pair[b]["q"],gauges[b])[both]
                    gap=np.abs(qa-qb)
                    thresholds[key]=float(max(
                        DISAGREEMENT_RANGE[0],
                        min(DISAGREEMENT_RANGE[1],np.quantile(gap,.95))))
                else:
                    thresholds[key]=None
            else:
                overlaps[key]=0
                thresholds[key]=None
    return AdaptivePairFitV41(
        gauges=gauges,
        train_pairwise_threshold=thresholds,
        train_pairwise_overlap=overlaps,
        v37_fit=v37,
    )


def _select_features(pair,calibration:AdaptivePairFitV41,*,enforce_consensus):
    """Can be unit-tested directly with synthetic q/support/n, no fitting/Y."""
    if set(pair)!=set(ORDER):
        raise ValueError("All registered pair receipts are required")
    n=len(np.asarray(pair["AB"]["q"]))
    raw_q=np.column_stack([pair[k]["q"] for k in ORDER]).astype(np.float64)
    raw_s=np.column_stack([pair[k]["support"] for k in ORDER]).astype(float)
    times=np.column_stack([pair[k]["paired_timepoints"] for k in ORDER]).astype(int)
    if (raw_q.shape!=(n,3) or raw_s.shape!=(n,3) or times.shape!=(n,3)
        or not np.isfinite(raw_q).all() or not np.isfinite(raw_s).all()
        or np.any(raw_q<0) or np.any(raw_q>1) or
        not np.isin(raw_s,[0.,1.]).all()
        or np.any(times<0) or np.any(times>16)
        or n<1):
        raise ValueError("Invalid actual masked independent source pair evidence")
    good=(raw_s>0)&(times>=PAIR_TIME_MIN)
    if enforce_consensus:
        trustworthy=np.array([calibration.gauges[k].calibrated for k in ORDER])
        good&=trustworthy[None,:]
        aligned=np.column_stack([
            (_gauge_transform(raw_q[:,i],calibration.gauges[k])
             if calibration.gauges[k].calibrated
             else np.full(n,.5))
            for i,k in enumerate(ORDER)
        ])
    else:
        aligned=raw_q
    n_supported=good.sum(axis=1)
    incompatible=np.zeros(n,dtype=bool)
    pair_conflicts=np.zeros((n,3),dtype=bool)
    # Compare ALL supported pairs, including the pair that would not be
    # selected as "best". This prevents cherry-picking compatible subsets.
    if enforce_consensus:
        t=0
        for i,a in enumerate(ORDER):
            for j,b in enumerate(ORDER):
                if j<=i:
                    continue
                both=good[:,i]&good[:,j]
                threshold=calibration.train_pairwise_threshold[a+"|"+b]
                disagreement=(
                    (np.abs(aligned[:,i]-aligned[:,j])>threshold)
                    if threshold is not None else np.ones(n,dtype=bool))
                pair_conflicts[:,t]=both&disagreement
                incompatible |= pair_conflicts[:,t]
                t+=1
    # Prefer most actually measured joint ON&OFF timesteps; deterministic
    # AB > AC > BC ties. NEVER maximize q or classifier risk.
    priorities=np.where(good,times,-1)
    chosen=np.argmax(priorities,axis=1)
    usable=(n_supported>0)&~incompatible
    selected=np.array(ORDER,dtype="U2")[chosen]
    selected[~usable]="--"
    q=np.where(usable,aligned[np.arange(n),chosen],.5)
    chosen_count=times[np.arange(n),chosen]
    chosen_count=np.where(usable,chosen_count,0)
    single=usable&(n_supported==1)
    corroborated=usable&(n_supported>=2)
    reason=np.where(
        n_supported==0,"ABSTAIN_NO_TRAIN_CERTIFIED_MEASURED_PAIR",
        np.where(incompatible,
                 "ABSTAIN_CONTRADICTORY_MEASURED_PAIRS",
                 np.where(single,"SINGLE_PAIR_UNCORROBORATED",
                          "MULTIPAIR_MEASURED_AGREEMENT")))
    if not enforce_consensus:
        reason=np.where(usable,"UNSAFE_UNCONFIRMED_ANY_PAIR",
                        "ABSTAIN_NO_MEASURED_PAIR")
    return {
        "q":q,
        "support":usable.astype(float),
        "selected_pair":selected,
        "selected_joint_measured_times":chosen_count.astype(int),
        "pair_membership_raw":raw_q,
        "pair_membership_train_aligned":aligned,
        "pair_support":good,
        "pair_timepoints":times,
        "n_supported_pairs":n_supported.astype(int),
        "pair_conflicts":pair_conflicts,
        "refused_due_to_disagreement":incompatible,
        "single_pair_uncorroborated":single,
        "two_or_three_pairs_corroborated":corroborated,
        "decision_reason":reason,
        "all_used_features_actually_measured":True,
        "single_pair_not_semantically_certified":True,
        "independent_wiring_and_ON_OFF_isolation_externally_assumed":True,
    }


def adaptive_pair_feature_v41(
    fit:AdaptivePairFitV41,x,observed_mask,
    *,enforce_consensus:bool=True,
):
    measured=_natural_pair_features(fit.v37_fit,x,observed_mask)
    return _select_features(
        measured,fit,enforce_consensus=enforce_consensus)


def unsafe_unconfirmed_any_pair_v41(fit:AdaptivePairFitV41,x,m):
    """Falsifier: no gauge or multi-view agreement, **NOT an identified concept**."""
    return adaptive_pair_feature_v41(fit,x,m,enforce_consensus=False)
