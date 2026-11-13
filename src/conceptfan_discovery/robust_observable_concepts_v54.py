"""V54: measured-group robust pooling and train-only spectral concept fallback.

Two distinct semantic contracts:
- coherent >=3 channel groups use a majority-median measured *source proxy*;
- mixed-loading data may only yield an *observable predictive subspace*,
  not identifiable physical factor coordinates.

Both are strictly concept-mediated signed additive classifiers; latent oracle
states, human concept labels and TEST targets are never used in source fitting.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from conceptfan_discovery.autonomous_temporal_bank_v53 import (
    AutonomousTemporalConceptFAN, TemporalVocabulary,
    check_measurements, group_signals, fit_temporal_vocabulary,
)
from conceptfan_discovery.lagged_rank import fit_lagged_predictive_rank


def robust_group_level(group,x,mask) -> tuple[np.ndarray,np.ndarray]:
    a,m=check_measurements(x,mask)
    cols=np.asarray(group.sensor_indices,dtype=int)
    obs=m[:,:,cols]
    means=np.asarray(group.train_means)[None,None,:]
    scales=np.asarray(group.train_stds)[None,None,:]
    signs=np.asarray(group.signs)[None,None,:]
    safe=np.where(obs,a[:,:,cols],means)
    aligned=(safe-means)*signs/scales
    count=obs.sum(axis=-1)
    if len(cols)<3:
        s,support,_,_=group_signals(group,a,m)
        return s,support
    # Observed-value-only per-time median. With three measured channels a
    # single gross outlier cannot move the median past the two healthy sensors.
    order=np.sort(np.where(obs,aligned,np.inf),axis=-1)
    left=np.maximum((count-1)//2,0)
    right=count//2
    a0=np.take_along_axis(order,left[...,None],axis=-1)[...,0]
    a1=np.take_along_axis(order,right[...,None],axis=-1)[...,0]
    median=np.where(count>0,(a0+a1)*.5,0.)
    return median,count/len(cols)


def fit_robust_vocabulary(x,mask,*,seed:int=20261301):
    source=fit_temporal_vocabulary(x,mask,seed=seed)
    revised=[]
    for group in source.groups:
        if len(group.sensor_indices)<3:
            revised.append(group)
            continue
        level,support=robust_group_level(group,x,mask)
        samples=level[support>0]
        if len(samples)<100:
            raise ValueError("Robust group without observed TRAIN measurements")
        lo,hi=np.quantile(samples,[.25,.75])
        revised.append(replace(group,state_tau=float(np.median(samples)),
                               state_width=float(max(.15,(hi-lo)/1.35))))
    return replace(source,groups=tuple(revised))


def robust_measured_concepts(vocabulary,x,mask):
    a,m=check_measurements(x,mask)
    if not vocabulary.groups:
        c=np.empty((len(a),a.shape[1],0),dtype=float)
        return {"trajectories":c,"summaries":c.mean(axis=1),
                "support":np.empty((len(a),0)),
                "names":tuple(),"coherence_rejected":np.zeros((len(a),0),bool)}
    seq=[];coverage=[];names=[]
    for group in vocabulary.groups:
        level,support=robust_group_level(group,a,m)
        state=.5+support*(expit((level-group.state_tau)/group.state_width)-.5)
        seq.append(state)
        coverage.append(support.mean(axis=1))
        names.append(group.group_id+"|robust-state")
    concepts=np.stack(seq,axis=-1)
    if not np.isfinite(concepts).all():
        raise AssertionError("Nonfinite robust concepts")
    return {"trajectories":concepts,"summaries":concepts.mean(axis=1),
            "support":np.stack(coverage,axis=1),
            "names":tuple(names),
            "coherence_rejected":np.zeros((len(a),len(vocabulary.groups)),bool)}


class RobustGroupConceptFAN(AutonomousTemporalConceptFAN):
    def __init__(self,vocabulary:TemporalVocabulary):
        super().__init__(vocabulary,include_surprise=False,guard=False)
    def extract(self,x,mask):
        return robust_measured_concepts(self.vocabulary,x,mask)
    def passport(self):
        p=super().passport()
        p["operator"]="TRAIN-calibrated source median if >=3 sensors; else source mean"
        p["gross_single_outlier_resistance_only_when_three_or_more_observed"]=True
        return p


@dataclass(frozen=True)
class ObservableSpectralDefinitions:
    means:tuple[float,...]
    scales:tuple[float,...]
    observable_loadings:tuple[tuple[float,...],...]
    thresholds:tuple[float,...]
    widths:tuple[float,...]
    train_rank_receipt:dict[str,Any]
    source_type:str="STATISTICAL_PREDICTIVE_SUBSPACE_NOT_PHYSICAL_CONCEPT"
    target_labels_used_to_fit:bool=False
    true_latent_states_used_to_fit:bool=False

    @property
    def rank(self)->int:
        return len(self.thresholds)
    def as_dict(self):
        return asdict(self)


def _masked_spectral_projection(definition:ObservableSpectralDefinitions,x,mask):
    a,m=check_measurements(x,mask)
    means=np.asarray(definition.means)[None,None,:]
    scales=np.asarray(definition.scales)[None,None,:]
    basis=np.asarray(definition.observable_loadings,dtype=float)
    if basis.shape!=(a.shape[-1],definition.rank):
        raise ValueError("Spectral coordinate basis incompatible with sensor count")
    # Hidden payloads are never used, even if they contain NaN.
    safe=np.where(m,a,means)
    standardized=(safe-means)/scales
    numerator=np.einsum("ntd,dk->ntk",standardized*m,basis,optimize=True)
    abs_basis=np.abs(basis)
    support_num=np.einsum("ntd,dk->ntk",m.astype(float),abs_basis,optimize=True)
    support=support_num/np.maximum(abs_basis.sum(axis=0)[None,None,:],1e-9)
    level=numerator/np.maximum(support_num,.2)
    return level,support


def fit_observable_spectral_definitions(train_x,train_mask,*,seed:int=20261301,
                                         permutations:int=60,bootstraps:int=20
                                         ) -> ObservableSpectralDefinitions:
    a,m=check_measurements(train_x,train_mask)
    # Input to original rank estimator must have zeros at absent measurements.
    safe=np.where(m,a,0.)
    evidence=fit_lagged_predictive_rank(
        safe,m.astype(float),lags=(1,2,3),permutations=permutations,
        patient_bootstraps=bootstraps,alpha=.05,seed=seed)
    count=m.sum(axis=(0,1))
    if np.any(count<100):
        raise ValueError("Spectral source definition requires observed TRAIN channels")
    means=(safe.sum(axis=(0,1))/count)
    centred=np.where(m,a-means[None,None,:],0.)
    scales=np.sqrt((centred**2).sum(axis=(0,1))/count).clip(.1,None)
    basis=evidence.left_observable_loadings[:,:evidence.rank]
    proto=ObservableSpectralDefinitions(
        tuple(means),tuple(scales),tuple(tuple(float(v) for v in row)
                                        for row in basis),
        tuple(0. for _ in range(evidence.rank)),
        tuple(1. for _ in range(evidence.rank)),
        evidence.to_json())
    if evidence.rank==0:
        return proto
    level,support=_masked_spectral_projection(proto,a,m)
    taus=[];widths=[]
    for k in range(evidence.rank):
        values=level[:,:,k][support[:,:,k]>0]
        if len(values)<100:
            raise ValueError("Not enough projected source observations")
        lo,hi=np.quantile(values,[.25,.75])
        taus.append(float(np.median(values)))
        widths.append(float(max(.15,(hi-lo)/1.35)))
    return ObservableSpectralDefinitions(proto.means,proto.scales,
               proto.observable_loadings,tuple(taus),tuple(widths),proto.train_rank_receipt)


class SpectralObservableConceptFAN:
    """SVD predictive coordinates (not source identities) in an exact risk head."""
    def __init__(self,definition:ObservableSpectralDefinitions):
        self.definition=definition
        self.bias:float|None=None
        self.beta:np.ndarray|None=None
        self.n_risk_labels=0

    def extract(self,x,mask):
        values,m=check_measurements(x,mask)
        if self.definition.rank==0:
            c=np.empty((len(values),values.shape[1],0),dtype=float)
            return {"trajectories":c,"summaries":c.mean(axis=1),
                    "names":tuple(),"support":np.empty((len(values),0))}
        level,support=_masked_spectral_projection(self.definition,values,m)
        tau=np.asarray(self.definition.thresholds)[None,None,:]
        width=np.asarray(self.definition.widths)[None,None,:]
        c=.5+support*(expit((level-tau)/width)-.5)
        return {"trajectories":c,"summaries":c.mean(axis=1),
                "names":tuple(f"observed-predictive-axis-{i}" for i in range(self.definition.rank)),
                "support":support.mean(axis=1)}

    def fit_risk_head(self,x,mask,y):
        yy=np.asarray(y).reshape(-1)
        features=self.extract(x,mask)["summaries"]-.5
        if len(yy)!=len(features) or not np.isin(yy,(0,1)).all() or len(np.unique(yy))!=2:
            raise ValueError("Invalid TRAIN outcome labels for spectral head")
        if features.shape[1]==0:
            p=float(np.clip(yy.mean(),1e-6,1.-1e-6))
            self.bias=float(np.log(p/(1.-p)))
            self.beta=np.empty(0)
        else:
            clf=LogisticRegression(C=1.,max_iter=1200)
            clf.fit(features,yy)
            self.bias=float(clf.intercept_[0])
            self.beta=clf.coef_[0].copy()
        self.n_risk_labels=len(yy)
        return self

    def predict(self,x,mask):
        if self.beta is None or self.bias is None:
            raise RuntimeError("Fit the concept-only TRAIN risk head")
        extracted=self.extract(x,mask)
        con=(extracted["summaries"]-.5)*self.beta[None,:]
        logit=self.bias+con.sum(axis=1)
        if not np.isfinite(logit).all():
            raise AssertionError("Nonfinite model logit")
        return {**extracted,"contributions":con,"logits":logit,
                "probability":expit(logit),"bias":self.bias,
                "betas":self.beta.copy(),
                "max_additive_error":float(np.max(np.abs(logit-(self.bias+con.sum(axis=1))))),
                "causal_physical_source_identity_claimed":False}

    def passport(self):
        if self.beta is None:
            raise RuntimeError("Unfitted spectral model")
        return {"source":self.definition.as_dict(),
                "bias":self.bias,"signed_weights":self.beta.tolist(),
                "n_risk_train_labels":self.n_risk_labels,
                "no_hidden_label_or_oracle_in_source_rank":True,
                "no_physical_axis_identity_identified":True,
                "exactly_additive":True}
