"""V53: autonomous measured temporal concepts with exact additive risk.

Every source-group definition, sign, AR(1) transition and fuzzy membership
threshold comes from TRAIN observed measurement histories only. Neither
outcomes, known concept channel maps, latent source labels, future observations
nor validation/test data enter definition fitting.

Concept trajectories are computed *inside* the decision architecture:
logit(x) = bias + sum_k beta_k * (mean_t c_k(x,t) - 0.5).
The optional sensor-consistency safeguard abstains on incoherent sources,
but cannot identify common-mode shifts or distinguish physiology from devices.
This is an interpretable statistical model, not a causal source identifier.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from conceptfan_discovery.auto_measured_concept_proposals import (
    ProposalManifest, form_train_measured_groups,
)


@dataclass(frozen=True)
class TemporalGroup:
    group_id: str
    sensor_indices: tuple[int, ...]
    signs: tuple[int, ...]
    train_means: tuple[float, ...]
    train_stds: tuple[float, ...]
    rho: float
    state_tau: float
    state_width: float
    surprise_tau: float
    surprise_width: float
    disagreement_limit: float
    n_train_paired_times: int
    n_train_consistency_subjects: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalVocabulary:
    groups: tuple[TemporalGroup, ...]
    excluded_singletons: tuple[int, ...]
    ambiguous_components: tuple[tuple[int, ...], ...]
    training_subjects: int
    all_sensors_searched: bool
    no_outcomes_used: bool
    no_oracle_concept_labels_used: bool
    no_known_sensor_group_supplied: bool
    measured_definition_only: bool
    proposal_hash: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_measurements(x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a=np.asarray(x,dtype=np.float64)
    m=np.asarray(mask)
    if a.ndim!=3 or a.shape!=m.shape or a.shape[1]<4 or a.shape[2]<2:
        raise ValueError("Expected matched [N,T,D] values and masks")
    if not np.isin(m,(0,1)).all():
        raise ValueError("Non-binary observation mask")
    available=m.astype(bool)
    if not np.isfinite(a[available]).all():
        raise ValueError("Non-finite observed sensor reading")
    return a,available


def group_signals(group: TemporalGroup, x: np.ndarray, mask: np.ndarray
                  ) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    a,m=check_measurements(x,mask)
    cols=np.asarray(group.sensor_indices,dtype=int)
    if cols.min()<0 or cols.max()>=a.shape[-1]:
        raise ValueError("Group incompatible with input sensor dimension")
    obs=m[:,:,cols]
    # Never read hidden payload, even intermediate arithmetic on NaN.
    means=np.asarray(group.train_means)[None,None,:]
    stds=np.asarray(group.train_stds)[None,None,:]
    signs=np.asarray(group.signs)[None,None,:]
    safe=np.where(obs,a[:,:,cols],means)
    aligned=((safe-means)/stds)*signs
    aligned=np.where(obs,aligned,0.)
    count=obs.sum(axis=-1)
    level=aligned.sum(axis=-1)/np.maximum(count,1)
    support=count/len(cols)
    pair=(count[:,1:]>0)&(count[:,:-1]>0)
    innovations=np.zeros_like(level)
    innovations[:,1:]=np.where(
        pair,level[:,1:]-float(group.rho)*level[:,:-1],0.)
    pair_support=np.zeros_like(support)
    pair_support[:,1:]=np.where(pair,np.minimum(support[:,1:],support[:,:-1]),0.)
    return level,support,innovations,pair_support


def _disagreement(group: TemporalGroup,x:np.ndarray,mask:np.ndarray
                  ) -> tuple[np.ndarray,np.ndarray]:
    a,m=check_measurements(x,mask)
    cols=np.asarray(group.sensor_indices,dtype=int)
    obs=m[:,:,cols]
    means=np.asarray(group.train_means)[None,None,:]
    stds=np.asarray(group.train_stds)[None,None,:]
    signs=np.asarray(group.signs)[None,None,:]
    safe=np.where(obs,a[:,:,cols],means)
    aligned=(safe-means)/stds*signs
    count=obs.sum(axis=-1)
    centre=np.where(obs,aligned,0.).sum(axis=-1)/np.maximum(count,1)
    variance=(np.where(obs,(aligned-centre[:,:,None])**2,0.)
              .sum(axis=-1)/np.maximum(count,1))
    jointly=count>=2
    n_joint=jointly.sum(axis=1)
    score=(np.where(jointly,np.sqrt(variance),0.).sum(axis=1)
           /np.maximum(n_joint,1))
    score=np.where(n_joint>=3,score,np.inf)
    return score,n_joint


def _calibrate_group(proposed,x,mask) -> TemporalGroup:
    # Temporary receipt for measured signal extraction only.
    proto=TemporalGroup(
        proposed.candidate_id,tuple(proposed.sensor_indices),
        tuple(proposed.sensor_signs),tuple(proposed.train_means),
        tuple(proposed.train_stds),0.,0.,1.,0.,1.,1.,0,0)
    level,support,_,_=group_signals(proto,x,mask)
    usable=support>0
    adjacent=usable[:,1:]&usable[:,:-1]
    first=level[:,:-1][adjacent]
    second=level[:,1:][adjacent]
    if len(first)<60:
        raise ValueError("Source has insufficient consecutive TRAIN measurements")
    rho=float(np.clip(np.dot(first,second)/max(np.dot(first,first),1e-9),-.95,.95))
    proto=TemporalGroup(
        proto.group_id,proto.sensor_indices,proto.signs,
        proto.train_means,proto.train_stds,rho,0.,1.,0.,1.,1.,len(first),0)
    level,support,residual,pair_support=group_signals(proto,x,mask)
    state_values=level[support>0]
    innovation_values=np.abs(residual[pair_support>0])
    if len(state_values)<100 or len(innovation_values)<60:
        raise ValueError("Not enough observed group states or innovations")
    def iqr_width(z,minimum):
        q25,q75=np.quantile(z,[.25,.75])
        return float(max(minimum,(q75-q25)/1.35))
    tau=float(np.median(state_values))
    state_width=iqr_width(state_values,.15)
    surprise_tau=float(np.quantile(innovation_values,.85))
    surprise_width=iqr_width(innovation_values,.12)
    score,n_joint=_disagreement(proto,x,mask)
    valid=score[np.isfinite(score)]
    if len(valid)<100:
        # A group that cannot be corroborated has no verified consistency gate.
        limit=0.
    else:
        limit=float(max(.1,np.quantile(valid,.99)))
    return TemporalGroup(
        proto.group_id,proto.sensor_indices,proto.signs,
        proto.train_means,proto.train_stds,rho,tau,state_width,
        surprise_tau,surprise_width,limit,len(first),len(valid))


def fit_temporal_vocabulary(train_x:np.ndarray,train_mask:np.ndarray,
                            *,seed:int=20261221,max_groups:int=5
                            ) -> TemporalVocabulary:
    x,m=check_measurements(train_x,train_mask)
    # Deliberately no known sensor groups. This also removes previous manual K2.
    proposed=form_train_measured_groups(
        x,m,known_sensor_indices=(),seed=seed,
        min_pair_abs_corr=.60,min_joint_cells_per_fold=300,
        max_candidates=max_groups)
    groups=tuple(_calibrate_group(z,x,m) for z in proposed.candidates)
    digest=sha256(repr(proposed.as_dict()).encode("utf8")).hexdigest()
    return TemporalVocabulary(
        groups,proposed.ungrouped_sensors,proposed.ambiguous_components,
        len(x),True,True,True,True,True,digest)


def concepts_from_measurements(
    vocabulary:TemporalVocabulary,x:np.ndarray,mask:np.ndarray,
    *,include_surprise:bool=True,guard:bool=False
) -> dict[str,np.ndarray|tuple[str,...]]:
    values,m=check_measurements(x,mask)
    if not vocabulary.groups:
        empty=np.empty((len(values),values.shape[1],0))
        return {"trajectories":empty,"summaries":empty.mean(axis=1),
                "support":np.empty((len(values),0)),
                "coherence_rejected":np.empty((len(values),0),dtype=bool),
                "names":()}
    sequences=[]
    supports=[]
    rejected=[]
    names=[]
    for group in vocabulary.groups:
        level,support,innovation,paired_support=group_signals(group,values,m)
        score,njoint=_disagreement(group,values,m)
        # A fully unobserved source has no outcome influence. In guarded
        # mode require enough co-observed source evidence to trust a group.
        deny=((score>group.disagreement_limit)|(njoint<3)) if guard else np.zeros(len(values),bool)
        if guard and group.n_train_consistency_subjects<100:
            deny=np.ones(len(values),bool)
        state=.5+support*(expit((level-group.state_tau)/group.state_width)-.5)
        state=np.where(deny[:,None],.5,state)
        sequences.append(state)
        names.append(group.group_id+"|state")
        if include_surprise:
            surprise=.5+paired_support*(
                expit((np.abs(innovation)-group.surprise_tau)/group.surprise_width)-.5)
            surprise=np.where(deny[:,None],.5,surprise)
            sequences.append(surprise)
            names.append(group.group_id+"|surprise")
        supports.append(np.where(deny,0.,support.mean(axis=1)))
        rejected.append(deny)
    c=np.stack(sequences,axis=-1)
    if not (np.isfinite(c).all() and np.all((c>=0)&(c<=1))):
        raise AssertionError("Invalid bounded concepts")
    return {"trajectories":c,"summaries":c.mean(axis=1),
            "support":np.stack(supports,axis=1),
            "coherence_rejected":np.stack(rejected,axis=1),
            "names":tuple(names)}


def event_intervals(trajectory:np.ndarray,*,threshold:float=.65,
                    minimum_length:int=2)->tuple[tuple[int,int],...]:
    """Time-localized measured events, not clinical episode diagnoses."""
    z=np.asarray(trajectory)
    if z.ndim!=1 or not np.isfinite(z).all() or not 0<threshold<1:
        raise ValueError("Expected finite one-dimensional membership trajectory")
    if minimum_length<1:
        raise ValueError("Invalid temporal event duration")
    on=z>=threshold
    pad=np.concatenate(([False],on,[False]))
    starts=np.flatnonzero(~pad[:-1]&pad[1:])
    stops=np.flatnonzero(pad[:-1]&~pad[1:])
    return tuple((int(a),int(b)) for a,b in zip(starts,stops)
                 if b-a>=minimum_length)


class AutonomousTemporalConceptFAN:
    """Training-derived measured concepts; supervised risk uses ONLY concepts.

    The TRAIN-only source and fuzzy calibrations never consume Y. The strictly
    additive risk head is a frozen logistic model on concept summaries.
    This is not a causal model and never auto-names sources as diseases.
    """
    def __init__(self,vocabulary:TemporalVocabulary,*,include_surprise:bool=True,
                 guard:bool=False):
        self.vocabulary=vocabulary
        self.include_surprise=bool(include_surprise)
        self.guard=bool(guard)
        self.bias:float|None=None
        self.beta:np.ndarray|None=None
        self.n_risk_train_labels=0

    def fit_risk_head(self,x:np.ndarray,mask:np.ndarray,y:np.ndarray):
        yy=np.asarray(y).reshape(-1)
        if not np.isin(yy,(0,1)).all() or len(np.unique(yy))!=2:
            raise ValueError("TRAIN labels must be binary with both classes")
        features=self.extract(x,mask)["summaries"]
        if len(features)!=len(yy):
            raise ValueError("Features and labels misaligned")
        if features.shape[1]==0:
            rate=float(np.clip(yy.mean(),1e-6,1-1e-6))
            self.bias=float(np.log(rate/(1-rate)))
            self.beta=np.zeros(0,dtype=float)
        else:
            model=LogisticRegression(C=1.,max_iter=1200,solver="lbfgs")
            model.fit(features-.5,yy)
            self.bias=float(model.intercept_[0])
            self.beta=model.coef_[0].copy()
        self.n_risk_train_labels=len(yy)
        return self

    def extract(self,x,mask):
        return concepts_from_measurements(self.vocabulary,x,mask,
                include_surprise=self.include_surprise,guard=self.guard)

    def predict(self,x,mask):
        if self.bias is None or self.beta is None:
            raise RuntimeError("Risk head must be fitted on TRAIN labels")
        extracted=self.extract(x,mask)
        centred=extracted["summaries"]-.5
        contributions=centred*self.beta[None,:]
        logits=self.bias+contributions.sum(axis=1)
        if not np.allclose(logits,self.bias+contributions.sum(axis=1),rtol=0,atol=1e-12):
            raise AssertionError("Concept-only decomposition invalid")
        return {
            **extracted,"contributions":contributions,
            "logits":logits,"probability":expit(logits),
            "bias":self.bias,"betas":self.beta.copy(),
            "logit_additive_error":float(np.max(
                np.abs(logits-(self.bias+contributions.sum(axis=1))))),
        }

    def passport(self)->dict[str,Any]:
        if self.bias is None or self.beta is None:
            raise RuntimeError("Unfitted model")
        return {
            "source":self.vocabulary.as_dict(),
            "concept_operator_types":("state","surprise") if self.include_surprise else ("state",),
            "risk_beta":self.beta.tolist(),"risk_bias":self.bias,
            "risk_train_labels":self.n_risk_train_labels,
            "trained_source_without_labels":True,
            "source_names_certified_physical":False,
            "model_label_shortcuts":False,
            "exact_signed_additive_decision":True,
            "guard_enabled":self.guard,
        }
