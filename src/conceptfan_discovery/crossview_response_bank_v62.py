"""V62 unlabeled train-only cross-view physical-response concept bank.

Rank selection is based on paired views of one real independent instrument
trial, with a null from independent permutations of entire TRAIN trials.
Neither gas-flow names, analyzer concentrations, clinical labels nor gas
source ground truth enter concept definition.

The recovered axes are shared observable response directions, NOT chemical
source identities, even when several survive an empirical null threshold.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from conceptfan_discovery.autonomous_temporal_bank_v53 import check_measurements
from conceptfan_discovery.robust_observable_concepts_v54 import (
    ObservableSpectralDefinitions,_masked_spectral_projection
)

SOURCE_SEED=20261801
NULL_PERMUTATIONS=100
WHOLE_RUN_BOOTSTRAPS=20
VIEW_BINS={
    "early_baseline":(0,6),
    "late_baseline":(6,12),
    "early_exposure":(12,30),
    "late_exposure":(30,48),
}


def _view_feature(x,mask,start:int,stop:int):
    obs=mask[:,start:stop,:]
    count=obs.sum(axis=1)
    if (count==0).any():
        raise ValueError("A preregistered source view has unavailable sensors")
    return np.where(obs,x[:,start:stop,:],0).sum(axis=1)/count


def make_response_views(x,mask):
    """Two nonoverlapping response and baseline views from every physical run."""
    a,m=check_measurements(x,mask)
    if a.shape[1]!=60:
        raise ValueError("Exact 300-second real gas run required (60x5 seconds)")
    early=_view_feature(a,m,*VIEW_BINS["early_exposure"])
    baseline_early=_view_feature(a,m,*VIEW_BINS["early_baseline"])
    late=_view_feature(a,m,*VIEW_BINS["late_exposure"])
    baseline_late=_view_feature(a,m,*VIEW_BINS["late_baseline"])
    return early-baseline_early,late-baseline_late


def crossview_rank_evidence(view_a,view_b,*,seed=SOURCE_SEED,
                             permutations=NULL_PERMUTATIONS,
                             bootstraps=WHOLE_RUN_BOOTSTRAPS):
    aa=np.asarray(view_a,dtype=np.float64)
    bb=np.asarray(view_b,dtype=np.float64)
    if (aa.ndim!=2 or bb.shape!=aa.shape or len(aa)<80 or
        aa.shape[1]<2 or not np.isfinite(aa).all() or
        not np.isfinite(bb).all() or permutations<25 or bootstraps<5):
        raise ValueError("Need >=80 independent paired TRAIN runs and sufficient null trials")
    n,d=aa.shape
    a=aa-aa.mean(axis=0)
    b=bb-bb.mean(axis=0)
    actual=a.T@b/(n-1)
    u,s,_=np.linalg.svd(actual,full_matrices=False)
    for j in range(d):
        chosen=int(np.argmax(np.abs(u[:,j])))
        if u[chosen,j]<0:
            u[:,j]*=-1
    rng=np.random.default_rng(seed)
    null_top=np.empty(permutations,dtype=float)
    for z in range(permutations):
        pairing=rng.permutation(n)
        null_top[z]=np.linalg.svd(a.T@b[pairing]/(n-1),
                                   compute_uv=False)[0]
    threshold=float(np.quantile(null_top,.95))
    rank=int(np.count_nonzero(s>threshold))
    bootstrap_ranks=[]
    for z in range(bootstraps):
        take=rng.integers(n,size=n)
        x=aa[take]-aa[take].mean(axis=0)
        y=bb[take]-bb[take].mean(axis=0)
        sval=np.linalg.svd(x.T@y/(n-1),compute_uv=False)
        bootstrap_ranks.append(int(np.count_nonzero(sval>threshold)))
    interval=np.quantile(bootstrap_ranks,[.05,.95])
    return {
        "rank":rank,
        "left_observable_loadings":u,
        "signal_singular_values":s,
        "max_singular_value_null_samples":null_top,
        "familywise_noise_max_threshold":threshold,
        "train_whole_run_bootstrap_ranks":bootstrap_ranks,
        "train_whole_run_rank_interval_05_95":[int(np.floor(interval[0])),
                                             int(np.ceil(interval[1]))],
        "source_fit_train_independent_runs":n,
        "null_pairings_are_whole_runs":True,
        "no_target_labels_or_gas_configurations_used":True,
        "physical_identity_proven":False,
    }


class TrainOnlyCrossViewBank:
    """V56 compatible source vocabulary from independent TRAIN physical runs."""

    def __init__(self,*,seed:int=SOURCE_SEED,
                 permutations:int=NULL_PERMUTATIONS,
                 bootstraps:int=WHOLE_RUN_BOOTSTRAPS):
        self.seed=int(seed)
        self.permutations=int(permutations)
        self.bootstraps=int(bootstraps)
        self.route=None
        self.source=None
        self.k=0
        self.tau=None
        self.width=None
        self.sha=None
        self.train_receipt=None

    def fit_unlabeled(self,train_x,train_mask):
        a,m=check_measurements(train_x,train_mask)
        if len(a)<80 or a.shape[1]!=60:
            raise ValueError("Requires >=80 whole 300s TRAIN trials, not dependent crops")
        va,vb=make_response_views(a,m)
        e=crossview_rank_evidence(va,vb,seed=self.seed,
                                  permutations=self.permutations,
                                  bootstraps=self.bootstraps)
        k=e["rank"]
        basis=e["left_observable_loadings"][:,:k]
        observed=m.sum(axis=(0,1))
        if (observed<100).any():
            raise ValueError("Insufficient TRAIN sensor observations")
        means=np.where(m,a,0.).sum(axis=(0,1))/observed
        centred=np.where(m,a-means[None,None,:],0.)
        scales=np.sqrt((centred*centred).sum(axis=(0,1))/observed).clip(.1,None)
        receipt={
            key:(value.tolist() if isinstance(value,np.ndarray) else value)
            for key,value in e.items()
            if key!="left_observable_loadings"
        }
        prototype=ObservableSpectralDefinitions(
            tuple(float(v) for v in means),
            tuple(float(v) for v in scales),
            tuple(tuple(float(v) for v in row) for row in basis),
            tuple(0. for _ in range(k)),
            tuple(1. for _ in range(k)),
            receipt
        )
        levels,support=(
            _masked_spectral_projection(prototype,a,m)
            if k else
            (np.empty((len(a),a.shape[1],0)),np.empty((len(a),a.shape[1],0)))
        )
        thresholds=[];widths=[]
        for index in range(k):
            values=levels[:,:,index][support[:,:,index]>0]
            if len(values)<100:
                raise ValueError("Insufficient actual TRAIN source projection values")
            q25,q75=np.quantile(values,[.25,.75])
            thresholds.append(float(np.median(values)))
            widths.append(float(max(.15,(q75-q25)/1.35)))
        self.source=ObservableSpectralDefinitions(
            prototype.means,prototype.scales,prototype.observable_loadings,
            tuple(thresholds),tuple(widths),receipt)
        self.k=k
        self.tau=np.asarray(thresholds,float)
        self.width=np.asarray(widths,float)
        self.route="CROSSVIEW_RESPONSE" if k else "OBSERVATION_ABSTAIN"
        self.sha=hashlib.sha256(json.dumps(
            self.source.as_dict(),sort_keys=True,allow_nan=False
        ).encode()).hexdigest()
        self.train_receipt={
            "source_route":self.route,
            "number_of_discovered_source_slots":k,
            "n_train_unlabeled":len(a),
            "null_is_whole_train_run_permutation":True,
            "source_sha256":self.sha,
            "sensor_response_view_definition":VIEW_BINS,
            "source_rank_evidence":receipt,
            "manual_source_number_or_gas_levels_used":False,
            "physical_concept_identity_certified":False,
            "source_reported_as_observable_response_axes_only":True,
        }
        return self

    def transform(self,x,mask):
        if self.source is None:
            raise RuntimeError("Fit unlabeled TRAIN source bank first")
        a,m=check_measurements(x,mask)
        if a.shape[-1]!=len(self.source.means):
            raise ValueError("Source dimensions changed after TRAIN fit")
        if not self.k:
            empty=np.empty((len(a),a.shape[1],0),dtype=np.float32)
            return empty,empty.copy()
        level,support=_masked_spectral_projection(self.source,a,m)
        if not np.isfinite(level).all() or not np.isfinite(support).all():
            raise AssertionError("Nonfinite observed response concepts")
        return level.astype(np.float32),support.astype(np.float32)

    def public_passport(self):
        if self.train_receipt is None:
            raise RuntimeError("Fit unlabeled source definition first")
        return {"source_receipt":self.train_receipt,
                "trained_threshold_initialization":self.tau.tolist(),
                "trained_width_initialization":self.width.tolist(),
                "source_manifest":self.source.as_dict()}
