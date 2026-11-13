"""V63 blind TRAIN innovation PCA/ICA concept vocabulary for real gas dynamics.

Physical gas setpoints are neither accepted nor used by this interface.
The rank choice is PCA(>=95% variance), bounded to <=8, not number of gases.
The ICA sensors are anonymous projections and never called chemical identities.
"""
from __future__ import annotations

import hashlib
import json
import warnings

import numpy as np
from scipy.special import expit
from sklearn.decomposition import FastICA,PCA
from sklearn.exceptions import ConvergenceWarning

from conceptfan_discovery.autonomous_temporal_bank_v53 import check_measurements


class TrainOnlyInnovationBank:
    def __init__(self,*,basis:str="ica",seed:int=20261913,
                 explained_variance_target:float=.95,max_k:int=8,
                 block_bootstraps:int=20):
        if basis not in ("pca","ica") or not 0.8<=explained_variance_target<=.99:
            raise ValueError("Unregistered innovation basis or variance cutoff")
        if not 1<=max_k<=8 or block_bootstraps<5:
            raise ValueError("K cap or block-bootstrap count is invalid")
        self.basis=basis
        self.seed=int(seed)
        self.target=float(explained_variance_target)
        self.max_k=int(max_k)
        self.n_boot=int(block_bootstraps)
        self.k=0
        self.route=None
        self.tau=None
        self.width=None
        self.sha=None
        self.loading=None
        self.train_receipt=None

    @staticmethod
    def _innovations(a:np.ndarray,m:np.ndarray):
        if a.shape[1]!=120:
            raise ValueError("Exactly 120 observed one-second bins per episode required")
        def obs_median(start:int,end:int):
            sliced=np.where(m[:,start:end],a[:,start:end],np.nan)
            if not m[:,start:end].all():
                raise ValueError("Missing source channels require a different predeclared protocol")
            return np.median(sliced,axis=1)
        return obs_median(90,120)-obs_median(0,30)

    @staticmethod
    def _pca_rank(innovation:np.ndarray,target:float,cap:int):
        p=PCA().fit(innovation)
        cumulative=np.cumsum(p.explained_variance_ratio_)
        unlimited=1+int(np.searchsorted(cumulative,target,side="left"))
        return int(min(max(unlimited,1),cap)),int(unlimited),p

    def fit_unlabeled(self,train_x,train_mask,*,train_halfhour_blocks):
        a,m=check_measurements(train_x,train_mask)
        if a.shape[1]!=120 or a.shape[2]!=16 or len(a)<300 or not m.all():
            raise ValueError("Need >=300 full 120-second TRAIN episodes, 16 fully measured sensors")
        blocks=np.asarray(train_halfhour_blocks)
        if blocks.shape!=(len(a),):
            raise ValueError("One TRAIN half-hour block ID per temporal episode required")
        groups,group_ids=np.unique(blocks,return_inverse=True)
        if len(groups)<12:
            raise ValueError("Block bootstrap requires at least 12 real TRAIN half-hour blocks")
        innovation=self._innovations(a,m)
        k,unlimited,pca=self._pca_rank(innovation,self.target,self.max_k)
        seed_rng=np.random.default_rng(self.seed+117)
        boot=[]
        for _ in range(self.n_boot):
            draw=seed_rng.integers(len(groups),size=len(groups))
            member=np.concatenate([np.flatnonzero(group_ids==g) for g in draw])
            rank,_,_=self._pca_rank(innovation[member],self.target,self.max_k)
            boot.append(rank)
        converge=True
        n_iter=None
        if self.basis=="pca":
            load=pca.components_[:k].T
        else:
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always",ConvergenceWarning)
                ica=FastICA(n_components=k,whiten="unit-variance",
                            random_state=self.seed,max_iter=1000,tol=1e-4)
                ica.fit(innovation)
            converge=not any(issubclass(w.category,ConvergenceWarning) for w in observed)
            n_iter=int(ica.n_iter_)
            load=ica.components_.T
        norms=np.linalg.norm(load,axis=0)
        if not np.isfinite(norms).all() or np.any(norms<1e-10):
            raise ValueError("Non-identifiable degenerate ICA/PCA axis")
        load=load/norms[None,:]
        for j in range(k):
            significant=int(np.argmax(np.abs(load[:,j])))
            if load[significant,j]<0:
                load[:,j]*=-1
        self.loading=load.astype(np.float64)
        self.k=k
        self.route="TRAIN_INNOVATION_"+self.basis.upper()
        level,support=self.transform(a,m)
        threshold=[];width=[]
        for j in range(k):
            train_values=level[:,:,j][support[:,:,j]>0]
            q25,q75=np.quantile(train_values,[.25,.75])
            threshold.append(float(np.median(train_values)))
            width.append(float(max(.15,(q75-q25)/1.35)))
        self.tau=np.asarray(threshold)
        self.width=np.asarray(width)
        detail={
            "type":self.route,
            "k_observable_statistical_axes":k,
            "pca_rank_before_8_axis_cap":unlimited,
            "pca_train_explained_variance_ratios":pca.explained_variance_ratio_.tolist(),
            "pca_cumulative_variance_at_selected_k":float(
                np.cumsum(pca.explained_variance_ratio_)[k-1]),
            "rank_bootstrap_20_blocks":boot,
            "bootstrap_05_95_rank_interval":list(np.quantile(boot,[.05,.95]).tolist()),
            "n_halfhour_source_blocks":int(len(groups)),
            "n_correlated_train_120s_windows":len(a),
            "n_independent_continuous_recording_sessions":2,
            "gas_labels_or_controller_setpoints_used":False,
            "causal_timestep_projection_uses_only_current_and_initial_sensor":True,
            "ica_converged":converge,
            "ica_n_iter":n_iter,
            "thresholds":threshold,
            "widths":width,
            "signed_sensor_projections":load.tolist(),
            "physical_axis_identity_proven":False,
        }
        self.train_receipt=detail
        self.sha=hashlib.sha256(json.dumps(
            detail,sort_keys=True,allow_nan=False).encode()).hexdigest()
        return self

    def transform(self,x,mask):
        if self.loading is None:
            # During fit only, before thresholds/hashes are initialized.
            raise RuntimeError("No unlabeled TRAIN sensor projection")
        a,m=check_measurements(x,mask)
        if a.shape[-1]!=self.loading.shape[0] or a.shape[1]!=120:
            raise ValueError("Wrong physical source sensor count/time length")
        # The anchor comes only from the FIRST observed second, never future.
        initial=np.where(m[:,0,:],a[:,0,:],0.)
        differences=np.where(m,a-initial[:,None,:],0.)
        numerator=np.einsum("ntd,dk->ntk",differences,self.loading,optimize=True)
        magnitude=np.abs(self.loading)
        supported=np.einsum("ntd,dk->ntk",m.astype(float),magnitude,optimize=True)
        coverage=supported/np.maximum(1e-9,magnitude.sum(axis=0)[None,None,:])
        levels=numerator/np.maximum(supported,.2)
        if not np.isfinite(levels).all() or not np.isfinite(coverage).all():
            raise AssertionError("Nonfinite masked innovation concept")
        return levels.astype(np.float32),coverage.astype(np.float32)

    def static_fuzzy(self,x,mask):
        if self.tau is None:
            raise RuntimeError("Unlabeled source calibration missing")
        z,s=self.transform(x,mask)
        return .5+s*(expit((z-self.tau[None,None,:])/
                             self.width[None,None,:])-.5)

    def public_passport(self):
        if self.train_receipt is None:
            raise RuntimeError("Unfitted source")
        return {"source_receipt":self.train_receipt,
                "source_sha256":self.sha,
                "source_risk_labels_used":False,
                "number_of_discovered_source_slots":self.k,
                "initial_tau":self.tau.tolist(),
                "initial_width":self.width.tolist()}
