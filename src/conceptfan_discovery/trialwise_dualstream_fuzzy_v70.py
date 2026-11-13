"""V70: static dual-stream fuzzy source construction on 58 genuine gas trials.

Input is n independent five-minute trials x 60 real 5-second bins x
16 physical sensors. The source learner receives NO acetone or ethanol
concentrations, gas names, batch labels or outcome parameters. Actual
physical-source count is NOT inferred from PCA statistical K.
"""
from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA


class TrialOnlyDualStreamBank:
    def __init__(self, *, variance_fraction:float=.95,k_cap:int=4):
        if variance_fraction!=.95 or k_cap!=4:
            raise ValueError("Only frozen V70 train-only PCA95 K-cap4 source allowed")
        self.variance_fraction=variance_fraction
        self.k_cap=k_cap
        self.normalizer_median=None
        self.normalizer_iqr=None
        self.loadings=None
        self.K=None
        self.uncapped_K=None
        self.tau=None
        self.width=None
        self.sha=None
        self.receipt=None

    @staticmethod
    def validate(x):
        v=np.asarray(x,dtype=np.float64)
        if v.ndim!=3 or v.shape[1:]!=(60,16) or not np.isfinite(v).all():
            raise ValueError("Need physically observed 60 five-second bins x 16 gas sensors")
        return v

    def normalized(self,x):
        if self.normalizer_median is None:
            raise RuntimeError("TRAIN-only physical sensor normalizer unavailable")
        v=self.validate(x)
        return np.clip((v-self.normalizer_median[None,None,:])/
                       self.normalizer_iqr[None,None,:],-8,8)

    def fit_unlabeled(self,train_x):
        raw=self.validate(train_x)
        if len(raw)!=39:
            # The 39 original physical TRAIN trials are the statistical units.
            raise ValueError("V70 frozen physical unit contract requires 39 real TRAIN trials")
        flat=raw.reshape(-1,16)
        self.normalizer_median=np.median(flat,axis=0)
        q25,q75=np.quantile(flat,[.25,.75],axis=0)
        dispersion=q75-q25
        if np.any(~np.isfinite(dispersion)) or np.any(dispersion<=1e-8):
            raise ValueError("Sensor not variable across physically measured TRAIN trials")
        self.normalizer_iqr=dispersion
        z=self.normalized(raw)
        innovation=np.median(z[:,-15:,:],axis=1)-np.median(z[:,:15,:],axis=1)
        pca=PCA(svd_solver="full").fit(innovation)
        coverage=np.cumsum(pca.explained_variance_ratio_)
        self.uncapped_K=1+int(np.searchsorted(coverage,self.variance_fraction))
        self.K=min(self.k_cap,self.uncapped_K)
        weights=pca.components_[:self.K].T.copy()
        for k in range(self.K):
            pivot=int(np.argmax(np.abs(weights[:,k])))
            if weights[pivot,k]<0:weights[:,k]*=-1.
        self.loadings=weights
        channels=self._two_channels(raw)
        thresholds=[];widths=[]
        for k in range(channels.shape[-1]):
            source=channels[:,:,k].ravel()
            a,b,c=np.quantile(source,[.25,.5,.75])
            thresholds.extend([float(a),float(b),float(c)])
            width=max(.15,(c-a)/1.35)
            widths.extend([float(width)]*3)
        self.tau=np.asarray(thresholds,dtype=float)
        self.width=np.asarray(widths,dtype=float)
        source_receipt={
            "number_of_original_independent_training_trials":len(raw),
            "physical_sensor_count":16,
            "true_time_bins_per_trial":60,
            "no_individual_sensor_time_bin_treated_as_independent_trial":True,
            "source_algorithm":"PCA95 TRAIN late15 minus early15 normalized sensor medians",
            "source_K_is_statistical_not_number_of_gases":self.K,
            "pca_uncapped_rank":self.uncapped_K,
            "pca_train_coverage_at_K":float(coverage[self.K-1]),
            "pca_explained_variance_ratios":pca.explained_variance_ratio_.tolist(),
            "physical_sensor_16byK_signed_loadings":weights.tolist(),
            "normalization_train_16_sensor_medians":self.normalizer_median.tolist(),
            "normalization_train_16_sensor_iqr":self.normalizer_iqr.tolist(),
            "absolute_plus_causal_delta_channels":2*self.K,
            "train_only_Q25_Q50_Q75_fuzzy_states":6*self.K,
            "fuzzy_positive_widths":self.width.tolist(),
            "fuzzy_train_only_thresholds":self.tau.tolist(),
            "features_after_whole_and_late_summaries":12*self.K,
            "concentration_references_or_labels_used_to_fit":False,
            "unique_physical_concept_identification_claimed":False,
        }
        self.receipt=source_receipt
        self.sha=sha256(json.dumps(
            source_receipt,sort_keys=True,allow_nan=False).encode()).hexdigest()
        return self

    def _two_channels(self,x):
        v=self.normalized(x)
        absolute=np.einsum("ntd,dk->ntk",v,self.loadings,optimize=True)
        delta=np.einsum("ntd,dk->ntk",v-v[:,:1,:],self.loadings,optimize=True)
        return np.concatenate((absolute,delta),axis=-1)

    def source_channels(self,x):
        if self.loadings is None:
            raise RuntimeError("Fit anonymous TRAIN source directions first")
        return self._two_channels(x)

    def fuzzy(self,x):
        if self.tau is None:
            raise RuntimeError("Fit TRAIN physical-state thresholds first")
        z=self.source_channels(x)
        q=np.repeat(z,3,axis=-1)
        result=expit((q-self.tau[None,None,:])/self.width[None,None,:])
        if np.any(~np.isfinite(result)) or result.min()<0 or result.max()>1:
            raise AssertionError("Out of bounded fuzzy source range")
        return result

    @staticmethod
    def both_time_summaries(states):
        z=np.asarray(states,dtype=float)
        if z.ndim!=3 or z.shape[1]!=60:
            raise ValueError("Exactly 60 actual observed time bins required")
        return np.concatenate((z.mean(axis=1),z[:,-15:,:].mean(axis=1)),axis=1)

    def fuzzy_features(self,x):
        return self.both_time_summaries(self.fuzzy(x))

    def linear_features(self,x):
        return self.both_time_summaries(self.source_channels(x))

    def raw_features(self,x):
        return self.both_time_summaries(self.normalized(x))

    def public_passport(self):
        if self.receipt is None:raise RuntimeError("Not fitted")
        return dict(self.receipt,source_sha256=self.sha)
