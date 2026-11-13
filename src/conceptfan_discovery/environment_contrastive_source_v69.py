"""V69: environment-contrastive anonymous source generator for UCI270.

Replace global PCA/ICA directions with TRAIN-only sensor directions that have
large WITHIN-acquisition-batch variation relative to global variation. This
uses historical batch IDs as environment metadata, never gas labels, ppm,
outcomes, VALID or TEST rows. 128 inputs are precomputed sensor descriptors,
NOT temporal timesteps. Derived fuzzy states are not chemical identities.

V69 is post-V67 already-viewed TEST exploratory work; see frozen protocol.
"""
from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA

from conceptfan_discovery.static_gas_drift_concepts_v67 import (
    SOURCE_K_CAP,SOURCE_PCA_TARGET,FrozenSixGasSourceBank
)


class TrainOnlyDriftContrastiveBank:
    """Measured-view covariance quotient: variation within / total variation."""

    def __init__(self, *, ridge_fraction:float=.10):
        if not np.isfinite(ridge_fraction) or ridge_fraction!=.10:
            raise ValueError("Only preregistered ridge_fraction=.10 is supported")
        self.ridge_fraction=float(ridge_fraction)
        self.raw_median=None
        self.raw_iqr=None
        self.global_mean=None
        self.loadings=None
        self.K=None
        self.pca_K_uncapped=None
        self.thresholds=None
        self.widths=None
        self.receipt=None
        self.sha=None

    def normalize(self,raw):
        if self.raw_median is None:
            raise RuntimeError("Source normalizer requires unlabeled TRAIN fitting")
        x=np.asarray(raw,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=128 or not np.isfinite(x).all():
            raise ValueError("Expected 128 actually measured finite descriptors")
        return np.clip((x-self.raw_median[None,:])/
                       self.raw_iqr[None,:],-8.,8.)

    def measured_views(self,x):
        return FrozenSixGasSourceBank.sensor_views(self.normalize(x))

    def fit_unlabeled(self,train_x,train_batch_ids):
        x=np.asarray(train_x,dtype=np.float64)
        batch=np.asarray(train_batch_ids)
        if x.ndim!=2 or x.shape[1]!=128 or len(x)<1000:
            raise ValueError("Need >1000 real original TRAIN sensor exposures")
        if len(batch)!=len(x) or not np.isfinite(x).all():
            raise ValueError("Mismatch of TRAIN sensor and environment IDs")
        try:
            environment=batch.astype(int)
        except (TypeError,ValueError,OverflowError) as error:
            raise ValueError("Only genuine historical integer TRAIN environment IDs") from error
        if not np.array_equal(environment,batch):
            raise ValueError("Historical batch IDs must be integers")
        if np.any(environment<1) or np.any(environment>7):
            raise ValueError("VALID and TEST batch metadata forbidden in source fitting")
        unique=np.unique(environment)
        if len(unique)<3:
            raise ValueError("At least three historical physical acquisition batches required")
        self.raw_median=np.median(x,axis=0)
        q25,q75=np.quantile(x,[.25,.75],axis=0)
        raw_iqr=q75-q25
        self.raw_iqr=np.where(raw_iqr>1e-9,raw_iqr,1.)
        view=self.measured_views(x)
        n,d=view.shape
        assert d==32
        self.global_mean=view.mean(axis=0)
        centered=view-self.global_mean

        pca=PCA(svd_solver="full").fit(view)
        explained=np.cumsum(pca.explained_variance_ratio_)
        self.pca_K_uncapped=int(np.searchsorted(explained,SOURCE_PCA_TARGET)+1)
        self.K=min(self.pca_K_uncapped,SOURCE_K_CAP)

        within=np.zeros((d,d),dtype=np.float64)
        between=np.zeros((d,d),dtype=np.float64)
        n_env={}
        for b in unique:
            group=view[environment==b]
            n_env[str(int(b))]=len(group)
            groupmean=group.mean(axis=0)
            demeaned=group-groupmean
            within+=demeaned.T@demeaned
            diff=groupmean-self.global_mean
            between+=len(group)*np.outer(diff,diff)
        within/=n
        between/=n
        # Covariance quotient depends exclusively on unlabeled TRAIN sensors
        # and true TRAIN environment batch IDs.
        regularizer=self.ridge_fraction*np.trace(within)/d
        if not np.isfinite(regularizer) or regularizer<=1e-10:
            raise ValueError("No nondegenerate TRAIN within-batch response variance")
        total=within+between+regularizer*np.eye(d)
        values,vectors=np.linalg.eigh(total)
        if values.min()<=0:
            raise ValueError("Regularized TRAIN covariance not positive definite")
        invsqrt=(vectors/np.sqrt(values)[None,:])@vectors.T
        quotient=invsqrt@within@invsqrt
        quotient=(quotient+quotient.T)/2.
        ratio,eigenbasis=np.linalg.eigh(quotient)
        order=np.argsort(-ratio,kind="stable")
        sorted_ratio=ratio[order]
        basis=invsqrt@eigenbasis[:,order[:self.K]]
        lengths=np.linalg.norm(basis,axis=0)
        if np.any(lengths<1e-12) or not np.isfinite(lengths).all():
            raise ValueError("Degenerate temporal batch nuisance-contrastive source")
        basis=basis/lengths[None,:]
        for j in range(self.K):
            largest=np.argmax(np.abs(basis[:,j]))
            if basis[largest,j]<0:basis[:,j]*=-1.
        self.loadings=basis
        source=(view-self.global_mean)@basis
        q=np.quantile(source,[.25,.50,.75],axis=0).T
        self.thresholds=q
        self.widths=np.maximum(.15,(q[:,2]-q[:,0])/1.35)
        self.receipt={
            "n_unlabeled_train_exposures":n,
            "training_environment_batches":sorted(int(b) for b in unique),
            "n_sensor_samples_by_training_environment":n_env,
            "measured_source_channels":"16_steady_views_plus_16_engineered_transient_views",
            "128_input_values_are_engineered_sensor_descriptors_not_timesteps":True,
            "any_gas_labels_or_ppm_supplied_to_source_fit":False,
            "unsupervised_objective":"top eigenvectors of T^(-1/2) * W * T^(-1/2)",
            "W":"TRAIN within-acquisition-batch sensor covariance",
            "B":"TRAIN between-acquisition-batch sensor mean covariance",
            "T":"W+B+0.10*trace(W)/32*I",
            "ridge_fraction_fixed_before_V69_outcomes":self.ridge_fraction,
            "eigenvalue_ratios_descending":sorted_ratio.tolist(),
            "chosen_pca95_source_rank_K":self.K,
            "pca95_uncapped_K":self.pca_K_uncapped,
            "pca95_train_coverage_at_K":float(explained[self.K-1]),
            "source_basis_32_by_K":basis.tolist(),
            "TRAIN_measured_view_mean":self.global_mean.tolist(),
            "TRAIN_raw_128_medians":self.raw_median.tolist(),
            "TRAIN_raw_128_iqr":self.raw_iqr.tolist(),
            "train_fuzzy_25_50_75_quantiles":q.tolist(),
            "train_fuzzy_widths":self.widths.tolist(),
            "n_fuzzy_states":3*self.K,
            "statistical_source_axes_are_not_chemical_gas_identities":True,
            "post_V67_inspected_TEST_exploratory":True,
        }
        self.sha=sha256(json.dumps(
            self.receipt,sort_keys=True,allow_nan=False).encode()).hexdigest()
        return self

    def source_levels(self,raw):
        if self.loadings is None:
            raise RuntimeError("Unfitted unlabeled sensor basis")
        z=(self.measured_views(raw)-self.global_mean)@self.loadings
        if not np.isfinite(z).all():
            raise AssertionError("Invalid source projection")
        return z

    def fuzzy(self,raw):
        if self.thresholds is None:
            raise RuntimeError("Unfitted unlabeled fuzzy calibrator")
        z=self.source_levels(raw)
        result=expit((z[:,:,None]-self.thresholds[None,:,:])/
                     self.widths[None,:,None]).reshape(len(z),3*self.K)
        if not np.isfinite(result).all() or np.any(result<0) or np.any(result>1):
            raise AssertionError("Invalid bounded fuzzy source states")
        return result

    def public_passport(self):
        if self.receipt is None:
            raise RuntimeError("Source bank not yet trained")
        return dict(self.receipt,source_sha256=self.sha)
