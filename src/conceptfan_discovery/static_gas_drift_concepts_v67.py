"""V67 independent new-gas-drift source vocabulary.

UCI270 contains 128 engineered FEATURES (16 physical sensors times 8
per-sensor descriptors), NOT a time trajectory. No recurrence, timestamps
or real gas labels are accepted by this source learner.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import re
import warnings
from zipfile import ZipFile

import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA,FastICA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

GAS_NAMES=("ethanol","ethylene","ammonia","acetaldehyde","acetone","toluene")
SOURCE_SEED=20261967
SOURCE_PCA_TARGET=.95
SOURCE_K_CAP=12
UCI_N=13910
BATCHES=tuple(range(1,11))
FEATURES_PER_SENSOR=8
N_SENSORS=16
N_RAW_FEATURES=128
FMT=re.compile(r"(?:^|/)batch([1-9]|10)\.dat$")


def parse_uci_record(line:bytes):
    parts=line.decode("utf-8").strip().split()
    if len(parts)!=129:
        raise ValueError("Require class;ppm plus all 128 explicitly observed features")
    lab,concentration=parts[0].split(";",1)
    gas=int(lab); ppm=float(concentration)
    if not 1<=gas<=6 or not np.isfinite(ppm) or ppm<=0:
        raise ValueError("Invalid official six-gas class or physical concentration")
    values=np.empty(N_RAW_FEATURES,dtype=np.float64)
    seen=np.zeros(N_RAW_FEATURES,dtype=bool)
    for entry in parts[1:]:
        k,v=entry.split(":",1)
        key=int(k)-1
        if not 0<=key<N_RAW_FEATURES or seen[key]:
            raise ValueError("Feature index absent from 1..128 or repeated")
        seen[key]=True
        values[key]=float(v)
    if not seen.all() or not np.isfinite(values).all():
        raise ValueError("Missing or nonfinite actual sensor response features")
    return gas,ppm,values


def read_original_ten_batches(path):
    with ZipFile(path) as archive:
        names={int(m.group(1)):name
               for name in archive.namelist()
               if (m:=FMT.search(name)) and not name.endswith("/")}
        if set(names)!=set(BATCHES):
            raise ValueError("Exactly 10 original UCI270 batch files required")
        ys=[];ppm=[];features=[];batches=[];counts={}
        for b in BATCHES:
            seen=Counter()
            with archive.open(names[b]) as inp:
                for line in inp:
                    if not line.strip():continue
                    gas,p,x=parse_uci_record(line)
                    ys.append(gas);ppm.append(p);features.append(x)
                    batches.append(b)
                    seen[gas]+=1
            counts[b]={str(k):int(seen[k]) for k in range(1,7)}
        y=np.asarray(ys,dtype=int)
        concentration=np.asarray(ppm,dtype=float)
        x=np.asarray(features,dtype=np.float64)
        batch=np.asarray(batches,dtype=np.int8)
    if x.shape!=(UCI_N,N_RAW_FEATURES):
        raise ValueError("Official UCI270 measurement count / schema differs from protocol")
    groups={
        "TRAIN":np.flatnonzero(batch<=7),
        "VALID":np.flatnonzero((batch>=8)&(batch<=9)),
        "TEST":np.flatnonzero(batch==10)}
    if any(set(np.unique(y[ix]))!=set(range(1,7)) for ix in groups.values()):
        raise ValueError("All six real pure-gas classes needed in each temporal partition")
    return x,y,concentration,batch,groups,{
        "n_actual_recorded_exposures":len(y),
        "counts_by_batch_and_original_label":counts,
        "per_sensor_features_are_precomputed_not_actual_timestamps":True,
        "sources_from_gas_identity_or_concentration":False,
        "time_split_batches":{"TRAIN":list(range(1,8)),"VALID":[8,9],"TEST":[10]},
    }


class FrozenSixGasSourceBank:
    def __init__(self):
        self.raw_median=None
        self.raw_iqr=None
        self.n_constant_raw_features=None
        self.ica=None
        self.sign=None
        self.K=None
        self.uncapped_K=None
        self.quantiles=None
        self.widths=None
        self.receipt=None
        self.sha=None

    def normalize(self,x):
        if self.raw_median is None:
            raise RuntimeError("Fit on TRAIN input sensors first")
        v=np.asarray(x,dtype=float)
        if v.ndim!=2 or v.shape[1]!=128 or not np.isfinite(v).all():
            raise ValueError("Require 128 real finite engineered sensor features")
        return np.clip((v-self.raw_median)/self.raw_iqr,-8.,8.)

    @staticmethod
    def sensor_views(normalized):
        v=np.asarray(normalized,dtype=float)
        if v.ndim!=2 or v.shape[1]!=128:
            raise ValueError("Real 16-sensor / 8-engineered-feature schema required")
        response=v.reshape(-1,16,8)
        # Feature descriptors are NOT chronological time points:
        steady=response[:,:,:2].mean(axis=2)
        transient=response[:,:,2:].mean(axis=2)
        return np.concatenate([steady,transient],axis=1)

    def fit_unlabeled(self,train_x):
        v=np.asarray(train_x,dtype=np.float64)
        if v.ndim!=2 or v.shape[1]!=128 or len(v)<1000 or not np.isfinite(v).all():
            raise ValueError("Insufficient valid TRAIN source measurements")
        self.raw_median=np.median(v,axis=0)
        q25,q75=np.quantile(v,[.25,.75],axis=0)
        delta=q75-q25
        self.n_constant_raw_features=int(np.sum(delta<=1e-9))
        self.raw_iqr=np.where(delta>1e-9,delta,1.)
        view=self.sensor_views(self.normalize(v))
        pca=PCA(svd_solver="full").fit(view)
        cumulative=np.cumsum(pca.explained_variance_ratio_)
        self.uncapped_K=int(1+np.searchsorted(cumulative,SOURCE_PCA_TARGET))
        self.K=int(min(SOURCE_K_CAP,self.uncapped_K))
        self.ica=FastICA(
            n_components=self.K,whiten="unit-variance",
            random_state=SOURCE_SEED,max_iter=1000,tol=1e-4)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always",ConvergenceWarning)
            z=self.ica.fit_transform(view)
        converged=not any(issubclass(q.category,ConvergenceWarning) for q in w)
        if not np.isfinite(z).all():
            raise ValueError("Unsupervised real source ICA generated nonfinite values")
        weights=self.ica.components_
        anchors=np.argmax(np.abs(weights),axis=1)
        orientation=np.sign(weights[np.arange(self.K),anchors])
        self.sign=np.where(orientation<0,-1.,1.)
        z*=self.sign
        quantiles=np.quantile(z,[.25,.50,.75],axis=0).T
        self.quantiles=quantiles
        self.widths=np.maximum(.15,(quantiles[:,2]-quantiles[:,0])/1.35)
        content={
            "training_measurements":len(v),
            "source_feature_type":"16_MEASURED_SENSOR_x_8_ENGINEERED_FEATURES_NOT_TIME_SERIES",
            "two_sensor_views":"16_NORMALIZED_STEADY_MEANS_PLUS_16_NORMALIZED_TRANSIENT_DESCRIPTORS",
            "training_source_labels_or_gas_ppm_used":False,
            "k_from_train_variance_95pct":self.K,
            "unclipped_pca_rank_to_95pct":self.uncapped_K,
            "pca_k_cap":SOURCE_K_CAP,
            "explained_variance_at_K":float(cumulative[self.K-1]),
            "ica_converged":bool(converged),
            "ica_iterations":int(self.ica.n_iter_),
            "fixed_seed":SOURCE_SEED,
            "not_a_physical_gas_source_count":True,
            "fuzzy_memberships_per_axis":3,
            "n_fuzzy_memberships":3*self.K,
            "source_raw_medians":self.raw_median.tolist(),
            "source_raw_iqrs":self.raw_iqr.tolist(),
            "n_nonvarying_engineered_features":self.n_constant_raw_features,
            "source_ica_projection_weights":(self.ica.components_*
                                             self.sign[:,None]).tolist(),
            "source_ica_center_mean":self.ica.mean_.tolist(),
            "source_quantiles_q25_q50_q75":quantiles.tolist(),
            "source_calibrated_positive_widths":self.widths.tolist(),
            "source_axes_have_no_labels_or_identified_chemical_identity":True,
        }
        self.receipt=content
        self.sha=sha256(json.dumps(content,sort_keys=True,
                                   allow_nan=False).encode()).hexdigest()
        return self

    def source_levels(self,x):
        if self.ica is None:
            raise RuntimeError("TRAIN source not fitted")
        z=self.ica.transform(self.sensor_views(self.normalize(x)))
        z=z*self.sign
        if not np.isfinite(z).all():
            raise ValueError("ICA source output nonfinite")
        return z

    def fuzzy(self,x):
        if self.quantiles is None:
            raise RuntimeError("TRAIN fuzzy state calibration missing")
        z=self.source_levels(x)
        c=expit((z[:,:,None]-self.quantiles[None,:,:])/
                self.widths[None,:,None]).reshape(len(z),3*self.K)
        if not np.isfinite(c).all() or c.min()<0 or c.max()>1:
            raise AssertionError("Invalid bounded source memberships")
        return c

    def public_passport(self):
        if self.receipt is None:
            raise RuntimeError("No TRAIN-only source definition")
        return {**self.receipt,"source_sha256":self.sha}


def one_vs_rest_source_auc_matrix(z,y):
    val=np.asarray(z,float)
    gas=np.asarray(y,int)
    if val.ndim!=2 or len(gas)!=len(val):
        raise ValueError("Gas reference and frozen source dimensions disagree")
    if set(np.unique(gas))!=set(range(1,7)):
        raise ValueError("Need all six real physical gases")
    out=np.empty((6,val.shape[1]))
    for g in range(1,7):
        indicator=(gas==g).astype(int)
        for j in range(val.shape[1]):
            out[g-1,j]=float(roc_auc_score(indicator,val[:,j]))
    return out


def fix_semantic_alignment_from_valid(valid_z,valid_y):
    auc=one_vs_rest_source_auc_matrix(valid_z,valid_y)
    signal=np.abs(auc-.5)
    assigned_gas,assigned_axis=linear_sum_assignment(-signal)
    mapping={}
    for g,k in zip(assigned_gas,assigned_axis):
        g=int(g)+1;k=int(k)
        mapping[str(g)]={
            "anonymous_ica_source_axis":k,
            "valid_direction":1 if auc[g-1,k]>=.5 else -1,
            "VALID_one_vs_rest_auc_unoriented":float(auc[g-1,k]),
            "VALID_oriented_auc":float(max(auc[g-1,k],1-auc[g-1,k])),
            "gas_name_applied_posthoc_using_valid_labels":GAS_NAMES[g-1],
        }
    return mapping,auc


def audit_fixed_mapping(mapping,valid_matrix,test_z,test_y,test_conc,
                        *,permutation_seed=20261967):
    actual=one_vs_rest_source_auc_matrix(test_z,test_y)
    rng=np.random.default_rng(permutation_seed)
    shuffled=one_vs_rest_source_auc_matrix(
        test_z,rng.permutation(test_y))
    results={}
    for cls in range(1,7):
        if str(cls) not in mapping:
            results[str(cls)]={
                "status":"ABSTAIN_NO_MATCHED_STATISTICAL_SOURCE",
                "test_auc":None}
            continue
        fit=mapping[str(cls)]
        k=fit["anonymous_ica_source_axis"]
        sign=fit["valid_direction"]
        oriented=float(actual[cls-1,k] if sign>0 else 1-actual[cls-1,k])
        shuffled_auc=float(shuffled[cls-1,k] if sign>0 else 1-shuffled[cls-1,k])
        mask=test_y==cls
        values=np.asarray(test_conc[mask],float)
        if mask.sum()>20 and np.std(values)>1e-8:
            cond=float(spearmanr(test_z[mask,k]*sign,values).statistic)
        else:
            cond=None
        results[str(cls)]={
            "status":"VALID_LABEL_MATCH_FIXED_BEFORE_TEST_NOT_AUTONOMOUS_CHEMICAL_IDENTITY",
            "anonymous_ica_axis":k,
            "VALID_oriented_auc":fit["VALID_oriented_auc"],
            "TEST_oriented_auc_no_rematching":oriented,
            "TEST_shuffled_labels_auc_control":shuffled_auc,
            "TEST_in_gas_concentration_spearman":cond,
            "TEST_gas_sample_count":int(mask.sum()),
            "physical_semantic_identity_proven":False,
        }
    return {
        "mapping":results,
        "all_6xK_VALID_auc_unoriented":valid_matrix.tolist(),
        "all_6xK_TEST_auc_unoriented":actual.tolist(),
        "all_6xK_TEST_shuffled_class_auc_unoriented":shuffled.tolist(),
        "valid_labels_used_only_for_postfit_semantic_mapping":True,
        "test_gas_labels_not_used_to_choose_source_or_matching":True,
        "unique_physical_gas_concepts_certified":False,
    }
