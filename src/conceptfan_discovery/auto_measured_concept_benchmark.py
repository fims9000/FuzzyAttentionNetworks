"""Controlled unlabeled source grouping benchmark: 12-channel nonlinear signals.

Extends the already executed V30 nonlinear switching/Laplace simulator with
two hidden nuisance/recording-process source pairs and a lone distractor.
Only the *evaluation* knows source identities. The proposal algorithm only
receives x, observation masks, and known-existing sensor indices [0..3].
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from conceptfan_discovery.nonlinear_source_bench_v30 import nonlinear_world

DATA_SEED = 20261113
N_CHANNELS = 12
TRAIN_N = 1700
VALID_N = 1200
TEST_N = 900
M_TRAIN = 20267201
M_VAL1 = 20267202
M_VAL2 = 20267203
M_TEST_BASE = 20267900
STRESS = ("complete", "MCAR20", "MCAR40", "MNAR", "block8")


def _extra_non_gaussian_source(
    rng: np.random.Generator,n: int,t: int, *, nonlinear_scale: float
) -> np.ndarray:
    z = np.empty((n,t),dtype=np.float32)
    z[:,0] = rng.laplace(0,.55,n).astype(np.float32)
    regime = rng.choice([-1.,1.],size=n)
    phase = rng.uniform(-np.pi,np.pi,size=n)
    for k in range(1,t):
        regime = np.where(rng.random(n)<.045,-regime,regime)
        shock = (rng.random(n)<.10)*rng.choice([-1.,1.],size=n)*rng.uniform(.5,1.4,size=n)
        z[:,k] = (.40*z[:,k-1]+nonlinear_scale*np.tanh(1.4*z[:,k-1])
                   +rng.laplace(0,.42,n)+.17*regime
                   +.12*np.sin(k/3+phase)+shock).astype(np.float32)
    return z


def extended_nonlinear_world(*, seed:int=DATA_SEED,
                              train_n:int=TRAIN_N,valid_n:int=VALID_N,
                              test_n:int=TEST_N)->dict:
    base=nonlinear_world(seed=seed,n_train=train_n,n_valid=valid_n,n_test=test_n)
    rng=np.random.default_rng(seed+49219)
    original=base["x"]
    n,t,d=original.shape
    assert d==7
    nuisance=_extra_non_gaussian_source(rng,n,t,nonlinear_scale=.68)
    artifact=_extra_non_gaussian_source(rng,n,t,nonlinear_scale=.73)
    result=np.empty((n,t,N_CHANNELS),dtype=np.float32)
    result[:,:,:7]=original
    for target,z,noise in ((7,nuisance,.37),(8,nuisance,.40),
                           (9,artifact,.37),(10,artifact,.40)):
        result[:,:,target]=(z+rng.normal(0,noise,(n,t))).astype(np.float32)
    # Unmatched singleton readout: not a stable independently corroborated source.
    result[:,:,11]=rng.laplace(0.,.9,(n,t)).astype(np.float32)
    known=base["true_concept_trajectories"]
    known_mean=known[:,:,:2].mean(axis=1)
    instrument_proxy=expit(1.1*artifact).mean(axis=1)
    base_logits=4.5*(known_mean[:,0]-.5)-4.0*(known_mean[:,1]-.5)-.12
    extra_logits=base_logits+5.0*(instrument_proxy-.5)
    artifact_outcome=(rng.random(n)<expit(extra_logits)).astype(np.int64)
    full={
        "x":result,
        "outcomes":{
            "physical_extra":base["labels_predictive"].copy(),
            "complete_nuisance":base["labels_nuisance"].copy(),
            "artifact_predictive":artifact_outcome,
        },
        "oracle_concepts_for_posthoc_TEST_audit_only":known,
        "oracle_nuisance_for_posthoc_TEST_audit_only":expit(1.1*nuisance).astype(np.float32),
        "oracle_artifact_for_posthoc_TEST_audit_only":expit(1.1*artifact).astype(np.float32),
        "train":base["train"],
        "valid_env1":base["valid_env1"],
        "valid_env2":base["valid_env2"],
        "test":base["test"],
        "generator_family":"nonlinear_v30_extended_three_unassigned_source_families",
        "evaluation_only_true_groups":{
            "physical_extra":[4,5,6],
            "nonpredictive_measurement":[7,8],
            "recording_artifact":[9,10],
            "unpaired_noise":[11],
        },
        "physical_vs_recording_interpretation_not_inferable_from_prediction":True,
    }
    return full


def masked_histories(x:np.ndarray,mode:str,*,seed:int)->np.ndarray:
    x=np.asarray(x)
    if x.ndim!=3 or x.shape[1:]!=(16,N_CHANNELS) or not np.isfinite(x).all():
        raise ValueError("Expected finite 16-step, 12-channel synthetic histories")
    rng=np.random.default_rng(seed)
    n,t,d=x.shape
    m=np.ones(x.shape,dtype=bool)
    if mode=="complete":
        return m
    if mode.startswith("MCAR"):
        return rng.random(x.shape)>=int(mode[4:])/100.0
    if mode=="MNAR":
        probability=.06+.75*expit(1.65*x)
        return rng.random(x.shape)>=probability
    if mode=="block8":
        m=rng.random(x.shape)>=.08
        st=rng.integers(0,t-8+1,size=(n,d))
        for i in range(n):
            for j in range(d):
                m[i,st[i,j]:st[i,j]+8,j]=False
        return m
    raise ValueError("Unregistered missingness mechanism")


def intervene(x:np.ndarray,scenario:str)->np.ndarray:
    result=np.asarray(x,dtype=np.float32).copy()
    if scenario=="stable":
        return result
    if scenario=="source_common_sign_flip":
        result[:,:,4:7]*=-1.
        return result
    if scenario=="artifact_common_sign_flip":
        result[:,:,9:11]*=-1.
        return result
    if scenario=="source_completely_unobserved":
        return result
    raise ValueError("Unregistered source intervention")


def apply_missing_source(mask:np.ndarray,scenario:str)->np.ndarray:
    m=np.asarray(mask).copy()
    if scenario=="source_completely_unobserved":
        m[:,:,4:7]=False
    return m


def raw_measured_features(x:np.ndarray,m:np.ndarray)->np.ndarray:
    if x.shape!=m.shape or x.shape[-1]!=N_CHANNELS:
        raise ValueError("Misaligned raw-input baseline features")
    obs=np.where(m,x,0.)
    count=np.maximum(1,m.sum(axis=1))
    recent=np.maximum(1,m[:,-6:,:].sum(axis=1))
    full=obs.sum(axis=1)/count
    late=obs[:,-6:,:].sum(axis=1)/recent
    return np.concatenate((full,late,m.mean(axis=1)),axis=1)
