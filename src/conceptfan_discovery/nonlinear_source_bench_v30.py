"""V30 nonlinear switching/shock temporal source benchmark (never clinical data).

Three independent non-Gaussian latent dynamical sources, two measurements each,
plus a third separately calibrated and independently-noisy source3 anchor.
Original K2 concept module may only see first FOUR sensor channels.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

N_STEPS=16
N_CHANNELS=7
SEED=20261110
TRAIN=1800
VALID=1200
TEST=900
STRESS=("complete","MCAR20","MCAR40","MNAR","block8")


def nonlinear_world(*, seed:int=SEED,n_train:int=TRAIN,
                    n_valid:int=VALID,n_test:int=TEST)->dict:
    if min(n_train,n_valid,n_test)<80 or n_valid%2:
        raise ValueError("Require 3 nonempty partitions and even two-environment VALID split")
    n=n_train+n_valid+n_test
    rng=np.random.default_rng(seed)
    z=np.empty((n,N_STEPS,3),dtype=np.float32)
    regimes=rng.choice(np.array([-1.,1.],dtype=np.float32),size=(n,3))
    phases=rng.uniform(-np.pi,np.pi,size=(n,3))
    z[:,0,:]=(rng.laplace(0.,.55,(n,3))+.15*regimes).astype(np.float32)
    for t in range(1,N_STEPS):
        # Nonlinear, non-Gaussian (Laplace shocks), driven, switching regime
        # unlike all V25-V29 stationary Gaussian AR(1) synthetic benchmarks.
        switches=rng.random((n,3))<.045
        regimes=np.where(switches,-regimes,regimes)
        shock=(rng.random((n,3))<.11)*rng.choice([-1.,1.],(n,3))*rng.uniform(.5,1.5,(n,3))
        innovation=rng.laplace(0.,.4,(n,3))
        seasonal=.16*np.sin((t/3.)+phases)
        z[:,t,:]=(.43*z[:,t-1,:]
                   +.68*np.tanh(1.35*z[:,t-1,:])
                   +innovation+.19*regimes+shock+seasonal).astype(np.float32)
    c=expit(1.1*z).astype(np.float32)
    x=np.empty((n,N_STEPS,N_CHANNELS),dtype=np.float32)
    for k in range(3):
        for sensor in range(2):
            x[:,:,2*k+sensor]=z[:,:,k]+rng.normal(0,.35,(n,N_STEPS)).astype(np.float32)
    # The assumed third independent sensor has another simulated readout
    # channel and independent draw of measurement error: no shared noise
    # beyond the underlying true latent source. NOT real device provenance.
    x[:,:,6]=z[:,:,2]+rng.normal(0,.45,(n,N_STEPS)).astype(np.float32)
    means=c.mean(axis=1)
    baseline=4.5*(means[:,0]-.5)-4.0*(means[:,1]-.5)-.12
    omitted=baseline+5.0*(means[:,2]-.5)
    uniforms=rng.random(n)
    y_complete=(uniforms<expit(baseline)).astype(np.int64)
    y_predictive=(uniforms<expit(omitted)).astype(np.int64)
    out={
        "x":x,"true_concept_trajectories":c,
        "labels_nuisance":y_complete,"labels_predictive":y_predictive,
        "train":np.arange(n_train),
        "valid_env1":np.arange(n_train,n_train+n_valid//2),
        "valid_env2":np.arange(n_train+n_valid//2,n_train+n_valid),
        "test":np.arange(n_train+n_valid,n),
        "generator_family":"nonlinear_switching_laplace_shocks_v30",
        "sensor_anchor_independent_noise_by_generator_assumption":True,
    }
    ids=[out[name] for name in ("train","valid_env1","valid_env2","test")]
    if len(np.unique(np.concatenate(ids)))!=n:
        raise AssertionError("Overlapping split identities")
    if not (len(np.unique(y_complete))==2 and len(np.unique(y_predictive))==2):
        raise AssertionError("Outcome law collapsed into single class")
    return out


def mask_view(x:np.ndarray,mode:str,*,seed:int)->np.ndarray:
    x=np.asarray(x)
    if x.ndim!=3 or x.shape[-2:]!=(N_STEPS,N_CHANNELS) or not np.isfinite(x).all():
        raise ValueError("Expected 16 by 7 finite sensor histories")
    rng=np.random.default_rng(seed)
    n,t,k=x.shape
    m=np.ones((n,t,k),dtype=bool)
    if mode=="complete":
        return m
    if mode.startswith("MCAR"):
        fraction=int(mode[4:])/100.
        return rng.random(m.shape)>=fraction
    if mode=="MNAR":
        p=.06+.75*expit(1.65*x)
        return rng.random(m.shape)>=p
    if mode=="block8":
        m=rng.random(m.shape)>=.08
        st=rng.integers(0,t-8+1,size=(n,k))
        for i in range(n):
            for j in range(k):
                m[i,st[i,j]:st[i,j]+8,j]=False
        return m
    raise ValueError(f"Unsupported V30 mechanism: {mode}")


def instrument_fault(x:np.ndarray,scenario:str)->np.ndarray:
    data=np.asarray(x,dtype=np.float32).copy()
    if scenario=="stable" or scenario=="nuisance":
        pass
    elif scenario=="single_pair_sign_flip":
        data[:,:,5]*=-1.
    elif scenario=="single_pair_bias":
        data[:,:,5]+=2.
    elif scenario=="both_pair_sign_flip":
        data[:,:,4:6]*=-1.
    elif scenario=="anchor_only_sign_flip":
        data[:,:,6]*=-1.
    elif scenario=="all_three_sign_flip":
        data[:,:,4:7]*=-1.
    else:
        raise ValueError("Unlisted V30 scenario")
    return data


def raw_observed_features(x:np.ndarray,m:np.ndarray)->np.ndarray:
    if x.shape!=m.shape or x.shape[-1]!=N_CHANNELS:
        raise ValueError("Misaligned sensor baseline")
    observed=np.where(m,x,0.)
    denom=np.maximum(1,m.sum(axis=1))
    dlate=np.maximum(1,m[:,-6:,:].sum(axis=1))
    full=observed.sum(axis=1)/denom
    late=observed[:,-6:,:].sum(axis=1)/dlate
    return np.concatenate((full,late,m.mean(axis=1)),axis=1)
