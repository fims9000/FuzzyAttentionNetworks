"""Three-environment Gaussian AR latent-source simulator (synthetic only).

Environment ID is observed; latent sources, mixing and environment variances
are strictly evaluation-only. There are NO treatment/action targets.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class EnvironmentSimulation:
    observations: np.ndarray
    environment: np.ndarray
    labels: np.ndarray
    true_sources: np.ndarray
    mixing: np.ndarray
    source_variances: np.ndarray
    mode: str


def make_environment_scm(*,k:int,seed:int,per_environment:int=1250,
    history:int=12,mode:str="distinct",measurement_noise:float=.11,
    mixing:np.ndarray|None=None,missing_fraction:float=0.0)->EnvironmentSimulation:
    if (not 2<=k<=8 or per_environment<200 or history<6 or
        not 0<=missing_fraction<1 or measurement_noise<0):
        raise ValueError("Invalid synthetic generator regime")
    if mode not in ("distinct","no_shift","tied_profiles","sensor_noise_shift","mean_only_shift","noise_shift_aligned"):
        raise ValueError("Unknown simulated environment variation")
    rng=np.random.default_rng(seed)
    d=k+3
    if mixing is None:
        b=rng.normal(scale=.28,size=(d,k))
        b[:k]+=1.3*np.eye(k)
    else:
        b=np.asarray(mixing,dtype=np.float64)
        if b.shape!=(d,k) or np.linalg.matrix_rank(b)<k:
            raise ValueError("Must supply full rank sensor mixing")
    b=b/np.linalg.norm(b,axis=0)[None,:]
    delta=.9+.2*np.arange(k)
    ratio=np.linspace(.22,3.4,k)
    base=np.full(k,.65)
    variance=np.stack((base,base+delta,base+delta*ratio))
    if mode in ("no_shift","mean_only_shift","sensor_noise_shift","noise_shift_aligned"):
        variance[:]=base
    if mode=="tied_profiles":
        ratio[:2]=1.0
        variance[2]=base+delta*ratio
    n=3*per_environment
    env=np.repeat(np.arange(3),per_environment)
    forecast=4
    steps=history+forecast
    source=np.zeros((n,steps,k))
    raw_noise=rng.normal(size=(n,steps,k))
    for e in range(3):
        idx=np.flatnonzero(env==e)
        std=np.sqrt(variance[e])
        source[idx,0]=std*raw_noise[idx,0]
        for t in range(1,steps):
            source[idx,t]=(.75*source[idx,t-1]
                            + np.sqrt(1-.75**2)*std*raw_noise[idx,t])
    # Same non-isotropic sensor noise matrix across environments in valid modes.
    noise_mixing=rng.normal(scale=.16,size=(d,d))
    sensor_noise=(rng.normal(size=(n,history,d))@noise_mixing.T)*measurement_noise/.16
    if mode=="noise_shift_aligned":
        # Adversarial observational equivalence: unchanged latent factors,
        # but sensor noise changes along the same mixing directions.
        # A covariance-only method cannot distinguish this from source shifts.
        for e in (1,2):
            idx=np.flatnonzero(env==e)
            delta_e=delta*(np.ones(k) if e==1 else ratio)
            colored=rng.normal(size=(len(idx),history,k))*np.sqrt(4.0*delta_e)
            sensor_noise[idx]+=colored@b.T
    if mode=="sensor_noise_shift":
        # Violates invariant sensor noise without changing latent source variances.
        # Can be statistically indistinguishable from latent heteroscedasticity.
        corrupt=np.diag(np.linspace(1.0,2.0,d))
        sensor_noise[env==1]*=2.2
        sensor_noise[env==2]=sensor_noise[env==2]@corrupt
    observed=source[:,:history]@b.T+sensor_noise
    if mode=="mean_only_shift":
        observed[env==1]+=1.4
        observed[env==2]-=1.4
    if missing_fraction:
        mask=rng.random(size=observed.shape)>=missing_fraction
        observed=np.where(mask,observed,np.nan)
    beta=np.linspace(.9,1.6,k)*np.where(np.arange(k)%2==0,1.,-1.)
    # Outcome is generated strictly AFTER the historical measurement window.
    score=-.1+np.sum(source[:,history:].mean(axis=1)*beta[None,:],axis=1)
    prob=1/(1+np.exp(-np.clip(score,-20,20)))
    outcome=(rng.random(n)<prob).astype(np.float32)
    return EnvironmentSimulation(observed,env,outcome,source[:,:history],b,variance,mode)
