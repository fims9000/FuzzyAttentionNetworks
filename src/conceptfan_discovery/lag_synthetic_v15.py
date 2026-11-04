"""V15 independent synthetic dynamic-source and environment-dependent sensor-noise states."""
from dataclasses import dataclass
import numpy as np

@dataclass
class Simulated:
    x:np.ndarray
    env:np.ndarray
    true_sources:np.ndarray
    mixing:np.ndarray
    sensor_noises:np.ndarray
    source_variances:np.ndarray
    mode:str

def make_lag_scm(*,k,seed,per_env=1000,steps=12,mode="latent_shift"):
    if (k not in (3,5,7) or per_env<180 or steps<8
        or mode not in ("latent_shift","white_sensor_shift","colored_sensor_shift",
                        "no_shift","tied_profiles")):
        raise ValueError("Unregistered generator mode")
    rng=np.random.default_rng(seed)
    d=k+3
    b=rng.normal(scale=.24,size=(d,k))
    b[:k]+=1.3*np.eye(k)
    b=b/np.linalg.norm(b,axis=0)
    env=np.repeat(np.arange(3),per_env)
    n=len(env)
    base=np.full(k,.72)
    delta=.9+.17*np.arange(k)
    ratio=np.linspace(.27,3.2,k)
    if mode=="tied_profiles":
        ratio[:2]=1.0
    dvar=np.stack([np.zeros(k),delta,delta*ratio])
    variances=np.tile(base,(3,1))+(dvar if mode in ("latent_shift","tied_profiles") else 0)
    rho=.73
    source=np.zeros((n,steps,k))
    colored=np.zeros((n,steps,k))
    for e in range(3):
        ids=np.flatnonzero(env==e)
        for t in range(steps):
            innovation=rng.normal(size=(len(ids),k))
            source[ids,t]=(np.sqrt(variances[e])*innovation if t==0 else
                rho*source[ids,t-1]+np.sqrt(1-rho*rho)*np.sqrt(variances[e])*innovation)
            if mode=="colored_sensor_shift":
                z=rng.normal(size=(len(ids),k))
                colored[ids,t]=(np.sqrt(dvar[e])*z if t==0 else
                    rho*colored[ids,t-1]+np.sqrt(1-rho*rho)*np.sqrt(dvar[e])*z)
            elif mode=="white_sensor_shift":
                colored[ids,t]=np.sqrt(dvar[e])*rng.normal(size=(len(ids),k))
    sensor_noise=rng.normal(scale=.13,size=(n,steps,d))+(colored@b.T)
    observed=source@b.T+sensor_noise
    return Simulated(observed,env,source,b,sensor_noise,variances,mode)
