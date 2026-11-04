"""Synthetic synchronized independent and correlated sensor-view noise tests."""
from dataclasses import dataclass
import numpy as np
@dataclass
class TwoViews:
    a:np.ndarray
    b:np.ndarray
    env:np.ndarray
    latent:np.ndarray
    mixing_a:np.ndarray
    mixing_b:np.ndarray
    mode:str

def make_crossview_scm(*,k,seed,per_env=1000,t=12,mode="latent_shift"):
    if k not in (3,5,7) or mode not in (
        "latent_shift","independent_colored_noise",
        "shared_colored_noise","white_noise_shift","no_shift"):
        raise ValueError("Unknown two-view generator")
    rng=np.random.default_rng(seed);d=k+3
    a=rng.normal(scale=.23,size=(d,k));a[:k]+=1.3*np.eye(k)
    b=rng.normal(scale=.23,size=(d,k));b[:k]+=1.3*np.eye(k)
    a/=np.linalg.norm(a,axis=0)
    b/=np.linalg.norm(b,axis=0)
    env=np.repeat(np.arange(3),per_env)
    n=len(env);rho=.74;base=np.full(k,.72)
    delta=.85+.15*np.arange(k);ratio=np.linspace(.3,3.2,k)
    dvar=np.stack((np.zeros(k),delta,delta*ratio))
    variances=np.broadcast_to(base[None,:]+(dvar if mode=="latent_shift" else 0),(3,k))
    latent=np.zeros((n,t,k));na=np.zeros_like(latent);nb=np.zeros_like(latent)
    for e in range(3):
        idx=np.flatnonzero(env==e)
        for step in range(t):
            def ar(scale,prev=None):
                z=rng.normal(size=(len(idx),k))*np.sqrt(scale)
                return z if prev is None else rho*prev+np.sqrt(1-rho*rho)*z
            latent[idx,step]=ar(variances[e],None if step==0 else latent[idx,step-1])
            if mode in ("independent_colored_noise","shared_colored_noise","white_noise_shift"):
                na[idx,step]=ar(dvar[e],None if step==0 or mode=="white_noise_shift" else na[idx,step-1])
                nb[idx,step]=(na[idx,step] if mode=="shared_colored_noise" else
                    ar(dvar[e],None if step==0 or mode=="white_noise_shift" else nb[idx,step-1]))
    xa=latent@a.T+na@a.T+rng.normal(scale=.12,size=(n,t,d))
    xb=latent@b.T+nb@b.T+rng.normal(scale=.12,size=(n,t,d))
    return TwoViews(xa,xb,env,latent,a,b,mode)
