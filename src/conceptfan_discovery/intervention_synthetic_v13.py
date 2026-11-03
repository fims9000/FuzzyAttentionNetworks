"""Fully synthetic independent randomized interventions, never real patient data."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class InstrumentedSimulation:
    observations: np.ndarray
    actions: np.ndarray
    labels: np.ndarray
    oracle_factors: np.ndarray
    planted_mixing: np.ndarray
    planted_action_gains: np.ndarray
    action_mode: str
    history_steps: int

def make_intervention_scm(*,n:int,k:int,d:int,seed:int,
        history_steps:int=12,horizon:int=2,action_mode:str="randomized",
        missing_fraction:float=0.0,mixing:np.ndarray|None=None)->InstrumentedSimulation:
    if n<120 or not 2<=k<=d or history_steps<8 or horizon<1:
        raise ValueError("Invalid factor/sample/time count")
    if action_mode not in ("randomized","confounded","collinear","none"):
        raise ValueError("Unregistered intervention design")
    if not 0<=missing_fraction<.8:
        raise ValueError("Invalid missingness regime")
    rng=np.random.default_rng(seed)
    if mixing is None:
        b=rng.normal(scale=.23,size=(d,k))
        b[:k]+=1.2*np.eye(k)
    else:
        b=np.asarray(mixing,dtype=np.float64)
        if b.shape!=(d,k) or np.linalg.matrix_rank(b)<k:
            raise ValueError("Valid full-column-rank fixed sensor mixing required")
    b=b/np.linalg.norm(b,axis=0)
    gain=.43+.08*np.arange(k)/max(k-1,1)
    steps=history_steps+horizon
    latent=np.zeros((n,steps,k),dtype=np.float64)
    actions=np.zeros((n,steps-1,k),dtype=np.float64)
    latent[:,0]=rng.normal(scale=.35,size=(n,k))
    for t in range(steps-1):
        if action_mode=="randomized":
            a=rng.choice([-1.,1.],size=(n,k)).astype(float)
        elif action_mode=="confounded":
            a=1.8*latent[:,t,:]+rng.normal(scale=.1,size=(n,k))
        elif action_mode=="collinear":
            shared=rng.choice([-1.,1.],size=(n,1))
            a=np.repeat(shared,k,axis=1)
        else:
            a=np.zeros((n,k))
        actions[:,t]=a
        latent[:,t+1]=.78*latent[:,t]+gain*a+rng.normal(scale=.08,size=(n,k))
    x=latent@b.T+rng.normal(scale=.06,size=(n,steps,d))
    if missing_fraction:
        mask=rng.random(size=x.shape)>=missing_fraction
        x=np.where(mask,x,np.nan)
    beta=np.linspace(1.0,1.7,k)*np.where(np.arange(k)%2==0,1.,-1.)
    logits=-.25+(latent[:,history_steps:].mean(axis=1)*beta).sum(axis=-1)
    probs=1/(1+np.exp(-np.clip(logits,-20,20)))
    labels=(rng.random(n)<probs).astype(np.float32)
    return InstrumentedSimulation(x,actions,labels,latent,b,gain,action_mode,history_steps)
