"""V20b: new nonlinear / regime / irregular-sampling synthetic stress families.

Train/validation/test originate from independent generator seeds and no
oracle states enter DiscoverySplit. This isolates native risk-head refit,
not source discovery or physiological identification.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import DiscoveryData, DiscoverySplit


FAMILIES=("nonlinear_delayed","regime_switch","informative_sampling","outcome_null")
SEEDS={"train":305011,"validation":305012,"test":305013}
SIZES={"train":1000,"validation":300,"test":300}
OBSERVATION_WINDOW=24
HORIZON=6
CHANNELS=tuple(f"sensor_{i}" for i in range(5))


@dataclass(frozen=True)
class IndependentFamilyData:
    discovery: DiscoveryData
    test: DiscoverySplit
    family: str


def _raw_episodes(family:str,seed:int,n:int):
    if family not in FAMILIES:
        raise ValueError("Unknown predefined V20b generator family")
    rng=np.random.default_rng(seed)
    t=OBSERVATION_WINDOW+HORIZON
    latent=np.zeros((n,t,3),dtype=np.float64)
    shocks=rng.normal(size=(n,t,3))
    latent[:,0,:]=.7*shocks[:,0,:]
    regimes=rng.integers(0,2,size=n)
    for i in range(1,t):
        a,b,c=latent[:,i-1,0],latent[:,i-1,1],latent[:,i-1,2]
        if family=="regime_switch":
            phi=np.where(regimes==0,.4,.7) if i<12 else np.where(regimes==0,.88,.25)
            latent[:,i,0]=.5*a+.2*np.tanh(b)+.55*shocks[:,i,0]
            latent[:,i,1]=.58*b+.16*np.sin(a)+.5*shocks[:,i,1]
            latent[:,i,2]=phi*c+.22*np.tanh(a*b)+.52*shocks[:,i,2]
        else:
            latent[:,i,0]=.57*a+.18*np.tanh(b)+.58*shocks[:,i,0]
            latent[:,i,1]=.66*b+.14*np.sin(a)+.54*shocks[:,i,1]
            # Time-delayed nonlinear feedback.
            delayed=latent[:,max(0,i-3),2]
            latent[:,i,2]=.56*c+.22*np.tanh(b*delayed)+.53*shocks[:,i,2]

    x=rng.normal(scale=.18,size=(n,t,5))
    x[:,:,0]+=.85*latent[:,:,0]
    x[:,:,1]+=.75*latent[:,:,1]+.12*latent[:,:,0]
    x[:,:,2]+=.88*latent[:,:,2]+.12*latent[:,:,1]
    x[:,:,3]+=.34*latent[:,:,0]+.25*latent[:,:,1]
    x[:,:,4]+=.29*latent[:,:,2]
    if family=="regime_switch":
        x[:,12:,2]+=.26*latent[:,12:,0]

    future=latent[:,OBSERVATION_WINDOW:,2].mean(1)
    last=latent[:,OBSERVATION_WINDOW-5:OBSERVATION_WINDOW,2].mean(1)
    if family=="regime_switch":
        signal=2.7*future+.3*(future-last)
    else:
        signal=2.8*future+.35*last
    probability=1/(1+np.exp(-np.clip(signal,-15,15)))
    if family=="outcome_null":
        probability=np.full(n,.5,dtype=np.float64)
    y=(rng.random(n)<probability).astype(np.float32)

    mask=np.ones_like(x,dtype=np.float32)
    if family=="informative_sampling":
        # Sensor sampling is informative but is not a physiological feature.
        z=latent
        chance=np.empty_like(x)
        chance[:,:,0]=.60+.28/(1+np.exp(-z[:,:,0]))
        chance[:,:,1]=.63+.20/(1+np.exp(-z[:,:,1]))
        chance[:,:,2]=.24+.48/(1+np.exp(-z[:,:,2]))
        chance[:,:,3]=.73
        chance[:,:,4]=.50+.22/(1+np.exp(-z[:,:,2]))
        mask=(rng.random(x.shape)<chance).astype(np.float32)
        # Guarantee at least one real measurement for stable normalization.
        mask[:,0,0]=1.
    return x,mask,y


def make_distinct_generator_data(family:str)->IndependentFamilyData:
    all_parts={}
    for part in ("train","validation","test"):
        all_parts[part]=_raw_episodes(family,SEEDS[part],SIZES[part])
    x_train,m_train,_=all_parts["train"]
    medians=[]
    scales=[]
    for i in range(x_train.shape[-1]):
        observed=x_train[:,:,i][m_train[:,:,i]>0]
        if len(observed)<100:
            raise ValueError("Generator does not have enough TRAIN-only observed values")
        medians.append(float(np.median(observed)))
        scales.append(float(max(np.quantile(observed,.75)-np.quantile(observed,.25),.05)))
    medians=np.asarray(medians,dtype=np.float64)
    scales=np.asarray(scales,dtype=np.float64)
    def make(part,offset):
        raw,mask,labels=all_parts[part]
        normalized=np.clip((raw-medians)/scales,-6,6).astype(np.float32)
        observed=mask.astype(np.float32)
        normalized*=observed
        past=normalized[:,:OBSERVATION_WINDOW,:]
        future=normalized[:,OBSERVATION_WINDOW:,:]
        past_mask=observed[:,:OBSERVATION_WINDOW,:]
        future_mask=observed[:,OBSERVATION_WINDOW:,:]
        return DiscoverySplit(
            x=past,mask=past_mask,
            future=future,future_mask=future_mask,
            y=labels,oracle_states=None,
            episode_ids=np.arange(offset,offset+len(labels),dtype=np.int64),
        )
    train=make("train",0)
    val=make("validation",10000)
    test=make("test",20000)
    if set(train.episode_ids)&set(val.episode_ids) or set(train.episode_ids)&set(test.episode_ids):
        raise AssertionError("TRAIN/VAL/TEST IDs must be disjoint")
    data=DiscoveryData(
        train=train,validation=val,
        receipt={
            "channel_names":list(CHANNELS),
            "generator_family":family,
            "test_outcomes_never_used_for_model_selection":True,
            "no_latent_oracle_exported":True,
            "different_data_generator_from_V18":True,
        },
        medians=medians,scales=scales,
    )
    return IndependentFamilyData(discovery=data,test=test,family=family)
