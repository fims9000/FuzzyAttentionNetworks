"""New V36 generator: two unknown instrument views, hidden AR factors (AUDIT only)."""
from __future__ import annotations
import numpy as np
from scipy.special import expit
from conceptfan_discovery.phase_pulse_benchmark_v31 import phase_pulse_world

SEED=20261206
STRESS=("complete","MCAR20","MCAR40","independent_block8",
        "synced_block8","missing_view_A","missing_view_B","no_views")
CASES=(
 ("healthy_energy","energy","none","none"),
 ("unseen_both_views_flip","energy","none","both_flip"),
 ("unseen_one_view_flip","energy","none","A_flip"),
 ("known_one_view_flip_VALID2","energy","A_flip","A_flip"),
 ("unseen_shared_colored_noise","energy","none","shared_artifact"),
 ("complete_dictionary","complete","none","none"),
 ("signed_source_target","signed","none","none"),
)


def generate_world_v36(*,seed:int=SEED,n_train:int=1200,n_valid:int=800,
                       n_test:int=600,matched_rho:bool=False):
    world=phase_pulse_world(seed=seed,n_train=n_train,
                             n_valid=n_valid,n_test=n_test)
    x=world["x"].copy()
    rng=np.random.default_rng(seed+2101)
    n,t=x.shape[:2]
    rho=np.array([.77,.74] if matched_rho else [.77,-.27])
    gain=rng.uniform(.60,1.75,size=(n,2))
    s=np.empty((n,t,2))
    s[:,0,:]=rng.normal(size=(n,2))
    for j in range(1,t):
        s[:,j,:]=rho*s[:,j-1,:]+np.sqrt(1-rho**2)*rng.normal(size=(n,2))
    s*=gain[:,None,:]
    # Generator-only unknown mixing; NEVER passed to estimation.
    true_A=np.array([[1.1,.6],[.45,1.2]])
    true_B=np.array([[.8,-.7],[1.15,.4]])
    x[:,:,4:6]=(s@true_A.T+rng.normal(0,.21,size=(n,t,2))).astype("float32")
    x[:,:,6:8]=(s@true_B.T+rng.normal(0,.21,size=(n,t,2))).astype("float32")
    known=world["true_known_CONCEPTS_AUDIT_ONLY"].mean(axis=1)
    base=3.7*(known[:,0]-.5)-3.6*(known[:,1]-.5)-.10
    energy=(s[:,:,0]**2).mean(axis=1)
    signed=s[:,:,0].mean(axis=1)
    coin=rng.random(n)
    labels={
        "complete":(coin<expit(base)).astype(np.int64),
        "energy":(coin<expit(base+1.7*(energy-1.4))).astype(np.int64),
        "signed":(coin<expit(base+3.*signed)).astype(np.int64),
    }
    return {
        "x":x,"outcomes":labels,"train":world["train"],
        "valid1":world["valid1"],"valid2":world["valid2"],
        "test":world["test"],"true_slow_energy_AUDIT_ONLY":energy,
        "true_slow_signed_AUDIT_ONLY":signed,
        "true_A_B_SIMULATOR_ONLY":(true_A,true_B),
        "true_rho_SIMULATOR_ONLY":rho,
        "generator_family":"two-view stationary AR with uncalibrated mixing",
        "no_signed_or_physical_identity_from_observations":True,
    }


def corrupt_views_v36(x,mode,*,seed=20263666):
    y=np.asarray(x,dtype=np.float32).copy()
    if mode=="none":
        pass
    elif mode=="A_flip":
        y[:,:,4:6]*=-1
    elif mode=="both_flip":
        y[:,:,4:8]*=-1
    elif mode=="shared_artifact":
        rng=np.random.default_rng(seed)
        shared=rng.normal(size=(len(y),16))
        for k in range(1,16):
            shared[:,k]+=.8*shared[:,k-1]
        y[:,:,4:6]+=1.5*shared[:,:,None]
        y[:,:,6:8]+=1.5*shared[:,:,None]
    else:
        raise ValueError("Unregistered source fault")
    np.testing.assert_array_equal(y[:,:,:4],np.asarray(x)[:,:,:4])
    return y


def mask_views_v36(x,mode,*,seed:int):
    x=np.asarray(x)
    if x.ndim!=3 or x.shape[1:]!=(16,12) or mode not in STRESS:
        raise ValueError("Unregistered V36 mask")
    rng=np.random.default_rng(seed)
    m=np.ones(x.shape,dtype=bool)
    if mode=="complete":
        return m
    rate=.4 if mode=="MCAR40" else .2
    m[:,:,4:8]=rng.random((len(x),16,4))>=rate
    if mode=="independent_block8":
        m[:,:,4:8]=True
        starts=rng.integers(0,9,size=(len(x),4))
        for i in range(len(x)):
            for j in range(4):
                m[i,starts[i,j]:starts[i,j]+8,4+j]=False
    if mode=="synced_block8":
        m[:,:,4:8]=True
        starts=rng.integers(0,9,size=len(x))
        for i,k in enumerate(starts):
            m[i,k:k+8,4:8]=False
    if mode=="missing_view_A":
        m[:,:,4:6]=False
    if mode=="missing_view_B":
        m[:,:,6:8]=False
    if mode=="no_views":
        m[:,:,4:8]=False
    return m
