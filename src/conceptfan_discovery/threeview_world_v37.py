"""V37: new independently sampled three-view measurement environments.

These matrices and latent source traces exist ONLY inside the simulator; the
estimator accepts 12-channel observations and masks, never H, S or diagnoses.
Known K2 first four sensors and outcomes are inherited only as independent
TRAIN-proxy tasks; new unknown groups: (4,5), (6,7), (8,9).
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit
from conceptfan_discovery.phase_pulse_benchmark_v31 import phase_pulse_world

MIX_SEEDS=(20261217,20261219)
UNKNOWN_VIEWS=((4,5),(6,7),(8,9))
STRESSES=("complete","MCAR20","MCAR40","independent_block8",
          "synchronized_block8","missing_A","missing_C","no_views")
CASES=(
 ("healthy_energy","energy","none","none"),
 ("TEST_offset_A_plus2","energy","none","offset_A"),
 ("TEST_offset_all_plus2","energy","none","offset_all"),
 ("TEST_common_AB_colored","energy","none","shared_AB"),
 ("TEST_common_ABC_colored","energy","none","shared_ABC"),
 ("TEST_one_view_A_flip","energy","none","flip_A"),
 ("VALID2_known_AB_colored","energy","shared_AB","shared_AB"),
 ("complete_dictionary","complete","none","none"),
 ("TEST_one_view_A_noise","energy","none","noise_A"),
)


def _sample_mix(rng):
    """Fixed per device, never supplied to the observed-data algorithm."""
    u,_=np.linalg.qr(rng.normal(size=(2,2)))
    v,_=np.linalg.qr(rng.normal(size=(2,2)))
    scales=rng.uniform(.75,1.30,size=2)
    A=u@np.diag(scales)@v.T
    if np.linalg.cond(A)>2.5:
        raise AssertionError("Generator produced non-registered ill-conditioned mix")
    return A


def generate_threeview_world_v37(
    *,mix_seed:int,n_train:int=1000,n_valid:int=800,n_test:int=600,
    tied_spectral_lags:bool=False,
):
    if mix_seed not in MIX_SEEDS or n_valid%2 or min(n_train,n_valid,n_test)<100:
        raise ValueError("Only frozen independent V37 mixing device seeds and split sizes")
    base=phase_pulse_world(seed=mix_seed+331,n_train=n_train,
                           n_valid=n_valid,n_test=n_test)
    x=base["x"].copy()
    rng=np.random.default_rng(mix_seed+77231)
    n,t=x.shape[:2]
    rho=np.array([.77,.74] if tied_spectral_lags else [.77,-.27])
    latent=rng.normal(size=(n,t,2))
    latent[:,0,:]=rng.normal(size=(n,2))
    for step in range(1,t):
        latent[:,step,:]=(
            rho*latent[:,step-1,:]+
            np.sqrt(1-rho**2)*rng.standard_normal((n,2))
        )
    latent*=rng.uniform(.65,1.65,size=(n,1,2))
    mixes=tuple(_sample_mix(rng) for _ in UNKNOWN_VIEWS)
    for mix,view in zip(mixes,UNKNOWN_VIEWS):
        x[:,:,list(view)]=(latent@mix.T+
                rng.normal(0,.21,size=(n,t,2))).astype(np.float32)
    x[:,:,10:]=rng.normal(size=(n,t,2)).astype(np.float32)
    known=base["true_known_CONCEPTS_AUDIT_ONLY"].mean(axis=1)
    zknown=3.7*(known[:,0]-.5)-3.6*(known[:,1]-.5)-.10
    centered=latent[:,:,0]-latent[:,:,0].mean(axis=1,keepdims=True)
    fluctuation_power=np.mean(centered**2,axis=1)
    common_u=rng.random(n)
    outcomes={
        "complete":(common_u<expit(zknown)).astype(np.int64),
        "energy":(common_u<expit(zknown+2.2*(fluctuation_power-.65)))
                 .astype(np.int64),
    }
    return {
        "x":x,
        "outcomes":outcomes,
        "train":base["train"],"valid1":base["valid1"],
        "valid2":base["valid2"],"test":base["test"],
        "synthetic_mixes_AUDIT_ONLY":[m.tolist() for m in mixes],
        "synthetic_latent_fluctuation_power_AUDIT_ONLY":fluctuation_power,
        "synthetic_lag_coefficients_AUDIT_ONLY":rho.tolist(),
        "independent_instrument_noise_is_simulator_assumption":True,
        "views_given_but_mixing_unknown_to_learner":True,
    }


def apply_device_world_v37(x,mode,*,seed:int):
    """One fixed fault per episode/test scenario; NEVER alter known K2."""
    x=np.asarray(x,dtype=np.float32)
    if x.ndim!=3 or x.shape[1:]!=(16,12):
        raise ValueError("Expected 16x12 input")
    out=x.copy()
    rng=np.random.default_rng(seed)
    if mode=="none":
        pass
    elif mode=="offset_A":
        out[:,:,4:6]+=2.
    elif mode=="offset_all":
        out[:,:,4:10]+=2.
    elif mode=="noise_A":
        out[:,:,4:6]+=rng.normal(0,.90,size=out[:,:,4:6].shape)
    elif mode=="flip_A":
        out[:,:,4:6]*=-1.
    elif mode in ("shared_AB","shared_ABC"):
        shared=rng.normal(size=(len(out),16))
        for t in range(1,16):
            shared[:,t]+=.8*shared[:,t-1]
        affected=range(4,10 if mode=="shared_ABC" else 8)
        for channel in affected:
            out[:,:,channel]+=(1.35*shared).astype(np.float32)
    else:
        raise ValueError("Unregistered V37 fault")
    np.testing.assert_array_equal(out[:,:,:4],x[:,:,:4])
    return out


def view_masks_v37(x,stress,*,seed:int,known_seed:int):
    x=np.asarray(x)
    if x.ndim!=3 or x.shape[1:]!=(16,12) or stress not in STRESSES:
        raise ValueError("Expected registered V37 missingness stress")
    rng=np.random.default_rng(seed)
    mask=np.ones(x.shape,dtype=bool)
    mask[:,:,:4]=np.random.default_rng(known_seed).random((len(x),16,4))>=.2
    if stress in ("MCAR20","MCAR40"):
        missing=.20 if stress=="MCAR20" else .40
        mask[:,:,4:10]=rng.random((len(x),16,6))>=missing
    elif stress=="independent_block8":
        starts=rng.integers(0,9,size=(len(x),6))
        for i in range(len(x)):
            for j in range(6):
                mask[i,starts[i,j]:starts[i,j]+8,4+j]=False
    elif stress=="synchronized_block8":
        starts=rng.integers(0,9,size=len(x))
        for i,j in enumerate(starts):
            mask[i,j:j+8,4:10]=False
    elif stress=="missing_A":
        mask[:,:,4:6]=False
    elif stress=="missing_C":
        mask[:,:,8:10]=False
    elif stress=="no_views":
        mask[:,:,4:10]=False
    return mask
