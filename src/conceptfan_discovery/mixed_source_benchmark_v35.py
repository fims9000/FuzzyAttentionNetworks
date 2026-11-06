"""V35 mixed-latent source TEST generator and label-blind masks.

The physical source coordinates are tied to independent synthetic DEVICE
excitation reference amplitudes. The estimator receives only calibrated H
from the phantom, not Htrue or oracle patient latent sources. This is an
external calibration assumption and not automatic causal concept discovery.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from conceptfan_discovery.phase_pulse_benchmark_v31 import phase_pulse_world
from conceptfan_discovery.target_observability_v35 import PHI

DATA_SEED=20261130
TRUE_H_SIMULATOR_ONLY=np.array([
    [1.,1.],
    [0.,1.],
    [1.,0.],
],dtype=np.float64)
TRAIN_N,VALID_N,TEST_N=1200,800,600
MASK_STRESS=(
    "MCAR20","MCAR40","only_mixture_4","only_nuisance_5",
    "only_target_6","mixture_plus_nuisance_4_5",
    "independent_block8","synchronized_block8","no_source_all16",
)
FAULT_CASES=(
    ("healthy","physical","none","none"),
    ("unseen_flip_sensor6","physical","none","flip6"),
    ("unseen_common_flip_456","physical","none","flipall"),
    ("known_common_flip_VALID2","physical","flipall","flipall"),
    ("complete_dictionary","complete","none","none"),
)


def mixed_reference_world(*,seed=DATA_SEED,
                          n_train=TRAIN_N,n_valid=VALID_N,
                          n_test=TEST_N):
    world=phase_pulse_world(
        seed=seed,n_train=n_train,n_valid=n_valid,n_test=n_test)
    x=world["x"].copy()
    rng=np.random.default_rng(seed+90019)
    a=np.column_stack([
        world["amplitude_CONCEPTS_AUDIT_ONLY"]["physical"],
        world["amplitude_CONCEPTS_AUDIT_ONLY"]["nuisance"],
    ])
    observed_signal=np.einsum(
        "nk,jk,t->ntj",a,TRUE_H_SIMULATOR_ONLY,PHI)
    x[:,:,4:7]=(
        observed_signal+rng.normal(0,.32,size=observed_signal.shape)
    ).astype(np.float32)
    return {
        "x":x,
        "outcomes":world["outcomes"],
        "train":world["train"],
        "valid1":world["valid1"],
        "valid2":world["valid2"],
        "test":world["test"],
        "true_a1_EVALUATION_ONLY":a[:,0],
        "true_a2_EVALUATION_ONLY":a[:,1],
        "generator_family":"V35_externally_calibrated_two_mixed_pulse_sources",
        "simulator_H_used_only_to_generate_measurements":True,
        "no_medical_semantics_certified":True,
    }


def faulted_source_readouts(x:np.ndarray,mode:str):
    out=np.asarray(x,dtype=np.float32).copy()
    if mode=="none":
        pass
    elif mode=="flip6":
        out[:,:,6]*=-1
    elif mode=="flipall":
        out[:,:,4:7]*=-1
    else:
        raise ValueError("Unregistered TEST device readout fault")
    np.testing.assert_array_equal(out[:,:,:4],np.asarray(x)[:,:,:4])
    return out


def observation_mask(x:np.ndarray,mode:str,*,seed:int,
                     common_known_seed:int=20263550)->np.ndarray:
    xx=np.asarray(x)
    if xx.ndim!=3 or xx.shape[1:]!=(16,12):
        raise ValueError("Expected 16 timepoints and 12 channels")
    if mode not in MASK_STRESS:
        raise ValueError("Unregistered TEST missingness")
    known_rng=np.random.default_rng(common_known_seed)
    rng=np.random.default_rng(seed)
    obs=known_rng.random(xx.shape)>=.20
    source=rng.random((len(xx),16,3))>=(
        .4 if mode=="MCAR40" else .2)
    if mode in (
        "only_mixture_4","only_nuisance_5","only_target_6",
        "mixture_plus_nuisance_4_5","no_source_all16",
    ):
        source[:]=False
        active={
            "only_mixture_4":(0,),
            "only_nuisance_5":(1,),
            "only_target_6":(2,),
            "mixture_plus_nuisance_4_5":(0,1),
            "no_source_all16":(),
        }[mode]
        for j in active:
            source[:,:,j]=True
    elif mode=="independent_block8":
        source[:]=True
        starts=rng.integers(0,9,size=(len(xx),3))
        for i in range(len(xx)):
            for j in range(3):
                source[i,starts[i,j]:starts[i,j]+8,j]=False
    elif mode=="synchronized_block8":
        source[:]=True
        starts=rng.integers(0,9,size=len(xx))
        for i,first in enumerate(starts):
            source[i,first:first+8,:]=False
    obs[:,:,4:7]=source
    return obs
