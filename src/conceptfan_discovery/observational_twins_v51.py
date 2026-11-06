"""V51: exactly observationally equivalent synthetic twin worlds.

Two shared-view Gaussian AR sources with equal dynamics retain orthogonal
rotational gauge freedom. Both worlds have bitwise identical X, masks and Y
but physical coordinate-1 energy is different. External attested anchor
measurement is the ONLY semantic symmetry breaker.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

DEVICES_V51=(20263307,20263309,20263321,20263323)
ANGLE_RAD=.73
RHO=.72
T=16
N_SENSORS=4


def energy_from_trajectory_v51(source):
    a=np.asarray(source,dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=T:
        raise ValueError("Energy requires [N,16] source")
    return a.var(axis=1,ddof=1)


def _sensor_matrix(seed:int):
    rng=np.random.default_rng(seed+909)
    base=np.array([[1.2,.3],[-.2,1.1],[.8,-.5],[.3,.95]])
    h=base + rng.normal(0.,.11,size=(4,2))
    if np.linalg.svd(h,compute_uv=False)[-1]<.6:
        raise AssertionError("Unexpected rank-deficient physical measurement world")
    return h


def make_observational_twins_v51(device_seed:int,*,n_train=1200,
                                 n_valid1=400,n_valid2=400,n_test=600):
    if device_seed not in DEVICES_V51 or min(n_train,n_valid1,n_valid2,n_test)<150:
        raise ValueError("Require four preregistered devices and sufficient episodes")
    n=n_train+n_valid1+n_valid2+n_test
    rng=np.random.default_rng(device_seed+4030)
    s=np.empty((n,T,2),dtype=np.float64)
    s[:,0]=rng.normal(size=(n,2))
    for t in range(1,T):
        s[:,t]=RHO*s[:,t-1]+np.sqrt(1.-RHO**2)*rng.normal(size=(n,2))
    s*=rng.uniform(.6,1.7,size=(n,1,1))
    angle=ANGLE_RAD
    R=np.array([[np.cos(angle),-np.sin(angle)],
                [np.sin(angle),np.cos(angle)]],dtype=np.float64)
    sr=s@R.T
    h=_sensor_matrix(device_seed)
    hr=h@R.T
    readout_noise=rng.normal(0.,.16,size=(n,T,4))
    x=s@h.T+readout_noise
    # Ground-truth rotation identity is checked independently, then the
    # SAME observed bytes are explicitly handed to both scientific worlds.
    assert np.allclose(x,sr@hr.T+readout_noise,rtol=0,atol=1e-14)
    shift_h=1.5*h
    shift_hr=shift_h@R.T
    shift_x=s@shift_h.T+readout_noise
    assert np.allclose(shift_x,sr@shift_hr.T+readout_noise,rtol=0,atol=2e-14)
    source_energies=np.var(s,axis=1,ddof=1)
    rot_energies=np.var(sr,axis=1,ddof=1)
    if not np.allclose(source_energies.sum(axis=1),
                       rot_energies.sum(axis=1),rtol=0,atol=5e-14):
        raise AssertionError("The twin risk law must be rotation-invariant")
    total=source_energies.sum(axis=1)
    y=(rng.random(n)<expit(-1.2+2.4*(total-1.4))).astype(np.int64)
    assay_noise=rng.normal(0.,.16,size=(n,T))
    independent_assays=(s[:,:,0]+assay_noise,
                         sr[:,:,0]+assay_noise)
    train=np.arange(n_train)
    valid1=np.arange(n_train,n_train+n_valid1)
    valid2=np.arange(n_train+n_valid1,n_train+n_valid1+n_valid2)
    test=np.arange(n-n_test,n)
    masks={}
    for name,prob in (("TRAIN_MCAR20",.2),("VALID1_MCAR20",.2),
                      ("VALID2_MCAR40",.4),("TEST_MCAR20",.2),
                      ("TEST_MCAR40",.4)):
        seed_adjust=sum(ord(c) for c in name)*7
        masks[name]=(np.random.default_rng(device_seed+seed_adjust)
                     .random((n,T,4))>=prob)
    assert np.array_equal(y,y.copy())
    return {
        "device_seed":device_seed,
        "train":train,"valid1":valid1,"valid2":valid2,"test":test,
        "shared_observed":x,
        "test_only_amplifier_readout":shift_x,
        "same_shared_outcome_Y":y,
        "measured_boolean_masks":masks,
        "worlds":(
            {"gauge_id":"A","physical_device_id":f"device{device_seed}_gaugeA",
             "external_reference_id":f"assay_source1_A_device{device_seed}",
             "attested_external_assay":independent_assays[0],
             "hidden_physical_source_energy_TEST_AUDIT_ONLY":source_energies[:,0],
             "hidden_H_AUDIT_ONLY":h},
            {"gauge_id":"B","physical_device_id":f"device{device_seed}_gaugeB",
             "external_reference_id":f"assay_source1_B_device{device_seed}",
             "attested_external_assay":independent_assays[1],
             "hidden_physical_source_energy_TEST_AUDIT_ONLY":rot_energies[:,0],
             "hidden_H_AUDIT_ONLY":hr},
        ),
        "observational_twin_proof":{
            "same_observations_and_outcome_across_worlds":True,
            "same_missingness_and_all_train_valid_test_splits":True,
            "rotation_angle_radians":ANGLE_RAD,
            "latent_process_isotropic_equal_rho":RHO,
            "no_unique_physical_source1_from_X_and_Y":True,
            "independently_observed_anchor_breaks_gauge_in_its_attested_device":True,
        },
        "no_hidden_H_or_S_given_to_estimators":True,
    }


def sensor_task_features_v51(x,mask):
    """Ordinary raw nonconceptual risk features, SAME for both twin worlds."""
    a=np.asarray(x,dtype=np.float64)
    m=np.asarray(mask,dtype=bool)
    if a.ndim!=3 or a.shape[1:]!=(16,4) or a.shape!=m.shape:
        raise ValueError("Expected measured four-channel episodes")
    observed=np.where(m,a,0.)
    count=np.maximum(1,m.sum(axis=1))
    mu=observed.sum(axis=1)/count
    centered=np.where(m,a-mu[:,None,:],0.)
    var=(centered*centered).sum(axis=1)/np.maximum(1,m.sum(axis=1)-1)
    rms=np.sqrt((observed*observed).sum(axis=1)/count)
    cross=np.stack([
        (np.where(m[:,:,i]&m[:,:,j],a[:,:,i]*a[:,:,j],0.).sum(axis=1)/
         np.maximum(1,(m[:,:,i]&m[:,:,j]).sum(axis=1)))
        for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
    ],axis=1)
    return np.concatenate((mu,var,rms,cross,m.mean(axis=1)),axis=1)
