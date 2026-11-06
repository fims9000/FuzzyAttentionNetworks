"""V38: independently randomized synthetic devices, external source probe and sentinel.

No health claim: the known-actuator target is defined by a SIMULATOR.
Source estimator receives only routine sensor values, paired on/off readouts,
actuator type and independently wired negative control sensor 10. The
generator-only H and hidden S are audit-only and never enter estimation.
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit

from conceptfan_discovery.phase_pulse_benchmark_v31 import phase_pulse_world

DEVICE_SEEDS=(20261231,20261301)
VIEW_PAIRS=((4,5),(6,7),(8,9))
STRESSES=("complete","MCAR20","MCAR40","synchronized_block8",
          "missing_C","missing_sentinel","missing_all_views")
CASES=(
    ("healthy","energy","none","none"),
    ("monitored_AB_common_noise","energy","none","shared_AB_monitored"),
    ("monitored_ABC_common_noise","energy","none","shared_ABC_monitored"),
    ("UNMONITORED_ABC_noise","energy","none","shared_ABC_unmonitored"),
    ("sentinel_false_alarm","energy","none","sentinel_only"),
    ("unseen_A_flip","energy","none","flip_A"),
    ("offset_all_plus2","energy","none","offset_all"),
    ("complete_dictionary","complete","none","none"),
    ("one_view_noise_shift","energy","none","noise_A"),
    ("VALID2_known_AB_noise","energy","shared_AB_monitored","shared_AB_monitored"),
)


def _mix(rng):
    a,_=np.linalg.qr(rng.normal(size=(2,2)))
    b,_=np.linalg.qr(rng.normal(size=(2,2)))
    scales=rng.uniform(.75,1.3,size=2)
    return a@np.diag(scales)@b.T


def _latent(rng,n):
    rho=np.array([.77,-.27])
    latent=rng.normal(size=(n,16,2))
    for t in range(1,16):
        latent[:,t,:]=(
            rho*latent[:,t-1,:]+np.sqrt(1-rho**2)*rng.standard_normal((n,2))
        )
    latent*=rng.uniform(.65,1.65,size=(n,1,2))
    return latent


def _measure(rng,latent,mixes):
    n=len(latent)
    observed=np.zeros((n,16,12),dtype=np.float32)
    for idx,mix in zip(VIEW_PAIRS,mixes):
        observed[:,:,list(idx)]=(latent@mix.T+
            rng.normal(0,.21,size=(n,16,2))).astype(np.float32)
    observed[:,:,10]=rng.normal(0,.08,size=(n,16))
    observed[:,:,11]=rng.normal(size=(n,16))
    return observed


def _paired_probes(rng,mixes,*,per_actuator=240,leak_into_sentinel=False):
    if per_actuator not in (180,240):
        raise ValueError("Only preregistered independent calibration cohort sizes")
    out={}
    deterministic_wave=np.r_[np.ones(8),-np.ones(8)]
    for name,axis in (("target",0),("nuisance",1)):
        s=_latent(rng,per_actuator)
        source_on=s.copy()
        # Known external ACTUATOR input, NOT supplied latent S.
        # Random sign in each case prevents incidental fixed-phase correlation.
        signs=rng.choice([-1.,1.],size=(per_actuator,1))
        command=signs*deterministic_wave[None,:]
        source_on[:,:,axis]+=1.8*command
        off=_measure(rng,s,mixes)
        on=_measure(rng,source_on,mixes)
        if leak_into_sentinel and name=="target":
            on[:,:,10]+=(1.8*command).astype(np.float32)
        mask=np.ones(off.shape,dtype=bool)
        mask[:,:,4:10]=rng.random((per_actuator,16,6))>=.20
        mask[:,:,10]=rng.random((per_actuator,16))>=.10
        # Deliberately omit latent and mixing. Calibration sees only
        # the paired measured readouts and externally recorded command.
        out[name]={
            "off":off,"on":on,"mask":mask,
            "known_actuator_command":command.astype(np.float32),
            "physical_source_identity_external_assumption":True,
        }
    return out


def generate_exogenous_world_v38(
    *,device_seed:int,n_train=1000,n_valid=800,n_test=600,
    per_actuator=240,
):
    if (device_seed not in DEVICE_SEEDS or min(n_train,n_valid,n_test)<100
        or n_valid%2 or per_actuator not in (180,240)):
        raise ValueError("Unregistered V38 device/split")
    base=phase_pulse_world(
        seed=device_seed+331,n_train=n_train,n_valid=n_valid,n_test=n_test)
    x=base["x"].copy()
    rng=np.random.default_rng(device_seed+99831)
    s=_latent(rng,len(x))
    mixes=tuple(_mix(rng) for _ in VIEW_PAIRS)
    measured=_measure(rng,s,mixes)
    x[:,:,4:]=measured[:,:,4:]
    reference=_paired_probes(rng,mixes,per_actuator=per_actuator)
    known=base["true_known_CONCEPTS_AUDIT_ONLY"].mean(axis=1)
    zknown=3.7*(known[:,0]-.5)-3.6*(known[:,1]-.5)-.10
    s_centered=s[:,:,0]-s[:,:,0].mean(axis=1,keepdims=True)
    energy=np.mean(s_centered*s_centered,axis=1)
    uniform=rng.random(len(x))
    outcomes={
        "energy":(uniform<expit(zknown+2.2*(energy-.65))).astype(np.int64),
        "complete":(uniform<expit(zknown)).astype(np.int64),
    }
    return {
        "x":x,"outcomes":outcomes,
        "train":base["train"],"valid1":base["valid1"],
        "valid2":base["valid2"],"test":base["test"],
        "independent_calibration_onoff":reference,
        "hidden_H_AUDIT_ONLY":[m.tolist() for m in mixes],
        "hidden_energy_AUDIT_ONLY":energy,
        "sensor10_is_EXTERNALLY_WIRED_artifact_monitor":True,
        "controlled_reference_targets_slow_simulator_source":True,
        "source_decomposition_must_not_read_hidden_H_or_energy":True,
        "no_clinical_measurements":True,
    }


def apply_artifact_v38(x,mode,*,seed:int):
    x=np.asarray(x,dtype=np.float32)
    if x.ndim!=3 or x.shape[1:]!=(16,12):
        raise ValueError("Unregistered input shape")
    y=x.copy()
    rng=np.random.default_rng(seed)
    if mode=="none":
        pass
    elif mode=="offset_all":
        y[:,:,4:10]+=2.
    elif mode=="flip_A":
        y[:,:,4:6]*=-1
    elif mode=="noise_A":
        y[:,:,4:6]+=rng.normal(0,.9,size=(len(y),16,2))
    elif mode in (
        "shared_AB_monitored","shared_ABC_monitored",
        "shared_ABC_unmonitored","sentinel_only",
    ):
        signal=rng.normal(size=(len(y),16))
        for t in range(1,16):
            signal[:,t]+=0.8*signal[:,t-1]
        signal*=1.35
        if mode!="sentinel_only":
            end=8 if mode=="shared_AB_monitored" else 10
            y[:,:,4:end]+=signal[:,:,None].astype(np.float32)
        if mode!="shared_ABC_unmonitored":
            y[:,:,10]+=signal.astype(np.float32)
    else:
        raise ValueError("Unregistered measurement fault")
    np.testing.assert_array_equal(y[:,:,:4],x[:,:,:4])
    return y


def mask_world_v38(x,stress,*,seed:int,known_seed:int):
    v=np.asarray(x)
    if v.ndim!=3 or v.shape[1:]!=(16,12) or stress not in STRESSES:
        raise ValueError("Unregistered V38 mask")
    rng=np.random.default_rng(seed)
    m=np.ones(v.shape,dtype=bool)
    m[:,:,:4]=np.random.default_rng(known_seed).random((len(v),16,4))>=.20
    if stress=="MCAR20":
        m[:,:,4:10]=rng.random((len(v),16,6))>=.2
    elif stress=="MCAR40":
        m[:,:,4:10]=rng.random((len(v),16,6))>=.4
    elif stress=="synchronized_block8":
        starts=rng.integers(0,9,len(v))
        for i,st in enumerate(starts):
            m[i,st:st+8,4:10]=False
    elif stress=="missing_C":
        m[:,:,8:10]=False
    elif stress=="missing_all_views":
        m[:,:,4:10]=False
    m[:,:,10]=rng.random((len(v),16))>=.20
    if stress=="missing_sentinel":
        m[:,:,10]=False
    return m
