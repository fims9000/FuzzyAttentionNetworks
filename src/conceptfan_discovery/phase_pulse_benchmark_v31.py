"""V31 nonlinear pulse-order synthetic benchmark (distinct from prior AR-like law).

Independent per-episode pulse order, nuisance and recording-device artifact
amplitudes are supplied by GENERATOR only. TRAIN feature/proposal algorithms
receive ONLY 12 noisy measurements/masks and train outcome labels at the
declared risk-head stage; no oracle amplitudes/true concepts are used in
candidate proposals or baseline K2 stage-one training.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

SEED=20261116
TRAIN_N,VALID_N,TEST_N=1500,1000,800
T=16
D=12
STRESS=("complete","MCAR20","MCAR40","MNAR","block8")
CASES=(
    ("physical_benign","physical","benign","none"),
    ("physical_known_phase_reversal","physical","flip_physical","flip_physical"),
    ("artifact_known_phase_reversal","artifact","flip_artifact","flip_artifact"),
    ("physical_unseen_phase_reversal","physical","benign","flip_physical"),
    ("complete_benign","complete","benign","none"),
)


def phase_pulse_world(*,seed:int=SEED,n_train:int=TRAIN_N,
                      n_valid:int=VALID_N,n_test:int=TEST_N)->dict:
    if min(n_train,n_valid,n_test)<100 or n_valid%2:
        raise ValueError("All splits require >=100 episodes and two even VALID cohorts")
    n=n_train+n_valid+n_test
    rng=np.random.default_rng(seed)
    t=np.arange(T,dtype=np.float64)[None,:]
    groups=[]
    for s in range(2):
        base=rng.uniform(-1.5,1.5,n)[:,None]
        phase=rng.uniform(-np.pi,np.pi,n)[:,None]
        trend=.30*np.sin(2*np.pi*t/T+phase)
        shocks=(rng.random((n,T))<.07)*rng.choice(
            [-1.,1.],size=(n,T))*rng.uniform(.2,.65,size=(n,T))
        z=(base+trend+shocks+.09*rng.standard_normal((n,T)))
        groups.append(z)
    amp=rng.uniform(-1.45,1.45,n)
    nuisance_amp=rng.uniform(-1.45,1.45,n)
    artifact_amp=rng.uniform(-1.45,1.45,n)
    waveform=np.r_[np.ones(T//2),-np.ones(T//2)][None,:]
    # Distinct from AR: signed phase pulse has near-zero full-episode mean.
    def pulse(a):
        noise=rng.normal(0,.07,(n,T))
        noise-=noise.mean(axis=1,keepdims=True)
        return a[:,None]*waveform+noise
    physical=pulse(amp)
    nuisance=pulse(nuisance_amp)
    artifact=pulse(artifact_amp)
    x=np.empty((n,T,D),dtype=np.float32)
    for z,channels,std in (
        (groups[0],(0,1),.26),
        (groups[1],(2,3),.26),
        (physical,(4,5,6),.30),
        (nuisance,(7,8),.30),
        (artifact,(9,10),.30),
    ):
        for j in channels:
            x[:,:,j]=(z+rng.normal(0,std,(n,T))).astype(np.float32)
    x[:,:,11]=rng.normal(0.,1.1,(n,T)).astype(np.float32)
    true_known=np.stack([expit(groups[0]),expit(groups[1])],axis=-1).astype(np.float32)
    mean_known=true_known.mean(axis=1)
    base_logit=3.7*(mean_known[:,0]-.5)-3.6*(mean_known[:,1]-.5)-.10
    # Pulse order is predictive; its WHOLE HISTORY MEAN is not.
    pred_logit=base_logit+2.5*amp
    artifact_logit=base_logit+2.5*artifact_amp
    # Share one Bernoulli draw across worlds to isolate outcome-law changes.
    u=rng.random(n)
    labels={
        "complete":(u<expit(base_logit)).astype(np.int64),
        "physical":(u<expit(pred_logit)).astype(np.int64),
        "artifact":(u<expit(artifact_logit)).astype(np.int64),
    }
    if any(len(np.unique(v))!=2 for v in labels.values()):
        raise AssertionError("All registered outcome laws must contain both labels")
    return {
        "x":x,"outcomes":labels,
        "train":np.arange(n_train),
        "valid1":np.arange(n_train,n_train+n_valid//2),
        "valid2":np.arange(n_train+n_valid//2,n_train+n_valid),
        "test":np.arange(n_train+n_valid,n),
        "true_known_CONCEPTS_AUDIT_ONLY":true_known,
        "amplitude_CONCEPTS_AUDIT_ONLY":{
            "physical":amp,"artifact":artifact_amp,"nuisance":nuisance_amp,
        },
        "real_source_group_mapping_AUDIT_ONLY":{
            "physical":[4,5,6],"nuisance":[7,8],
            "recording_artifact":[9,10],"distractor":[11],
        },
        "generator_family":"non_autoregressive_signed_two_phase_pulse_contrast",
        "timestamp_count":T,
        "no_physical_causal_identity_inferred_from_outcomes":True,
        "pulse_has_zero_true_full_history_mean":bool(
            np.max(np.abs(physical.mean(axis=1)))<1e-6),
    }


def device_environment(x:np.ndarray,regime:str)->np.ndarray:
    y=np.asarray(x,dtype=np.float32)
    if y.ndim!=3 or y.shape[1:]!=(T,D) or not np.isfinite(y).all():
        raise ValueError("Sensor environment must be finite [episode,16,12]")
    out=y.copy()
    if regime=="none":
        pass
    elif regime=="benign":
        out[:,:,4:12]=1.10*out[:,:,4:12]+.05
    elif regime=="flip_physical":
        out[:,:,4:7]*=-1.
    elif regime=="flip_artifact":
        out[:,:,9:11]*=-1.
    else:
        raise ValueError("Unknown preregistered device regime")
    np.testing.assert_array_equal(out[:,:,:4],y[:,:,:4])
    return out


def measured_mask(x:np.ndarray,mode:str,*,seed:int)->np.ndarray:
    values=np.asarray(x)
    if values.ndim!=3 or values.shape[1:]!=(T,D) or not np.isfinite(values).all():
        raise ValueError("Invalid finite 12-channel 16-step waveform")
    rng=np.random.default_rng(seed)
    mask=np.ones(values.shape,dtype=bool)
    if mode=="complete":
        return mask
    if mode in ("MCAR20","MCAR40"):
        p={"MCAR20":.2,"MCAR40":.4}[mode]
        return rng.random(values.shape)>=p
    if mode=="MNAR":
        p=.07+.7*expit(1.25*values)
        return rng.random(values.shape)>=p
    if mode=="block8":
        mask=rng.random(values.shape)>=.05
        starts=rng.integers(0,T-8+1,size=(len(values),D))
        for i in range(len(values)):
            for j in range(D):
                mask[i,starts[i,j]:starts[i,j]+8,j]=False
        return mask
    raise ValueError("Unregistered TEST missingness mode")


def temporal_sensor_features(x:np.ndarray,mask:np.ndarray)->np.ndarray:
    """Strong raw logistic with early/late temporal evidence, not mean-only."""
    v=np.asarray(x,dtype=np.float64)
    m=np.asarray(mask)
    if v.ndim!=3 or v.shape!=m.shape or v.shape[1:]!=(T,D):
        raise ValueError("Misaligned raw temporal feature input")
    parts=[]
    for section in (slice(None),slice(0,T//2),slice(T//2,None)):
        w=m[:,section,:]
        p=np.where(w,v[:,section,:],0.)
        summed=p.sum(axis=1)/np.maximum(1,w.sum(axis=1))
        parts.extend((summed,w.mean(axis=1)))
    return np.concatenate(parts,axis=1)
