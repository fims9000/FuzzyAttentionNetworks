"""V36: outcome-blind two-view lag-spectrum source energy (classical SOBI-style).

X_A=A S+noiseA, X_B=B S+noiseB; C_AB(1) C_AB(0)^-1
recovers the lag-ordered A directions only under independent-view errors,
full mixing rank, distinct lag signatures and zero-mean stationary sources.
An automatically found spectral direction is NOT a named signed physical factor.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class GaugeSourceFitV36:
    mixing_a: np.ndarray
    mixing_b: np.ndarray
    inv_a: np.ndarray
    inv_b: np.ndarray
    lag_ratios: np.ndarray
    lag_gap: float
    boot_abs_cos_q10: float
    train_pair_count_lag0: int
    train_pair_count_lag1: int
    train_median_power: float
    train_power_scale: float
    independent_view_errors_documented: bool
    true_signed_source_direction_certified: bool=False
    physical_semantics_certified: bool=False

    def receipt(self):
        return {
            "source_definition": "blind largest lag-one eigenvalue cross-view source energy",
            "lag_ratios": self.lag_ratios.tolist(),
            "min_lag_gap": self.lag_gap,
            "bootstrap_cosine_q10": self.boot_abs_cos_q10,
            "pair_count_lag0": self.train_pair_count_lag0,
            "pair_count_lag1": self.train_pair_count_lag1,
            "train_median_power": self.train_median_power,
            "train_power_scale": self.train_power_scale,
            "assumption_independent_view_errors_documented": self.independent_view_errors_documented,
            "signed_source_identity_certified": False,
            "causal_physical_concept_certified": False,
            "original_mixing_matrix_supplied": False,
            "not_a_new_BSS_theorem": True,
        }


def _inputs(x, m):
    x=np.asarray(x,dtype=np.float64)
    m=np.asarray(m,dtype=bool)
    if (x.ndim!=3 or x.shape!=m.shape or x.shape[1]!=16
        or x.shape[2]!=12 or len(x)<100 or not np.isfinite(x[m]).all()):
        raise ValueError("ABSTAIN: expected aligned [N,16,12] observations and masks")
    return x,m


def _cross(x, m, lag):
    if lag not in (0,1):
        raise ValueError("Only preregistered lags 0 and 1")
    paired=m[:,:,4:8].all(axis=-1)
    both=paired[:,lag:] & paired[:,:16-lag]
    a=x[:,lag:,4:6][both]
    b=x[:,:16-lag,6:8][both]
    if len(a)<350:
        raise ValueError("ABSTAIN: insufficient two-view paired moments")
    return a.T@b/len(a),int(len(a))


def _directions(c0,c1):
    if np.linalg.cond(c0)>1e3 or abs(np.linalg.det(c0))<1e-4:
        raise ValueError("ABSTAIN: zero-lag cross-view rank deficient")
    vals,vec=np.linalg.eig(c1@np.linalg.inv(c0))
    if np.max(abs(vals.imag))>.03 or np.max(abs(vec.imag))>.03:
        raise ValueError("ABSTAIN: non-real lagged source directions")
    idx=np.argsort(vals.real)[::-1]
    vals=vals.real[idx]
    vec=vec.real[:,idx]
    if vals[0]-vals[1]<.25:
        raise ValueError("ABSTAIN: temporal signatures not separated")
    if (np.max(abs(vals))>.99 or np.linalg.cond(vec)>50):
        raise ValueError("ABSTAIN: unstable lag-spectrum")
    vec=vec/np.linalg.norm(vec,axis=0)[None,:]
    piv=np.argmax(np.abs(vec),axis=0)
    vec*=np.sign(vec[piv,np.arange(2)])[None,:]
    b=np.linalg.solve(vec,c0).T
    if abs(np.linalg.det(b))<1e-5 or np.linalg.cond(b)>100:
        raise ValueError("ABSTAIN: other view singular")
    return vals,vec,b


def source_energy_v36(fit,x,m,*,minimum_pairs=4,min_positive_power=.10):
    x,m=_inputs(x,m)
    if minimum_pairs!=4 or min_positive_power!=.10:
        raise ValueError("Frozen V36 observation thresholds changed")
    paired=m[:,:,4:8].all(axis=-1)
    counts=paired.sum(axis=1)
    a=x[:,:,4:6]@fit.inv_a.T
    b=x[:,:,6:8]@fit.inv_b.T
    cross=a[:,:,0]*b[:,:,0]
    energy=np.where(paired,cross,0.).sum(axis=1)/np.maximum(counts,1)
    supported=(counts>=minimum_pairs)&(energy>=min_positive_power)
    q=np.where(supported,expit((energy-fit.train_median_power)
                               /fit.train_power_scale),.5)
    return {
        "q":q,"support":supported.astype(float),
        "paired_sensor_times":counts,
        "actually_observed_pair_fraction":counts/16.,
        "estimated_crossview_energy":energy,
        "no_external_mixing_matrix_used":True,
        "simultaneous_global_sign_invariant":True,
        "physically_named_signed_source_is_identified":False,
        "unobserved_data_are_not_imputed":True,
    }


def fit_blind_gauge_source_v36(
    train_x,train_mask,*,independent_view_errors_documented:bool,
    seed:int=20263601,bootstrap:int=24,
)->GaugeSourceFitV36:
    if not independent_view_errors_documented:
        raise ValueError("ABSTAIN: cross-view independence needs external provenance")
    if bootstrap not in (24,32):
        raise ValueError("Frozen bootstrap resample count changed")
    x,m=_inputs(train_x,train_mask)
    c0,n0=_cross(x,m,0)
    c1,n1=_cross(x,m,1)
    ratio,a,b=_directions(c0,c1)
    rng=np.random.default_rng(seed)
    overlap=[]
    for _ in range(bootstrap):
        index=rng.integers(len(x),size=len(x))
        z0,_=_cross(x[index],m[index],0)
        z1,_=_cross(x[index],m[index],1)
        try:
            _,av,_=_directions(z0,z1)
        except ValueError as exc:
            raise ValueError("ABSTAIN: bootstrap source recovery unstable") from exc
        overlap.append(abs(float(a[:,0]@av[:,0])))
    cos10=float(np.quantile(overlap,.10))
    if cos10<.80:
        raise ValueError("ABSTAIN: source direction is unstable across TRAIN resamples")
    prelim=GaugeSourceFitV36(a,b,np.linalg.inv(a),np.linalg.inv(b),
         ratio,float(ratio[0]-ratio[1]),cos10,n0,n1,0.,1.,True)
    rec=source_energy_v36(prelim,x,m)
    z=rec["estimated_crossview_energy"][rec["support"]>0]
    if len(z)<.72*len(x):
        raise ValueError("ABSTAIN: too few TRAIN episodes with measured two-view energy")
    tau=float(np.median(z))
    width=float(max(.2,(np.quantile(z,.75)-np.quantile(z,.25))/1.349))
    return GaugeSourceFitV36(a,b,np.linalg.inv(a),np.linalg.inv(b),
         ratio,float(ratio[0]-ratio[1]),cos10,n0,n1,tau,width,True)
