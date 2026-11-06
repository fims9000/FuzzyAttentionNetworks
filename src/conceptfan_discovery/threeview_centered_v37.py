"""V37: per-episode mean-centered, two- and three-view blind source-energy.

Classical second-order lag moment diagonalization, *not* a new BSS theorem.
All three 2x2 unknown mixing matrices are learned from masked TRAIN input
without oracle H, latent S, or outcome Y. Sensor view partition and
independent noise are STILL externally assumed by this controlled simulator.

The intended source is unsigned, episode-mean-centered fluctuation energy,
NOT a signed physical amplitude, clinical mechanism or causal concept.

New alpha/T3 layers do not resolve the independent-view noise assumption.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import expit

PAIRS={
    "AB":((4,5),(6,7)),
    "AC":((4,5),(8,9)),
    "BC":((6,7),(8,9)),
}


@dataclass(frozen=True)
class CenteredPairFitV37:
    pair_name:str
    view_a:tuple[int,int]
    view_b:tuple[int,int]
    left_eigenvectors:np.ndarray
    right_mixing:np.ndarray
    inv_left:np.ndarray
    inv_right:np.ndarray
    eigenvalues:np.ndarray
    spectral_gap:float
    bootstrap_abs_cos_q10:float
    train_lag0_pair_count:int
    train_lag1_pair_count:int
    train_energy_median:float
    train_energy_scale:float
    original_mixing_supplied:bool=False
    physically_signed_source_identified:bool=False

    def audit(self)->dict:
        return {
            "pair":self.pair_name,
            "view_a":self.view_a,
            "view_b":self.view_b,
            "lag_ordered_eigenvalues":self.eigenvalues.tolist(),
            "eigenvalue_gap":self.spectral_gap,
            "bootstrap_abs_cos_q10":self.bootstrap_abs_cos_q10,
            "train_lag0_pair_count":self.train_lag0_pair_count,
            "train_lag1_pair_count":self.train_lag1_pair_count,
            "center_policy":"measured-time episode-wise paired demeaning",
            "original_mixing_supplied":False,
            "physically_signed_source_identified":False,
        }


@dataclass(frozen=True)
class ThreeViewFitV37:
    pairs:dict[str,CenteredPairFitV37]
    train_max_pairwise_q_discrepancy_q95:float
    train_calibration_supported_episodes:int
    train_fit_episodes:int
    train_calibration_episodes:int
    view_partition_externally_assumed:bool=True
    independent_view_noise_externally_assumed:bool=True
    true_physical_semantics_certified:bool=False

    def audit(self)->dict:
        return {
            "pairs":{key:fit.audit() for key,fit in self.pairs.items()},
            "discrepancy_threshold":self.train_max_pairwise_q_discrepancy_q95,
            "discrepancy_threshold_derived_only_from_train":True,
            "train_calibration_supported_episodes":
                self.train_calibration_supported_episodes,
            "train_fit_episodes":self.train_fit_episodes,
            "train_calibration_episodes":self.train_calibration_episodes,
            "independent_view_noise_external_assumption":True,
            "view_partition_external_assumption":True,
            "latent_sources_and_true_mixing_not_used":True,
            "true_physical_semantics_certified":False,
        }


def _validate(x,m,*,min_episodes=1):
    values=np.asarray(x,dtype=np.float64)
    mask=np.asarray(m,dtype=bool)
    if (values.shape!=mask.shape or values.ndim!=3
        or values.shape[1:]!=(16,12) or len(values)<min_episodes
        or not np.isfinite(values[mask]).all()):
        raise ValueError("ABSTAIN: invalid masked 16x12 measurements")
    return values,mask


def _centered_pair(x,m,pair_name):
    if pair_name not in PAIRS:
        raise ValueError("Unknown predetermined independent view pair")
    a_idx,b_idx=PAIRS[pair_name]
    all_idx=list(a_idx+b_idx)
    paired=m[:,:,all_idx].all(axis=-1)
    n=paired.sum(axis=1)
    a=np.where(paired[:,:,None],x[:,:,list(a_idx)],0.)
    b=np.where(paired[:,:,None],x[:,:,list(b_idx)],0.)
    ca=a.sum(axis=1)/np.maximum(n,1)[:,None]
    cb=b.sum(axis=1)/np.maximum(n,1)[:,None]
    a=np.where(paired[:,:,None],a-ca[:,None,:],0.)
    b=np.where(paired[:,:,None],b-cb[:,None,:],0.)
    return a,b,paired,n


def _moments(a,b,paired):
    c0count=int(paired.sum())
    lag_mask=paired[:,1:] & paired[:,:-1]
    c1count=int(lag_mask.sum())
    if c0count<300 or c1count<250:
        raise ValueError("ABSTAIN: insufficient actual two-view lagged observations")
    c0=np.einsum("nti,ntj->ij",a,b,optimize=True)/c0count
    c1=np.einsum(
        "nti,ntj->ij",
        np.where(lag_mask[:,:,None],a[:,1:,:],0.),
        np.where(lag_mask[:,:,None],b[:,:-1,:],0.),
        optimize=True,
    )/c1count
    return c0,c1,c0count,c1count


def _blind_directions(c0,c1):
    if abs(float(np.linalg.det(c0)))<1e-5 or np.linalg.cond(c0)>1e3:
        raise ValueError("ABSTAIN: crossview matrix not full rank")
    eig,vec=np.linalg.eig(np.linalg.solve(c0.T,c1.T).T)
    if np.max(abs(eig.imag))>.03 or np.max(abs(vec.imag))>.03:
        raise ValueError("ABSTAIN: lag eigenvectors not real")
    idx=np.argsort(eig.real)[::-1]
    eig=eig.real[idx]
    vec=vec.real[:,idx]
    gap=float(eig[0]-eig[1])
    if gap<.25 or np.max(abs(eig))>1.1 or np.linalg.cond(vec)>50:
        raise ValueError("ABSTAIN: eigenvectors not spectrally identifiable")
    vec=vec/np.linalg.norm(vec,axis=0)[None,:]
    pivot=np.argmax(abs(vec),axis=0)
    vec*=np.where(vec[pivot,np.arange(2)]<0,-1.,1.)[None,:]
    right=np.linalg.solve(vec,c0).T
    if abs(float(np.linalg.det(right)))<1e-5 or np.linalg.cond(right)>100:
        raise ValueError("ABSTAIN: right view spectral factor singular")
    return eig,vec,right,gap


def pair_feature_v37(fit:CenteredPairFitV37,x,m,*,strict_min_pair_times=4):
    if strict_min_pair_times!=4:
        raise ValueError("Registered minimum measured paired times changed")
    v,mask=_validate(x,m)
    a,b,paired,n=_centered_pair(v,mask,fit.pair_name)
    left=(a@fit.inv_left.T)[:,:,0]
    right=(b@fit.inv_right.T)[:,:,0]
    energy=(left*right).sum(axis=1)/np.maximum(n,1)
    approved=(n>=4)&(energy>=.10)
    q=np.where(approved,
               expit((energy-fit.train_energy_median)/fit.train_energy_scale),.5)
    return {
        "q":q,
        "support":approved.astype(float),
        "raw_crossview_fluctuation_energy":energy,
        "paired_timepoints":n,
        "observed_paired_time_fraction":n/16.,
        "evidence_is_measured_not_generated":True,
        "real_signed_source_direction_identified":False,
    }


def _fit_pair(x,m,name,*,seed,bootstrap):
    v,mask=_validate(x,m,min_episodes=180)
    a,b,paired,_=_centered_pair(v,mask,name)
    c0,c1,n0,n1=_moments(a,b,paired)
    vals,vec,right,gap=_blind_directions(c0,c1)
    rng=np.random.default_rng(seed)
    cos=[]
    for _ in range(bootstrap):
        indices=rng.integers(0,len(v),len(v))
        try:
            ca,cb,_,_=_moments(a[indices],b[indices],paired[indices])
            _,bvec,_,_=_blind_directions(ca,cb)
        except ValueError as exc:
            raise ValueError("ABSTAIN: unstable source eigensystem under TRAIN bootstrap") from exc
        cos.append(abs(float(vec[:,0]@bvec[:,0])))
    cosq=float(np.quantile(cos,.10))
    if cosq<.80:
        raise ValueError("ABSTAIN: nonreproducible slow-source axis on TRAIN")
    first=CenteredPairFitV37(
        pair_name=name,view_a=PAIRS[name][0],view_b=PAIRS[name][1],
        left_eigenvectors=vec,right_mixing=right,
        inv_left=np.linalg.inv(vec),inv_right=np.linalg.inv(right),
        eigenvalues=vals,spectral_gap=gap,
        bootstrap_abs_cos_q10=cosq,
        train_lag0_pair_count=n0,train_lag1_pair_count=n1,
        train_energy_median=0.,train_energy_scale=1.,
    )
    r=pair_feature_v37(first,v,mask)
    good=r["support"]>0
    if good.mean()<.50:
        raise ValueError("ABSTAIN: source power unobservable in half of TRAIN")
    e=r["raw_crossview_fluctuation_energy"][good]
    med=float(np.median(e))
    scale=float(max(.20,(np.quantile(e,.75)-np.quantile(e,.25))/1.349))
    return CenteredPairFitV37(
        pair_name=name,view_a=PAIRS[name][0],view_b=PAIRS[name][1],
        left_eigenvectors=vec,right_mixing=right,
        inv_left=np.linalg.inv(vec),inv_right=np.linalg.inv(right),
        eigenvalues=vals,spectral_gap=gap,
        bootstrap_abs_cos_q10=cosq,
        train_lag0_pair_count=n0,train_lag1_pair_count=n1,
        train_energy_median=med,train_energy_scale=scale,
    )


def fit_centered_threeview_v37(
    train_x,train_mask,*,independent_view_noise_documented:bool,
    seed:int=20263701,bootstrap:int=24,
)->ThreeViewFitV37:
    if not independent_view_noise_documented:
        raise ValueError("ABSTAIN: no externally documented independent instrument views")
    if bootstrap!=24:
        raise ValueError("Frozen TRAIN direction bootstrap count changed")
    v,m=_validate(train_x,train_mask,min_episodes=300)
    rng=np.random.default_rng(seed)
    ix=rng.permutation(len(v))
    cut=int(.70*len(v))
    fit_ix,cal_ix=ix[:cut],ix[cut:]
    fits={
        name:_fit_pair(v[fit_ix],m[fit_ix],name,
                       seed=seed+91*i,bootstrap=bootstrap)
        for i,name in enumerate(PAIRS)
    }
    calibrated=[
        pair_feature_v37(fits[name],v[cal_ix],m[cal_ix])
        for name in PAIRS
    ]
    supported=np.stack([r["support"]>0 for r in calibrated],axis=1).all(axis=1)
    if int(supported.sum())<35:
        raise ValueError("ABSTAIN: insufficient TRAIN calibration triple-view support")
    qs=np.stack([r["q"] for r in calibrated],axis=1)
    gap=qs.max(axis=1)-qs.min(axis=1)
    threshold=float(max(.12,np.quantile(gap[supported],.95)))
    return ThreeViewFitV37(
        pairs=fits,
        train_max_pairwise_q_discrepancy_q95=threshold,
        train_calibration_supported_episodes=int(supported.sum()),
        train_fit_episodes=len(fit_ix),
        train_calibration_episodes=len(cal_ix),
    )


def threeview_feature_v37(fit:ThreeViewFitV37,x,m):
    v,mask=_validate(x,m)
    features={name:pair_feature_v37(fit.pairs[name],v,mask)
              for name in PAIRS}
    Q=np.column_stack([features[name]["q"] for name in PAIRS])
    S=np.column_stack([features[name]["support"] for name in PAIRS])
    all_supported=(S>0).all(axis=1)
    disagreement=Q.max(axis=1)-Q.min(axis=1)
    admitted=all_supported & (disagreement<=fit.train_max_pairwise_q_discrepancy_q95)
    q=np.where(admitted,np.median(Q,axis=1),.5)
    return {
        "q":q,
        "support":admitted.astype(float),
        "all_three_pairs_measured":all_supported,
        "pair_support":S,
        "pair_memberships":Q,
        "pair_energy":np.column_stack([
            features[name]["raw_crossview_fluctuation_energy"]
            for name in PAIRS]),
        "pair_timepoints":np.column_stack([
            features[name]["paired_timepoints"] for name in PAIRS]),
        "pairwise_max_membership_discrepancy":disagreement,
        "source_refused_due_to_missing_measurement":~all_supported,
        "source_refused_due_to_pair_disagreement":(
            all_supported & ~admitted),
        "no_original_mixing_used":True,
        "not_physically_certified":True,
        "no_imputation":True,
    }


def raw_ab_v36_feature(train_fit_or_none,x,m):
    """Explicit historic V36 uncentered two-view method, abstain if FIT fails.

    The V36 implementation remains frozen; it may fail TRAIN eigen recovery
    on a new random mixing. We do not modify the old function to force fit.
    """
    if train_fit_or_none is None:
        return {
            "q":np.full(len(x),.5,dtype=np.float64),
            "support":np.zeros(len(x),dtype=np.float64),
            "raw_method_fit_failed":True,
        }
    from conceptfan_discovery.gauge_source_v36 import source_energy_v36
    old=source_energy_v36(train_fit_or_none,x,m)
    return {**old,"raw_method_fit_failed":False}
