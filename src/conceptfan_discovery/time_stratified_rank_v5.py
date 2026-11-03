"""Time-stratified predictive covariance under explicitly limited missingness assumptions.

Observed-only pairwise lagged covariance in the v3 estimator effectively gives
hours with the most measurements more influence. For irregular clinical data
this changes the estimand when observation policies vary over time.

Here each supported source-hour receives EQUAL weight, using within-hour,
across-patient centering and only jointly measured lag pairs. In the
population it targets a time-averaged positive-lag cross-covariance when
missingness is independent of measurement VALUES conditional on hour and
environment (time-conditional MAR/MCAR) and there is positivity.

No estimator based only on masks can generally repair patient-treatment-
dependent / latent-dependent MNAR. Inadequate observed or permuted support
causes an explicit rejection, never a fabricated zero covariance.

A SINGLE max-singular-value threshold from patient-permutation null samples
controls false positives conservatively, conditional on the selected fixed
channels and temporal support protocol. No outcome labels or oracle states.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


class NotEstimableRank(ValueError):
    """Insufficient matched measurement support for the announced estimand."""


@dataclass(frozen=True)
class TimeStratifiedRank:
    rank: int
    singular_values: np.ndarray
    familywise_null_threshold: float
    source_direction: np.ndarray
    observed_time_strata_per_channel_pair: np.ndarray
    null_min_supported_time_strata_per_draw: np.ndarray
    minimum_pairs_per_hour: int
    minimum_time_strata: int
    hours_considered: int
    lags: tuple[int,...]
    seed: int
    n_train_patients: int
    null_repetitions: int

    def summary(self)->dict:
        return {
            "rank":int(self.rank),
            "singular_values":self.singular_values.tolist(),
            "familywise_null_threshold":float(self.familywise_null_threshold),
            "signed_observable_basis":self.source_direction.tolist(),
            "observed_time_strata_per_channel_pair":
                self.observed_time_strata_per_channel_pair.tolist(),
            "min_null_time_strata_per_draw":
                self.null_min_supported_time_strata_per_draw.astype(int).tolist(),
            "minimum_pairs_per_hour":int(self.minimum_pairs_per_hour),
            "minimum_time_strata":int(self.minimum_time_strata),
            "hours_considered":int(self.hours_considered),
            "lags":list(self.lags),
            "seed":int(self.seed),
            "patients":int(self.n_train_patients),
            "null_repetitions":int(self.null_repetitions),
            "train_observable_values_masks_only":True,
            "no_human_or_synthetic_concept_labels":True,
            "assumptions":[
                "independent patients within a stable observational environment",
                "missingness independent of values conditional on time and measured environment",
                "enough jointly observed time points for EVERY channel pair",
                "positive-lag observable and excited predictive dynamics",
                "finite independent-observation-noise support; no causal interpretation",
            ],
            "limitation":"time-dependent propensity balancing does NOT identify latent-dependent MNAR or true physiological ontology",
        }


def _precheck(x:np.ndarray, m:np.ndarray, lags:tuple[int,...])->tuple[np.ndarray,np.ndarray]:
    x=np.asarray(x,dtype=np.float64)
    m=np.asarray(m,dtype=np.float64)
    if x.ndim!=3 or x.shape!=m.shape or x.shape[0]<80 or x.shape[2]<2:
        raise ValueError("Expected >=80 independent train histories and [N,T,D] masks")
    if not np.isfinite(x).all() or not np.isfinite(m).all():
        raise ValueError("Measurements and observation masks must be finite")
    if not np.logical_or(m==0.,m==1.).all():
        raise ValueError("Masks must be exact 0/1, not implicit observation probabilities")
    if not lags or len(set(lags))!=len(lags) or min(lags)<1 or max(lags)>=x.shape[1]:
        raise ValueError("Lag values must be unique positive integers smaller than T")
    return x,m


def _time_centered(x:np.ndarray,m:np.ndarray)->np.ndarray:
    # Time-specific means eliminate between-patient time-policy mean drift,
    # without imputing or forecasting any patient-specific missing values.
    count=m.sum(axis=0)
    means=np.divide((x*m).sum(axis=0),count,
                    out=np.zeros_like(count),where=count>0)
    return (x-means[None,:,:])*m


def _stratified_operator(
    centered:np.ndarray,
    mask:np.ndarray,
    lags:tuple[int,...],
    *,
    permutation:np.ndarray|None,
    min_pairs:int,
    min_strata:int,
)->tuple[np.ndarray,np.ndarray]:
    n,t,d=centered.shape
    if permutation is None:
        permutation=np.arange(n)
    if permutation.shape!=(n,) or len(set(permutation.tolist()))!=n:
        raise ValueError("Patient index must be a whole-history permutation")
    blocks=[]
    all_support=[]
    for lag in lags:
        numerator=np.zeros((d,d),dtype=np.float64)
        strata=np.zeros((d,d),dtype=np.int32)
        for hour in range(t-lag):
            a=centered[:,hour,:]
            b=centered[permutation,hour+lag,:]
            am=mask[:,hour,:]
            bm=mask[permutation,hour+lag,:]
            count=am.T@bm
            valid=count>=min_pairs
            if valid.any():
                cross=a.T@b
                numerator+=np.divide(cross,count,
                                     out=np.zeros((d,d)),where=valid)
                strata+=valid.astype(np.int32)
        if (strata<min_strata).any():
            raise NotEstimableRank(
                f"Positive-lag channels lack {min_strata} supported hourly strata; "
                f"lag={lag}, minimum={int(strata.min())}, pairs/hour={min_pairs}"
            )
        blocks.append(numerator/strata)
        all_support.append(strata)
    return np.concatenate(blocks,axis=1),np.stack(all_support)


def fit_time_stratified_rank(
    train_values:np.ndarray,
    train_mask:np.ndarray,
    *,
    lags:tuple[int,...]=(1,2,3),
    min_pairs_per_hour:int=12,
    min_time_strata:int=3,
    permutations:int=80,
    alpha:float=.05,
    seed:int=20270101,
)->TimeStratifiedRank:
    x,m=_precheck(train_values,train_mask,lags)
    if min_pairs_per_hour<4 or min_time_strata<2 or permutations<25 or not .001<alpha<.25:
        raise ValueError("Unsupported per-hour joint support or permutation test protocol")
    centered=_time_centered(x,m)
    signal,observed_support=_stratified_operator(
        centered,m,lags,permutation=None,
        min_pairs=min_pairs_per_hour,min_strata=min_time_strata,
    )
    u,s,_=np.linalg.svd(signal,full_matrices=False)
    for j in range(u.shape[1]):
        sign_anchor=np.argmax(np.abs(u[:,j]))
        if u[sign_anchor,j]<0:
            u[:,j]*=-1

    rng=np.random.default_rng(int(seed))
    max_singular=np.zeros(permutations,dtype=np.float64)
    min_null_support=np.empty(permutations,dtype=np.int32)
    for p in range(permutations):
        # Do not silently redraw an invalid permutation; it is evidence
        # that the measurement process cannot support this null.
        shuffled=rng.permutation(len(x))
        null,ns=_stratified_operator(
            centered,m,lags,permutation=shuffled,
            min_pairs=min_pairs_per_hour,min_strata=min_time_strata,
        )
        min_null_support[p]=int(ns.min())
        max_singular[p]=np.linalg.svd(null,compute_uv=False)[0]
    threshold=float(np.quantile(max_singular,1-alpha))
    rank=int(np.count_nonzero(s>threshold))
    return TimeStratifiedRank(
        rank=rank,singular_values=s,
        familywise_null_threshold=threshold,
        source_direction=u,
        observed_time_strata_per_channel_pair=observed_support,
        null_min_supported_time_strata_per_draw=min_null_support,
        minimum_pairs_per_hour=min_pairs_per_hour,
        minimum_time_strata=min_time_strata,
        hours_considered=x.shape[1],
        lags=tuple(int(v) for v in lags),
        seed=int(seed),n_train_patients=len(x),
        null_repetitions=permutations,
    )
