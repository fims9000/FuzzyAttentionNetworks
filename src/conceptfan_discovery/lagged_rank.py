"""Train-only, permutation-calibrated predictive rank from observation dynamics.

A distinct alternative to the existing square block-Hankel estimator. Stack
positive-lag sensor cross-covariances horizontally; under a sufficiently
excited observable linear latent-state model, their row rank identifies the
dimension of the detectable predictive subspace up to n_channels.

Null calibration permutes entire future patient trajectories against the past
to preserve within-patient autocorrelations, channel marginals and masks.
This is NOT a general identifiability theorem for nonlinear clinical data.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LaggedRankEvidence:
    rank: int
    singular_values: np.ndarray
    noise_max_threshold: float
    null_rank_thresholds: np.ndarray
    left_observable_loadings: np.ndarray
    bootstrap_rank_counts: np.ndarray
    patient_bootstrap_interval: tuple[int,int]
    lags: tuple[int,...]
    permutation_repetitions: int
    seed: int
    n_train_episodes: int

    def to_json(self) -> dict:
        return {
            "rank":int(self.rank),
            "singular_values":self.singular_values.tolist(),
            "familywise_noise_max_threshold":float(self.noise_max_threshold),
            "null_rank_thresholds":self.null_rank_thresholds.tolist(),
            "left_observable_loadings":self.left_observable_loadings.tolist(),
            "patient_bootstrap_rank_counts":self.bootstrap_rank_counts.tolist(),
            "patient_bootstrap_interval":[int(v) for v in self.patient_bootstrap_interval],
            "lags":list(self.lags),
            "permutation_repetitions":int(self.permutation_repetitions),
            "seed":int(self.seed),
            "train_episodes":int(self.n_train_episodes),
            "input_used":"train measurements and observation masks only",
            "latent_labels_used":False,
            "interpretation":"detected rank of positive-lag observable predictive covariance; no causal claim",
        }


def _matrix_from_centered(
    centered: np.ndarray,
    mask: np.ndarray,
    lags: tuple[int,...],
    future_permutation: np.ndarray | None = None,
) -> np.ndarray:
    n, _, d = centered.shape
    if future_permutation is None:
        future_permutation = np.arange(n)
    blocks=[]
    for lag in lags:
        past=centered[:,:-lag,:].reshape(-1,d)
        future=centered[future_permutation,lag:,:].reshape(-1,d)
        pm=mask[:,:-lag,:].reshape(-1,d)
        fm=mask[future_permutation,lag:,:].reshape(-1,d)
        count=pm.T@fm
        if np.any(count < 20):
            raise ValueError("Insufficient jointly observed positive-lag channel pairs")
        blocks.append((past.T@future)/count)
    return np.concatenate(blocks,axis=1)


def _center(x:np.ndarray,m:np.ndarray)->np.ndarray:
    means=(x*m).sum(axis=(0,1))/np.maximum(m.sum(axis=(0,1)),1.)
    return (x-means[None,None,:])*m


def fit_lagged_predictive_rank(
    train_values:np.ndarray,
    train_observed_mask:np.ndarray,
    *,
    lags:tuple[int,...]=(1,2,3),
    permutations:int=120,
    patient_bootstraps:int=50,
    alpha:float=.05,
    seed:int=20261221,
)->LaggedRankEvidence:
    x=np.asarray(train_values,dtype=np.float64)
    m=np.asarray(train_observed_mask,dtype=np.float64)
    if x.ndim!=3 or x.shape!=m.shape or x.shape[0]<80 or x.shape[2]<2:
        raise ValueError("Need >=80 train patients with matched [N,T,channels] arrays")
    if not np.isfinite(x).all() or not np.isfinite(m).all() or np.any((m<0)|(m>1)):
        raise ValueError("Measurements/masks must be finite and masks in [0,1]")
    if not lags or len(set(lags))!=len(lags) or any(v<1 or v>=x.shape[1] for v in lags):
        raise ValueError("Lags must be unique positive integers smaller than T")
    if permutations<25 or patient_bootstraps<5 or not .001<alpha<.25:
        raise ValueError("Unsupported permutation/bootstrap significance protocol")
    n,t,d=x.shape
    centered=_center(x,m)
    signal=_matrix_from_centered(centered,m,lags)
    u,s,_=np.linalg.svd(signal,full_matrices=False)
    # Orientation is fixed entirely by the train-observed largest loading.
    for j in range(d):
        anchor=int(np.argmax(np.abs(u[:,j])))
        if u[anchor,j]<0:
            u[:,j]*=-1
    rng=np.random.default_rng(int(seed))
    shuffled=np.empty((permutations,d))
    for b in range(permutations):
        perm=rng.permutation(n)
        null=_matrix_from_centered(centered,m,lags,perm)
        shuffled[b]=np.linalg.svd(null,compute_uv=False)
    threshold=float(np.quantile(shuffled[:,0],1-alpha))
    # The SAME null spectral-norm threshold is used on every singular value:
    # a conservative family-wise control, unlike per-rank thresholds.
    rank=int(np.count_nonzero(s>threshold))
    bootstrap=np.zeros(d+1,dtype=int)
    for _ in range(patient_bootstraps):
        take=rng.integers(n,size=n)
        xx,mm=x[take],m[take]
        sb=np.linalg.svd(_matrix_from_centered(_center(xx,mm),mm,lags),compute_uv=False)
        bootstrap[int(np.count_nonzero(sb>threshold))]+=1
    quantile_sample=np.repeat(np.arange(d+1),bootstrap)
    q0,q1=np.quantile(quantile_sample,(.025,.975))
    return LaggedRankEvidence(
        rank=rank,singular_values=s,
        noise_max_threshold=threshold,
        null_rank_thresholds=np.quantile(shuffled,1-alpha,axis=0),
        left_observable_loadings=u,
        bootstrap_rank_counts=bootstrap,
        patient_bootstrap_interval=(int(np.floor(q0)),int(np.ceil(q1))),
        lags=tuple(lags),permutation_repetitions=permutations,
        seed=int(seed),n_train_episodes=n,
    )
