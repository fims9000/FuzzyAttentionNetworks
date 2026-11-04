"""V22: elementary TRAIN-only second-order multisensor source unmixing.

This is a LIMITED established covariance-based BSS baseline, not a new
algorithm or theorem: whiten the pooled rank-two observations, diagonalize
their cross-environment covariance difference and test whether the two
directions are reliably distinguishable.

The source signatures can equally well represent physical dynamics or
instrument artifacts. Unknown physical truth never enters proposal.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
from hashlib import sha256

import numpy as np

from .mixed_source_data_v22 import (
    ObservableEnvironments,TrainWhitening,
    train_only_whitening,HISTORY,SOURCE_COUNT,N_SENSORS,
)


@dataclass(frozen=True)
class SourceSeparationEvidence:
    candidate_id: str
    status: str
    source_vectors_whitened: np.ndarray
    source_filters_original_sensor_space: np.ndarray
    environment_contrast_eigenvalues: tuple[float,float]
    contrast_eigen_gap_standardized: float
    environment_permutation_p: float
    bootstrap_min_direction_cosine_q10: float
    environment_sensor_subspace_min_cosine: float
    train_sample_count: int
    environment_counts: tuple[int,int]
    source_whitener: TrainWhitening
    source_origin_known_physically: bool = False
    oracle_latent_states_used_for_fit: bool = False
    task_labels_used_for_source_fit: bool = False
    external_source_identity_verified: bool = False


def _cov(a):
    return np.cov(a.reshape(-1,a.shape[-1]).T)


def _environment_cov(x,environment):
    cov=[]
    for e in (0,1):
        selected=x[environment==e]
        if len(selected)<100:
            raise ValueError("Each independent TRAIN environment needs >=100 episodes")
        cov.append(_cov(selected))
    return cov


def _two_source_subspace_overlap(x,environment):
    sensor_basis=[]
    for e in (0,1):
        cov=_cov(x[environment==e])
        values,vectors=np.linalg.eigh(cov)
        sensor_basis.append(vectors[:,-SOURCE_COUNT:])
    singular=np.linalg.svd(sensor_basis[0].T@sensor_basis[1],compute_uv=False)
    return float(np.min(singular))


def fit_train_only_environment_source_directions(
    observable:ObservableEnvironments,
    *,
    whiten:TrainWhitening|None=None,
    seed:int=522221,
    permutations:int=199,
    bootstraps:int=48,
    min_contrast_gap_standardized:float=.20,
    min_bootstrap_direction_abs_cosine:float=.80,
    min_environment_subspace_cosine:float=.99,
)->SourceSeparationEvidence:
    if not isinstance(observable,ObservableEnvironments):
        raise ValueError("Only independently registered observable TRAIN cohorts supported")
    if (not 99<=permutations<=999
            or not 24<=bootstraps<=999
            or not .1<=min_contrast_gap_standardized<=.8
            or not .5<=min_bootstrap_direction_abs_cosine<=.99
            or not .90<=min_environment_subspace_cosine<=.99999):
        raise ValueError("Invalid frozen environment-identification thresholds")
    train=np.asarray(observable.train_x_full[:,:HISTORY],dtype=np.float64)
    labels=np.asarray(observable.train_environment,dtype=np.int64)
    if (train.ndim!=3 or train.shape[-1]!=N_SENSORS
            or len(train)<400 or labels.shape!=(len(train),)
            or not np.array_equal(np.unique(labels),np.array([0,1]))
            or not np.isfinite(train).all()):
        raise ValueError("ABSTAIN: invalid independently observed TRAIN environment cohort")
    whitening=whiten if whiten is not None else train_only_whitening(observable)
    if whitening.trained_episodes!=len(train):
        raise ValueError("ABSTAIN: whitened map not fit to declared TRAIN cohort")
    if whitening.linear_map.shape!=(N_SENSORS,SOURCE_COUNT):
        raise ValueError("Invalid rank-two TRAIN whitened map")
    counts=tuple(int(np.sum(labels==e)) for e in (0,1))
    if min(counts)<150:
        raise ValueError("ABSTAIN: too few TRAIN patients in an environment")
    overlap=_two_source_subspace_overlap(train,labels)
    whitened=(train-whitening.center)@whitening.linear_map
    cov0,cov1=_environment_cov(whitened,labels)
    contrast=cov1-cov0
    val,eigenvec=np.linalg.eigh(contrast)
    order=np.argsort(val)[::-1]
    val=val[order]
    eigenvec=eigenvec[:,order]
    scaled_gap=float((val[0]-val[1]) /
                     max(float(np.trace((cov1+cov0)/2)),1e-10))
    # Preserve patient trajectories under the null by reassigning whole
    # episodes to environments instead of shuffling within trajectories.
    rng=np.random.default_rng(seed)
    null=np.empty(permutations,dtype=np.float64)
    for i in range(permutations):
        labels_perm=rng.permutation(labels)
        a,b=_environment_cov(whitened,labels_perm)
        delta_eig=np.linalg.eigvalsh(b-a)
        null[i]=float((delta_eig[-1]-delta_eig[0])/
                      max(float(np.trace((a+b)/2)),1e-10))
    p=float((1+np.count_nonzero(null>=scaled_gap))/(1+permutations))
    boot=np.empty(bootstraps,dtype=np.float64)
    for i in range(bootstraps):
        sampled=[]
        for e in (0,1):
            indices=np.flatnonzero(labels==e)
            selected=whitened[rng.choice(indices,size=len(indices),replace=True)]
            sampled.append(_cov(selected))
        _,b_evec=np.linalg.eigh(sampled[1]-sampled[0])
        b_evec=b_evec[:,::-1]
        # The estimand is the fixed *two source axes*, not the rank-two span
        # (which remains identifiable even when component directions are not).
        boot[i]=float(np.min(np.abs(np.diag(eigenvec.T@b_evec))))
    q10=float(np.quantile(boot,.1))
    raw_filters=whitening.linear_map@eigenvec
    # Fix a deterministic sign convention using the largest RAW-sensor
    # coefficient. This does not align to physical or instrument oracle truth.
    for k in range(SOURCE_COUNT):
        pivot=int(np.argmax(np.abs(raw_filters[:,k])))
        if raw_filters[pivot,k]<0:
            eigenvec[:,k]*=-1
            raw_filters[:,k]*=-1
    if overlap<min_environment_subspace_cosine:
        status="ABSTAIN_ENVIRONMENT_MIXING_CHANGE"
    elif scaled_gap<min_contrast_gap_standardized:
        status="ABSTAIN_NONIDENTIFIABLE_PROPORTIONAL_ENVIRONMENT_SIGNATURES"
    elif p>.01:
        status="ABSTAIN_NO_SIGNIFICANT_ENVIRONMENT_CONTRAST"
    elif q10<min_bootstrap_direction_abs_cosine:
        status="ABSTAIN_SOURCE_ORIENTATION_UNSTABLE_ON_TRAIN"
    else:
        status="ACCEPT_OBSERVATIONAL_ENVIRONMENT_UNMIXING"
    digest=sha256(np.round(raw_filters,8).tobytes()).hexdigest()[:16]
    return SourceSeparationEvidence(
        candidate_id="mixed-observable-environments-"+digest,
        status=status,
        source_vectors_whitened=eigenvec.astype(np.float64),
        source_filters_original_sensor_space=raw_filters.astype(np.float64),
        environment_contrast_eigenvalues=(float(val[0]),float(val[1])),
        contrast_eigen_gap_standardized=scaled_gap,
        environment_permutation_p=p,
        bootstrap_min_direction_cosine_q10=q10,
        environment_sensor_subspace_min_cosine=overlap,
        train_sample_count=len(train),
        environment_counts=counts,
        source_whitener=whitening,
    )


def separation_passport(value:SourceSeparationEvidence)->dict:
    data=asdict(value)
    data["source_vectors_whitened"]=value.source_vectors_whitened.tolist()
    data["source_filters_original_sensor_space"]=value.source_filters_original_sensor_space.tolist()
    data["source_whitener"]={
        "train_centered_sensor_means":value.source_whitener.center.tolist(),
        "train_rank_two_projection":value.source_whitener.linear_map.tolist(),
        "pooled_train_eigenvalues":value.source_whitener.pooled_eigenvalues.tolist(),
        "train_only":True,
    }
    data["source_identification_up_to_sign_and_permutation_only"]=True
    data["physical_vs_instrument_disambiguation_by_statistics_alone"]=False
    data["source_task_outcome_predictive_status_known"]=False
    data["declared_environment_mixing_change_can_escape_detection"]=True
    return data


def native_basis_from_separation(value:SourceSeparationEvidence)->np.ndarray:
    if value.status!="ACCEPT_OBSERVATIONAL_ENVIRONMENT_UNMIXING":
        raise ValueError("ABSTAIN: cannot insert nonidentifiable environment source in native model")
    mat=np.zeros((N_SENSORS,SOURCE_COUNT),dtype=np.float64)
    mat[:SOURCE_COUNT,:]=value.source_vectors_whitened
    if not np.allclose(mat.T@mat,np.eye(SOURCE_COUNT),atol=1e-8):
        raise AssertionError("Native BSS source axes must be orthonormal after whitening")
    return mat
