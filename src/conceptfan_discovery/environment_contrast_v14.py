"""V14: conditional, *observational* multi-environment source separation.

Assume X_e = B S_e + eta_e, shared full-column-rank B, source covariance
Cov(S_e)=diag(v_e), and COMMON measurement-noise covariance Cov(eta_e)=Omega.
With three ordered environment labels (0,1,2):
D1=Cov(X_1)-Cov(X_0)=B diag(v1-v0) B.T;
D2=Cov(X_2)-Cov(X_0)=B diag(v2-v0) B.T.
If every (v1-v0)>0 and the ratios (v2-v0)/(v1-v0) are all distinct,
whitening D1 then eigendecomposing D2 identifies columns of B up to
permutation, sign and nonzero scale (not names, causes, or diagnoses).

This is a *classical* covariance-difference / second-order blind-source
separation argument. It is NOT a general nonlinear ICA theorem. Statistical
checks are insufficient to verify independent latent sources or invariant
sensor noise; external domain evidence for those assumptions is necessary.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EnvironmentContrastCertificate:
    signature: np.ndarray
    demixing: np.ndarray
    variance_ratios: np.ndarray
    signal_eigenvalues: np.ndarray
    response_overlap_q10: np.ndarray
    ratio_gap_q10: np.ndarray
    permutation_null_p: float
    selected_rank: int
    n_train_per_environment: tuple[int, int, int]
    source_window: int
    random_seed: int

    def to_json(self) -> dict:
        return {
            "status": "CONDITIONAL_SECOND_ORDER_ENVIRONMENT_AXES",
            "selected_rank_from_train_only": self.selected_rank,
            "environment_labels_used": [0, 1, 2],
            "n_train_per_environment": list(self.n_train_per_environment),
            "history_window": self.source_window,
            "seed": self.random_seed,
            "variance_ratios_descending": self.variance_ratios.tolist(),
            "signal_eigenvalues": self.signal_eigenvalues.tolist(),
            "bootstrap_sensor_signature_cosine_q10": self.response_overlap_q10.tolist(),
            "bootstrap_adjacent_eigenvalue_gap_q10": self.ratio_gap_q10.tolist(),
            "environment_exchangeability_null_p": self.permutation_null_p,
            "unit_normalized_signed_sensor_mixing": self.signature.tolist(),
            "sensor_demixing": self.demixing.tolist(),
            "latent_factor_count_proven": False,
            "biological_or_causal_names_identified": False,
            "identifiable_up_to": "permutation, pivot-sign convention and source scale",
            "not_testable_from_covariance": [
                "conditional latent independence", "invariant sensor noise", "stable sensor mixing"
            ],
            "no_outcomes_no_oracle_no_intervention_targets": True,
        }


def _cov(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64)
    c = v - v.mean(axis=0, keepdims=True)
    return (c.T @ c) / (len(v) - 1)


def _moments(groups: tuple[np.ndarray, ...]) -> tuple[np.ndarray,np.ndarray]:
    s = [_cov(group) for group in groups]
    d1=s[1]-s[0]
    d2=s[2]-s[0]
    return (d1+d1.T)*.5,(d2+d2.T)*.5


def _factor(d1:np.ndarray,d2:np.ndarray,k:int):
    eigen,u=np.linalg.eigh(d1)
    order=np.argsort(eigen)[::-1]
    eigen,u=eigen[order],u[:,order]
    if not np.isfinite(eigen).all() or eigen[k-1] <= 1e-7:
        raise ValueError("ABSTAIN: positive environment covariance contrast has insufficient rank")
    whiten=u[:,:k] / np.sqrt(eigen[:k])[None,:]
    h=whiten.T @ d2 @ whiten
    h=.5*(h+h.T)
    ratio,v=np.linalg.eigh(h)
    order=np.argsort(ratio)[::-1]
    ratio,v=ratio[order],v[:,order]
    unmix=whiten @ v
    mix=np.linalg.pinv(unmix).T
    norms=np.linalg.norm(mix,axis=0)
    if np.min(norms)<=1e-10:
        raise ValueError("ABSTAIN: degenerate observable source direction")
    mix/=norms
    pivot=np.argmax(np.abs(mix),axis=0)
    signs=np.sign(mix[pivot,np.arange(k)])
    mix*=signs[None,:]
    # This rescaling preserves the duality signature.T @ demixing = I.
    demix=np.linalg.pinv(mix).T
    return mix,demix,ratio,eigen


def fit_environment_contrast(
    train_history:np.ndarray, train_environment:np.ndarray, *,
    seed:int=20261104, window:int=4, permutations:int=99,
    bootstraps:int=48, min_variance_ratio_gap:float=.10,
    min_signature_bootstrap_cosine_q10:float=.85,
    min_null_p:float=.05,
    ) -> EnvironmentContrastCertificate:
    """Train-only, unknown K, unknown action-to-factor map, no oracle labels.

    Rank is selected relative to a permutation largest-eigenvalue null,
    not a planted source count. The independent resampling unit is patient
    / episode, not correlated time points. Refuses unclear rank/rotations.
    """
    x=np.asarray(train_history,dtype=np.float64)
    env=np.asarray(train_environment)
    if (x.ndim!=3 or x.shape[0]!=len(env) or x.shape[1]<window or
        x.shape[2]<3 or not 2<=window<=x.shape[1] or
        not np.isfinite(x).all() or not np.isin(env,[0,1,2]).all() or
        permutations<99 or bootstraps<24 or
        not 0<min_variance_ratio_gap<1 or
        not 0<min_signature_bootstrap_cosine_q10<1 or
        not 0<min_null_p<.2):
        raise ValueError("ABSTAIN: strict observed three-environment TRAIN history required")
    counts=tuple(int((env==i).sum()) for i in range(3))
    if min(counts)<160:
        raise ValueError("ABSTAIN: too few independent episodes per environment")
    summary=x[:,-window:,:].mean(axis=1)
    groups=tuple(summary[env==e] for e in (0,1,2))
    d1,d2=_moments(groups)
    rng=np.random.default_rng(seed)
    pooled=np.concatenate(groups[:2])
    n0=len(groups[0]);n01=len(pooled)
    null_max=np.empty(permutations)
    for i in range(permutations):
        perm=rng.permutation(n01)
        delta=_cov(pooled[perm[n0:]])-_cov(pooled[perm[:n0]])
        null_max[i]=np.linalg.eigvalsh(delta)[-1]
    eigen=np.linalg.eigvalsh(d1)[::-1]
    # Conservative separation of signal eigenvalues from finite-sample
    # environment-label permutation noise; not an exact rank confidence bound.
    null_cut=np.quantile(null_max,.95)
    p=(1+np.count_nonzero(null_max>=eigen[0]))/(1+permutations)
    if p>min_null_p:
        raise ValueError("ABSTAIN: no statistically separable variance shift across environments")
    cutoff=max(null_cut*1.25,.045*max(eigen[0],0),.035)
    k=int(np.sum(eigen>cutoff))
    if k<2:
        raise ValueError("ABSTAIN: fewer than two empirical covariance-shift directions")
    if k>=x.shape[2]:
        raise ValueError("ABSTAIN: covariance contrast saturates sensors; invariant-noise directions untestable")
    signature,demix,ratios,eigenvalues=_factor(d1,d2,k)
    gaps=ratios[:-1]-ratios[1:]
    if np.min(gaps)<min_variance_ratio_gap:
        raise ValueError("ABSTAIN: indistinguishable environment variance profiles")
    overlap=np.empty((bootstraps,k))
    boot_gap=np.empty((bootstraps,k-1))
    for b in range(bootstraps):
        resample=tuple(g[rng.integers(len(g),size=len(g))] for g in groups)
        bd1,bd2=_moments(resample)
        try:
            bsig,_,bratio,beigen=_factor(bd1,bd2,k)
        except ValueError as exc:
            raise ValueError("ABSTAIN: bootstrap source rank unstable") from exc
        # Rankings are fixed by distinct variance profiles, not oracle alignment.
        overlap[b]=np.sum(signature*bsig,axis=0)
        boot_gap[b]=bratio[:-1]-bratio[1:]
    q10=np.quantile(overlap,.1,axis=0)
    gap10=np.quantile(boot_gap,.1,axis=0)
    if (np.min(q10)<min_signature_bootstrap_cosine_q10 or
        np.min(gap10)<min_variance_ratio_gap*.5):
        raise ValueError("ABSTAIN: bootstrap directions or variance-ratio gaps not reproducible")
    return EnvironmentContrastCertificate(
        signature,demix,ratios,eigenvalues[:k],q10,gap10,float(p),
        k,counts,window,int(seed),
    )


def project_environment_concepts(history:np.ndarray,certificate:EnvironmentContrastCertificate)->np.ndarray:
    x=np.asarray(history,dtype=np.float64)
    if (x.ndim!=3 or x.shape[-1]!=certificate.signature.shape[0] or
        x.shape[1]<certificate.source_window or not np.isfinite(x).all()):
        raise ValueError("Only complete measured sensor history is supported")
    avg=x[:,-certificate.source_window:,:].mean(axis=1)
    return (avg @ certificate.demixing).astype(np.float64)
