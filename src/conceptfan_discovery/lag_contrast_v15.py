"""V15 time-lag environment contrasts, conditional on temporally white sensor noise.

If X[e,t] = B Z[e,t] + eta[e,t], Cov(Z[e,t+1],Z[e,t])
= diag(rho * v[e]), and eta is source-independent, temporally uncorrelated
even when its instantaneous covariance Omega[e] changes by environment, then
L[e] = Cov(X[e,t+1],X[e,t]) = B diag(rho * v[e]) B.T.
Three environments with distinct relative contrasts conditionally identify
mixing column directions up to permutation, sign, scale. This is a classical
SOBI-style observation, NOT a new identification theorem. Correlated noise
in time defeats it; external measurement protocol must justify assumptions.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class LagCertificate:
    signature:np.ndarray
    demixing:np.ndarray
    variance_ratios:np.ndarray
    source_rank:int
    p_environment_null:float
    bootstrap_min_cosine_q10:np.ndarray
    bootstrap_min_gap_q10:np.ndarray
    n_per_environment:tuple[int,int,int]
    lag:int=1

    def to_json(self):
        return {
            "status":"CONDITIONAL_LAGGED_ENVIRONMENT_SOURCE_AXES",
            "n_source_directions_not_true_factor_count":self.source_rank,
            "n_per_environment":list(self.n_per_environment),
            "lag":self.lag,
            "permutation_p":self.p_environment_null,
            "bootstrapped_cosine_q10":self.bootstrap_min_cosine_q10.tolist(),
            "ratio_gap_q10":self.bootstrap_min_gap_q10.tolist(),
            "variance_ratios":self.variance_ratios.tolist(),
            "source_signatures":self.signature.tolist(),
            "white_sensor_noise_verified_from_observations":False,
            "unmeasured_colored_sensor_noise_identifiable":False,
            "scientific_concept_labels_identified":False,
            "concept_ambiguity":"permutation, sign and scale",
            "only_train_history_and_environment_used":True,
        }

def _lag(x:np.ndarray)->np.ndarray:
    if x.ndim!=3 or x.shape[1]<4:
        raise ValueError("ABSTAIN: need complete dynamic episodes")
    before=x[:,:-1,:].reshape(-1,x.shape[-1])
    after=x[:,1:,:].reshape(-1,x.shape[-1])
    before=before-before.mean(0)
    after=after-after.mean(0)
    c=after.T@before/(len(before)-1)
    return (c+c.T)*.5

def _moments(groups):
    c=[_lag(g) for g in groups]
    return c[1]-c[0],c[2]-c[0]

def _factor(d1,d2,k):
    vals,basis=np.linalg.eigh(d1)
    order=np.argsort(vals)[::-1]
    vals,basis=vals[order],basis[:,order]
    if k<2 or vals[k-1]<1e-8:
        raise ValueError("ABSTAIN: lag covariance contrast lacks positive rank")
    w=basis[:,:k]/np.sqrt(vals[:k])[None,:]
    h=w.T@d2@w
    h=(h+h.T)*.5
    ratios,v=np.linalg.eigh(h)
    order=np.argsort(ratios)[::-1]
    ratios,v=ratios[order],v[:,order]
    b=np.linalg.pinv(w@v).T
    norms=np.linalg.norm(b,axis=0)
    if min(norms)<1e-8:
        raise ValueError("ABSTAIN: degenerate recovered sensor direction")
    b=b/norms
    piv=np.argmax(np.abs(b),axis=0)
    b=b*np.sign(b[piv,np.arange(k)])[None,:]
    return b,np.linalg.pinv(b).T,ratios

def fit_lagged_contrast(train_history,train_env,*,
        sensor_temporal_white_documented:bool=False,
        seed:int=20261104,permutations:int=199,bootstraps:int=32,
        min_ratio_gap:float=.12,min_bootstrap_cosine:float=.83,
        null_alpha:float=.01)->LagCertificate:
    """Require external measurement-noise provenance, never infer it from X."""
    if not sensor_temporal_white_documented:
        raise ValueError("ABSTAIN: temporal whiteness requires external documentation")
    x=np.asarray(train_history,dtype=np.float64)
    labels=np.asarray(train_env)
    if (x.ndim!=3 or x.shape[0]!=len(labels) or x.shape[1]<8
        or x.shape[-1]<4 or not np.isfinite(x).all()
        or not np.isin(labels,[0,1,2]).all() or permutations<99
        or bootstraps<24 or not 0<null_alpha<.1):
        raise ValueError("ABSTAIN: complete labeled three-environment episodes required")
    groups=tuple(x[labels==e] for e in (0,1,2))
    counts=tuple(len(g) for g in groups)
    if min(counts)<160:
        raise ValueError("ABSTAIN: insufficient independent episodes per environment")
    d1,d2=_moments(groups)
    rng=np.random.default_rng(seed)
    pooled=np.concatenate(groups[:2])
    n0=len(groups[0])
    null=np.empty(permutations)
    for j in range(permutations):
        p=rng.permutation(len(pooled))
        null[j]=np.linalg.eigvalsh(_lag(pooled[p[n0:]])-_lag(pooled[p[:n0]]))[-1]
    vals=np.linalg.eigvalsh(d1)[::-1]
    pval=(1+int(np.count_nonzero(null>=vals[0])))/(permutations+1)
    if pval>null_alpha:
        raise ValueError("ABSTAIN: no reliable dynamic variance shift")
    cutoff=max(np.quantile(null,.99)*1.45,.04*max(vals[0],0),.023)
    k=int(np.sum(vals>cutoff))
    if k<2 or k>=x.shape[-1]:
        raise ValueError("ABSTAIN: dynamic contrast rank ambiguous or sensors saturated")
    sig,dual,ratios=_factor(d1,d2,k)
    gaps=-np.diff(ratios)
    if min(gaps)<min_ratio_gap:
        raise ValueError("ABSTAIN: dynamic variance ratio profiles not distinct")
    overlaps=np.empty((bootstraps,k))
    bgaps=np.empty((bootstraps,k-1))
    for j in range(bootstraps):
        bgs=tuple(g[rng.integers(len(g),size=len(g))] for g in groups)
        bd1,bd2=_moments(bgs)
        try:
            bs,_,br=_factor(bd1,bd2,k)
        except ValueError as exc:
            raise ValueError("ABSTAIN: dynamic rank not bootstrap reproducible") from exc
        overlaps[j]=np.sum(sig*bs,axis=0)
        bgaps[j]=-np.diff(br)
    q=np.quantile(overlaps,.1,axis=0)
    gapq=np.quantile(bgaps,.1,axis=0)
    if min(q)<min_bootstrap_cosine or min(gapq)<min_ratio_gap*.5:
        raise ValueError("ABSTAIN: directions or variance ratios unstable")
    return LagCertificate(sig,dual,ratios,k,float(pval),q,gapq,counts)

def project_lag_concepts(histories,cert):
    x=np.asarray(histories,dtype=float)
    if x.ndim!=3 or x.shape[-1]!=cert.signature.shape[0] or not np.isfinite(x).all():
        raise ValueError("ABSTAIN: incomplete or incompatible measurements")
    return x[:,-4:,:].mean(axis=1)@cert.demixing
