"""V15 paired-view environment contrasts; not a new BSS theorem.

x_A=A z+eta_A, x_B=B z+eta_B, with Cov(eta_A,eta_B|e)=0 and
source-sensor cross-noise covariance zero in all environments. Then
Cov(x_A,x_B|e)=A diag(v_e) B.T. Time-colored sensor noise is allowed
within each view. Cross-view correlated noise defeats the guarantee.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class CrossViewCertificate:
    signature_a:np.ndarray
    signature_b:np.ndarray
    demixing_a:np.ndarray
    source_rank:int
    variance_ratios:np.ndarray
    ratio_gap_q10:np.ndarray
    signature_cosine_q10:np.ndarray
    exchange_null_p:float
    counts:tuple[int,int,int]
    def to_json(self):
        return {"status":"CONDITIONAL_INDEPENDENT_CROSSVIEW_DIRECTIONS",
                "n_resolved_directions_not_true_count":self.source_rank,
                "n_per_environment":list(self.counts),
                "permutation_p":self.exchange_null_p,
                "ratios":self.variance_ratios.tolist(),
                "bootstrap_cosine_q10":self.signature_cosine_q10.tolist(),
                "bootstrap_ratio_gap_q10":self.ratio_gap_q10.tolist(),
                "independent_sensor_views_verified_from_data":False,
                "semantic_latent_identity_proven":False}

def _xcov(a,b):
    x=a-a.mean(axis=0);y=b-b.mean(axis=0)
    return x.T@y/(len(x)-1)

def _moments(g):
    c=[_xcov(a,b) for a,b in g]
    return c[1]-c[0],c[2]-c[0]

def _factor(d1,d2,k):
    u,s,vt=np.linalg.svd(d1,full_matrices=False)
    if k<2 or s[k-1]<1e-8:
        raise ValueError("ABSTAIN: cross-view contrast deficient rank")
    left=u[:,:k]/np.sqrt(s[:k])
    right=vt[:k,:].T/np.sqrt(s[:k])
    eigvals,eigvecs=np.linalg.eig(left.T@d2@right)
    if np.max(np.abs(eigvals.imag))>1e-4 or np.max(np.abs(eigvecs.imag))>1e-3:
        raise ValueError("ABSTAIN: complex generalized eigenstructure")
    order=np.argsort(eigvals.real)[::-1]
    ratio=eigvals.real[order]
    q=eigvecs.real[:,order]
    if np.linalg.cond(q)>1e4:
        raise ValueError("ABSTAIN: eigenvectors poorly conditioned")
    mix=u[:,:k]@np.diag(np.sqrt(s[:k]))@q
    norms=np.linalg.norm(mix,axis=0)
    if min(norms)<1e-10:
        raise ValueError("ABSTAIN: degenerate cross-view direction")
    mix=mix/norms
    pivot=np.argmax(np.abs(mix),axis=0)
    mix*=np.sign(mix[pivot,np.arange(k)])[None,:]
    other=d1.T@np.linalg.pinv(mix).T
    other/=np.maximum(np.linalg.norm(other,axis=0),1e-12)
    return mix,other,np.linalg.pinv(mix).T,ratio

def fit_crossview_environment(train_a,train_b,labels,*,
        independently_noisy_views_documented=False,seed=20261104,
        permutations=199,bootstraps=32,null_alpha=.01,
        min_gap=.13,min_signature_cosine=.82):
    if not independently_noisy_views_documented:
        raise ValueError("ABSTAIN: independent view-error provenance unavailable")
    a=np.asarray(train_a,dtype=float);b=np.asarray(train_b,dtype=float)
    e=np.asarray(labels)
    if (a.ndim!=3 or b.ndim!=3 or a.shape[:2]!=b.shape[:2]
        or len(a)!=len(e) or a.shape[1]<6
        or min(a.shape[-1],b.shape[-1])<3
        or not np.isfinite(a).all() or not np.isfinite(b).all()
        or not np.isin(e,[0,1,2]).all() or permutations<99 or bootstraps<24):
        raise ValueError("ABSTAIN: complete synchronous two-view episodes required")
    aa=a[:,-4:].mean(axis=1);bb=b[:,-4:].mean(axis=1)
    groups=tuple((aa[e==j],bb[e==j]) for j in range(3))
    counts=tuple(len(g[0]) for g in groups)
    if min(counts)<160:
        raise ValueError("ABSTAIN: too few independent episodes")
    d1,d2=_moments(groups)
    rng=np.random.default_rng(seed)
    pa=np.concatenate((groups[0][0],groups[1][0]))
    pb=np.concatenate((groups[0][1],groups[1][1]))
    n0=counts[0]
    null=np.empty(permutations)
    for i in range(permutations):
        perm=rng.permutation(len(pa))
        left=(pa[perm[:n0]],pb[perm[:n0]])
        right=(pa[perm[n0:]],pb[perm[n0:]])
        null[i]=np.linalg.svd(_xcov(*right)-_xcov(*left),compute_uv=False)[0]
    singular=np.linalg.svd(d1,compute_uv=False)
    p=(1+int((null>=singular[0]).sum()))/(permutations+1)
    if p>null_alpha:
        raise ValueError("ABSTAIN: no reliable cross-view shift")
    threshold=max(np.quantile(null,.99)*1.4,.04*singular[0],.018)
    k=int((singular>threshold).sum())
    if k<2 or k>=min(a.shape[-1],b.shape[-1]):
        raise ValueError("ABSTAIN: unresolved number or saturated view")
    sa,sb,dual,ratio=_factor(d1,d2,k)
    if min(-np.diff(ratio))<min_gap:
        raise ValueError("ABSTAIN: tied variance profiles")
    overlap=np.empty((bootstraps,k));gaps=np.empty((bootstraps,k-1))
    for i in range(bootstraps):
        sampled=tuple((x[idx],y[idx]) for x,y in groups
                      for idx in [rng.integers(len(x),size=len(x))])
        bd1,bd2=_moments(sampled)
        try:bs,_,_,br=_factor(bd1,bd2,k)
        except ValueError as exc:
            raise ValueError("ABSTAIN: cross-view bootstrap rank unstable") from exc
        overlap[i]=np.sum(sa*bs,axis=0)
        gaps[i]=-np.diff(br)
    cq=np.quantile(overlap,.1,axis=0);gq=np.quantile(gaps,.1,axis=0)
    if min(cq)<min_signature_cosine or min(gq)<min_gap*.5:
        raise ValueError("ABSTAIN: unstable cross-view signatures")
    return CrossViewCertificate(sa,sb,dual,k,ratio,gq,cq,float(p),counts)
