"""Pre-registered measurement environments and falsifiable rank-invariance gate.

No artificial-mask environment is presented as an independent hospital. A
random thinning experiment can only test the response to a *known additional*
observation process, not correct original MNAR/clinical measurement policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .time_stratified_rank_v5 import NotEstimableRank


def apply_measurement_environment(
    observed_values:np.ndarray,
    observed_mask:np.ndarray,
    *,
    environment:str,
    seed:int,
)->tuple[np.ndarray,np.ndarray,dict]:
    x=np.asarray(observed_values,dtype=np.float32)
    mask=np.asarray(observed_mask,dtype=np.float32)
    if x.ndim!=3 or x.shape!=mask.shape or not np.isfinite(x).all() or not np.isfinite(mask).all():
        raise ValueError("Expected finite measured histories and masks")
    if not np.logical_or(mask==0,mask==1).all():
        raise ValueError("Masks must be exact binary observation events")
    n,t,d=x.shape
    rng=np.random.default_rng(int(seed))
    if environment=="original":
        keep_prob=np.ones((1,t,d),dtype=np.float32)
    elif environment=="mcar_thin":
        keep_prob=np.full((1,t,d),.65,dtype=np.float32)
    elif environment=="time_channel_mar":
        # Vary measurement opportunity by HOUR and sensor. Independent of
        # values conditional on time/channel; a test of time-MAR handling.
        hours=np.arange(t,dtype=np.float32)[None,:,None]
        sensors=np.arange(d,dtype=np.float32)[None,None,:]
        keep_prob=np.clip(.60+.24*np.sin(.4*hours+.7*sensors),.34,.84)
    elif environment=="observed_value_mnar":
        # Negative stress: more high-amplitude observed measurements become
        # missing. This simulates value-selective observation of measurements
        # that were originally available. It cannot recreate full clinical MNAR.
        keep_prob=np.clip(.57-.23*np.tanh(np.abs(x)),.26,.75)
    else:
        raise ValueError("Unknown prespecified measurement environment")
    # A common rng stream does not confer patient-independence between
    # treatments; every experiment is a nested view of the same training
    # patients and must not be used as independent replication.
    retained=(rng.random(size=(n,t,d))<keep_prob)&(mask>0)
    after=retained.astype(np.float32)
    changed=x*after
    receipt={
        "environment":environment,
        "seed":int(seed),
        "initial_observed_fraction":float(mask.mean()),
        "after_observed_fraction":float(after.mean()),
        "relative_retained_observations":float(after.sum()/max(mask.sum(),1.)),
        "selection_depends_on_current_observed_value":environment=="observed_value_mnar",
        "patient_latent_oracle_used":False,
        "source_patient_count":int(n),
        "independent_research_cohort":False,
        "observation_process_assumptions":"MCAR/time-channel conditional MAR" if environment!="observed_value_mnar"
                                       else "MNAR stress, no unbiased rank claim",
    }
    return changed,after,receipt


def principal_subspace_overlap(
    a:np.ndarray,b:np.ndarray,ka:int,kb:int,
)->float|None:
    if min(ka,kb)<1:
        return None
    ua=np.asarray(a,dtype=np.float64)[:,:ka]
    ub=np.asarray(b,dtype=np.float64)[:,:kb]
    if ua.ndim!=2 or ub.ndim!=2 or ua.shape[0]!=ub.shape[0]:
        raise ValueError("Subspaces must describe the same observed channels")
    singular=np.linalg.svd(ua.T@ub,compute_uv=False)
    return float((singular[:min(ka,kb)]**2).mean())


@dataclass(frozen=True)
class GateResult:
    status:str
    consensus_k:int|None
    reasons:tuple[str,...]
    method:str
    environment_ranks:dict[str,int|None]
    split_half_ranks:dict[str,tuple[int|None,int|None]]
    split_half_principal_overlap:dict[str,float|None]

    def json(self)->dict:
        return {
            "status":self.status,
            "consensus_k":self.consensus_k,
            "reasons":list(self.reasons),
            "estimator":self.method,
            "environments_ranks":self.environment_ranks,
            "half_sample_ranks":{k:list(v) for k,v in self.split_half_ranks.items()},
            "half_sample_subspace_overlap":self.split_half_principal_overlap,
            "interpretation":"structural rank stability criterion; no clinical semantic proof",
        }


def evaluate_rank_invariance(
    fitted:dict[str,dict],
    *,
    method:str,
    min_squared_principal_cosine:float=.75,
)->GateResult:
    """A predefined all-environments, all-subsamples gate; no oracle K input."""
    if not .5<=min_squared_principal_cosine<=1.:
        raise ValueError("Subspace overlap threshold must be in [.5,1]")
    if not fitted:
        raise ValueError("No prespecified observation environments")
    ranks,halves,overlaps={}, {},{}
    reasons=[]
    for environment,record in fitted.items():
        whole=record.get("whole")
        sa,sb=record.get("halves",(None,None))
        ranks[environment]=None if whole is None else int(whole.rank)
        halves[environment]=(
            None if sa is None else int(sa.rank),
            None if sb is None else int(sb.rank),
        )
        if whole is None or sa is None or sb is None:
            reasons.append(f"{environment}:insufficient_joint_measurement_support")
            overlaps[environment]=None
            continue
        if whole.rank<1:
            reasons.append(f"{environment}:no_predictive_dynamic_dimension")
        if not (whole.rank==sa.rank==sb.rank):
            reasons.append(f"{environment}:nonreproducible_split_half_rank")
        similarity=principal_subspace_overlap(
            sa.source_direction,sb.source_direction,sa.rank,sb.rank,
        )
        overlaps[environment]=similarity
        if similarity is None or similarity<min_squared_principal_cosine:
            reasons.append(f"{environment}:weak_split_half_subspace_overlap")
    unique={k for k in ranks.values() if k is not None}
    if len(unique)>1:
        reasons.append("noninvariant_rank_across_observation_environments")
    all_ranks={k for k in ranks.values() if k is not None}
    consensus=int(next(iter(all_ranks))) if not reasons and len(all_ranks)==1 else None
    return GateResult(
        status="MEASUREMENT_INVARIANT_OBSERVABLE_RANK" if consensus is not None else "ABSTAIN_UNIDENTIFIABLE",
        consensus_k=consensus,
        reasons=tuple(reasons),
        method=method,
        environment_ranks=ranks,
        split_half_ranks=halves,
        split_half_principal_overlap=overlaps,
    )
