"""V60: sparse jointly calibrated native PyTorch measured-program risk head.

Unlike original K2's preset pair proxies, the source definitions are compiled
from an explicit causal grammar. Risk sees ONLY their bounded memberships;
no latent, raw-window or direct sensor-to-risk bypass.

Two variants: pure program head (constant TRAIN prior), and frozen K2
residual. Both are fitted on TRAIN Y with equal per-TRAIN-patient weight.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TOTAL_STEPS=220
REFIT_STEPS=80
LEARNING_RATE=.035
GROUP_L1=.0025
SLOPE_ANCHOR=.01
OFFSET_ANCHOR=.005
MAX_ACTIVE=6
MIN_SUPPORT=.5
MIN_GAIN=.005
MIN_VALID_GROUPS=4
BOOTSTRAP_REPS=399


@dataclass(frozen=True)
class ProgramHeadFitV60:
    variant:str
    n_candidates:int
    n_active:int
    active_indices:tuple[int,...]
    alpha:tuple[float,...]
    offset:tuple[float,...]
    gamma:tuple[float,...]
    train_prior_log_odds:float
    n_train:int
    n_train_subjects:int
    train_objective_first:float
    train_objective_last:float
    train_patient_macro_NLL:float
    uses_train_labels_for_optimization:bool=True
    raw_to_risk_direct_path:bool=False
    source_programs_fixed_without_Y:bool=True
    source_physical_identity_certified:bool=False
    stochastic_seeded_initialization_std:float=0.
    fixed_outcome_blind_source_shortlist:bool=False

    def passport(self):
        return asdict(self)


class SparseCompiledConceptHeadV60(nn.Module):
    """Native interpretable additive neural concept-only head."""

    def __init__(self,n):
        super().__init__()
        if not(1<=n<=24):raise ValueError("Needs 1..24 observed compiled concepts")
        self.alpha=nn.Parameter(torch.ones(n,dtype=torch.float64))
        self.offset=nn.Parameter(torch.zeros(n,dtype=torch.float64))
        self.gamma=nn.Parameter(torch.zeros(n,dtype=torch.float64))

    def forward(self,measured,support,baseline):
        x=torch.as_tensor(measured,dtype=torch.float64)
        s=torch.as_tensor(support,dtype=torch.float64)
        base=torch.as_tensor(baseline,dtype=torch.float64).reshape(-1)
        if (x.ndim!=2 or x.shape!=s.shape or x.shape[1]!=len(self.gamma)
            or len(base)!=len(x) or not bool(torch.isfinite(x).all())
            or not bool(torch.isfinite(s).all()) or
            not bool(torch.isfinite(base).all()) or
            bool((s<0).any()) or bool((s>1).any())):
            raise ValueError("Program-only risk requires finite measured values/masks")
        used=s>=MIN_SUPPORT
        bounded_slope=self.alpha.clamp(.5,3.)
        bounded_offset=self.offset.clamp(-2.,2.)
        q=.5+s*(torch.sigmoid(bounded_slope*x-bounded_offset)-.5)
        q=torch.where(used,q,.5)
        contributions=self.gamma*(q-.5)
        # Exact zero for missing concept, including nonzero alpha/gamma.
        contributions=torch.where(used,contributions,torch.zeros_like(contributions))
        risk=base+contributions.sum(dim=1)
        return {"logits":risk,"memberships":q,
                "used":used,"support":s,"signed_contributions":contributions,
                "baseline":base}


def _subject_equal_weights(groups):
    ids,inv,counts=np.unique(np.asarray(groups),return_inverse=True,return_counts=True)
    if len(ids)<12 or np.any(counts<=0):
        raise ValueError("TRAIN group identities required")
    weight=1./counts[inv]
    weight/=weight.mean()
    return weight,len(ids)


def _loss(output,y,weights,head,*,regularize=True):
    err=F.binary_cross_entropy_with_logits(
        output["logits"],y,reduction="none")
    task=(err*weights).mean()
    if not regularize:return task
    reg=(GROUP_L1*torch.sqrt(head.gamma.square()+1e-6).sum()
         +SLOPE_ANCHOR*(head.alpha-1.).square().mean()
         +OFFSET_ANCHOR*head.offset.square().mean())
    return task+reg


def _np_nll(y,z):
    return np.logaddexp(0.,z)-y*z


def group_macro_nll_v60(y,z,groups):
    yy=np.asarray(y,np.float64).reshape(-1)
    zz=np.asarray(z,np.float64).reshape(-1)
    group=np.asarray(groups)
    if (len(yy)!=len(zz) or len(group)!=len(yy)
        or not np.isfinite(zz).all() or not np.isin(yy,[0.,1.]).all()):
        raise ValueError("Bad subject-macro paired loss")
    err=_np_nll(yy,zz)
    return float(np.mean([err[group==p].mean() for p in np.unique(group)]))


def fit_program_head_v60(values,support,base_logits,train_y,train_groups,
                         *,variant,seed=20266021,
                         train_initialization_std=0.,
                         retain_all_predeclared_sources=False):
    x=np.asarray(values,np.float64)
    s=np.asarray(support,np.float64)
    base0=np.asarray(base_logits,np.float64).reshape(-1)
    y=np.asarray(train_y,np.float64).reshape(-1)
    if (variant not in ("PURE","RESIDUAL") or x.ndim!=2 or
        x.shape!=s.shape or len(x)!=len(y) or len(base0)!=len(y) or
        not(1<=x.shape[1]<=24) or len(y)<300 or
        not np.isin(y,[0.,1.]).all() or np.unique(y).size!=2 or
        not np.isfinite(x).all() or not np.isfinite(base0).all() or
        not 0.<=train_initialization_std<=.05 or
        (retain_all_predeclared_sources and x.shape[1]>MAX_ACTIVE)):
        raise ValueError("Need TRAIN only compiled measurement and binary outcomes")
    weight,n_groups=_subject_equal_weights(train_groups)
    prior=np.clip(np.mean(y),1e-5,1.-1e-5)
    prior_log_odds=float(np.log(prior/(1.-prior)))
    base=(np.full(len(y),prior_log_odds) if variant=="PURE" else base0)
    torch.manual_seed(seed)
    net=SparseCompiledConceptHeadV60(x.shape[1])
    if train_initialization_std>0.:
        # V61R genuinely distinct seeded neural initial conditions;
        # V60/V61 defaults remain exactly unchanged.
        with torch.no_grad():
            net.alpha.add_(train_initialization_std*torch.randn_like(net.alpha))
            net.offset.add_(train_initialization_std*torch.randn_like(net.offset))
            net.gamma.add_(train_initialization_std*torch.randn_like(net.gamma))
            net.alpha.clamp_(.5,3.)
            net.offset.clamp_(-2.,2.)
            net.gamma.clamp_(-10.,10.)
    xx=torch.as_tensor(x,dtype=torch.float64)
    ss=torch.as_tensor(s,dtype=torch.float64)
    bb=torch.as_tensor(base,dtype=torch.float64)
    yy=torch.as_tensor(y,dtype=torch.float64)
    ww=torch.as_tensor(weight,dtype=torch.float64)
    optimizer=torch.optim.Adam(net.parameters(),lr=LEARNING_RATE)
    history=[]
    for _ in range(TOTAL_STEPS):
        out=net(xx,ss,bb)
        loss=_loss(out,yy,ww,net)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            net.alpha.clamp_(.5,3.)
            net.offset.clamp_(-2.,2.)
            net.gamma.clamp_(-10.,10.)
        history.append(float(loss.detach()))
    # TRAIN-only sparse selection. Do not choose feature count by VALID/TEST.
    if retain_all_predeclared_sources:
        # Definition shortlist is frozen using TRAIN X only, not task gamma.
        active=tuple(range(x.shape[1]))
    else:
        active=np.argsort(-np.abs(net.gamma.detach().numpy()))[:MAX_ACTIVE]
        active=tuple(sorted(int(i) for i in active))
    not_selected=np.ones(x.shape[1],dtype=bool)
    not_selected[list(active)]=False
    others=torch.as_tensor(not_selected,dtype=torch.bool)
    with torch.no_grad():
        net.gamma[others]=0.
        net.alpha.requires_grad_(False)
        net.offset.requires_grad_(False)
    optimizer=torch.optim.Adam([net.gamma],lr=LEARNING_RATE)
    for _ in range(REFIT_STEPS):
        out=net(xx,ss,bb)
        loss=_loss(out,yy,ww,net,regularize=False)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        net.gamma.grad[others]=0.
        optimizer.step()
        with torch.no_grad():
            net.gamma[others]=0.
            net.gamma.clamp_(-10.,10.)
    with torch.no_grad():
        final=net(xx,ss,bb)["logits"].numpy()
        gamma=net.gamma.numpy()
        if not retain_all_predeclared_sources:
            active=tuple(i for i in active if abs(gamma[i])>.1)
        # For V61R: keep the entire outcome-blind shortlist, even when a
        # learned source gamma is weak. Never re-rank by label-trained gamma.
        keep=np.zeros(len(gamma),bool);keep[list(active)]=True
        net.gamma[~torch.as_tensor(keep)]=0.
        final=net(xx,ss,bb)["logits"].numpy()
    result=ProgramHeadFitV60(
        variant=variant,n_candidates=x.shape[1],n_active=len(active),
        active_indices=active,
        alpha=tuple(map(float,net.alpha.detach().numpy())),
        offset=tuple(map(float,net.offset.detach().numpy())),
        gamma=tuple(map(float,net.gamma.detach().numpy())),
        train_prior_log_odds=prior_log_odds,
        n_train=len(x),n_train_subjects=n_groups,
        train_objective_first=history[0],train_objective_last=history[-1],
        train_patient_macro_NLL=group_macro_nll_v60(y,final,train_groups),
        stochastic_seeded_initialization_std=float(train_initialization_std),
        fixed_outcome_blind_source_shortlist=bool(retain_all_predeclared_sources),
    )
    return result


def infer_program_head_v60(fit,measured,support,frozen_k2_logits):
    x=np.asarray(measured,np.float64)
    s=np.asarray(support,np.float64)
    base0=np.asarray(frozen_k2_logits,np.float64).reshape(-1)
    if (x.ndim!=2 or x.shape!=s.shape or x.shape[1]!=fit.n_candidates
        or len(x)!=len(base0)):
        raise ValueError("Changed program grammar or risk source dimension")
    prior=(np.full(len(base0),fit.train_prior_log_odds)
           if fit.variant=="PURE" else base0)
    network=SparseCompiledConceptHeadV60(fit.n_candidates)
    with torch.no_grad():
        network.alpha.copy_(torch.as_tensor(fit.alpha,dtype=torch.float64))
        network.offset.copy_(torch.as_tensor(fit.offset,dtype=torch.float64))
        network.gamma.copy_(torch.as_tensor(fit.gamma,dtype=torch.float64))
        output=network(x,s,prior)
        return {key:(value.detach().numpy().copy()
                     if isinstance(value,torch.Tensor) else value)
                for key,value in output.items()}


def _one_environment(y,baseline,proposed,used,groups,*,seed):
    group=np.asarray(groups)
    unique=np.unique(group)
    base=np.asarray(baseline,np.float64)
    pred=np.asarray(proposed,np.float64)
    labels=np.asarray(y,np.float64)
    coverage=float(np.mean(np.asarray(used,bool).any(axis=1)))
    if (len(unique)<MIN_VALID_GROUPS or len(labels)!=len(group)
        or len(labels)!=len(base) or len(pred)!=len(labels)):
        raise ValueError("At least 4 distinct patient groups required for admission")
    delta=_np_nll(labels,base)-_np_nll(labels,pred)
    per_group=np.asarray([delta[group==p].mean() for p in unique])
    gain=float(per_group.mean())
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(unique),size=(BOOTSTRAP_REPS,len(unique)))
    lower=float(np.quantile(per_group[idx].mean(axis=1),.01))
    ok=bool(gain>=MIN_GAIN and lower>0. and coverage>=.5)
    return {"patient_groups":len(unique),
            "patient_macro_NLL_gain":gain,
            "patient_bootstrap_99pct_lower":lower,
            "supported_episode_fraction":coverage,
            "passed":ok,
            "window_replicates_not_independent_patients":True}


def admit_program_head_v60(fit,valid1,valid2):
    """Whole-patient-group dual VALID gate; rejects to exact frozen native K2.

    Each valid tuple = (y, K2_logits, proposed_logits, program_use_matrix,
                       true_patient_group_ids).
    """
    v1=_one_environment(*valid1,seed=20266041)
    v2=_one_environment(*valid2,seed=20266042)
    approved=bool(fit.n_active>0 and v1["passed"] and v2["passed"])
    return {
        "chosen_variant":fit.variant,
        "approved":approved,
        "reason":("ADMITTED_TWO_PATIENT_CLUSTER_VALID"
                  if approved else "ABSTAIN_NO_REPLICATED_PATIENT_CLUSTER_GAIN"),
        "valid1":v1,"valid2":v2,
        "source_physical_identity_proven":False,
        "fit_used_only_train_labels":True,
        "validation_bootstrap_unit":"patient","bootstrap_reps":BOOTSTRAP_REPS,
    }


def apply_program_admission_v60(frozen_k2,proposed,used,*,approved):
    base=np.asarray(frozen_k2,np.float64).reshape(-1)
    p=np.asarray(proposed,np.float64).reshape(-1)
    evidence=np.asarray(used,bool)
    if evidence.shape[0]!=len(base) or p.shape!=base.shape:
        raise ValueError("Wrong admission test dimensions")
    # For a PURE program head, an episode with no programs must also
    # fall back to frozen K2 (not the unrelated pure TRAIN prior).
    eligible=evidence.any(axis=1)&bool(approved)
    result=np.where(eligible,p,base)
    assert np.array_equal(result[~eligible],base[~eligible])
    return result,eligible
