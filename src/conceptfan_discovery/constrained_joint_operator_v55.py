"""V55: TRAIN-only constrained joint calibration of a discovered measured concept.

A frozen V54 source passport fixes which sensors, lag and operator are allowed.
TRAIN Y may optimize *only* a positive two-input mixture, bounded temperature
and the single signed coefficient in the native concept-only risk path.
No physical/medical semantics are inferred; no hidden raw predictor path.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .lagged_operator_concepts_v53 import _check
from .operator_library_concepts_v54 import (
    MeasuredOperatorConceptV54,infer_operator_concept_v54,OP_EVEN,OP_SIGNED,
)
from .isolated_source_admission_v28 import train_source_coefficient

TRAIN_STEPS=80
LEARNING_RATE=.03
MAX_ABS_GAMMA=10.
MIN_TEMPERATURE=.5
MAX_TEMPERATURE=2.
Q_ANCHOR_WEIGHT=.10
MIX_WEIGHT=.01
TEMPERATURE_WEIGHT=.01
GAMMA_WEIGHT=.0005


@dataclass(frozen=True)
class JointMeasuredConceptFitV55:
    candidate: MeasuredOperatorConceptV54
    source_weight: float
    readout_weight: float
    temperature: float
    gamma: float
    initial_gamma: float
    n_train: int
    train_objective_first: float
    train_objective_last: float
    supervised_definition_tuned_on_train_only: bool = True
    candidate_sensor_lag_operator_fixed_from_unlabeled_train: bool = True
    no_latent_or_raw_risk_bypass: bool = True
    physical_identity_attested: bool = False

    def passport(self):
        return asdict(self)


class ConstrainedMeasuredOperatorV55(nn.Module):
    def __init__(self,candidate:MeasuredOperatorConceptV54,gamma0=0.):
        super().__init__()
        if (candidate.family not in (OP_EVEN,OP_SIGNED)
            or candidate.readout_sign not in (-1,1)
            or candidate.source_scale<=0 or candidate.readout_scale<=0
            or not np.isfinite([candidate.source_center,candidate.source_scale,
                                candidate.readout_center,candidate.readout_scale]).all()):
            raise ValueError("Invalid fixed measured operator passport")
        self.candidate=candidate
        self.mix_logits=nn.Parameter(torch.zeros(2,dtype=torch.float64))
        self.log_temperature=nn.Parameter(torch.zeros((),dtype=torch.float64))
        self.gamma=nn.Parameter(torch.tensor(float(gamma0),dtype=torch.float64))

    def memberships(self,x,mask,last_steps=6):
        c=self.candidate
        if (x.ndim!=3 or x.shape!=mask.shape or x.dtype!=torch.float64
            or mask.dtype!=torch.bool
            or not(1<=c.lag_steps<x.shape[1])
            or not(0<=c.source_channel<x.shape[2])
            or not(0<=c.readout_channel<x.shape[2])
            or not(1<=last_steps<=x.shape[1])):
            raise ValueError("Incompatible measured temporal input")
        i,j,lag=c.source_channel,c.readout_channel,c.lag_steps
        observed=mask[:,:-lag,i]&mask[:,lag:,j]
        safe=torch.where(mask,x,torch.zeros_like(x))
        first=safe[:,:-lag,i]
        second=safe[:,lag:,j]
        if c.family==OP_EVEN:
            first=first.square()
            second=second.square()
        aa=(first-c.source_center)/c.source_scale
        bb=c.readout_sign*(second-c.readout_center)/c.readout_scale
        weight=F.softmax(self.mix_logits,dim=0)
        temperature=self.log_temperature.exp()
        value=torch.sigmoid(temperature*(weight[0]*aa+weight[1]*bb))
        trajectory=torch.full((len(x),x.shape[1]),.5,dtype=torch.float64,device=x.device)
        trajectory[:,lag:]=torch.where(observed,value,.5)
        available=torch.zeros((len(x),x.shape[1]),dtype=torch.bool,device=x.device)
        available[:,lag:]=observed
        recent=available[:,-last_steps:]
        count=recent.sum(dim=1)
        support=count.to(torch.float64)/last_steps
        centered=(trajectory[:,-last_steps:]-.5).sum(dim=1)/count.clamp(min=1)
        q=torch.clamp(.5+support*centered,0.,1.)
        return q,support,trajectory,available

    def forward(self,base_logits,x,mask):
        q,support,trajectory,available=self.memberships(x,mask)
        used=support>=.50
        signed=torch.where(used,self.gamma*(q-.5),torch.zeros_like(q))
        return base_logits+signed,q,support,used,trajectory,available


def _tensor_inputs(values,mask):
    x,m=_check(values,mask)
    return (torch.as_tensor(np.where(m,x,0.),dtype=torch.float64),
            torch.as_tensor(m,dtype=torch.bool))


def fit_joint_operator_v55(candidate,train_x,train_mask,base_logits,train_y,
                           *,seed=20265505):
    x,m=_tensor_inputs(train_x,train_mask)
    n=len(x)
    yy=np.asarray(train_y,dtype=np.float64).reshape(-1)
    base=np.asarray(base_logits,dtype=np.float64).reshape(-1)
    if (n<300 or len(yy)!=n or len(base)!=n or
        not np.isin(yy,[0.,1.]).all() or not np.isfinite(base).all()):
        raise ValueError("Independent TRAIN Y and K2 logits required")
    original=infer_operator_concept_v54(candidate,train_x,train_mask)
    original_q=torch.as_tensor(original["summary"],dtype=torch.float64)
    gamma0=float(np.clip(train_source_coefficient(base,original["summary"],yy),
                         -MAX_ABS_GAMMA,MAX_ABS_GAMMA))
    model=ConstrainedMeasuredOperatorV55(candidate,gamma0)
    with torch.no_grad():
        before=model.memberships(x,m)[0]
        if not torch.allclose(before,original_q,atol=1e-12,rtol=0):
            raise AssertionError("Uncalibrated V55 does not reproduce frozen V54 q")
    optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    y=torch.as_tensor(yy,dtype=torch.float64)
    baseline=torch.as_tensor(base,dtype=torch.float64)
    trace=[]
    for _ in range(TRAIN_STEPS):
        z,q,support,used,_,_=model(baseline,x,m)
        weight=F.softmax(model.mix_logits,dim=0)
        task=F.binary_cross_entropy_with_logits(z,y)
        reg=(Q_ANCHOR_WEIGHT*((q-original_q)**2).mean()
             +MIX_WEIGHT*((weight-.5)**2).sum()
             +TEMPERATURE_WEIGHT*model.log_temperature.square()
             +GAMMA_WEIGHT*model.gamma.square())
        loss=task+reg
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite constrained concept objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.log_temperature.clamp_(min=float(np.log(MIN_TEMPERATURE)),
                                         max=float(np.log(MAX_TEMPERATURE)))
            model.gamma.clamp_(min=-MAX_ABS_GAMMA,max=MAX_ABS_GAMMA)
        trace.append(float(loss.detach()))
    with torch.no_grad():
        w=F.softmax(model.mix_logits,dim=0).numpy()
        temperature=float(model.log_temperature.exp())
        gamma=float(model.gamma)
    return JointMeasuredConceptFitV55(
        candidate=candidate,source_weight=float(w[0]),
        readout_weight=float(w[1]),temperature=temperature,gamma=gamma,
        initial_gamma=gamma0,n_train=n,
        train_objective_first=trace[0],train_objective_last=trace[-1])


def infer_joint_operator_v55(fitted:JointMeasuredConceptFitV55,values,mask):
    x,m=_tensor_inputs(values,mask)
    if fitted.n_train<300 or not(0<fitted.source_weight<1 and 0<fitted.readout_weight<1):
        raise ValueError("Invalid TRAIN-only joint operator fit")
    if abs(fitted.source_weight+fitted.readout_weight-1.)>1e-8:
        raise ValueError("Mixture is not convex")
    if not(MIN_TEMPERATURE<=fitted.temperature<=MAX_TEMPERATURE):
        raise ValueError("Out-of-scope concept temperature")
    model=ConstrainedMeasuredOperatorV55(fitted.candidate,gamma0=fitted.gamma)
    with torch.no_grad():
        model.mix_logits.copy_(torch.as_tensor(np.log([
            fitted.source_weight,fitted.readout_weight]),dtype=torch.float64))
        model.log_temperature.fill_(float(np.log(fitted.temperature)))
        q,support,trajectory,available=model.memberships(x,m)
        qq=q.numpy();ss=support.numpy()
        return {"summary":qq,"support":ss,
                "trajectory":trajectory.numpy(),
                "available":available.numpy(),
                "gamma":fitted.gamma,"operator_family":fitted.candidate.family,
                "statistical_not_physical":True}
