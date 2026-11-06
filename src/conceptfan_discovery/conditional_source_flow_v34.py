"""Small conditional rectified-flow source amplitude model (V34).

This is CONDITIONAL FLOW MATCHING, not a general time-series diffusion model:
it learns a 1-D conditional distribution of the TRAIN measured WLS source
amplitude from artificially weakened TRAIN observations. The "target" is a
full-OBSERVED TRAIN WLS estimate, NEVER latent synthetic oracle amplitude.
Samples are approximations, NOT certified Bayesian posterior draws.

Base: Gaussian epsilon.  x_t=(1-t)*epsilon+t*a_full.
Target velocity: a_full-epsilon.  MSE over t~Uniform[0,1].
The input context contains only measured amplitude, information and
observed early/late time fractions; labels Y never appear in the fit.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from conceptfan_discovery.masked_temporal_template import (
    MaskedRank1Template,project_masked_template,
)
from conceptfan_discovery.auto_measured_concept_proposals import MeasuredGroup


class SourceAmplitudeFlow(nn.Module):
    def __init__(self,hidden_dim:int=40):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(6,hidden_dim),nn.SiLU(),
            nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),
            nn.Linear(hidden_dim,1),
        )

    def forward(self,z_t:torch.Tensor,t:torch.Tensor,context:torch.Tensor):
        if context.ndim!=2 or context.shape[1]!=4:
            raise ValueError("Flow conditional context must contain four measured features")
        return self.net(torch.cat((z_t[:,None],t[:,None],context),dim=1))[:,0]


@dataclass
class TrainedSourceFlow:
    model: SourceAmplitudeFlow
    context_mean: np.ndarray
    context_std: np.ndarray
    target_mean: float
    target_std: float
    train_reconstruction_mse_first: float
    train_reconstruction_mse_last: float
    heldout_train_nominal_80pct_coverage: float
    heldout_train_width_multiplier: float
    n_fit_samples: int
    n_calibration_samples: int
    training_used_outcomes: bool=False
    training_used_latent_source_truth: bool=False
    final_flow_is_a_bayesian_posterior: bool=False


def observed_flow_context(
    template:MaskedRank1Template,group:MeasuredGroup,
    x:np.ndarray,m:np.ndarray,
)->tuple[np.ndarray,np.ndarray]:
    z=project_masked_template(template,group,x,m)
    indices=group.sensor_indices
    seen=np.asarray(m,dtype=bool)[:,:,indices].any(axis=-1)
    early=seen[:,:8].mean(axis=1)
    late=seen[:,8:].mean(axis=1)
    c=np.column_stack((
        z["amplitude_estimate"],
        z["observed_template_information_fraction"],
        early,late,
    )).astype(np.float32)
    return c,z["observed_template_information_fraction"]


def _sample_normalized(
    model:SourceAmplitudeFlow,context:np.ndarray,
    *,seed:int,n_samples:int=48,steps:int=12,
)->np.ndarray:
    if not 8<=n_samples<=512 or not 6<=steps<=64:
        raise ValueError("Flow empirical sample count/solver steps out of preregistered range")
    c=np.asarray(context,dtype=np.float32)
    if c.ndim!=2 or c.shape[1]!=4:
        raise ValueError("Conditional context dimension mismatch")
    n=len(c)
    rng=np.random.default_rng(seed)
    z=torch.from_numpy(rng.normal(size=(n*n_samples,)).astype(np.float32))
    cc=torch.from_numpy(np.repeat(c,n_samples,axis=0))
    model.eval()
    with torch.no_grad():
        for j in range(steps):
            t=torch.full_like(z,(j+.5)/steps)
            z=z+model(z,t,cc)/steps
    return z.reshape(n,n_samples).numpy()


def flow_amplitude_samples(
    trained:TrainedSourceFlow,
    template:MaskedRank1Template,group:MeasuredGroup,
    x:np.ndarray,m:np.ndarray,
    *,seed:int,n_samples:int=48,
)->dict:
    context,observed_info=observed_flow_context(template,group,x,m)
    normalized=(context-trained.context_mean)/trained.context_std
    samples=_sample_normalized(
        trained.model,normalized,seed=seed,n_samples=n_samples)
    amp=trained.target_mean+trained.target_std*samples
    q=np.quantile(amp,[.1,.5,.9],axis=1)
    lo=q[0]
    hi=q[2]
    midpoint=amp.mean(axis=1)
    width=0.5*(hi-lo)*trained.heldout_train_width_multiplier
    low=midpoint-width
    high=midpoint+width
    # Calibration is only with synthetic masking of TRAIN proxy targets.
    # Never claim nominal coverage on new TEST physical latent factors.
    measured=project_masked_template(template,group,x,m)
    eligible=(observed_info>=.50)
    return {
        "amplitude_mean":np.where(eligible,midpoint,0.).astype(np.float32),
        "amplitude_sd":np.where(eligible,amp.std(axis=1),0.).astype(np.float32),
        "amplitude_interval_80pct_proxy_calibrated":np.column_stack((
            np.where(eligible,low,0.),
            np.where(eligible,high,0.),
        )).astype(np.float32),
        "source_observed_information":observed_info.astype(np.float32),
        "eligible":eligible,
        "empirical_samples":amp.astype(np.float32),
        "actual_observed_readout_amplitude":measured["amplitude_estimate"].astype(np.float32),
        "samples_are_generated_not_observed":True,
        "physical_source_identity_verified":False,
    }


def train_conditional_measured_source_flow(
    template:MaskedRank1Template,group:MeasuredGroup,
    train_x:np.ndarray,train_mask:np.ndarray,
    *,seed:int=20263401,steps:int=220,
)->TrainedSourceFlow:
    """Self-supervised additional TRAIN masking, no outcomes / latent oracle."""
    x=np.asarray(train_x,dtype=np.float32)
    m=np.asarray(train_mask,dtype=bool)
    if x.ndim!=3 or x.shape!=m.shape or len(x)<240 or steps not in (20,220):
        raise ValueError("Needs >=240 TRAIN episodes and registered optimizer steps")
    rng=np.random.default_rng(seed)
    # TRAIN target estimated with ALL actually measured cells (not the
    # simulator oracle); if poor original support then omit example.
    target=project_masked_template(template,group,x,m)
    qualify=target["observed_template_information_fraction"]>=.70
    artificial=m & (rng.random(m.shape)>.33)
    context,condinfo=observed_flow_context(template,group,x,artificial)
    qualify=qualify & (condinfo>=.30)
    if int(qualify.sum())<160:
        raise ValueError("Insufficient measurable TRAIN source pairs")
    target_value=target["amplitude_estimate"][qualify].astype(np.float32)
    context=context[qualify]
    n=len(target_value)
    permutation=rng.permutation(n)
    cut=int(.8*n)
    fit_ids,calib_ids=permutation[:cut],permutation[cut:]
    c_mean=context[fit_ids].mean(axis=0)
    c_std=np.maximum(context[fit_ids].std(axis=0),.05)
    y_mean=float(target_value[fit_ids].mean())
    y_std=float(max(target_value[fit_ids].std(),.20))
    features=(context-c_mean)/c_std
    targets=(target_value-y_mean)/y_std
    torch.manual_seed(seed)
    model=SourceAmplitudeFlow()
    optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    X=torch.from_numpy(features[fit_ids].astype(np.float32))
    Y=torch.from_numpy(targets[fit_ids].astype(np.float32))
    generator=torch.Generator().manual_seed(seed+17)
    trace=[]
    model.train()
    for k in range(steps):
        index=torch.randint(len(fit_ids),(min(256,len(fit_ids)),),generator=generator)
        target_batch=Y[index]
        ctx=X[index]
        t=torch.rand(len(index),generator=generator)
        eps=torch.randn(len(index),generator=generator)
        zt=(1-t)*eps+t*target_batch
        velocity=target_batch-eps
        prediction=model(zt,t,ctx)
        loss=F.mse_loss(prediction,velocity)
        if not torch.isfinite(loss):
            raise FloatingPointError("Flow training became nonfinite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
        optimizer.step()
        if k==0 or k==steps-1:
            trace.append(float(loss.detach()))
    # Fit interval scale from a TRAIN HELDOUT pseudo-target only, not
    # official VALID or TEST labels, nor actual latent physiological A.
    calibration_samples=_sample_normalized(
        model,features[calib_ids],seed=seed+31,n_samples=48)
    calib_amp=y_mean+y_std*calibration_samples
    center=calib_amp.mean(axis=1)
    halfwidth=np.maximum(.02,
        .5*(np.quantile(calib_amp,.9,axis=1)-np.quantile(calib_amp,.1,axis=1)))
    ratios=np.abs(target_value[calib_ids]-center)/halfwidth
    factor=float(np.clip(np.quantile(ratios,.80),.40,4.))
    empirical=float(np.mean(ratios<=factor))
    model.eval()
    return TrainedSourceFlow(
        model=model,
        context_mean=c_mean.astype(np.float32),
        context_std=c_std.astype(np.float32),
        target_mean=y_mean,target_std=y_std,
        train_reconstruction_mse_first=trace[0],
        train_reconstruction_mse_last=trace[-1],
        heldout_train_nominal_80pct_coverage=empirical,
        heldout_train_width_multiplier=factor,
        n_fit_samples=len(fit_ids),
        n_calibration_samples=len(calib_ids),
    )
