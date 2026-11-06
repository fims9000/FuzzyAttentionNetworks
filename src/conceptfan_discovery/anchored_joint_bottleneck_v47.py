"""V47 strict joint concept-only predictor with measured-proxy anchoring.

Real Temporal ConceptFAN encoder is unfrozen, while ALL risk information
flows through two known measured concept summaries plus one measured
outcome-blind V44 source q. Not clinical or causal concept certification.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class JointConceptBottleneckOutputV47:
    logits: torch.Tensor
    concept_trajectories: torch.Tensor
    known_concepts: torch.Tensor
    signed_contributions: torch.Tensor
    accepted_source: torch.Tensor
    source_score: torch.Tensor

    @property
    def probs(self):
        return torch.sigmoid(self.logits)


class AnchoredJointConceptBottleneckV47(nn.Module):
    """Trainable native Transformer, no direct latent-to-risk/raw-to-risk shortcut."""

    def __init__(self, pretrained_native, initial_weights, initial_bias,
                 initial_gamma=0.):
        super().__init__()
        self.concept_encoder=copy.deepcopy(pretrained_native)
        self.known_weight=nn.Parameter(
            torch.as_tensor(initial_weights,dtype=torch.float32).reshape(2).clone())
        self.bias=nn.Parameter(torch.tensor(float(initial_bias),dtype=torch.float32))
        self.gamma=nn.Parameter(torch.tensor(float(initial_gamma),dtype=torch.float32))

    def forward(self, observed_known, known_mask, source_q, source_support):
        x=torch.as_tensor(observed_known,dtype=torch.float32)
        m=torch.as_tensor(known_mask,dtype=torch.bool)
        q=torch.as_tensor(source_q,dtype=torch.float32).reshape(-1)
        support=torch.as_tensor(source_support,dtype=torch.bool).reshape(-1)
        if (x.ndim!=3 or x.shape[1:]!=(16,4) or x.shape!=m.shape
                or len(q)!=len(x) or len(support)!=len(x)
                or not bool(torch.isfinite(x[m]).all())
                or not bool(torch.isfinite(q).all())
                or bool((q<0).any()) or bool((q>1).any())):
            raise ValueError("Only masked [B,16,4] known readings and [B] measured q permitted")
        masked=torch.where(m,x,torch.zeros_like(x))
        features=torch.cat((masked,m.to(x.dtype)),dim=-1)
        native=self.concept_encoder(features)
        concept=native.concept_summaries
        if concept.shape!=(len(x),2) or native.concept_trajectories.shape!=(len(x),16,2):
            raise AssertionError("Native concept encoder must expose exactly two K2 concepts")
        k2=(concept-.5)*self.known_weight[None,:]
        source=torch.where(support,self.gamma*(q-.5),torch.zeros_like(q))
        contributions=torch.cat((k2,source[:,None]),dim=-1)
        z=self.bias+contributions.sum(dim=-1)
        if not bool(torch.isfinite(z).all()):
            raise FloatingPointError("Nonfinite signed-concept bottleneck")
        return JointConceptBottleneckOutputV47(
            logits=z,concept_trajectories=native.concept_trajectories,
            known_concepts=concept,signed_contributions=contributions,
            accepted_source=support,source_score=q,
        )


def known_measured_proxy_v47(x,m):
    """Label-free targets from observed sensor pairs (0,1) and (2,3)."""
    values=torch.as_tensor(x,dtype=torch.float32)
    mask=torch.as_tensor(m,dtype=torch.bool)
    if (values.ndim!=3 or values.shape[1:]!=(16,4) or mask.shape!=values.shape
            or not bool(torch.isfinite(values[mask]).all())):
        raise ValueError("Incorrect measured K2 proxy tensor")
    targets=[];exists=[]
    for a in (0,2):
        seen=mask[:,:,a:a+2]
        data=torch.where(seen,values[:,:,a:a+2],torch.zeros_like(values[:,:,a:a+2]))
        count=seen.sum(dim=-1)
        targets.append(torch.sigmoid(data.sum(dim=-1)/count.clamp(min=1)))
        exists.append(count>0)
    return torch.stack(targets,dim=-1),torch.stack(exists,dim=-1)


def masked_proxy_loss_v47(output,known_input,known_mask):
    target,information=known_measured_proxy_v47(known_input,known_mask)
    valid=information.to(output.concept_trajectories.dtype)
    return (((output.concept_trajectories-target)**2)*valid).sum()/valid.sum().clamp_min(1)


def run_joint_epochs_v47(model,known_train,mask_train,source_q,source_support,y_train,
                         *,seed:int,consistency_weight:float,epochs:int=8,
                         batch_size:int=128,proxy_weight:float=5.):
    """TRAIN-only end-to-end BCE + observed proxy + optional missing-mask invariance."""
    if (consistency_weight not in (0.,.25) or epochs<1 or epochs>12
            or batch_size!=128 or proxy_weight!=5.):
        raise ValueError("Frozen V47 training hyperparameters changed")
    x=torch.from_numpy(np.asarray(known_train,dtype=np.float32))
    m=torch.from_numpy(np.asarray(mask_train,dtype=bool))
    q=torch.from_numpy(np.asarray(source_q,dtype=np.float32))
    support=torch.from_numpy(np.asarray(source_support,dtype=bool))
    y=torch.from_numpy(np.asarray(y_train,dtype=np.float32))
    if (len(x)!=len(m) or len(x)!=len(q) or len(x)!=len(y)
            or len(x)!=len(support) or len(x)<300
            or not bool(torch.isfinite(y).all())
            or not bool(((y==0)|(y==1)).all())):
        raise ValueError("TRAIN labels and independent measured inputs are invalid")
    # Do not optimize the native unused decision head; only K2 concept path
    # plus the three explicit signed contributions can receive task gradients.
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=.001,weight_decay=.001)
    rng=np.random.default_rng(20264701+seed)
    model.train()
    objective_trace=[];proxy_trace=[];cons_trace=[]
    for epoch in range(epochs):
        order=rng.permutation(len(x))
        batch_losses=[];batch_proxy=[];batch_cons=[]
        for offset in range(0,len(order),batch_size):
            ix=order[offset:offset+batch_size]
            out=model(x[ix],m[ix],q[ix],support[ix])
            task=F.binary_cross_entropy_with_logits(out.logits,y[ix])
            proxy=masked_proxy_loss_v47(out,x[ix],m[ix])
            if consistency_weight>0:
                random_keep=torch.from_numpy(rng.random(m[ix].shape)>=.10)
                reduced=m[ix]&random_keep
                masked_out=model(x[ix],reduced,q[ix],support[ix])
                # Invariance is anchored to the same *measured episode*;
                # consistency alone is not a physical source certificate.
                consist=(out.known_concepts-masked_out.known_concepts).square().mean()
            else:
                consist=task.new_zeros(())
            objective=task+proxy_weight*proxy+consistency_weight*consist
            if not torch.isfinite(objective):
                raise FloatingPointError("Nonfinite joint measured concept objective")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(params,1.)
            optimizer.step()
            batch_losses.append(float(objective.detach()))
            batch_proxy.append(float(proxy.detach()))
            batch_cons.append(float(consist.detach()))
        objective_trace.append(float(np.mean(batch_losses)))
        proxy_trace.append(float(np.mean(batch_proxy)))
        cons_trace.append(float(np.mean(batch_cons)))
    return {
        "first_epoch_objective":objective_trace[0],
        "last_epoch_objective":objective_trace[-1],
        "last_epoch_proxy_loss":proxy_trace[-1],
        "last_epoch_mask_consistency_loss":cons_trace[-1],
        "n_joint_epochs":epochs,
        "consistency_weight":float(consistency_weight),
        "same_original_TRAIN_Y_for_encoder_and_head":True,
        "new_source_q_proposed_without_outcomes":True,
        "risk_has_no_raw_bypass":True,
        "physical_concept_identity_certified":False,
    }


@torch.no_grad()
def predict_joint_v47(model,values,observed,source_q,source_support):
    model.eval()
    x=np.asarray(values,dtype=np.float32)
    m=np.asarray(observed,dtype=bool)
    q=np.asarray(source_q,dtype=np.float32)
    sup=np.asarray(source_support,dtype=bool)
    pieces=[];summaries=[]
    max_error=0.
    for start in range(0,len(x),256):
        t=model(torch.from_numpy(x[start:start+256]),
                torch.from_numpy(m[start:start+256]),
                torch.from_numpy(q[start:start+256]),
                torch.from_numpy(sup[start:start+256]))
        err=(t.logits-(model.bias+t.signed_contributions.sum(dim=1))).abs().max().item()
        max_error=max(max_error,float(err))
        pieces.append(t.logits.numpy().astype(np.float64))
        summaries.append(t.known_concepts.numpy().astype(np.float64))
    if max_error>1e-6:
        raise AssertionError("Non-additive joint risk: direct latent bypass")
    return {"logits":np.concatenate(pieces),
            "known_concepts":np.concatenate(summaries),
            "max_additivity_error":max_error}


def safe_joint_logits_v47(frozen_k2_logits,joint_logits,source_support,*,approved):
    k=np.asarray(frozen_k2_logits,dtype=np.float64).reshape(-1)
    z=np.asarray(joint_logits,dtype=np.float64).reshape(-1)
    support=np.asarray(source_support,dtype=bool).reshape(-1)
    if k.shape!=z.shape or k.shape!=support.shape or not np.isfinite(k).all() or not np.isfinite(z).all():
        raise ValueError("Misaligned original fallback and proposed joint model")
    used=support & bool(approved)
    result=np.where(used,z,k)
    if not np.array_equal(result[~used],k[~used]):
        raise AssertionError("Unsupported source must return exact frozen K2")
    return result,used
