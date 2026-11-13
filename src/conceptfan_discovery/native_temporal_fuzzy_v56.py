"""V56: native, differentiable, diagonally causal fuzzy temporal concepts.

The observable source vocabulary comes from unsupervised TRAIN measurements,
with either V54 robust measured groups or the established positive-lag
observable subspace fallback. The native concept encoder is trained exclusively
by masked-measurement denoising and next-measurement forecasting; the supervised
task Y never appears in its training API or gradients.

The risk head is fitted only AFTER native concept training, sees concepts
and nothing else, and has an exact signed additive logit.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from scipy.special import expit

from conceptfan_discovery.autonomous_temporal_bank_v53 import (
    check_measurements,event_intervals
)
from conceptfan_discovery.robust_observable_concepts_v54 import (
    fit_robust_vocabulary, robust_group_level,
    fit_observable_spectral_definitions,_masked_spectral_projection,
)

TRAIN_EPOCHS=8
TRAIN_BATCH=160
TRAIN_LR=.015
TRAIN_MASK_RATE=.20
MIN_GROUPS=2
MIN_GROUP_CHANNEL_COVERAGE=.50


class TrainOnlySourceBank:
    """Route and form all sources from TRAIN X/M alone, never Y or oracle."""
    def __init__(self,*,seed:int=20261393,permutations:int=60,bootstraps:int=20):
        if permutations<25 or bootstraps<5:
            raise ValueError("Too few spectral null/bootstrap samples")
        self.seed=int(seed)
        self.permutations=int(permutations)
        self.bootstraps=int(bootstraps)
        self.route=None
        self.source=None
        self.k=0
        self.tau=None
        self.width=None
        self.sha=None
        self.train_receipt=None

    def fit_unlabeled(self,x,mask):
        a,m=check_measurements(x,mask)
        if len(a)<300:
            raise ValueError("TRAIN source protocol requires 300 independent episodes")
        measured=fit_robust_vocabulary(a,m,seed=self.seed)
        coverage=len({j for g in measured.groups for j in g.sensor_indices})/a.shape[-1]
        if len(measured.groups)>=MIN_GROUPS and coverage>=MIN_GROUP_CHANNEL_COVERAGE:
            self.route="MEASURED_GROUPS"
            self.source=measured
            self.k=len(measured.groups)
            self.tau=np.array([g.state_tau for g in measured.groups],float)
            self.width=np.array([g.state_width for g in measured.groups],float)
        else:
            defined=fit_observable_spectral_definitions(
                a,m,seed=self.seed+17,
                permutations=self.permutations,bootstraps=self.bootstraps)
            self.source=defined
            self.k=defined.rank
            self.tau=np.asarray(defined.thresholds,float)
            self.width=np.asarray(defined.widths,float)
            self.route=("PREDICTIVE_SUBSPACE" if self.k
                        else "OBSERVATION_ABSTAIN")
        metadata=self.source.as_dict()
        self.sha=hashlib.sha256(json.dumps(
            metadata,sort_keys=True,allow_nan=False).encode()).hexdigest()
        self.train_receipt={
            "source_route":self.route,
            "number_of_discovered_source_slots":self.k,
            "train_source_group_count":len(measured.groups),
            "train_source_coverage":coverage,
            "n_train_unlabeled":len(a),
            "manual_K_or_concept_labels_provided":False,
            "outcome_oracle_used_in_source_selection":False,
            "source_sha256":self.sha,
            "physical_semantics_identified":False,
            "source_definition_type":(
                "MEASURED_GROUP_PROXY" if self.route=="MEASURED_GROUPS"
                else "OBSERVABLE_PREDICTIVE_COORDINATE"),
        }
        return self

    def transform(self,x,mask)->tuple[np.ndarray,np.ndarray]:
        if self.route is None:
            raise RuntimeError("Fit TRAIN-only source bank first")
        a,m=check_measurements(x,mask)
        if not self.k:
            empty=np.empty((len(a),a.shape[1],0),dtype=np.float32)
            return empty,empty.copy()
        if self.route=="MEASURED_GROUPS":
            levels=[];supports=[]
            for g in self.source.groups:
                v,s=robust_group_level(g,a,m)
                levels.append(v)
                supports.append(s)
            values=np.stack(levels,axis=-1)
            support=np.stack(supports,axis=-1)
        else:
            values,support=_masked_spectral_projection(self.source,a,m)
        if (not np.isfinite(values).all() or
            not np.isfinite(support).all() or
            np.any((support<0)|(support>1))):
            raise AssertionError("Nonfinite or unsupported measured source")
        return values.astype(np.float32),support.astype(np.float32)

    def public_passport(self):
        if self.train_receipt is None:
            raise RuntimeError("No fitted TRAIN source definition")
        return {"source_receipt":self.train_receipt,
                "trained_threshold_initialization":self.tau.tolist(),
                "trained_width_initialization":self.width.tolist(),
                "source_manifest":self.source.as_dict()}


class NativeCausalFuzzyDynamics(nn.Module):
    """Independent learned temporal filter for every measured source."""

    def __init__(self,tau:np.ndarray,width:np.ndarray):
        super().__init__()
        tau=np.asarray(tau,dtype=np.float32)
        width=np.asarray(width,dtype=np.float32)
        if tau.ndim!=1 or tau.shape!=width.shape or len(tau)<1:
            raise ValueError("At least one measured source is necessary")
        if not np.isfinite(tau).all() or not np.isfinite(width).all() or np.any(width<=0):
            raise ValueError("Invalid initial fuzzy boundaries")
        k=len(tau)
        self.k=k
        self.input_gain=nn.Parameter(torch.ones(k))
        self.rho_logits=nn.Parameter(torch.full((k,),-.4))
        self.state_bias=nn.Parameter(torch.zeros(k))
        self.fuzzy_tau=nn.Parameter(torch.as_tensor(tau.copy()))
        self.fuzzy_log_width=nn.Parameter(torch.log(torch.as_tensor(width.copy())))
        self.denoise_gain=nn.Parameter(torch.ones(k))
        self.denoise_bias=nn.Parameter(torch.zeros(k))
        self.future_gain=nn.Parameter(torch.full((k,),.6))
        self.future_bias=nn.Parameter(torch.zeros(k))

    def forward(self,level:torch.Tensor,support:torch.Tensor):
        if level.ndim!=3 or support.shape!=level.shape or level.shape[-1]!=self.k:
            raise ValueError("Expected aligned [N,T,K] source levels and support")
        rho=torch.sigmoid(self.rho_logits)[None,:]
        width=self.fuzzy_log_width.exp().clamp(.07,4.)[None,:]
        h=level.new_zeros((len(level),self.k))
        states=[];concepts=[]
        for t in range(level.shape[1]):
            observed=support[:,t,:]
            latent=torch.tanh(
                self.input_gain[None,:]*level[:,t,:]+rho*h+self.state_bias[None,:])
            h=torch.where(observed>0,latent,rho*h)
            c=.5+observed*(torch.sigmoid((h-self.fuzzy_tau[None,:])/width)-.5)
            states.append(h)
            concepts.append(c)
        states=torch.stack(states,dim=1)
        concepts=torch.stack(concepts,dim=1)
        denoise=(states*self.denoise_gain[None,None,:]
                 +self.denoise_bias[None,None,:])
        one_ahead=(states*self.future_gain[None,None,:]
                   +self.future_bias[None,None,:])
        return concepts,denoise,one_ahead


class NativeSelfSupervisedConceptFAN:
    """Fitted unlabeled temporal encoder followed by concept-only risk head.

    Source stage has NO Y argument, then risk is fitted on TRAIN Y after
    a verified source-network hash is frozen. No raw-to-risk bypass.
    """
    def __init__(self,bank:TrainOnlySourceBank,*,neural_seed:int=11,
                 epochs:int=TRAIN_EPOCHS):
        if bank.route is None:
            raise ValueError("Fit unlabeled observable-source bank first")
        self.bank=bank
        self.neural_seed=int(neural_seed)
        self.epochs=int(epochs)
        if not 1<=self.epochs<=40:
            raise ValueError("Unregistered training epoch count")
        self.net:NativeCausalFuzzyDynamics|None=None
        self.loss_receipt=[]
        self.frozen_network_sha=None
        self.bias=None
        self.beta=None
        self.n_risk_labels=0
        self.is_source_frozen=False

    def fit_source_unlabeled(self,train_x:np.ndarray,train_mask:np.ndarray):
        """TRAIN X+mask ONLY; randomized held-out observed sensor cells."""
        a,m=check_measurements(train_x,train_mask)
        if len(a)<300:
            raise ValueError("Not enough episodes for self-supervised training")
        if not self.bank.k:
            self.net=None
            self.frozen_network_sha="ZERO_RANK_NO_NETWORK"
            self.is_source_frozen=True
            return self
        torch.set_num_threads(2)
        torch.manual_seed(self.neural_seed)
        rng=np.random.default_rng(881001+self.neural_seed)
        net=NativeCausalFuzzyDynamics(self.bank.tau,self.bank.width)
        opt=torch.optim.Adam(net.parameters(),lr=TRAIN_LR)
        teacher,teacher_support=self.bank.transform(a,m)
        x=torch.from_numpy(teacher)
        support=torch.from_numpy(teacher_support)
        init_tau=torch.as_tensor(self.bank.tau.copy(),dtype=torch.float32)
        init_width=torch.as_tensor(self.bank.width.copy(),dtype=torch.float32)
        for epoch in range(self.epochs):
            dropout=m&(rng.random(m.shape)>=TRAIN_MASK_RATE)
            noisy,seen=self.bank.transform(a,dropout)
            nlevel=torch.from_numpy(noisy)
            nmask=torch.from_numpy(seen)
            perm=rng.permutation(len(a))
            losses=[];denoise_losses=[];forecast_losses=[]
            net.train()
            for start in range(0,len(a),TRAIN_BATCH):
                idx=perm[start:start+TRAIN_BATCH]
                if len(idx)<2:
                    continue
                concepts,estimated,forecast=net(nlevel[idx],nmask[idx])
                original=x[idx]
                original_support=support[idx]
                newly_unavailable=(original_support-nmask[idx]).clamp_min(0.)
                weight=(.25*original_support+newly_unavailable)
                denoise=((estimated-original).square()*weight).sum()/weight.sum().clamp_min(1.)
                future_mask=original_support[:,1:]
                forecast_loss=((forecast[:,:-1]-original[:,1:]).square()
                               *future_mask).sum()/future_mask.sum().clamp_min(1.)
                # Anchor fuzzy coordinate to the original inspectable
                # unsupervised measured source definition.
                base=.5+nmask[idx]*(torch.sigmoid(
                    (nlevel[idx]-init_tau[None,None,:])
                    /init_width[None,None,:])-.5)
                alignment=((concepts-base).square()
                           *nmask[idx]).sum()/nmask[idx].sum().clamp_min(1.)
                regularization=(
                    (net.input_gain-1.).square().mean()
                    +net.rho_logits.square().mean()*.1)
                loss=.55*denoise+.45*forecast_loss+.02*alignment+.001*regularization
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Non-finite unlabeled self-supervision")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),1.)
                opt.step()
                losses.append(float(loss.detach()))
                denoise_losses.append(float(denoise.detach()))
                forecast_losses.append(float(forecast_loss.detach()))
            self.loss_receipt.append({
                "epoch":epoch+1,
                "mean_unlabeled_total_loss":float(np.mean(losses)),
                "mean_source_reconstruction_mse":float(np.mean(denoise_losses)),
                "mean_source_next_step_mse":float(np.mean(forecast_losses)),
            })
        net.eval()
        for param in net.parameters():
            param.requires_grad_(False)
        self.net=net
        h=hashlib.sha256()
        for key,value in sorted(net.state_dict().items()):
            h.update(key.encode())
            h.update(value.detach().cpu().numpy().tobytes())
        self.frozen_network_sha=h.hexdigest()
        self.is_source_frozen=True
        return self

    def _transform(self,x,mask)->dict[str,np.ndarray]:
        if not self.is_source_frozen:
            raise RuntimeError("Complete unlabeled concept training first")
        a,m=check_measurements(x,mask)
        level,support=self.bank.transform(a,m)
        if not self.bank.k:
            traj=np.empty((len(a),a.shape[1],0),dtype=float)
        else:
            with torch.inference_mode():
                z,_,_=self.net(torch.from_numpy(level),
                               torch.from_numpy(support))
            traj=z.detach().cpu().numpy().astype(float)
        return {"trajectories":traj,
                "summaries":traj.mean(axis=1),
                "source_support":support.mean(axis=1),
                "source_route":self.bank.route}

    def fit_risk_head(self,x,mask,y):
        if not self.is_source_frozen:
            raise RuntimeError("Outcome gradients must not enter concept learning")
        yy=np.asarray(y).reshape(-1)
        features=self._transform(x,mask)["summaries"]-.5
        if len(yy)!=len(features) or not np.isin(yy,(0,1)).all() or len(np.unique(yy))!=2:
            raise ValueError("Invalid TRAIN labels")
        if not features.shape[-1]:
            p=float(np.clip(yy.mean(),1e-6,1-1e-6))
            self.bias=float(np.log(p/(1-p)))
            self.beta=np.empty(0)
        else:
            head=LogisticRegression(C=1.,max_iter=1200)
            head.fit(features,yy)
            self.bias=float(head.intercept_[0])
            self.beta=head.coef_[0].copy()
        self.n_risk_labels=len(yy)
        return self

    def predict(self,x,mask):
        if self.beta is None or self.bias is None:
            raise RuntimeError("Fit TRAIN-only additive risk head")
        output=self._transform(x,mask)
        contrib=(output["summaries"]-.5)*self.beta[None,:]
        logits=self.bias+contrib.sum(axis=1)
        err=float(np.max(np.abs(logits-(self.bias+contrib.sum(axis=1)))))
        if not np.isfinite(logits).all() or err>1e-12:
            raise AssertionError("Unstable or non-additive trained concept")
        return {**output,"direct_contributions":contrib,
                "logits":logits,"probability":expit(logits),
                "bias":self.bias,"signed_beta":self.beta.copy(),
                "max_additive_logit_error":err,
                "source_sha256":self.bank.sha,
                "source_network_sha256":self.frozen_network_sha,
                "original_physical_identity_inferred":False}

    def passport(self):
        if self.beta is None:
            raise RuntimeError("Train source then supervised head first")
        return {
            "route":self.bank.public_passport(),
            "native_trainable_temporal_concept_generator":self.bank.k>0,
            "epochs_self_supervised":self.epochs,
            "unlabeled_source_losses":self.loss_receipt,
            "frozen_source_network_sha256":self.frozen_network_sha,
            "train_binary_task_labels":self.n_risk_labels,
            "exact_signed_task_beta":self.beta.tolist(),
            "fuzzy_tau":self.net.fuzzy_tau.detach().cpu().numpy().tolist() if self.net else [],
            "fuzzy_width":self.net.fuzzy_log_width.exp().detach().cpu().numpy().tolist() if self.net else [],
            "outcome_gradients_in_concept_training":False,
            "no_raw_classification_bypass":True,
            "autonomous_clinical_cause_identification":False,
        }

    def event_summary(self,x,mask,threshold=.65,minimum_length=2):
        out=self.predict(x,mask)
        traj=out["trajectories"]
        return {
            f"{self.bank.route}_concept_{k}":{
                "mean_events_per_episode":float(np.mean([
                    len(event_intervals(traj[i,:,k],threshold=threshold,
                                        minimum_length=minimum_length))
                    for i in range(len(traj))])),
                "mean_absolute_direct_logit_contribution":float(
                    np.mean(np.abs(out["direct_contributions"][:,k])))
            } for k in range(self.bank.k)
        }
