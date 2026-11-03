"""Independent MAGNETS-style mask-and-aggregate BINARY adaptation for V12.

Cites: Forest, Wei & Fink (Data Mining and Knowledge Discovery, 2026),
"When, how long and how much? Interpretable neural networks for time
series regression by learning to mask and aggregate".
https://doi.org/10.1007/s10618-026-01267-y

This is an INDEPENDENT reimplementation inspired by their published
mask -> observed aggregation -> linear bottleneck -> additive decision
scheme. The authors' official reference code is for REGRESSION, with
Gumbel masks, and has NOT been used here. DO NOT claim reproduction
of their official method or compare numbers to original TSER papers.

This classifier is a separately trained concept model, NOT a post-hoc
extractor. No 36-48h readings, oracle factors, or hidden-to-logit bypass.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


@dataclass
class MaskAggregateResult:
    logits:torch.Tensor
    contributions:torch.Tensor
    concepts:torch.Tensor
    masks:torch.Tensor

    def decomposition_error(self,bias:torch.Tensor)->torch.Tensor:
        return (self.logits-(self.contributions.sum(dim=-1)+bias)).abs().max()


class MaskAggregateBinaryConcepts(nn.Module):
    """Train-time learned channel-time mask and exact additive logit.

    K=source TRAIN-computational rank for fair budget, M=4 trainable masks.
    No concept labels; learned aggregators need an independent semantic audit.
    """
    def __init__(self,n_channels:int,n_concepts:int,history_hours:int=36,
                 n_masks:int=4,hidden:int=32,temperature:float=.7):
        super().__init__()
        if not (n_channels>=2 and n_concepts>=2 and history_hours>=8
                and 1<=n_masks<=8 and hidden>=4 and 0<temperature<=1.):
            raise ValueError("Invalid MAGNETS-style binary architecture capacity")
        self.n_channels=n_channels
        self.n_concepts=n_concepts
        self.history_hours=history_hours
        self.n_masks=n_masks
        self.temperature=temperature
        self.masker=nn.Sequential(
            nn.Conv1d(3*n_channels,hidden,kernel_size=5,padding=2),
            nn.GELU(),
            nn.Conv1d(hidden,hidden,kernel_size=5,padding=2),
            nn.GELU(),
            nn.Conv1d(hidden,n_masks*n_channels,kernel_size=1),
        )
        self.bottleneck=nn.Linear(n_masks*n_channels,n_concepts,bias=False)
        self.decision=nn.Parameter(torch.randn(n_concepts)*.05)
        self.decision_bias=nn.Parameter(torch.zeros(()))

    def forward(self,values:torch.Tensor,visible:torch.Tensor,
                recorded:torch.Tensor)->MaskAggregateResult:
        if (values.ndim!=3 or values.shape!=visible.shape or
            values.shape!=recorded.shape or values.shape[1:]!=(
                self.history_hours,self.n_channels
            )):
            raise ValueError("MAGNETS-style input shape must match historical [B,T,D]")
        if (not torch.isfinite(values).all() or
            not torch.isfinite(visible).all() or not torch.isfinite(recorded).all()
            or torch.any(visible<0) or torch.any(visible>1)
            or torch.any(recorded<0) or torch.any(recorded>1)
            or torch.any(visible>recorded)):
            raise ValueError("Visible observations must be subset of clinician-recorded")
        x=values*visible
        mask_input=torch.cat((x,visible,recorded),dim=-1).transpose(1,2)
        logits=self.masker(mask_input).reshape(
            len(x),self.n_masks,self.n_channels,self.history_hours
        )
        soft=torch.sigmoid(logits/self.temperature)
        hard=(soft>=.5).to(soft.dtype)
        selection=hard.detach()-soft.detach()+soft if self.training else hard
        masked=(x.transpose(1,2).unsqueeze(1)*
                visible.transpose(1,2).unsqueeze(1)*selection)
        # Fixed /T makes integrated magnitude interpretable and comparable
        # regardless of number of observed hours. Time duration is retained.
        aggregates=masked.sum(dim=-1)/self.history_hours
        concepts=self.bottleneck(aggregates.flatten(1))
        contributions=concepts*self.decision.view(1,-1)
        logit=self.decision_bias+contributions.sum(dim=-1)
        return MaskAggregateResult(logit,contributions,concepts,selection)

    def training_regularizers(self,output:MaskAggregateResult)->dict[str,torch.Tensor]:
        mask=output.masks
        sparse=mask.mean()
        smooth=(mask[:,:,:,1:]-mask[:,:,:,:-1]).abs().mean()
        weight=self.bottleneck.weight
        concept_sparse=weight.abs().mean()
        normalized=F.normalize(weight,dim=-1,eps=1e-8)
        gram=normalized@normalized.T
        redundancy=((gram-torch.diag_embed(gram.diagonal())).square()).mean()
        return {"mask_size":sparse,"mask_total_variation":smooth,
                "bottleneck_l1":concept_sparse,"concept_redundancy":redundancy}


def fit_magnets_binary_adaptation(
    train,seed:int,*,n_concepts:int,epochs:int=18,batch_size:int=128,
    learning_rate:float=.001,weight_decay:float=.0001,
    augmentation_fraction:float=.20,n_masks:int=4,
)->tuple[MaskAggregateBinaryConcepts,dict]:
    """Fit on SAME Set A train as V10/V11/V12, not dev or heldout Set B."""
    if not 0<augmentation_fraction<1:
        raise ValueError("Artificial masking fraction must be nontrivial")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    model=MaskAggregateBinaryConcepts(
        train.x.shape[-1],n_concepts,train.x.shape[1],n_masks=n_masks,
    )
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,
                                weight_decay=weight_decay)
    labels=np.asarray(train.y,dtype=np.float32)
    n_positive=float(labels.sum())
    if not 0<n_positive<len(labels):
        raise ValueError("Training task requires both labels")
    pos_weight=torch.tensor((len(labels)-n_positive)/n_positive,dtype=torch.float32)
    loader=train.loader(batch_size,True,seed=1001)
    rng=torch.Generator().manual_seed(int(seed)+71021)
    counters={"bce":0.,"mask_size":0.,"mask_tv":0.,"sparsity":0.}
    steps=0
    for epoch in range(epochs):
        model.train()
        for x,source_mask,_,_,y in loader:
            optimizer.zero_grad(set_to_none=True)
            visible=(torch.rand(x.shape,generator=rng)>=augmentation_fraction).to(x.dtype)*source_mask
            out=model(x*visible,visible,source_mask)
            task=F.binary_cross_entropy_with_logits(out.logits,y,pos_weight=pos_weight)
            r=model.training_regularizers(out)
            loss=(task+.006*r["mask_size"]+.01*r["mask_total_variation"]
                  +.01*r["bottleneck_l1"]+.01*r["concept_redundancy"])
            if not torch.isfinite(loss) or out.decomposition_error(model.decision_bias)>1e-5:
                raise FloatingPointError("Invalid independent mask-aggregate training step")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            optimizer.step()
            steps+=1
            for key,value in (("bce",task),("mask_size",r["mask_size"]),
                              ("mask_tv",r["mask_total_variation"]),
                              ("sparsity",r["bottleneck_l1"])):
                counters[key]+=float(value.detach())
    model.eval()
    return model,{
        "model":"MAGNETS_STYLE_BINARY_ADAPTATION_NOT_AUTHORS_ORIGINAL",
        "training_set":"Set A TRAIN ONLY",
        "n_train_steps":steps,
        "epochs":epochs,"n_masks":n_masks,
        "n_concepts":n_concepts,
        "trainable_parameters":sum(p.numel() for p in model.parameters()),
        "zero_outcome_holdout_access":True,
        "exact_additive_logit":True,
        "reference_paper_task_regression_not_binary":True,
        **{f"mean_train_{k}":v/max(steps,1) for k,v in counters.items()},
    }


def predict_magnets_binary(
    model:MaskAggregateBinaryConcepts,values:np.ndarray,
    visible:np.ndarray,recorded:np.ndarray,*,batch_size:int=128,
)->tuple[np.ndarray,np.ndarray]:
    x=np.asarray(values,dtype=np.float32)
    m=np.asarray(visible,dtype=np.float32)
    r=np.asarray(recorded,dtype=np.float32)
    probs,terms=[],[]
    model.eval()
    with torch.no_grad():
        for i in range(0,len(x),batch_size):
            z=model(torch.from_numpy(x[i:i+batch_size]*m[i:i+batch_size]),
                    torch.from_numpy(m[i:i+batch_size]),
                    torch.from_numpy(r[i:i+batch_size]))
            if z.decomposition_error(model.decision_bias)>1e-5:
                raise AssertionError("External concept model lost exact additivity")
            probs.append(torch.sigmoid(z.logits).numpy())
            terms.append(z.contributions.numpy())
    return np.concatenate(probs).astype(np.float64),np.concatenate(terms).astype(np.float64)


def train_only_concept_permutation(reference:np.ndarray,other:np.ndarray,*,min_abs_spearman:float=.2)->np.ndarray:
    """Hungarian permutation from TRAIN features ONLY; NEVER align using DEV/B.

    Absolute Spearman chooses the closest shape; the sign is NOT corrected
    after matching, because opposite signed risk effects are a true failure.
    """
    a=np.asarray(reference,dtype=np.float64)
    b=np.asarray(other,dtype=np.float64)
    if a.shape!=b.shape or a.ndim!=2 or len(a)<10 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Matching requires same TRAIN patient set and K")
    if not 0<min_abs_spearman<1:
        raise ValueError("Positive minimum independent TRAIN alignment required")
    k=a.shape[1]
    cost=np.ones((k,k))
    for i in range(k):
        for j in range(k):
            if np.std(a[:,i])>1e-8 and np.std(b[:,j])>1e-8:
                rho=float(spearmanr(a[:,i],b[:,j]).statistic)
                cost[i,j]=1.-abs(rho) if np.isfinite(rho) else 1.
    row,col=linear_sum_assignment(cost)
    if not np.array_equal(row,np.arange(k)):
        raise AssertionError("Unassigned concept slot in Hungarian permutation")
    result=col.astype(np.int64)
    # An arbitrary Hungarian assignment on unrelated or flat concepts is
    # NOT evidence of stable matched identities. Mark those slots ABSTAIN.
    for i in range(k):
        if cost[i,result[i]]>1.-min_abs_spearman:
            result[i]=-1
    return result
