"""Independent, reproducible external classification baselines for V10.

All models use exactly 36h of train-normalized observed values and recorded
sampling masks; no future value, hidden synthetic state, hand-made concept
label, PhysioNet calibration/test ID, or pretrained external cohort.

These are locally implemented standard architectures, NOT reproduced
published checkpoint scores. GRU-D-style includes input and hidden decay
with hours-since-observation, but does not claim exact original hyperparams.
"""
from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _validate_view(
    x: np.ndarray, visible_mask: np.ndarray, recorded_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x=np.asarray(x,dtype=np.float32)
    visible_mask=np.asarray(visible_mask,dtype=np.float32)
    recorded_mask=np.asarray(recorded_mask,dtype=np.float32)
    if x.ndim!=3 or x.shape!=visible_mask.shape or x.shape!=recorded_mask.shape:
        raise ValueError("Expect identical [patients,history_hours,channels] arrays")
    if not (np.isfinite(x).all() and np.isfinite(visible_mask).all()
            and np.isfinite(recorded_mask).all()):
        raise ValueError("Measured input arrays must be finite")
    if not (np.logical_or(visible_mask==0,visible_mask==1).all()
            and np.logical_or(recorded_mask==0,recorded_mask==1).all()):
        raise ValueError("Masks must be binary")
    if np.any(visible_mask>recorded_mask):
        raise ValueError("Artificial visibility cannot exceed original recorded mask")
    return x,visible_mask,recorded_mask


def _observed_summary(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Numeric and observation-time features, all computed from past data."""
    b,t,c=x.shape
    count=mask.sum(axis=1)
    den=np.maximum(count,1.)
    valid=np.where(mask>0,x,0.)
    mean=valid.sum(axis=1)/den
    std=np.sqrt(
        (((valid-mean[:,None,:])**2)*mask).sum(axis=1)/den
    )
    minimum=np.where(count>0,np.where(mask>0,x,np.inf).min(axis=1),0.)
    maximum=np.where(count>0,np.where(mask>0,x,-np.inf).max(axis=1),0.)
    index_first=mask.argmax(axis=1)
    index_last=t-1-mask[:,::-1,:].argmax(axis=1)
    first=np.take_along_axis(x,index_first[:,None,:],axis=1)[:,0,:]
    last=np.take_along_axis(x,index_last[:,None,:],axis=1)[:,0,:]
    first=np.where(count>0,first,0.)
    last=np.where(count>0,last,0.)
    hours=np.arange(t,dtype=np.float32)[None,:,None]
    average_hour=(mask*hours).sum(axis=1)/den
    centered=hours-average_hour[:,None,:]
    slope=(mask*centered*(valid-mean[:,None,:])).sum(axis=1)/np.maximum(
        (mask*centered**2).sum(axis=1),1.
    )
    no_obs=np.where(count>0,0.,1.)
    since=np.where(count>0,(t-1-index_last)/max(t-1,1),1.)
    recent6=mask[:,-min(6,t):,:].mean(axis=1)
    recent12=mask[:,-min(12,t):,:].mean(axis=1)
    recent_mean=(valid[:,-min(6,t):,:].sum(axis=1)
                /np.maximum(mask[:,-min(6,t):,:].sum(axis=1),1.))
    transitions=np.abs(mask[:,1:,:]-mask[:,:-1,:]).mean(axis=1)
    return np.concatenate(
        (mean,std,minimum,maximum,first,last,last-first,slope,
         recent_mean,count/t,recent6,recent12,since,no_obs,transitions),
        axis=1,
    )


def engineer_clinical_history(
    values: np.ndarray, visible_mask: np.ndarray, recorded_mask: np.ndarray,
) -> np.ndarray:
    """Same raw patient-time observables for Logistic and Gradient Boosting."""
    x,m,s=_validate_view(values,visible_mask,recorded_mask)
    # Intentionally do not use values wherever visible=0, even if a caller
    # supplies impossible sentinel values in those cells.
    # Observed-source metadata only: coverage, recent coverage, staleness,
    # never raw values that were hidden in a simulated measurement scenario.
    sampling=_observed_summary(np.zeros_like(x),s)[:,-6*x.shape[-1]:]
    return np.concatenate((_observed_summary(x*m,m),sampling),axis=1).astype(np.float32)


def class_balanced_weights(y: np.ndarray) -> np.ndarray:
    y=np.asarray(y,dtype=np.float32)
    if y.ndim!=1 or not np.isin(y,[0.,1.]).all() or min(y.sum(),len(y)-y.sum())<=0:
        raise ValueError("Binary labels with both classes required")
    positive=(len(y)-float(y.sum()))/float(y.sum())
    return np.where(y==1,positive,1.).astype(np.float64)


def fit_tabular_baseline(
    method: str,
    x: np.ndarray,
    visible_mask: np.ndarray,
    recorded_mask: np.ndarray,
    y: np.ndarray,
    *,
    augmentation_fraction: float,
    seed: int,
):
    if method not in ("logistic_history","hist_gradient_boosting"):
        raise ValueError("Unsupported tabular comparison")
    if not 0<augmentation_fraction<1:
        raise ValueError("Training augmentation fraction must lie strictly inside (0,1)")
    x,m,s=_validate_view(x,visible_mask,recorded_mask)
    rng=np.random.default_rng(int(seed)+33007)
    retained=(rng.random(m.shape)>=augmentation_fraction).astype(np.float32)*m
    # One augmented replica does not add distinct patients. This only makes
    # baseline training robust to the SAME artificial missing-observation
    # mechanism used in the V10 physical branch.
    features=np.concatenate(
        (engineer_clinical_history(x,m,s),
         engineer_clinical_history(x*retained,retained,s)),
        axis=0,
    )
    labels=np.tile(np.asarray(y,dtype=np.float32),2)
    if method=="logistic_history":
        classifier=make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.,max_iter=1200,solver="lbfgs",
                               random_state=int(seed),class_weight="balanced"),
        )
        classifier.fit(features,labels)
    else:
        classifier=HistGradientBoostingClassifier(
            learning_rate=.05,max_iter=180,max_leaf_nodes=15,
            min_samples_leaf=25,l2_regularization=1.,
            early_stopping=False,random_state=int(seed),
        )
        classifier.fit(features,labels,sample_weight=class_balanced_weights(labels))
    return classifier


def predict_tabular(
    model, values: np.ndarray, visible_mask: np.ndarray,
    recorded_mask: np.ndarray,
) -> np.ndarray:
    f=engineer_clinical_history(values,visible_mask,recorded_mask)
    return np.asarray(model.predict_proba(f)[:,1],dtype=np.float64)


def time_since_previous_observation(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim!=3:
        raise ValueError("Expected time-channel mask")
    out=torch.zeros_like(mask)
    prev=torch.zeros(mask.shape[0],mask.shape[-1],dtype=mask.dtype,device=mask.device)
    for step in range(1,mask.shape[1]):
        prev=1. + (1.-mask[:,step-1])*prev
        out[:,step]=prev
    return out


class GRUDStyleClassifier(nn.Module):
    """Trainable input/hidden time-since-observation decay (Che et al. 2018)."""

    def __init__(self,n_channels:int,hidden_size:int=64,train_mean:np.ndarray|None=None):
        super().__init__()
        if n_channels<2 or hidden_size<4:
            raise ValueError("Invalid recurrent dimensions")
        self.n_channels=n_channels
        if train_mean is None:
            train_mean=np.zeros(n_channels,dtype=np.float32)
        if np.asarray(train_mean).shape!=(n_channels,):
            raise ValueError("Train-only channel mean must match channels")
        self.register_buffer("mean",torch.tensor(train_mean,dtype=torch.float32))
        self.input_decay_raw=nn.Parameter(torch.zeros(n_channels))
        self.input_decay_bias=nn.Parameter(torch.zeros(n_channels))
        self.hidden_decay_raw=nn.Parameter(torch.randn(hidden_size,n_channels)*.01)
        self.hidden_decay_bias=nn.Parameter(torch.zeros(hidden_size))
        self.cell=nn.GRUCell(3*n_channels,hidden_size)
        self.head=nn.Linear(hidden_size,1)

    def forward(
        self, values:torch.Tensor,visible_mask:torch.Tensor,
        recorded_mask:torch.Tensor,
    )->torch.Tensor:
        if values.ndim!=3 or values.shape!=visible_mask.shape or values.shape!=recorded_mask.shape:
            raise ValueError("GRU-D arrays must match [B,T,C]")
        if values.shape[-1]!=self.n_channels:
            raise ValueError("Expected exactly trained observed channels")
        d=time_since_previous_observation(visible_mask)
        last=self.mean.view(1,-1).expand(len(values),-1)
        hidden=values.new_zeros((len(values),self.cell.hidden_size))
        for t in range(values.shape[1]):
            mask=visible_mask[:,t]
            delta=d[:,t]
            gamma_x=torch.exp(-F.relu(F.softplus(self.input_decay_raw)*delta+
                                      self.input_decay_bias))
            gamma_h=torch.exp(-F.relu(F.linear(delta,
                                              F.softplus(self.hidden_decay_raw),
                                              self.hidden_decay_bias)))
            imputed=mask*values[:,t]+(1-mask)*(gamma_x*last+(1-gamma_x)*self.mean)
            hidden=self.cell(
                torch.cat((imputed,mask,recorded_mask[:,t]),dim=-1),
                hidden*gamma_h,
            )
            last=mask*values[:,t]+(1-mask)*last
        return self.head(hidden).flatten()


class TimeSeriesTransformerClassifier(nn.Module):
    """Small non-concept Transformer, 36h mask-aware time-series benchmark."""

    def __init__(
        self,n_channels:int,history_hours:int=36,
        d_model:int=64,n_heads:int=4,n_layers:int=2,
    ):
        super().__init__()
        if d_model%n_heads or history_hours<4:
            raise ValueError("Invalid attention architecture")
        self.n_channels=n_channels
        self.history_hours=history_hours
        self.embed=nn.Linear(4*n_channels,d_model)
        self.positions=nn.Parameter(torch.zeros(1,history_hours+1,d_model))
        nn.init.normal_(self.positions,std=.02)
        self.cls=nn.Parameter(torch.randn(1,1,d_model)*.02)
        layer=nn.TransformerEncoderLayer(
            d_model=d_model,nhead=n_heads,dim_feedforward=d_model*2,
            dropout=.1,batch_first=True,
        )
        self.encoder=nn.TransformerEncoder(layer,num_layers=n_layers)
        self.head=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,1))

    def forward(
        self, values:torch.Tensor,visible_mask:torch.Tensor,
        recorded_mask:torch.Tensor,
    )->torch.Tensor:
        if values.ndim!=3 or values.shape!=visible_mask.shape or values.shape!=recorded_mask.shape:
            raise ValueError("Transformer input/mask shape mismatch")
        if values.shape[-1]!=self.n_channels or values.shape[1]>self.history_hours:
            raise ValueError("Input exceeds trained sensor/time shape")
        delta=time_since_previous_observation(visible_mask)/float(self.history_hours)
        feat=torch.cat(
            (values*visible_mask,visible_mask,recorded_mask,delta),
            dim=-1,
        )
        z=self.embed(feat)
        initial=self.cls.expand(z.shape[0],-1,-1)
        hidden=self.encoder(
            torch.cat((initial,z),dim=1)+self.positions[:,:z.shape[1]+1],
        )
        return self.head(hidden[:,0]).flatten()


def seed_research(seed:int)->None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)


def observed_train_mean(values:np.ndarray,mask:np.ndarray)->np.ndarray:
    values,mask,_=_validate_view(values,mask,mask)
    numerator=(values*mask).sum(axis=(0,1))
    denominator=np.maximum(mask.sum(axis=(0,1)),1.)
    return (numerator/denominator).astype(np.float32)


def fit_neural_baseline(
    method:str,train,*,seed:int,epochs:int=18,batch_size:int=128,
    learning_rate:float=.001,weight_decay:float=.0001,
    augmentation_fraction:float=.2,
)->tuple[nn.Module,dict]:
    if method not in ("gru_d_style","transformer_timeseries"):
        raise ValueError("Unknown temporal baseline")
    if not 0<augmentation_fraction<1:
        raise ValueError("Artificial observation dropout must be in (0,1)")
    seed_research(int(seed))
    nchan=train.x.shape[-1]
    if method=="gru_d_style":
        model=GRUDStyleClassifier(nchan,hidden_size=64,
                                  train_mean=observed_train_mean(train.x,train.mask))
    else:
        model=TimeSeriesTransformerClassifier(nchan,history_hours=train.x.shape[1])
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,
                                weight_decay=weight_decay)
    weights=class_balanced_weights(train.y)
    positive=torch.tensor(float(weights[train.y==1][0]),dtype=torch.float32)
    loader=train.loader(batch_size,True,seed=1001)
    thinning=torch.Generator().manual_seed(int(seed)+71021)
    minibatches=0
    for epoch in range(int(epochs)):
        model.train()
        for x,source_mask,_,_,y in loader:
            optimizer.zero_grad(set_to_none=True)
            retained=(torch.rand(x.shape,generator=thinning)>=augmentation_fraction).float()*source_mask
            logit=model(x*retained,retained,source_mask)
            loss=F.binary_cross_entropy_with_logits(logit,y,pos_weight=positive)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {method} loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            optimizer.step()
            minibatches+=1
    model.eval()
    return model,{
        "architecture":method,
        "epochs":int(epochs),
        "minibatches":minibatches,
        "parameters":sum(p.numel() for p in model.parameters()),
        "validation_used_for_optimization":False,
    }


def predict_neural(
    model:nn.Module,values:np.ndarray,visible_mask:np.ndarray,
    recorded_mask:np.ndarray,batch_size:int=128,
)->np.ndarray:
    x,m,s=_validate_view(values,visible_mask,recorded_mask)
    model.eval()
    probs=[]
    with torch.no_grad():
        for first in range(0,len(x),batch_size):
            last=min(len(x),first+batch_size)
            logit=model(torch.from_numpy(x[first:last]*m[first:last]),
                        torch.from_numpy(m[first:last]),
                        torch.from_numpy(s[first:last]))
            if not torch.isfinite(logit).all():
                raise FloatingPointError("Nonfinite external prediction")
            probs.append(torch.sigmoid(logit).numpy())
    return np.concatenate(probs).astype(np.float64)
