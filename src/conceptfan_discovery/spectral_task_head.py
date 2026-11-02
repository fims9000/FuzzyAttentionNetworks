"""Canonical task-specific logit head learned on TRAIN-detected dynamic concepts.

This is a statistical control for cross-run allocation instability:
the head uses labels Y, but NEVER human concept labels or synthetic latent
states, and is fitted *before* every independently trained decoder.
The coefficients belong to measurable spectral components, not disease names.

A head shared across seeds is stable BY DESIGN. Only patient-specific excess
agreement over chance, prediction performance and latent correspondence are
valid tests of whether this construct helps.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from .spectral_concepts_v3 import SpectralConceptFAN


@dataclass(frozen=True)
class CanonicalTaskHead:
    signed_coefficient: np.ndarray
    bias: float
    train_only_probability_mean: float
    n_training_patients: int
    clipped_beta_count: int

    def to_json(self) -> dict:
        return {
            "beta":self.signed_coefficient.tolist(),
            "bias":float(self.bias),
            "n_train":int(self.n_training_patients),
            "mean_predicted_training_probability":float(self.train_only_probability_mean),
            "coefficient_clipped_count":int(self.clipped_beta_count),
            "train_only_task_labels":True,
            "human_concept_labels":False,
            "latent_oracle_labels":False,
            "interpretation":"shared direct logit effect on train-observable spectral concepts, not real-world intervention",
        }


def fit_canonical_task_head(
    train_values: np.ndarray,
    train_mask: np.ndarray,
    train_outcomes: np.ndarray,
    *,
    observable_basis:np.ndarray,
    channels:tuple[str,...],
    forecast_horizon:int,
    regularization_c:float=1.0,
    max_beta:float=2.8,
)->CanonicalTaskHead:
    x=np.asarray(train_values,dtype=np.float32)
    m=np.asarray(train_mask,dtype=np.float32)
    y=np.asarray(train_outcomes,dtype=np.float32)
    if x.ndim!=3 or m.shape!=x.shape or y.shape!=(len(x),):
        raise ValueError("Expected matched train patient trajectories and labels")
    if np.unique(y).size!=2 or regularization_c<=0 or not 0<max_beta<3:
        raise ValueError("Need both binary classes and supported fixed logistic regularization")
    # This reference mapping is fixed entirely by train sensor autocovariance
    # and hyperparameters; random neural initialization cannot influence it.
    reference=SpectralConceptFAN(
        observable_loadings=observable_basis,
        channels=channels,
        forecast_horizon=forecast_horizon,
        freeze_definitions=True,
    )
    reference.eval()
    signal=[]
    with torch.no_grad():
        for start in range(0,len(x),128):
            out=reference(
                torch.from_numpy(x[start:start+128]),
                torch.from_numpy(m[start:start+128]),
            )
            signal.append((out.concept_summary-.5*out.gates).cpu().numpy())
    features=np.concatenate(signal,axis=0)
    if not np.isfinite(features).all() or np.any(features.var(axis=0)<1e-9):
        raise ValueError("Degenerate observed concepts: cannot fit a truthful task head")
    clf=LogisticRegression(
        C=float(regularization_c),solver="liblinear",max_iter=2000,
    )
    clf.fit(features,y.astype(int))
    natural=clf.coef_.reshape(-1)
    beta=np.clip(natural,-max_beta,max_beta)
    scores=clf.intercept_[0]+features@beta
    return CanonicalTaskHead(
        signed_coefficient=beta.astype(np.float32),
        bias=float(clf.intercept_[0]),
        train_only_probability_mean=float((1/(1+np.exp(-scores))).mean()),
        n_training_patients=int(len(x)),
        clipped_beta_count=int(np.count_nonzero(natural!=beta)),
    )


def initialize_head(
    model:SpectralConceptFAN,
    head:CanonicalTaskHead,
    *,
    freeze_head:bool,
)->None:
    if model.n_slots!=len(head.signed_coefficient):
        raise ValueError("Head dimension differs from train-observed predictive rank")
    beta=np.asarray(head.signed_coefficient,dtype=np.float64)
    if np.any(abs(beta)>=3.):
        raise ValueError("Canonical beta exceeds model's bounded logistic head")
    raw=np.arctanh(beta/3.0).astype(np.float32)
    with torch.no_grad():
        model.decision_raw_weight.copy_(torch.from_numpy(raw))
        model.decision_bias.fill_(float(head.bias))
    model.decision_raw_weight.requires_grad_(not freeze_head)
    model.decision_bias.requires_grad_(not freeze_head)


def task_head_alignment_penalty(model:SpectralConceptFAN,head:CanonicalTaskHead)->torch.Tensor:
    target=model.decision_raw_weight.new_tensor(head.signed_coefficient)
    residual=(model.decision_weights()-target).square().mean()
    return residual+(model.decision_bias-float(head.bias)).square()
