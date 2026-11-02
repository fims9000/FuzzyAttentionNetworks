"""Transparent V8 care sampling concepts fixed to TRAIN-only mask subspace.

The process concept count, signed definition anchors and orientation are
derived entirely from the TRAIN patient observation masks, using
permutation-calibrated between-operator covariance rank. No human concept
labels or patient diagnosis names are used.

Two predeclared controlled fits:
  source_only: source operator mix fixed, task beta may train freely;
  source_plus_train_task_prior: same source mix, task beta initialized and
      penalized relative to one TRAIN-only logistic mask-concept head.

In BOTH cases the observation process is a separate explicitly named
source of outcome evidence, not "physiology" or a treatment causal effect.
Shared definitions or coefficients stable by construction are NOT alone
proof of patient-specific semantic stability.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from sklearn.linear_model import LogisticRegression

from .care_process_rank_v8 import CarePolicyRank
from .observation_process_concepts_v7 import DualProcessConceptFAN


@dataclass(frozen=True)
class CareTaskPrior:
    beta:np.ndarray
    n_train:int
    class_prevalence:float
    clipped_coefficients:int

    def receipt(self)->dict:
        return {
            "signed_logit_coefficients":self.beta.tolist(),
            "train_episodes":int(self.n_train),
            "train_outcome_prevalence":float(self.class_prevalence),
            "clipped_coefficients":int(self.clipped_coefficients),
            "fitted_from_mask_only_natively_defined_train_concepts":True,
            "physiological_concept_labels_supplied":False,
            "meaning":"shared training-task coefficient prior, not direct clinical effect",
        }


class MaskRankAnchoredDualConceptFAN(DualProcessConceptFAN):
    def __init__(
        self,*,
        physiological_observable_basis:np.ndarray,
        care_rank:CarePolicyRank,
        channels:tuple[str,...],
        forecast_horizon:int,
    ):
        if len(channels)!=care_rank.n_observed_sensor_channels:
            raise ValueError("The rank-selected sensor dictionary must match clinical mask")
        super().__init__(
            observable_loadings=physiological_observable_basis,
            channels=channels,forecast_horizon=forecast_horizon,
            n_observation_concepts=care_rank.chosen_k,
        )
        target=torch.as_tensor(
            care_rank.positive_negative_simplex_anchors,
            dtype=self.process.feature_logits.dtype
        )
        with torch.no_grad():
            self.process.feature_logits.copy_(target)
            self.process.threshold_logits.zero_()
            self.process.width_logits.fill_(-1.75)
        self.process.feature_logits.requires_grad_(False)
        self.register_buffer(
            "frozen_observation_source_definition",target.detach().clone()
        )
        with torch.no_grad():
            torch.testing.assert_close(
                self.process.feature_weights(),target,atol=1e-5,rtol=1e-5
            )

    def care_operator_definition_error(self)->torch.Tensor:
        return (
            self.process.feature_weights()
            -self.frozen_observation_source_definition
        ).abs().max()


def fit_train_only_care_task_prior(
    reference:MaskRankAnchoredDualConceptFAN,
    train_mask:np.ndarray,
    labels:np.ndarray,
    *,
    c:float=1.,
    clip_beta:float=2.7,
)->CareTaskPrior:
    m=np.asarray(train_mask,dtype=np.float32)
    y=np.asarray(labels,dtype=np.float32)
    if m.ndim!=3 or m.shape[0]!=len(y) or np.unique(y).size!=2:
        raise ValueError("Train mask samples and binary outcomes must align")
    if c<=0 or not 0<clip_beta<3.:
        raise ValueError("Invalid predeclared logistic mask task-head hyperparameters")
    reference.eval()
    features=[]
    with torch.no_grad():
        for i in range(0,len(y),128):
            scores,_,_,_=reference.process(
                torch.from_numpy(m[i:i+128])
            )
            features.append((scores-.5).detach().cpu().numpy())
    x=np.concatenate(features,axis=0)
    if not np.isfinite(x).all() or (x.var(axis=0)<1e-9).any():
        raise ValueError("Undefined process concept under train masks")
    classifier=LogisticRegression(
        C=float(c),solver="liblinear",max_iter=2000,
    )
    classifier.fit(x,y.astype(int))
    coefficients=classifier.coef_.ravel().astype(float)
    clipped=np.clip(coefficients,-clip_beta,clip_beta)
    return CareTaskPrior(
        beta=clipped.astype(np.float32),
        n_train=len(y),
        class_prevalence=float(y.mean()),
        clipped_coefficients=int(np.count_nonzero(clipped!=coefficients)),
    )


def initialize_task_prior(
    model:MaskRankAnchoredDualConceptFAN,
    head:CareTaskPrior,
)->None:
    beta=np.asarray(head.beta,dtype=float)
    if beta.shape!=(model.process.n_concepts,):
        raise ValueError("The task-head concept dimension differs from train policy rank")
    if (abs(beta)>=3.).any():
        raise ValueError("Out-of-range canonical signed effect")
    with torch.no_grad():
        model.process.signed_beta_raw.copy_(
            torch.from_numpy(np.arctanh(beta/3.).astype(np.float32))
        )


def process_head_prior_loss(
    model:MaskRankAnchoredDualConceptFAN,prior:CareTaskPrior,
)->torch.Tensor:
    target=model.process.signed_beta_raw.new_tensor(prior.beta)
    return (model.process.direct_weights()-target).square().mean()
