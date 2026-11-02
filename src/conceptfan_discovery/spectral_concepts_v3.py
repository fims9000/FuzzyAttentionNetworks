"""Transparent, train-only spectrally discovered native concept model.

An *unsupervised* positive-lag covariance rank supplies K and a signed linear
sensor subspace. Every slot is then a concrete nonnegative sparsemax mixture
over the existing signed OBSERVABLE level operators. Concepts participate
in BCE, future observation prediction and masked denoising during training.
No clinician labels, synthetic latent truth or post-hoc factor extractor.

Frozen-feature and fine-tuned-feature variants must be compared. Shared
frozen definitions are stable by construction and CANNOT alone substantiate
nontrivial patient-specific explanation stability.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_v2 import DenoisingPredictiveConceptFAN


class FixedOpenGates(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.register_buffer("enabled", torch.ones(k,dtype=torch.float32))

    def forward(self, hard: bool = False) -> torch.Tensor:
        return self.enabled

    def deterministic(self) -> torch.Tensor:
        return self.enabled

    def expected_l0(self) -> torch.Tensor:
        return self.enabled


def spectral_operator_definitions(
    left_loadings: np.ndarray,
    *,
    n_channels: int,
    n_operators: int = 4,
) -> torch.Tensor:
    """Project a signed sensor-space basis into inspectable level operators.

    This is an explicit projection of linear sensor directions onto a bounded
    tanh-level operator dictionary, not a claim the nonlinear transforms are
    exactly the linear eigenvectors from the rank estimator.
    """
    a=np.asarray(left_loadings,dtype=np.float64)
    if a.ndim!=2 or a.shape[0]!=n_channels or not 1<=a.shape[1]<=n_channels:
        raise ValueError("Need [observable_channels, detected_rank] basis")
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite subspace basis")
    if not np.allclose(a.T@a,np.eye(a.shape[1]),atol=1e-4):
        raise ValueError("Basis must have orthonormal columns")
    k=a.shape[1]
    d=n_channels*n_operators*2
    weight=np.zeros((k,d),dtype=np.float32)
    for i in range(k):
        col=a[:,i]
        magnitude=np.abs(col)
        if magnitude.sum()<1e-12:
            raise ValueError("No measurable loading for a candidate concept")
        magnitude/=magnitude.sum()
        for channel in range(n_channels):
            direction_offset=0 if col[channel]>=0 else n_channels*n_operators
            weight[i,direction_offset+channel]=float(magnitude[channel])
    if not np.allclose(weight.sum(axis=1),1,atol=1e-6):
        raise AssertionError("Sparse operator definitions are not normalized")
    return torch.from_numpy(weight)


class SpectralConceptFAN(DenoisingPredictiveConceptFAN):
    """Automatically ranked and semantically anchored concepts trained end-to-end."""

    def __init__(
        self,
        *,
        observable_loadings: np.ndarray,
        channels: tuple[str,...],
        forecast_horizon: int,
        freeze_definitions: bool,
        min_width: float = .08,
    ):
        basis=np.asarray(observable_loadings,dtype=np.float64)
        n_channels=len(channels)
        if basis.ndim!=2 or basis.shape[0]!=n_channels or basis.shape[1]<1:
            raise ValueError("Detected predictive basis must have positive rank")
        # The vanilla class demands K>=2, reflecting its manual budget. This
        # spectral model deliberately refuses to invent a second concept
        # when train-only observed predictive rank is truly K=1.
        if basis.shape[1]<2:
            raise ValueError("Detected K=1: disclose it rather than artificially expanding concept count")
        super().__init__(
            n_slots=basis.shape[1],
            n_channels=n_channels,
            channels=channels,
            forecast_horizon=forecast_horizon,
            min_width=min_width,
        )
        template=spectral_operator_definitions(basis,n_channels=n_channels)
        self.register_buffer("operator_reference",template.detach().clone())
        with torch.no_grad():
            self.feature_logits.copy_(template)
        self.feature_logits.requires_grad_(not freeze_definitions)
        self.gates=FixedOpenGates(basis.shape[1])
        self.freeze_definitions=bool(freeze_definitions)
        self.register_buffer(
            "source_predictive_basis",torch.tensor(basis,dtype=torch.float32).clone()
        )

    def spectral_definition_penalty(self) -> dict[str,torch.Tensor]:
        w=self.feature_weights()
        displacement=(w-self.operator_reference).square().sum(dim=1).mean()
        c=self.n_channels
        signed=w[:, :c]-w[:, 4*c:5*c]
        normalized=F.normalize(signed,dim=1,eps=1e-9)
        gram=normalized@normalized.T
        off=gram-torch.diag_embed(gram.diagonal())
        orthogonality=off.square().sum()/max(self.n_slots*(self.n_slots-1),1)
        return {
            "operator_displacement":displacement,
            "observable_axis_nonorthogonality":orthogonality,
        }

    def concept_basis_manifest(self)->dict:
        return {
            "n_detected_slots":self.n_slots,
            "observable_channel_names":list(self.channels),
            "source_basis_columns":self.source_predictive_basis.detach().cpu().numpy().tolist(),
            "signed_level_operator_weights":self.feature_weights().detach().cpu().numpy().tolist(),
            "fixed_operator_definitions":self.freeze_definitions,
            "concept_labels_from_humans":False,
            "oracle_latent_labels_used":False,
            "all_slots_enabled_by_detected_train_only_predictive_rank":True,
            "warning":"spectral directions need not equal identifiable physiological mechanisms",
        }
