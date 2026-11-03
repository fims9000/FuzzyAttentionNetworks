"""Matched V11 training: replace arbitrary SVD-axis anchoring by group-projector loss.

Both models use the SAME native V10 measured-operator concepts and separate
mask-only clinical observation-process head. Unlike post-hoc axis matching,
group constraints influence the model DURING outcome/forecast/denoising fit.

The new mask consistency option compares rotation-invariant *group* temporal
patterns, so directions within unresolved repeated-eigenvalue groups are
not forced to have arbitrary stable names.
"""
from __future__ import annotations

import random

import numpy as np
import torch

from .data import DiscoveryData
from .identifiability_v11 import (
    PredictiveCertificate, torch_rotation_invariant_group_loss,
)
from .model_v2 import masked_observation_view, v2_loss
from .multiscale_physiology_v10 import (
    MultiScaleDualProcessConceptFAN, physiological_scale_entropy,
)
from .observation_process_concepts_v7 import dual_training_objective
from .rotation_consistency_v11 import grouped_rotation_invariant_view_consistency
from .unknown_neutral_v6 import cross_view_concept_consistency


def fit_identifiability_aware_v11(
    data:DiscoveryData, source_basis:np.ndarray,
    certificate:PredictiveCertificate,
    seed:int,base_config:dict,v7_config:dict,
    v10_config:dict,v11_config:dict, *,
    variant:str,
)->tuple[MultiScaleDualProcessConceptFAN,dict]:
    """Train exact same native architecture with different *principled* losses.

    v11_projector_axis_view: replace only SVD axis anchor; preserve V10
      per-axis observation-thinning consistency.
    v11_projector_group_view: also replace per-axis consistency with
      group Gram consistency (rotation-invariant by construction).

    The groups are fitted from TRAIN only; neither validation label nor
    any latent source identity enters the loss.
    """
    if variant not in ("v11_projector_axis_view","v11_projector_group_view"):
        raise ValueError("Unregistered research arm")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    t=base_config["training"]
    dual=v7_config["dual_head"]
    v10=v10_config["training"]
    v11=v11_config["training"]
    if len(certificate.groups)<1 or certificate.selected_computational_rank!=source_basis.shape[1]:
        raise ValueError("Train-only certificate must match model rank")
    model=MultiScaleDualProcessConceptFAN(
        observable_loadings=source_basis,
        channels=tuple(data.receipt["channel_names"]),
        forecast_horizon=data.train.future.shape[1],
        n_observation_concepts=int(dual["n_observation_process_concepts"]),
        scales=tuple(int(s) for s in v10_config["architecture"]["scales_hours"]),
    )
    optimizer=torch.optim.AdamW(
        model.parameters(),lr=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
    )
    n_positive=float(data.train.y.sum())
    weight=torch.tensor(
        (len(data.train.y)-n_positive)/max(n_positive,1.),
        dtype=torch.float32,
    )
    loader=data.train.loader(int(t["batch_size"]),True,int(t["batch_order_seed"]))
    rng_main=torch.Generator().manual_seed(int(seed)+71021)
    rng_other=torch.Generator().manual_seed(int(seed)+71022)
    summed={"projector_loss":0.,"group_view_loss":0.,
            "joint_reliability":0.,"scale_entropy":0.}
    count=0
    for epoch in range(int(t["epochs"])):
        model.train()
        for x,mask,future,future_mask,y in loader:
            optimizer.zero_grad(set_to_none=True)
            corrupted,visible,heldout=masked_observation_view(
                x,mask,fraction=float(t["main_mask_fraction"]),
                generator=rng_main,
            )
            out=model(corrupted,visible,process_mask=mask)
            physical_loss,physical_terms=v2_loss(
                model.physical,out.physical,y,future,future_mask,
                x,heldout,mode="full",pos_weight=weight,
                loss_weights=t["loss_weights"],
            )
            total,_=dual_training_objective(
                model,out,y,physical_loss,physical_terms["task"],
                pos_weight=weight,
                mask_coeff_l1=float(dual["signed_mask_coefficient_l1"]),
                mask_redundancy_weight=float(dual["mask_definition_redundancy"]),
            )
            group_penalty=torch_rotation_invariant_group_loss(
                model.physical.feature_weights(),
                model.physical.source_predictive_basis,
                certificate.groups,
                epsilon=float(v11["projector_ridge"]),
            )
            regularizers=model.physical.spectral_definition_penalty()
            # The old per-axis operator_reference MSE is intentionally NOT
            # included; its arbitrary SVD-axis orientation is the problem.
            total+=float(v11["group_projector_weight"])*group_penalty
            total+=float(t["spectral_axis_independence_weight"])*regularizers[
                "observable_axis_nonorthogonality"
            ]
            entropy=physiological_scale_entropy(model)
            total+=float(v10["scale_entropy_weight"])*entropy
            altered,altered_mask,_=masked_observation_view(
                x,mask,fraction=float(v10["second_mask_fraction"]),
                generator=rng_other,
            )
            other=model(altered,altered_mask,process_mask=mask)
            if variant=="v11_projector_axis_view":
                view_loss,reliability=cross_view_concept_consistency(
                    out.physical,other.physical,
                    minimum_joint_reliability=float(v10["minimum_joint_reliability"]),
                )
            else:
                view_loss,reliability=grouped_rotation_invariant_view_consistency(
                    out.physical,other.physical,certificate.groups,
                    minimum_joint_reliability=float(v10["minimum_joint_reliability"]),
                )
            total+=float(v10["consistency_weight"])*view_loss
            if not torch.isfinite(total):
                raise FloatingPointError(f"Nonfinite V11 loss, seed {seed} epoch {epoch}")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),float(t["gradient_clip"]))
            optimizer.step()
            count+=1
            summed["projector_loss"]+=float(group_penalty.detach())
            summed["group_view_loss"]+=float(view_loss.detach())
            summed["joint_reliability"]+=float(reliability.detach())
            summed["scale_entropy"]+=float(entropy.detach())
    model.eval()
    return model,{
        "source_basis_is_train_only":True,
        "concepts_are_native_to_training":True,
        "arbitrary_svd_axis_anchor_disabled":True,
        "variant":variant,
        "n_training_minibatches":count,
        "group_count":len(certificate.groups),
        "n_singleton_observable_certificates":sum(
            g.status=="INDIVIDUAL_OBSERVABLE_AXIS_STABLE"
            for g in certificate.groups
        ),
        **{f"mean_train_{k}":v/max(count,1) for k,v in summed.items()},
    }
