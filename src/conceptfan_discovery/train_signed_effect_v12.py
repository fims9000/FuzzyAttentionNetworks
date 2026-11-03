"""V12: train native concepts with stable observable singleton orientation.

Keep exact V11 architecture and all V11 projector/rotation-invariant losses.
New gradients are added to the same native concept path and direct logit,
not post-hoc explanations or a second classifier. Source anchors and
supported task signs are derived ONLY from Set A TRAIN before fit.
Unresolved source groups receive no individual identity or sign constraints.
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
from .signed_effect_v12 import SignedAnchorCertificate, orient_only_resolved_singletons, signed_effect_training_regularizers


def fit_signed_effect_v12(
    data:DiscoveryData, source_basis:np.ndarray,
    certificate:PredictiveCertificate,anchors:SignedAnchorCertificate,
    seed:int,base_config:dict,v7_config:dict,
    v10_config:dict,v11_config:dict,v12_config:dict, *,
    variant:str,
)->tuple[MultiScaleDualProcessConceptFAN,dict]:
    """Ablate observable anchoring vs additionally training signed effects."""
    if variant not in ("v12_observable_anchor","v12_anchor_signed_effect"):
        raise ValueError("Unregistered V12 research arm")
    oriented,individual=orient_only_resolved_singletons(certificate)
    if (source_basis.shape!=oriented.shape or
        not np.allclose(source_basis,oriented,atol=1e-8) or
        not np.array_equal(anchors.singleton,individual) or
        not np.allclose(anchors.basis,oriented,atol=1e-8)):
        raise ValueError("V12 source basis MUST be train-only canonically oriented")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    t=base_config["training"]
    dual=v7_config["dual_head"]
    v10=v10_config["training"]
    v11=v11_config["training"]
    v12=v12_config["training"]
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
            "joint_reliability":0.,"scale_entropy":0.,
            "v12_observable_anchor":0.,"v12_signed_effect_view":0.,
            "v12_risk_sign_regularizer":0.}
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
            view_loss,reliability=grouped_rotation_invariant_view_consistency(
                out.physical,other.physical,certificate.groups,
                minimum_joint_reliability=float(v10["minimum_joint_reliability"]),
            )
            total+=float(v10["consistency_weight"])*view_loss
            # Anchor is constructed from ONLY observed original TRAIN history;
            # masked train views cannot read missing or later measurements.
            extra=signed_effect_training_regularizers(
                model,out.physical,other.physical,x,mask,anchors,
                history_steps=int(v12["anchor_history_hours"]),
                margin=float(v12["risk_sign_margin"]),
            )
            total+=float(v12["observable_anchor_weight"])*extra["observable_anchor"]
            if variant=="v12_anchor_signed_effect":
                total+=float(v12["signed_effect_view_weight"])*extra["signed_effect_view"]
                total+=float(v12["train_association_sign_weight"])*extra["train_association_risk_sign"]
            if not torch.isfinite(total):
                raise FloatingPointError(f"Nonfinite V12 loss, seed {seed} epoch {epoch}")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),float(t["gradient_clip"]))
            optimizer.step()
            count+=1
            summed["projector_loss"]+=float(group_penalty.detach())
            summed["group_view_loss"]+=float(view_loss.detach())
            summed["joint_reliability"]+=float(reliability.detach())
            summed["scale_entropy"]+=float(entropy.detach())
            summed["v12_observable_anchor"]+=float(extra["observable_anchor"].detach())
            summed["v12_signed_effect_view"]+=float(extra["signed_effect_view"].detach())
            summed["v12_risk_sign_regularizer"]+=float(extra["train_association_risk_sign"].detach())
    model.eval()
    return model,{
        "source_basis_is_train_only":True,
        "concepts_are_native_to_training":True,
        "arbitrary_svd_axis_anchor_disabled":True,
        "variant":variant,
        "source_anchor_train_only":True,
        "forced_risk_signs_selected_train_only":True,
        "n_train_risk_signs":int(np.count_nonzero(anchors.risk_sign)),
        "signed_effect_regularization":variant=="v12_anchor_signed_effect",
        "no_new_concept_classifier_or_posthoc_explainer":True,
        "n_training_minibatches":count,
        "group_count":len(certificate.groups),
        "n_singleton_observable_certificates":sum(
            g.status=="INDIVIDUAL_OBSERVABLE_AXIS_STABLE"
            for g in certificate.groups
        ),
        **{f"mean_train_{k}":v/max(count,1) for k,v in summed.items()},
    }
