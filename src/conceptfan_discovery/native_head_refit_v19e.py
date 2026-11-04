"""V19e: fit the *existing* native ConceptFAN risk head after concept training.

This is a second, TRAIN-only phase of model fitting. It never attaches a
post-hoc student or raw-input classifier to inference. Physical concepts,
forecasting, masked reconstruction and care observation concepts are frozen.
Only actual native decision_raw_weight and decision_bias are updated.

The refit is not a causal/medical proof; it addresses optimizer competition
between a discriminative risk head and auxiliary dynamics objectives.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class NativeHeadRefitReceipt:
    n_train_episodes: int
    native_slots: int
    original_objective: float
    final_objective: float
    original_train_bce: float
    final_train_bce: float
    optimized_native_parameters: tuple[str, ...]
    refused_validation_oracle: bool = True
    no_new_inference_head: bool = True


def _native_betas(raw: torch.Tensor, policy: torch.Tensor) -> torch.Tensor:
    basic=3.0*torch.tanh(raw)
    orientation=policy.to(device=basic.device,dtype=basic.dtype)
    signed=torch.where(orientation==0,basic,orientation*basic.abs())
    # Native descriptive-only slot policy 2 is a structural zero:
    # even separate TRAIN head refits cannot convert it to a risk signal.
    return torch.where(orientation==2,torch.zeros_like(basic),signed)


def fit_train_only_native_risk_head(
    model,
    train_split,
    *,
    declared_train_episode_ids,
    weight_l2: float = .001,
    max_iter: int = 120,
    batch_size: int = 256,
) -> NativeHeadRefitReceipt:
    """Refit only actual physical native coefficients and bias from TRAIN.

    Train membership is checked on *each* entry. Avoids heldout labels,
    oracle states, current input raw operators outside already learned
    native C, and independently fitted post-hoc student models. Enforces
    zero-drift of all other model tensors by only writing two parameters.
    """
    model_params=list(model.parameters())
    if not model_params:
        raise ValueError("No trainable native model")
    if not (np.isfinite(weight_l2) and 0<=weight_l2<=.05 and
            10<=max_iter<=500 and batch_size>=1):
        raise ValueError("Invalid preregistered head-refit controls")
    expected={int(i) for i in declared_train_episode_ids}
    given=[int(i) for i in train_split.episode_ids.tolist()]
    if (len(expected)==0 or len(expected)!=len(given) or
        set(given)!=expected or len(set(given))!=len(given)):
        raise ValueError("ABSTAIN: only declared TRAIN IDs may inform native refit")
    if getattr(train_split,"oracle_states",None) is not None:
        # Oracle trajectories may exist on a synthetic data object, but
        # not needed here and not read; reject them to make the contract
        # simpler and mechanically auditable.
        raise ValueError("ABSTAIN: oracle-state data attached to native head refit")
    if len(np.unique(train_split.y)) != 2:
        raise ValueError("ABSTAIN: risk refit requires both outcome classes")

    was_training=model.training
    model.eval()
    device=model_params[0].device
    physical=model.physical
    if not hasattr(physical,"signed_risk_policy"):
        raise ValueError("Refit supports only inspected V18/V19 physical heads")
    if physical.decision_raw_weight.numel()!=physical.n_slots:
        raise ValueError("Native risk coefficient dimensional mismatch")
    raw_orig=physical.decision_raw_weight.detach().clone()
    bias_orig=physical.decision_bias.detach().clone()
    expected_native=physical.decision_weights().detach()
    current_native=_native_betas(raw_orig,physical.signed_risk_policy)
    if not torch.allclose(expected_native,current_native,atol=1e-6,rtol=1e-6):
        raise ValueError("ABSTAIN: unsupported nonstandard native head formula")

    X=[];care=[]
    with torch.no_grad():
        for i in range(0,len(given),batch_size):
            stop=min(i+batch_size,len(given))
            values=torch.from_numpy(train_split.x[i:stop]).to(device)
            mask=torch.from_numpy(train_split.mask[i:stop]).to(device)
            out=model(values,mask,process_mask=mask)
            if float(out.decomposition_error())>2e-5:
                raise AssertionError("Original model is not an exact native additive bottleneck")
            X.append((out.physical.concept_summary-.5*out.physical.gates.view(1,-1)).detach())
            care.append(out.observation_contributions.sum(dim=1).detach())
    features=torch.cat(X,dim=0)
    fixed_care=torch.cat(care,dim=0)
    target=torch.as_tensor(train_split.y,device=device,dtype=features.dtype)
    if features.shape!=(len(given),physical.n_slots) or target.shape!=(len(given),):
        raise ValueError("Native concept-only fitting dimensions are invalid")
    if not torch.isfinite(features).all() or not torch.isfinite(fixed_care).all():
        raise ValueError("Nonfinite concept features or observation-process contribution")
    if not torch.isfinite(target).all():
        raise ValueError("Nonfinite train labels")

    theta=torch.nn.Parameter(raw_orig)
    bias=torch.nn.Parameter(bias_orig)
    policy=physical.signed_risk_policy
    def calculate():
        coefficients=_native_betas(theta,policy)
        logits=bias+features@coefficients+fixed_care
        task=F.binary_cross_entropy_with_logits(logits,target)
        objective=task+float(weight_l2)*coefficients.square().sum()
        return objective,task

    with torch.no_grad():
        initial,task_before=calculate()
        first_obj=float(initial)
        first_task=float(task_before)
    optimizer=torch.optim.LBFGS(
        [theta,bias],lr=1.0,max_iter=int(max_iter),
        line_search_fn="strong_wolfe",tolerance_grad=1e-9,
    )
    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss,_=calculate()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite native head-only task objective")
        loss.backward()
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        final,task_after=calculate()
        if not torch.isfinite(final) or float(final)>first_obj+1e-6:
            raise FloatingPointError("Native head refit failed its own TRAIN objective")
        physical.decision_raw_weight.copy_(theta)
        physical.decision_bias.copy_(bias)
    model.train(was_training)
    return NativeHeadRefitReceipt(
        n_train_episodes=len(given),
        native_slots=physical.n_slots,
        original_objective=first_obj,
        final_objective=float(final),
        original_train_bce=first_task,
        final_train_bce=float(task_after),
        optimized_native_parameters=(
            "physical.decision_raw_weight","physical.decision_bias",
        ),
    )
