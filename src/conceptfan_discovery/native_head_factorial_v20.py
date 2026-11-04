"""V20: isolate native decision optimizer vs original class-weighted task loss.

Do not add a post-hoc classifier. This refits ONLY the original physical
ConceptFAN decision_raw_weight and decision_bias using frozen TRAIN-native
concept summaries. The mask-only care contribution remains fixed.

All refits start from the *same* original native parameters. Disjoint
validation and sealed test are not passed to the optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .native_head_refit_v19e import _native_betas


@dataclass(frozen=True)
class FactorialHeadReceipt:
    method: str
    pos_weighted: bool
    positive_weight_from_train: float
    initial_objective: float
    final_objective: float
    initial_unweighted_bce: float
    final_unweighted_bce: float
    initial_weighted_bce: float
    final_weighted_bce: float
    effective_beta_l2: float
    adamw_steps: int
    initial_signed_coefficients: tuple[float, ...]
    final_signed_coefficients: tuple[float, ...]


def fit_native_head_factorial_v20(
    model,
    train_split,
    *,
    declared_train_episode_ids,
    method: str,
    pos_weighted: bool,
    effective_beta_l2: float = .001,
    lbfgs_max_iter: int = 120,
    adamw_steps: int = 300,
    adamw_lr: float = .03,
    batch_size: int = 256,
) -> FactorialHeadReceipt:
    """Refit actual existing native beta/bias with a predeclared full-data loss.

    Unlike external logistic regression, the *inference graph* remains the
    same ConceptFAN. No learned patient features from outside native C.
    """
    if method not in ("lbfgs","adamw"):
        raise ValueError("Unsupported native risk optimizer")
    if (not np.isfinite(effective_beta_l2) or not 0 <= effective_beta_l2 <= .05
            or not 10 <= lbfgs_max_iter <= 500
            or not 10 <= adamw_steps <= 2000
            or not np.isfinite(adamw_lr) or not 0 < adamw_lr < 1
            or batch_size<1):
        raise ValueError("Invalid native head refit parameters")
    expected={int(v) for v in declared_train_episode_ids}
    actual=[int(v) for v in train_split.episode_ids.tolist()]
    if (len(expected)<32 or len(expected)!=len(actual)
            or len(set(actual))!=len(actual) or set(actual)!=expected):
        raise ValueError("ABSTAIN: native head optimization requires declared TRAIN-only IDs")
    if getattr(train_split,"oracle_states",None) is not None:
        raise ValueError("ABSTAIN: oracle states cannot enter risk head fitting")
    if train_split.y.ndim != 1 or len(train_split.y) != len(actual):
        raise ValueError("Invalid binary TRAIN labels")
    unique=np.unique(train_split.y)
    if not np.array_equal(unique,np.array([0.,1.])):
        raise ValueError("ABSTAIN: task refit requires two binary outcomes")
    if not hasattr(model.physical,"signed_risk_policy"):
        raise ValueError("Native V10+ risk head is required")
    was_training=model.training
    model.eval()
    physical=model.physical
    device=physical.decision_raw_weight.device
    features=[]
    care_offsets=[]
    with torch.no_grad():
        for i in range(0,len(actual),batch_size):
            end=min(i+batch_size,len(actual))
            x=torch.from_numpy(train_split.x[i:end]).to(device)
            observed=torch.from_numpy(train_split.mask[i:end]).to(device)
            output=model(x,observed,process_mask=observed)
            if float(output.decomposition_error())>2e-5:
                raise AssertionError("Non-additive original ConceptFAN native graph")
            features.append((
                output.physical.concept_summary
                - .5*output.physical.gates.view(1,-1)
            ).detach())
            care_offsets.append(output.observation_contributions.sum(-1).detach())
    X=torch.cat(features,dim=0)
    care=torch.cat(care_offsets,dim=0)
    target=torch.as_tensor(train_split.y,device=device,dtype=X.dtype)
    if X.shape!=(len(actual),physical.n_slots) or care.shape!=target.shape:
        raise ValueError("Unexpected native feature dimensions")
    if not torch.isfinite(X).all() or not torch.isfinite(care).all():
        raise ValueError("Nonfinite native concept feature")
    positives=int(np.count_nonzero(train_split.y))
    neg=len(actual)-positives
    weight=float(neg/positives)
    pos_weight=torch.tensor(weight,device=device,dtype=X.dtype)

    raw_initial=physical.decision_raw_weight.detach().clone()
    bias_initial=physical.decision_bias.detach().clone()
    theta=torch.nn.Parameter(raw_initial)
    bias=torch.nn.Parameter(bias_initial)
    policy=physical.signed_risk_policy

    def objective():
        beta=_native_betas(theta,policy)
        logits=bias+X@beta+care
        weighted=F.binary_cross_entropy_with_logits(
            logits,target,pos_weight=pos_weight,
        )
        plain=F.binary_cross_entropy_with_logits(logits,target)
        penalty=effective_beta_l2*beta.square().sum()
        return (weighted if pos_weighted else plain)+penalty,weighted,plain

    with torch.no_grad():
        before,weighted_start,plain_start=objective()
        signed_before=_native_betas(theta,policy).detach().cpu().tolist()
        initial_loss=float(before)
    if method=="lbfgs":
        optimizer=torch.optim.LBFGS(
            [theta,bias],lr=1.,max_iter=int(lbfgs_max_iter),
            line_search_fn="strong_wolfe",tolerance_grad=1e-9,
        )
        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss,_,_=objective()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite native LBFGS risk loss")
            loss.backward()
            return loss
        optimizer.step(closure)
    else:
        optimizer=torch.optim.AdamW(
            [theta,bias],lr=float(adamw_lr),weight_decay=0.,
        )
        # Retain the best TRAIN objective (not VALIDATION-selected) so
        # a poor late optimizer step cannot silently corrupt the model.
        best_raw=raw_initial.clone()
        best_bias=bias_initial.clone()
        best_loss=initial_loss
        for _ in range(int(adamw_steps)):
            optimizer.zero_grad(set_to_none=True)
            loss,_,_=objective()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite native AdamW task loss")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                current,_,_=objective()
                if float(current)<best_loss:
                    best_loss=float(current)
                    best_raw.copy_(theta)
                    best_bias.copy_(bias)
        with torch.no_grad():
            theta.copy_(best_raw)
            bias.copy_(best_bias)
    with torch.no_grad():
        final,weighted_last,plain_last=objective()
        if not torch.isfinite(final) or float(final)>initial_loss+1e-6:
            raise FloatingPointError("Frozen TRAIN-only native head objective failed to decrease")
        physical.decision_raw_weight.copy_(theta)
        physical.decision_bias.copy_(bias)
        signed_after=_native_betas(theta,policy).cpu().tolist()
    model.train(was_training)
    return FactorialHeadReceipt(
        method=method,
        pos_weighted=pos_weighted,
        positive_weight_from_train=weight,
        initial_objective=initial_loss,
        final_objective=float(final),
        initial_unweighted_bce=float(plain_start),
        final_unweighted_bce=float(plain_last),
        initial_weighted_bce=float(weighted_start),
        final_weighted_bce=float(weighted_last),
        effective_beta_l2=float(effective_beta_l2),
        adamw_steps=(int(adamw_steps) if method=="adamw" else 0),
        initial_signed_coefficients=tuple(map(float,signed_before)),
        final_signed_coefficients=tuple(map(float,signed_after)),
    )
