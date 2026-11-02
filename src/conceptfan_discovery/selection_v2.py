"""Separate and reproducible validation selection of dynamic K, with denoising.

Selection uses one fixed artificially hidden 20% of observed entries and
held-out patients disjoint from the final audit. No oracle factor K is used.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, log_loss

from .data import DiscoverySplit
from .evaluation import evaluate_discovery
from .model_v2 import DenoisingPredictiveConceptFAN, masked_observation_view
from .selection import SelectedConceptModel


def _score(
    model: DenoisingPredictiveConceptFAN,
    data: DiscoverySplit,
    active: list[int],
    seed: int,
) -> tuple[dict, None]:
    selection = torch.zeros(model.n_slots,dtype=torch.float32)
    selection[active] = 1.
    clean = evaluate_discovery(model, data, selected_slots=selection)
    # For every proposed subset, the masking is IDENTICAL (same patients,
    # same observed cells and same generator seed). Avoid search noise.
    generator=torch.Generator().manual_seed(int(seed))
    x=torch.from_numpy(data.x)
    mask=torch.from_numpy(data.mask)
    corrupted, visible, withheld=masked_observation_view(
        x,mask,fraction=.20,generator=generator
    )
    with torch.no_grad():
        masked_out=model(corrupted,visible,selected_slots=selection)
        residual=(masked_out.reconstruction-x).square()*withheld
        denoise_rmse=float((residual.sum()/withheld.sum().clamp_min(1.)).sqrt())
    return {
        "k":len(active),
        "enabled":active.copy(),
        "auprc":float(average_precision_score(data.y, clean.probabilities)),
        "nll":float(log_loss(data.y, clean.probabilities,labels=[0,1])),
        "future_rmse":float(clean.metrics["future_observed_rmse"]),
        "masked_reconstruction_rmse":denoise_rmse,
        "exact_additive_error":float(clean.metrics["max_additive_error"]),
    },None


def select_v2_cardinality(
    model: DenoisingPredictiveConceptFAN,
    selection: DiscoverySplit,
    audit: DiscoverySplit,
    *,
    mode: str,
    seed: int = 20261113,
    max_auprc_drop: float = .02,
    max_nll_increase: float = .03,
    max_future_rmse_increase: float = .035,
    max_masked_rmse_increase: float = .025,
) -> SelectedConceptModel:
    if mode not in ("task_only","task_future","full_without_denoising","full"):
        raise ValueError("Unknown frozen model objective")
    if len(set(selection.episode_ids)&set(audit.episode_ids)):
        raise ValueError("Disjoint validation episodes required")
    if min(max_auprc_drop,max_nll_increase,
           max_future_rmse_increase,max_masked_rmse_increase)<0:
        raise ValueError("No negative error tolerances allowed")
    model.eval()
    active=list(range(model.n_slots))
    baseline,_=_score(model,selection,active,seed)
    path=[{"step":0,"removed":None,"selection":baseline}]
    while len(active)>1:
        proposals=[]
        for removed in active:
            enabled=[v for v in active if v!=removed]
            metric,_=_score(model,selection,enabled,seed)
            acceptable=(
                metric["auprc"]>=baseline["auprc"]-max_auprc_drop
                and metric["nll"]<=baseline["nll"]+max_nll_increase
                and (mode=="task_only" or
                    metric["future_rmse"]<=baseline["future_rmse"]+max_future_rmse_increase)
                and (mode!="full" or
                    metric["masked_reconstruction_rmse"]<=
                    baseline["masked_reconstruction_rmse"]+max_masked_rmse_increase)
            )
            if acceptable:
                priority=metric["nll"]+(.5*metric["future_rmse"] if mode!="task_only" else 0.)
                if mode=="full":
                    priority+=.75*metric["masked_reconstruction_rmse"]
                proposals.append((priority,removed,enabled,metric))
        if not proposals:
            break
        _,removed,active,metric=min(proposals,key=lambda v:(v[0],v[1]))
        path.append({"step":len(path),"removed":removed,"selection":metric})
    selected,_=_score(model,selection,active,seed)
    outer,_=_score(model,audit,active,seed+1)
    mask=np.zeros(model.n_slots,dtype=np.float32)
    mask[active]=1.
    outer["selected_slots"]=active.copy()
    outer["no_outcome_from_audit_used_for_selection"]=True
    outer["selection_method"]="greedy native slot elimination with fixed prediction+forecast+denoising noninferiority"
    return SelectedConceptModel(mask,path,selected,outer,
                                selection.episode_ids.copy(),audit.episode_ids.copy())
