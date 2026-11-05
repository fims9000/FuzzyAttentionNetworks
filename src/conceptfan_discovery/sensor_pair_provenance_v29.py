"""V29: unlabeled redundancy check for one explicitly declared sensor pair.

Only observed historical co-recordings are used. This is a *consistency*
monitor under a known two-readout assumption, NOT physical/causal certification
and NOT a detector of arbitrary common-mode calibration or sign errors.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PairIntegrityPolicy:
    first_channel: int
    second_channel: int
    min_joint_steps: int
    min_pearson_corr: float
    max_mean_abs_disagreement: float
    train_subjects: int
    valid_train_subjects: int
    low_corr_quantile: float
    high_gap_quantile: float
    does_not_certify_common_mode_faults: bool = True
    outcome_labels_used: bool = False

    def as_dict(self):
        return asdict(self)


def pair_diagnostics(
    values: np.ndarray, observed_mask: np.ndarray, *,
    first_channel: int = 4, second_channel: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per episode (n_joint, signed corr, mean |difference|).

    All unmeasured payloads (including NaN) are ignored. Correlation is
    undefined (NaN) when too few observations or a zero-variance readout,
    and an integrity check must refuse those cases.
    """
    x=np.asarray(values,dtype=np.float64)
    m=np.asarray(observed_mask)
    if (x.ndim!=3 or x.shape!=m.shape or x.shape[1]<2
            or first_channel==second_channel
            or not (0<=first_channel<x.shape[-1])
            or not (0<=second_channel<x.shape[-1])):
        raise ValueError("Expected aligned historical sensor values and two distinct named channels")
    if not np.logical_or(m==0,m==1).all():
        raise ValueError("Sensor availability mask must be strictly binary")
    m=m.astype(bool)
    if not np.isfinite(x[m]).all():
        raise ValueError("Observed measurements contain nonfinite values")
    jointly=m[:,:,first_channel]&m[:,:,second_channel]
    n=jointly.sum(axis=1)
    corr=np.full(len(x),np.nan)
    gap=np.full(len(x),np.nan)
    for i in range(len(x)):
        if n[i]<2:
            continue
        a=x[i,jointly[i],first_channel]
        b=x[i,jointly[i],second_channel]
        gap[i]=float(np.abs(a-b).mean())
        a=a-a.mean()
        b=b-b.mean()
        d=np.sqrt(np.dot(a,a)*np.dot(b,b))
        if d>1e-12:
            corr[i]=float(np.clip(np.dot(a,b)/d,-1.,1.))
    return n,corr,gap


def fit_unlabelled_pair_guard(
    train_values: np.ndarray,
    train_observed_mask: np.ndarray, *,
    min_joint_steps: int = 5,
    low_corr_quantile: float = .01,
    high_gap_quantile: float = .99,
) -> PairIntegrityPolicy:
    """Fixed TRAIN-only quantile thresholds; no outcome labels or TEST tuning."""
    if min_joint_steps<3 or not 0<low_corr_quantile<.5 or not .5<high_gap_quantile<1:
        raise ValueError("Invalid prespecified observed-pair guard thresholds")
    n,corr,gap=pair_diagnostics(train_values,train_observed_mask)
    valid=(n>=min_joint_steps)&np.isfinite(corr)&np.isfinite(gap)
    if valid.sum()<100:
        raise ValueError("Insufficient jointly observed TRAIN episodes to set pair policy")
    corr_min=max(0.,float(np.quantile(corr[valid],low_corr_quantile)))
    gap_max=float(np.quantile(gap[valid],high_gap_quantile))
    if not np.isfinite(gap_max) or gap_max<=0:
        raise ValueError("Invalid observed-pair disagreement distribution")
    return PairIntegrityPolicy(
        first_channel=4,second_channel=5,
        min_joint_steps=min_joint_steps,
        min_pearson_corr=corr_min,
        max_mean_abs_disagreement=gap_max,
        train_subjects=len(n),
        valid_train_subjects=int(valid.sum()),
        low_corr_quantile=low_corr_quantile,
        high_gap_quantile=high_gap_quantile,
    )


def pair_is_credible(
    policy:PairIntegrityPolicy,
    values:np.ndarray,
    observed_mask:np.ndarray,
)->dict[str,np.ndarray]:
    n,corr,gap=pair_diagnostics(
        values,observed_mask,
        first_channel=policy.first_channel,
        second_channel=policy.second_channel,
    )
    has_history=n>=policy.min_joint_steps
    matched=(has_history & np.isfinite(corr) & np.isfinite(gap)
             & (corr>=policy.min_pearson_corr)
             & (gap<=policy.max_mean_abs_disagreement))
    return {
        "integrity_passed":matched,
        "enough_joint_history":has_history,
        "measured_joint_steps":n,
        "corr":corr,
        "mean_abs_gap":gap,
        "corr_passed":np.isfinite(corr)&(corr>=policy.min_pearson_corr),
        "gap_passed":np.isfinite(gap)&(gap<=policy.max_mean_abs_disagreement),
    }
