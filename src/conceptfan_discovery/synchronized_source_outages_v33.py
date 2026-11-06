"""V33: explicit synchronized source loss and a conservative evidence gate.

ALL sensor indices in a synthetic source group may share the same outage.
The simulator-defined group is used ONLY to construct TEST stress masks,
never to form TRAIN proposals, source templates or risk coefficients.

Phase corroboration is a deterministic measured-coverage criterion:
source must be actually observed at >=2 of 8 time positions in BOTH halves.
It never imputes hidden readings and returns exact neutral candidate upon
failure, even if the older rank-1 WLS template could extrapolate a score.
This is a conservative heuristic, NOT a clinical robustness certificate.
"""
from __future__ import annotations

from hashlib import sha256

import numpy as np

from conceptfan_discovery.masked_temporal_template import (
    MaskedRank1Template,project_masked_template,
)
from conceptfan_discovery.auto_measured_concept_proposals import (
    MeasuredGroup,
)
from conceptfan_discovery.phase_pulse_benchmark_v31 import (
    measured_mask,
)


STRESS_V33=(
    "complete","MCAR20","MCAR40","MNAR","independent_block8",
    "synchronized_early8","synchronized_late8",
    "synchronized_block8","synchronized_full16",
)
SIMULATOR_PHYSICAL_SENSOR_GROUP=(4,5,6)


def synchronized_outage_mask(
    x:np.ndarray,
    mode:str,
    *,
    seed:int,
    simulator_stress_only_indices:tuple[int,...]=SIMULATOR_PHYSICAL_SENSOR_GROUP,
)->np.ndarray:
    """Construct fixed TEST-only stress; no Y or hidden source state.

    Pre-existing independent_block8 is the V32 mask, not the synchronized
    case. All four new scenarios retain MCAR20 on unrelated sensors.
    """
    xx=np.asarray(x)
    if xx.ndim!=3 or xx.shape[1:]!=(16,12) or not np.isfinite(xx).all():
        raise ValueError("Expected 16 timepoints of 12 finite measured channels")
    idx=tuple(int(j) for j in simulator_stress_only_indices)
    if not idx or len(set(idx))!=len(idx) or min(idx)<0 or max(idx)>=12:
        raise ValueError("Invalid synthetic TEST fault target source group")
    if mode=="independent_block8":
        return measured_mask(xx,"block8",seed=seed)
    if mode in ("complete","MCAR20","MCAR40","MNAR"):
        return measured_mask(xx,mode,seed=seed)
    if mode not in STRESS_V33:
        raise ValueError("Unregistered source dropout stress")
    observed=measured_mask(xx,"MCAR20",seed=seed)
    if mode=="synchronized_full16":
        observed[:,:,idx]=False
    elif mode=="synchronized_early8":
        observed[:,:8,idx]=False
    elif mode=="synchronized_late8":
        observed[:,8:,idx]=False
    elif mode=="synchronized_block8":
        starts=np.random.default_rng(seed+11021).integers(
            0,9,size=len(xx))
        for i,first in enumerate(starts):
            observed[i,first:first+8,idx]=False
    if observed.shape!=xx.shape:
        raise AssertionError("Synchronized mask lost source alignment")
    return observed


def phase_corroborated_template(
    template:MaskedRank1Template,
    group:MeasuredGroup,
    x:np.ndarray,
    mask:np.ndarray,
    *,
    minimum_observed_positions_per_half:int=2,
)->dict:
    """Only count ACTUALLY observed source times in each temporal phase."""
    if minimum_observed_positions_per_half!=2:
        raise ValueError("V33 fixed phase corroboration threshold is 2/8")
    core=project_masked_template(
        template,group,x,mask,min_information_fraction=.50)
    observed=np.asarray(mask)
    if observed.shape!=np.asarray(x).shape or not np.isin(observed,[0,1]).all():
        raise ValueError("Observation mask must align to source histories")
    per_time=observed[:,:,group.sensor_indices].any(axis=2)
    early=per_time[:,:8].sum(axis=1)
    late=per_time[:,8:].sum(axis=1)
    sufficient=(
        core["sufficient_observations"] &
        (early>=minimum_observed_positions_per_half) &
        (late>=minimum_observed_positions_per_half)
    )
    candidate=np.where(sufficient,core["q"],.5)
    support=np.where(sufficient,core["source_support"],0.)
    return {
        "q":candidate,
        "source_support":support,
        "measured_time_positions_early":early,
        "measured_time_positions_late":late,
        "phase_corroboration_passed":sufficient,
        "wls_base_eligible":core["sufficient_observations"],
        "WLS_information_fraction":core[
            "observed_template_information_fraction"],
        "source_temporal_template_sha256":core[
            "source_temporal_template_sha256"],
        "no_unsupported_missing_values_read":True,
        "no_outcomes_or_hidden_sources_used":True,
    }


def phase_concept(
    template:MaskedRank1Template,
    group:MeasuredGroup,
    x:np.ndarray,
    mask:np.ndarray,
)->tuple[np.ndarray,np.ndarray]:
    z=phase_corroborated_template(template,group,x,mask)
    return z["q"],z["source_support"]


def observed_payload_digest(x:np.ndarray,mask:np.ndarray)->str:
    """Hash observed payload and mask only, not withheld hidden values."""
    xx=np.asarray(x)
    mm=np.asarray(mask).astype(bool)
    if xx.shape!=mm.shape or not np.isfinite(xx[mm]).all():
        raise ValueError("Invalid observed payload")
    values=np.asarray(xx[mm],dtype=np.float64)
    return sha256(
        np.asarray(mm,dtype=np.uint8).tobytes()+values.tobytes()
    ).hexdigest()
