"""V21c: TRAIN-disjoint paired measured repeatability after instrument reset.

This is a measurement provenance screening test, not physical truth. A
common-mode environmental artifact can remain highly reproducible even
after nominal hardware reset; the frozen persistent-artifact scenario is
an explicit falsification of overclaiming this test.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
from typing import Iterable

import numpy as np

from .candidate_admission_v17 import ProposedObservableConcept
from .instrument_reset_data_v21c import PairedMeasuredAcquisitions


@dataclass(frozen=True)
class PairedSourceRepeatability:
    candidate_id: str
    status: str
    n_independent_calibration_episodes: int
    first_repeat_pearson_r: float
    permutation_p_two_sided: float
    bootstrap_ci95: tuple[float,float]
    paired_two_half_correlations: tuple[float,float]
    before_reference_signed_r: float
    after_reference_signed_r: float
    external_reference_repeat_r: float
    acquisition_reset_declared: bool
    physical_causal_identity_established: bool = False
    persistent_commonmode_interference_ruled_out: bool = False


def _pearson(a,b):
    if float(np.std(a))<1e-10 or float(np.std(b))<1e-10:
        return 0.
    return float(np.mean((a-a.mean())*(b-b.mean()))/(a.std()*b.std()))


def assess_candidate_after_independent_acquisition_reset(
    candidate:ProposedObservableConcept,
    pairs:PairedMeasuredAcquisitions,
    *,
    discovery_episode_ids:Iterable[int],
    outcome_screen_episode_ids:Iterable[int],
    reference_screen_episode_ids:Iterable[int],
    seed:int=211144,
    window:int=6,
    permutations:int=199,
    bootstraps:int=199,
    minimum_r:float=.65,
    bootstrap_lower_r:float=.50,
    two_half_min_r:float=.45,
)->PairedSourceRepeatability:
    if not isinstance(candidate,ProposedObservableConcept) or not isinstance(pairs,PairedMeasuredAcquisitions):
        raise ValueError("Expected V17 TRAIN-only candidate and controlled paired measurements")
    before=np.asarray(pairs.first_model_visible_measurements,dtype=np.float64)
    after=np.asarray(pairs.repeated_model_visible_measurements,dtype=np.float64)
    r1=np.asarray(pairs.first_external_reference_measurement,dtype=np.float64)
    r2=np.asarray(pairs.repeated_external_reference_measurement,dtype=np.float64)
    first_mask=np.asarray(pairs.first_recorded_mask)
    after_mask=np.asarray(pairs.repeat_recorded_mask)
    ids=[int(i) for i in pairs.calibration_episode_ids]
    if (before.ndim!=3 or before.shape!=after.shape or before.shape!=first_mask.shape
            or after.shape!=after_mask.shape or r1.shape!=before.shape[:2]
            or r2.shape!=before.shape[:2] or candidate.direction.shape!=(before.shape[-1],)
            or len(ids)!=before.shape[0] or before.shape[0]<180
            or not 2<=window<=before.shape[1]
            or not np.isfinite(before).all() or not np.isfinite(after).all()
            or not np.isfinite(r1).all() or not np.isfinite(r2).all()
            or not np.array_equal(first_mask,np.ones_like(first_mask))
            or not np.array_equal(after_mask,np.ones_like(after_mask))):
        raise ValueError("ABSTAIN: invalid complete paired acquisition measurements")
    d=set(map(int,discovery_episode_ids))
    o=set(map(int,outcome_screen_episode_ids))
    r=set(map(int,reference_screen_episode_ids))
    if (not d or not o or not r or len(ids)!=len(set(ids)) or
            set(ids)&(d|o|r) or d&o or d&r or o&r):
        raise ValueError("ABSTAIN: independent calibration/source/risk/reference cohorts must be disjoint")
    if (not pairs.controlled_instrument_reset_declared or
            not pairs.same_physical_system_held_fixed_in_simulator or
            not pairs.no_generator_latent_states_or_outcomes_exposed or
            pairs.patient_histories_from_train_validation_and_test_reused):
        raise ValueError("ABSTAIN: unverified controlled reset / source-provenance protocol")
    if (not 0<two_half_min_r<bootstrap_lower_r<minimum_r<=.99
            or permutations<99 or bootstraps<99):
        raise ValueError("Invalid preregistered repeatability thresholds")
    score_before=(before[:,-window:]@candidate.direction).mean(1)
    score_after=(after[:,-window:]@candidate.direction).mean(1)
    ref_before=r1[:,-window:].mean(1)
    ref_after=r2[:,-window:].mean(1)
    if score_before.std()<1e-8 or score_after.std()<1e-8:
        return PairedSourceRepeatability(
            candidate.candidate_id,
            "ABSTAIN_UNMEASURABLE_PAIRED_SOURCE",
            len(ids),0.,1.,(0.,0.),(0.,0.),
            _pearson(score_before,ref_before),
            _pearson(score_after,ref_after),
            _pearson(ref_before,ref_after),
            True,
        )
    observed=_pearson(score_before,score_after)
    rng=np.random.default_rng(seed)
    null=np.asarray([
        abs(_pearson(score_before,rng.permutation(score_after)))
        for _ in range(permutations)
    ])
    p=float((1+int(np.count_nonzero(null>=abs(observed))))/(1+permutations))
    bootstrap=np.asarray([
        _pearson(score_before[idx],score_after[idx])
        for idx in (rng.integers(0,len(ids),size=len(ids)) for _ in range(bootstraps))
    ])
    lower,upper=map(float,np.quantile(bootstrap,[.025,.975]))
    # Split halves are prepared with one frozen random ordering; no adaptive
    # selection of the better split or patient-signed realignment.
    perm=rng.permutation(len(ids))
    half_a,half_b=np.array_split(perm,2)
    halves=(
        _pearson(score_before[half_a],score_after[half_a]),
        _pearson(score_before[half_b],score_after[half_b]),
    )
    accepted=(observed>=minimum_r and p<=.01 and
              lower>bootstrap_lower_r and all(h>=two_half_min_r for h in halves))
    return PairedSourceRepeatability(
        candidate_id=candidate.candidate_id,
        status=("ACCEPT_ACQUISITION_REPEATABILITY" if accepted
                else "ABSTAIN_UNSTABLE_UNDER_INSTRUMENT_RESET"),
        n_independent_calibration_episodes=len(ids),
        first_repeat_pearson_r=observed,
        permutation_p_two_sided=p,
        bootstrap_ci95=(lower,upper),
        paired_two_half_correlations=tuple(map(float,halves)),
        before_reference_signed_r=_pearson(score_before,ref_before),
        after_reference_signed_r=_pearson(score_after,ref_after),
        external_reference_repeat_r=_pearson(ref_before,ref_after),
        acquisition_reset_declared=True,
    )


def paired_repeatability_passport(value:PairedSourceRepeatability)->dict:
    return {
        **asdict(value),
        "measured_acquisition_only_no_hidden_oracle":True,
        "outcome_labels_entered_paired_test":False,
        "clinical_physical_identity_certified":False,
        "persistent_common_mode_can_spoof_paired_repeat":True,
        "patient_level_paired_measurements_exported":False,
    }
