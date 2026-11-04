"""V21c: paired repeat-acquisition source diagnostic with instrument reset.

Purely synthetic controlled experiment. Original same underlying physical
system is observed twice with separately reinitialized acquisition devices.
The persistent-environmental-interference control deliberately violates
the assumption that resetting the instruments removes all artifacts.

Only paired observed measurements are exposed. This data object's API does
not return latent states or outcome labels. Those are generator internals,
not allowed to enter source proposal or repetition screening.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .provenance_data_v21 import (
    _simulate,make_v21_provenance_cohort,
    FAMILIES,HISTORY,HORIZON,
)

RESET_FAMILIES=(*FAMILIES,"artifact_environmental_persistent")
CALIBRATION_SEED=405111
INDEPENDENT_RESET_SEED=405112
CALIBRATION_EPISODES=240
SENSOR_COUNT=4
CALIBRATION_START_ID=40000


@dataclass(frozen=True)
class PairedMeasuredAcquisitions:
    scenario: str
    source_training_family: str
    first_model_visible_measurements: np.ndarray
    repeated_model_visible_measurements: np.ndarray
    first_external_reference_measurement: np.ndarray
    repeated_external_reference_measurement: np.ndarray
    calibration_episode_ids: np.ndarray
    first_recorded_mask: np.ndarray
    repeat_recorded_mask: np.ndarray
    controlled_instrument_reset_declared: bool
    same_physical_system_held_fixed_in_simulator: bool
    no_generator_latent_states_or_outcomes_exposed: bool
    patient_histories_from_train_validation_and_test_reused: bool


def _new_independent_colored_instrument(
    rng:np.random.Generator,n:int,t:int,
)->np.ndarray:
    raw=np.zeros((n,t),dtype=np.float64)
    for step in range(t):
        innovation=.43*rng.normal(size=n)
        raw[:,step]=innovation if step==0 else .92*raw[:,step-1]+innovation
    return raw


def make_v21c_repeat_acquisitions(
    scenario:str,
)->PairedMeasuredAcquisitions:
    if scenario not in RESET_FAMILIES:
        raise ValueError("Unregistered V21c paired calibration scenario")
    family=(
        "artifact_common_mode_reference"
        if scenario=="artifact_environmental_persistent" else scenario
    )
    raw,reference,labels,physical_source,instrument=_simulate(
        family,CALIBRATION_SEED,CALIBRATION_EPISODES,
    )
    # Retain the same physiological process across these two simulated
    # acquisitions. Entirely separate measurement noise and, except for
    # the persistent-environmental-control, independently reset artifact.
    n,t,d=raw.shape
    if n!=CALIBRATION_EPISODES or t!=HISTORY+HORIZON or d!=SENSOR_COUNT:
        raise AssertionError("Independent acquisition simulator dimensions changed")
    rng=np.random.default_rng(INDEPENDENT_RESET_SEED)
    new_artifact=_new_independent_colored_instrument(rng,n,t)
    if scenario=="artifact_environmental_persistent":
        # Hardware was reset, yet environmental interference survives:
        # repeated device readings are still contaminated.
        new_artifact=instrument.copy()
    repeated=raw.copy()
    repeated[:,:,:2]=raw[:,:,:2]+rng.normal(scale=.08,size=(n,t,2))
    if family.startswith("physical_"):
        repeated[:,:,2]=1.02*physical_source+rng.normal(scale=.11,size=(n,t))
        repeated[:,:,3]=rng.normal(scale=.20,size=(n,t))
    else:
        repeated[:,:,2]=rng.normal(scale=.20,size=(n,t))
        repeated[:,:,3]=1.08*new_artifact+rng.normal(scale=.11,size=(n,t))
    if family in ("artifact_common_mode_reference",):
        repeated_reference=(
            .95*new_artifact+.06*physical_source
            +rng.normal(scale=.13,size=(n,t))
        )
    else:
        repeated_reference=(
            .92*physical_source+rng.normal(scale=.13,size=(n,t))
        )

    # Critically, both repeated acquisitions use ORIGINAL source-family
    # TRAIN medians and scales; calibration does not fit scalers or leak
    # source/oracle supervision through normalization.
    original_train=make_v21_provenance_cohort(family).discovery
    loc=original_train.medians
    scale=original_train.scales
    first=np.clip((raw[:,:HISTORY]-loc)/scale,-6,6).astype(np.float32)
    second=np.clip((repeated[:,:HISTORY]-loc)/scale,-6,6).astype(np.float32)
    before_reference=reference[:,:HISTORY].astype(np.float32)
    after_reference=repeated_reference[:,:HISTORY].astype(np.float32)
    if not (
        np.isfinite(first).all() and np.isfinite(second).all()
        and np.isfinite(before_reference).all()
        and np.isfinite(after_reference).all()
    ):
        raise ValueError("Calibration observations must be finite")
    # Deliberately never return labels, hidden physical source, or
    # hidden instrumental source.
    del labels,physical_source,instrument,raw,repeated
    return PairedMeasuredAcquisitions(
        scenario=scenario,source_training_family=family,
        first_model_visible_measurements=first,
        repeated_model_visible_measurements=second,
        first_external_reference_measurement=before_reference,
        repeated_external_reference_measurement=after_reference,
        calibration_episode_ids=np.arange(
            CALIBRATION_START_ID,CALIBRATION_START_ID+n,dtype=np.int64,
        ),
        first_recorded_mask=np.ones_like(first,dtype=np.float32),
        repeat_recorded_mask=np.ones_like(second,dtype=np.float32),
        controlled_instrument_reset_declared=True,
        same_physical_system_held_fixed_in_simulator=True,
        no_generator_latent_states_or_outcomes_exposed=True,
        patient_histories_from_train_validation_and_test_reused=False,
    )
