"""Pilot-only temporal coverage selection for clinical missingness-aware rank.

Choose a SINGLE COMMON sensor dictionary using ONLY a disjoint subset
of TRAIN patients' observation masks. The actual rank estimation remains
on other TRAIN patients, with two independently audited halves. Never
choose sensors or hour binning because their estimated rank looks good.

Per-hour joint support is required under actual within-patient pairing
AND expected whole-patient-shuffle null pairing. An independently
thinned observation environment is prospectively accommodated by
inflating the pilot minimum count using a prespecified worst retention
probability. Selection has no outcome labels, measurement magnitudes,
validation, calibration, test or latent oracle.
"""
from __future__ import annotations

import math
import numpy as np


def time_pair_support(
    observed_mask:np.ndarray,
    *,
    lags:tuple[int,...]=(1,2),
    min_observed_pairs:int=12,
    min_permuted_expected_pairs:float=20.,
)->dict:
    m=np.asarray(observed_mask,dtype=np.float64)
    if m.ndim!=3 or m.shape[0]<40 or m.shape[2]<2 or not np.isfinite(m).all():
        raise ValueError("Need finite masks from >=40 pilot patients")
    if not np.logical_or(m==0,m==1).all():
        raise ValueError("Only binary observed masks accepted")
    if not lags or min(lags)<1 or max(lags)>=m.shape[1] or len(set(lags))!=len(lags):
        raise ValueError("Distinct legal positive time lags required")
    if min_observed_pairs<4 or min_permuted_expected_pairs<min_observed_pairs:
        raise ValueError("Expected null overlap must exceed positive observed minimum")
    n,t,d=m.shape
    coverage=[]
    observed_min=[]
    null_min=[]
    for lag in lags:
        time_usable=np.zeros((d,d),dtype=np.int32)
        original_min=np.full((d,d),np.inf)
        expected_min=np.full((d,d),np.inf)
        for hour in range(t-lag):
            past=m[:,hour,:]
            future=m[:,hour+lag,:]
            actual=past.T@future
            expected=np.outer(past.sum(axis=0),future.sum(axis=0))/n
            adequate=(actual>=min_observed_pairs)&(expected>=min_permuted_expected_pairs)
            time_usable+=adequate.astype(np.int32)
            original_min=np.minimum(original_min,actual)
            expected_min=np.minimum(expected_min,expected)
        coverage.append(time_usable)
        observed_min.append(original_min)
        null_min.append(expected_min)
    return {
        "time_bins_with_adequate_both_processes":np.stack(coverage),
        "min_real_pairs_any_bin":np.stack(observed_min),
        "min_expected_null_pairs_any_bin":np.stack(null_min),
    }


def choose_common_temporal_sensors(
    pilot_mask:np.ndarray,
    channel_names:list[str],
    *,
    lags:tuple[int,...]=(1,2),
    minimum_supported_strata:int=3,
    minimum_observed_pairs_per_stratum:int=12,
    minimum_expected_null_pairs_per_stratum:float=20.,
    worst_additional_observation_retention:float=.65,
)->dict:
    if not .15<=worst_additional_observation_retention<=1.:
        raise ValueError("Unrealistic or unannounced retention probability")
    m=np.asarray(pilot_mask)
    if m.ndim!=3 or m.shape[2]!=len(channel_names):
        raise ValueError("Pilot masks and observable channel order must agree")
    if minimum_supported_strata<2 or minimum_supported_strata>=m.shape[1]:
        raise ValueError("Unsupported source-hour strata minimum")
    actual_requirement=int(math.ceil(
        minimum_observed_pairs_per_stratum/
        worst_additional_observation_retention**2
    ))
    null_requirement=max(minimum_expected_null_pairs_per_stratum,
                         float(actual_requirement))
    support=time_pair_support(
        m,lags=lags,min_observed_pairs=actual_requirement,
        min_permuted_expected_pairs=null_requirement,
    )["time_bins_with_adequate_both_processes"]
    minimum_by_lag=support.min(axis=0)
    own=np.diag(minimum_by_lag)
    candidates=[
        int(i) for i in np.argsort(-own,kind="stable")
        if own[i]>=minimum_supported_strata
    ]
    chosen=[]
    for candidate in candidates:
        if all(
            minimum_by_lag[candidate,prior]>=minimum_supported_strata
            and minimum_by_lag[prior,candidate]>=minimum_supported_strata
            for prior in chosen
        ):
            chosen.append(candidate)
    adequate=len(chosen)>=2
    return {
        "status":"PILOT_SENSORS_ELIGIBLE" if adequate
                 else "ABSTAIN_INSUFFICIENT_PILOT_TEMPORAL_SUPPORT",
        "selected_indices":chosen,
        "selected_names":[str(channel_names[i]) for i in chosen],
        "excluded_names":[str(channel_names[i]) for i in range(len(channel_names))
                          if i not in chosen],
        "pilot_n":len(m),
        "positive_lags":list(lags),
        "worst_retention_probability_prespecified":float(
            worst_additional_observation_retention
        ),
        "pilot_min_original_pair_count":int(actual_requirement),
        "pilot_min_expected_null_pair_count":float(null_requirement),
        "min_required_distinct_time_strata":int(minimum_supported_strata),
        "min_eligible_strata_within_selected":
            int(minimum_by_lag[np.ix_(chosen,chosen)].min()) if chosen else 0,
        "all_selected_sensor_pairs_adequate_in_pilot":bool(
            adequate and (
                minimum_by_lag[np.ix_(chosen,chosen)]>=minimum_supported_strata
            ).all()
        ),
        "outcomes_or_measurement_values_used":False,
        "not_claimed":"informative missingness controlled or physiological factors identified",
    }
