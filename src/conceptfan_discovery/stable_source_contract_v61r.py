"""V61R: outcome-blind, deterministic limited conditional sensor dictionary.

This is a *posthoc engineering repair*, not a new independent scientific
confirmation. V61's gamma-ranked top-six was numerically unstable under
minuscule Ridge residualizer changes; V61R shortlists <=6 programs using
TRAIN measured X alone, with source-family diversity and conditional
residual dispersion, before reading any outcome Y for task training.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass,replace
import hashlib
import json
import numpy as np

from .conditional_innovation_v61 import (
    ConditionalSourceV61,infer_conditional_sources_v61,
)

MAX_FROZEN_SOURCES=6
REDUNDANCY_LIMIT=.95


@dataclass(frozen=True)
class StableSourceDictionaryV61R:
    selected_raw_indices:tuple[int,...]
    selected_candidate_ids:tuple[str,...]
    selected_operator_names:tuple[str,...]
    selected_channel_groups:tuple[tuple[int,...],...]
    total_conditional_pool:int
    shortlist_limit:int
    selected_residual_min_train_fold_std:tuple[float,...]
    selection_used_train_Y:bool=False
    selection_used_test_Y:bool=False
    independent_physical_identity_proven:bool=False

    def passport(self):
        return asdict(self)


def _fingerprint(d):
    return hashlib.sha256(json.dumps(
        d,sort_keys=True,separators=(",",":"),
        ensure_ascii=True).encode()).hexdigest()


def shortlist_conditional_sources_v61r(full:ConditionalSourceV61,
                                      bank,train_values,train_support,
                                      train_raw_mask,train_subject_groups,
                                      *,limit=MAX_FROZEN_SOURCES):
    if not (1<=limit<=MAX_FROZEN_SOURCES):
        raise ValueError("Frozen V61R shortlist must contain at most six sources")
    if full is None:return None,None
    z,s=infer_conditional_sources_v61(
        full,train_values,train_support,train_raw_mask)
    labels=np.asarray(train_subject_groups)
    if labels.shape!=(len(z),):
        raise ValueError("TRAIN subject groups required for canonical dictionary")
    if not z.shape[1]:
        return None,None
    # Candidate scores depend ONLY on observed TRAIN masks/values.
    descriptors=[]
    for j,index in enumerate(full.raw_source_indices):
        c=bank.candidates[index]
        if 0 in c.channels:
            raise AssertionError("Novel dictionary source reads PULSE")
        observed=s[:,j]>=.5
        coverage=float(observed.mean())
        fold_min=float(min(full.train_fold_std[j]))
        # Dispersion is measured on already TRAIN-scaled residuals.
        # It is NOT a supervised metric or model importance.
        quality=fold_min*coverage
        descriptors.append({
            "j":j,"raw":int(index),
            "family":tuple(c.channels),
            "operator":c.operator,"id":c.candidate_id,
            "quality":float(quality),"min_fold_std":fold_min,
        })
    descriptors.sort(key=lambda c:(-round(c["quality"],9),c["id"]))
    selected=[];used_families=set()
    # First ensure representation of each *distinct* non-PULSE channel
    # combination if observed and non-redundant: avoids a single sensor
    # consuming all six slots due to near-tied task gamma weights.
    def distinct(candidate):
        j=candidate["j"]
        mask=s[:,j]>=.5
        for other in selected:
            k=other["j"]
            common=mask&(s[:,k]>=.5)
            if int(common.sum())<100:continue
            a=z[common,j];b=z[common,k]
            if min(a.std(),b.std())<1e-10:continue
            if abs(float(np.corrcoef(a,b)[0,1]))>REDUNDANCY_LIMIT:
                return False
        return True
    for candidate in descriptors:
        if (candidate["family"] not in used_families
            and distinct(candidate)):
            selected.append(candidate)
            used_families.add(candidate["family"])
            if len(selected)==limit:break
    # Fill remainder with independent measured expressions ordered by
    # TRAIN-only residual support/variation, never target risk or gamma.
    for candidate in descriptors:
        if len(selected)>=limit:break
        if candidate in selected:continue
        if distinct(candidate):selected.append(candidate)
    if not selected:return None,None
    # Stable program order: original registered grammar order. This is
    # independent of numerical ranking and frozen across site inference.
    positions=tuple(sorted(item["j"] for item in selected))
    def pick(field):
        return tuple(getattr(full,field)[j] for j in positions)
    final=replace(
        full,
        raw_source_indices=pick("raw_source_indices"),
        coefficients=pick("coefficients"),
        intercepts=pick("intercepts"),
        train_medians=pick("train_medians"),
        train_iqrs=pick("train_iqrs"),
        train_residual_r2=pick("train_residual_r2"),
        train_fold_std=pick("train_fold_std"),
    )
    dictionary=StableSourceDictionaryV61R(
        selected_raw_indices=final.raw_source_indices,
        selected_candidate_ids=tuple(
            bank.candidates[k].candidate_id for k in final.raw_source_indices),
        selected_operator_names=tuple(
            bank.candidates[k].operator for k in final.raw_source_indices),
        selected_channel_groups=tuple(
            bank.candidates[k].channels for k in final.raw_source_indices),
        total_conditional_pool=len(full.raw_source_indices),
        shortlist_limit=limit,
        selected_residual_min_train_fold_std=tuple(
            min(x) for x in final.train_fold_std),
    )
    return final,dictionary


def source_contract_fingerprint_v61r(source,dictionary):
    if source is None or dictionary is None:
        return "NO_CONDITIONALLY_OBSERVED_NEW_SOURCE"
    return _fingerprint({
        "source":source.passport(),
        "dictionary":dictionary.passport(),
    })
