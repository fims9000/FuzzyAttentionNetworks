"""Prespecified observation-environment changes and paired risk uncertainty.

Only synthetic device readout transformations. Source labels do NOT change,
TRAIN/VALID subject membership does NOT change and no TEST outcomes enter
gate admission. VALID1 is clean; VALID2 is distinct and may be benign or
adversarial; some TEST shifts are deliberately unseen in VALID2.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score


@dataclass(frozen=True)
class EnvironmentCase:
    name: str
    outcome_law: str
    validation2_device_regime: str
    test_device_regime: str
    explanation: str


CASES = (
    EnvironmentCase(
        "physical_benign_shift","physical_extra","benign_drift","none",
        "KNOWN benign channel-gain offset at VALID2, normal TEST"
    ),
    EnvironmentCase(
        "physical_known_flip","physical_extra","flip_source","flip_source",
        "Third physical-source channels inverted in VALID2 and TEST"
    ),
    EnvironmentCase(
        "artifact_known_flip","artifact_predictive","flip_artifact","flip_artifact",
        "Recording artifact channels inverted in VALID2 and TEST"
    ),
    EnvironmentCase(
        "physical_UNSEEN_flip","physical_extra","benign_drift","flip_source",
        "TEST-only inversion not represented by any VALID cohort"
    ),
    EnvironmentCase(
        "complete_benign_shift","complete_nuisance","benign_drift","none",
        "No omitted outcome-relevant factor"
    ),
)


def changed_sensor_environment(values:np.ndarray, regime:str)->np.ndarray:
    x=np.asarray(values)
    if (x.ndim!=3 or x.shape[-1]!=12 or x.shape[1]!=16
            or not np.isfinite(x).all()):
        raise ValueError("Expected fully allocated synthetic 16-step, 12-sensor histories")
    output=x.astype(np.float32, copy=True)
    if regime=="none":
        pass
    elif regime=="benign_drift":
        output[:,:,4:12]=1.15*output[:,:,4:12]+.10
    elif regime=="flip_source":
        output[:,:,4:7]*=-1.
    elif regime=="flip_artifact":
        output[:,:,9:11]*=-1.
    else:
        raise ValueError(f"Not a registered sensor environment: {regime}")
    # The original K2 representation sees ONLY sensors0:4.
    np.testing.assert_array_equal(output[:,:,:4],x[:,:,:4])
    return output


def paired_test_ap_difference(
    y:np.ndarray,
    a:np.ndarray,
    b:np.ndarray,
    *,
    seed:int,
    resamples:int=399,
)->dict:
    """Paired patient bootstrap, same sample IDs and same 3 NN seeds.

    A and B are [fit_seed,TEST_subject]. Confidence interval conditions
    on the THREE trained models and ONE TEST generator realization.
    Do not advertise as a CI across independently trained populations.
    """
    yy=np.asarray(y,dtype=np.int64).reshape(-1)
    aa=np.asarray(a,dtype=np.float64)
    bb=np.asarray(b,dtype=np.float64)
    if (aa.ndim!=2 or aa.shape!=bb.shape or len(yy)!=aa.shape[1]
        or aa.shape[0]<1 or len(yy)<100
        or not np.isin(yy,[0,1]).all() or len(np.unique(yy))<2
        or not np.isfinite(aa).all() or not np.isfinite(bb).all()):
        raise ValueError("Expected paired, valid TEST patient predictions for each NN fit")
    if not (199<=resamples<=4999):
        raise ValueError("Unsupported frozen number of patient bootstraps")
    def ap_gap(index):
        sample=yy[index]
        if len(np.unique(sample))<2:
            raise ValueError("Insufficient positive/negative subjects in bootstrap")
        return float(np.mean([
            average_precision_score(sample,aa[s,index])-
            average_precision_score(sample,bb[s,index])
            for s in range(aa.shape[0])
        ]))
    observed=ap_gap(np.arange(len(yy)))
    rng=np.random.default_rng(seed)
    draws=np.empty(resamples,dtype=np.float64)
    for j in range(resamples):
        i=rng.integers(0,len(yy),size=len(yy))
        draws[j]=ap_gap(i)
    return {
        "difference_AUPRC_a_minus_b":observed,
        "paired_TEST_subject_bootstrap_95pct_ci":[
            float(v) for v in np.quantile(draws,[.025,.975])],
        "n_test_subjects":len(yy),
        "n_nn_fits_shared_subjects":aa.shape[0],
        "resamples":resamples,
        "condition_on_frozen_fits_and_one_test_generator":True,
    }


def case_manifest()->list[dict]:
    return [asdict(c) for c in CASES]
