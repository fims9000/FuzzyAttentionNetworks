"""V46 measured ON/OFF corroboration condition; existing instruments only.

This is unsupervised change detection, not source identification.
A large matched ON+OFF background need not be a faulty difference.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
import numpy as np
from .off_reference_reliability_v45 import measured_off_dynamic_energy
from .adaptive_pair_bench_world_v41 import measured_difference_v41

MIN_PAIRED_TIMES=4
MIN_PAIRED_CHANNELS=4
TRAIN_MIN=300
MIN_VALID_SUPPORT=.90
MAX_VALID_FALSE_ALARM=.05


def measured_paired_mismatch_v46(on,off,mask_on,mask_off):
    """Variances of TRUE paired ON-OFF readings, never synthesize missing data."""
    difference,observed=measured_difference_v41(on,off,mask_on,mask_off)
    a=np.asarray(difference,dtype=np.float64)[:,:,4:10]
    m=np.asarray(observed,dtype=bool)[:,:,4:10]
    counts=m.sum(axis=1)
    eligible=counts>=MIN_PAIRED_TIMES
    supported=eligible.sum(axis=1)>=MIN_PAIRED_CHANNELS
    avg=np.where(m,a,0.).sum(axis=1)/np.maximum(counts,1)
    diff=np.where(m,a-avg[:,None,:],0.)
    variance=(diff*diff).sum(axis=1)/np.maximum(counts-1,1)
    score=(np.where(eligible,variance,0.).sum(axis=1)/
           np.maximum(eligible.sum(axis=1),1))
    return {
        "score":np.where(supported,score,0.),
        "support":supported,
        "n_paired_channels":eligible.sum(axis=1),
        "min_actual_times_among_supported":np.min(np.where(eligible,counts,16),axis=1),
        "outcome_and_hidden_source_values_used":False,
        "no_imputation":True,
    }


def _observations(on,off,mask_on,mask_off):
    off_energy=measured_off_dynamic_energy(off,mask_off)
    mismatch=measured_paired_mismatch_v46(on,off,mask_on,mask_off)
    supported=off_energy["support"]&mismatch["support"]
    return off_energy,mismatch,supported


@dataclass(frozen=True)
class MeasuredONOFFConditionV46:
    off_train_limit:float
    diff_train_limit:float
    n_train_sufficient:int
    n_train:int
    valid1_supported_fraction:float
    valid2_supported_fraction:float
    valid1_joint_alarm_fraction:float
    valid2_joint_alarm_fraction:float
    approved_two_healthy_VALID:bool
    uses_outcomes_Y:bool=False
    uses_hidden_H_S_U_E:bool=False
    physical_concept_identity_certified:bool=False

    def audit(self):
        return asdict(self)


def fit_paired_condition_v46(
    tr_on,tr_off,tr_mon,tr_moff,
    v1_on,v1_off,v1_mon,v1_moff,
    v2_on,v2_off,v2_mon,v2_moff,
):
    """TRAIN(.99 OFF, .95 DIFF) and disjoint VALID false alarm <=5%."""
    a,b,support=_observations(tr_on,tr_off,tr_mon,tr_moff)
    if len(support)<TRAIN_MIN or float(support.mean())<MIN_VALID_SUPPORT:
        raise ValueError("Insufficient TRAIN paired ON/OFF measurements")
    off_limit=float(np.quantile(a["score"][support],.99))
    diff_limit=float(np.quantile(b["score"][support],.95))
    if not (np.isfinite(off_limit) and np.isfinite(diff_limit) and
            off_limit>0 and diff_limit>0):
        raise ValueError("Cannot certify pair-monitor TRAIN calibration")
    validations=[]
    for on,off,m_on,m_off in (
        (v1_on,v1_off,v1_mon,v1_moff),
        (v2_on,v2_off,v2_mon,v2_moff),
    ):
        va,vb,seen=_observations(on,off,m_on,m_off)
        fpr=(float(np.mean((va["score"][seen]>off_limit)&
                            (vb["score"][seen]>diff_limit))) if seen.any() else 1.)
        validations.append((float(seen.mean()),fpr))
    approved=all(coverage>=MIN_VALID_SUPPORT and fpr<=MAX_VALID_FALSE_ALARM
                 for coverage,fpr in validations)
    return MeasuredONOFFConditionV46(
        off_train_limit=off_limit,diff_train_limit=diff_limit,
        n_train_sufficient=int(support.sum()),n_train=len(support),
        valid1_supported_fraction=validations[0][0],
        valid2_supported_fraction=validations[1][0],
        valid1_joint_alarm_fraction=validations[0][1],
        valid2_joint_alarm_fraction=validations[1][1],
        approved_two_healthy_VALID=approved,
    )


def apply_paired_condition_v46(fit:MeasuredONOFFConditionV46,
                               on,off,mask_on,mask_off):
    a,b,support=_observations(on,off,mask_on,mask_off)
    off_alarm=support&(a["score"]>fit.off_train_limit)
    diff_alarm=support&(b["score"]>fit.diff_train_limit)
    alarm=off_alarm&diff_alarm
    passed=support&~alarm&fit.approved_two_healthy_VALID
    reason=np.where(~support,"MISSING_REAL_ON_OFF_PAIR",
                    np.where(~fit.approved_two_healthy_VALID,
                             "NO_TWO_HEALTHY_VALID_CERTIFICATES",
                             np.where(alarm,"JOINT_ON_OFF_CONDITION_ANOMALY",
                                      "MEASURED_ON_OFF_CONDITION_SUPPORTED")))
    return {
        "usable_for_additional_concept":passed,
        "off_background_alarm":off_alarm,
        "diff_discrepancy_alarm":diff_alarm,
        "joint_condition_alarm":alarm,
        "condition_support":support,
        "score":a["score"],
        "difference_score":b["score"],
        "decision_reason":reason,
        "monitor_saw_outcomes_Y":False,
        "physical_semantics_certified":False,
    }
