"""V48 frozen monotone bounded log-odds transform of measured source membership.

This is an *algebraic link function*, not a new fuzzy-set type-reduction,
uncertainty certificate, physical source label or outcome probability.
"""
from __future__ import annotations
import numpy as np

EPS=.001
MAX_LOGIT=4.
LOGIT_DIVISOR=8.


def bounded_membership_logodds_v48(q):
    values=np.asarray(q,dtype=np.float64)
    if values.ndim!=1 or len(values)==0 or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite measured 1D concept membership")
    if (values<0).any() or (values>1).any():
        raise ValueError("Measured membership must be in [0,1]")
    bounded=np.clip(values,EPS,1-EPS)
    return np.clip(np.log(bounded/(1-bounded)),-MAX_LOGIT,MAX_LOGIT)/LOGIT_DIVISOR


def risk_concept_input_v48(q,*,link:str):
    """Returns [0,1] pseudo-membership for the exact V47 native head:
    gamma*(pseudo_membership - .5). Source provenance and original q unchanged.
    """
    values=np.asarray(q,dtype=np.float64)
    if values.ndim!=1 or not np.isfinite(values).all() or (values<0).any() or (values>1).any():
        raise ValueError("Invalid outcome-blind V44 membership")
    if link=="linear_membership":
        return values.copy()
    if link=="bounded_logodds":
        return .5+bounded_membership_logodds_v48(values)
    raise ValueError("Unregistered V48 source-risk link")


def describe_source_link_v48(link:str)->dict:
    if link not in ("linear_membership","bounded_logodds"):
        raise ValueError("Unknown frozen concept-risk relation")
    return {
        "name":link,
        "algebraic_semantics":(
            "gamma*(q-0.5)" if link=="linear_membership" else
            "gamma*clip(logit(clip(q,.001,.999)),-4,4)/8"),
        "source_q_formed_without_outcome_label_Y":True,
        "preserves_signed_additive_decomposition":True,
        "no_oracle_source_or_raw_sensor_risk_bypass":True,
        "not_physical_identity_or_uncertainty_certificate":True,
    }
