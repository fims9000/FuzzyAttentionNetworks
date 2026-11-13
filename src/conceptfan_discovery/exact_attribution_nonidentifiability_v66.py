"""V66 exact bounded-concept equivalence: formal counterexample to semantic identity.

Exact risk additivity, bounded fuzzy memberships and unchanged predictions do
not by themselves determine uniquely meaningful concept coordinates. This
does NOT refute identifiability under stronger physical/ICA/interventional
assumptions that explicitly rule out cross-concept mixing.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class EquivalenceAudit:
    k: int
    n: int
    alpha: float
    mixing_eigenvalue_orthogonal: float
    matrix_condition_number: float
    max_logit_difference: float
    max_probability_difference: float
    max_signed_contribution_sum_difference: float
    max_fuzzy_membership_difference: float
    bounded_state_violations: int
    fraction_changed_top_abs_contribution_axis: float
    individual_attribution_mean_abs_difference: float
    semantics_identified: bool = False

    def as_dict(self):
        from dataclasses import asdict
        return asdict(self)


def bounded_mixing_matrix(k: int, alpha: float):
    """Invertible nonnegative doubly stochastic concept mixing, K>=2."""
    if not isinstance(k,int) or k<2:
        raise ValueError("A nontrivial semantic mixing counterexample requires K>=2")
    if not np.isfinite(alpha) or not 0.0<alpha<1.0:
        raise ValueError("Mixing strength must be strictly between zero and one")
    return (1.-alpha)*np.eye(k)+alpha*np.ones((k,k))/k


def equivalent_concept_head(concepts, beta, bias, *, alpha:float=.35):
    """Return risk-exact new [0,1] concepts and signed coefficients.

    Let A=(1-alpha)I+alpha * J/K. Then A1=1, A>=0 and A^-1 exists.
    Therefore (C A-0.5) @ A^-1 beta == (C-0.5) @ beta for all C.
    """
    c=np.asarray(concepts,dtype=np.float64)
    weights=np.asarray(beta,dtype=np.float64)
    if c.ndim!=2 or len(c)<1 or weights.shape!=(c.shape[1],):
        raise ValueError("Expected bounded [N,K] memberships and K signed weights")
    if not np.isfinite(c).all() or not np.isfinite(weights).all():
        raise ValueError("Nonfinite fuzzy memberships or signed coefficients")
    if (c<0).any() or (c>1).any():
        raise ValueError("The counterexample assumes fuzzy membership bounds [0,1]")
    if not np.isfinite(bias):
        raise ValueError("Nonfinite source-only risk intercept")
    a=bounded_mixing_matrix(c.shape[1],float(alpha))
    changed=c@a
    transformed_beta=np.linalg.solve(a,weights)
    original_parts=(c-.5)*weights
    updated_parts=(changed-.5)*transformed_beta
    raw_logits=float(bias)+original_parts.sum(axis=1)
    same_logits=float(bias)+updated_parts.sum(axis=1)
    if np.max(np.abs(raw_logits-same_logits))>1e-9:
        raise AssertionError("Bounded remixed concepts unexpectedly changed risk logits")
    return {
        "matrix":a,
        "concepts":changed,
        "signed_weights":transformed_beta,
        "bias":float(bias),
        "original_parts":original_parts,
        "remixed_parts":updated_parts,
        "original_logits":raw_logits,
        "remixed_logits":same_logits,
    }


def audit_equivalence(concepts,beta,bias,*,alpha=.35):
    c=np.asarray(concepts,dtype=np.float64)
    e=equivalent_concept_head(c,beta,bias,alpha=alpha)
    o=e["original_parts"];t=e["remixed_parts"]
    n,k=c.shape
    return EquivalenceAudit(
        k=k,n=n,alpha=float(alpha),
        mixing_eigenvalue_orthogonal=1.-alpha,
        matrix_condition_number=float(np.linalg.cond(e["matrix"])),
        max_logit_difference=float(np.max(np.abs(e["original_logits"]-e["remixed_logits"]))),
        max_probability_difference=float(np.max(np.abs(
            expit(e["original_logits"])-expit(e["remixed_logits"])))),
        max_signed_contribution_sum_difference=float(np.max(np.abs(
            o.sum(axis=1)-t.sum(axis=1)))),
        max_fuzzy_membership_difference=float(np.max(np.abs(c-e["concepts"]))),
        bounded_state_violations=int(np.count_nonzero(
            (e["concepts"]<0)|(e["concepts"]>1))),
        fraction_changed_top_abs_contribution_axis=float(np.mean(
            np.argmax(np.abs(o),axis=1)!=np.argmax(np.abs(t),axis=1))),
        individual_attribution_mean_abs_difference=float(np.mean(np.abs(o-t))),
    )
