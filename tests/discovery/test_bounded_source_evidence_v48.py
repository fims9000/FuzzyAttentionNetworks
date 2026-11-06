"""V48 frozen risk-evidence link: monotonicity, boundedness, no hidden inputs."""
import numpy as np
import pytest
from conceptfan_discovery.bounded_source_evidence_v48 import (
    bounded_membership_logodds_v48,risk_concept_input_v48,describe_source_link_v48,
)


def test_monotone_and_finite_for_edges_and_symmetry():
    q=np.array([0,.0001,.001,.01,.1,.2,.5,.8,.9,.99,.999,1.])
    e=bounded_membership_logodds_v48(q)
    assert np.isfinite(e).all()
    assert (np.diff(e)>=0).all()
    assert (e>=-.5).all() and (e<=.5).all()
    np.testing.assert_allclose(e+e[::-1],np.zeros_like(e),atol=1e-12,rtol=0.)
    assert e[6]==0


def test_two_registered_links_match_exact_centering_and_no_prob_interpretation():
    q=np.array([.1,.3,.5,.7,.9])
    linear=risk_concept_input_v48(q,link="linear_membership")
    stretch=risk_concept_input_v48(q,link="bounded_logodds")
    np.testing.assert_array_equal(linear,q)
    np.testing.assert_allclose(stretch-.5,bounded_membership_logodds_v48(q))
    assert stretch[2]==.5
    assert not np.array_equal(stretch,linear)
    assert describe_source_link_v48("bounded_logodds")["not_physical_identity_or_uncertainty_certificate"]


def test_invalid_concept_input_fails_closed():
    with pytest.raises(ValueError):
        risk_concept_input_v48(np.array([.8,1.2]),link="bounded_logodds")
    with pytest.raises(ValueError):
        risk_concept_input_v48(np.array([.8,np.nan]),link="bounded_logodds")
    with pytest.raises(ValueError):
        risk_concept_input_v48(np.array([.5]),link="invented")
