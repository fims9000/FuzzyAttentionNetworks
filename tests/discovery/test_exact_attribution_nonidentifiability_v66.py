import numpy as np
import pytest
from scipy.special import expit
from conceptfan_discovery.exact_attribution_nonidentifiability_v66 import (
    bounded_mixing_matrix,equivalent_concept_head,audit_equivalence
)

@pytest.mark.parametrize("k",[3,18,36])
@pytest.mark.parametrize("alpha",[.05,.35,.75])
def test_exact_bounded_concept_reparameterization_can_change_explanation(k,alpha):
    rng=np.random.default_rng(20261966+k)
    concepts=rng.beta(1.5,1.5,size=(1000,k))
    beta=rng.normal(size=k)
    beta[0]+=2.8
    bias=-.41
    q=equivalent_concept_head(concepts,beta,bias,alpha=alpha)
    np.testing.assert_allclose(q["original_logits"],q["remixed_logits"],
                               rtol=0,atol=1e-12)
    np.testing.assert_allclose(
        expit(q["original_logits"]),expit(q["remixed_logits"]),
        rtol=0,atol=1e-13)
    assert np.all(q["concepts"]>=0) and np.all(q["concepts"]<=1)
    assert np.max(np.abs(q["concepts"]-concepts))>0.002
    assert np.mean(np.abs(q["original_parts"]-q["remixed_parts"]))>0.
    assert np.max(np.abs(q["matrix"].sum(axis=0)-1.))<1e-14
    assert np.max(np.abs(q["matrix"].sum(axis=1)-1.))<1e-14
    audit=audit_equivalence(concepts,beta,bias,alpha=alpha)
    assert audit.k==k and audit.n==1000
    assert audit.bounded_state_violations==0
    assert not audit.semantics_identified
    assert audit.max_logit_difference<1e-12
    assert audit.max_probability_difference<1e-12

def test_nontrivial_top_attribution_ranking_changes_while_risk_identical():
    rng=np.random.default_rng(88)
    c=rng.beta(1.2,2.5,size=(3000,3))
    beta=np.array([.2,2.0,1.])
    e=audit_equivalence(c,beta,bias=.3,alpha=.75)
    assert e.fraction_changed_top_abs_contribution_axis>.01
    assert e.max_probability_difference<1e-12

@pytest.mark.parametrize("k,alpha",[(1,.3),(3,0),(3,1),(0,.5),(3,-1),(3,float("nan"))])
def test_invalid_semantic_equivalence_domain_rejected(k,alpha):
    with pytest.raises(ValueError):
        bounded_mixing_matrix(k,alpha)

def test_nonfinite_and_out_of_bounds_memberships_refused():
    with pytest.raises(ValueError):
        equivalent_concept_head(np.array([[.2,1.4]]),[1,1],.1)
    with pytest.raises(ValueError):
        equivalent_concept_head(np.array([[np.nan,.4]]),[1,1],.1)
    with pytest.raises(ValueError):
        equivalent_concept_head(np.array([[.2,.4]]),[1,np.nan],.1)
