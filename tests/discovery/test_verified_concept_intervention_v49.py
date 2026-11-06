"""V49 tests for expert-only concept changes and label-free review selection."""
import numpy as np
import pytest
from scipy.special import expit,logit

from conceptfan_discovery.verified_concept_intervention_v49 import (
    fit_independent_reference_v49,synthetic_expert_verdict_v49,
    choose_review_patients_v49,apply_verified_concept_interventions_v49,
)


def test_assay_train_only_and_adversarial_test_reference():
    rng=np.random.default_rng(49)
    energy=rng.normal(size=800)
    q=expit(1.1*energy-.2+.1*rng.normal(size=len(energy)))
    fit=fit_independent_reference_v49(q,np.ones(len(q),dtype=bool),energy)
    assert fit.certified_available
    assert fit.supported_train_count==800
    assert fit.reference_is_ideal_synthetic_assay
    qtest=expit(1.1*energy[:100]-.2+.1*rng.normal(size=100))
    select=np.zeros(len(qtest),dtype=bool)
    select[:70]=True
    actions,receipt=synthetic_expert_verdict_v49(fit,qtest,energy[:100],select)
    assert receipt["reviewed"]==70
    assert receipt["rejected"]<20
    changed=qtest.copy()
    changed[:70]=.999
    bad,proof=synthetic_expert_verdict_v49(fit,changed,energy[:100],select)
    assert proof["rejected"]>25
    assert (bad[~select]=="keep").all()


def test_top_budget_is_reproducible_and_never_needs_y_or_assay():
    base=expit(np.array([0.,0.,0.,0.,0.]))
    p=expit(np.array([0.1,0.9,-1.5,.5,.2]))
    used=np.array([True,True,True,False,True])
    chosen=choose_review_patients_v49(p,base,used,budget=2,selection="active")
    assert np.flatnonzero(chosen).tolist()==[1,2]
    random1=choose_review_patients_v49(p,base,used,budget=2,
                                      selection="random",seed=99)
    random2=choose_review_patients_v49(p,base,used,budget=2,
                                      selection="random",seed=99)
    np.testing.assert_array_equal(random1,random2)
    assert random1.sum()==2
    assert not random1[3]


def test_reject_exact_frozen_baseline_and_confirm_does_not_mutate_other_subjects():
    baseline=expit(np.linspace(-2,2,6))
    current=expit(logit(baseline)+np.array([.5,.1,-.2,.7,0.,-.6]))
    used=np.ones(6,dtype=bool)
    q=np.array([.65,.9,.3,.75,.5,.1])
    decisions=np.array(["keep","confirm","reject","keep","keep","confirm"])
    refs=["","independent_assay", "independent_assay","","","assay_5"]
    result=apply_verified_concept_interventions_v49(
        baseline,current,q,used,decisions,
        link="bounded_logodds",gamma=1.5,reference_ids=refs)
    got=result["probability"]
    assert got[2]==baseline[2]
    assert got[1]==current[1] and got[5]==current[5]
    assert np.array_equal(got[[0,3,4]],current[[0,3,4]])
    assert result["n_verified_reviews"]==3
    assert result["n_rejected_sources"]==1


def test_verified_replace_changes_only_single_signed_contribution():
    b=np.array([.35,.55,.25])
    q=np.array([.2,.5,.8])
    used=np.array([True,True,True])
    gamma=2.0
    before=expit(logit(b)+gamma*(q-.5))
    corrected=np.array([.9,np.nan,np.nan])
    actions=np.array(["replace","keep","keep"])
    out=apply_verified_concept_interventions_v49(
        b,before,q,used,actions,
        link="linear_membership",gamma=gamma,
        corrected_membership=corrected,
        reference_ids=["independent_measurement","",""])
    expected=expit(logit(b[0])+gamma*(corrected[0]-.5))
    np.testing.assert_allclose(out["probability"][0],expected,atol=1e-12,rtol=0)
    np.testing.assert_array_equal(out["probability"][1:],before[1:])
    assert out["n_membership_replacements"]==1


def test_no_forced_unsupported_source_or_unverified_action():
    b=np.array([.4,.6])
    q=np.array([.2,.7])
    with pytest.raises(ValueError):
        apply_verified_concept_interventions_v49(
            b,b,q,np.array([True,False]),["keep","reject"],
            link="linear_membership",gamma=1.0,reference_ids=["","verified"])
    with pytest.raises(ValueError):
        apply_verified_concept_interventions_v49(
            b,b,q,np.array([True,True]),["reject","keep"],
            link="linear_membership",gamma=1.0,reference_ids=["",""])
    with pytest.raises(ValueError):
        apply_verified_concept_interventions_v49(
            b,b,q,np.array([True,True]),["replace","keep"],
            link="linear_membership",gamma=1.0,corrected_membership=np.array([3.,.5]),
            reference_ids=["verified",""])
