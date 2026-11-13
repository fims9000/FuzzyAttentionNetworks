import inspect

import numpy as np
import pytest

from conceptfan_discovery.reference_anchored_dose_v71 import (
    SupervisedConditionalReferenceBank
)
from scripts.experiment_reference_anchored_mixture_v71 import (
    descriptive_specificity,predict_refit
)


def two_analyte_synthetic_trial_features(seed=71):
    rng=np.random.default_rng(seed)
    n=58
    prep=rng.choice([0.,.1,.3,1.],size=(n,2))
    mix=rng.normal(size=(2,32))
    x=prep@mix+rng.normal(scale=.02,size=(n,32))
    return x,prep


def test_supervision_is_explicit_not_hidden_chemical_discovery():
    x,y=two_analyte_synthetic_trial_features()
    sig=inspect.signature(SupervisedConditionalReferenceBank.fit_reference_anchored)
    assert list(sig.parameters)==[
        "self","TRAIN_features","TRAIN_water_dilution_volpercent"]
    bank=SupervisedConditionalReferenceBank().fit_reference_anchored(
        x[:39],y[:39])
    passport=bank.public_passport()
    assert passport["target_information_used_in_source_fit"] is True
    assert passport["not_an_autonomous_source_discovery_algorithm"]
    assert not passport["statistical_sources_not_independent_gas_assays"] is False
    assert passport["reference_units"].startswith("nominal water dilution")
    assert bank.score_proxies(x[39:]).shape==(19,2)
    assert bank.fuzzy(x[39:]).shape==(19,6)
    assert np.all((bank.fuzzy(x[39:])>=0)&(bank.fuzzy(x[39:])<=1))


def test_explicit_reference_label_change_really_changes_concept_definition():
    x,y=two_analyte_synthetic_trial_features()
    a=SupervisedConditionalReferenceBank().fit_reference_anchored(
        x[:39],y[:39])
    changed=np.random.default_rng(100).permutation(y[:39])
    b=SupervisedConditionalReferenceBank().fit_reference_anchored(
        x[:39],changed)
    assert a.sha!=b.sha
    assert not np.allclose(a.loading,b.loading)
    assert a.public_passport()["source_sha256"]==a.sha


def test_exact_concept_only_proxy_prediction_is_frozen_and_finite():
    x,y=two_analyte_synthetic_trial_features()
    source=SupervisedConditionalReferenceBank().fit_reference_anchored(
        x[:39],y[:39])
    q=source.fuzzy(x)
    pred,receipt=predict_refit(
        "V71_SUPERVISED_REFERENCE_FUZZY",
        q[:39],y[:39,0],q[39:])
    assert pred.shape==(19,)
    assert np.isfinite(pred).all()
    assert receipt["max_source_only_signed_additive_prediction_error"]<1e-9
    assert receipt["no_original_sensor_to_risk_bypass"]
    assert source.sha==source.public_passport()["source_sha256"]


def test_reference_source_selectivity_is_measured_not_assumed():
    x,y=two_analyte_synthetic_trial_features()
    b=SupervisedConditionalReferenceBank().fit_reference_anchored(
        x[:39],y[:39])
    report=descriptive_specificity(
        b,x,y,{"TRAIN":np.arange(39),
               "VALID":np.arange(39,50),
               "TEST":np.arange(50,58)})
    assert set(report)=={"TRAIN","VALID","TEST"}
    assert len(report["TEST"])==2
    assert report["TEST"][0]["n_independent_original_trials"]==8
    assert all(not v["gas_analyte_molecular_identity_proven"]
               for v in report["TEST"])


def test_never_accept_more_than_original_physical_training_trials():
    x,y=two_analyte_synthetic_trial_features()
    with pytest.raises(ValueError):
        SupervisedConditionalReferenceBank().fit_reference_anchored(
            x[:38],y[:38])
    with pytest.raises(ValueError):
        SupervisedConditionalReferenceBank().fit_reference_anchored(
            x[:39],y[:39,:1])
    with pytest.raises(ValueError):
        SupervisedConditionalReferenceBank().fit_reference_anchored(
            x[:39],np.full((39,2),np.nan))
    with pytest.raises(ValueError):
        SupervisedConditionalReferenceBank().fit_reference_anchored(
            x[:39],np.full((39,2),.4))
