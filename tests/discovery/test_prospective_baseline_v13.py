"""Sanity checks for simple matched intervention-aware baseline, without real data."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from conceptfan_discovery.intervention_synthetic_v13 import make_intervention_scm
from conceptfan_discovery.instrumented_v13 import (
    InterventionProtocol, identify_action_anchored_concepts,
)
from conceptfan_discovery.prospective_baseline_v13 import last_four_sensor_projection

def test_training_only_observable_signature_and_additive_logit():
    simulation=make_intervention_scm(n=320,k=3,d=6,seed=1214)
    x=simulation.observations[:,:12]
    u=simulation.actions[:,:11]
    p=InterventionProtocol(True,True,True,True,("target_0","target_1","target_2"))
    cert=identify_action_anchored_concepts(
        x[:240],u[:240],p,seed=12,permutations=99,bootstraps=24)
    train=last_four_sensor_projection(x[:240],cert.dual)
    test=last_four_sensor_projection(x[240:],cert.dual)
    model=LogisticRegression(C=1.,solver="lbfgs",max_iter=1000)
    model.fit(train,simulation.labels[:240])
    terms=test*model.coef_[0][None,:]
    np.testing.assert_allclose(terms.sum(1)+model.intercept_[0],
                               model.decision_function(test),atol=1e-10)
    assert terms.shape==(80,3)

def test_sensor_dimensions_and_missing_values_refused():
    with pytest.raises(ValueError):
        last_four_sensor_projection(np.ones((10,3,6)),np.ones((3,6)))
    with pytest.raises(ValueError):
        last_four_sensor_projection(np.ones((10,5,5)),np.ones((3,6)))
    bad=np.ones((10,5,6))
    bad[0,0,0]=float("nan")
    with pytest.raises(ValueError):
        last_four_sensor_projection(bad,np.ones((3,6)))
