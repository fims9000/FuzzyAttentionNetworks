"""V26 falsification guards; actual native network integrated by experiment smoke."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.experiment_frozen_concept_head_refit_v26 import (
    LinearNativeHead, refit_linear_concept_head, refit_positive_platt,
)


def test_linear_native_refit_keeps_additive_contributions_and_no_shortcut():
    rng=np.random.default_rng(20261106)
    q=rng.normal(size=(500,3))
    y=rng.binomial(1,1/(1+np.exp(-(q[:,0]-q[:,1]+.5*q[:,2]))))
    head=refit_linear_concept_head(q,y)
    logit, contrib=head(q)
    np.testing.assert_allclose(logit, head.bias+contrib.sum(axis=1))
    assert contrib.shape==(500,3)
    assert np.isfinite(logit).all()


def test_platt_positive_calibration_cannot_improve_rank():
    rng=np.random.default_rng(123)
    raw=rng.normal(size=420)
    y=rng.binomial(1,1/(1+np.exp(-raw)))
    cal=refit_positive_platt(raw,y)
    assert cal.slope>0
    np.testing.assert_array_equal(np.argsort(raw),np.argsort(cal(raw)))
    assert np.isfinite(cal(raw)).all()


def test_head_rejects_non_concept_inputs():
    head=LinearNativeHead(0.,(1.,-1.))
    with pytest.raises(ValueError):
        head(np.zeros((5,5)))
    with pytest.raises(ValueError):
        head(np.full((5,2),np.nan))
