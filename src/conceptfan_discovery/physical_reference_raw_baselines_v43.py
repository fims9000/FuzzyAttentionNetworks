"""V43 strong matched physical-reference R-supervised raw sensor regressors.

Four preregistered direct ordinary predictors, trained on exactly the
same **noisy independent external physical R** made available to V42:
either TRAIN300 or TRAIN300+VALID1 200+VALID2 200 (ALL700), no TEST.
They see the same real source ON/OFF measurements and masks as V42,
including known original channels 0:3, but NOT hidden latent H/S/E/Y,
nor R from TEST. No concept definition or physically certified views.

Gating by a new independent nuisance instrument is evaluated for both
raw predictors and ConceptFAN, never reserved exclusively for ConceptFAN.
"""
from __future__ import annotations
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

from .physical_reference_certificate_v42 import (
    measured_external_on_off_difference_v42,
)
from .matched_raw_baselines_v40 import (
    measured_temporal_quadratic_features_v40,
)

METHODS=("RAW_RIDGE_R300","RAW_HGB_R300",
         "RAW_RIDGE_R700","RAW_HGB_R700")


def features_measured_reference_v43(on,off,on_mask,off_mask):
    diff,observed=measured_external_on_off_difference_v42(
        on,off,on_mask,off_mask)
    features=measured_temporal_quadratic_features_v40(
        on,on_mask,diff,observed,use_reference=True)
    source_count=observed[:,:,4:10].sum(axis=(1,2))
    # This is merely a real-ON/OFF-measurement availability policy;
    # it does not convey the source's physical name.
    enough=source_count>=8
    return features,enough


def fit_raw_reference_v43(method,x,measured_R):
    if method not in METHODS:
        raise ValueError("Unregistered physical R comparison method")
    features=np.asarray(x,dtype=np.float64)
    truth=np.asarray(measured_R,dtype=np.float64)
    required=700 if method.endswith("R700") else 300
    if (features.ndim!=2 or len(features)!=required
        or truth.shape!=(required,)
        or not np.isfinite(features).all() or not np.isfinite(truth).all()):
        raise ValueError("Raw baseline needs EXACT registered noisy physical R labels")
    if method.startswith("RAW_RIDGE"):
        model=make_pipeline(StandardScaler(),Ridge(alpha=10.))
    else:
        model=HistGradientBoostingRegressor(
            max_leaf_nodes=15,max_iter=120,learning_rate=.065,
            max_depth=4,min_samples_leaf=20,l2_regularization=1.,
            early_stopping=False,random_state=20264301,
        )
    model.fit(features,np.maximum(truth,0.))
    return model


def predict_raw_reference_v43(model,x):
    p=np.asarray(model.predict(np.asarray(x,dtype=np.float64)),
                 dtype=np.float64)
    if p.ndim!=1 or not np.isfinite(p).all():
        raise AssertionError("Numerical failure in physical R raw comparator")
    return np.maximum(0.,p)
