"""V71 target-anchored conditional *preparation-dose proxy*, NOT autonomous concepts.

This source explicitly uses TRAIN39 reference acetone/ethanol WATER dilution
vol.% labels to fit two supervised conditional sensor directions. It is an
upper-bound/ablation of the V70 unlabeled physical source identification
problem, and must NEVER be called a separately assayed airborne gas source.

V71 was designed AFTER V70 TEST8 was inspected. All results exploratory.
"""
from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
from scipy.special import expit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

REFERENCE_NAMES=("acetone_preparation_dose","ethanol_preparation_dose")


class SupervisedConditionalReferenceBank:
    def __init__(self):
        self.scaler=None
        self.loading=None
        self.thresholds=None
        self.widths=None
        self.dose_residual_receipts=None
        self.sha=None
        self.receipt=None

    @staticmethod
    def _check_features(x):
        v=np.asarray(x,dtype=float)
        if v.ndim!=2 or v.shape[-1]!=32 or not np.isfinite(v).all():
            raise ValueError("Exactly 32 true physical sensor summary measurements required")
        return v

    def fit_reference_anchored(self,TRAIN_features,TRAIN_water_dilution_volpercent):
        x=self._check_features(TRAIN_features)
        y=np.asarray(TRAIN_water_dilution_volpercent,dtype=float)
        if len(x)!=39 or y.shape!=(39,2) or not np.isfinite(y).all():
            raise ValueError("Physical dose supervision is permitted ONLY on original 39 TRAIN trials")
        if (y<0).any() or (y>1).any():
            raise ValueError("Reference fields must be original water-dilution vol.% not gas-phase ppm")
        self.scaler=StandardScaler().fit(x)
        normalized=self.scaler.transform(x)
        coef=np.zeros((32,2),dtype=np.float64)
        residual_receipts=[]
        for j,name in enumerate(REFERENCE_NAMES):
            other=y[:,1-j]
            o=np.column_stack((np.ones(39),other))
            residualization=np.linalg.lstsq(o,y[:,j],rcond=None)[0]
            conditioned=y[:,j]-o@residualization
            if np.std(conditioned)<=1e-8:
                raise ValueError("An analyte reference is not conditionally varied on TRAIN")
            fit=Ridge(alpha=1.).fit(normalized,conditioned)
            loading=np.asarray(fit.coef_,dtype=np.float64)
            score=normalized@loading
            orientation=1. if np.cov(score,conditioned)[0,1]>=0 else -1.
            coef[:,j]=orientation*loading
            residual_receipts.append({
                "reference_name_supplied_by_TRAIN_supervision":name,
                "other_reference_regression_intercept":float(residualization[0]),
                "other_reference_regression_coefficient":float(residualization[1]),
                "residual_target_std_nominal_water_volpercent":float(np.std(conditioned)),
                "ridge_alpha_precommitted":1.,
                "orientation_precommitted_from_TRAIN_only":orientation,
                "direct_supervision_from_known_preparation_reference":True,
                "physical_airborne_analyte_assay_available":False,
            })
        self.loading=coef
        z=self.score_proxies(x)
        quantiles=np.quantile(z,[.25,.50,.75],axis=0).T
        widths=np.maximum(.15,(quantiles[:,2]-quantiles[:,0])/1.35)
        self.thresholds=quantiles
        self.widths=widths
        self.dose_residual_receipts=residual_receipts
        self.receipt={
            "number_of_independent_original_training_trials":39,
            "two_source_axis_names_PROVIDED_BY_REFERENCE_TRAIN_LABELS":list(REFERENCE_NAMES),
            "target_information_used_in_source_fit":True,
            "target_information_used_in_downstream_head_too":True,
            "not_an_autonomous_source_discovery_algorithm":True,
            "statistical_sources_not_independent_gas_assays":True,
            "reference_units":"nominal water dilution vol.percent, NOT gas ppm",
            "original_32_raw_sensor_feature_scaler_mean":self.scaler.mean_.tolist(),
            "original_32_raw_sensor_feature_scaler_scale":self.scaler.scale_.tolist(),
            "two_supervised_32_by_2_dose_residual_proxy_loadings":coef.tolist(),
            "six_TRAIN_calibrated_fuzzy_thresholds":quantiles.tolist(),
            "two_positive_train_widths":widths.tolist(),
            "cross_reference_residualization":residual_receipts,
            "post_V70_TEST_inspected_exploratory":True,
        }
        self.sha=sha256(json.dumps(
            self.receipt,sort_keys=True,allow_nan=False).encode()).hexdigest()
        return self

    def score_proxies(self,x):
        if self.loading is None:
            raise RuntimeError("Reference supervised source must be fitted on TRAIN first")
        v=self._check_features(x)
        z=self.scaler.transform(v)@self.loading
        if not np.isfinite(z).all():
            raise AssertionError("Nonfinite supervised physical reference proxy")
        return z

    def fuzzy(self,x):
        if self.thresholds is None:
            raise RuntimeError("Missing TRAIN reference proxy fuzzy membership calibration")
        z=self.score_proxies(x)
        s=expit((z[:,:,None]-self.thresholds[None,:,:])/
                self.widths[None,:,None]).reshape(len(z),6)
        if not np.isfinite(s).all() or np.any((s<0)|(s>1)):
            raise AssertionError("Invalid bounded supervised dose memberships")
        return s

    def public_passport(self):
        if self.receipt is None:
            raise RuntimeError("Supervised TRAIN fit not completed")
        return dict(self.receipt,source_sha256=self.sha)
