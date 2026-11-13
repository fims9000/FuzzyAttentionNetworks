"""V55 autonomous adaptive concept vocabulary for masked time series.

No oracle concept targets or manual channel groups are accepted. Unlabeled
TRAIN X and observation mask deterministically choose between:
  (a) robust measured-source grouped fuzzy concepts, and
  (b) anonymous observable predictive spectral coordinates.
Only afterwards does the independent supervised, exact additive risk head
see TRAIN Y. No claim of physical factor identity or new spectral theory.
"""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Any
import numpy as np

from conceptfan_discovery.autonomous_temporal_bank_v53 import (
    check_measurements, event_intervals
)
from conceptfan_discovery.robust_observable_concepts_v54 import (
    RobustGroupConceptFAN, SpectralObservableConceptFAN,
    fit_robust_vocabulary, fit_observable_spectral_definitions,
)

MIN_GROUPS=2
MIN_COVERAGE=0.50
SPECTRAL_LAGS=(1,2,3)
SPECTRAL_PERMUTATIONS=60
SPECTRAL_BOOTSTRAPS=20


class AdaptiveConceptFAN:
    """One fitted prediction API, with source definitions formed before labels.

    Source route is decided only from TRAIN measured x/m. A stable supported
    observed group is a *statistical measurement* concept, while predictive
    SVD coordinates are explicitly unnamed. Both use risk coefficients trained
    only after source-fitting and are strictly additive in logit space.
    """
    def __init__(self, *, seed:int=20261331, permutations:int=SPECTRAL_PERMUTATIONS,
                 bootstraps:int=SPECTRAL_BOOTSTRAPS):
        if permutations<25 or bootstraps<5:
            raise ValueError("Insufficient fixed spectral null repetitions")
        self.seed=int(seed)
        self.permutations=int(permutations)
        self.bootstraps=int(bootstraps)
        self.route:str|None=None
        self.model:RobustGroupConceptFAN|SpectralObservableConceptFAN|None=None
        self.selected_k=0
        self.route_receipt:dict[str,Any]|None=None
        self.source_hash:str|None=None

    def fit(self,train_x:np.ndarray,train_mask:np.ndarray,train_y:np.ndarray):
        x,m=check_measurements(train_x,train_mask)
        yy=np.asarray(train_y).reshape(-1)
        if len(yy)!=len(x) or not np.isin(yy,(0,1)).all() or len(np.unique(yy))!=2:
            raise ValueError("Risk training requires aligned binary labels")
        # ALL source fitting must happen before passing labels to fit_risk_head.
        discovered=fit_robust_vocabulary(x,m,seed=self.seed)
        valid_groups=len(discovered.groups)
        covered=len(set(j for group in discovered.groups
                        for j in group.sensor_indices))
        coverage=covered/x.shape[-1]
        is_measured=(valid_groups>=MIN_GROUPS and coverage>=MIN_COVERAGE)
        if is_measured:
            candidate=RobustGroupConceptFAN(discovered)
            route="MEASURED_GROUPS"
            k=valid_groups
            spectral_receipt=None
        else:
            definition=fit_observable_spectral_definitions(
                x,m,seed=self.seed+17,
                permutations=self.permutations,
                bootstraps=self.bootstraps)
            candidate=SpectralObservableConceptFAN(definition)
            k=definition.rank
            route="PREDICTIVE_SUBSPACE" if k else "OBSERVATION_ABSTAIN"
            spectral_receipt=definition.train_rank_receipt
        # No labels enter candidate/group/source discovery above this line.
        candidate.fit_risk_head(x,m,yy)
        self.model=candidate
        self.route=route
        self.selected_k=k
        source_payload=candidate.passport().get("source")
        if source_payload is None:
            raise AssertionError("Model lacks measured-source provenance")
        self.source_hash=sha256(json.dumps(
            source_payload,sort_keys=True,allow_nan=False).encode("utf8")).hexdigest()
        self.route_receipt={
            "source_definition_train_only":True,
            "source_routing_uses_outcome_labels":False,
            "known_concepts_or_oracle_latents_used":False,
            "no_physical_identity_certified":True,
            "selected_route":route,
            "selected_concept_count":k,
            "detected_group_count":valid_groups,
            "group_channel_coverage":float(coverage),
            "train_episode_count":int(len(x)),
            "train_sensor_count":int(x.shape[-1]),
            "train_event_length":int(x.shape[1]),
            "route_requirement_group_count":MIN_GROUPS,
            "route_requirement_channel_coverage":MIN_COVERAGE,
            "null_rank_state":bool(k==0),
            "spectral_rank_receipt":spectral_receipt,
            "source_sha256":self.source_hash,
            "risk_train_outcomes_seen_only_after_route_frozen":True,
            "interpretation":(
                "Measured coherent sensor-source proxies, not causal semantics"
                if route=="MEASURED_GROUPS" else
                "Observable predictive signed basis, not physical factor identity"
            )
        }
        return self

    def predict(self,x:np.ndarray,mask:np.ndarray)->dict[str,Any]:
        if self.model is None or self.route_receipt is None:
            raise RuntimeError("Fit vocabulary and risk on TRAIN before predict")
        out=self.model.predict(x,mask)
        err=out.get("logit_additive_error",out.get("max_additive_error"))
        if err is None or float(err)>1e-12:
            raise AssertionError("Non-exact risk attribution")
        return {**out,"selected_source_route":self.route,
                "selected_concept_count":self.selected_k,
                "fitted_source_sha256":self.source_hash,
                "max_additive_logit_error":float(err),
                "source_semantics":(
                    "MEASURED_ASSOCIATION_ONLY" if self.route=="MEASURED_GROUPS"
                    else "OBSERVABLE_PREDICTIVE_SUBSPACE_ONLY")}

    def public_passport(self)->dict[str,Any]:
        if self.route_receipt is None or self.model is None:
            raise RuntimeError("No completed TRAIN fit")
        return {
            "route":self.route_receipt,
            "source_model":self.model.passport(),
            "direct_concepts_are_observed_time_series_operators":True,
            "clinical_disease_labels_inferred":False,
            "classifier_has_direct_raw_feature_shortcut":False,
        }

    def event_summary(self,x,mask,*,threshold:float=.65,minimum_length:int=2):
        """Event counts / time bounds, never diagnosis or causal intervention."""
        out=self.predict(x,mask)
        z=out["trajectories"]
        names=out["names"]
        return {
            name:{
                "episode_fraction_with_event":float(np.mean([
                    bool(event_intervals(z[i,:,j],threshold=threshold,
                                         minimum_length=minimum_length))
                    for i in range(len(z))])),
                "mean_events_per_episode":float(np.mean([
                    len(event_intervals(z[i,:,j],threshold=threshold,
                                        minimum_length=minimum_length))
                    for i in range(len(z))])),
                "mean_signed_direct_logit_contribution":float(
                    out["contributions"][:,j].mean()),
            } for j,name in enumerate(names)
        }
