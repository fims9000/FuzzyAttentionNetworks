from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from conceptfan_discovery.identifiability_v11 import PredictiveCertificate,PredictiveGroup
from conceptfan_discovery.multiscale_physiology_v10 import MultiScaleDualProcessConceptFAN
from conceptfan_discovery.passports_v11 import identified_predictive_passports


def _model_and_cert():
    model=MultiScaleDualProcessConceptFAN(
        observable_loadings=np.eye(5,3,dtype=np.float64),
        channels=("pressure","heart_rate","creatinine","lactate","temperature"),
        forecast_horizon=6,
        n_observation_concepts=3,
    )
    g=[
        PredictiveGroup(0,1,"INDIVIDUAL_OBSERVABLE_AXIS_STABLE",.98,.99,.98,(0,)),
        PredictiveGroup(1,3,"GROUP_SUBSPACE_STABLE_AXES_UNRESOLVED",.91,.95,None,(1,2)),
    ]
    c=PredictiveCertificate(
        observed_singular_values=np.asarray([3.,2.,1.,.1,.01]),
        bootstrap_singular_values=np.ones((10,5)),
        reference_basis=np.eye(5),
        groups=tuple(g),
        certified_boundaries=(1,),
        boundary_evidence=tuple(),
        selected_computational_rank=3,
        diagnostics={"clinical_concepts_identified":False},
    )
    return model,c


def test_pasports_disclose_true_learned_thresholds_and_no_fabricated_physical_cutoffs():
    model,cert=_model_and_cert()
    model.eval()
    passport=identified_predictive_passports(model,cert)
    assert passport["n_predictive_slots"]==3
    assert passport["no_patient_identifiers_or_values"]
    assert len(passport["groups"])==2
    first=passport["groups"][0]["member_parameter_formulas"][0]
    tau,width=model.physical.concept_boundaries()
    assert first["feature_activation_threshold_in_composite_score"]==pytest.approx(float(tau[0].detach()))
    assert first["transition_width_in_composite_score"]==pytest.approx(float(width[0].detach()))
    assert first["single_physical_channel_clinical_cutoff"] is None
    assert first["top_observed_operator_terms"]
    for group in passport["groups"]:
        for unit in group["member_parameter_formulas"]:
            assert sum(unit["temporal_scale_mixture_hours"].values())==pytest.approx(1.)
            assert unit["unique_physiological_identity_verified"] is False


def test_unresolved_rotating_pair_remains_group_without_fabricated_identity():
    model,cert=_model_and_cert()
    items=identified_predictive_passports(model,cert)["groups"]
    assert not items[0]["no_individual_identity_claim"]
    assert items[1]["no_individual_identity_claim"]
    assert items[1]["members"]==[1,2]
    assert len(items[1]["member_parameter_formulas"])==2
    assert all(s["single_physical_channel_clinical_cutoff"] is None
               for s in items[1]["member_parameter_formulas"])


def test_wrong_rank_and_top_features_rejected_without_relaxing_ontology():
    model,cert=_model_and_cert()
    with pytest.raises(ValueError):
        identified_predictive_passports(model,cert,max_operator_terms=0)
    with pytest.raises(ValueError):
        identified_predictive_passports(model,replace(cert,selected_computational_rank=2))


def test_native_formula_and_bottleneck_logit_preserve_hard_missing_case():
    model,cert=_model_and_cert()
    model.eval()
    x=torch.randn(4,12,5)
    mask=torch.zeros_like(x)
    with torch.no_grad():
        output=model(x,mask,process_mask=mask)
    assert output.decomposition_error().item()<1e-6
    # Float32 0.5-valued concept histories can accumulate ~1e-8
    # rounding at weighted temporal pooling; 1e-7 is the same exact
    # additive-logit tolerance used elsewhere, not a license for
    # informative missingness.
    assert float(output.physical.contributions.abs().max())<1e-7
    p=identified_predictive_passports(model,cert)
    assert p["unreliable_missing_history_is_unknown_neutral"]
