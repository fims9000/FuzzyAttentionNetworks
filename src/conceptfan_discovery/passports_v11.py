"""Trustworthy V11 concept passports from actual trained native parameters.

A concept is a measured signed operator mixture with a learned activation
boundary and uncertainty width, not a clinical diagnosis. Unresolved train
spectral groups receive a GROUP-only interpretation; do not assign
individual member names. This file never inspects patient records or labels.
"""
from __future__ import annotations

import numpy as np
import torch

from .identifiability_v11 import PredictiveCertificate
from .multiscale_physiology_v10 import MultiScaleDualProcessConceptFAN
from .operators import OPERATORS


def identified_predictive_passports(
    model:MultiScaleDualProcessConceptFAN,
    certificate:PredictiveCertificate,
    *,
    max_operator_terms:int=8,
)->dict:
    """Render formulas for native learned concepts; no post-hoc fitted model."""
    physical=model.physical
    if not 1<=max_operator_terms<=physical.feature_dim:
        raise ValueError("Invalid feature threshold")
    if certificate.selected_computational_rank!=physical.n_slots or (
        certificate.reference_basis.shape[0]!=physical.n_channels
    ):
        raise ValueError("Certificate and native learned sensory model do not align")
    c=physical.n_channels
    names=[
        f"{direction}|{name}|{channel}"
        for direction in ("positive","negative")
        for name in OPERATORS
        for channel in physical.channels
    ]
    if len(names)!=physical.feature_dim:
        raise ValueError("Operator dictionary definition mismatch")
    model.eval()
    with torch.no_grad():
        w=physical.feature_weights().cpu().numpy()
        scales=physical.scale_weights().cpu().numpy()
        gates=physical.gates.deterministic().cpu().numpy()
        boundaries,widths=physical.concept_boundaries()
        boundaries,widths=boundaries.cpu().numpy(),widths.cpu().numpy()
        beta=physical.decision_weights().cpu().numpy()
        process=model.process.passports(tuple(physical.channels))
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(scales)):
        raise ValueError("Refuse nonfinite learned concept definitions")
    groups=[]
    for g in certificate.groups:
        slots=[]
        for slot in g.members:
            selected=np.argsort(-w[slot])[:max_operator_terms]
            terms=[
                {"operator":names[i],"learned_simplex_weight":float(w[slot,i])}
                for i in selected if w[slot,i]>1e-8
            ]
            # top-N is only a display summary. All nonzero operator
            # coefficients are retained to reconstruct the EXACT native
            # formula, including tiny but genuine sparsemax entries.
            all_terms=[
                {"operator":names[i],"learned_simplex_weight":float(w[slot,i])}
                for i in range(len(names)) if w[slot,i]>0.0
            ]
            slots.append({
                "slot":int(slot),
                "operator_formula":"signed simplex-weighted observed operators at 1h/4h/12h/24h history, reliability-normalized",
                "temporal_scale_mixture_hours":{
                    str(int(h)):float(v)
                    for h,v in zip(physical.scales,scales[slot])
                },
                "top_observed_operator_terms":terms,
                "all_nonzero_observed_operator_terms":all_terms,
                "all_weights_sum_to_one":float(w[slot].sum()),
                "feature_activation_threshold_in_composite_score":
                    float(boundaries[slot]),
                "transition_width_in_composite_score":float(widths[slot]),
                "gate":float(gates[slot]),
                "signed_direct_logit_weight":float(beta[slot]),
                "single_physical_channel_clinical_cutoff":None,
                "reason_no_single_clinical_cutoff":
                    "multichannel signed nonlinear operators and temporal mixtures define a composite learned score",
                "each_slot_inside_native_training_graph":True,
                "unique_physiological_identity_verified":False,
            })
        groups.append({
            "members":list(g.members),
            "observed_predictive_status":g.status,
            "stability_q10_source_projector":float(g.q10_subspace_overlap),
            "no_individual_identity_claim":g.status!="INDIVIDUAL_OBSERVABLE_AXIS_STABLE",
            "member_parameter_formulas":slots,
            "exact_model_logit_group_contribution":
                "sum(beta[k] * (mean(last_6h, concept[k]) - 0.5*gate[k]) for k in group)",
            "can_infer_true_etiology_or_intervention_effect":False,
            "individual_member_formula_does_not_imply_individual_identifiability":True,
        })
    return {
        "description":"Actual trained model-native signed operator and temporal concept definitions",
        "source":"train-only predictive bootstrap groups and trained V11 network",
        "n_predictive_slots":physical.n_slots,
        "n_empirically_split_groups":len(certificate.groups),
        "operator_bank":names,
        "groups":groups,
        "original_observation_process_head":process,
        "no_patient_identifiers_or_values":True,
        "formula_parameters_are_model_internal_not_physiological_diagnoses":True,
        "unreliable_missing_history_is_unknown_neutral":True,
        "no_hidden_prediction_bypass":True,
    }
