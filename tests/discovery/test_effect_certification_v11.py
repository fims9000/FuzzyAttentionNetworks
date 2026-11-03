from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conceptfan_discovery.effect_certification_v11 import joint_concept_screen


def _inputs():
    cert={"groups":[
        {"first_slot_zero_index":0,"stop_exclusive":1,
         "slots_zero_index":[0],"certificate":"INDIVIDUAL_OBSERVABLE_AXIS_STABLE"},
        {"first_slot_zero_index":1,"stop_exclusive":3,
         "slots_zero_index":[1,2],
         "certificate":"GROUP_SUBSPACE_STABLE_AXES_UNRESOLVED"},
    ]}
    rows=[]
    for members,rank,overlap in [([0],.92,.94),([1,2],.89,.91)]:
        for a,b in ((11,23),(11,37),(23,37)):
            rows.append({"method":"model_A","seed_a":a,"seed_b":b,
                         "slots":str(members),"certificate":
                            cert["groups"][0]["certificate"] if len(members)==1 else
                            cert["groups"][1]["certificate"],
                         "same_patient_spearman":rank,
                         "definition_subspace_overlap":overlap})
    return cert,pd.DataFrame(rows)


def test_joint_certificate_retains_individual_and_group_distinction():
    cert,rows=_inputs()
    result=joint_concept_screen(cert,rows)
    a=result["methods"]["model_A"]
    assert a["n_candidate_individual_effects"]==1
    assert a["n_candidate_group_only"]==1
    assert a["n_abstained"]==0
    assert a["groups"][1]["clinical_meaning_verified"] is False
    assert result["statistical_significance_claim"] is False


def test_stable_spectral_axis_with_negative_direct_effect_must_abstain():
    cert,rows=_inputs()
    rows.loc[rows.slots.eq("[0]") & rows.seed_a.eq(11),
             "same_patient_spearman"]=-.94
    result=joint_concept_screen(cert,rows)
    g=result["methods"]["model_A"]["groups"][0]
    assert g["decision"]=="ABSTAIN_CROSS_SEED_EFFECT_OR_DEFINITION"
    assert result["methods"]["model_A"]["n_candidate_individual_effects"]==0


def test_group_definition_mismatch_cannot_pass_with_good_risk_prediction():
    cert,rows=_inputs()
    rows.loc[rows.slots.eq("[1, 2]"),"definition_subspace_overlap"]=.5
    result=joint_concept_screen(cert,rows)
    assert result["methods"]["model_A"]["n_candidate_group_only"]==0


def test_missing_pair_cannot_masquerade_as_reproducible_model():
    cert,rows=_inputs()
    with pytest.raises(ValueError):
        joint_concept_screen(cert,rows.iloc[:-1])


def test_unstable_structural_group_always_abstains_even_if_effect_looks_good():
    cert,rows=_inputs()
    cert["groups"][0]["certificate"]="ABSTAIN_UNSTABLE_SUBSPACE"
    result=joint_concept_screen(cert,rows)
    assert result["methods"]["model_A"]["groups"][0]["decision"]=="ABSTAIN_STRUCTURAL_SUBSPACE"


def test_report_does_not_convert_three_overlapping_pairs_to_independent_trials():
    cert,rows=_inputs()
    report=joint_concept_screen(cert,rows)
    assert report["n_independent_source_initializations"]==3
    assert report["n_pair_scores_not_independent"]==3
    assert report["already_repeatedly_inspected_development"]
    assert not report["not_causal_identification"] is False
