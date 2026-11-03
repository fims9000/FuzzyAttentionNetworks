"""Falsify post-hoc one-shot report inference on invented aggregate fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from conceptfan_discovery.report_one_shot_v11 import (
    ARCHITECTURES, format_one_shot_scientific_report,
)


def _synthetic_aggregate(root:Path, *,
                         v10_auprc:float=.38,v11_auprc:float=.35,
                         candidates_v10:int=1,candidates_v11:int=1):
    root.mkdir(parents=True,exist_ok=True)
    (root/"STATUS.json").write_text(json.dumps({
        "status":"ONE_SHOT_SET_B_INDEPENDENT_PATIENT_COHORT_COMPLETED",
        "not_independent_hospital":True,
        "set_a_original_calibration_test_sealed":True,
        "not_used_for_model_selection_or_retraining":True,
        "previous_set_a_dev_verified_all_models":True,
        "no_patient_id_predictions_or_checkpoints_in_report":True,
    }))
    scores=[]
    for model in ARCHITECTURES:
        mean=(
            v10_auprc if model==ARCHITECTURES[0] else
            v11_auprc if model==ARCHITECTURES[2] else .37
        )
        scores.append({
            "scenario":"original","method":model,
            "independent_model_initializations":
                1 if model in ("logistic_history","hist_gradient_boosting") else 3,
            "mean_auprc":mean,
            "mean_auroc":.76,
            "mean_brier_uncalibrated":.2,
            "n_new_patients":4000,
            "outcome_prevalence":.14,
        })
    pd.DataFrame(scores).to_csv(root/"SET_B_ONE_SHOT_AGGREGATE.csv",index=False)
    pairs=[]
    for comparator in (ARCHITECTURES[0],"transformer_timeseries"):
        pairs.append({
            "scenario":"original","candidate":ARCHITECTURES[2],
            "comparator":comparator,
            "mean_auprc_difference_candidate_minus_comparator":v11_auprc-.38,
            "paired_patient_and_model_bootstrap_interval":"[-0.045, 0.01]",
        })
    pd.DataFrame(pairs).to_csv(
        root/"SET_B_PAIRED_PATIENT_MODEL_BOOTSTRAP.csv",index=False
    )
    obj={"methods":{}}
    for model in ARCHITECTURES[:3]:
        obj["methods"][model]={
            "n_candidate_individual_effects":
                candidates_v11 if model==ARCHITECTURES[2] else candidates_v10,
            "n_candidate_group_only":0,
            "n_abstained":6,
        }
    (root/"SET_B_FUNCTIONAL_IDENTIFIABILITY_AUDIT.json").write_text(
        json.dumps(obj)
    )


def test_negative_one_shot_result_is_reported_not_oversold(tmp_path):
    _synthetic_aggregate(tmp_path)
    report=format_one_shot_scientific_report(tmp_path)
    assert "NEGATIVE: V11 loses" in report
    assert "NOT an independent hospital" in report
    assert "No raw patient-level predictions" in report
    assert "new theory of latent causal" in report
    assert "0.3500" in report
    assert "-0.0300" in report


def test_potential_gain_not_interpreted_as_causal_identity(tmp_path):
    _synthetic_aggregate(tmp_path,
                         v10_auprc=.38,v11_auprc=.385,
                         candidates_v10=1,candidates_v11=3)
    report=format_one_shot_scientific_report(tmp_path)
    assert "PROMISING BUT INSUFFICIENT" in report
    assert "not verified physiological mechanisms" in report


def test_missing_or_duplicated_method_invalidates_result(tmp_path):
    _synthetic_aggregate(tmp_path)
    csv=tmp_path/"SET_B_ONE_SHOT_AGGREGATE.csv"
    frame=pd.read_csv(csv).iloc[:-1]
    frame.to_csv(csv,index=False)
    with pytest.raises(ValueError,match="Missing or duplicated"):
        format_one_shot_scientific_report(tmp_path)


def test_unverified_or_retuned_holdout_cannot_generate_paper_report(tmp_path):
    _synthetic_aggregate(tmp_path)
    source=tmp_path/"STATUS.json"
    d=json.loads(source.read_text())
    d["not_used_for_model_selection_or_retraining"]=False
    source.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="unretuned"):
        format_one_shot_scientific_report(tmp_path)


def test_no_positive_story_when_auprc_good_but_no_extra_stable_effect(tmp_path):
    _synthetic_aggregate(tmp_path,
                         v10_auprc=.38,v11_auprc=.39,
                         candidates_v10=1,candidates_v11=1)
    report=format_one_shot_scientific_report(tmp_path)
    assert "INCONCLUSIVE OR UNFAVORABLE" in report
