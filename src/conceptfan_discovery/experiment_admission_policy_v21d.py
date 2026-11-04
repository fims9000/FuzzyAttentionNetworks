"""V21d: executed six-condition bounded concept admission policy audit.

Do not certify physical identity from prediction, independent device
corroboration, reproducibility or even known-condition acquisition reset.
The persistent environmental artifact is a declared false positive.
No real clinician is represented by the synthetic instrument veto control.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .provenance_data_v21 import (
    CHANNELS,DISCOVERY_N,SCREEN_N,CORROBORATION_N,
    make_v21_provenance_cohort,
)
from .instrument_reset_data_v21c import RESET_FAMILIES,make_v21c_repeat_acquisitions
from .candidate_admission_v17 import propose_observable_residual_concept
from .risk_screen_v18 import screen_risk_association
from .measurement_corroboration_v21 import corroborate_candidate_on_disjoint_measured_reference
from .instrument_repeatability_v21c import assess_candidate_after_independent_acquisition_reset
from .concept_evidence_policy_v21d import (
    review_candidate_evidence_for_native_admission,
    expand_original_native_model_if_provisionally_measured,
    bounded_evidence_passport,
    ALLOW_RISK,ALLOW_NO_RISK,ABSTAIN_REFERENCE,ABSTAIN_REPEAT,BLOCK_ARTIFACT,
)
from .multiscale_physiology_v10 import MultiScaleDualProcessConceptFAN


EXPECTED={
    "physical_predictive":ALLOW_RISK,
    "physical_nontask":ALLOW_NO_RISK,
    "artifact_predictive":ABSTAIN_REFERENCE,
    "artifact_nontask":ABSTAIN_REFERENCE,
    "artifact_common_mode_reference":ABSTAIN_REPEAT,
    "artifact_environmental_persistent":ALLOW_RISK,
}


def _new_k2_research_model():
    torch.manual_seed(21713)
    return MultiScaleDualProcessConceptFAN(
        observable_loadings=np.eye(4,2),
        channels=CHANNELS,
        forecast_horizon=6,
        n_observation_concepts=2,
        scales=(1,4,12),
        adaptive_feature_gradients=True,
    ).eval()


def _assert_no_drift(before,after,split):
    original=before.eval()
    expanded=after.eval()
    with torch.no_grad():
        x=torch.from_numpy(split.x[:48])
        mask=torch.from_numpy(split.mask[:48])
        old=original(x,mask,process_mask=mask)
        new=expanded(x,mask,process_mask=mask)
    if not torch.allclose(old.logits,new.logits,atol=1e-6,rtol=1e-6):
        raise AssertionError("Source added unsupported direct risk influence")
    if not torch.allclose(
        old.physical.future_prediction,new.physical.future_prediction,
        atol=1e-6,rtol=1e-6,
    ):
        raise AssertionError("Untrained new source changed future measurement prediction")
    if not torch.allclose(
        old.physical.reconstruction,new.physical.reconstruction,
        atol=1e-6,rtol=1e-6,
    ):
        raise AssertionError("Untrained new source changed physical reconstruction")
    if float(new.decomposition_error())>2e-5:
        raise AssertionError("Native concept decision is no longer exactly additive")
    return float((old.logits-new.logits).abs().max())


def run(out_dir:Path):
    if out_dir.exists():
        raise FileExistsError("Never overwrite concept evidence-admission records")
    rows=[]
    counterfactual={}
    for scenario in RESET_FAMILIES:
        paired=make_v21c_repeat_acquisitions(scenario)
        source_family=paired.source_training_family
        cohort=make_v21_provenance_cohort(source_family)
        train=cohort.discovery.train
        basis=np.eye(len(CHANNELS),2)
        discovery=slice(0,DISCOVERY_N)
        risk_part=slice(DISCOVERY_N,DISCOVERY_N+SCREEN_N)
        measured_part=slice(DISCOVERY_N+SCREEN_N,
                            DISCOVERY_N+SCREEN_N+CORROBORATION_N)
        candidate=propose_observable_residual_concept(
            train.x[discovery],train.mask[discovery],
            basis,CHANNELS,seed=211131,permutations=199,bootstraps=32,
        )
        risk=screen_risk_association(
            candidate,train.x[risk_part],train.mask[risk_part],
            train.y[risk_part],
            screening_episode_ids=train.episode_ids[risk_part],
            discovery_episode_ids=train.episode_ids[discovery],
            seed=211132,permutations=199,bootstraps=199,
        )
        independent=corroborate_candidate_on_disjoint_measured_reference(
            candidate,
            train.x[measured_part],
            cohort.independent_train_reference[measured_part],
            provenance_episode_ids=train.episode_ids[measured_part],
            discovery_episode_ids=train.episode_ids[discovery],
            risk_screen_episode_ids=train.episode_ids[risk_part],
            model_channel_devices=cohort.model_sensor_devices,
            reference_device_id=cohort.reference_device_id,
            seed=211133,permutations=199,bootstraps=199,
        )
        repeat=assess_candidate_after_independent_acquisition_reset(
            candidate,paired,
            discovery_episode_ids=train.episode_ids[discovery],
            outcome_screen_episode_ids=train.episode_ids[risk_part],
            reference_screen_episode_ids=train.episode_ids[measured_part],
            seed=211144,permutations=199,bootstraps=199,
        )
        decision=review_candidate_evidence_for_native_admission(
            candidate,risk,independent,repeat,
            reviewer_source_provenance_choice="unknown",
        )
        if decision.status!=EXPECTED[scenario]:
            raise AssertionError(
                f"Unreported source-admission evidence status in {scenario}: {decision.status}"
            )
        untouched=_new_k2_research_model()
        if decision.can_insert_provisional_native_slot:
            expanded=expand_original_native_model_if_provisionally_measured(
                untouched,candidate,decision,
            )
            drift=_assert_no_drift(untouched,expanded,train)
            policy=int(expanded.physical.signed_risk_policy[-1])
            if decision.status==ALLOW_NO_RISK:
                if policy!=2:
                    raise AssertionError("Descriptive source missing structural zero-risk mask")
                with torch.no_grad():
                    expanded.physical.decision_raw_weight[-1].fill_(2.)
                    if float(expanded.physical.decision_weights()[-1])!=0.:
                        raise AssertionError("Native descriptive risk gate not robust to future weight changes")
            elif policy!=0:
                raise AssertionError("Measured predictive source accidentally sign-certified")
        else:
            drift=0.
            policy=None

        if scenario=="artifact_environmental_persistent":
            # This is a counterfactual independent, externally obtained
            # artifact provenance judgement. It is NOT an actual clinician.
            veto=review_candidate_evidence_for_native_admission(
                candidate,risk,independent,repeat,
                reviewer_source_provenance_choice="instrument_artifact",
            )
            if veto.status!=BLOCK_ARTIFACT or veto.can_insert_provisional_native_slot:
                raise AssertionError("An explicit instrument provenance veto must override all statistics")
            try:
                expand_original_native_model_if_provisionally_measured(
                    untouched,candidate,veto,
                )
            except ValueError:
                pass
            else:
                raise AssertionError("A vetoed source cannot enter native inference graph")
            counterfactual={
                "scenario":scenario,
                "without_independent_source_veto":bounded_evidence_passport(decision),
                "with_explicit_instrument_artifact_veto":bounded_evidence_passport(veto),
                "synthetic_manual_veto_not_an_actual_human_or_doctor":True,
                "physical_cause_still_not_certified":True,
            }

        rows.append({
            "scenario":scenario,
            "candidate_id":candidate.candidate_id,
            "risk_assoc_status":risk.status,
            "independent_device_status":independent.status,
            "paired_acquisition_status":repeat.status,
            "candidate_level_review":"unknown",
            "bounded_admission_status":decision.status,
            "native_provisional_slot_allowed":decision.can_insert_provisional_native_slot,
            "native_risk_training_permitted":decision.can_train_native_risk_after_insertion,
            "new_slot_native_signed_risk_policy":policy,
            "original_inference_drift_max_abs_logit":drift,
            "physical_causation_certified":False,
            "actual_doctor_annotated":False,
            "known_persistent_artifact_false_positive_test":(
                scenario=="artifact_environmental_persistent"
            ),
        })
        print(scenario,
              "bounded decision",decision.status,
              "risk training allowed",decision.can_train_native_risk_after_insertion,
              flush=True)
    if len(rows)!=6 or not counterfactual:
        raise AssertionError("Missing full evidence gate and veto falsification")
    out_dir.mkdir(parents=True,exist_ok=False)
    with (out_dir/"ALL_SIX_CONCEPT_ADMISSION_POLICIES.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    (out_dir/"PERSISTENT_ARTIFACT_HUMAN_VETO_COUNTERFACTUAL.json").write_text(
        json.dumps(counterfactual,indent=2,allow_nan=False)
    )
    status={
        "status":"V21D_BOUNDED_MEASURED_CONCEPT_ADMISSION_AND_ARTIFACT_VETO",
        "n_predeclared_admission_conditions":len(rows),
        "n_accepted_provisional_native_measured_concepts":sum(
            r["native_provisional_slot_allowed"] for r in rows),
        "n_rejected_instrument_suspected_concepts":sum(
            not r["native_provisional_slot_allowed"] for r in rows),
        "n_descriptive_only_native_zero_risk_slots":sum(
            r["new_slot_native_signed_risk_policy"]==2 for r in rows),
        "persistent_artifact_false_positive_explicit":True,
        "counterfactual_artifact_provenance_veto_blocks_native_admission":True,
        "reviewer_was_real_clinician":False,
        "all_numerical_gates_are_observational_or_calibration_evidence":True,
        "no_physical_or_causal_source_certification":True,
        "new_provisional_slots_have_zero_inference_drift_before_training":True,
        "native_decision_path_never_raw_input_bypassed":True,
        "six_case_full_denominator_preserved":True,
        "per_scenario":rows,
        "independent_source_veto_counterfactual":counterfactual,
    }
    (out_dir/"STATUS.json").write_text(json.dumps(status,indent=2,allow_nan=False))
    return status


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",type=Path,required=True)
    options=p.parse_args()
    run(options.out_dir)
