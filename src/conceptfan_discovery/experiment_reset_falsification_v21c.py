"""V21c: execute six frozen paired-acquisition instrument-reset falsifications.

V17 candidate proposal from original V21 TRAIN measurements, disjoint TRAIN
V18 outcome association and independent reference device corroboration.
Then test an INDEPENDENT measured paired-calibration set, never accessing
target labels, hidden source states, or test/validation patient histories.

No auto physical/casual certification is made even if every gate accepts.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .provenance_data_v21 import (
    CHANNELS,DISCOVERY_N,SCREEN_N,CORROBORATION_N,
    make_v21_provenance_cohort,
)
from .instrument_reset_data_v21c import (
    RESET_FAMILIES,CALIBRATION_SEED,INDEPENDENT_RESET_SEED,
    make_v21c_repeat_acquisitions,
)
from .candidate_admission_v17 import (
    propose_observable_residual_concept,candidate_passport,
)
from .risk_screen_v18 import (
    screen_risk_association,risk_evidence_passport,
)
from .measurement_corroboration_v21 import (
    corroborate_candidate_on_disjoint_measured_reference,provenance_passport,
)
from .instrument_repeatability_v21c import (
    assess_candidate_after_independent_acquisition_reset,
    paired_repeatability_passport,
)


def run(outdir:Path):
    if outdir.exists():
        raise FileExistsError("Never overwrite a scientific V21c calibration run")
    rows=[]
    receipts=[]
    for scenario in RESET_FAMILIES:
        pair=make_v21c_repeat_acquisitions(scenario)
        family=pair.source_training_family
        cohort=make_v21_provenance_cohort(family)
        train=cohort.discovery.train
        if train.oracle_states is not None or cohort.discovery.validation.oracle_states is not None:
            raise AssertionError("No source oracle may be available to proposal or reference screens")
        basis=np.eye(len(CHANNELS),2)
        discovery=slice(0,DISCOVERY_N)
        risk_part=slice(DISCOVERY_N,DISCOVERY_N+SCREEN_N)
        provenance_part=slice(DISCOVERY_N+SCREEN_N,
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
        measured=corroborate_candidate_on_disjoint_measured_reference(
            candidate,
            train.x[provenance_part],
            cohort.independent_train_reference[provenance_part],
            provenance_episode_ids=train.episode_ids[provenance_part],
            discovery_episode_ids=train.episode_ids[discovery],
            risk_screen_episode_ids=train.episode_ids[risk_part],
            model_channel_devices=cohort.model_sensor_devices,
            reference_device_id=cohort.reference_device_id,
            seed=211133,permutations=199,bootstraps=199,
        )
        reset=assess_candidate_after_independent_acquisition_reset(
            candidate,pair,
            discovery_episode_ids=train.episode_ids[discovery],
            outcome_screen_episode_ids=train.episode_ids[risk_part],
            reference_screen_episode_ids=train.episode_ids[provenance_part],
            seed=211144,permutations=199,bootstraps=199,
        )
        rows.append({
            "scenario":scenario,
            "underlying_V21_train_generator":family,
            "n_independent_calibration_episodes":len(pair.calibration_episode_ids),
            "most_influential_train_sensor":candidate.top_signed_sensors[0][0],
            "train_sensor_lag_direction_p":candidate.permutation_p,
            "train_risk_screen_status":risk.status,
            "train_risk_association_point_biserial_r":risk.point_biserial_r,
            "separate_reference_status":measured.status,
            "separate_reference_correlated_r":measured.observed_signed_pearson_r,
            "controlled_acquisition_repeatability_status":reset.status,
            "paired_repeat_measurement_r":reset.first_repeat_pearson_r,
            "paired_repeat_permutation_p":reset.permutation_p_two_sided,
            "paired_repeat_95pct_bootstrap_ci_lower":reset.bootstrap_ci95[0],
            "paired_repeat_95pct_bootstrap_ci_upper":reset.bootstrap_ci95[1],
            "paired_first_half_r":reset.paired_two_half_correlations[0],
            "paired_second_half_r":reset.paired_two_half_correlations[1],
            "first_acquisition_sensor_vs_reference_r":reset.before_reference_signed_r,
            "after_reset_sensor_vs_reference_r":reset.after_reference_signed_r,
            "paired_reference_repeat_r":reset.external_reference_repeat_r,
            "predictive_relevance_is_not_physical_identity":True,
            "instrument_reset_repeatability_is_not_physical_identity":True,
            "actual_patient_diagnosis_or_causation_proven":False,
        })
        receipts.append({
            "scenario":scenario,
            "candidate":candidate_passport(candidate),
            "outcome_risk":risk_evidence_passport(risk),
            "cross_device":provenance_passport(measured),
            "paired_instrument_reset":paired_repeatability_passport(reset),
            "source_cohort_size":len(train.y),
            "calibration_data_contains_hidden_true_latent_states":False,
            "same_underlying_physical_state_is_simulator_intervention":True,
        })
        print(scenario,
              "risk",risk.status,
              "device",measured.status,
              "reset",reset.status,
              "paired-r",round(reset.first_repeat_pearson_r,5),
              flush=True)
    if len(rows)!=6:
        raise AssertionError("Missing a predeclared persistent-artifact negative control")
    outdir.mkdir(parents=True,exist_ok=False)
    with (outdir/"ALL_SIX_INSTRUMENT_RESET_CONTROLS.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (outdir/"FULL_SOURCE_PROVENANCE_AUDIT.json").write_text(
        json.dumps(receipts,indent=2,allow_nan=False)
    )
    receipt={
        "status":"V21C_DISJOINT_INSTRUMENT_RESET_PAIRED_SOURCE_AUDIT",
        "n_predeclared_calibration_scenarios":len(rows),
        "n_observed_paired_calibration_episodes_each":240,
        "external_calibration_seed":CALIBRATION_SEED,
        "instrument_reset_seed":INDEPENDENT_RESET_SEED,
        "no_val_test_oracle_source_in_calibration":True,
        "calibration_only_uses_train_fitted_normalizer":True,
        "observed_paired_readouts_only_no_target_oracle":True,
        "source_risk_reference_calibration_cohorts_all_disjoint":True,
        "persistent_commonmode_negative_case_retained":True,
        "real_clinical_or_physical_identity_claimed":False,
        "new_physician_labels_provided":False,
        "real_icu_dataset_evaluated":False,
        "all_six_scenarios_reported":True,
        "results":rows,
        "full_source_screen_passports":receipts,
    }
    (outdir/"STATUS.json").write_text(json.dumps(receipt,indent=2,allow_nan=False))
    return receipt


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args()
    run(args.out_dir)
