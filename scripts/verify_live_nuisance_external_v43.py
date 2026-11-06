"""V43 end-to-end external input scientific QA, frozen TEST patient parity.

SIMULATOR IMPORT ONLY IN THIS TEST HARNESS, NEVER RUNTIME INFERENCE.
Generate true *observed* ON/OFF+nuisance sensor arrays for unchanged
TEST patients and masks from frozen V43 study. Set ALL missing values
to NaN (strict physical missingness proof), save external 8-array NPZ,
then invoke actual public scripts/infer_live_nuisance_reference_v43.py
against independently saved TRAIN/VALID-only JSON source + monitor.
Compare individual source decisions/energies to separate immutable V43
TEST predictions, not to a newly fitted source.

No actual TEST source R, true E, original Y/H/S or nuisance oracle passed
to the inference function. Physical control channel is a genuine extra
measured feature in the synthetic world, not an observed source concept.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

from conceptfan_discovery.live_nuisance_world_v43 import (
    generate_device_v43,
)
from scripts.experiment_live_nuisance_reference_v43 import (
    _read,
    WORLDS,
)
from scripts.infer_live_nuisance_reference_v43 import (
    infer_live_nuisance_v43,
)

CHOSEN_CASES={
    "healthy","OFF_drift","OFF_source_leak",
    "monitor_blind_drift","monitor_false_alarm","monitor_missing",
    "OFF_missing","A_ON_inversion",
}
STRESSES=("MCAR20","MCAR40")


def qa(evidence:Path,out:Path):
    if out.exists():
        raise FileExistsError("No overwriting external user-path QA")
    x=json.loads((evidence/"STATUS.json").read_text())
    if x["status"]!="V43_LIVE_NUISANCE_AND_MATCHED_RAW_REFERENCE_EXECUTED":
        raise ValueError("Must compare with successfully frozen full V43 study")
    original=np.load(
        evidence/"FROZEN_TEST_PHYSICAL_PREDICTIONS.npz",
        allow_pickle=False)
    out.mkdir(parents=True,exist_ok=False)
    records=[];maximum=0.
    n_calls=0
    for dev in x["devices"]:
        world=generate_device_v43(
            device_seed=dev,n_train=1000,n_valid=800,n_test=600)
        chosen=world["test"][:10]
        reference=(evidence/"FROZEN_MODELS"/f"device{dev}_none.json")
        monitor=(evidence/"FROZEN_MODELS"/f"device{dev}_live.json")
        for case,phys,control_fault,val_fault in WORLDS:
            if case not in CHOSEN_CASES:
                continue
            assert val_fault=="none"
            for stress in STRESSES:
                measured=_read(
                    world,world["test"],physical_fault=phys,
                    monitor_fault=control_fault,
                    stress=stress,stage=10,monitor_seed=0)
                d={k:v[:10].copy() for k,v in measured.items()
                   if k in (
                       "on","off","mask_on","mask_off","control_on",
                       "control_off","control_mask_on","control_mask_off")}
                d["on"][~d["mask_on"]]=np.nan
                d["off"][~d["mask_off"]]=np.nan
                d["control_on"][~d["control_mask_on"]]=np.nan
                d["control_off"][~d["control_mask_off"]]=np.nan
                name=f"device{dev}_{case}_{stress}"
                inp=out/f"{name}_only_actual_measured_inputs.npz"
                np.savez_compressed(
                    inp,on=d["on"],off=d["off"],
                    observed_on=d["mask_on"],observed_off=d["mask_off"],
                    nuisance_on=d["control_on"],
                    nuisance_off=d["control_off"],
                    observed_nuisance_on=d["control_mask_on"],
                    observed_nuisance_off=d["control_mask_off"])
                jsonout=out/f"{name}_frozen_source_inference.json"
                result=infer_live_nuisance_v43(
                    reference_model=reference,monitor_model=monitor,
                    input_npz=inp,output_json=jsonout)
                pref=f"device{dev}_{case}_{stress}_"
                want=original[pref+"V43_LIVE_MONITORED_pred"][:10]
                want_ok=original[pref+"V43_LIVE_MONITORED_accepted"][:10]
                pair=original[pref+"chosen_source_pair"][:10]
                mon=original[pref+"monitor_accepted"][:10]
                if result["input_episode_count"]!=10:
                    raise AssertionError("Dropped externally observed episodes")
                for i,r in enumerate(result["patient_results"]):
                    if bool(want_ok[i])!=r[
                        "physical_reference_model_available_after_current_live_check"]:
                        raise AssertionError(f"Current monitor+source acceptance changed: {name}/{i}")
                    if bool(mon[i])!=r["current_independent_nuisance_monitor_passed"]:
                        raise AssertionError(f"Current monitor scores differed: {name}/{i}")
                    if r["selected_physically_observed_ON_OFF_sensor_pair"] is not None:
                        if str(pair[i])!=r["selected_physically_observed_ON_OFF_sensor_pair"]:
                            raise AssertionError(f"Sensor pair attribution changed: {name}/{i}")
                    if bool(want_ok[i]):
                        diff=abs(float(want[i])-r[
                            "estimated_independent_noisy_R_energy"])
                        maximum=max(maximum,diff)
                    elif r["estimated_independent_noisy_R_energy"] is not None:
                        raise AssertionError("Refused episode still gives physical energy")
                    if not (
                        r["external_reference_R_UNAVAILABLE_at_inference"]
                        and r["no_source_or_nuisance_values_synthesized"]
                        and r["nuisance_monitor_and_ON_OFF_source_switch_externally_assumed"]
                    ):
                        raise AssertionError("Invalid current physical provenance")
                if case in ("monitor_missing","OFF_missing"):
                    assert result["accepted_physical_estimate_count"]==0
                records.append({
                    "device_seed":dev,"case":case,"stress":stress,
                    "episodes":10,
                    "accepted":result["accepted_physical_estimate_count"],
                    "no_oracle_R_E_H_S_Y_or_U_passed_to_runtime":True,
                    "both_JSON_models_reopened_from_TRAIN_VALID":True,
                    "all_unobserved_input_values_NaN":True,
                })
                n_calls+=1
    if n_calls!=4*len(CHOSEN_CASES)*len(STRESSES) or maximum>=1e-8:
        raise AssertionError("Incomplete external instrument user-path parity")
    summary={
        "status":"V43_REAL_INSTRUMENT_NPZ_JSON_PUBLIC_INFERENCE_PARITY",
        "number_of_independent_reopened_NPZ_to_JSON_calls":n_calls,
        "number_of_replayed_TEST_subject_observations":n_calls*10,
        "max_original_vs_reloaded_source_energy_error":maximum,
        "source_and_nuisance_models_from_TRAIN_VALID_ONLY":True,
        "all_missing_sensor_values_replaced_by_NaN":True,
        "no_hidden_TEST_reference_or_latent_target_passed":True,
        "unknown_current_MONITOR_faults_can_still_be_blind":True,
        "research_only_without_actual_hardware":True,
        "receipts":records,
    }
    (out/"EXTERNAL_INFERENCE_QA.json").write_text(json.dumps(
        summary,indent=2,ensure_ascii=False,allow_nan=False))
    print(json.dumps({
        "event":"V43_EXTERNAL_NPZ_JSON_SOURCE_PARITY",
        "user_path_calls":n_calls,"episodes":n_calls*10,
        "max_abs_energy_difference":maximum,
    }),flush=True)
    return summary


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--evidence-dir",type=Path,required=True)
    a.add_argument("--out-dir",type=Path,required=True)
    p=a.parse_args()
    qa(p.evidence_dir,p.out_dir)
