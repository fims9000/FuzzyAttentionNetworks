"""V43 external real paired ON/OFF + nuisance control -> JSON, no simulator.

Must provide 8 arrays in NPZ:
  on, off, observed_on, observed_off                     [N,16,12]
  nuisance_on, nuisance_off, observed_nuisance_on,
  observed_nuisance_off                                 [N,16]
The nuisance-on/OFF measurements are an **independent actual instrument**,
not invented by this software. Source ON/OFF and monitor must be matched
to the SAME subject/time. Their physical origin and channel gain are
assumed; impossible to verify from an NPZ file alone.

No outcome Y, TEST source truth E/S/H, independent physical energy R,
or generation/model simulator is imported, reconstructed or assumed
to be observed. Historical R mapping and live nuisance monitor are
read from separate TRAIN/VALID-only JSON audit snapshots.

Usage:
PYTHONPATH=src:. python scripts/infer_live_nuisance_reference_v43.py \
  --reference-model JSON --monitor-model JSON \
  --input-npz paired_real_observations_and_controls.npz \
  --output-json physical_source_with_current_monitor.json
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

from conceptfan_discovery.physical_reference_certificate_v42 import (
    load_physical_reference_v42,predict_physical_reference_v42,
    measured_external_on_off_difference_v42,
)
from conceptfan_discovery.live_nuisance_monitor_v43 import (
    load_live_monitor_v43,evaluate_live_nuisance_monitor_v43,
    gate_physical_reference_v43,
)

REQUIRED={
    "on","off","observed_on","observed_off",
    "nuisance_on","nuisance_off",
    "observed_nuisance_on","observed_nuisance_off",
}


def infer_live_nuisance_v43(
    *,
    reference_model:Path,monitor_model:Path,
    input_npz:Path,output_json:Path,
)->dict:
    if Path(output_json).exists():
        raise FileExistsError("Refuse to overwrite existing source research result")
    with np.load(input_npz,allow_pickle=False) as inputs:
        if set(inputs.files)!=REQUIRED:
            raise ValueError("Must contain EXACTLY 8 measured ON/OFF and nuisance arrays")
        d={k:inputs[k] for k in REQUIRED}
    n=len(d["on"])
    if not 1<=n<=10000:
        raise ValueError("External V43 input cohort 1..10000")
    ref=load_physical_reference_v42(reference_model)
    monitor=load_live_monitor_v43(monitor_model)
    measured,mask=measured_external_on_off_difference_v42(
        d["on"],d["off"],d["observed_on"],d["observed_off"])
    old=predict_physical_reference_v42(ref,measured,mask)
    live=evaluate_live_nuisance_monitor_v43(
        monitor,d["nuisance_on"],d["nuisance_off"],
        d["observed_nuisance_on"],d["observed_nuisance_off"])
    current=gate_physical_reference_v43(old,live)
    if len(current["physical_reference_allowed"])!=n:
        raise AssertionError("Physical instrument time alignment/patient IDs invalid")
    records=[]
    for i in range(n):
        yes=bool(current["physical_reference_allowed"][i])
        historical=bool(old["physical_reference_allowed"][i])
        mon_accept=bool(live["accepted_matching_background"][i])
        if yes!=(historical and mon_accept):
            raise AssertionError("Current nuisance gate bypassed current alarm")
        energy=current["physical_reference_estimated_energy"][i]
        records.append({
            "input_episode_index":i,
            "physical_reference_model_available_after_current_live_check":yes,
            "estimated_independent_noisy_R_energy":(
                float(energy) if yes else None),
            "nominal_interval_for_historical_NOISY_R_only":(
                [float(current["physical_reference_interval_lower"][i]),
                 float(current["physical_reference_interval_upper"][i])]
                if yes else None),
            "historical_independent_reference_VALID1_VALID2_allowed":historical,
            "current_independent_nuisance_monitor_passed":mon_accept,
            "measured_nuisance_ON_OFF_difference_RMS":float(
                live["current_MONITOR_RMS"][i]),
            "measured_nuisance_timepoints":int(
                live["current_MONITOR_joint_timesteps"][i]),
            "selected_physically_observed_ON_OFF_sensor_pair":(
                str(current["selected_pair"][i])
                if current["support"][i]>0 else None),
            "selected_pair_joint_measured_times":int(
                current["selected_joint_measured_times"][i]),
            "single_sensor_pair_uncorroborated":bool(
                current["single_pair_uncorroborated"][i]),
            "multiple_sensor_pairs_currently_agree":bool(
                current["two_or_three_pairs_corroborated"][i]),
            "no_source_or_nuisance_values_synthesized":True,
            "physical_current_monitor_result":str(
                live["decision_reason"][i]),
            "historical_reference_result":str(
                old["physical_reference_reason"][i]),
            "final_physical_estimate_decision":str(
                current["physical_reference_reason"][i]),
            "external_reference_R_UNAVAILABLE_at_inference":True,
            "nuisance_monitor_and_ON_OFF_source_switch_externally_assumed":True,
            "does_not_guarantee_sensor_integrity_or_causal_semantics":True,
        })
    payload={
        "schema":"conceptfan-v43-real-nuisance-controlled-source-v1",
        "research_only_NOT_medical_diagnostic":True,
        "input_episode_count":n,
        "accepted_physical_estimate_count":int(
            current["physical_reference_allowed"].sum()),
        "nominal_interval_targets":"historical noisy independent reference R, not guaranteed true E",
        "new_external_physical_nuisance_monitor_required":True,
        "source_path_isolation_quality_not_proven_by_nuisance_only":True,
        "patient_results":records,
    }
    path=Path(output_json)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(
        payload,indent=2,ensure_ascii=False,allow_nan=False))
    return payload


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--reference-model",type=Path,required=True)
    p.add_argument("--monitor-model",type=Path,required=True)
    p.add_argument("--input-npz",type=Path,required=True)
    p.add_argument("--output-json",type=Path,required=True)
    args=p.parse_args()
    data=infer_live_nuisance_v43(
        reference_model=args.reference_model,
        monitor_model=args.monitor_model,
        input_npz=args.input_npz,
        output_json=args.output_json)
    print(json.dumps({
        "event":"V43_INDEPENDENT_LIVE_NUISANCE_EXTERNAL_INFERENCE",
        "n_subjects":data["input_episode_count"],
        "physical_energy_estimates_issued":data[
            "accepted_physical_estimate_count"],
        "not_clinically_certified":True,
    }))
