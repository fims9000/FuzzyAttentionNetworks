"""V42 research-only external ON/OFF NPZ -> JSON physical-source inference.

Input NPZ: exact four arrays,
  on            [N,16,12] actual ON readings
  off           [N,16,12] actually measured separate OFF readings
  observed_on   [N,16,12] strictly numpy BOOL real observation mask
  observed_off  [N,16,12] strictly numpy BOOL real observation mask

No TEST reference R, latent truth S/H, Y, or simulator required. The
saved JSON bundle includes TRAIN source fits and separately privileged
independent-reference calibration knots from TRAIN/VALID only.
An independent real-world reference device and physically functioning
source-off switch are still ASSUMED, not verified by this CLI.

Output is a MODEL ESTIMATE of noisy reference energy, NOT clinical risk,
a causal effect, proof of current sensor correctness or patient diagnosis.

PYTHONPATH=src:. python scripts/infer_physical_reference_v42.py \
  --reference-model V42_EVIDENCE/REFERENCE_MODELS/device20261607_none_physical_reference_v42.json \
  --input-npz paired_observations.npz \
  --output-json paired_source_estimates.json
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

from conceptfan_discovery.physical_reference_certificate_v42 import (
    load_physical_reference_v42,predict_physical_reference_v42,
    measured_external_on_off_difference_v42,
)


def inference_physical_reference_npz_v42(
    *,model_path:Path,input_npz:Path,output_json:Path,
)->dict:
    if Path(output_json).exists():
        raise FileExistsError("Will never overwrite existing reference inference")
    if not Path(input_npz).is_file():
        raise FileNotFoundError("Absent actual paired instrument ON/OFF observation NPZ")
    with np.load(input_npz,allow_pickle=False) as bundle:
        if set(bundle.files)!={
            "on","off","observed_on","observed_off"}:
            raise ValueError("Expected exactly on/off/observed_on/observed_off")
        x,z=bundle["on"],bundle["off"]
        ma,mb=bundle["observed_on"],bundle["observed_off"]
    if (x.ndim!=3 or x.shape[1:]!=(16,12) or
        not 1<=len(x)<=10000 or
        z.shape!=x.shape or ma.shape!=x.shape or mb.shape!=x.shape or
        ma.dtype!=np.bool_ or mb.dtype!=np.bool_ or
        not np.issubdtype(x.dtype,np.number) or
        not np.issubdtype(z.dtype,np.number) or
        not np.isfinite(x[ma]).all() or
        not np.isfinite(z[mb]).all()):
        raise ValueError(
            "Malformed paired [N,16,12] finite observed readings and strict BOOL masks")
    model=load_physical_reference_v42(model_path)
    diff,observed=measured_external_on_off_difference_v42(x,z,ma,mb)
    f=predict_physical_reference_v42(model,diff,observed)
    result=[]
    for i in range(len(x)):
        approved=bool(f["physical_reference_allowed"][i])
        source_statistic=bool(f["support"][i]>0)
        if approved and not source_statistic:
            raise AssertionError("Unmeasured source illegally acquired physical value")
        reason=str(f["physical_reference_reason"][i])
        result.append({
            "index_in_input_batch":i,
            "historically_calibrated_source_value_available":approved,
            "model_estimated_physical_reference_energy":(
                float(f["physical_reference_estimated_energy"][i])
                if approved else None),
            "nominal_90pct_interval_for_NOISY_REFERENCE_R_only":(
                [float(f["physical_reference_interval_lower"][i]),
                 float(f["physical_reference_interval_upper"][i])]
                if approved else None),
            "uncalibrated_V41_observational_q_NOT_energy":(
                float(f["q"][i]) if source_statistic else None),
            "selected_real_ON_OFF_sensor_pair":(
                str(f["selected_pair"][i]) if source_statistic else None),
            "observed_joint_ON_OFF_timesteps":int(
                f["selected_joint_measured_times"][i]),
            "single_pair_not_independently_confirmed_for_this_episode":
                bool(f["single_pair_uncorroborated"][i]),
            "multi_pair_agreement_on_this_episode":
                bool(f["two_or_three_pairs_corroborated"][i]),
            "source_measurement_decision":str(f["decision_reason"][i]),
            "physical_reference_decision":reason,
            "external_sensor_wiring_and_source_OFF_switch_assumed":True,
            "independent_reference_was_AVAILABLE_only_on_TRAIN_VALID":True,
            "independent_reference_R_NOT_provided_at_inference":True,
            "realworld_physical_identity_or_current_fault_exclusion_PROVEN":False,
            "clinical_or_causal_claim":False,
        })
    payload={
        "schema":"conceptfan-v42-external-reference-inference-v1",
        "research_only_not_clinical_diagnostic":True,
        "target_of_interval":"noisy calibrated independent reference R, not latent E",
        "physical_view_independence_and_ON_OFF_source_isolation_externally_assumed":True,
        "number_of_input_episodes":len(result),
        "physical_source_available_count":int(
            sum(q["historically_calibrated_source_value_available"]
                for q in result)),
        "episodes":result,
    }
    path=Path(output_json)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(
        payload,indent=2,ensure_ascii=False,allow_nan=False))
    return payload


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--reference-model",type=Path,required=True)
    p.add_argument("--input-npz",type=Path,required=True)
    p.add_argument("--output-json",type=Path,required=True)
    opts=p.parse_args()
    r=inference_physical_reference_npz_v42(
        model_path=opts.reference_model,input_npz=opts.input_npz,
        output_json=opts.output_json)
    print(json.dumps({
        "status":"V42_EXTERNAL_PAIRED_REFERENCE_INFERENCE_COMPLETE",
        "n_episodes":r["number_of_input_episodes"],
        "n_reference_estimates":r["physical_source_available_count"],
        "synthetic_research_only":True,
    }))
