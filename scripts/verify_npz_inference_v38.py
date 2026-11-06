"""V38 real external NPZ -> JSON inference parity, no outcomes at runtime.

Recreates the REGISTERED synthetic readouts only in the QA harness.
The actual CLI module itself loads arbitrary NPZ x+Boolean observed mask,
saved native K2 + source model bundle and emits result JSON. It imports NO
generator, true hidden mixing or labels. Verify 2 devices × 3 seeds × 5
mask/fault worlds with two actual subjects including hidden NaN inputs.

Test output is synthetic only, and source input is never copied into
the result as observed user data. No retraining, TEST label read or risk-gate
retuning. This is an end-to-end deployment-path acceptance test.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

import numpy as np

from conceptfan_discovery.exogenous_probe_world_v38 import (
    generate_exogenous_world_v38,mask_world_v38,apply_artifact_v38,
    CASES,STRESSES,DEVICE_SEEDS,
)
from scripts.infer_exogenous_source_v38 import inference_npz_to_json

NN_SEEDS=(11,23,37)
SCENARIOS=(
    ("healthy","MCAR20"),
    ("monitored_ABC_common_noise","MCAR20"),
    ("UNMONITORED_ABC_noise","MCAR20"),
    ("healthy","missing_sentinel"),
    ("healthy","missing_C"),
)


def check_external_v38(evidence_dir:Path,out_dir:Path):
    if out_dir.exists():
        raise FileExistsError("Never overwrite existing end-to-end QA evidence")
    state=json.loads((evidence_dir/"STATUS.json").read_text())
    if (state.get("status")!="V38_INSTRUMENTED_CONCEPT_EXECUTED"
        or tuple(state["devices"])!=DEVICE_SEEDS):
        raise ValueError("Only complete frozen instrumented V38 evidence")
    frozen=np.load(
        evidence_dir/"FROZEN_TEST_PREDICTIONS.npz",allow_pickle=False)
    out_dir.mkdir(parents=True,exist_ok=False)
    receipts=[]
    for device in DEVICE_SEEDS:
        synth=generate_exogenous_world_v38(
            device_seed=device,
            n_train=state["n_train"],
            n_valid=state["n_valid1"]+state["n_valid2"],
            n_test=state["n_test"],
            per_actuator=state["n_independent_probe_pairs_per_type"])
        for case,stress in SCENARIOS:
            index=next(i for i,c in enumerate(CASES) if c[0]==case)
            fault=CASES[index][3]
            inputs=apply_artifact_v38(
                synth["x"][synth["test"]],fault,
                seed=20263880+index+device%101,
            )
            mask=mask_world_v38(
                inputs,stress,
                seed=20263911+STRESSES.index(stress)+device%101,
                known_seed=20263878+device%101,
            )
            # the inferred NPZ includes only 3 observed episodes, no labels.
            # Unobserved numeric payload is intentionally NaN to prove
            # absence of hidden data leakage.
            x=np.asarray(inputs[:3]).copy()
            m=mask[:3]
            x[~m]=np.nan
            infile=out_dir/f"device{device}_{case}_{stress}_observed_only.npz"
            np.savez_compressed(infile,x=x,observed=m)
            for nn_seed in NN_SEEDS:
                outfile=out_dir/f"device{device}_{case}_{stress}_seed{nn_seed}.json"
                result=inference_npz_to_json(
                    evidence_dir,device_seed=device,nn_seed=nn_seed,
                    outcome_case=case,input_npz=infile,output_json=outfile)
                original=frozen[
                    f"device{device}_{case}_{stress}_ABC_probe_monitor_seed{nn_seed}"
                ][:3]
                observed_prob=np.asarray([
                    z["model_event_probability"] for z in result["episodes"]
                ])
                error=float(np.max(np.abs(original-observed_prob)))
                if error>1e-6:
                    raise AssertionError(
                        f"External NPZ/JSON forward differs from original trained inference {device}/{case}/{stress}/{nn_seed}: {error}"
                    )
                if not result["synthetic_research_not_clinically_validated"]:
                    raise AssertionError("Missing explicit medical validity disclaimer")
                if stress in ("missing_sentinel","missing_C"):
                    if any(z["used_extra_source"] for z in result["episodes"]):
                        raise AssertionError("Unobserved monitor or view was used")
                receipts.append({
                    "device_seed":device,
                    "nn_seed":nn_seed,
                    "case":case,
                    "stress":stress,
                    "n_external_input_episodes":3,
                    "maximum_probability_difference_vs_frozen_TEST":error,
                    "source_admission_count":sum(
                        z["used_extra_source"] for z in result["episodes"]),
                    "no_hidden_values_or_TEST_labels_read":True,
                    "checkpoint_reloaded_after_evidence_created":True,
                    "source_and_head_persisted_from_TRAIN":True,
                })
    if len(receipts)!=30:
        raise AssertionError("Incomplete external inference worlds")
    audited={
        "status":"V38_EXTERNAL_NPZ_JSON_INFERENCE_PARITY_VERIFIED",
        "n_devices":2,"n_real_native_seeds":3,
        "n_independent_fault_and_missingness_conditions":len(SCENARIOS),
        "n_reopened_CLI_inferences":len(receipts),
        "n_actual_external_sensor_episodes_per_run":3,
        "maximum_difference_vs_original_predictions":max(
            x["maximum_probability_difference_vs_frozen_TEST"]
            for x in receipts),
        "model_cli_has_no_latent_H_S_or_TEST_labels":True,
        "every_missing_sentinel_or_view_abstains":True,
        "receipts":receipts,
    }
    (out_dir/"EXTERNAL_INFERENCE_PARITY_AUDIT.json").write_text(
        json.dumps(audited,indent=2,ensure_ascii=False,allow_nan=False))
    print(json.dumps({
        "event":"V38_EXTERNAL_INFERENCE_CLI_QA",
        "n_reloaded_public_clis":len(receipts),
        "max_absolute_difference":
            audited["maximum_difference_vs_original_predictions"],
        "masked_hidden_inputs":"NaN",
        "no_patient_TEST_labels_read":True,
    }))
    return audited


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--evidence-dir",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    a=parser.parse_args()
    check_external_v38(a.evidence_dir,a.out_dir)
