"""V38 COMPLETE model REOPEN/inference QA from frozen published artifacts.

Recreates only the registered synthetic observed device stream, reads the
source fit from safe JSON, loads actual K2 Transformer weights and the
TRAIN-only two-concept head/extra gamma + two-VALID admission from saved
bundle, then predicts TEST x from scratch. NEVER reads outcome TEST labels
or retrains anything; the archived risk probabilities are used ONLY for
comparison after the forward pass. This tests that CI results are really
deployable through a persist/reopen API, not just ephemeral summary scores.

Usage:
 PYTHONPATH=src:. python scripts/replay_exogenous_source_v38.py \
   --evidence-dir V38_EVIDENCE \
   --out-file V38_EVIDENCE/REOPEN_INFERENCE_AUDIT.json
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit

from conceptfan_discovery.exogenous_probe_world_v38 import (
    DEVICE_SEEDS,CASES,STRESSES,
    generate_exogenous_world_v38,apply_artifact_v38,mask_world_v38,
)
from conceptfan_discovery.source_persistence_v38 import (
    load_source_state_v38,
)
from conceptfan_discovery.exogenous_probe_gate_v38 import (
    source_with_independent_evidence_v38,
)
from conceptfan_discovery.auto_measured_concept_admission import (
    admitted_test_logit,
)
from scripts.experiment_nonlinear_anchor_repair_v30 import make_firewalled_k2
from scripts.experiment_trained_conceptfan_omission_v25 import (
    predict_native,sha_state,
)

SEEDS=(11,23,37)
CASES_TO_REPLAY=(
    ("healthy","MCAR20"),
    ("monitored_AB_common_noise","MCAR20"),
    ("UNMONITORED_ABC_noise","MCAR20"),
    ("healthy","missing_sentinel"),
    ("healthy","missing_C"),
)


def _mask(x,stress,*,device_seed,extra_seed):
    return mask_world_v38(
        x,stress,seed=extra_seed+device_seed%101,
        known_seed=20263878+device_seed%101)


def check_all_reopen_inferences_v38(evidence_dir:Path,out_file:Path):
    if out_file.exists():
        raise FileExistsError("Do not overwrite immutable model-replay QA")
    state=json.loads((evidence_dir/"STATUS.json").read_text())
    manifests=json.loads((evidence_dir/"INFERENCE_BUNDLES.json").read_text())
    archive=np.load(evidence_dir/"FROZEN_TEST_PREDICTIONS.npz",allow_pickle=False)
    if (state["status"]!="V38_INSTRUMENTED_CONCEPT_EXECUTED"
        or tuple(state["devices"])!=DEVICE_SEEDS
        or state["n_saved_outcome_specific_inference_bundles"]!=60):
        raise ValueError("Only fully frozen, reopened V38 benchmark")
    failures=[]
    receipts=[]
    worlds={}
    for device in DEVICE_SEEDS:
        # Synthetic generator's internal hidden H is not used by the model.
        # Reproducing deterministic TEST inputs is necessary because raw
        # patient histories are intentionally not saved in the evidence zip.
        world=generate_exogenous_world_v38(
            device_seed=device,
            n_train=state["n_train"],
            n_valid=state["n_valid1"]+state["n_valid2"],
            n_test=state["n_test"],
            per_actuator=state["n_independent_probe_pairs_per_type"])
        worlds[device]=world
        reopened=load_source_state_v38(
            evidence_dir/"SOURCE_STATES"/f"device{device}_V38_source.json")
        for seed in SEEDS:
            actual_bundle=[v for v in manifests if (
                v["device_seed"],v["NN_seed"],v["outcome_case"]
            )==(device,seed,"healthy")]
            if len(actual_bundle)!=1:
                raise AssertionError("Missing original TRAIN outcome head")
            snapshot=torch.load(
                evidence_dir/actual_bundle[0]["native_checkpoint"],
                map_location="cpu",weights_only=True)
            original=make_firewalled_k2(seed)
            original.load_state_dict(snapshot["state_dict"],strict=True)
            original.eval()
            if sha_state(original)!=actual_bundle[0]["frozen_old_K2_weights_SHA"]:
                raise AssertionError("Stored K2 Transformer differs from frozen source checkpoint")
            for case,stress in CASES_TO_REPLAY:
                k=(device,seed,case)
                bundles=[v for v in manifests if (
                    v["device_seed"],v["NN_seed"],v["outcome_case"]
                )==k]
                if len(bundles)!=1:
                    raise AssertionError("Missing saved TRAIN two-concept head / independent VALID gate")
                b=bundles[0]
                index=next(i for i,z in enumerate(CASES) if z[0]==case)
                test_fault=CASES[index][3]
                x=apply_artifact_v38(
                    world["x"][world["test"]],test_fault,
                    seed=20263880+index+device%101)
                stress_idx=STRESSES.index(stress)
                mask=_mask(x,stress,device_seed=device,
                           extra_seed=20263911+stress_idx)
                q=predict_native(original,x[:,:,:4],
                                 mask[:,:,:4])["concept_summary"]
                head=b["two_concept_original_TRAIN_head"]
                base=float(head["bias"])+q@np.asarray(
                    head["weights"],dtype=np.float64)
                source=source_with_independent_evidence_v38(
                    reopened.source,reopened.monitor,reopened.probe,x,mask)
                logits,used=admitted_test_logit(
                    base,source["q"],source["support"],
                    gamma=b["new_source_gamma_TRAIN_only"],
                    approved=b["new_source_passed_two_independent_VALID"],
                    minimum_fraction=.50)
                np.testing.assert_array_equal(logits[~used],base[~used])
                prediction=expit(logits)
                saved=archive[
                    f"device{device}_{case}_{stress}_ABC_probe_monitor_seed{seed}"
                ]
                if len(saved)!=len(prediction):
                    raise AssertionError("Saved record identifiers mismatch")
                max_err=float(np.max(abs(saved-prediction)))
                receipts.append({
                    "device_seed":device,"NN_seed":seed,
                    "case":case,"stress":stress,
                    "max_abs_difference_saved_vs_reopened_probs":max_err,
                    "source_use":float(used.mean()),
                    "original_native_model_sha_validated":True,
                    "saved_source_calibration_thresholds_reused":True,
                    "no_TEST_outcomes_read":True,
                })
                if max_err>1e-6:
                    failures.append((device,seed,case,stress,max_err))
    if len(manifests)!=60 or len(receipts)!=len(DEVICE_SEEDS)*len(SEEDS)*len(CASES_TO_REPLAY):
        raise AssertionError("V38 outcome-head persistence incomplete")
    if failures:
        raise AssertionError("Reopened risk score differs: "+str(failures))
    result={
        "status":"V38_REOPENED_FULL_NATIVE_AND_SOURCE_INFERENCE_VERIFIED",
        "n_outcome_specific_bundles":len(manifests),
        "n_reopened_real_K2_and_source_forward_passes":len(receipts),
        "max_abs_reopen_prediction_difference":max(
            x["max_abs_difference_saved_vs_reopened_probs"] for x in receipts),
        "no_dataset_labels_read_for_replay":True,
        "source_model_loaded_from_safe_JSON":True,
        "original_native_K2_loaded_from_snapshot":True,
        "no_retraining_or_TEST_calibration":True,
        "receipts":receipts,
    }
    out_file.parent.mkdir(parents=True,exist_ok=True)
    out_file.write_text(json.dumps(
        result,indent=2,ensure_ascii=False,allow_nan=False))
    print(json.dumps({
        "event":"V38_REOPEN_QA","n_frozen_bundles":len(manifests),
        "actual_forward_passes":len(receipts),
        "max_absolute_prediction_error":
            result["max_abs_reopen_prediction_difference"],
    }),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--evidence-dir",type=Path,required=True)
    parser.add_argument("--out-file",type=Path,required=True)
    args=parser.parse_args()
    check_all_reopen_inferences_v38(args.evidence_dir,args.out_file)
