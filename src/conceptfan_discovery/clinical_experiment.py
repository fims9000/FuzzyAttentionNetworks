"""Clinical exploratory run: discover concepts jointly, not from predefined proxies.

Validation is split again into cardinality-selection and independent
cardinality-audit halves. No calibration/test patient is evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .evaluation import (
    compare_independent_discoveries, concept_passports, evaluate_discovery,
)
from .experiment import _run_one
from .model import PredictiveDynamicsConceptFAN
from .physionet import prepare_physionet_discovery
from .selection import select_concept_cardinality, split_validation_for_cardinality


def json_write(path: Path, content: dict | list) -> None:
    path.write_text(json.dumps(content, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--raw-zip", type=Path)
    p.add_argument("--outcomes", type=Path)
    args = p.parse_args()
    if args.out_dir.exists():
        p.error("Refusing to overwrite a prior analysis")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if cfg["protocol"]["test_locked"] is not True or cfg["reporting"]["no_test_evaluation"] is not True:
        p.error("Unsealed test protocol forbidden")
    if set(cfg["training"]["modes"]) != {"task_only", "task_future", "full"}:
        p.error("All predefined three-armed ablations must be run")
    root = Path(__file__).resolve().parents[2]
    raw_zip = args.raw_zip or root / cfg["data"]["set_a_zip"]
    outcome = args.outcomes or root / cfg["data"]["outcomes_a"]
    with tempfile.TemporaryDirectory(prefix="pd_conceptfan_private_split_") as tmp:
        data = prepare_physionet_discovery(
            raw_zip, outcome, Path(tmp),
            split_seed=int(cfg["data"]["split_seed"]),
            input_window=int(cfg["data"]["input_window"]),
            forecast_horizon=int(cfg["data"]["forecast_horizon"]),
            minimum_train_patient_coverage=float(cfg["data"]["minimum_train_patient_coverage"]),
        )
    torch.set_num_threads(int(cfg["training"]["cpu_threads"]))
    selecting, auditing = split_validation_for_cardinality(
        data.validation,
        random_seed=int(cfg["reporting"]["inner_outer_validation_selection_seed"]),
    )
    assert data.train.oracle_states is None and data.validation.oracle_states is None
    args.out_dir.mkdir(parents=True, exist_ok=False)
    json_write(args.out_dir / "data_receipt_no_concept_targets.json", data.receipt)
    rows, pairs = [], []
    evidences = {}
    for method in cfg["training"]["modes"]:
        evidences[method] = {}
        for init_seed, order_seed in itertools.product(cfg["training"]["init_seeds"], cfg["training"]["order_seeds"]):
            record, _ = _run_one(cfg, data, method, int(init_seed), int(order_seed), args.out_dir)
            run_id = record["run_id"]
            checkpoint = args.out_dir / "runs" / method / run_id / "model.pt"
            package = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if package.get("channels") != data.receipt["channel_names"]:
                raise ValueError("Model channel fingerprint differs from train-selected variables")
            model = PredictiveDynamicsConceptFAN(
                n_slots=int(cfg["architecture"]["n_candidate_slots"]),
                n_channels=len(package["channels"]),
                channels=tuple(package["channels"]),
                forecast_horizon=int(cfg["data"]["forecast_horizon"]),
                min_width=float(cfg["architecture"]["min_concept_width"]),
            )
            model.load_state_dict(package["model_state_dict"], strict=True)
            model.eval()
            with torch.no_grad():
                selection = select_concept_cardinality(
                    model, selecting, auditing, mode=method,
                    auprc_tolerance=float(cfg["reporting"]["auprc_tolerance"]),
                    nll_tolerance=float(cfg["reporting"]["nll_tolerance"]),
                    future_rmse_tolerance=float(cfg["reporting"]["future_rmse_tolerance"]),
                )
                selected = evaluate_discovery(
                    model, auditing,
                    selected_slots=torch.from_numpy(selection.selected_mask),
                )
                before = evaluate_discovery(model, auditing)
            passports = concept_passports(
                selected, model, auditing, data.medians, data.scales,
                include_examples=False,
            )
            # Aggregated definitions and metrics only; no patient record IDs,
            # scores, temporal traces, or patient-level source files released.
            detail = {
                "method": method, "run_id": run_id,
                "selected_slots": np.flatnonzero(selection.selected_mask).astype(int).tolist(),
                "selection_trace": selection.path,
                "concepts": passports,
                "oracle_labels_supplied": False,
                "clinical_intervention_claim": False,
            }
            run_dir = args.out_dir / "runs" / method / run_id
            json_write(run_dir / "selected_concept_definitions_no_patient_examples.json", detail)
            rows.append({
                "method": method, "run_id": run_id,
                "init_seed": int(init_seed), "order_seed": int(order_seed),
                "k_selected": int(selection.selected_mask.sum()),
                "original_auprc_full_validation": record["auprc"],
                "baseline_outer_auprc": before.metrics["auprc"],
                "selected_outer_auprc": selected.metrics["auprc"],
                "selected_outer_nll": selection.audit_metrics["nll"],
                "baseline_outer_future_rmse": before.metrics["future_observed_rmse"],
                "selected_outer_future_rmse": selected.metrics["future_observed_rmse"],
                "selected_outer_additive_error": selected.metrics["max_additive_error"],
                "example_patient_ids_published": False,
            })
            evidences[method][run_id] = selected
        for left, right in itertools.combinations(sorted(evidences[method]), 2):
            pairs.append({"method": method, "run_a": left, "run_b": right,
                          **compare_independent_discoveries(
                              evidences[method][left], evidences[method][right])})
    run_frame, pair_frame = pd.DataFrame(rows), pd.DataFrame(pairs)
    run_frame.to_csv(args.out_dir / "clinical_per_run_aggregates.csv", index=False)
    pair_frame.to_csv(args.out_dir / "clinical_model_pair_aggregates.csv", index=False)
    summary = {}
    for method, sub in run_frame.groupby("method"):
        cross = pair_frame[pair_frame.method.eq(method)]
        valid = cross["excess_over_unmatched_episode_null_spearman"].dropna()
        summary[method] = {
            "trained_runs": len(sub),
            "mean_auto_selected_k": float(sub.k_selected.mean()),
            "auto_selected_k_min_max": [int(sub.k_selected.min()),int(sub.k_selected.max())],
            "mean_outer_auprc_before": float(sub.baseline_outer_auprc.mean()),
            "mean_outer_auprc_after": float(sub.selected_outer_auprc.mean()),
            "mean_outer_future_rmse_after": float(sub.selected_outer_future_rmse.mean()),
            "mean_case_specific_spearman_excess":
                float(valid.mean()) if len(valid) else None,
            "paired_run_comparisons": len(cross),
            "fraction_invalid_pairs_after_k_selection": float(1 - len(valid) / max(len(cross), 1)),
        }
    json_write(args.out_dir / "clinical_validation_only_summary.json", {
        "interpretation": "exploratory, observational, not clinical concept validation",
        "no_manual_concept_targets": True,
        "no_test_or_calibration_prediction_evaluated": True,
        "model_selection_then_independent_audit": True,
        "post_pilot_method_selection": True,
        "models": len(run_frame),
        "results": summary,
        "privacy": "Only aggregate metrics and signed measurement operator formulas released",
        "limitations": [
            "No gold-standard clinical concept identities",
            "Irregular ICU measurements and informative missingness",
            "Future patient treatments uncontrolled",
            "Four initialization seeds provide no confirmatory inference",
            "Original study has previously accessed same published benchmark test; do not claim clean held-out test for novel architecture",
        ],
    })
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
