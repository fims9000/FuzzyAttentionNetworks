"""Frozen 30-run conditional residual study: proxy-only vs proxy+residuals.

No checkpoint is modified, no neural model is trained, and old reports stay
immutable. Proxy features and patient IDs are identical across both arms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss

from conceptfan_realdata.conditional_residual import fit_probe_and_predict, proxy_summary
from conceptfan_realdata.data import load_prepared
from conceptfan_realdata.leakage import _residual_summary
from conceptfan_realdata.residual_grid import hierarchical_delta_ci
from conceptfan_realdata.stability import _run_dirs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args()
    if args.out_dir.exists():
        p.error(f"Output destination already exists: {args.out_dir}")
    prepared = args.artifacts_dir / "prepared" / "prepared_physionet2012.npz"
    if not prepared.is_file():
        p.error(f"Missing frozen prepared patient data: {prepared}")
    data = load_prepared(prepared)
    runs = _run_dirs(args.artifacts_dir / "runs", "ConceptFAN-NoAlpha")
    if len(runs) != 30:
        p.error(f"Expected 30 completed ConceptFAN fits; found {len(runs)}")

    names = ("train", "validation", "calibration", "test")
    positions = {split: data.indices(split) for split in names}
    baselines = {split: proxy_summary(data.concepts[idx], data.concept_mask[idx]) for split, idx in positions.items()}
    labels = {split: data.y[idx].astype(np.int64) for split, idx in positions.items()}
    x_train, x_val, x_cal, x_test = (baselines[split] for split in names)
    y_train, y_val, y_cal, y_test = (labels[split] for split in names)
    base_pred, base_c = fit_probe_and_predict(x_train, y_train, x_val, y_val, x_cal, y_cal, x_test)

    extended = []
    rows = []
    for run in runs:
        matrices = {}
        for split in names:
            ids, residuals, run_positions = _residual_summary(run, split, data)
            expected = positions[split]
            if len(run_positions) != len(expected) or set(run_positions) != set(expected):
                raise ValueError(f"Split mismatch in {run.name}/{split}")
            position_to_row = {int(index): j for j, index in enumerate(run_positions)}
            order = [position_to_row[int(index)] for index in expected]
            aligned_residuals = residuals[order]
            matrices[split] = np.column_stack([baselines[split], aligned_residuals])
        p_extended, c = fit_probe_and_predict(
            matrices["train"], y_train, matrices["validation"], y_val,
            matrices["calibration"], y_cal, matrices["test"],
        )
        extended.append(p_extended)
        rows.append({
            "run_id": run.name,
            "selected_C_baseline": base_c,
            "selected_C_extended": c,
            "delta_auprc": float(average_precision_score(y_test, p_extended) -
                                 average_precision_score(y_test, base_pred)),
            "delta_nll": float(log_loss(y_test, p_extended, labels=[0, 1]) -
                              log_loss(y_test, base_pred, labels=[0, 1])),
        })

    result = hierarchical_delta_ci(y_test, base_pred, np.stack(extended), repetitions=args.bootstrap)
    result["status"] = "CONDITIONAL_RESIDUAL_GRID_REANALYSIS_COMPLETE"
    result["model_arm"] = "ConceptFAN-NoAlpha"
    result["runs"] = [r.name for r in runs]
    result["test_positives"] = int(y_test.sum())
    result["frozen_inputs_only"] = True
    result["claim_limit"] = "Additional predictive association conditional on proxy summaries; not causal bypass evidence"
    args.out_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(args.out_dir / "per_run_deltas.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
