"""Run supplementary conditional residual audit on an existing frozen run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from conceptfan_realdata.conditional_residual import (
    fit_probe_and_predict, paired_patient_bootstrap, proxy_summary,
)
from conceptfan_realdata.data import load_prepared
from conceptfan_realdata.leakage import _residual_summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    a = p.parse_args()
    data_path = a.artifacts_dir / "prepared" / "prepared_physionet2012.npz"
    if not data_path.is_file():
        p.error(f"Frozen prepared data not found: {data_path}")
    data = load_prepared(data_path)
    matrices = {}
    labels = {}
    for split in ("train", "validation", "calibration", "test"):
        ids, residuals, positions = _residual_summary(a.run_dir, split, data)
        if not np.array_equal(ids, data.record_ids[positions]):
            raise ValueError(f"Residual record alignment failed: {split}")
        if set(positions.tolist()) != set(data.indices(split).tolist()):
            raise ValueError(f"Split patient IDs mismatched: {split}")
        concepts = proxy_summary(data.concepts[positions], data.concept_mask[positions])
        matrices[split] = (concepts, np.concatenate([concepts, residuals], axis=1))
        labels[split] = data.y[positions].astype(np.int64)
    predictions = []
    selected_cs = []
    for arm in range(2):
        probability, c = fit_probe_and_predict(
            matrices["train"][arm], labels["train"],
            matrices["validation"][arm], labels["validation"],
            matrices["calibration"][arm], labels["calibration"],
            matrices["test"][arm],
        )
        predictions.append(probability)
        selected_cs.append(c)
    result = paired_patient_bootstrap(labels["test"], *predictions, repetitions=a.bootstrap)
    result["selected_C"] = {"proxies_only": selected_cs[0], "proxies_plus_residuals": selected_cs[1]}
    result["test_patients"] = len(labels["test"])
    result["data_protocol"] = "frozen patient-level train/validation/calibration/test"
    result["status"] = "CONDITIONAL_RESIDUAL_DIAGNOSTIC_COMPUTED"
    result["warning"] = "Do not infer causality or bypass from a predictive association"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
