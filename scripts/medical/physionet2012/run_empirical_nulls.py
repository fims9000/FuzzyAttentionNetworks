"""Analyse empirical null agreement on frozen per-model contributions.

Do not overwrite accepted-report statistics or refit any model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from conceptfan_realdata.stability import _contribution_matrix, _run_dirs
from conceptfan_realdata.stability_nulls import agreement_nulls


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--arm", default="ConceptFAN-NoAlpha")
    p.add_argument("--permutations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260717)
    args = p.parse_args()
    runs = _run_dirs(args.runs_root, args.arm)
    if len(runs) != 30:
        p.error(f"Expected 30 complete runs for {args.arm}, found {len(runs)}")
    ids, first = _contribution_matrix(runs[0])
    vectors = [first]
    for run in runs[1:]:
        other_ids, values = _contribution_matrix(run)
        if not np.array_equal(ids, other_ids):
            raise ValueError(f"Different held-out patients or order in {run}")
        vectors.append(values)
    result = agreement_nulls(np.stack(vectors), args.permutations, args.seed)
    result["model_arm"] = args.arm
    result["frozen_record_id_count"] = len(ids)
    result["run_ids"] = [r.name for r in runs]
    result["status"] = "EMPIRICAL_NULL_REANALYSIS_COMPLETE"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        p.error(f"Output exists; refusing to overwrite an existing audit: {args.out}")
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
