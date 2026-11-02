"""Run every available frozen ConceptFAN reanalysis without touching old evidence.

The suite is deliberately non-destructive. Missing local patient-level
artifacts are reported as NOT_AVAILABLE, never fabricated from aggregates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts" / "medical" / "physionet2012"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts" / "physionet2012_last_run")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--two-way-bootstrap", type=int, default=2000)
    p.add_argument("--null-permutations", type=int, default=1000)
    p.add_argument("--conditional-bootstrap", type=int, default=2000)
    p.add_argument("--tests", action="store_true")
    args = p.parse_args()
    if args.out_dir.exists():
        p.error("Refusing to overwrite an existing reanalysis directory")
    out = args.out_dir.resolve()
    artifacts = args.artifacts_dir.resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")])
    steps = []

    def step(name: str, required: list[Path], argv: list[str]) -> None:
        missing = [str(x) for x in required if not x.is_file()]
        if missing:
            steps.append({"name": name, "status": "NOT_AVAILABLE", "missing": missing})
            return
        result = subprocess.run([sys.executable, *argv], cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        record = {"name": name, "status": "PASS" if result.returncode == 0 else "FAIL",
                  "exit_code": result.returncode, "output_tail": result.stdout[-3000:]}
        steps.append(record)
        print(name, record["status"], flush=True)

    out.mkdir(parents=True, exist_ok=False)
    repo_pair = ROOT / "reports/physionet2012/posthoc_attribution_audit/TABLES/posthoc_pairwise_stability.csv"
    step("posthoc_model_only_ci", [repo_pair], [
        str(SCRIPTS / "recompute_posthoc_model_only_ci.py"),
        "--input", str(repo_pair), "--out", str(out / "model_only_ci.csv"),
    ])
    episode = artifacts / "TABLES" / "episode_pairwise_stability.parquet"
    pair = artifacts / "TABLES" / "stability_pair_summary.parquet"
    step("two_way_stability_ci", [episode, pair], [
        str(SCRIPTS / "recompute_stability_ci.py"),
        "--artifacts-dir", str(artifacts),
        "--out", str(out / "two_way_stability_ci.csv"),
        "--repetitions", str(args.two_way_bootstrap),
    ])
    runs = artifacts / "runs" / "ConceptFAN-NoAlpha"
    contribution_files = sorted(runs.glob("run_*/contributions_test.parquet"))
    required_null = contribution_files if len(contribution_files) == 30 else [runs / "30_run_contributions_required"]
    step("empirical_nulls", required_null, [
        str(SCRIPTS / "run_empirical_nulls.py"),
        "--runs-root", str(artifacts / "runs"),
        "--out", str(out / "empirical_nulls.json"),
        "--permutations", str(args.null_permutations),
    ])
    prepared = artifacts / "prepared" / "prepared_physionet2012.npz"
    run_dirs = sorted(runs.glob("run_*"))
    required_conditional = [prepared]
    if len(run_dirs) == 30:
        required_conditional += [r / f"concepts_{split}.parquet" for r in run_dirs
                                 for split in ("train", "validation", "calibration", "test")]
    else:
        required_conditional.append(runs / "30_complete_run_directories_required")
    step("conditional_residual_grid", required_conditional, [
        str(SCRIPTS / "run_conditional_residual_grid.py"),
        "--artifacts-dir", str(artifacts),
        "--out-dir", str(out / "conditional_grid_30"),
        "--bootstrap", str(args.conditional_bootstrap),
    ])
    if args.tests:
        step("physionet_regression_tests", [], [
            "-m", "pytest", "tests/physionet2012", "-q",
        ])
    result = {
        "status": "COMPLETE" if all(s["status"] == "PASS" for s in steps) else "PARTIAL_OR_FAILED",
        "source_branch": "fix/physionet-evidence-bootstrap-20261102",
        "scientific_status": "reanalysis, no model retraining or new causal claim",
        "artifacts_dir": str(artifacts),
        "steps": steps,
        "output_sha256": {x.name: sha256(x) for x in sorted(out.iterdir()) if x.is_file()},
    }
    (out / "STATUS.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if any(s["status"] == "FAIL" for s in steps):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
