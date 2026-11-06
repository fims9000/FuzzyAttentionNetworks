"""Independent V45 audit based exclusively on saved TEST probabilities, not source generators."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss, brier_score_loss

PAIRS = (
    ("healthy_energy", "MCAR40", "V45_OFF_GUARD", "V44_PARTIAL_VIEW"),
    ("unknown_OFF_drift", "MCAR40", "V45_OFF_GUARD", "V44_PARTIAL_VIEW"),
    ("unknown_OFF_drift", "MCAR40", "V45_OFF_GUARD", "K2"),
    ("healthy_energy", "MCAR40", "V45_OFF_GUARD", "RAW_DIFF_HGB_ALL1800"),
    ("unknown_OFF_drift", "MCAR40", "V45_OFF_GUARD", "RAW_DIFF_HGB_ALL1800_OFF_GUARD"),
    ("common_U8_matched", "MCAR40", "V45_OFF_GUARD", "V44_PARTIAL_VIEW"),
)
CONCEPTS = {"K2", "V44_PARTIAL_VIEW", "V45_OFF_GUARD",
            "RAW_DIFF_HGB_ALL1800_OFF_GUARD"}

def _preds(arr, device, case, stress, method, seeds):
    use = seeds if method in CONCEPTS else (None,)
    p = np.stack([arr[f"device{device}_{case}_{stress}_{method}_seed{seed}"] for seed in use])
    return np.repeat(p, len(seeds), axis=0) if len(use) == 1 and len(seeds) > 1 else p


def _paired_ci(y, a, b, seed, nboot):
    rng = np.random.default_rng(seed)
    n = len(y)
    def gap(ix):
        return float(np.mean([average_precision_score(y[ix], a[j, ix]) -
                              average_precision_score(y[ix], b[j, ix])
                              for j in range(len(a))]))
    delta = gap(np.arange(n))
    draws = [gap(rng.integers(0, n, n)) for _ in range(nboot)]
    return delta, np.quantile(draws, [.025, .975])


def audit(evidence: Path, output: Path):
    status = json.loads((evidence / "STATUS.json").read_text())
    rows = json.loads((evidence / "TEST_ROWS.json").read_text())
    summaries = json.loads((evidence / "SUMMARY.json").read_text())
    paired = json.loads((evidence / "PAIRED_CIS.json").read_text())
    arr = np.load(evidence / "TEST_PREDICTIONS.npz", allow_pickle=False)
    receipts = np.load(evidence / "TEST_OFF_MONITOR_SOURCE_RECEIPTS.npz", allow_pickle=False)
    n_episodes = len(status["device_seeds"]) * 7 * 5
    if len(receipts.files) != n_episodes*6 + n_episodes*2*3:
        raise AssertionError("Incomplete per-subject sensor monitor/source receipt archive")
    seeds = sorted({r["seed"] for r in rows if r["method"] == "K2"})
    maximum = 0.0
    fallback = 0
    for r in rows:
        d, c, s, m, seed = (r[z] for z in ("device", "case", "stress", "method", "seed"))
        y = arr[f"device{d}_{c}_TEST_Y"]
        p = arr[f"device{d}_{c}_{s}_{m}_seed{seed}"]
        if (p.shape != y.shape or len(y) != status["n_test"]
                or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all()
                or (p < 0).any() or (p > 1).any()):
            raise AssertionError("Invalid frozen TEST patient predictions")
        check = {
            "n": len(y), "positives": int(y.sum()),
            "auprc": average_precision_score(y, p),
            "auroc": roc_auc_score(y, p),
            "nll": log_loss(y, p, labels=[0, 1]),
            "brier": brier_score_loss(y, p),
        }
        for name, value in check.items():
            error = abs(float(value) - float(r["metrics"][name]))
            maximum = max(maximum, error)
            if error > 1e-12:
                raise AssertionError(f"TEST metric mismatch: {d}/{c}/{s}/{m}/{name}")
        prefix = f"device{d}_{c}_{s}"
        monitor = receipts[prefix+"_off_usable"]
        monitored = receipts[prefix+"_off_support"]
        alarm = receipts[prefix+"_off_alarm"]
        if (monitor.shape != p.shape or not np.isin(monitor,[0,1]).all()
                or not np.logical_not(monitor & alarm).all()
                or not np.all(monitor <= monitored)):
            raise AssertionError("Malformed current OFF monitor/availability audit")
        if m in ("V44_PARTIAL_VIEW","V45_OFF_GUARD"):
            evidence_use=receipts[f"{prefix}_{m}_seed{seed}_used"]
            supported=receipts[prefix+"_"+m+"_measured_support"]
            if not np.array_equal(evidence_use,supported & bool(r["gate_approved"])):
                raise AssertionError("Source intervention acceptance differs from frozen VALID gate")
            if abs(float(evidence_use.mean())-r["source_use"]) > 1e-12:
                raise AssertionError("Patient-level source use does not match summary")
            if m == "V45_OFF_GUARD" and not np.all(supported <= monitor):
                raise AssertionError("V45 permitted source without measured OFF validation")
        if m in ("V44_PARTIAL_VIEW", "V45_OFF_GUARD") and (
                s == "missing_reference" or c in ("known_VALID2_OFF_drift", "complete_K2_dictionary")):
            base = arr[f"device{d}_{c}_{s}_K2_seed{seed}"]
            if not np.array_equal(p, base):
                raise AssertionError("Expected exact native K2 source refusal")
            fallback += 1
        if m == "RAW_DIFF_HGB_ALL1800_OFF_GUARD":
            plain = arr[f"device{d}_{c}_{s}_RAW_DIFF_HGB_ALL1800_seedNone"]
            base = arr[f"device{d}_{c}_{s}_K2_seed{seed}"]
            if not np.array_equal(p, np.where(monitor, plain, base)):
                raise AssertionError("Raw guard does not match actual saved OFF gate")
            if s == "missing_reference":
                if not np.array_equal(p, base):
                    raise AssertionError("No OFF raw-hybrid fallback failed")
                fallback += 1

    for summary in summaries:
        candidates = [r for r in rows if all(r[k] == summary[k] for k in
                      ("device", "case", "stress", "method"))]
        expected = len(seeds) if summary["method"] in CONCEPTS else 1
        if len(candidates) != expected:
            raise AssertionError("Missing TEST scores underlying summary")
        for label, metric in (("AUPRC", "auprc"), ("NLL", "nll")):
            if abs(np.mean([r["metrics"][metric] for r in candidates]) - summary[label]) > 1e-12:
                raise AssertionError("Aggregate TEST metric mismatch")
        if abs(np.mean([r["source_use"] for r in candidates]) - summary["source_use"]) > 1e-12:
            raise AssertionError("Aggregate source usage mismatch")

    ci_error = 0.
    for comparison in paired:
        d = comparison["device"]
        contrast = (comparison["case"], comparison["stress"], comparison["a"], comparison["b"])
        if contrast not in PAIRS:
            raise AssertionError("Unregistered TEST contrast")
        j = PAIRS.index(contrast)
        y = arr[f"device{d}_{contrast[0]}_TEST_Y"]
        a = _preds(arr, d, contrast[0], contrast[1], contrast[2], seeds)
        b = _preds(arr, d, contrast[0], contrast[1], contrast[3], seeds)
        obs, ci = _paired_ci(y, a, b, 20264601 + d%101*100 + j, comparison["resamples"])
        error = max(abs(obs - comparison["difference_AUPRC_a_minus_b"]),
                    np.max(np.abs(ci - comparison["paired_TEST_subject_bootstrap_95pct_ci"])))
        ci_error = max(ci_error, float(error))
        if error > 1e-12:
            raise AssertionError("Paired TEST-subject bootstrap mismatch")

    result = {
        "status": "V45_ALL_FROZEN_TEST_PREDICTIONS_INDEPENDENTLY_RECALCULATED",
        "n_devices": len(status["device_seeds"]),
        "verified_TEST_method_rows": len(rows),
        "verified_TEST_metrics": len(rows)*6,
        "verified_TEST_summaries": len(summaries),
        "verified_paired_AP_CIs": len(paired),
        "exact_K2_fallback_array_comparisons": fallback,
        "verified_TEST_sensor_receipt_arrays": len(receipts.files),
        "maximum_absolute_metric_error": maximum,
        "maximum_absolute_paired_CI_error": ci_error,
        "auditor_imported_model_or_generator": False,
        "physical_or_clinical_semantics_claimed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.evidence_dir, args.out_file), ensure_ascii=False))
