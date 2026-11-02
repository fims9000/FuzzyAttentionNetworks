"""Train/validation-only, multi-seed automatic discovery research protocol.

The synthetic generator's five latent states are NEVER passed to model.fit,
feature design, loss calculation, or model selection. They enter only the
post-training evaluator as an optional independent identifiability audit.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .data import DiscoveryData, make_synthetic_discovery_data
from .evaluation import (
    compare_independent_discoveries,
    concept_passports,
    evaluate_discovery,
    oracle_correspondence,
)
from .model import PredictiveDynamicsConceptFAN, discovery_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _valid_float(value: float | int) -> float | None:
    v = float(value)
    return v if np.isfinite(v) else None


def _model_bootstrap(
    pair_frame: pd.DataFrame,
    models: list[str],
    metric: str,
    *,
    repetitions: int,
    seed: int,
) -> dict:
    """Bootstrap original model IDs, never overlapping model pairs as rows."""
    pairs = list(itertools.combinations(sorted(models), 2))
    lookup = {(str(row.run_a), str(row.run_b)): float(getattr(row, metric))
              for row in pair_frame.itertuples(index=False)}
    scores = np.asarray([lookup.get((a, b), lookup.get((b, a), np.nan)) for a, b in pairs])
    if not np.isfinite(scores).all() or len(models) < 3:
        return {"mean": None, "ci95": None, "reason": "Missing comparable pairs"}
    index = {name: i for i, name in enumerate(sorted(models))}
    edges = np.asarray([(index[a], index[b]) for a, b in pairs])
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        counts = np.bincount(rng.integers(len(models), size=len(models)), minlength=len(models))
        weights = counts[edges[:, 0]] * counts[edges[:, 1]]
        denominator = weights.sum()
        if denominator:
            estimates.append(float(weights @ scores / denominator))
    return {
        "mean": float(scores.mean()),
        "ci95": [float(v) for v in np.quantile(estimates, (.025, .975))] if estimates else None,
        "bootstrap_repetitions_valid": len(estimates),
        "qualification": "model-only bootstrap; shared patient cohort and crossed seed structure remain fixed",
    }


def _run_one(
    cfg: dict,
    dataset: DiscoveryData,
    mode: str,
    init_seed: int,
    order_seed: int,
    out_dir: Path,
) -> tuple[dict, object]:
    set_seed(init_seed)
    model = PredictiveDynamicsConceptFAN(
        n_slots=int(cfg["architecture"]["n_candidate_slots"]),
        forecast_horizon=int(cfg["data"]["forecast_horizon"]),
        min_width=float(cfg["architecture"]["min_concept_width"]),
    )
    train = dataset.train
    classes = float(train.y.sum())
    with torch.no_grad():
        model.decision_bias.fill_(
            float(np.log(np.clip(classes / (len(train.y) - classes), 1e-5, 1e5)))
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    pos_weight = torch.tensor((len(train.y) - classes) / max(classes, 1.0), dtype=torch.float32)
    loader = train.loader(int(cfg["training"]["batch_size"]), True, order_seed)
    epochs = int(cfg["training"]["epochs"])
    loss_weights = cfg["training"]["loss_weights"]
    started = time.perf_counter()
    history = []
    for epoch in range(epochs):
        model.train()
        terms = []
        for values, observed_mask, future, future_mask, label in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(values, observed_mask)
            loss, parts = discovery_loss(
                model, output, label, future, future_mask, mode=mode,
                pos_weight=pos_weight, loss_weights=loss_weights,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss: {mode} epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg["training"]["gradient_clip"])
            )
            optimizer.step()
            terms.append({key: float(value.detach()) for key, value in parts.items()})
        history.append({
            "epoch": epoch + 1,
            **{key: float(np.mean([row[key] for row in terms])) for key in terms[0]},
        })

    # No re-tuning or early stopping based on validation outcomes.
    model.eval()
    evidence = evaluate_discovery(model, dataset.validation)
    pruned_evidence = evaluate_discovery(model, dataset.validation, hard_gates=True)
    oracle = oracle_correspondence(evidence, dataset.validation.oracle_states)
    passports = concept_passports(
        evidence, model, dataset.validation, dataset.medians, dataset.scales
    )
    identity = f"init_{init_seed}_order_{order_seed}"
    record = {
        "method": mode, "run_id": identity,
        "init_seed": int(init_seed), "order_seed": int(order_seed),
        **evidence.metrics,
        "pruned_auprc": pruned_evidence.metrics["auprc"],
        "pruned_future_observed_rmse": pruned_evidence.metrics["future_observed_rmse"],
        "oracle_mean_abs_corr": oracle["oracle_mean_abs_corr"],
        "oracle_matched": oracle["oracle_matched"],
        "elapsed_seconds": time.perf_counter() - started,
        "validation_only": True, "oracle_supervision": False,
    }
    run_dir = out_dir / "runs" / mode / identity
    run_dir.mkdir(parents=True, exist_ok=False)
    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": cfg["architecture"],
        "config_fingerprint": hashlib.sha256(
            yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest(),
        "init_seed": init_seed, "order_seed": order_seed,
        "method": mode, "validation_only": True,
    }, run_dir / "model.pt")
    _write_json(run_dir / "operator_concepts.json", passports)
    _write_json(run_dir / "oracle_evaluation_only.json", oracle)
    _write_json(run_dir / "learning_curve.json", history)
    _write_json(run_dir / "metrics.json", record)
    print(f"{mode:12s} {identity:24s} AUPRC={record['auprc']:.4f}"
          f" active={record['n_active_and_nonconstant_slots']}"
          f" future_RMSE={record['future_observed_rmse']:.4f}", flush=True)
    return record, evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("Output directory exists: refusing to overwrite prior scientific evidence")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not (cfg["protocol"]["original_test_locked"] and
            cfg["reporting"]["no_test_evaluation"] and
            cfg["reporting"]["no_confirmatory_pvalues"]):
        parser.error("Train/validation-only contract must be explicitly enabled")
    if set(cfg["training"]["modes"]) != {"task_only", "task_future", "full"}:
        parser.error("All three predefined objective ablations required")
    torch.set_num_threads(int(cfg["training"]["cpu_threads"]))
    data = make_synthetic_discovery_data(
        generator_seed=int(cfg["data"]["generator_seed"]),
        split_seed=int(cfg["data"]["split_seed"]),
        n_episodes=int(cfg["data"]["n_episodes"]),
        input_window=int(cfg["data"]["input_window"]),
        horizon=int(cfg["data"]["forecast_horizon"]),
    )
    init_seeds = [int(v) for v in cfg["training"]["init_seeds"]]
    order_seeds = [int(v) for v in cfg["training"]["order_seeds"]]
    if len(set(init_seeds)) != len(init_seeds) or len(set(order_seeds)) != len(order_seeds):
        parser.error("Duplicate train seeds undermine replication design")
    if len(init_seeds) < 3:
        parser.error("At least three initialization seeds required")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=False)
    _write_json(out_dir / "train_validation_only_receipt.json", data.receipt)
    run_records, pair_records = [], []
    by_method = {}
    for method in cfg["training"]["modes"]:
        by_method[method] = {}
        for init, order in itertools.product(init_seeds, order_seeds):
            record, ev = _run_one(cfg, data, method, init, order, out_dir)
            run_records.append(record)
            by_method[method][record["run_id"]] = ev
        keys = sorted(by_method[method])
        for a, b in itertools.combinations(keys, 2):
            comparison = compare_independent_discoveries(
                by_method[method][a], by_method[method][b]
            )
            pair_records.append({"method": method, "run_a": a, "run_b": b, **comparison})

    run_frame = pd.DataFrame(run_records)
    pairs_frame = pd.DataFrame(pair_records)
    run_frame.to_csv(out_dir / "per_run_metrics.csv", index=False)
    pairs_frame.to_csv(out_dir / "pairwise_definition_matched_stability.csv", index=False)
    report = {}
    for idx, method in enumerate(cfg["training"]["modes"]):
        mrun = run_frame.loc[run_frame["method"].eq(method)]
        mpairs = pairs_frame.loc[pairs_frame["method"].eq(method)]
        bootstrap = {}
        for metric in ("definition_cosine_normalized",
                       "excess_over_unmatched_episode_null_spearman",
                       "excess_over_unmatched_episode_null_top1"):
            bootstrap[metric] = _model_bootstrap(
                mpairs, sorted(by_method[method]), metric,
                repetitions=int(cfg["reporting"]["bootstrap_model_repetitions"]),
                seed=int(cfg["reporting"]["bootstrap_seed"]) + idx * 10 + len(bootstrap),
            )
        report[method] = {
            "mean_validation_auprc": float(mrun.auprc.mean()),
            "mean_future_rmse": float(mrun.future_observed_rmse.mean()),
            "mean_n_active_slots": float(mrun.n_active_and_nonconstant_slots.mean()),
            "mean_hard_selected_slots": float(mrun.n_hard_gate_slots.mean()),
            "mean_expected_l0": float(mrun.expected_l0.mean()),
            "mean_oracle_abs_corr_evaluation_only": float(mrun.oracle_mean_abs_corr.mean()),
            "mean_hard_pruned_auprc": float(mrun.pruned_auprc.mean()),
            "fraction_with_fewer_than_two_usable_slots":
                float((mrun.n_active_and_nonconstant_slots < 2).mean()),
            "pairs": len(mpairs), "seed_factor_counts": [len(init_seeds), len(order_seeds)],
            "model_bootstrap_sensitivity": bootstrap,
        }
    summary = {
        "scope": "exploratory Med-CircuitBench train/validation only",
        "methodology": "shared causal operator dictionary; no oracle labels in model or optimizer; true state used only for evaluation",
        "models_total": len(run_frame),
        "model_pairs_total": len(pairs_frame),
        "model_aggregation_warning": "seed cells are crossed (init x order); model-only CIs are descriptive sensitivity, not confirmatory inference",
        "test_ever_evaluated": False,
        "no_automatic_favorable_result_selection": True,
        "hypothesis_status": "untested beyond this run; failure cases retained",
        "results": report,
    }
    _write_json(out_dir / "summary.json", summary)
    files = sorted(x for x in out_dir.rglob("*") if x.is_file())
    provenance = {
        "config": cfg, "input_receipt": data.receipt,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "files_sha256": {str(path.relative_to(out_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in files},
        "no_original_test_evaluated": True,
        "oracle_labels_in_fit_or_model": False,
    }
    _write_json(out_dir / "provenance.json", provenance)
    print(json.dumps({"runs": len(run_frame), "pairs": len(pairs_frame),
                      "results": {k: {n: _valid_float(v) for n, v in x.items() if n.startswith("mean_validation_") or n == "mean_n_active_slots"}
                                  for k, x in report.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
