"""Re-evaluate frozen jointly trained concept models with honest, data-driven K.

Select slot count on one half of validation and publish ALL metrics on a
disjoint half. This reuses native model checkpoints, not a post-hoc concept
extractor and not the original sealed test set.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .data import make_synthetic_discovery_data
from .evaluation import (
    compare_independent_discoveries,
    concept_passports,
    evaluate_discovery,
    oracle_correspondence,
)
from .model import PredictiveDynamicsConceptFAN
from .selection import select_concept_cardinality, split_validation_for_cardinality


def save_json(path: Path, obj: dict | list) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoints-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--selection-seed", type=int, default=20261107)
    ap.add_argument("--auprc-tolerance", type=float, default=0.025)
    ap.add_argument("--nll-tolerance", type=float, default=0.035)
    ap.add_argument("--future-rmse-tolerance", type=float, default=0.045)
    args = ap.parse_args()
    if args.out_dir.exists():
        ap.error("Refusing to overwrite previous audit")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config_digest = hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest()
    torch.set_num_threads(int(cfg["training"]["cpu_threads"]))
    data = make_synthetic_discovery_data(
        generator_seed=int(cfg["data"]["generator_seed"]),
        split_seed=int(cfg["data"]["split_seed"]),
        n_episodes=int(cfg["data"]["n_episodes"]),
        input_window=int(cfg["data"]["input_window"]),
        horizon=int(cfg["data"]["forecast_horizon"]),
    )
    selecting, audit = split_validation_for_cardinality(
        data.validation, random_seed=args.selection_seed,
    )
    checkpoint_root = args.checkpoints_root
    if (checkpoint_root / "published_discovery" / "runs").is_dir():
        checkpoint_root = checkpoint_root / "published_discovery"
    elif (checkpoint_root / "runs").is_dir():
        pass
    else:
        ap.error("Checkpoints root must contain a runs/ folder")

    rows = []
    comparisons = []
    evidences = {}
    provenance = {
        "selection_fraction": len(selecting.y) / len(data.validation.y),
        "selection_episodes": len(selecting.y),
        "audit_episodes": len(audit.y),
        "selection_episode_ids_sha256": hashlib.sha256(
            np.ascontiguousarray(selecting.episode_ids).tobytes()).hexdigest(),
        "audit_episode_ids_sha256": hashlib.sha256(
            np.ascontiguousarray(audit.episode_ids).tobytes()).hexdigest(),
        "frozen_model_config_sha256": config_digest,
        "oracle_used_in_selection": False,
        "test_ids_inspected": False,
        "selection_protocol": "greedy concept-slot elimination: independent inner selection / outer audit",
        "tolerances": {"auprc": args.auprc_tolerance,
                       "nll": args.nll_tolerance,
                       "future_rmse": args.future_rmse_tolerance},
        "warning": "Exploratory post-pilot change; this outer validation audit does not establish prospective confirmatory significance",
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "passports").mkdir()
    for method in cfg["training"]["modes"]:
        evidences[method] = {}
        for seed, order in itertools.product(cfg["training"]["init_seeds"], cfg["training"]["order_seeds"]):
            run_id = f"init_{seed}_order_{order}"
            path = checkpoint_root / "runs" / method / run_id / "model.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Missing frozen trained model: {path}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint["config_fingerprint"] != config_digest or checkpoint["method"] != method:
                raise ValueError(f"Source checkpoint is not from this frozen protocol: {path}")
            model = PredictiveDynamicsConceptFAN(
                n_slots=int(cfg["architecture"]["n_candidate_slots"]),
                forecast_horizon=int(cfg["data"]["forecast_horizon"]),
                min_width=float(cfg["architecture"]["min_concept_width"]),
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()
            with torch.no_grad():
                selected = select_concept_cardinality(
                    model, selecting, audit, mode=method,
                    auprc_tolerance=args.auprc_tolerance,
                    nll_tolerance=args.nll_tolerance,
                    future_rmse_tolerance=args.future_rmse_tolerance,
                )
                before = evaluate_discovery(model, audit)
                after = evaluate_discovery(
                    model, audit, selected_slots=torch.from_numpy(selected.selected_mask),
                )
            truth = oracle_correspondence(after, audit.oracle_states)
            passport = concept_passports(after, model, audit, data.medians, data.scales)
            save_json(args.out_dir / "passports" / f"{method}_{run_id}.json", {
                "source_checkpoint": str(path),
                "slots_selected": np.flatnonzero(selected.selected_mask).astype(int).tolist(),
                "selection_trace": selected.path,
                "audit_only_oracle": truth,
                "concepts": passport,
            })
            rows.append({
                "method": method, "run_id": run_id, "init_seed": int(seed), "order_seed": int(order),
                "k_selected": int(selected.selected_mask.sum()),
                "before_auprc_audit": before.metrics["auprc"],
                "selected_auprc_audit": after.metrics["auprc"],
                "selected_nll_audit": selected.audit_metrics["nll"],
                "before_future_rmse_audit": before.metrics["future_observed_rmse"],
                "selected_future_rmse_audit": after.metrics["future_observed_rmse"],
                "selected_mean_effective_concepts": after.metrics["mean_effective_concept_count"],
                "selected_oracle_corr_evaluation_only": truth["oracle_mean_abs_corr"],
                "oracle_factor_coverage_evaluation_only": truth["oracle_factor_coverage"],
                "oracle_coverage_adjusted_corr_evaluation_only": truth["oracle_coverage_adjusted_abs_corr"],
                "selection_auc": selected.selection_metrics["auprc"],
                "selection_nll": selected.selection_metrics["nll"],
                "full_additive_error": before.metrics["max_additive_error"],
                "selected_additive_error": after.metrics["max_additive_error"],
            })
            evidences[method][run_id] = after
            print(method, run_id, "selected", int(selected.selected_mask.sum()),
                  "audit AUPRC", round(after.metrics["auprc"], 4), flush=True)
        for left, right in itertools.combinations(sorted(evidences[method]), 2):
            result = compare_independent_discoveries(evidences[method][left], evidences[method][right])
            comparisons.append({"method": method, "run_a": left, "run_b": right, **result})

    by_run = pd.DataFrame(rows)
    by_pair = pd.DataFrame(comparisons)
    by_run.to_csv(args.out_dir / "cardinality_selected_per_run.csv", index=False)
    by_pair.to_csv(args.out_dir / "selected_pairwise_stability.csv", index=False)
    summary = {}
    for method, df in by_run.groupby("method"):
        pairs = by_pair[by_pair.method.eq(method)]
        summary[method] = {
            "k_selected_mean": float(df.k_selected.mean()),
            "k_selected_min": int(df.k_selected.min()),
            "k_selected_max": int(df.k_selected.max()),
            "mean_preselection_audit_auprc": float(df.before_auprc_audit.mean()),
            "mean_selected_audit_auprc": float(df.selected_auprc_audit.mean()),
            "mean_selected_audit_future_rmse": float(df.selected_future_rmse_audit.mean()),
            "mean_selected_oracle_corr_on_matched_only": float(df.selected_oracle_corr_evaluation_only.mean()),
            "mean_true_factor_coverage": float(df.oracle_factor_coverage_evaluation_only.mean()),
            "mean_oracle_coverage_adjusted_corr": float(df.oracle_coverage_adjusted_corr_evaluation_only.mean()),
            "fraction_pruned": float((df.k_selected < cfg["architecture"]["n_candidate_slots"]).mean()),
            "fraction_below_two_slots": float((df.k_selected < 2).mean()),
            "valid_comparison_pair_fraction": float(np.isfinite(pairs.paired_spearman).mean()),
            "mean_case_specific_spearman_excess": (
                float(pairs.excess_over_unmatched_episode_null_spearman.mean())
                if np.isfinite(pairs.excess_over_unmatched_episode_null_spearman).any() else None
            ),
            "mean_definition_similarity": float(pairs.definition_cosine_normalized.mean()),
        }
    save_json(args.out_dir / "cardinality_selection_summary.json", {
        "experimental_status": "exploratory validation-only native concept selection",
        "results": summary,
        "test_used": False,
        "selection_not_a_posthoc_concept_extractor": True,
        "failure_if_k_max_not_reduced": True,
    })
    save_json(args.out_dir / "selection_provenance.json", provenance)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
