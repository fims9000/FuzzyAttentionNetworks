"""Frozen exploratory V16 training: native concept grounding from sparse proxy reviews.

This is NOT clinical feedback and does NOT discover physical mechanisms.
The simulated reviewer calls a recent *observed sensor* pattern present or absent.
No generator latent oracle, target labels, or holdout outcomes are used to
choose reviews or to fit the basis. All runs share the V10 training protocol.
"""
from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
import torch
import yaml

from .variable_factor_data import generate_variable_factor_data
from .expert_feedback_v16 import ExpertReview, prepare_feedback, safe_feedback_manifest
from .clinical_multiscale_v10 import fit_multiscale

GENERATOR_SEED = 20281104
SPLIT_SEED = 20300419
DATA_N = 960
N_CHANNELS = 8
N_SLOTS = 4              # prespecified capacity; NOT an inferred factor count
TRUE_K_GENERATOR = 3     # used ONLY to define synthetic data generating process
INPUT_STEPS = 36
HORIZON = 6
SEEDS = (11, 23, 37)
ARMS = ("no_reviews", "exploration_only", "correct_proxy", "inverted_proxy")
REVIEW_LIMIT_EACH_CLASS = 56
REVIEW_START = 30
PROXY_CHANNEL = 0
PROXY_MARGIN = .12
FEEDBACK_WEIGHT = 1.0
TRAIN_EPOCHS = 12
BATCH_SIZE = 128


def train_only_basis(train):
    """No latent-oracle access: SVD of observed TRAIN measurements only."""
    x = train.x.reshape(-1, train.x.shape[-1]).astype(np.float64)
    values = x - x.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(values, full_matrices=False)
    basis = vt.T[:, :N_SLOTS]
    # Deterministic observable-sensor orientation, not true factor orientation.
    for j in range(N_SLOTS):
        pivot = int(np.argmax(np.abs(basis[:, j])))
        if basis[pivot, j] < 0:
            basis[:, j] *= -1
    return np.ascontiguousarray(basis)


def observed_proxy_labels(split):
    """Illustrative clinician question: is observed recent sensor_1 level high?

    This question is a reproducible measurement-level proxy, not a diagnosis.
    A judgement requires >=4 of the last 6 sensor samples and an unambiguous
    observed average outside [-margin, +margin]. No future data used.
    """
    m = split.mask[:, REVIEW_START:INPUT_STEPS, PROXY_CHANNEL]
    x = split.x[:, REVIEW_START:INPUT_STEPS, PROXY_CHANNEL]
    n = m.sum(axis=1)
    value = (x*m).sum(axis=1) / np.maximum(n, 1)
    confident = (n >= 4) & (np.abs(value) > PROXY_MARGIN)
    return confident, (value > 0).astype(np.int64), value


def build_review_sets(data, slot):
    valid, label, _ = observed_proxy_labels(data.train)
    rng = np.random.default_rng(20301001)
    keep = []
    for target in (0, 1):
        candidates = np.flatnonzero(valid & (label == target))
        if len(candidates) < 20:
            raise RuntimeError("Insufficient observed patterns for both simulated expert answers")
        selected = rng.choice(
            candidates, size=min(REVIEW_LIMIT_EACH_CLASS, len(candidates)), replace=False,
        )
        keep.extend((int(i), int(target)) for i in selected)
    keep.sort()
    def reviews(flip):
        return [
            ExpertReview(
                int(data.train.episode_ids[i]), int(slot),
                REVIEW_START, INPUT_STEPS,
                "yes" if (target ^ flip) else "no", 1.0, "synthetic_rule_not_clinician",
            )
            for i, target in keep
        ]
    correct = prepare_feedback(
        reviews(0), data.train.episode_ids, n_slots=N_SLOTS, n_steps=INPUT_STEPS,
    )
    wrong = prepare_feedback(
        reviews(1), data.train.episode_ids, n_slots=N_SLOTS, n_steps=INPUT_STEPS,
    )
    assert correct.audit["accepted"] == wrong.audit["accepted"] == len(keep)
    return correct, wrong


def evaluate(model, split, slot):
    model.eval()
    estimates = []
    probabilities = []
    signed = []
    decomposition = 0.
    with torch.no_grad():
        for start in range(0, len(split.y), 128):
            stop = min(start+128, len(split.y))
            x = torch.from_numpy(split.x[start:stop])
            m = torch.from_numpy(split.mask[start:stop])
            out = model(x, m, process_mask=m)
            raw = out.physical.raw_activation[:, REVIEW_START:INPUT_STEPS, slot]
            support = out.physical.availability[:, REVIEW_START:INPUT_STEPS, slot]
            estimated = (raw*support).sum(1)/support.sum(1).clamp_min(1e-8)
            estimates.extend(estimated.cpu().numpy().tolist())
            probabilities.extend(out.probabilities.cpu().numpy().tolist())
            signed.extend(out.physical.contributions[:, slot].cpu().numpy().tolist())
            decomposition = max(decomposition,float(out.decomposition_error()))
    estimated = np.asarray(estimates,dtype=np.float64)
    probs = np.asarray(probabilities,dtype=np.float64)
    signed = np.asarray(signed,dtype=np.float64)
    valid, target, _ = observed_proxy_labels(split)
    if np.unique(target[valid]).size != 2:
        raise ValueError("Validation proxy needs both observed states")
    oracle = split.oracle_states
    # Oracle consulted ONLY after fitting, for a descriptive diagnostic.
    factor0 = oracle[:, REVIEW_START:INPUT_STEPS, 0].mean(axis=1)
    oracle_rho = spearmanr(estimated, factor0).statistic
    return {
        "metrics": {
            "n_validation_episodes":len(split.y),
            "proxy_validation_episodes":int(valid.sum()),
            "outcome_auprc":float(average_precision_score(split.y, probs)),
            "outcome_auroc":float(roc_auc_score(split.y, probs)),
            "outcome_brier":float(brier_score_loss(split.y, probs)),
            "observable_proxy_auroc":float(roc_auc_score(target[valid],estimated[valid])),
            "observable_proxy_auprc":float(average_precision_score(target[valid],estimated[valid])),
            "oracle_factor0_abs_spearman_eval_only":float(abs(oracle_rho)),
            "max_additive_logit_error":decomposition,
        },
        "signed":signed,
        "activation":estimated,
    }


def run(out_dir:Path):
    if out_dir.exists():
        raise FileExistsError("Never replace previously executed evidence")
    torch.set_num_threads(1)
    data = generate_variable_factor_data(
        n_true_factors=TRUE_K_GENERATOR,seed=GENERATOR_SEED,split_seed=SPLIT_SEED,
        n_samples=DATA_N,n_channels=N_CHANNELS,input_steps=INPUT_STEPS,horizon=HORIZON,
    )
    basis = train_only_basis(data.train)
    chosen_slot = int(np.argmax(np.abs(basis[PROXY_CHANNEL,:])))
    correct,wrong = build_review_sets(data, chosen_slot)
    reference = yaml.safe_load(Path("configs/medical/pd_clinical_unknown_neutral_v6.yaml").read_text())
    dual = yaml.safe_load(Path("configs/medical/pd_clinical_dual_process_v7.yaml").read_text())
    multiscale = yaml.safe_load(Path("configs/medical/pd_clinical_v10_multiscale.yaml").read_text())
    reference["training"]["epochs"] = TRAIN_EPOCHS
    reference["training"]["batch_size"] = BATCH_SIZE

    rows = []
    signed_by_arm = {}
    for arm in ARMS:
        signed_by_arm[arm] = []
        for seed in SEEDS:
            feedback = (correct if arm == "correct_proxy" else wrong if arm == "inverted_proxy" else None)
            model,train_receipt = fit_multiscale(
                data, basis, int(seed), reference, dual, multiscale,
                consistency=True,
                expert_feedback=feedback,
                expert_feedback_weight=0.0 if feedback is None else FEEDBACK_WEIGHT,
                feature_exploration=(arm != "no_reviews"),
            )
            result = evaluate(model,data.validation,chosen_slot)
            record = {"arm":arm,"initialization_seed":seed,
                      "model_selected_slot_train_sensor_only":chosen_slot,
                      "n_review_intervals":0 if feedback is None else len(feedback.accepted),
                      **result["metrics"],
                      "train_supervised_batches":train_receipt["expert_feedback_supervised_batches"],
                      "feature_exploration_enabled":train_receipt["expert_operator_gradient_bridge_enabled"],
                      "train_mean_feedback_loss":train_receipt["mean_train_expert_feedback_loss"]}
            if record["max_additive_logit_error"] > 2e-5:
                raise AssertionError("Bottleneck additive decomposition violated")
            rows.append(record)
            signed_by_arm[arm].append(result["signed"])
            print(arm,seed, "AUPRC", round(record["outcome_auprc"],5),
                  "proxy-AUROC",round(record["observable_proxy_auroc"],5),
                  "oracle-rho",round(record["oracle_factor0_abs_spearman_eval_only"],5),
                  flush=True)

    summary = []
    for arm in ARMS:
        group = [r for r in rows if r["arm"] == arm]
        effect = signed_by_arm[arm]
        pair_corr = []
        for i in range(len(effect)):
            for j in range(i+1,len(effect)):
                rho = spearmanr(effect[i],effect[j]).statistic
                if np.isfinite(rho):
                    pair_corr.append(float(rho))
        summary.append({
            "arm":arm,
            "n_independent_model_initializations":len(group),
            "mean_outcome_auprc":float(np.mean([r["outcome_auprc"] for r in group])),
            "mean_observable_proxy_auroc":float(np.mean([r["observable_proxy_auroc"] for r in group])),
            "mean_oracle_factor0_abs_spearman":float(np.mean([r["oracle_factor0_abs_spearman_eval_only"] for r in group])),
            "mean_same_slot_signed_effect_spearman_three_dependent_pairs":
                float(np.mean(pair_corr)) if pair_corr else None,
            "not_confirmatory_p_value":True,
        })

    out_dir.mkdir(parents=True,exist_ok=False)
    with (out_dir/"ALL_MODEL_FITS.csv").open("w",newline="") as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir/"AGGREGATE.csv").open("w",newline="") as file:
        writer = csv.DictWriter(file,fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    status={
        "status":"V16_EXPLORATORY_SYNTHETIC_PROXY_EXPERT_ABLATION",
        "arbitrated_concept_source":"observed sensor_1 threshold from causal history, NOT clinician",
        "human_doctor_annotations_provided":False,
        "new_concept_discovery_demonstrated":False,
        "training_on_true_hidden_factors":False,
        "oracle_factor_diagnostics_only_after_training":True,
        "validation_outcomes_used_in_training_or_review_selection":False,
        "test_ids_sealed":True,
        "training_only_svd_basis":True,
        "basis_sha256":sha256(basis.tobytes()).hexdigest(),
        "n_slots_predeclared":N_SLOTS,
        "factor_count_only_generator_definition":TRUE_K_GENERATOR,
        "chosen_slot_by_train_sensor":chosen_slot,
        "new_seed_generator":GENERATOR_SEED,"split_seed":SPLIT_SEED,
        "train_ids_sha256":data.receipt["train_ids_sha256"],
        "validation_ids_sha256":data.receipt["validation_ids_sha256"],
        "test_ids_sha256":data.receipt["test_ids_sha256"],
        "fixed_train_epochs":TRAIN_EPOCHS,"feedback_weight":FEEDBACK_WEIGHT,
        "one_review_per_selected_train_episode":True,
        "correct_review_aggregate":safe_feedback_manifest(correct),
        "wrong_review_aggregate":safe_feedback_manifest(wrong),
        "twelve_fits_all_counted":len(rows)==12,
        "no_clinical_source_names":True,
        "per_seed_results":rows,
        "aggregate":summary,
    }
    (out_dir/"STATUS.json").write_text(json.dumps(status,indent=2,allow_nan=False))
    return status


if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    run(args.out_dir)
