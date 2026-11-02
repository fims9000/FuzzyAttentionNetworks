"""Reproducible Med-CircuitBench TRAIN/VALIDATION-only ConceptFAN anchor pilot.

This is an exploratory development experiment, NOT evidence of significant
improvement and NOT an evaluation on the frozen PIERE / PhysioNet test sets.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from conceptfan_realdata.models import (
    AdditiveConceptModel,
    exact_decomposition_error,
    masked_concept_loss,
)
from conceptfan_realdata.semantic_anchors import (
    AnchoredConceptFAN,
    fit_semantic_anchor,
    semantic_concept_regularizers,
)
from med_circuitbench.benchmark.generator import BenchmarkConfig, generate_episode, split_ids


def freeze_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def checksum_ndarray(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    return hashlib.sha256(a.tobytes()).hexdigest()


def load_split(config: dict) -> tuple[dict, dict]:
    data_cfg = config["dataset"]
    n = int(data_cfg["n_samples"])
    generator_cfg = BenchmarkConfig(
        seed=int(data_cfg["generator_seed"]), n_samples=n,
        sequence_length=int(data_cfg["sequence_length"]),
        input_window=int(data_cfg["input_window"]),
        target_threshold=float(data_cfg["target_threshold"]),
        infection_prevalence=float(data_cfg["infection_prevalence"]),
        infection_impulse_strength=float(data_cfg["infection_impulse_strength"]),
    )
    rng = np.random.default_rng(int(data_cfg["generator_seed"]))
    episodes = [generate_episode(i, generator_cfg, rng) for i in range(n)]
    # No hidden test outcome/feature is loaded into the optimizer, evaluator,
    # scaler, or anchor fitting procedure.
    splits = split_ids(n, int(data_cfg["split_seed"]))
    train_ids = np.array(splits["train"], dtype=int)
    validation_ids = np.array(splits["validation"], dtype=int)
    sealed_ids = np.array(splits["test"], dtype=int)
    inputs = np.stack([episodes[i]["model_input"] for i in np.r_[train_ids, validation_ids]]).astype(np.float32)
    concepts = np.stack([episodes[i]["states"][:int(data_cfg["input_window"])] for i in np.r_[train_ids, validation_ids]]).astype(np.float32)
    outcomes = np.asarray([episodes[i]["target"] for i in np.r_[train_ids, validation_ids]], dtype=np.float32)
    n_train = len(train_ids)
    norm = StandardScaler()
    x_train = inputs[:n_train]
    x_val = inputs[n_train:]
    norm.fit(x_train.reshape(-1, x_train.shape[-1]))
    x_train = norm.transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape).astype(np.float32)
    x_val = norm.transform(x_val.reshape(-1, x_val.shape[-1])).reshape(x_val.shape).astype(np.float32)
    data = {
        "train": (x_train, outcomes[:n_train], concepts[:n_train]),
        "validation": (x_val, outcomes[n_train:], concepts[n_train:]),
    }
    if any(np.unique(data[s][1]).size != 2 for s in data):
        raise ValueError("Train and validation must both contain both classes; do not silently change protocol")
    receipt = {
        "generator_seed": int(data_cfg["generator_seed"]),
        "train_count": n_train,
        "validation_count": len(validation_ids),
        "sealed_test_count": len(sealed_ids),
        "train_ids_sha256": checksum_ndarray(train_ids),
        "validation_ids_sha256": checksum_ndarray(validation_ids),
        "sealed_test_ids_sha256": checksum_ndarray(sealed_ids),
        "train_positive_rate": float(data["train"][1].mean()),
        "validation_positive_rate": float(data["validation"][1].mean()),
        "train_only_normalization_mean": norm.mean_.tolist(),
        "train_only_normalization_scale": norm.scale_.tolist(),
        "no_test_features_or_labels_used": True,
    }
    return data, receipt


def make_model(method: str, config: dict, anchor, input_dim: int) -> torch.nn.Module:
    if method == "legacy":
        return AdditiveConceptModel(config, input_dim, fuzzy=True)
    return AnchoredConceptFAN(config, input_dim, anchor)


def evaluate_validation(model, validation: tuple[np.ndarray, np.ndarray, np.ndarray], batch_size: int) -> dict:
    x, y, target_concepts = validation
    model.eval()
    probabilities, contributions, concept_predictions, errors = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            out = model(torch.from_numpy(x[start:start+batch_size]))
            probabilities.append(out.probability.detach().numpy())
            contributions.append(out.contributions.detach().numpy())
            concept_predictions.append(out.concept_trajectories.detach().numpy())
            errors.append(exact_decomposition_error(out).detach().numpy())
    p = np.concatenate(probabilities)
    v = np.concatenate(contributions)
    c = np.concatenate(concept_predictions)
    decomp = float(np.max(np.concatenate(errors)))
    if decomp > 1e-5:
        raise AssertionError(f"Additive decomposition contract broken: {decomp}")
    weight = np.abs(v) / np.maximum(np.abs(v).sum(axis=1, keepdims=True), 1e-12)
    effective_count = np.exp(-(weight * np.log(np.maximum(weight, 1e-12))).sum(axis=1))
    return {
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "concept_mae": float(np.mean(np.abs(c - target_concepts))),
        "concept_dynamics_mae": float(np.mean(np.abs(np.diff(c,axis=1)-np.diff(target_concepts,axis=1)))),
        "max_decomposition_error": decomp,
        "mean_effective_concept_count": float(effective_count.mean()),
        "probability": p,
        "contributions": v,
    }


def train_one(method: str, initial_seed: int, config: dict, anchor, data: dict, *, order_seed: int | None = None) -> dict:
    freeze_seed(initial_seed)
    x, y, c = data["train"]
    cfg = config["training"]
    model = make_model(method, config, anchor, x.shape[-1])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=float(cfg["learning_rate"]),weight_decay=float(cfg["weight_decay"]))
    generator = torch.Generator().manual_seed(int(cfg["data_order_seed"] if order_seed is None else order_seed))
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(c))
    loader = DataLoader(ds,batch_size=int(cfg["batch_size"]),shuffle=True,generator=generator)
    pos_weight = torch.tensor([(len(y)-float(y.sum()))/max(float(y.sum()),1.)])
    for epoch in range(int(cfg["epochs"])):
        model.train()
        for xb,yb,cb in loader:
            optimizer.zero_grad(set_to_none=True)
            out=model(xb)
            base = F.binary_cross_entropy_with_logits(out.logit,yb,pos_weight=pos_weight)
            concept=masked_concept_loss(out.concept_trajectories,cb,torch.ones_like(cb))
            loss=base+float(cfg["lambda_concept"])*concept
            if method=="anchor_regularized":
                extra=semantic_concept_regularizers(out,cb,torch.ones_like(cb),model)
                loss=loss+float(cfg["lambda_temporal"])*extra["temporal"]
                loss=loss+float(cfg["lambda_oracle"])*extra["oracle_evidence"]
                loss=loss+float(cfg["lambda_cross_residual"])*extra["cross_residual"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss: {method} seed {initial_seed} epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],float(cfg["gradient_clip"])
            )
            optimizer.step()
    return evaluate_validation(model,data["validation"],int(cfg["batch_size"]))


def agreement(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        raise ValueError("Paired models must evaluate same patients and concepts")
    rank = []
    for x,y in zip(a,b):
        if np.std(x)<1e-10 or np.std(y)<1e-10:
            rank.append(0.)
        else:
            value=float(spearmanr(x,y).statistic)
            rank.append(value if np.isfinite(value) else 0.)
    abs_a=np.abs(a)
    abs_b=np.abs(b)
    top_a=np.argmax(abs_a,axis=1)
    top_b=np.argmax(abs_b,axis=1)
    return {
        "spearman_signed":float(np.mean(rank)),
        "top1_abs":float(np.mean(top_a==top_b)),
        "sign_agreement":float(np.mean(np.sign(a)==np.sign(b))),
        "cosine_signed":float(np.mean(
            (a*b).sum(axis=1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-12)
        )),
    }


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    args=ap.parse_args()
    if args.out_dir.exists():
        ap.error("Refusing to overwrite a previous validation run")
    cfg=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if cfg["evaluation"]["no_test_access"] is not True or cfg["evaluation"]["no_hyperparameter_selection_using_validation"] is not True:
        ap.error("Validation-only guard violated")
    torch.set_num_threads(int(cfg["training"]["cpu_threads"]))
    data,receipt=load_split(cfg)
    a=cfg["anchor"]
    anchor=fit_semantic_anchor(
        data["train"][2],data["train"][1],
        recency_decay=float(a["recency_decay"]),
        logistic_c=float(a["logistic_c"]),
        minimum_width=float(a["minimum_width"]),
    )
    summary=(data["train"][2]*anchor.time_weights[None,:,None]).sum(axis=1)
    oracle_prob=1/(1+np.exp(-(anchor.bias+((
        1/(1+np.exp(-(summary-anchor.center)/anchor.width))-0.5
    )*anchor.beta).sum(axis=1))))
    receipt["oracle_train_auprc"]=float(average_precision_score(data["train"][1],oracle_prob))
    outcomes=[]
    pairrows=[]
    methods=cfg["evaluation"]["methods"]
    init_seeds=[int(v) for v in cfg["training"]["init_seeds"]]
    if len(set(init_seeds))<3:
        raise ValueError("Exploratory pilot needs >=3 independent starts")
    for method in methods:
        saved={}
        for seed in init_seeds:
            report=train_one(method,seed,cfg,anchor,data)
            saved[seed]=report
            outcomes.append({"method":method,"seed":seed,
                             **{k:v for k,v in report.items() if not isinstance(v,np.ndarray)}})
            print(method,seed,"AUPRC",round(report["auprc"],4),
                  "MAE",round(report["concept_mae"],4),flush=True)
        for seed_a,seed_b in itertools.combinations(init_seeds,2):
            pairrows.append({"method":method,"seed_a":seed_a,"seed_b":seed_b,
                             **agreement(saved[seed_a]["contributions"],saved[seed_b]["contributions"])})
    args.out_dir.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(outcomes).to_csv(args.out_dir/"validation_per_run.csv",index=False)
    pd.DataFrame(pairrows).to_csv(args.out_dir/"validation_pairwise_stability.csv",index=False)
    metadata={
        "protocol":cfg["protocol"],"frozen_config":cfg,"data_receipt":receipt,
        "train_only_semantic_anchor":anchor.to_dict(),
        "git_head":subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),
        "scope":"train/validation only: test IDs sealed and never used for selection",
        "statistical_note":"3-seed pilot: descriptive statistics only, no p-values or confirmatory claims",
        "methods":methods,
    }
    (args.out_dir/"provenance.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(json.dumps({"methods":methods,"runs":len(outcomes),"pairs":len(pairrows),
                      "baseline_train_oracle_auprc":metadata["data_receipt"]["oracle_train_auprc"],
                      "test_evaluated":False},indent=2))


if __name__=="__main__":
    main()
