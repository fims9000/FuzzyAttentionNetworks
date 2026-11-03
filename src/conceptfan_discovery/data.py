"""Strict train/validation-only Med-CircuitBench data protocol for discovery.

Synthetic oracle latent states are stored for external evaluation and never
enter input tensors, optimizer losses, hyperparameter tuning, or feature bank.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from med_circuitbench.benchmark.generator import BenchmarkConfig, generate_episode, split_ids
from .operators import CHANNELS


@dataclass
class DiscoverySplit:
    x: np.ndarray
    mask: np.ndarray
    future: np.ndarray
    future_mask: np.ndarray
    y: np.ndarray
    oracle_states: np.ndarray | None
    episode_ids: np.ndarray

    def loader(self, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
        # Oracle ground-truth states are deliberately absent.
        dataset = TensorDataset(
            torch.from_numpy(self.x),
            torch.from_numpy(self.mask),
            torch.from_numpy(self.future),
            torch.from_numpy(self.future_mask),
            torch.from_numpy(self.y),
        )
        return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle,
                          generator=torch.Generator().manual_seed(int(seed)), num_workers=0)


@dataclass
class DiscoveryData:
    train: DiscoverySplit
    validation: DiscoverySplit
    receipt: dict
    medians: np.ndarray
    scales: np.ndarray


def _checksum_ids(ids: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(ids.astype("<i8")).tobytes()).hexdigest()


def make_synthetic_discovery_data(
    *,
    generator_seed: int,
    split_seed: int,
    n_episodes: int = 960,
    input_window: int = 36,
    horizon: int = 6,
) -> DiscoveryData:
    if n_episodes < 100 or input_window < 8 or horizon < 1:
        raise ValueError("Need an adequate observational cohort and nonempty future horizon")
    cfg = BenchmarkConfig(
        seed=int(generator_seed), n_samples=int(n_episodes),
        sequence_length=input_window + horizon,
        input_window=input_window, prediction_horizon=horizon,
        target_threshold=0.0715, allow_target_fallback=False,
    )
    split_ids_map = split_ids(n_episodes, int(split_seed))
    train_ids = np.asarray(split_ids_map["train"], dtype=np.int64)
    val_ids = np.asarray(split_ids_map["validation"], dtype=np.int64)
    test_ids = np.asarray(split_ids_map["test"], dtype=np.int64)
    all_ids = set(train_ids.tolist()) | set(val_ids.tolist()) | set(test_ids.tolist())
    if len(all_ids) != n_episodes or (set(train_ids) & set(val_ids)) or (set(test_ids) & set(train_ids)):
        raise AssertionError("Split integrity violated")

    rng = np.random.default_rng(int(generator_seed))
    episodes = [generate_episode(idx, cfg, rng) for idx in range(n_episodes)]

    def collect(ids: np.ndarray) -> dict[str, np.ndarray]:
        selected = [episodes[int(i)] for i in ids]
        raw = np.stack([np.asarray(e["observations"], dtype=np.float64)[:input_window] for e in selected])
        mask = np.stack([np.asarray(e["masks"], dtype=np.float64)[:input_window] for e in selected])
        future = np.stack([np.asarray(e["observations"], dtype=np.float64)[input_window:input_window + horizon] for e in selected])
        future_mask = np.stack([np.asarray(e["masks"], dtype=np.float64)[input_window:input_window + horizon] for e in selected])
        labels = np.asarray([int(e["target"]) for e in selected], dtype=np.float32)
        oracle = np.stack([np.asarray(e["states"], dtype=np.float64)[:input_window] for e in selected])
        return {"raw": raw, "mask": mask, "future": future, "future_mask": future_mask,
                "y": labels, "oracle": oracle}

    train = collect(train_ids)
    val = collect(val_ids)
    # No records from the sealed test ID set are collected or inspected.
    if np.unique(train["y"]).size != 2 or np.unique(val["y"]).size != 2:
        raise ValueError("Train and validation each need both classes")

    observed_train = np.where(train["mask"] > 0, train["raw"], np.nan)
    medians = np.nanmedian(observed_train, axis=(0, 1))
    q75 = np.nanquantile(observed_train, .75, axis=(0, 1))
    q25 = np.nanquantile(observed_train, .25, axis=(0, 1))
    scales = np.maximum(q75 - q25, 1e-3)
    if not np.isfinite(medians).all() or not np.isfinite(scales).all():
        raise ValueError("Missing channel without training observations")

    def standardized(data: dict, ids: np.ndarray) -> DiscoverySplit:
        x = np.clip((data["raw"] - medians) / scales, -6., 6.)
        future = np.clip((data["future"] - medians) / scales, -6., 6.)
        x = np.where(data["mask"] > 0, x, 0.).astype(np.float32)
        future = np.where(data["future_mask"] > 0, future, 0.).astype(np.float32)
        return DiscoverySplit(
            x=x, mask=data["mask"].astype(np.float32),
            future=future, future_mask=data["future_mask"].astype(np.float32),
            y=data["y"], oracle_states=data["oracle"].astype(np.float32),
            episode_ids=ids,
        )

    train_split, val_split = standardized(train, train_ids), standardized(val, val_ids)
    receipt = {
        "data": "Med-CircuitBench newly synthesized",
        "generator_seed": int(generator_seed),
        "split_seed": int(split_seed),
        "n_total_generated": int(n_episodes),
        "train_n": len(train_ids),
        "validation_n": len(val_ids),
        "test_n_sealed": len(test_ids),
        "train_ids_sha256": _checksum_ids(train_ids),
        "validation_ids_sha256": _checksum_ids(val_ids),
        "sealed_test_ids_sha256": _checksum_ids(test_ids),
        "positive_train": int(train_split.y.sum()),
        "positive_validation": int(val_split.y.sum()),
        "channel_names": list(CHANNELS),
        "train_only_medians": medians.tolist(),
        "train_only_iqr": scales.tolist(),
        "oracle_states_in_optimizers": False,
        "test_used_for_training_selection_or_evaluation": False,
        "forecast_horizon": horizon,
        "input_window": input_window,
    }
    return DiscoveryData(train_split, val_split, receipt, medians, scales)
