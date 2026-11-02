"""Annotation-free PhysioNet 2012 time-series data adapter.

Only raw observed parameter names, values, timestamps and outcomes are used.
The five manually specified proxy concept definitions from the earlier paper
are deliberately NOT imported, generated, or provided as training labels.
The old 600-case test split is sealed; no test prediction is made.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from conceptfan_realdata.data import (
    FORBIDDEN_INPUTS,
    KNOWN_PARAMETERS,
    STATIC_PARAMETERS,
    _aggregate_patient,
    _read_record,
    create_split,
    load_outcomes,
    sha256_file,
)
from .data import DiscoveryData, DiscoverySplit


def prepare_physionet_discovery(
    set_a_zip: Path,
    outcomes_a: Path,
    temporary_split_dir: Path,
    *,
    split_seed: int = 20260717,
    input_window: int = 36,
    forecast_horizon: int = 12,
    minimum_train_patient_coverage: float = 0.15,
) -> DiscoveryData:
    """Prepare only TRAIN/VALIDATION while preserving the old frozen split.

    This is a new exploratory protocol. Since the original clinical test was
    already consulted by a different study, it is not valid for new method
    selection or an unbiased primary confirmatory result.
    """
    if not set_a_zip.is_file() or not outcomes_a.is_file():
        raise FileNotFoundError("Official PhysioNet Set A and Outcomes-a.txt are required")
    if input_window < 8 or forecast_horizon < 1 or input_window + forecast_horizon != 48:
        raise ValueError("Use a full 48h history+forecast protocol")
    if not 0.01 <= minimum_train_patient_coverage <= 1.0:
        raise ValueError("Invalid prespecified train-only observability cutoff")
    outcomes = load_outcomes(outcomes_a)
    if len(outcomes) != 4000:
        raise ValueError("Expected the 4000 official Set A outcome records")
    official_vars = sorted(KNOWN_PARAMETERS - STATIC_PARAMETERS - FORBIDDEN_INPUTS)
    if not official_vars:
        raise ValueError("No observable clinical variables")
    values = np.full((4000, 48, len(official_vars)), np.nan, dtype=np.float32)
    mask = np.zeros_like(values)
    identifiers, icu_types = [], []
    with zipfile.ZipFile(set_a_zip) as archive:
        if archive.testzip() is not None:
            raise ValueError("Corrupt clinical ZIP member")
        files = sorted(x for x in archive.namelist() if x.lower().endswith(".txt"))
        if len(files) != 4000:
            raise ValueError("Expected exactly 4000 clinical patient files")
        for pos, filename in enumerate(files):
            record = _read_record(archive.read(filename))
            record_id = int(Path(filename).stem)
            ids_inside = record.loc[record["Parameter"].eq("RecordID"), "Value"].dropna().astype(int).unique()
            if len(ids_inside) != 1 or int(ids_inside[0]) != record_id:
                raise ValueError("Patient record name/id mismatch")
            current, observed, static = _aggregate_patient(record, official_vars, hours=48)
            values[pos] = current
            mask[pos] = observed
            identifiers.append(record_id)
            icu_type = static.get("ICUType", float("nan"))
            icu_types.append(int(icu_type) if np.isfinite(icu_type) else -1)
    if len(set(identifiers)) != 4000 or set(identifiers) != set(outcomes["RecordID"].astype(int)):
        raise ValueError("Duplicate or missing patient/outcome alignment")
    outcomes = outcomes.set_index("RecordID")
    manifest = pd.DataFrame({
        "record_id": identifiers,
        "icu_type": icu_types,
        "In-hospital_death": [int(outcomes.loc[int(i), "In-hospital_death"]) for i in identifiers],
    })
    # Same published stratification and seed, including ICU type. The exact
    # split IDs are not uploaded as public model artifacts.
    split = create_split(manifest, temporary_split_dir, seed=int(split_seed))
    membership = dict(zip(split.RecordID.astype(int), split.split.astype(str)))
    labels = np.asarray([int(outcomes.loc[int(i), "In-hospital_death"]) for i in identifiers], dtype=np.float32)
    train_idx = np.asarray([i for i,r in enumerate(identifiers) if membership[r]=="train"],dtype=int)
    val_idx = np.asarray([i for i,r in enumerate(identifiers) if membership[r]=="validation"],dtype=int)
    cal_idx = np.asarray([i for i,r in enumerate(identifiers) if membership[r]=="calibration"],dtype=int)
    test_idx = np.asarray([i for i,r in enumerate(identifiers) if membership[r]=="test"],dtype=int)
    if tuple(map(len, (train_idx,val_idx,cal_idx,test_idx))) != (2400,600,400,600):
        raise ValueError("Published train/validation/calibration/test partition differs")

    coverage = (mask[train_idx, :input_window].sum(axis=1)>0).mean(axis=0)
    keep = coverage >= minimum_train_patient_coverage
    names_initial = np.asarray(official_vars)
    # Normalization excludes any data outside TRAIN and uses observed readings,
    # never hospital outcomes, downstream manually defined concepts, or
    # longitudinal future information from patients in validation.
    observed_training = np.where(mask[train_idx, :input_window]>0,
                                 values[train_idx, :input_window],np.nan)
    candidate = observed_training[:, :, keep]
    median_candidate = np.nanmedian(candidate, axis=(0,1))
    iqr_candidate = (
        np.nanquantile(candidate,.75,axis=(0,1))-
        np.nanquantile(candidate,.25,axis=(0,1))
    )
    finite_nondegenerate = np.isfinite(median_candidate) & np.isfinite(iqr_candidate) & (iqr_candidate>1e-5)
    selected_channels = names_initial[keep][finite_nondegenerate].tolist()
    chosen = np.flatnonzero(keep)[finite_nondegenerate]
    if len(selected_channels)<2:
        raise ValueError("Insufficient independently observable nonconstant physiological channels")
    medians = median_candidate[finite_nondegenerate]
    scales = np.maximum(iqr_candidate[finite_nondegenerate],1e-3)
    values = values[:, :, chosen]
    mask = mask[:, :, chosen]

    def to_split(indices: np.ndarray) -> DiscoverySplit:
        history = values[indices, :input_window]
        history_mask = mask[indices, :input_window]
        future = values[indices, input_window:input_window+forecast_horizon]
        future_mask = mask[indices, input_window:input_window+forecast_horizon]
        x = np.where(history_mask>0, np.clip((history-medians)/scales,-6,6),0).astype(np.float32)
        yf = np.where(future_mask>0,np.clip((future-medians)/scales,-6,6),0).astype(np.float32)
        # Do not publish original RecordID in per-example explanations.
        return DiscoverySplit(x=x,mask=history_mask.astype(np.float32),
                              future=yf,future_mask=future_mask.astype(np.float32),
                              y=labels[indices],oracle_states=None,
                              episode_ids=indices.astype(np.int64))
    train,validation=to_split(train_idx),to_split(val_idx)
    if np.unique(train.y).size != 2 or np.unique(validation.y).size != 2:
        raise ValueError("Both mortality classes required in research splits")
    receipt = {
        "dataset": "PhysioNet/CinC Challenge 2012 Set A, official public source",
        "patients": 4000,
        "train": 2400, "validation": 600,
        "calibration_sealed": 400, "test_sealed": 600,
        "split_seed": int(split_seed),
        "set_a_sha256": sha256_file(set_a_zip),
        "outcomes_sha256": sha256_file(outcomes_a),
        "train_record_ids_sha256": sha256(np.sort(np.asarray(identifiers)[train_idx]).tobytes()).hexdigest(),
        "validation_record_ids_sha256": sha256(np.sort(np.asarray(identifiers)[val_idx]).tobytes()).hexdigest(),
        "selected_operator_channel_names": selected_channels,
        "channel_names": selected_channels,
        "n_channels": len(selected_channels),
        "minimum_train_patient_coverage": float(minimum_train_patient_coverage),
        "train_normalization_medians": medians.astype(float).tolist(),
        "train_normalization_iqr": scales.astype(float).tolist(),
        "train_positive_count": int(train.y.sum()),
        "validation_positive_count": int(validation.y.sum()),
        "train_only_feature_selection": True,
        "no_manual_concept_definitions_or_labels": True,
        "calibration_or_test_used": False,
        "diagnostic_warning": "PhysioNet ICU records are irregular, highly missing and not clinical ground-truth concepts",
    }
    return DiscoveryData(train, validation, receipt, medians, scales)
