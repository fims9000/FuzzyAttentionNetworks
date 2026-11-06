"""V38 actual deployable SOURCE-MODEL persistence, no unsafe pickle.

The original frozen Temporal ConceptFAN checkpoint and outcome-specific
signed head coefficients are stored separately. This file is a read-only
JSON representation of the MEASURED source: blind learned projection
matrices, TRAIN-only q ranges, monitor threshold, and independent
actuator calibration receipts. NEVER stores SIMULATOR H/latent source S,
patient records, medical label or TEST labels.

This provides a reproducible reopen boundary, rather than just a CI
benchmark whose frozen source fit is lost after training.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
import json
from pathlib import Path
import numpy as np

from .threeview_centered_v37 import (
    CenteredPairFitV37,ThreeViewFitV37,PAIRS,
)
from .exogenous_probe_gate_v38 import (
    SentinelFitV38,ProbeGateV38,
)

VERSION=1


@dataclass(frozen=True)
class ReopenedInstrumentedSourceV38:
    source:ThreeViewFitV37
    monitor:SentinelFitV38
    probe:ProbeGateV38


def _safe_json(v):
    if isinstance(v,np.ndarray):
        return _safe_json(v.tolist())
    if isinstance(v,dict):
        return {str(k):_safe_json(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):
        return [_safe_json(x) for x in v]
    if isinstance(v,(np.floating,float)):
        z=float(v)
        if not np.isfinite(z):
            raise ValueError("Cannot save nonfinite scientific source state")
        return z
    if isinstance(v,(np.integer,int)) and not isinstance(v,(bool,np.bool_)):
        return int(v)
    if isinstance(v,(bool,np.bool_,str)):
        return bool(v) if isinstance(v,(bool,np.bool_)) else v
    raise TypeError("Unsupported source state serialization type")


def _encode(source:ThreeViewFitV37,
            monitor:SentinelFitV38,
            probe:ProbeGateV38):
    if (set(source.pairs)!=set(PAIRS)
        or source.true_physical_semantics_certified
        or monitor.physical_source_identity_certified
        or probe.biological_causal_semantics_proven
        or probe.latent_H_or_S_supplied_to_calibrator
        or not source.view_partition_externally_assumed
        or not source.independent_view_noise_externally_assumed
        or not monitor.independently_wired_instrument_artifact_monitor_assumed):
        raise ValueError("Refuse to persist unverified physical semantics as certified")
    return {
        "schema_version":VERSION,
        "schema":"conceptfan-v38-measured-threeview-external-evidence",
        "source_fits":_safe_json(asdict(source)),
        "sentinel_train_only":_safe_json(asdict(monitor)),
        "external_actuator_audit":_safe_json(asdict(probe)),
        "contains_true_H_or_latent_source":False,
        "source_semantics_medically_certified":False,
        "actuator_mapping_and_control_wiring_are_external_assumptions":True,
    }


def save_source_state_v38(
    file:Path,source:ThreeViewFitV37,
    monitor:SentinelFitV38,probe:ProbeGateV38,
)->Path:
    path=Path(file)
    if path.exists():
        raise FileExistsError("Refuse to overwrite frozen trained source state")
    payload=_encode(source,monitor,probe)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(
        payload,indent=2,ensure_ascii=False,allow_nan=False))
    return path


def load_source_state_v38(path:Path)->ReopenedInstrumentedSourceV38:
    payload=json.loads(Path(path).read_text())
    if (payload.get("schema_version")!=VERSION or
        payload.get("schema")!="conceptfan-v38-measured-threeview-external-evidence"
        or payload.get("contains_true_H_or_latent_source") is not False or
        payload.get("source_semantics_medically_certified") is not False or
        payload.get("actuator_mapping_and_control_wiring_are_external_assumptions") is not True):
        raise ValueError("Unsupported or falsely certified V38 source model state")
    raw=payload["source_fits"]
    if set(raw["pairs"])!=set(PAIRS):
        raise ValueError("Missing registered independent measured view fit")
    pairs={}
    for key,expected in PAIRS.items():
        v=dict(raw["pairs"][key])
        if (v.get("pair_name")!=key
            or tuple(v.get("view_a",()))!=expected[0]
            or tuple(v.get("view_b",()))!=expected[1]
            or v.get("original_mixing_supplied") is not False
            or v.get("physically_signed_source_identified") is not False):
            raise ValueError("Invalid view partition or false source identity")
        for field in (
            "left_eigenvectors","right_mixing","inv_left","inv_right",
        ):
            arr=np.asarray(v[field],dtype=np.float64)
            if arr.shape!=(2,2) or not np.isfinite(arr).all():
                raise ValueError("Invalid stored 2x2 blind source decomposition")
            v[field]=arr
        ev=np.asarray(v["eigenvalues"],dtype=np.float64)
        if ev.shape!=(2,) or not np.isfinite(ev).all():
            raise ValueError("Invalid registered TRAIN source eigensystem")
        v["eigenvalues"]=ev
        if (not np.allclose(v["inv_left"]@v["left_eigenvectors"],np.eye(2),atol=1e-6)
            or not np.allclose(v["inv_right"]@v["right_mixing"],np.eye(2),atol=1e-6)):
            raise ValueError("Corrupt blind source projection inverse")
        v["view_a"]=tuple(v["view_a"])
        v["view_b"]=tuple(v["view_b"])
        pairs[key]=CenteredPairFitV37(**v)
    base={k:v for k,v in raw.items() if k!="pairs"}
    source=ThreeViewFitV37(pairs=pairs,**base)
    monitor=SentinelFitV38(**payload["sentinel_train_only"])
    p=dict(payload["external_actuator_audit"])
    for field in (
        "target_mean_effect_half","nuisance_mean_effect_half",
        "target_bootstrap_lower_half","target_vs_nuisance_half",
        "target_supported_pairs_half","nuisance_supported_pairs_half",
    ):
        p[field]=tuple(p[field])
        if len(p[field])!=2:
            raise ValueError("Invalid independently checked calibration halves")
    probe=ProbeGateV38(**p)
    if (source.true_physical_semantics_certified
        or monitor.physical_source_identity_certified
        or probe.biological_causal_semantics_proven
        or probe.latent_H_or_S_supplied_to_calibrator
        or not monitor.independently_wired_instrument_artifact_monitor_assumed
        or monitor.min_observed_times!=10
        or source.train_max_pairwise_q_discrepancy_q95<.12):
        raise ValueError("Invalid source, instrument or calibration provenance")
    return ReopenedInstrumentedSourceV38(source,monitor,probe)
