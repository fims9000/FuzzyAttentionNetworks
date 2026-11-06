"""V38 production-shaped, source-grounded SINGLE and BATCH inference.

No generator, oracle H/S, training, validation, or TEST labels are imported
or needed for inference. Source is loaded from safe validated JSON;
native K2 source weights from the experiment checkpoint (trusted file only)
and the original TRAIN-only risk head from a validated JSON manifest.

V38 is a synthetic RESEARCH model: these risk numbers are not medical
probabilities, causal explanations, or calibrated clinical predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch
from scipy.special import expit

from .source_persistence_v38 import (
    ReopenedInstrumentedSourceV38,load_source_state_v38,
)
from .exogenous_probe_gate_v38 import (
    source_with_independent_evidence_v38,
)
from .auto_measured_concept_admission import admitted_test_logit

# Current trained K2 architecture is canonical V30/V38 temporal attention.
from scripts.experiment_nonlinear_anchor_repair_v30 import make_firewalled_k2
from scripts.experiment_trained_conceptfan_omission_v25 import (
    predict_native,sha_state,
)


@dataclass
class V38Inference:
    device_seed:int
    nn_seed:int
    outcome_case:str
    native_model:torch.nn.Module
    model_sha:str
    head_weights:np.ndarray
    head_bias:float
    new_gamma:float
    validation_approved:bool
    source:ReopenedInstrumentedSourceV38

    def predict(self,values,observed,*,max_episodes:int=10000)->dict:
        """Score a real inference batch, with precise per-episode refusals."""
        x=np.asarray(values)
        mask=np.asarray(observed)
        if (x.ndim!=3 or x.shape[1:]!=(16,12) or
            mask.shape!=x.shape or mask.dtype!=np.bool_
            or not 1<=len(x)<=max_episodes
            or not np.issubdtype(x.dtype,np.number)
            or np.isinf(x).any() or not np.isfinite(x[mask]).all()):
            raise ValueError("Expected [N,16,12] finite observed sensor values and Boolean observed mask")
        # No unobserved payload is ever given to the model.
        zero=np.where(mask,x,0.).astype(np.float32,copy=False)
        with torch.no_grad():
            q=predict_native(
                self.native_model,zero[:,:,:4],mask[:,:,:4],
            )["concept_summary"]
        if q.shape!=(len(zero),2):
            raise AssertionError("Expected two authentic frozen neural concepts")
        base=self.head_bias+q@self.head_weights
        source=source_with_independent_evidence_v38(
            self.source.source,self.source.monitor,self.source.probe,zero,mask)
        logit,used=admitted_test_logit(
            base,source["q"],source["support"],
            gamma=self.new_gamma,approved=self.validation_approved,
            minimum_fraction=.50,
        )
        np.testing.assert_array_equal(logit[~used],base[~used])
        risk=expit(logit)
        baseline=expit(base)
        monitor=source["instrument_control"]
        n_times=np.asarray(source["pair_timepoints"])
        disagree=np.asarray(source["source_refused_due_to_pair_disagreement"])
        enough=np.asarray(source["all_three_pairs_measured"])
        result=[]
        for i in range(len(zero)):
            if not self.source.probe.passed:
                reason="EXTERNAL_CALIBRATION_REFUSED"
            elif int(monitor["sentinel_observed_timesteps"][i])<10:
                reason="MONITOR_UNOBSERVED"
            elif not bool(monitor["eligible"][i]):
                reason="MONITOR_DETECTED_ANOMALY"
            elif not bool(enough[i]):
                reason="INSUFFICIENT_PAIRED_MEASUREMENTS"
            elif bool(disagree[i]):
                reason="SOURCE_VIEWS_DISAGREE"
            elif not self.validation_approved:
                reason="FAILED_TWO_INDEPENDENT_VALIDATIONS"
            elif not bool(used[i]):
                reason="SOURCE_EVIDENCE_INSUFFICIENT"
            else:
                reason="SOURCE_ACCEPTED"
            entry={
                "episode_index_in_batch":i,
                "model_event_probability":float(risk[i]),
                "model_event_probability_without_extra_source":
                    float(baseline[i]),
                "source_probability_change_percentage_points":
                    100.*float(risk[i]-baseline[i]),
                "used_extra_source":bool(used[i]),
                "decision_code":reason,
                "source_channels":{
                    "A":[4,5],"B":[6,7],"C":[8,9],
                    "independent_noise_monitor":[10],
                },
                "joint_observed_times":{
                    "AB":int(n_times[i,0]),
                    "AC":int(n_times[i,1]),
                    "BC":int(n_times[i,2]),
                    "total_times":16,
                },
                "monitor_observed_times":int(
                    monitor["sentinel_observed_timesteps"][i]),
                "monitor_variance":float(
                    monitor["sentinel_centered_variance"][i]),
                "monitor_TRAIN_threshold":float(
                    self.source.monitor.threshold_train_only),
                "observational_concept_value_not_a_probability":
                    float(source["q"][i]) if bool(used[i]) else None,
                "validation_gate_approved":self.validation_approved,
                "external_actuator_certificate_passed":self.source.probe.passed,
                "uncertainty":{
                    "independent_monitor_wiring_unverified_in_real_world":True,
                    "source_physical_identity_unverified":True,
                    "unmonitored_aligned_artifact_may_be_indistinguishable":True,
                    "no_imputed_values_used":True,
                },
            }
            if not entry["used_extra_source"] and not np.isclose(
                entry["model_event_probability"],
                entry["model_event_probability_without_extra_source"],
                atol=1e-12,rtol=0,
            ):
                raise AssertionError("Refusal modified the frozen K2 prediction")
            result.append(entry)
        return {
            "schema":"conceptfan-v38-research-inference-v1",
            "device_seed":self.device_seed,
            "nn_seed":self.nn_seed,
            "outcome_case":self.outcome_case,
            "native_K2_state_SHA":self.model_sha,
            "synthetic_research_not_clinically_validated":True,
            "physical_or_causal_semantics_certified":False,
            "n_episodes":len(result),
            "episodes":result,
        }


def _safe_bundle_file(base:Path,name:str)->Path:
    if not isinstance(name,str) or len(name)>300:
        raise ValueError("Invalid model bundle relative path")
    root=base.resolve()
    path=(root/name).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError("Missing or unsafe model bundle path")
    return path


def load_v38_inference_model(
    artifact_dir:Path,
    *,
    device_seed:int,
    nn_seed:int,
    outcome_case:str,
)->V38Inference:
    """Load only audited signed original native K2, not arbitrary python objects."""
    base=Path(artifact_dir)
    manifests=json.loads((base/"INFERENCE_BUNDLES.json").read_text())
    candidates=[x for x in manifests if (
        x.get("device_seed"),x.get("NN_seed"),x.get("outcome_case")
    )==(device_seed,nn_seed,outcome_case)]
    if len(candidates)!=1:
        raise ValueError("No unique frozen TRAIN risk head for requested device/seed/outcome")
    m=candidates[0]
    if m.get("source_physically_certified_realworld") is not False:
        raise ValueError("Refuse unsupported certified causal/physical source")
    if m.get("new_source_requires_independent_probe_pass") is not True:
        raise ValueError("Cannot omit the external reference-assumption boundary")
    weights=np.asarray(m["two_concept_original_TRAIN_head"]["weights"],
                       dtype=np.float64)
    bias=float(m["two_concept_original_TRAIN_head"]["bias"])
    gamma=float(m["new_source_gamma_TRAIN_only"])
    if (weights.shape!=(2,) or not np.isfinite(weights).all()
        or not np.isfinite(bias) or not np.isfinite(gamma)
        or not isinstance(m.get("new_source_passed_two_independent_VALID"),bool)):
        raise ValueError("Missing trusted TRAIN-only risk or validation parameters")
    source=load_source_state_v38(
        _safe_bundle_file(base,m["source_model_state"]))
    # Only TRUSTED, versioned local model snapshots. Torch weights_only=True
    # does not turn an arbitrary downloaded checkpoint into an authenticated one.
    saved=torch.load(
        _safe_bundle_file(base,m["native_checkpoint"]),
        map_location="cpu",weights_only=True,
    )
    if not isinstance(saved,dict) or "state_dict" not in saved:
        raise ValueError("Invalid frozen native model checkpoint")
    native=make_firewalled_k2(nn_seed)
    native.load_state_dict(saved["state_dict"],strict=True)
    native.eval()
    digest=sha_state(native)
    if digest!=m["frozen_old_K2_weights_SHA"]:
        raise ValueError("Frozen original K2 checkpoint checksum mismatch")
    return V38Inference(
        device_seed,nn_seed,outcome_case,native,digest,
        weights,bias,gamma,
        m["new_source_passed_two_independent_VALID"],source,
    )
