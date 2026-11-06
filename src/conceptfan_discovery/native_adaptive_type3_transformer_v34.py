"""V34 native Temporal ConceptFAN + in-graph audited higher-order concept slot.

The known concept encoder is the REAL MultiSetAdditiveTemporalConceptFAN
Transformer of this repository. Known concept definitions, temporal
aggregation and two-concept TRAIN-fitted risk head are frozen. A new source
slot enters THE SAME torch.forward inference graph through its membership
and separately trainable gamma, not by posthoc textual explanations.
There is no raw sensor bypass to prediction: unknown sensor groups are
consumed only by the measured-source/flow representation adapter.

This initial native extension preserves K2 neurons. It does NOT claim to
retrofit a new trainable source into the original transformer's self-attention
layers; doing that would invalidate exact preservation and demands a
separate controlled ablation.
"""
from __future__ import annotations

from hashlib import sha256
from dataclasses import asdict
from typing import Any
import numpy as np
import torch
from torch import nn

from .adaptive_higher_order_concepts_v34 import (
    AdditiveFuzzySourceSlot,FuzzySourceDefinition,
)


class NativeAuditedFuzzyConceptFANV34(nn.Module):
    def __init__(
        self,
        core:nn.Module,
        original_k2_head_weights:np.ndarray,
        original_k2_head_bias:float,
        source_definition:FuzzySourceDefinition,
        *,
        fuzzy_order:str="auto",
    ):
        super().__init__()
        weights=torch.as_tensor(original_k2_head_weights,dtype=torch.float32)
        if weights.shape!=(2,) or not bool(torch.isfinite(weights).all()):
            raise ValueError("An actual TRAIN-fitted two-concept head is required")
        if not np.isfinite(original_k2_head_bias):
            raise ValueError("A finite frozen head bias is required")
        self.core=core
        self.core.requires_grad_(False)
        self.core.eval()
        self.register_buffer("known_weights",weights.clone())
        self.register_buffer("known_bias",
                             torch.tensor(float(original_k2_head_bias)))
        self.source_slot=AdditiveFuzzySourceSlot(
            source_definition,order=fuzzy_order)
        self.source_definition=source_definition

    def train(self,mode:bool=True):
        """Never unfreeze or change the original known-concept encoder."""
        super().train(mode)
        self.core.eval()
        return self

    def known_model_sha256(self)->str:
        state=self.core.state_dict()
        payload=b"".join(
            name.encode("utf-8")+b"\x00"+
            tensor.detach().cpu().contiguous().numpy().tobytes()
            for name,tensor in sorted(state.items())
        )
        return sha256(payload).hexdigest()

    def forward(
        self,
        measured_known_values:torch.Tensor,
        known_observation_mask:torch.Tensor,
        *,
        extra_source_amplitude:torch.Tensor,
        extra_source_uncertainty:torch.Tensor,
        extra_source_observed_information:torch.Tensor,
        extra_source_gate_approved:bool=False,
    )->dict[str,Any]:
        values=torch.as_tensor(measured_known_values,dtype=torch.float32)
        mask=torch.as_tensor(known_observation_mask,dtype=torch.float32)
        if (values.ndim!=3 or values.shape!=mask.shape
            or values.shape[1:]!=(16,4)
            or not bool(torch.isfinite(values[mask.bool()]).all())
            or not bool(torch.isfinite(mask).all())
            or bool(((mask!=0)&(mask!=1)).any())):
            raise ValueError("Original K2 Transformer accepts only observed [N,16,4] channels")
        clean=torch.where(mask.bool(),values,torch.zeros_like(values))
        original_input=torch.cat([clean,mask],dim=-1)
        with torch.no_grad():
            original=self.core(original_input)
        known=original.concept_summaries
        if known.shape!=(len(values),2):
            raise AssertionError("Native Temporal ConceptFAN no longer exposes two concept slots")
        contributions=known*self.known_weights.view(1,-1)
        base=self.known_bias+contributions.sum(dim=1)
        additional=self.source_slot(
            base,extra_source_amplitude,extra_source_uncertainty,
            extra_source_observed_information,
            approved=extra_source_gate_approved)
        logits=additional["logit"]
        if not bool(torch.allclose(
            logits,self.known_bias+contributions.sum(dim=1)+
            additional["source_contribution"],atol=1e-6,rtol=0)):
            raise AssertionError("Native fuzzy concept prediction bypassed additive head")
        return {
            "logit":logits,
            "probability":torch.sigmoid(logits),
            "known_concepts":known,
            "known_concept_trajectories":original.concept_trajectories,
            "known_temporal_attention":original.temporal_concept_weights,
            "known_contributions":contributions,
            "known_bias":self.known_bias,
            "old_k2_logit":base,
            "new_concept_value":additional["source_membership"].center,
            "new_concept_membership_bundle":additional["source_membership"],
            "new_concept_used":additional["source_used"],
            "new_concept_contribution":additional["source_contribution"],
            "new_concept_risk_lower":additional["source_contribution_lower"],
            "new_concept_risk_upper":additional["source_contribution_upper"],
            "source_definition":self.source_definition,
            "source_gate_approved_independent_VALID":bool(
                extra_source_gate_approved),
            "new_source_input_is_inferred_not_oracle":True,
            "original_native_transformer_frozen":True,
            "raw_source_input_risk_bypass":False,
            "physical_concept_identity_certified":False,
        }


def readable_source_explanation(
    native_output:dict, index:int, *,
    has_independent_external_device_calibration:bool,
)->dict:
    """Human-facing trace with observables, uncertainty and explicit limits.

    No invented clinical label, causal claim or "80% disease confidence".
    """
    logits=native_output["logit"].detach().cpu()
    q=native_output["new_concept_value"].detach().cpu()
    used=native_output["new_concept_used"].detach().cpu()
    info=(native_output["new_concept_membership_bundle"]
          .source_observed_information.detach().cpu())
    deltas=native_output["new_concept_contribution"].detach().cpu()
    lo=native_output["new_concept_risk_lower"].detach().cpu()
    hi=native_output["new_concept_risk_upper"].detach().cpu()
    d=native_output["source_definition"]
    if not 0<=index<len(logits):
        raise IndexError("Patient index outside saved native inference")
    ok=bool(used[index])
    reason=("Недостаточно подтвержденных измерений или источник не "
            "прошёл независимую проверку") if not ok else (
                "Измерительный временной источник допущен, "
                "но физический смысл не установлен")
    return {
        "source_id":d.source_id,
        "source_sensor_channels":list(d.sensor_indices),
        "source_kind":"наблюдательный временной паттерн",
        "source_physically_identified":False,
        "source_observed_information_fraction":round(float(info[index]),5),
        "source_evidence_origin":"измеренные датчики и помеченная модельная реконструкция",
        "source_membership_degree_not_probability":round(float(q[index]),5),
        "fuzzy_order":native_output[
            "new_concept_membership_bundle"].selected_order,
        "independent_device_calibration_available":bool(
            has_independent_external_device_calibration),
        "new_source_used":ok,
        "new_source_risk_logit_contribution":round(float(deltas[index]),5),
        "new_source_risk_contribution_algebraic_lower":round(float(lo[index]),5),
        "new_source_risk_contribution_algebraic_upper":round(float(hi[index]),5),
        "final_risk_probability":round(float(torch.sigmoid(logits[index])),5),
        "explanation_in_plain_russian":reason,
        "membership_is_not_outcome_probability":True,
        "algebraic_interval_is_not_statistical_confidence":True,
        "unobserved_real_values_are_not_proven_to_exist":True,
    }


def render_concept_explanation_ru(
    native_output:dict,index:int,
    *,
    observed_source_template:list[float]|tuple[float,...]|None=None,
)->dict:
    """Readable FIRST-PERSON-NEUTRAL user report; numbers only for risk effect.

    Fuzzy membership degrees and alpha planes remain in a separate
    technical audit and are NEVER presented as patient event probabilities.
    Description is based on measured TRAIN source template, not an LLM
    guessed medical label. The actual generator truth is not available.
    """
    raw=readable_source_explanation(
        native_output,index,
        has_independent_external_device_calibration=(
            native_output["source_definition"].external_anchor_verified))
    risk=native_output["probability"].detach().cpu()
    baseline=torch.sigmoid(native_output["old_k2_logit"].detach().cpu())
    n=len(risk)
    if not 0<=index<n:
        raise IndexError("Observation outside native ConceptFAN inference")
    delta_pp=100.*float(risk[index]-baseline[index])
    channels=", ".join(str(j) for j in raw["source_sensor_channels"])
    t="Временная структура источника определяется по TRAIN-измерениям."
    if observed_source_template is not None:
        phi=np.asarray(observed_source_template,dtype=float)
        if phi.shape!=(16,) or not np.isfinite(phi).all():
            raise ValueError("Only complete TRAIN-derived 16-step templates")
        early=float(phi[:8].mean())
        late=float(phi[8:].mean())
        if early-late>0.35:
            t="В исходных TRAIN-данных наблюдался более высокий сигнал в начале ряда, затем снижение."
        elif late-early>0.35:
            t="В исходных TRAIN-данных наблюдалось повышение сигнала к концу ряда."
        else:
            t="Устойчивого направления раннего и позднего участка не установлено."
    info=raw["source_observed_information_fraction"]
    source_used=raw["new_source_used"]
    if not source_used:
        if info<.50:
            explanation=("Новый временной источник не использован: "
                         "фактических измерений недостаточно для оценки.")
        else:
            explanation=("Новый временной источник не использован: "
                         "его вклад не прошёл независимую проверку.")
        decision="Оставлен прогноз исходных двух концептов."
    else:
        if delta_pp>=.2:
            action="повысил"
        elif delta_pp<=-.2:
            action="снизил"
        else:
            action="практически не изменил"
        explanation=(
            f"Группа датчиков {channels} показывает согласованную "
            f"временную структуру. {t} Часть скрытой траектории "
            f"представлена модельными оценками, а не измерениями.")
        decision=(f"Добавленный источник {action} оценку вероятности "
                  f"события данной моделью. Это не доказательство причины.")
    return {
        "name":f"Автоматически найденный временной паттерн: датчики {channels}",
        "what_was_observed":t,
        "source_channels":raw["source_sensor_channels"],
        "measurement_evidence":(
            f"По доступным измерениям определяется {100*info:.0f}% "
            "информации об обученном временном шаблоне."),
        "what_was_inferred":(
            "Ненаблюдаемые участки оцениваются статистической моделью; "
            "они не выдаются за реальные измерения."),
        "what_the_model_did":explanation,
        "how_prediction_changed":decision,
        "before_after_model_probabilities":{
            "without_added_source":round(float(baseline[index]),4),
            "with_added_source":round(float(risk[index]),4),
            "difference_in_percentage_points":round(delta_pp,2),
        },
        "insufficient_evidence_causes_abstention":not source_used,
        "scientific_limitation":(
            "Физический смысл источника не подтверждён. Изменение "
            "прогноза не является причинным эффектом."),
        "technical_audit":raw,
        "fuzzy_membership_not_event_probability":True,
        "no_invented_medical_label":True,
    }
