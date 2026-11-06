"""V37 end-user evidence passport: operational explanation, NOT a clinical name.

Never display raw fuzzy grade or eigenvector as the explanation. Original
and expanded probability are model predictions, not causal interventions.
Exactly distinguish absent observation vs mismatched independent views vs
failed two-disjoint-VALID gate. A mean-centered fluctuation has no signed
physical identity even if outcome-predictive.
"""
from __future__ import annotations
import numpy as np


def explain_threeview_source_v37(
    measured:dict,original_event_probability:np.ndarray,
    new_event_probability:np.ndarray,
    *,
    approved_by_two_valid_environments:bool,
    trained_pair_discrepancy_limit:float,
    index:int,
)->dict:
    old=np.asarray(original_event_probability,dtype=np.float64).reshape(-1)
    new=np.asarray(new_event_probability,dtype=np.float64).reshape(-1)
    src=np.asarray(measured["support"],dtype=np.float64).reshape(-1)
    paired=np.asarray(measured["pair_timepoints"],dtype=np.int64)
    discrepancy=np.asarray(
        measured["pairwise_max_membership_discrepancy"],dtype=np.float64).reshape(-1)
    if (old.shape!=new.shape or old.shape!=src.shape or
        paired.shape!=(len(old),3) or discrepancy.shape!=old.shape or
        not (0<=index<len(old)) or
        not np.isfinite(old).all() or not np.isfinite(new).all()
        or np.any((old<0)|(old>1)|(new<0)|(new>1))
        or not np.isfinite(trained_pair_discrepancy_limit)
        or trained_pair_discrepancy_limit<0):
        raise ValueError("Invalid model/measurement provenance for user explanation")
    enough=bool(measured["all_three_pairs_measured"][index])
    admitted=bool(src[index]>0 and approved_by_two_valid_environments)
    if not approved_by_two_valid_environments:
        explanation=(
            "Новый временной признак не прошёл две независимые проверки "
            "прогностической полезности; исходный прогноз сохранён.")
        reason="FAILED_TWO_VALID_ENVIRONMENTS"
    elif not enough:
        explanation=(
            "Для одной или нескольких пар измерительных каналов "
            "не хватает совместно наблюдавшихся временных точек. "
            "Восстанавливать скрытую траекторию вместо измерений "
            "модель не стала; сохранён исходный прогноз.")
        reason="INSUFFICIENT_OBSERVED_PAIRED_TIMES"
    elif not admitted:
        explanation=(
            "Три независимые оценки временного источника противоречат "
            "друг другу сильнее, чем допускала обучающая выборка. "
            "Вклад источника отклонён.")
        reason="CROSS_VIEW_DISAGREEMENT"
    else:
        pp=100*(new[index]-old[index])
        if abs(pp)<.05:
            impact="почти не изменил"
        elif pp>0:
            impact="повысил"
        else:
            impact="понизил"
        explanation=(
            "Обнаружены согласованные временные колебания нескольких "
            "измерительных каналов. После вычитания среднего каждого "
            "эпизода этот наблюдательный признак "+impact+
            " оценку вероятности события модели. Это не означает "
            "доказанной причины события.")
        reason="INDEPENDENT_VIEWS_AGREE_AND_TWO_VALID_PASS"
    if not admitted and not np.isclose(new[index],old[index],atol=2e-12,rtol=0):
        raise AssertionError("Unsupported concept changed risk despite required abstention")
    return {
        "name":"Автоматически обнаруженные временные колебания измерений",
        "kind":"измеренный временной паттерн, не клинический диагноз",
        "measured_channels":{
            "канал_A":[4,5],"канал_B":[6,7],"канал_C":[8,9],
        },
        "measured_joint_timesteps":{
            "AB":int(paired[index,0]),
            "AC":int(paired[index,1]),
            "BC":int(paired[index,2]),
            "out_of_total":16,
        },
        "what_was_observed":(
            "Сравнены реально наблюдавшиеся совместные временные точки "
            "трёх пар каналов; средние значения внутри эпизода вычтены."),
        "what_was_not_observed":(
            "Пропущенные значения не генерировались и не выдавались "
            "за реальные измерения."),
        "how_model_decided":explanation,
        "reason_code":reason,
        "source_was_used":admitted,
        "predicted_event_probability_without_new_source":round(float(old[index]),5),
        "predicted_event_probability_with_new_source":round(float(new[index]),5),
        "change_in_percentage_points":round(100*float(new[index]-old[index]),3),
        "not_a_causal_intervention":True,
        "limitations":(
            "Физический смысл, независимость ошибок приборов и корректность "
            "группировки каналов во внешнем мире не подтверждены. "
            "Согласованность нескольких датчиков может возникнуть "
            "из-за общей приборной помехи."),
        "technical_audit":{
            "pair_membership_not_probabilities":
                np.asarray(measured["pair_memberships"][index],dtype=np.float64).tolist(),
            "pair_cross_energies":
                np.asarray(measured["pair_energy"][index],dtype=np.float64).tolist(),
            "pair_discrepancy":float(discrepancy[index]),
            "train_only_discrepancy_max":float(trained_pair_discrepancy_limit),
        },
        "no_invented_medical_semantics":True,
    }
