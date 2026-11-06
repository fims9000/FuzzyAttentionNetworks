"""V38: real Temporal ConceptFAN + external probe and negative-control guard.

RESEARCH ONLY. A known randomized actuator and independently instrumented
sentinel are simulator-supplied EXTERNAL supervision, not unsupervised discovery
of physiological or causal concepts. H, hidden S and TEST Y never enter the
sensor-source fitting/gate. All TEST worlds evaluated without cherry-picking.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import torch
from scipy.special import expit,logit

from conceptfan_discovery.exogenous_probe_world_v38 import (
    DEVICE_SEEDS,CASES,STRESSES,generate_exogenous_world_v38,
    apply_artifact_v38,mask_world_v38,
)
from conceptfan_discovery.exogenous_probe_gate_v38 import (
    fit_sentinel_train_only_v38,fit_probe_intervention_v38,
    source_with_independent_evidence_v38,negative_control_runtime_v38,
)
from conceptfan_discovery.source_persistence_v38 import (
    save_source_state_v38,load_source_state_v38,
)
from conceptfan_discovery.threeview_centered_v37 import (
    fit_centered_threeview_v37,threeview_feature_v37,pair_feature_v37,
)
from conceptfan_discovery.auto_measured_concept_admission import (
    evaluate_two_independent_environments,admitted_test_logit,
)
from conceptfan_discovery.isolated_source_admission_v28 import (
    train_source_coefficient,
)
from conceptfan_discovery.cross_environment_transport import (
    paired_test_ap_difference,
)
from conceptfan_discovery.adaptive_higher_order_concepts_v34 import (
    FuzzySourceDefinition,
)
from conceptfan_discovery.native_adaptive_type3_transformer_v34 import (
    NativeAuditedFuzzyConceptFANV34,
)
from scripts.experiment_automatic_measured_concepts import (
    train_masked_measured_known_concepts,logit_from_head,
)
from scripts.experiment_nonlinear_anchor_repair_v30 import (
    make_firewalled_k2,
)
from scripts.experiment_frozen_concept_head_refit_v26 import (
    refit_linear_concept_head,
)
from scripts.experiment_trained_conceptfan_omission_v25 import (
    predict_native,sha_state,metrics,
)
from scripts.experiment_threeview_centered_v37 import (
    _raw_fit,_raw_feature,
)

NATIVE_SEEDS=(11,23,37)
SOURCE_METHODS=("AB_centered","ABC_threeview",
                "ABC_monitor","ABC_probe_monitor")
NATIVE_METHODS=("K2",)+SOURCE_METHODS
RAW_METHODS=("raw_temporal_ALL1800","raw_quadratic_ALL1800")
METHODS=NATIVE_METHODS+RAW_METHODS
BOOT_COMPARISONS=(
    ("healthy","MCAR20","ABC_probe_monitor","K2"),
    ("healthy","MCAR20","ABC_probe_monitor","ABC_threeview"),
    ("healthy","MCAR20","ABC_probe_monitor","raw_quadratic_ALL1800"),
    ("monitored_AB_common_noise","MCAR20","ABC_probe_monitor","ABC_threeview"),
    ("monitored_ABC_common_noise","MCAR20","ABC_probe_monitor","ABC_threeview"),
    ("UNMONITORED_ABC_noise","MCAR20","ABC_probe_monitor","K2"),
    ("sentinel_false_alarm","MCAR20","ABC_probe_monitor","ABC_threeview"),
    ("healthy","missing_sentinel","ABC_probe_monitor","K2"),
    ("one_view_noise_shift","MCAR20","ABC_probe_monitor","ABC_threeview"),
)


def _mask(x,stress,*,device_seed,extra_seed):
    return mask_world_v38(x,stress,
                          seed=extra_seed+device_seed%101,
                          known_seed=20263878+device_seed%101)


def _source_methods(fit,sentinel,probe,x,mask):
    original=threeview_feature_v37(fit,x,mask)
    ab=pair_feature_v37(fit.pairs["AB"],x,mask)
    guard=source_with_independent_evidence_v38(
        fit,sentinel,probe,x,mask,require_probe=False)
    certified=source_with_independent_evidence_v38(
        fit,sentinel,probe,x,mask,require_probe=True)
    return {
        "AB_centered":ab,
        "ABC_threeview":original,
        "ABC_monitor":guard,
        "ABC_probe_monitor":certified,
    }


def _aggregate(rows,device,case,stress,method,n_native):
    subset=[r for r in rows if (
        r["device_seed"],r["case"],r["stress"],r["method"]
    )==(device,case,stress,method)]
    expected=n_native if method in NATIVE_METHODS else 1
    if len(subset)!=expected:
        raise AssertionError(f"Missing evaluated method: {device}/{case}/{stress}/{method}")
    return {
        "device_seed":device,"case":case,"stress":stress,
        "method":method,
        "AUPRC":float(np.mean([s["metrics"]["auprc"] for s in subset])),
        "AUROC":float(np.mean([s["metrics"]["auroc"] for s in subset])),
        "NLL":float(np.mean([s["metrics"]["nll"] for s in subset])),
        "Brier":float(np.mean([s["metrics"]["brier"] for s in subset])),
        "source_use":float(np.mean([s["source_use"] for s in subset])),
    }


def _human_explanation(
    measured,source_used,base_probability,new_probability,
    *,probe_gate,sentinel_fit,case,stress,
):
    monitor=measured["instrument_control"]
    used=bool(source_used)
    timepoints=np.asarray(measured["pair_timepoints"][0],dtype=int)
    sentinel_times=int(monitor["sentinel_observed_timesteps"][0])
    if not probe_gate.passed:
        reason="Новый источник не допущен: независимый контрольный сигнал не подтвердил избирательную реакцию измерений."
        code="PROBE_CALIBRATION_FAILED"
    elif sentinel_times<10:
        reason="Контрольный датчик прибора недоступен: сведения о возможной общей помехе отсутствуют. Прогноз оставлен исходным."
        code="SENTINEL_MISSING"
    elif not monitor["eligible"][0]:
        reason="Контрольный приборный канал обнаружил аномальную изменчивость: источник исключён из прогноза."
        code="SENTINEL_ANOMALY"
    elif not measured["all_three_pairs_measured"][0]:
        reason="Не хватает совместных измерений одной или нескольких пар датчиков. Источник исключён."
        code="INSUFFICIENT_SOURCE_MEASUREMENTS"
    elif not used:
        reason="Данные пар датчиков расходятся или проверка полезности на двух независимых VALID отклонила источник."
        code="SOURCE_DISAGREEMENT_OR_VALID_REJECTION"
    else:
        reason=("Три измерительные пары согласованы. Источник связан с известным "
                "контрольным воздействием на измерительную систему. Изменение прогноза "
                "не доказывает причинности или физического смысла в реальном мире.")
        code="SOURCE_USED_WITH_EXTERNAL_PROBE_AND_MONITOR"
    if not used and not np.isclose(new_probability,base_probability,atol=1e-7):
        raise AssertionError("Refused source changed original K2 prediction")
    return {
        "case":case,"stress":stress,
        "source_name":"Измеримая энергия временных колебаний, связанная с внешним эталонным воздействием",
        "observed_views":{"A":[4,5],"B":[6,7],"C":[8,9],"artifact_monitor":[10]},
        "actually_measured_joint_timepoints":{
            "AB":int(timepoints[0]),"AC":int(timepoints[1]),
            "BC":int(timepoints[2]),"max":16,
        },
        "artifact_monitor_observed_timepoints":sentinel_times,
        "probe_is_known_external_physical_intervention":True,
        "probe_is_supervision_not_automatic_semantic_identification":True,
        "source_used":used,
        "decision_reason":reason,"reason_code":code,
        "before_model_probability":round(float(base_probability),5),
        "after_model_probability":round(float(new_probability),5),
        "change_in_percentage_points":round(
            100*(float(new_probability)-float(base_probability)),3),
        "limitations":(
            "Контрольный датчик должен быть независимо подключён к общей приборной помехе. "
            "Помеха, не попавшая в контрольный канал, может остаться незамеченной. "
            "Эксперимент синтетический, медицинский смысл концепта не установлен."),
        "fuzzy_membership_is_not_event_probability":True,
        "source_physically_identified_in_real_world":False,
        "no_missing_values_claimed_as_measurements":True,
    }


def run(output:Path,*,smoke=False):
    if output.exists():
        raise FileExistsError("Must preserve earlier immutable research receipts")
    torch.set_num_threads(2)
    n_train,n_valid,n_test=(360,300,240) if smoke else (1000,800,600)
    n_probe=180 if smoke else 240
    seeds=(11,) if smoke else NATIVE_SEEDS
    devices=DEVICE_SEEDS[:1] if smoke else DEVICE_SEEDS
    resamples=199 if smoke else 399
    all_rows=[];all_gates=[];all_source_fits=[];all_training=[]
    predictions={};labels_by_case={};checkpoints=[];human=[]
    source_states_to_save={};replay_source_records={}
    inference_bundles=[]
    n_fit=0

    def store_prob(device,case,stress,method,nn_seed,p):
        predictions[(device,case,stress,method,nn_seed)]=(
            np.asarray(p,dtype=np.float64).copy())

    for device in devices:
        world=generate_exogenous_world_v38(
            device_seed=device,n_train=n_train,n_valid=n_valid,
            n_test=n_test,per_actuator=n_probe)
        x=world["x"]
        tr,v1,v2,te=(world[k] for k in ("train","valid1","valid2","test"))
        train_mask=_mask(x[tr],"MCAR20",device_seed=device,extra_seed=20263801)
        valid1_mask=_mask(x[v1],"MCAR20",device_seed=device,extra_seed=20263802)
        valid2_mask=_mask(x[v2],"MCAR20",device_seed=device,extra_seed=20263803)
        measured_fit=fit_centered_threeview_v37(
            x[tr],train_mask,independent_view_noise_documented=True,
            seed=20263811+device%101,bootstrap=24)
        monitor_fit=fit_sentinel_train_only_v38(x[tr],train_mask)
        probe_fit=fit_probe_intervention_v38(
            measured_fit,world["independent_calibration_onoff"],
            seed=20263821+device%101,bootstrap_reps=199)
        source_states_to_save[device]=(measured_fit,monitor_fit,probe_fit)
        source_train=_source_methods(
            measured_fit,monitor_fit,probe_fit,x[tr],train_mask)
        source_valid1=_source_methods(
            measured_fit,monitor_fit,probe_fit,x[v1],valid1_mask)
        all_source_fits.append({
            "device_seed":device,
            "measured_fit":measured_fit.audit(),
            "source_identity_external_probe_gate":probe_fit.audit(),
            "negative_control_sentinel_train_fit":{
                "threshold":monitor_fit.threshold_train_only,
                "eligible_train_count":monitor_fit.train_monitor_episodes,
                "min_observed_sentinel_timepoints":monitor_fit.min_observed_times,
                "independently_wired_monitor_assumed":True,
            },
            "physical_semantics_verified":False,
            "latent_mix_and_sources_used_for_fit":False,
        })
        case_data={}
        for ci,(case,law,validation_fault,test_fault) in enumerate(CASES):
            valid2_x=apply_artifact_v38(
                x[v2],validation_fault,seed=20263860+ci+device%101)
            test_x=apply_artifact_v38(
                x[te],test_fault,seed=20263880+ci+device%101)
            stress_data={}
            for si,stress in enumerate(STRESSES):
                mask=_mask(test_x,stress,device_seed=device,
                           extra_seed=20263911+si)
                source=_source_methods(
                    measured_fit,monitor_fit,probe_fit,test_x,mask)
                stress_data[stress]={
                    "x":test_x,"mask":mask,"source":source,
                }
            case_data[case]={
                "label_law":law,
                "valid2_x":valid2_x,
                "valid2_source":_source_methods(
                    measured_fit,monitor_fit,probe_fit,
                    valid2_x,valid2_mask),
                "test":stress_data,
            }
            labels_by_case[(device,case)]=world["outcomes"][law][te]
        healthy_case=case_data["healthy"]["test"]["MCAR20"]
        replay_source_records[device]=(
            healthy_case["x"][:2].copy(),
            healthy_case["mask"][:2].copy(),
        )

        # For one and the SAME TEST subject/mask, offset +2 must
        # change neither the physical source energy nor sentinel decision.
        for method in SOURCE_METHODS:
            normal=case_data["healthy"]["test"]["MCAR20"]["source"][method]
            offset=case_data["offset_all_plus2"]["test"]["MCAR20"]["source"][method]
            if not np.array_equal(normal["support"],offset["support"]):
                raise AssertionError("TRAIN-centered source lost support invariance under constant offset")
            if not np.allclose(normal["q"],offset["q"],atol=5e-6,rtol=0):
                raise AssertionError("TRAIN-centered source lost value invariance under constant offset")

        for seed in seeds:
            core=make_firewalled_k2(seed)
            fit_metrics=train_masked_measured_known_concepts(
                core,x[tr,:,:4],train_mask[:,:,:4],
                seed=seed,epochs=2 if smoke else 10)
            sha0=sha_state(core)
            n_fit+=1
            checkpoints.append({
                "device_seed":device,"NN_seed":seed,"model_sha":sha0,
                "state_dict":{k:v.detach().cpu().clone()
                              for k,v in core.state_dict().items()},
            })
            all_training.append({
                "device_seed":device,"NN_seed":seed,
                "model_sha":sha0,**fit_metrics,
            })
            qtr=predict_native(
                core,x[tr,:,:4],train_mask[:,:,:4])["concept_summary"]
            qv1=predict_native(
                core,x[v1,:,:4],valid1_mask[:,:,:4])["concept_summary"]
            qv2=predict_native(
                core,x[v2,:,:4],valid2_mask[:,:,:4])["concept_summary"]
            test_known_mask=case_data["healthy"]["test"]["MCAR20"]["mask"][:,:,:4]
            qtest=predict_native(
                core,x[te,:,:4],test_known_mask)["concept_summary"]
            for ci,(case,law,validation_fault,test_fault) in enumerate(CASES):
                y=world["outcomes"][law]
                head=refit_linear_concept_head(qtr,y[tr])
                base_train=logit_from_head(head,qtr)
                base_v1=logit_from_head(head,qv1)
                base_v2=logit_from_head(head,qv2)
                base_test=logit_from_head(head,qtest)
                trained_gate={}
                for method in SOURCE_METHODS:
                    train_feature=source_train[method]
                    gamma=train_source_coefficient(
                        base_train,train_feature["q"],y[tr])
                    v1feature=source_valid1[method]
                    v2feature=case_data[case]["valid2_source"][method]
                    gate=evaluate_two_independent_environments(
                        method,gamma,
                        ("VALID1",y[v1],base_v1,
                         v1feature["q"],v1feature["support"]),
                        ("VALID2",y[v2],base_v2,
                         v2feature["q"],v2feature["support"]),
                        train_label_count=len(tr),
                        min_mean_gain=.003,min_support_fraction=.25,
                        min_subject_support=.50,
                        bootstrap_reps=resamples,
                        seed=20263960+device%101*100+ci*11+seed,
                    )
                    trained_gate[method]=(gamma,gate)
                    all_gates.append({
                        "device_seed":device,"case":case,"NN_seed":seed,
                        "method":method,
                        "gamma_train_only":gamma,
                        "gate":gate.as_dict(),
                    })
                primary_gamma,primary_gate=trained_gate["ABC_probe_monitor"]
                inference_bundles.append({
                    "device_seed":device,"NN_seed":seed,
                    "outcome_case":case,"outcome_law":law,
                    "native_checkpoint":(
                        f"CHECKPOINTS/native_K2_device{device}_seed{seed}.pt"),
                    "source_model_state":f"SOURCE_STATES/device{device}_V38_source.json",
                    "two_concept_original_TRAIN_head":{
                        "weights":[float(v) for v in np.asarray(head.weights)],
                        "bias":float(head.bias),
                    },
                    "new_source_gamma_TRAIN_only":float(primary_gamma),
                    "new_source_passed_two_independent_VALID":bool(primary_gate.approved),
                    "new_source_requires_independent_probe_pass":True,
                    "frozen_old_K2_weights_SHA":sha0,
                    "source_physically_certified_realworld":False,
                })
                for stress in STRESSES:
                    test=case_data[case]["test"][stress]
                    baseline=expit(base_test)
                    store_prob(device,case,stress,"K2",seed,baseline)
                    all_rows.append({
                        "device_seed":device,"case":case,"stress":stress,
                        "method":"K2","NN_seed":seed,
                        "metrics":metrics(y[te],baseline),
                        "source_use":0.,
                    })
                    for method in SOURCE_METHODS:
                        f=test["source"][method]
                        gamma,gate=trained_gate[method]
                        z,used=admitted_test_logit(
                            base_test,f["q"],f["support"],
                            gamma=gamma,approved=gate.approved,
                            minimum_fraction=.50)
                        np.testing.assert_array_equal(z[~used],base_test[~used])
                        p=expit(z)
                        store_prob(device,case,stress,method,seed,p)
                        all_rows.append({
                            "device_seed":device,"case":case,"stress":stress,
                            "method":method,"NN_seed":seed,
                            "metrics":metrics(y[te],p),
                            "source_use":float(used.mean()),
                            "independent_VALID_approved":bool(gate.approved),
                        })
                        if (case=="healthy" and stress=="MCAR20"
                            and method=="ABC_probe_monitor"):
                            n=min(12,len(te))
                            definition=FuzzySourceDefinition(
                                source_id="probe_anchored_temporal_fluctuation_energy",
                                sensor_indices=(4,5,6,7,8,9,10),
                                threshold_train=0.,
                                width_train=1.,
                                train_missingness_spread=0.,
                                boundary_spread_anchor=0.,
                                certified_calibration_environments=0,
                                external_anchor_verified=False,
                            )
                            native=NativeAuditedFuzzyConceptFANV34(
                                core,np.asarray(head.weights),head.bias,
                                definition,fuzzy_order="type1")
                            with torch.no_grad():
                                native.source_slot.gamma.fill_(gamma)
                            q=f["q"][:n]
                            native_output=native(
                                torch.from_numpy(test["x"][:n,:,:4]),
                                torch.from_numpy(test["mask"][:n,:,:4]),
                                extra_source_amplitude=torch.from_numpy(
                                    logit(np.clip(q,1e-6,1-1e-6)).astype(np.float32)),
                                extra_source_uncertainty=torch.zeros(n),
                                extra_source_observed_information=torch.from_numpy(
                                    f["support"][:n].astype(np.float32)),
                                extra_source_gate_approved=gate.approved)
                            predicted_native=native_output["probability"].detach().numpy()
                            if np.max(abs(predicted_native-p[:n]))>3e-5:
                                raise AssertionError("Native Transformer extension differs from evaluated model")
                            human.append({
                                "device_seed":device,"NN_seed":seed,
                                "native_model_sha":sha0,
                                "source_provenance":"measured 3 views + external actuator + external negative control",
                                "sample0":_human_explanation(
                                    f,used[0],baseline[0],p[0],
                                    probe_gate=probe_fit,sentinel_fit=monitor_fit,
                                    case=case,stress=stress),
                                "signed_physical_source_certified":False,
                            })
                if sha_state(core)!=sha0:
                    raise AssertionError("Source changed original frozen K2 weights")
            print(json.dumps({
                "event":"V38_ACTUAL_NATIVE_CONCEPTFAN_TRAINED",
                "device_seed":device,"NN_seed":seed,
                "K2_state_sha":sha0,
                "probe_external_validation_passed":probe_fit.passed,
            }),flush=True)

        # Strong raw temporal and quadratic baselines fit ALL available Y
        # (1000 TRAIN + 400 VALID1 + 400 VALID2); not equal FLOPs.
        for case,law,vfault,tfault in CASES:
            y=world["outcomes"][law]
            allx=np.concatenate((
                x[tr],x[v1],case_data[case]["valid2_x"]))
            allm=np.concatenate((train_mask,valid1_mask,valid2_mask))
            ally=np.concatenate((y[tr],y[v1],y[v2]))
            for method,quad in (
                ("raw_temporal_ALL1800",False),
                ("raw_quadratic_ALL1800",True),
            ):
                fitted=_raw_fit(allx,allm,ally,quad)
                for stress in STRESSES:
                    test=case_data[case]["test"][stress]
                    feat=_raw_feature(test["x"],test["mask"],quadratic=quad)
                    p=fitted.predict_proba(feat)[:,1]
                    store_prob(device,case,stress,method,None,p)
                    all_rows.append({
                        "device_seed":device,"case":case,"stress":stress,
                        "method":method,"NN_seed":None,
                        "metrics":metrics(y[te],p),
                        "source_use":0.,
                    })

    aggregate=[
        _aggregate(all_rows,device,case,stress,method,len(seeds))
        for device in devices for case,*_ in CASES
        for stress in STRESSES for method in METHODS
    ]
    ci=[]
    for device in devices:
        for k,(case,stress,a,b) in enumerate(BOOT_COMPARISONS):
            y=labels_by_case[(device,case)]
            def preds(method):
                z=seeds if method in NATIVE_METHODS else (None,)
                arr=np.stack([
                    predictions[(device,case,stress,method,seed)]
                    for seed in z
                ])
                return arr if len(arr)==len(seeds) else np.repeat(arr,len(seeds),axis=0)
            ci.append({
                "device_seed":device,"case":case,"stress":stress,
                "method_a":a,"method_b":b,
                **paired_test_ap_difference(
                    y,preds(a),preds(b),
                    seed=20264000+device%101*100+k,
                    resamples=resamples,
                ),
            })
    result={
        "status":"V38_SMOKE_ONLY" if smoke else "V38_INSTRUMENTED_CONCEPT_EXECUTED",
        "pre_registered_protocol":"docs/medical/CONCEPTFAN_V38_FROZEN_INSTRUMENTED_CONCEPT_PROTOCOL_20261106.md",
        "devices":list(devices),
        "n_train":n_train,"n_valid1":n_valid//2,
        "n_valid2":n_valid//2,"n_test":n_test,
        "n_independent_probe_pairs_per_type":n_probe,
        "native_seed_list":list(seeds),
        "n_real_native_models":n_fit,"n_test_method_rows":len(all_rows),
        "n_aggregated_results":len(aggregate),
        "n_two_VALID_risk_gates":len(all_gates),
        "n_paired_AP_contrasts":len(ci),
        "n_persisted_reopenable_source_models":len(source_states_to_save),
        "n_saved_outcome_specific_inference_bundles":len(inference_bundles),
        "source_model_persistence_json_no_pickle":True,
        "assumes_control_sensor10_really_wired_to_shared_artifact":True,
        "assumes_controlled_actuator_physically_targets_one_sim_source":True,
        "calibration_is_external_supervision_not_unsupervised_discovery":True,
        "source_fit_has_no_privileged_latent_truth_or_mixing_H":True,
        "no_TEST_outcome_used_in_calibration_or_thresholds":True,
        "unmonitored_common_noise_not_guaranteed_detectable":True,
        "no_clinical_or_causal_semantics_certified":True,
        "source_models":all_source_fits,
        "native_training":all_training,
        "dual_VALID_receipts":all_gates,
        "test_rows":all_rows,
        "summary":aggregate,
        "paired_AP_bootstrap":ci,
        "human_model_explanations":human,
        "outcome_head_and_source_state_manifests_saved":True,
    }
    output.mkdir(parents=True,exist_ok=False)
    for name,record in (
        ("STATUS.json",result),
        ("SUMMARY.json",aggregate),
        ("VALID_GATES.json",all_gates),
        ("SOURCE_CALIBRATION.json",all_source_fits),
        ("PAIRED_AP_BOOTSTRAP.json",ci),
        ("EXPLANATIONS_RU.json",human),
        ("INFERENCE_BUNDLES.json",inference_bundles),
    ):
        (output/name).write_text(
            json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False))
    stored={
        f"device{dev}_{case}_{stress}_{method}_seed{seed}":p
        for (dev,case,stress,method,seed),p in predictions.items()
    }
    for (dev,case),y in labels_by_case.items():
        stored[f"device{dev}_{case}_TEST_LABELS"]=y
    np.savez_compressed(output/"FROZEN_TEST_PREDICTIONS.npz",**stored)
    (output/"SOURCE_STATES").mkdir()
    for device,(fit,monitor,probe) in source_states_to_save.items():
        source_path=output/"SOURCE_STATES"/f"device{device}_V38_source.json"
        save_source_state_v38(source_path,fit,monitor,probe)
        reopened=load_source_state_v38(source_path)
        measured,mask=replay_source_records[device]
        before=source_with_independent_evidence_v38(
            fit,monitor,probe,measured,mask)
        after=source_with_independent_evidence_v38(
            reopened.source,reopened.monitor,reopened.probe,
            measured,mask)
        np.testing.assert_array_equal(before["q"],after["q"])
        np.testing.assert_array_equal(before["support"],after["support"])
    (output/"CHECKPOINTS").mkdir()
    for ckpt in checkpoints:
        torch.save(
            ckpt,output/"CHECKPOINTS"/(
                f"native_K2_device{ckpt['device_seed']}_seed{ckpt['NN_seed']}.pt"))
    print(json.dumps({
        "event":"V38_COMPLETE",
        "status":result["status"],
        "n_models":n_fit,
        "n_rows":len(all_rows),
        "n_summary":len(aggregate),
        "n_gates":len(all_gates),
        "probe_per_device":[
            {"device_seed":rec["device_seed"],
             "approved":rec["source_identity_external_probe_gate"]["passed"],
             "target_mean":rec["source_identity_external_probe_gate"]["target_mean_effect_half"],
             "nuisance_mean":rec["source_identity_external_probe_gate"]["nuisance_mean_effect_half"]}
            for rec in all_source_fits
        ],
        "healthy_MCAR20":[r for r in aggregate
            if r["case"]=="healthy" and r["stress"]=="MCAR20"],
        "monitored_ABC_MCAR20":[r for r in aggregate
            if r["case"]=="monitored_ABC_common_noise" and r["stress"]=="MCAR20"],
        "UNMONITORED_ABC_MCAR20":[r for r in aggregate
            if r["case"]=="UNMONITORED_ABC_noise" and r["stress"]=="MCAR20"],
    }),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--out-dir",type=Path,required=True)
    options=parser.parse_args()
    run(options.out_dir,smoke=options.smoke)
