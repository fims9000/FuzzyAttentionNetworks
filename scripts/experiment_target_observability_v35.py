"""V35 executed mathematical source-identifiability experiment on REAL Temporal ConceptFAN.

Published classical nullspace/GLS mathematics, NOT a newly discovered theorem.
Sensor topology is supplied by a separate independently calibrated device
phantom with a known external two-source reference. Without the reference
this source identity is NOT automatically identifiable.

K2 network sees only measured first 4 sensor channels, trained on observed
proxy concept targets. Frozen original risk readout. Source gamma fitted
TRAIN labels, source acceptance screened using two independent VALID cohorts.
TEST never tunes source rank, SE threshold, residual check or gamma.

Synthetic mixed-source data not actual clinical data; recorded device
reference and outcome source coordinate are privileged task DESIGN but are
never patient oracle labels fed to source discovery. A truly task-automatic
source mapping WITHOUT calibrated excitation is not demonstrated here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conceptfan_discovery.target_observability_v35 import (
    PHI,SUPPORT,make_independent_phantom,fit_external_reference,
    estimate_calibrated_source,measured_target_membership,
    linear_logit_sensitivity_radius,nullspace_counterexample,METHODS,
)
from conceptfan_discovery.mixed_source_benchmark_v35 import (
    mixed_reference_world,faulted_source_readouts,observation_mask,
    MASK_STRESS,FAULT_CASES,TRAIN_N,VALID_N,TEST_N,DATA_SEED,
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
from conceptfan_discovery.phase_pulse_benchmark_v31 import (
    temporal_sensor_features,
)
from conceptfan_discovery.adaptive_higher_order_concepts_v34 import (
    FuzzySourceDefinition,
)
from conceptfan_discovery.native_adaptive_type3_transformer_v34 import (
    NativeAuditedFuzzyConceptFANV34,render_concept_explanation_ru,
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

SEEDS=(11,23,37)
SE_LIMIT=.15
CALIB_SEED=20263500
TRAIN_MASK_SEED=20263550
VAL1_MASK_SEED=20263551
VAL2_MASK_SEED=20263552
TEST_KNOWN_MASK_SEED=20263553
TEST_SOURCE_BASE=20263560
METHOD_NAMES=(
    "frozen_K2","naive_WLS_coverage",
    "unsafe_mask_fraction","calibrated_full_rank",
    "calibrated_target_rowspace","calibrated_target_residual",
    "oracle_TRUE_a1_PRIVILEGED","raw_temporal_TRAIN1200",
    "raw_temporal_ALL2000",
)
TRAINED_SOURCE_METHODS=METHOD_NAMES[1:6]
NATIVE_METHODS=METHOD_NAMES[:7]
BOOTSTRAP_CASES=(
    ("healthy","only_target_6","calibrated_target_rowspace","calibrated_full_rank"),
    ("healthy","only_mixture_4","calibrated_target_rowspace","unsafe_mask_fraction"),
    ("healthy","MCAR20","calibrated_target_rowspace","calibrated_full_rank"),
    ("unseen_flip_sensor6","MCAR20","calibrated_target_residual","calibrated_target_rowspace"),
    ("unseen_common_flip_456","MCAR20","calibrated_target_residual","calibrated_target_rowspace"),
)


def _raw(x,m,y):
    pipe=make_pipeline(
        StandardScaler(),LogisticRegression(C=1.,max_iter=1200))
    pipe.fit(temporal_sensor_features(x,m),y)
    return pipe


def _naive_pooled(x,m):
    obs=m[:,:,4:7]
    measurements=np.where(obs,x[:,:,4:7],0.)
    weights=PHI[None,:,None]
    counts=obs.sum(axis=(1,2))
    amp=(measurements*weights).sum(axis=(1,2))/np.maximum(1,counts)
    return amp.astype(float),counts/48.


def _train_definition(amp,used):
    values=np.asarray(amp,dtype=float)[np.asarray(used,dtype=bool)]
    if len(values)<100:
        raise ValueError("No TRAIN-only source amplitudes support a definition")
    center=float(np.median(values))
    width=float(max(.15,(np.quantile(values,.75)-np.quantile(values,.25))/1.349))
    return center,width


def _features_for(
    method,x,m,calibration,
    *,
    tau,width,residual_threshold,
):
    if method=="naive_WLS_coverage":
        amp,support=_naive_pooled(x,m)
        used=support>=1/3
        q=np.where(used,expit((amp-tau)/width),.5)
        return q,used.astype(float),{
            "estimated_target":amp,
            "target_se":np.full(len(x),float("inf")),
            "rank":np.zeros(len(x),dtype=int),
            "target_in_rowspace":np.zeros(len(x),dtype=bool),
            "normalized_residual":np.full(len(x),float("inf")),
            "measured_fraction":support,
            "nullspace_target_fraction":np.ones(len(x)),
            "concept_identifiable":False,
            "uses_externally_calibrated_H":False,
        }
    result=estimate_calibrated_source(
        x,m,calibration,se_limit=SE_LIMIT,
        residual_limit=residual_threshold)
    q,s=measured_target_membership(
        result,method,train_tau=tau,train_width=width)
    return q,s,result


def _summary(rows,case,stress,method,n_fits):
    subset=[r for r in rows
            if r["case"]==case and r["stress"]==stress
            and r["method"]==method]
    count=n_fits if method in NATIVE_METHODS else 1
    if len(subset)!=count:
        raise AssertionError(f"{case}/{stress}/{method} results {len(subset)}, expected {count}")
    return {
        "case":case,"stress":stress,"method":method,
        "n_native_k2_fits":count if method in NATIVE_METHODS else 0,
        "AUPRC":float(np.mean([r["metric"]["auprc"] for r in subset])),
        "AUROC":float(np.mean([r["metric"]["auroc"] for r in subset])),
        "NLL":float(np.mean([r["metric"]["nll"] for r in subset])),
        "Brier":float(np.mean([r["metric"]["brier"] for r in subset])),
        "source_use":float(np.mean([r["source_use"] for r in subset])),
        "target_latent_error_on_eligible":(
            float(np.mean([r["target_latent_error_on_eligible"]
                           for r in subset
                           if r["target_latent_error_on_eligible"] is not None]))
            if any(r["target_latent_error_on_eligible"] is not None
                   for r in subset) else None),
        "approx_gaussian_95pct_latent_coverage_on_eligible":(
            float(np.mean([r["approx_gaussian_95pct_latent_coverage_on_eligible"]
                           for r in subset if
                           r["approx_gaussian_95pct_latent_coverage_on_eligible"]
                           is not None]))
            if any(r["approx_gaussian_95pct_latent_coverage_on_eligible"] is not None
                   for r in subset) else None),
    }


def _evaluate_source_prediction(
    base,method,metrics_y,q,s,record,gamma,global_approved,
    true_a1,
    *,
    tau,width,
):
    logit,used=admitted_test_logit(
        base,q,s,gamma=gamma,approved=global_approved,
        minimum_fraction=.50)
    np.testing.assert_array_equal(logit[~used],base[~used])
    result={
        "metric":metrics(metrics_y,expit(logit)),
        "source_use":float(used.mean()),
        "globally_admitted_from_BOTH_VALID":bool(global_approved),
        "target_latent_error_on_eligible":None,
        "approx_gaussian_95pct_latent_coverage_on_eligible":None,
        "estimated_calibrated_target_rank_mean":float(
            np.asarray(record["rank"],dtype=float).mean()),
        "target_in_rowspace_fraction":float(
            np.asarray(record["target_in_rowspace"]).mean()),
    }
    if method in METHODS and used.any():
        estimates=np.asarray(record["estimated_target"])[used]
        ses=np.asarray(record["target_se"])[used]
        if np.isfinite(ses).all():
            result["target_latent_error_on_eligible"]=float(
                np.mean(np.abs(estimates-true_a1[used])))
            result["approx_gaussian_95pct_latent_coverage_on_eligible"]=float(
                np.mean(np.abs(estimates-true_a1[used])<=1.96*ses))
            radius=linear_logit_sensitivity_radius(
                gamma,record["target_se"],train_width=width,eligible=used)
            result["mean_algebraic_logit_radius_on_used"]=float(
                radius[used].mean())
    return logit,used,result


def _source_passport(
    model_output,raw,record,idx, *,
    calibration,method,mode,source_gamma,
):
    # V35 source coordinates are tied to EXTERNAL DEVICE PHANTOM
    # calibration, not automatically identified TRAIN physiology. V34's
    # generic membership metadata passes the boolean eligibility as 1.0;
    # never misrepresent it as 100% actually measured source coverage.
    friendly=render_concept_explanation_ru(
        model_output,idx,observed_source_template=None)
    measured_fraction=float(record["measured_fraction"][idx])
    measured_rows=int(record["n_rows"][idx])
    target_identifiable=bool(record["target_in_rowspace"][idx])
    friendly["name"]=(
        "Калиброванный временной компонент измерений: датчики 4, 5 и 6")
    friendly["what_was_observed"]=(
        "Внешний измерительный эталон определяет форму отклика: "
        "положительная ранняя и отрицательная поздняя фазы. "
        "Это не доказательство физиологического смысла.")
    friendly["measurement_evidence"]=(
        f"Наблюдались {measured_rows} из 48 потенциальных "
        f"измерений ({measured_fraction*100:.1f}%). "
        f"Нужный компонент {'определим' if target_identifiable else 'не определим'} "
        "по имеющимся измерениям при принятой калибровке.")
    friendly["what_was_inferred"]=(
        "Амплитуда рассчитана по наблюдаемым отсчётам с помощью "
        "внешне калиброванной линейной модели. Пропущенные значения "
        "не объявляются реальными измерениями. Диффузия не применялась."
        if measured_rows else
        "Нет измерений источника: амплитуда не восстановлена.")
    friendly["technical_audit"]["v34_source_support_field_was_boolean_gate"]=True
    friendly["technical_audit"]["source_observed_information_fraction"]=None
    friendly["technical_audit"]["source_measured_fraction"]=measured_fraction
    # Reference calibration of H (V35) and uncertainty in fuzzy
    # definition boundaries (V34 Type-3) are INDEPENDENT evidence classes.
    # V35 has verified H but deliberately has NO Type-3 boundary anchor.
    friendly["technical_audit"]["device_calibration_is_external"]=True
    friendly["technical_audit"]["independent_device_calibration_available"]=True
    friendly["technical_audit"]["type3_boundary_calibration_available"]=False
    friendly["technical_audit"]["anchor_semantics"]=(
        "Verified synthetic device mixing H; no independently verified "
        "fuzzy membership boundary or clinical physical identity")
    return {
        "version":"V35_TARGET_ROWSPACE",
        "method":method,"stress":mode,
        "source_concept":"Амплитуда заданного калиброванного временного источника 1",
        "source_group":[4,5,6],
        "calibration_ref":"independent synthetic known-reference device excitation",
        "reference_samples":calibration.reference_count,
        "calibration_sha256":calibration.calibration_sha256,
        "observed_measurement_fraction":float(record["measured_fraction"][idx]),
        "observed_measurement_rows":int(record["n_rows"][idx]),
        "measurement_operator_rank":int(record["rank"][idx]),
        "target_in_measured_rowspace":bool(record["target_in_rowspace"][idx]),
        "target_nullspace_component":float(record["nullspace_target_fraction"][idx]),
        "target_se_plugin":(
            float(record["target_se"][idx])
            if np.isfinite(record["target_se"][idx]) else None),
        "standardized_device_model_residual":(
            float(record["normalized_residual"][idx])
            if np.isfinite(record["normalized_residual"][idx]) else None),
        "null_uncertainty_value_means_NOT_IDENTIFIABLE":(
            not np.isfinite(record["target_se"][idx])),
        "source_signed_gamma":float(source_gamma),
        "human_facing_explanation":friendly,
        "no_certified_clinical_semantics":True,
        "identifiability_conditional_on_trusted_H":True,
        "type3_or_flow_does_not_prove_identity":True,
    }


def run(out_dir:Path,*,smoke:bool=False)->dict:
    if out_dir.exists():
        raise FileExistsError("V35 evidence immutable; choose a new directory")
    torch.set_num_threads(2)
    seeds=(11,) if smoke else SEEDS
    epochs=2 if smoke else 10
    bootstrap=199 if smoke else 399
    ntr,nval,nte=(300,300,240) if smoke else (
        TRAIN_N,VALID_N,TEST_N)
    data=mixed_reference_world(
        seed=DATA_SEED,n_train=ntr,n_valid=nval,n_test=nte)
    x=data["x"]
    tr,v1,v2,te=(data["train"],data["valid1"],
                 data["valid2"],data["test"])
    phantom_in,phantom_out=make_independent_phantom(
        seed=CALIB_SEED,n=900)
    calibration=fit_external_reference(
        phantom_in,phantom_out,
        physically_verified_routing_support=SUPPORT,
        independent_reference=True)
    mtr=observation_mask(x[tr],"MCAR20",seed=TRAIN_MASK_SEED,
                         common_known_seed=TRAIN_MASK_SEED)
    mv1=observation_mask(x[v1],"MCAR20",seed=VAL1_MASK_SEED,
                         common_known_seed=VAL1_MASK_SEED)
    mv2=observation_mask(x[v2],"MCAR20",seed=VAL2_MASK_SEED,
                         common_known_seed=VAL2_MASK_SEED)
    # Unsupervised residual warning calibrates only on TRAIN measured
    # observed cells. It is not an OUTCOME-aware source gate.
    train_inverse=estimate_calibrated_source(
        x[tr],mtr,calibration,se_limit=SE_LIMIT)
    good=train_inverse["calibrated_target_rowspace"]
    residual_threshold=float(np.quantile(
        train_inverse["normalized_residual"][good],.99))
    if not .4<residual_threshold<3:
        raise AssertionError("TRAIN calibration residual distribution implausible")
    train_inverse=estimate_calibrated_source(
        x[tr],mtr,calibration,se_limit=SE_LIMIT,
        residual_limit=residual_threshold)
    tau,width=_train_definition(
        train_inverse["estimated_target"],
        train_inverse["calibrated_target_rowspace"])
    naive_train,naive_support=_naive_pooled(x[tr],mtr)
    naive_tau,naive_width=_train_definition(
        naive_train,naive_support>=1/3)
    definitions={
        method:(naive_tau,naive_width)
        if method=="naive_WLS_coverage" else (tau,width)
        for method in TRAINED_SOURCE_METHODS
    }

    # Cache VALID and TEST observation-only inverse computations across
    # all native K2 trained seeds. TEST true_a1 is never accessed here.
    inverse_cache={
        ("train","none","MCAR20"):train_inverse,
    }
    methods_train={
        method:_features_for(
            method,x[tr],mtr,calibration,
            tau=definitions[method][0],
            width=definitions[method][1],
            residual_threshold=residual_threshold)
        for method in TRAINED_SOURCE_METHODS
    }
    # Scenario masks, sensor flips and labels are kept separate. ALL
    # TEST faults use identical TEST600 subject IDs and outcome draws.
    case_data={}
    for case_index,(case,law,valid_shift,test_shift) in enumerate(FAULT_CASES):
        vv2=faulted_source_readouts(x[v2],valid_shift)
        tt=faulted_source_readouts(x[te],test_shift)
        y=data["outcomes"][law]
        case_data[case]={
            "y":y,
            "xv2":vv2,
            "xtest":tt,
            "mv2":mv2.copy(),
            "v1":{},
            "v2":{},
            "test":{},
        }
        for method in TRAINED_SOURCE_METHODS:
            ta,wd=definitions[method]
            case_data[case]["v1"][method]=_features_for(
                method,x[v1],mv1,calibration,
                tau=ta,width=wd,
                residual_threshold=residual_threshold)
            case_data[case]["v2"][method]=_features_for(
                method,vv2,mv2,calibration,
                tau=ta,width=wd,
                residual_threshold=residual_threshold)
        for stress_index,stress in enumerate(MASK_STRESS):
            mm=observation_mask(
                tt,stress,seed=TEST_SOURCE_BASE+stress_index,
                common_known_seed=TEST_KNOWN_MASK_SEED)
            case_data[case]["test"][stress]={"mask":mm,"features":{}}
            for method in TRAINED_SOURCE_METHODS:
                ta,wd=definitions[method]
                case_data[case]["test"][stress]["features"][method]=(
                    _features_for(
                        method,tt,mm,calibration,
                        tau=ta,width=wd,
                        residual_threshold=residual_threshold))
    rows,gates,training,passports=[],[],[],[]
    predictions={}
    test_labels={}
    saved_native=[]
    for case,law,v_shift,t_shift in FAULT_CASES:
        y=case_data[case]["y"]
        test_labels[case]=y[te].copy()
        tr_raw=_raw(x[tr],mtr,y[tr])
        all_raw=_raw(
            np.concatenate([x[tr],x[v1],case_data[case]["xv2"]]),
            np.concatenate([mtr,mv1,mv2]),
            np.concatenate([y[tr],y[v1],y[v2]]))
        for stress in MASK_STRESS:
            xx=case_data[case]["xtest"]
            mm=case_data[case]["test"][stress]["mask"]
            xx_features=temporal_sensor_features(xx,mm)
            for method,rawmodel in (
                ("raw_temporal_TRAIN1200",tr_raw),
                ("raw_temporal_ALL2000",all_raw)):
                p=rawmodel.predict_proba(xx_features)[:,1]
                rows.append({
                    "case":case,"stress":stress,"method":method,
                    "NN_seed":None,"metric":metrics(y[te],p),
                    "source_use":0.,"target_latent_error_on_eligible":None,
                    "approx_gaussian_95pct_latent_coverage_on_eligible":None,
                })
                predictions[(case,stress,method,None)]=p
    # Freeze and test actual Transformer K2 known concepts once per seed.
    for seed in seeds:
        core=make_firewalled_k2(seed)
        stage=train_masked_measured_known_concepts(
            core,x[tr,:,:4],mtr[:,:,:4],seed=seed,epochs=epochs)
        before_sha=sha_state(core)
        saved_native.append({
            "nn_seed":seed,
            "state_dict":{k:v.detach().cpu().clone()
                          for k,v in core.state_dict().items()},
        })
        training.append({
            "NN_seed":seed,"native_K2_SHA256":before_sha,
            "n_known_concepts":2,**stage,
        })
        qt=predict_native(core,x[tr,:,:4],
                          mtr[:,:,:4])["concept_summary"]
        qv1=predict_native(core,x[v1,:,:4],
                          mv1[:,:,:4])["concept_summary"]
        qv2=predict_native(core,x[v2,:,:4],
                          mv2[:,:,:4])["concept_summary"]
        mt=case_data["healthy"]["test"]["MCAR20"]["mask"]
        qtest=predict_native(
            core,x[te,:,:4],mt[:,:,:4])["concept_summary"]
        for case_index,(case,law,valid_shift,test_shift) in enumerate(FAULT_CASES):
            y=case_data[case]["y"]
            head=refit_linear_concept_head(qt,y[tr])
            old_train=logit_from_head(head,qt)
            old_v1=logit_from_head(head,qv1)
            old_v2=logit_from_head(head,qv2)
            old_test=logit_from_head(head,qtest)
            approved={}
            for method in TRAINED_SOURCE_METHODS:
                trainq,trains,_=methods_train[method]
                val1q,val1s,_=case_data[case]["v1"][method]
                val2q,val2s,_=case_data[case]["v2"][method]
                gamma=train_source_coefficient(
                    old_train,trainq,y[tr])
                gate=evaluate_two_independent_environments(
                    method,gamma,
                    ("VALID1",y[v1],old_v1,val1q,val1s),
                    ("VALID2",y[v2],old_v2,val2q,val2s),
                    train_label_count=len(tr),min_mean_gain=.003,
                    min_support_fraction=.75,
                    min_subject_support=.50,
                    bootstrap_reps=bootstrap,
                    seed=20263580+case_index*30+seed)
                approved[method]=(gamma,gate)
                gates.append({
                    "case":case,"NN_seed":seed,
                    "method":method,"gamma_train_only":gamma,
                    "gate":gate.as_dict(),
                })
            oracle_qtrain=expit(
                (data["true_a1_EVALUATION_ONLY"][tr]-tau)/width)
            oracle_gamma=train_source_coefficient(
                old_train,oracle_qtrain,y[tr])
            for stress_index,stress in enumerate(MASK_STRESS):
                observed=case_data[case]["test"][stress]
                base=old_test.copy()
                np.testing.assert_array_equal(
                    base,logit_from_head(head,qtest))
                prob_base=expit(base)
                predictions[(case,stress,"frozen_K2",seed)]=prob_base
                rows.append({
                    "case":case,"stress":stress,
                    "method":"frozen_K2","NN_seed":seed,
                    "metric":metrics(y[te],prob_base),
                    "source_use":0.,
                    "target_latent_error_on_eligible":None,
                    "approx_gaussian_95pct_latent_coverage_on_eligible":None,
                })
                for method in TRAINED_SOURCE_METHODS:
                    q,s,rec=observed["features"][method]
                    gamma,gate=approved[method]
                    z,used,details=_evaluate_source_prediction(
                        base,method,y[te],q,s,rec,gamma,
                        gate.approved,
                        data["true_a1_EVALUATION_ONLY"][te],
                        tau=definitions[method][0],
                        width=definitions[method][1])
                    if stress=="no_source_all16":
                        np.testing.assert_array_equal(z,base)
                        assert not used.any()
                    predictions[(case,stress,method,seed)]=expit(z)
                    rows.append({
                        "case":case,"stress":stress,
                        "method":method,"NN_seed":seed,
                        **details,
                    })
                    if (case=="healthy" and seed==seeds[0]
                        and stress in (
                            "MCAR20","only_mixture_4",
                            "only_target_6","no_source_all16")
                        and method=="calibrated_target_rowspace"):
                        ta,wd=definitions[method]
                        concept_def=FuzzySourceDefinition(
                            source_id="external_reference_component_1",
                            sensor_indices=(4,5,6),
                            threshold_train=ta,width_train=wd,
                            train_missingness_spread=0.,
                            boundary_spread_anchor=0.,
                            certified_calibration_environments=0,
                            external_anchor_verified=False,
                        )
                        model=NativeAuditedFuzzyConceptFANV34(
                            core,np.asarray(head.weights),head.bias,
                            concept_def,fuzzy_order="type1")
                        with torch.no_grad():
                            model.source_slot.gamma.fill_(gamma)
                        # Ineligible pseudo-GLS amplitudes are not evidence.
                        amp=np.asarray(rec["estimated_target"][:24],
                                       dtype=np.float32)
                        se=np.asarray(rec["target_se"][:24],
                                      dtype=np.float32)
                        safe_se=np.where(np.isfinite(se),se,0.)
                        native=model(
                            torch.from_numpy(
                                case_data[case]["xtest"][:24,:,:4]),
                            torch.from_numpy(
                                observed["mask"][:24,:,:4]),
                            extra_source_amplitude=torch.from_numpy(amp),
                            extra_source_uncertainty=torch.from_numpy(safe_se),
                            extra_source_observed_information=torch.from_numpy(
                                s[:24].astype(np.float32)),
                            extra_source_gate_approved=gate.approved,
                        )
                        if max(abs(native["logit"].detach().numpy()-z[:24]))>5e-5:
                            raise AssertionError(
                                "V35 concept route not equal to native torch forward")
                        passport=_source_passport(
                            native,rec,rec,0,calibration=calibration,
                            method=method,mode=stress,source_gamma=gamma)
                        passports.append(passport)
                oracle_qtest=expit(
                    (data["true_a1_EVALUATION_ONLY"][te]-tau)/width)
                z_oracle=base+oracle_gamma*(oracle_qtest-.5)
                p_oracle=expit(z_oracle)
                predictions[(case,stress,"oracle_TRUE_a1_PRIVILEGED",seed)]=p_oracle
                rows.append({
                    "case":case,"stress":stress,
                    "method":"oracle_TRUE_a1_PRIVILEGED","NN_seed":seed,
                    "metric":metrics(y[te],p_oracle),
                    "source_use":1.,
                    "target_latent_error_on_eligible":0.,
                    "approx_gaussian_95pct_latent_coverage_on_eligible":None,
                    "oracle_source_used_for_any_train_or_VALID_gate":False,
                    "oracle_source_used_only_for_privileged_ceiling":True,
                })
            if before_sha!=sha_state(core):
                raise AssertionError("V35 altered frozen known Transformer")
        print(json.dumps({
            "event":"V35_NATIVE_K2_FIT_DONE",
            "NN_seed":seed,
            "original_transformer_sha256":before_sha,
            "native_model_trained_on_observed_proxy_not_Y":True,
        }),flush=True)
    summary=[
        _summary(rows,case,stress,method,len(seeds))
        for case,_,_,_ in FAULT_CASES
        for stress in MASK_STRESS
        for method in METHOD_NAMES
    ]
    comparisons=[]
    for j,(case,stress,a,b) in enumerate(BOOTSTRAP_CASES):
        yy=test_labels[case]
        aa=np.stack([
            predictions[(case,stress,a,seed)] for seed in seeds])
        bb=np.stack([
            predictions[(case,stress,b,seed)] for seed in seeds])
        comparison=paired_test_ap_difference(
            yy,aa,bb,seed=20263990+j,resamples=bootstrap)
        comparisons.append({
            "case":case,"stress":stress,
            "method_a":a,"method_b":b,
            **comparison,
        })
    # Counterfactual proof is separate from performance/label tuning.
    counterexample=nullspace_counterexample()
    if not counterexample["both_worlds_exactly_observationally_indistinguishable"]:
        raise AssertionError("Nonidentifiability falsifier broken")
    result={
        "status":"SMOKE_ONLY" if smoke else "V35_TARGET_SPECIFIC_IDENTIFIABILITY_EXECUTED",
        "source_branch":"experiment/conceptfan-v35-target-observability-20261106",
        "registered_protocol":"docs/medical/CONCEPTFAN_V35_FROZEN_TARGET_OBSERVABILITY_PROTOCOL_20261106.md",
        "data_seed":DATA_SEED,
        "n_train":ntr,"n_valid1":len(v1),"n_valid2":len(v2),
        "n_test":len(te),
        "n_K2_native_transformer_fits":len(seeds),
        "n_results":len(rows),"n_summary":len(summary),
        "n_VALID_candidate_gates":len(gates),
        "n_paired_TEST_bootstrap_comparisons":len(comparisons),
        "external_hardware_reference_truth_ASSUMED":True,
        "external_source_routing_support_VERIFIED_BY_SIMULATOR":True,
        "calibrated_true_H_is_not_fed_to_inverse":True,
        "concept_identity_without_calibration_proved":False,
        "physical_or_clinical_semantics_proved":False,
        "mathematical_rowspace_test_is_classical":True,
        "target_rowspace_weaker_than_full_rank":True,
        "new_source_in_real_native_transformer_torch_forward":True,
        "trained_native_K2_unchanged_at_source_admission":True,
        "TEST_used_to_tune_source_policy":False,
        "independently_calibrated_H":calibration.response.tolist(),
        "independently_calibrated_noise_SD":calibration.noise_sd.tolist(),
        "independently_verified_wiring_mask":SUPPORT.tolist(),
        "device_phantom_ref_size":calibration.reference_count,
        "device_H_sha256":calibration.calibration_sha256,
        "source_residual_threshold_TRAIN_99percentile":residual_threshold,
        "TRAIN_concept_membership_threshold":tau,
        "TRAIN_concept_membership_width":width,
        "TRAIN_naive_membership_threshold":naive_tau,
        "TRAIN_naive_membership_width":naive_width,
        "counterexample":counterexample,
        "training":training,
        "gates":gates,
        "rows":rows,
        "summary":summary,
        "confidence_intervals":comparisons,
        "human_explanation_passports":passports,
    }
    out_dir.mkdir(parents=True,exist_ok=False)
    for path,content in (
        ("STATUS.json",result),
        ("FULL_ROWS.json",rows),
        ("SUMMARY.json",summary),
        ("GATES.json",gates),
        ("PAIRED_TEST_CI.json",comparisons),
        ("CONCEPT_EXPLANATIONS.json",passports),
    ):
        (out_dir/path).write_text(
            json.dumps(content,indent=2,ensure_ascii=False,allow_nan=False))
    frozen={
        f"{case}_{stress}_{method}_seed{seed}":p
        for (case,stress,method,seed),p in predictions.items()
    }
    for case,y in test_labels.items():
        frozen[f"{case}_TEST_Y"]=y
    np.savez_compressed(out_dir/"FROZEN_TEST_PATIENT_PREDICTIONS.npz",**frozen)
    checkpoint_dir=out_dir/"CHECKPOINTS"
    checkpoint_dir.mkdir()
    for item in saved_native:
        torch.save({
            "seed":item["nn_seed"],
            "model":"REAL native K2 Temporal ConceptFAN",
            "state_dict":item["state_dict"]},
            checkpoint_dir/f"native_K2_seed{item['nn_seed']}.pt")
    print(json.dumps({
        "event":"V35_FULL_RESEARCH_DONE",
        "status":result["status"],
        "n_models":len(seeds),
        "n_rows":len(rows),"n_summary":len(summary),
        "n_gates":len(gates),"n_CI":len(comparisons),
        "selected_results":[s for s in summary
            if s["case"] in (
                "healthy","unseen_flip_sensor6",
                "unseen_common_flip_456","complete_dictionary")
            and s["stress"] in (
                "MCAR20","only_mixture_4","only_target_6",
                "mixture_plus_nuisance_4_5","no_source_all16")
            and s["method"] in (
                "frozen_K2","unsafe_mask_fraction",
                "calibrated_full_rank","calibrated_target_rowspace",
                "calibrated_target_residual","raw_temporal_ALL2000")],
        "CI":comparisons,
    }),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args()
    run(args.out_dir,smoke=args.smoke)
