"""V39 preregistered synthetic paired ON/OFF source isolation benchmark.

Six REAL native K2 Temporal ConceptFAN models over two independently drawn
devices, 1000 TRAIN + 400+400 VALID + 600 TEST each. All source discovery
uses masked measurements ONLY: no true mixing H, latent S or patient outcome
in source extraction. Signed additional concept gamma fits TRAIN outcomes
and is gated on independent VALID1 and VALID2, never TEST.

The ON/OFF physical blocking switch is EXTERNAL SUPERVISION assumed by the
SIMULATOR. Healthy artifact cancellation is an algebraic property of these
simulator constraints, NOT a guarantee for arbitrary instruments.
"""
from __future__ import annotations

import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit,logit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from conceptfan_discovery.paired_isolation_world_v39 import (
    DEVICE_SEEDS,STRESSES,CASES,
    generate_paired_isolation_world_v39,apply_paired_fault_v39,
    masks_v39,observed_paired_difference_v39,
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
from scripts.experiment_nonlinear_anchor_repair_v30 import make_firewalled_k2
from scripts.experiment_automatic_measured_concepts import (
    train_masked_measured_known_concepts,logit_from_head,
)
from scripts.experiment_frozen_concept_head_refit_v26 import (
    refit_linear_concept_head,
)
from scripts.experiment_trained_conceptfan_omission_v25 import (
    predict_native,sha_state,metrics,
)
from scripts.experiment_threeview_centered_v37 import (
    _raw_feature,
)

NATIVE_SEEDS=(11,23,37)
SOURCE_METHODS=("ON_AB","ON_ABC","DIFFERENCE_AB","DIFFERENCE_ABC")
NATIVE_METHODS=("K2",)+SOURCE_METHODS
RAW_METHODS=("RAW_ON_QUADRATIC_ALL1800","RAW_ON_OFF_QUADRATIC_ALL1800")
METHODS=NATIVE_METHODS+RAW_METHODS
PAIRED_TESTS=(
    ("healthy","MCAR20","DIFFERENCE_ABC","K2"),
    ("healthy","MCAR20","DIFFERENCE_ABC","ON_ABC"),
    ("healthy","MCAR20","DIFFERENCE_ABC","RAW_ON_OFF_QUADRATIC_ALL1800"),
    ("TEST_aligned_artifact_4x","MCAR20","DIFFERENCE_ABC","ON_ABC"),
    ("TEST_aligned_artifact_8x","MCAR20","DIFFERENCE_ABC","ON_ABC"),
    ("TEST_OFF_artifact_drift","MCAR20","DIFFERENCE_ABC","ON_ABC"),
    ("TEST_source_isolation_leak_50pct","MCAR20","DIFFERENCE_ABC","ON_ABC"),
    ("TEST_reference_missing","MCAR20","DIFFERENCE_ABC","K2"),
    ("healthy","synchronized_block8","DIFFERENCE_ABC","ON_ABC"),
    ("K2_complete_dictionary","MCAR20","DIFFERENCE_ABC","K2"),
)


def _mask(on,off,stress,*,device_seed,mask_seed):
    return masks_v39(
        on,off,stress,
        on_seed=mask_seed+device_seed%101,
        off_seed=mask_seed+991+device_seed%101,
    )


def _source_features(
    on,m_on,difference,m_difference,
    *,on_fit,difference_fit,
):
    return {
        "ON_AB":pair_feature_v37(on_fit.pairs["AB"],on,m_on),
        "ON_ABC":threeview_feature_v37(on_fit,on,m_on),
        "DIFFERENCE_AB":pair_feature_v37(
            difference_fit.pairs["AB"],difference,m_difference),
        "DIFFERENCE_ABC":threeview_feature_v37(
            difference_fit,difference,m_difference),
    }


def _raw_features(on,m_on,diff,m_diff,*,use_reference):
    a=_raw_feature(on,m_on,quadratic=True)
    if not use_reference:
        return a
    b=_raw_feature(diff,m_diff,quadratic=True)
    return np.column_stack((a,b))


def _fit_raw(on,mo,d,md,y,*,use_reference):
    model=make_pipeline(StandardScaler(),
                        LogisticRegression(C=1.,max_iter=1000))
    model.fit(_raw_features(on,mo,d,md,
                           use_reference=use_reference),y)
    return model


def _summary(rows,device,case,stress,method,n_models):
    subset=[v for v in rows if (
        v["device_seed"],v["case"],v["stress"],v["method"]
    )==(device,case,stress,method)]
    n=n_models if method in NATIVE_METHODS else 1
    if len(subset)!=n:
        raise AssertionError("Missing TEST predictions from frozen method")
    return {
        "device_seed":device,"case":case,"stress":stress,
        "method":method,
        "AUPRC":float(np.mean([r["metrics"]["auprc"] for r in subset])),
        "AUROC":float(np.mean([r["metrics"]["auroc"] for r in subset])),
        "NLL":float(np.mean([r["metrics"]["nll"] for r in subset])),
        "Brier":float(np.mean([r["metrics"]["brier"] for r in subset])),
        "source_use":float(np.mean([r["source_use"] for r in subset])),
    }


def _human_explanation(source,used,pbase,pnew,*,case,stress,gate):
    observed=np.asarray(source["pair_timepoints"])[0]
    if not gate:
        reason="Отказ: источник не улучшил два независимых VALID набора."
        code="TWO_VALID_SOURCE_REJECTED"
    elif not bool(source["all_three_pairs_measured"][0]):
        reason="Отказ: источниковой и изолированной опорной траектории недостаточно для сравнения."
        code="ON_OFF_REFERENCE_MISSING"
    elif not used:
        reason="Отказ: две формы измерения или три межканальные оценки не согласованы."
        code="SOURCE_MEASUREMENT_DISAGREEMENT"
    else:
        reason=("Использовано сравнение текущих и изолированных измерений. "
                "Из этой разности вычтен общий приборный сигнал при ПРЕДПОЛОЖЕНИИ "
                "его неизменности между ON и OFF. "
                "Физический смысл и причинность исхода не подтверждены.")
        code="MEASURED_PAIRED_SOURCE_ACCEPTED"
    if not used and pnew!=pbase:
        raise AssertionError("Source refusal mutated original neural K2 score")
    return {
        "concept_name":"Энергия временных колебаний источника, выделенная разностью ON−OFF",
        "decision_code":code,
        "decision_explanation":reason,
        "source_used":bool(used),
        "sensor_pairs":{"AB":[4,5,6,7],"AC":[4,5,8,9],"BC":[6,7,8,9]},
        "actual_joint_ON_OFF_timepoints_by_pair":{
            "AB":int(observed[0]),"AC":int(observed[1]),
            "BC":int(observed[2]),"total_timepoints":16,
        },
        "predicted_probability_original_model":float(pbase),
        "predicted_probability_with_source":float(pnew),
        "change_in_percentage_points":100.*float(pnew-pbase),
        "physical_switch_metadata_is_external_supervision":True,
        "source_vs_instrument_separation_requires_matched_artifact":True,
        "unmatched_instrument_drift_invalidates_claim":True,
        "not_an_estimated_clinical_causal_effect":True,
        "no_missing_off_values_generated":True,
        "test_case":case,"missingness":stress,
    }


def run(output:Path,*,smoke=False):
    if output.exists():
        raise FileExistsError("Never overwrite immutable V39 source/TEST evidence")
    torch.set_num_threads(2)
    n_train,n_valid,n_test=(360,300,240) if smoke else (1000,800,600)
    seeds=(11,) if smoke else NATIVE_SEEDS
    devices=DEVICE_SEEDS[:1] if smoke else DEVICE_SEEDS
    boot=199 if smoke else 399
    rows=[];valid_gates=[];source_fits=[];training=[];human=[]
    forecasts={};labels={};checkpoints=[]
    def save_prediction(device,case,stress,method,seed,p):
        forecasts[(device,case,stress,method,seed)]=np.asarray(
            p,dtype=np.float64).copy()

    for device in devices:
        world=generate_paired_isolation_world_v39(
            device_seed=device,n_train=n_train,
            n_valid=n_valid,n_test=n_test)
        tr,v1,v2,te=(world[k] for k in ("train","valid1","valid2","test"))
        on,off=world["on"],world["off"]
        mtr_on,mtr_off=_mask(on[tr],off[tr],"MCAR20",
                             device_seed=device,mask_seed=20263901)
        mv1_on,mv1_off=_mask(on[v1],off[v1],"MCAR20",
                             device_seed=device,mask_seed=20263902)
        mv2_on,mv2_off=_mask(on[v2],off[v2],"MCAR20",
                             device_seed=device,mask_seed=20263903)
        dt,mt=observed_paired_difference_v39(
            on[tr],off[tr],mtr_on,mtr_off)
        dv1,mdv1=observed_paired_difference_v39(
            on[v1],off[v1],mv1_on,mv1_off)
        on_fit=fit_centered_threeview_v37(
            on[tr],mtr_on,independent_view_noise_documented=True,
            seed=20263911+device%101,bootstrap=24)
        difference_fit=fit_centered_threeview_v37(
            dt,mt,independent_view_noise_documented=True,
            seed=20263921+device%101,bootstrap=24)
        src_train=_source_features(
            on[tr],mtr_on,dt,mt,
            on_fit=on_fit,difference_fit=difference_fit)
        src_v1=_source_features(
            on[v1],mv1_on,dv1,mdv1,
            on_fit=on_fit,difference_fit=difference_fit)
        source_fits.append({
            "device_seed":device,
            "observational_on_only":on_fit.audit(),
            "paired_source_subtraction":difference_fit.audit(),
            "no_true_H_or_S_in_source_fit":True,
            "external_source_isolation_switch_assumption":True,
            "matched_artifact_assumption":True,
            "source_physical_identity_certified":False,
        })
        case_data={}
        for ci,(case,law,fault_v2,fault_te) in enumerate(CASES):
            v2on,v2off=apply_paired_fault_v39(
                on[v2],off[v2],mode=fault_v2,
                seed=20263931+ci+device%101,
                hidden_H_SIMULATOR_ONLY=world["hidden_mixing_AUDIT_ONLY"],
                hidden_S_SIMULATOR_ONLY=world["hidden_S_AUDIT_ONLY"][v2])
            v2d,v2dm=observed_paired_difference_v39(
                v2on,v2off,mv2_on,mv2_off)
            source_v2=_source_features(
                v2on,mv2_on,v2d,v2dm,
                on_fit=on_fit,difference_fit=difference_fit)
            teston,testoff=apply_paired_fault_v39(
                on[te],off[te],mode=fault_te,
                seed=20263960+ci+device%101,
                hidden_H_SIMULATOR_ONLY=world["hidden_mixing_AUDIT_ONLY"],
                hidden_S_SIMULATOR_ONLY=world["hidden_S_AUDIT_ONLY"][te])
            by_mask={}
            for mi,stress in enumerate(STRESSES):
                m_on,m_off=_mask(
                    teston,testoff,stress,device_seed=device,
                    mask_seed=20264011+mi)
                if case=="TEST_reference_missing":
                    m_off[:,:,4:10]=False
                diff,available=observed_paired_difference_v39(
                    teston,testoff,m_on,m_off)
                by_mask[stress]={
                    "on":teston,"off":testoff,
                    "m_on":m_on,"m_off":m_off,
                    "difference":diff,"m_difference":available,
                    "source":_source_features(
                        teston,m_on,diff,available,
                        on_fit=on_fit,difference_fit=difference_fit),
                }
            case_data[case]={
                "label_law":law,
                "v2on":v2on,"v2off":v2off,
                "v2d":v2d,"v2dm":v2dm,
                "v2source":source_v2,
                "test":by_mask,
            }
            labels[(device,case)]=world["outcomes"][law][te]

        # Reference-only comparison: paired source invariant to aligned
        # common nuisance added *equally* to both modes, before y metrics.
        for case in ("TEST_aligned_artifact_4x","TEST_aligned_artifact_8x"):
            stable=case_data["healthy"]["test"]["MCAR20"]["source"]["DIFFERENCE_ABC"]
            altered=case_data[case]["test"]["MCAR20"]["source"]["DIFFERENCE_ABC"]
            if (not np.array_equal(stable["support"],altered["support"])
                or not np.allclose(stable["q"],altered["q"],rtol=0,atol=5e-6)):
                raise AssertionError(
                    "Instrument artifact did not cancel under paired ON−OFF measurements")
        # External source label is unavailable to estimation. Use only
        # AFTER model/source fit to compute reconstruction correlation.
        truth=world["true_source_fluctuation_power_AUDIT_ONLY"][te]
        source=case_data["healthy"]["test"]["MCAR20"]["source"]["DIFFERENCE_ABC"]
        support=source["support"]>0
        if support.sum()<30:
            correlation=None
        else:
            correlation=float(np.corrcoef(
                truth[support],source["q"][support])[0,1])
        source_fits[-1]["TEST_truth_corr_AUDIT_ONLY"]=correlation

        for seed in seeds:
            native=make_firewalled_k2(seed)
            learned=train_masked_measured_known_concepts(
                native,on[tr,:,:4],mtr_on[:,:,:4],
                seed=seed,epochs=2 if smoke else 10)
            model_sha=sha_state(native)
            training.append({
                "device_seed":device,"NN_seed":seed,
                "frozen_native_sha":model_sha,**learned})
            checkpoints.append({
                "device_seed":device,"NN_seed":seed,
                "frozen_native_SHA":model_sha,
                "state_dict":{k:v.detach().cpu().clone()
                              for k,v in native.state_dict().items()},
            })
            qt=predict_native(
                native,on[tr,:,:4],mtr_on[:,:,:4])["concept_summary"]
            q1=predict_native(
                native,on[v1,:,:4],mv1_on[:,:,:4])["concept_summary"]
            q2=predict_native(
                native,on[v2,:,:4],mv2_on[:,:,:4])["concept_summary"]
            known_mask=case_data["healthy"]["test"]["MCAR20"]["m_on"][:,:,:4]
            qe=predict_native(
                native,on[te,:,:4],known_mask)["concept_summary"]
            for ci,(case,law,fault_v2,fault_te) in enumerate(CASES):
                y=world["outcomes"][law]
                head=refit_linear_concept_head(qt,y[tr])
                zt=logit_from_head(head,qt)
                z1=logit_from_head(head,q1)
                z2=logit_from_head(head,q2)
                zbase=logit_from_head(head,qe)
                fitters={}
                for method in SOURCE_METHODS:
                    f=src_train[method]
                    gamma=train_source_coefficient(zt,f["q"],y[tr])
                    a=src_v1[method]
                    b=case_data[case]["v2source"][method]
                    gate=evaluate_two_independent_environments(
                        method,gamma,
                        ("VALID1",y[v1],z1,a["q"],a["support"]),
                        ("VALID2",y[v2],z2,b["q"],b["support"]),
                        train_label_count=len(tr),
                        min_mean_gain=.003,min_support_fraction=.25,
                        min_subject_support=.50,
                        bootstrap_reps=boot,
                        seed=20264091+device%101*100+ci*11+seed)
                    fitters[method]=(gamma,gate)
                    valid_gates.append({
                        "device_seed":device,"case":case,"NN_seed":seed,
                        "method":method,"TRAIN_gamma":gamma,
                        "gate":gate.as_dict(),
                    })
                for stress in STRESSES:
                    record=case_data[case]["test"][stress]
                    base=expit(zbase)
                    save_prediction(device,case,stress,"K2",seed,base)
                    rows.append({
                        "device_seed":device,"case":case,"stress":stress,
                        "method":"K2","NN_seed":seed,
                        "metrics":metrics(y[te],base),"source_use":0.,
                    })
                    for method in SOURCE_METHODS:
                        gamma,gate=fitters[method]
                        measured=record["source"][method]
                        z,used=admitted_test_logit(
                            zbase,measured["q"],measured["support"],
                            gamma=gamma,approved=gate.approved,
                            minimum_fraction=.50)
                        np.testing.assert_array_equal(z[~used],zbase[~used])
                        prob=expit(z)
                        save_prediction(
                            device,case,stress,method,seed,prob)
                        rows.append({
                            "device_seed":device,"case":case,"stress":stress,
                            "method":method,"NN_seed":seed,
                            "metrics":metrics(y[te],prob),
                            "source_use":float(used.mean()),
                            "source_approved_both_VALID":bool(gate.approved),
                        })
                        if (case=="healthy" and stress=="MCAR20"
                            and method=="DIFFERENCE_ABC"):
                            n=12
                            definition=FuzzySourceDefinition(
                                source_id="paired_actively_source_isolated_slow_fluctuation_energy",
                                sensor_indices=(4,5,6,7,8,9),
                                threshold_train=0.,width_train=1.,
                                train_missingness_spread=0.,
                                boundary_spread_anchor=0.,
                                certified_calibration_environments=0,
                                external_anchor_verified=False)
                            wrapper=NativeAuditedFuzzyConceptFANV34(
                                native,np.asarray(head.weights),head.bias,
                                definition,fuzzy_order="type1")
                            with torch.no_grad():
                                wrapper.source_slot.gamma.fill_(gamma)
                            q=measured["q"][:n]
                            native_out=wrapper(
                                torch.from_numpy(
                                    record["on"][:n,:,:4]),
                                torch.from_numpy(
                                    record["m_on"][:n,:,:4]),
                                extra_source_amplitude=torch.from_numpy(
                                    logit(np.clip(q,1e-6,1-1e-6)).astype(np.float32)),
                                extra_source_uncertainty=torch.zeros(n),
                                extra_source_observed_information=torch.from_numpy(
                                    measured["support"][:n].astype(np.float32)),
                                extra_source_gate_approved=gate.approved)
                            out=native_out["probability"].detach().numpy()
                            if np.max(abs(out-prob[:n]))>3e-5:
                                raise AssertionError("Native PyTorch source slot not in real risk forward")
                            human.append({
                                "device_seed":device,"NN_seed":seed,
                                "native_SHA":model_sha,
                                "measured_ON_OFF_reference_used":True,
                                "human_facing":_human_explanation(
                                    measured,used[0],base[0],prob[0],
                                    case=case,stress=stress,
                                    gate=gate.approved),
                                "no_physical_identity_claim":True,
                            })
                if sha_state(native)!=model_sha:
                    raise AssertionError("Frozen K2 changed under new measured source")
            print(json.dumps({
                "event":"V39_TRAINED_ACTUAL_NATIVE_TRANSFORMER",
                "device_seed":device,"NN_seed":seed,
                "original_K2_SHA":model_sha}),flush=True)

        # Same 1800 labels as in source TRAIN+VALID combined, but no
        # claim of matched optimizer FLOPs. A raw baseline is not a CBM.
        for ci,(case,law,fault_v2,fault_te) in enumerate(CASES):
            y=world["outcomes"][law]
            train_on=np.concatenate((on[tr],on[v1],case_data[case]["v2on"]))
            train_mask_on=np.concatenate((mtr_on,mv1_on,mv2_on))
            train_d=np.concatenate((dt,dv1,case_data[case]["v2d"]))
            train_md=np.concatenate((mt,mdv1,case_data[case]["v2dm"]))
            train_labels=np.concatenate((y[tr],y[v1],y[v2]))
            for method,use_reference in (
                ("RAW_ON_QUADRATIC_ALL1800",False),
                ("RAW_ON_OFF_QUADRATIC_ALL1800",True),
            ):
                clf=_fit_raw(
                    train_on,train_mask_on,train_d,train_md,
                    train_labels,use_reference=use_reference)
                for stress in STRESSES:
                    r=case_data[case]["test"][stress]
                    feat=_raw_features(
                        r["on"],r["m_on"],
                        r["difference"],r["m_difference"],
                        use_reference=use_reference)
                    p=clf.predict_proba(feat)[:,1]
                    save_prediction(device,case,stress,method,None,p)
                    rows.append({
                        "device_seed":device,"case":case,"stress":stress,
                        "method":method,"NN_seed":None,
                        "metrics":metrics(y[te],p),"source_use":0.,
                    })
    aggregated=[
        _summary(rows,device,case,stress,method,len(seeds))
        for device in devices for case,*_ in CASES
        for stress in STRESSES for method in METHODS
    ]
    bootstrap=[]
    for device in devices:
        for k,(case,stress,a,b) in enumerate(PAIRED_TESTS):
            actual=labels[(device,case)]
            def stack(method):
                which=seeds if method in NATIVE_METHODS else (None,)
                p=np.stack([
                    forecasts[(device,case,stress,method,s)] for s in which
                ])
                return p if len(which)==len(seeds) else np.repeat(
                    p,len(seeds),axis=0)
            bootstrap.append({
                "device_seed":device,"case":case,"stress":stress,
                "method_a":a,"method_b":b,
                **paired_test_ap_difference(
                    actual,stack(a),stack(b),
                    seed=20264191+device%101*100+k,resamples=boot),
            })
    result={
        "status":"V39_SMOKE_ONLY" if smoke else "V39_PAIRED_ACTIVE_SOURCE_ISOLATION_EXECUTED",
        "pre_registration":"docs/medical/CONCEPTFAN_V39_FROZEN_PAIRED_SOURCE_ISOLATION_PROTOCOL_20261106.md",
        "devices":list(devices),
        "n_train":n_train,"n_valid1":n_valid//2,
        "n_valid2":n_valid//2,"n_test":n_test,
        "native_K2_seeds":list(seeds),
        "n_real_native_models":len(training),
        "n_test_rows":len(rows),
        "n_summary":len(aggregated),
        "n_two_VALID_source_gates":len(valid_gates),
        "n_paired_TEST_AP_CIs":len(bootstrap),
        "external_source_isolation_switch_assumed_not_proven":True,
        "on_off_instrument_artifact_invariance_assumed_not_inferred":True,
        "source_fit_receives_no_original_H_S_or_patient_outcome":True,
        "source_inferred_only_from_jointly_measured_ON_and_OFF":True,
        "no_patient_TEST_outcomes_for_model_selection":True,
        "no_real_world_clinical_semantics":True,
        "source_fits":source_fits,"native_training":training,
        "dual_VALID_gates":valid_gates,
        "test_rows":rows,"summary":aggregated,
        "paired_TEST_AP_CI":bootstrap,
        "human_explanations":human,
    }
    output.mkdir(parents=True,exist_ok=False)
    for filename,record in (
        ("STATUS.json",result),("SUMMARY.json",aggregated),
        ("SOURCE_FITS.json",source_fits),
        ("VALID_GATES.json",valid_gates),
        ("PAIRED_AP_CI.json",bootstrap),
        ("EXPLANATIONS_RU.json",human),
    ):
        (output/filename).write_text(json.dumps(
            record,ensure_ascii=False,indent=2,allow_nan=False))
    saved={
        f"device{dev}_{case}_{stress}_{method}_seed{seed}":p
        for (dev,case,stress,method,seed),p in forecasts.items()
    }
    for (dev,case),y in labels.items():
        saved[f"device{dev}_{case}_TEST_Y"]=y
    np.savez_compressed(output/"FROZEN_TEST_PREDICTIONS.npz",**saved)
    checkpoints_dir=output/"CHECKPOINTS"
    checkpoints_dir.mkdir(exist_ok=False)
    for model in checkpoints:
        torch.save(model,checkpoints_dir/(
            f"native_K2_device{model['device_seed']}_seed{model['NN_seed']}.pt"))
    print(json.dumps({
        "event":"V39_COMPLETE",
        "status":result["status"],
        "n_models":len(training),
        "n_rows":len(rows),"n_summary":len(aggregated),
        "n_two_VALID_gates":len(valid_gates),
        "healthy_MCAR20":[v for v in aggregated if
            v["case"]=="healthy" and v["stress"]=="MCAR20"],
        "source_aligned_8x_MCAR20":[v for v in aggregated if
            v["case"]=="TEST_aligned_artifact_8x" and v["stress"]=="MCAR20"],
        "reference_OFF_drift_MCAR20":[v for v in aggregated if
            v["case"]=="TEST_OFF_artifact_drift" and v["stress"]=="MCAR20"],
        "source_leak_MCAR20":[v for v in aggregated if
            v["case"]=="TEST_source_isolation_leak_50pct" and v["stress"]=="MCAR20"],
    }),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--out-dir",type=Path,required=True)
    opt=p.parse_args()
    run(opt.out_dir,smoke=opt.smoke)
