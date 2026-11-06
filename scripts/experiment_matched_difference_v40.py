"""V40 preregistered equal-label RAW-difference vs actual Temporal ConceptFAN.

Two NEW linear and two independent nonlinear/heavy-tailed instrument devices.
Strict use of ORIGINAL observed ON and ON/OFF reference in all comparator
methods. No synthetic oracle mixing, latent source S, or outcome Y in
source discovery. NN K2 encoder gets measured proxy concepts only (no Y
gradient). Original K2 is frozen; registered residual source gamma fits
1000 TRAIN Y and passes two independently held-out 400-case VALID cohorts.

See docs/medical/CONCEPTFAN_V40_FROZEN_MATCHED_DIFFERENCE_BASELINES_PROTOCOL_20261106.md
The claim under test is NOT that source concepts beat well-tuned supervised
risk classifiers, and this controlled source-isolation switch is external
physical information rather than passive unsupervised source semantics.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit,logit

from conceptfan_discovery.difference_bench_world_v40 import (
    REGISTERED_DEVICES,STRESSES,CASES,
    generate_device_v40,render_observed_v40,
    masks_v40,measured_difference_v40,
)
from conceptfan_discovery.matched_raw_baselines_v40 import (
    REGISTERED_BASELINES,measured_temporal_quadratic_features_v40,
    fit_registered_raw_v40,
)
from conceptfan_discovery.threeview_centered_v37 import (
    fit_centered_threeview_v37,
    threeview_feature_v37,pair_feature_v37,
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
from scripts.experiment_nonlinear_anchor_repair_v30 import (
    make_firewalled_k2,
)
from scripts.experiment_automatic_measured_concepts import (
    train_masked_measured_known_concepts,logit_from_head,
)
from scripts.experiment_frozen_concept_head_refit_v26 import (
    refit_linear_concept_head,
)
from scripts.experiment_trained_conceptfan_omission_v25 import (
    predict_native,sha_state,metrics,
)

NATIVE_SEEDS=(11,23,37)
SOURCES=("ON_AB","DIFF_AB","DIFF_ABC")
NATIVE_METHODS=("K2",)+SOURCES
METHODS=NATIVE_METHODS+REGISTERED_BASELINES
CI_CONTRASTS=(
    ("healthy_energy","MCAR20","DIFF_AB","RAW_DIFF_LOGIT_TRAIN1000"),
    ("healthy_energy","MCAR20","DIFF_AB","RAW_DIFF_HGB_TRAIN1000"),
    ("healthy_energy","MCAR20","DIFF_AB","RAW_DIFF_HGB_ALL1800"),
    ("healthy_energy","MCAR20","DIFF_AB","K2"),
    ("common_U8_matched_ON_OFF","MCAR20","DIFF_AB","ON_AB"),
    ("common_U8_matched_ON_OFF","MCAR20","DIFF_AB","RAW_DIFF_HGB_TRAIN1000"),
    ("OFF_only_U4_drift","MCAR20","DIFF_AB","ON_AB"),
    ("OFF_source_leak_50pct","MCAR20","DIFF_AB","ON_AB"),
)


def _masks(on,off,stress,device,seed,known_mask=None):
    return masks_v40(
        on,off,stress,device_seed=device,
        mask_seed=seed,known_TEST_first4=known_mask)


def _missing_source(n,reason):
    return {
        "q":np.full(n,.5,dtype=np.float64),
        "support":np.zeros(n,dtype=np.float64),
        "pair_timepoints":np.zeros((n,3),dtype=np.int64),
        "all_three_pairs_measured":np.zeros(n,dtype=bool),
        "reason_no_train_identification":reason,
    }


def _fit_safe(x,m,*,seed):
    try:
        return fit_centered_threeview_v37(
            x,m,independent_view_noise_documented=True,
            seed=seed,bootstrap=24),None
    except (ValueError,AssertionError) as err:
        # Frozen identifiability protocol: a FAILED source fit must be
        # reported as abstention, not repaired using hidden simulator H.
        return None,str(err)


def _features(on,mask_on,diff,mask_diff,*,fit_on,fit_diff,fit_failures):
    n=len(on)
    return {
        "ON_AB":(
            pair_feature_v37(fit_on.pairs["AB"],on,mask_on)
            if fit_on is not None else
            _missing_source(n,fit_failures["ON"])
        ),
        "DIFF_AB":(
            pair_feature_v37(fit_diff.pairs["AB"],diff,mask_diff)
            if fit_diff is not None else
            _missing_source(n,fit_failures["DIFF"])
        ),
        "DIFF_ABC":(
            threeview_feature_v37(fit_diff,diff,mask_diff)
            if fit_diff is not None else
            _missing_source(n,fit_failures["DIFF"])
        ),
    }


def _raw_features(c):
    return {
        "on":measured_temporal_quadratic_features_v40(
            c["on"],c["mask_on"],c["diff"],c["mask_diff"],
            use_reference=False),
        "diff":measured_temporal_quadratic_features_v40(
            c["on"],c["mask_on"],c["diff"],c["mask_diff"],
            use_reference=True),
    }


def _method_use_diff(method):
    return method.startswith("RAW_DIFF_")


def _fit_one_raw(method,base_train,valid1,valid2,labels,splits,*,law):
    budget1800=method.endswith("ALL1800")
    source="diff" if _method_use_diff(method) else "on"
    yy=labels[law]
    if budget1800:
        X=np.vstack((
            base_train["raw"][source],
            valid1["raw"][source],
            valid2["raw"][source],
        ))
        y=np.concatenate((yy[splits["train"]],
                          yy[splits["valid1"]],
                          yy[splits["valid2"]]))
    else:
        X=base_train["raw"][source]
        y=yy[splits["train"]]
    return fit_registered_raw_v40(method,X,y)


def _mean_results(rows,device,family,case,stress,method,nseeds):
    subset=[r for r in rows if (
        r["device_seed"],r["case"],r["stress"],r["method"]
    )==(device,case,stress,method)]
    expected=nseeds if method in NATIVE_METHODS else 1
    if len(subset)!=expected:
        raise AssertionError(
            f"Incomplete TEST method cohort {device}/{case}/{stress}/{method}")
    return {
        "device_seed":device,"family":family,"case":case,
        "stress":stress,"method":method,
        "AUPRC":float(np.mean([x["metrics"]["auprc"] for x in subset])),
        "AUROC":float(np.mean([x["metrics"]["auroc"] for x in subset])),
        "NLL":float(np.mean([x["metrics"]["nll"] for x in subset])),
        "Brier":float(np.mean([x["metrics"]["brier"] for x in subset])),
        "source_use":float(np.mean([x["source_use"] for x in subset])),
    }


def _as_case(on,off,mask_on,mask_off,source_fitters,failures):
    d,md=measured_difference_v40(on,off,mask_on,mask_off)
    case={
        "on":on,"off":off,"mask_on":mask_on,"mask_off":mask_off,
        "diff":d,"mask_diff":md,
    }
    case["source"]=_features(
        on,mask_on,d,md,fit_on=source_fitters["ON"],
        fit_diff=source_fitters["DIFF"],fit_failures=failures)
    case["raw"]=_raw_features(case)
    return case


def _source_head_admission(
    method,source_train,source_v1,source_v2,
    *,
    y,train,valid1,valid2,
    base_train,base_v1,base_v2,
    bootstrap,seed,
):
    train_source=source_train[method]
    if not np.any(train_source["support"]):
        # Don't invoke optimizer on a source which is IDENTIFIABLY
        # absent. The exact safe output is gamma=0 and approved=False.
        return 0.,{
            "approved":False,
            "reason":"ABSTAIN_NO_TRAIN_SOURCE_IDENTIFIABILITY",
            "environment_receipts":[],
            "train_label_count":len(train),
            "valid_label_count":len(valid1)+len(valid2),
        }
    gamma=train_source_coefficient(
        base_train,train_source["q"],y[train])
    e1=source_v1[method]
    e2=source_v2[method]
    gate=evaluate_two_independent_environments(
        method,gamma,
        ("VALID1",y[valid1],base_v1,e1["q"],e1["support"]),
        ("VALID2",y[valid2],base_v2,e2["q"],e2["support"]),
        train_label_count=len(train),
        min_mean_gain=.003,min_support_fraction=.25,
        min_subject_support=.50,
        bootstrap_reps=bootstrap,seed=seed,
    )
    return gamma,gate.as_dict()


def run(out:Path,*,smoke:bool=False):
    if out.exists():
        raise FileExistsError("Frozen V40 evidence directory exists; never overwrite")
    torch.set_num_threads(2)
    ntr,nva,nte=(360,300,240) if smoke else (1000,800,600)
    seeds=(11,) if smoke else NATIVE_SEEDS
    devices=(20261407,) if smoke else tuple(REGISTERED_DEVICES)
    boot=199 if smoke else 399
    rows=[];summary=[];gates=[];source_state=[];native_training=[]
    predictions={};y_test={};real_model_states=[];explanations=[]
    def record_pred(device,case,stress,method,seed,p):
        predictions[(device,case,stress,method,seed)]=np.asarray(
            p,dtype=np.float64).copy()

    for device in devices:
        world=generate_device_v40(
            device_seed=device,n_train=ntr,n_valid=nva,n_test=nte)
        family=world["family"]
        split={key:world[key] for key in ("train","valid1","valid2","test")}
        tr,v1,v2,te=(split[k] for k in split)
        # Unlike an accidentally varying K2 baseline across stress
        # conditions, the ORIGINAL K2 TEST sensor masks are identical
        # for every candidate, strong raw comparator and fault world.
        shared_test_known=(
            np.random.default_rng(20264175+device%101).random(
                (len(te),16,4))>=.20)
        ontr,offtr=render_observed_v40(
            world,tr,fault="none",seed=20264011+device%101)
        onv1,offv1=render_observed_v40(
            world,v1,fault="none",seed=20264012+device%101)
        trainmask=_masks(ontr,offtr,"MCAR20",device,20264021)
        val1mask=_masks(onv1,offv1,"MCAR20",device,20264022)
        train_base={} # set after fits initialized
        fit_failures={}
        on_fit,e1=_fit_safe(ontr,trainmask[0],
                            seed=20264031+device%101)
        dtr,dmtr=measured_difference_v40(
            ontr,offtr,*trainmask)
        diff_fit,e2=_fit_safe(dtr,dmtr,seed=20264041+device%101)
        fitters={"ON":on_fit,"DIFF":diff_fit}
        fit_failures={"ON":e1,"DIFF":e2}
        source_state.append({
            "device_seed":device,"family":family,
            "ON_fit_error":e1,"DIFF_fit_error":e2,
            "ON_train_source":on_fit.audit() if on_fit else None,
            "DIFF_train_source":diff_fit.audit() if diff_fit else None,
            "learned_mixing_matrices_are_not_simulator_H":True,
            "prior_known_true_source_variables_used":False,
            "physical_ON_OFF_reference_external_assumption":True,
            "nonlinear_source_transfer_may_invalidate_reference_subtraction":
                family=="nonlinear_switching",
        })
        train_base=_as_case(
            ontr,offtr,*trainmask,fitters,fit_failures)
        valid1_base=_as_case(
            onv1,offv1,*val1mask,fitters,fit_failures)
        valid2_cache={}
        for valid_fault in ("none","off_drift"):
            onv2,offv2=render_observed_v40(
                world,v2,fault=valid_fault,
                seed=20264050+device%101)
            m2=_masks(
                onv2,offv2,"MCAR20",device,20264023)
            valid2_cache[valid_fault]=_as_case(
                onv2,offv2,*m2,fitters,fit_failures)
        tests={}
        for ci,(case,law,v2fault,testfault) in enumerate(CASES):
            onte,offte=render_observed_v40(
                world,te,fault=testfault,
                seed=20264101+ci+device%101)
            one={}
            for mi,stress in enumerate(STRESSES):
                mo,mr=_masks(
                    onte,offte,stress,device,
                    seed=20264141+mi,
                    known_mask=shared_test_known,
                )
                if testfault=="no_reference":
                    mr[:,:,4:10]=False
                one[stress]=_as_case(
                    onte,offte,mo,mr,fitters,fit_failures)
            tests[case]=one
            y_test[(device,case)]=world["outcomes"][law][te]

        # Pre-score algebraic probe: only the LINEAR family guarantees
        # exact cancellation of physically aligned +shared U during ON/OFF.
        for case in ("common_U4_matched_ON_OFF",
                     "common_U8_matched_ON_OFF"):
            clean=tests["healthy_energy"]["MCAR20"]
            changed=tests[case]["MCAR20"]
            error=float(np.max(abs(
                clean["diff"]-changed["diff"])))
            if family=="linear_AR" and error>6e-6:
                raise AssertionError(
                    "Failed exact matched OFF artifact cancellation in linear family")
            if family=="nonlinear_switching" and error<.03:
                raise AssertionError(
                    "Falsifier disappeared: nonlinear transfer unexpectedly subtracts exactly")

        # No TEST labels involved in source fit. Truth correlation is an
        # EXPLICITLY separate synthetic post-fit audit.
        actual_energy=world["true_slow_source_energy_POSTFIT_AUDIT_ONLY"][te]
        for method in SOURCES:
            v=tests["healthy_energy"]["MCAR20"]["source"][method]
            support=v["support"]>0
            correlation=(
                float(np.corrcoef(actual_energy[support],v["q"][support])[0,1])
                if int(support.sum())>30 else None)
            source_state[-1][f"TEST_true_energy_correlation_{method}_AUDIT_ONLY"]=correlation

        raw_models={}
        for case,law,v2fault,_ in CASES:
            for method in REGISTERED_BASELINES:
                key=(law,v2fault,method)
                if key in raw_models:
                    continue
                raw_models[key]=_fit_one_raw(
                    method,train_base,valid1_base,
                    valid2_cache[v2fault],
                    world["outcomes"],split,law=law)

        # Calculate direct conventional comparisons with all 6 methods
        # ONCE per case/stress, independent of NN random initialization.
        for case,law,v2fault,_ in CASES:
            y=world["outcomes"][law][te]
            for stress in STRESSES:
                r=tests[case][stress]
                for method in REGISTERED_BASELINES:
                    raw=raw_models[(law,v2fault,method)]
                    which="diff" if _method_use_diff(method) else "on"
                    prob=raw.predict_proba(r["raw"][which])[:,1]
                    record_pred(device,case,stress,method,None,prob)
                    rows.append({
                        "device_seed":device,"family":family,
                        "case":case,"stress":stress,"method":method,
                        "NN_seed":None,"n_outcome_training_labels":
                            ntr+nva if method.endswith("ALL1800") else ntr,
                        "metrics":metrics(y,prob),"source_use":0.,
                    })

        for nn_seed in seeds:
            native=make_firewalled_k2(nn_seed)
            train_receipt=train_masked_measured_known_concepts(
                native,ontr[:,:,:4],trainmask[0][:,:,:4],
                seed=nn_seed,epochs=2 if smoke else 10)
            model_sha=sha_state(native)
            native_training.append({
                "device_seed":device,"family":family,"NN_seed":nn_seed,
                "native_model_sha":model_sha,
                **train_receipt,
            })
            real_model_states.append({
                "device_seed":device,"NN_seed":nn_seed,
                "frozen_model_sha":model_sha,
                "state_dict":{k:v.detach().cpu().clone()
                              for k,v in native.state_dict().items()},
            })
            qtr=predict_native(
                native,ontr[:,:,:4],trainmask[0][:,:,:4])["concept_summary"]
            q1=predict_native(
                native,onv1[:,:,:4],val1mask[0][:,:,:4])["concept_summary"]
            # Same known channels & same known mask for every fault/stress.
            onte=tests["healthy_energy"]["MCAR20"]["on"]
            qe=predict_native(
                native,onte[:,:,:4],shared_test_known)["concept_summary"]
            gate_cache={}
            for ci,(case,law,v2fault,testfault) in enumerate(CASES):
                key=(law,v2fault)
                if key not in gate_cache:
                    valid2=valid2_cache[v2fault]
                    q2=predict_native(
                        native,valid2["on"][:,:,:4],
                        valid2["mask_on"][:,:,:4],
                    )["concept_summary"]
                    head=refit_linear_concept_head(qtr,world["outcomes"][law][tr])
                    ztr=logit_from_head(head,qtr)
                    zv1=logit_from_head(head,q1)
                    zv2=logit_from_head(head,q2)
                    ztest=logit_from_head(head,qe)
                    source_heads={}
                    for method in SOURCES:
                        gamma,receipt=_source_head_admission(
                            method,train_base["source"],
                            valid1_base["source"],valid2["source"],
                            y=world["outcomes"][law],
                            train=tr,valid1=v1,valid2=v2,
                            base_train=ztr,base_v1=zv1,base_v2=zv2,
                            bootstrap=boot,
                            seed=20264201+device%101*100+
                                 ci*13+nn_seed,
                        )
                        source_heads[method]=(gamma,receipt)
                    gate_cache[key]=(head,ztest,source_heads)
                head,ztest,source_heads=gate_cache[key]
                baseline=expit(ztest)
                for stress in STRESSES:
                    rec=tests[case][stress]
                    record_pred(device,case,stress,"K2",nn_seed,baseline)
                    rows.append({
                        "device_seed":device,"family":family,
                        "case":case,"stress":stress,"method":"K2",
                        "NN_seed":nn_seed,
                        "n_outcome_training_labels":ntr,
                        "metrics":metrics(world["outcomes"][law][te],baseline),
                        "source_use":0.,
                    })
                    for method in SOURCES:
                        gamma,gate=source_heads[method]
                        f=rec["source"][method]
                        z,used=admitted_test_logit(
                            ztest,f["q"],f["support"],
                            gamma=gamma,approved=gate["approved"],
                            minimum_fraction=.50)
                        np.testing.assert_array_equal(z[~used],ztest[~used])
                        p=expit(z)
                        record_pred(device,case,stress,method,nn_seed,p)
                        rows.append({
                            "device_seed":device,"family":family,
                            "case":case,"stress":stress,
                            "method":method,"NN_seed":nn_seed,
                            "n_outcome_training_labels":ntr,
                            "metrics":metrics(
                                world["outcomes"][law][te],p),
                            "source_use":float(used.mean()),
                            "source_approved_on_two_disjoint_VALID":
                                bool(gate["approved"]),
                        })
                        if (case=="healthy_energy" and
                            stress=="MCAR20" and method=="DIFF_AB"):
                            # Native in-graph evidence: this measured
                            # concept is wired into the native PyTorch
                            # decision computation, NOT post-hoc SHAP.
                            concept=FuzzySourceDefinition(
                                source_id="paired_measured_slow_source_energy",
                                sensor_indices=(4,5,6,7),
                                threshold_train=0.,width_train=1.,
                                train_missingness_spread=0.,
                                boundary_spread_anchor=0.,
                                certified_calibration_environments=0,
                                external_anchor_verified=False)
                            wrapper=NativeAuditedFuzzyConceptFANV34(
                                native,np.asarray(head.weights),
                                head.bias,concept,fuzzy_order="type1")
                            with torch.no_grad():
                                wrapper.source_slot.gamma.fill_(gamma)
                            sample_n=12
                            outnative=wrapper(
                                torch.from_numpy(
                                    rec["on"][:sample_n,:,:4]),
                                torch.from_numpy(
                                    rec["mask_on"][:sample_n,:,:4]),
                                extra_source_amplitude=torch.from_numpy(
                                    logit(np.clip(
                                        f["q"][:sample_n],1e-6,1-1e-6)
                                    ).astype(np.float32)),
                                extra_source_uncertainty=torch.zeros(sample_n),
                                extra_source_observed_information=torch.from_numpy(
                                    f["support"][:sample_n].astype(np.float32)),
                                extra_source_gate_approved=gate["approved"],
                            )
                            if np.max(abs(
                                outnative["probability"].detach().numpy()-
                                p[:sample_n]))>3e-5:
                                raise AssertionError(
                                    "New measured concept not in real K2+source risk computation")
                            explanations.append({
                                "device_seed":device,"family":family,
                                "NN_seed":nn_seed,
                                "source":"ON−OFF paired measured slow fluctuation energy; physical isolation external",
                                "source_used_for_example0":bool(used[0]),
                                "observed_joint_pair_timepoints_example0":
                                    int(f["paired_timepoints"][0]),
                                "original_probability_example0":float(baseline[0]),
                                "expanded_probability_example0":float(p[0]),
                                "change_percentage_points":
                                    100*float(p[0]-baseline[0]),
                                "source_gate_reason":gate.get("reason"),
                                "source_fit_error":fit_failures["DIFF"],
                                "not_clinically_or_causally_certified":True,
                                "unmonitored_OFF_drift_remains_unsafe":True,
                            })
                for method in SOURCES:
                    gamma,receipt=source_heads[method]
                    # Every CASE still has an explicit named receipt,
                    # even if identical TRAIN/VALID method results are
                    # reused to avoid wasteful refitting.
                    gates.append({
                        "device_seed":device,"family":family,
                        "NN_seed":nn_seed,"case":case,
                        "method":method,
                        "TRAIN_gamma":gamma,"receipt":receipt,
                        "n_total_labels_for_gamma":ntr,
                    })
                if sha_state(native)!=model_sha:
                    raise AssertionError("Additional concept modified frozen K2")
            print(json.dumps({
                "event":"V40_NATIVE_TRANSFORMER_TRAINED",
                "device_seed":device,"family":family,
                "NN_seed":nn_seed,"frozen_K2_SHA":model_sha,
            }),flush=True)

        for case,*_ in CASES:
            for stress in STRESSES:
                for method in METHODS:
                    summary.append(_mean_results(
                        rows,device,family,case,stress,method,len(seeds)))
        print(json.dumps({
            "event":"V40_DEVICE_COMPLETE","device_seed":device,
            "family":family,
            "source_fit_failures":fit_failures,
            "n_summary_so_far":len(summary),
        }),flush=True)

    comparisons=[]
    for device in devices:
        for k,(case,stress,method_a,method_b) in enumerate(CI_CONTRASTS):
            y=y_test[(device,case)]
            def preds(method):
                members=seeds if method in NATIVE_METHODS else (None,)
                arr=np.stack([
                    predictions[(device,case,stress,method,s)]
                    for s in members])
                if len(members)==1 and len(seeds)!=1:
                    arr=np.repeat(arr,len(seeds),axis=0)
                return arr
            comparisons.append({
                "device_seed":device,
                "family":REGISTERED_DEVICES[device],
                "case":case,"stress":stress,
                "method_a":method_a,"method_b":method_b,
                **paired_test_ap_difference(
                    y,preds(method_a),preds(method_b),
                    seed=20264301+device%101*100+k,
                    resamples=boot),
            })
    result={
        "status":"V40_SMOKE_ONLY" if smoke else "V40_MATCHED_RAW_DIFFERENCE_NONLINEAR_EXECUTED",
        "frozen_before_TEST_protocol":
          "docs/medical/CONCEPTFAN_V40_FROZEN_MATCHED_DIFFERENCE_BASELINES_PROTOCOL_20261106.md",
        "device_families":{str(k):REGISTERED_DEVICES[k] for k in devices},
        "devices":list(devices),
        "n_train":ntr,"n_valid1":nva//2,
        "n_valid2":nva//2,"n_test":nte,
        "NN_seeds":list(seeds),
        "n_real_native_K2_fits":len(native_training),
        "n_TEST_metric_rows":len(rows),
        "n_summary":len(summary),
        "n_dual_VALID_receipts":len(gates),
        "n_PRE_REGISTERED_AP_CIs":len(comparisons),
        "source_definition_frozen_before_TEST":True,
        "same_TEST_patient_K2_sensor_mask_across_every_method":True,
        "raw_diff_baselines_see_true_measured_pair_ON_OFF_no_H_or_S":True,
        "raw_logit_and_HGB_have_1000_TRAIN_and_1800_ALL_budgets":True,
        "no_TEST_y_used_in_feature_fit_or_gate":True,
        "physical_switch_is_external_supervision":True,
        "nonlinear_transfer_negates_general_ON_OFF_algebra":True,
        "no_claim_of_clinical_or_physical_source_identity":True,
        "source_fit_provenance":source_state,
        "native_training":native_training,
        "dual_VALID_gates":gates,
        "test_metric_rows":rows,
        "summary":summary,
        "paired_AP_CIs":comparisons,
        "native_concept_explanation_examples":explanations,
    }
    out.mkdir(parents=True,exist_ok=False)
    for filename,content in (
        ("STATUS.json",result),
        ("SUMMARY.json",summary),
        ("TEST_ROWS.json",rows),
        ("SOURCE_FITS.json",source_state),
        ("DUAL_VALID_GATES.json",gates),
        ("PAIRED_BOOTSTRAP_AP.json",comparisons),
        ("EXPLANATIONS_RU.json",explanations),
    ):
        (out/filename).write_text(
            json.dumps(content,indent=2,ensure_ascii=False,allow_nan=False))
    frozen={
        f"device{dev}_{case}_{stress}_{method}_seed{seed}":p
        for (dev,case,stress,method,seed),p in predictions.items()
    }
    for (dev,case),y in y_test.items():
        frozen[f"device{dev}_{case}_TEST_Y"]=y
    np.savez_compressed(out/"FROZEN_TEST_PREDICTIONS.npz",**frozen)
    ckpt_dir=out/"CHECKPOINTS"
    ckpt_dir.mkdir()
    for ck in real_model_states:
        torch.save(ck,ckpt_dir/
            f"native_K2_device{ck['device_seed']}_seed{ck['NN_seed']}.pt")
    print(json.dumps({
        "event":"V40_COMPLETE",
        "status":result["status"],
        "n_native":len(native_training),
        "n_test_metric_rows":len(rows),
        "n_summary":len(summary),
        "n_gates":len(gates),
        "n_paired_AP_CIs":len(comparisons),
        "healthy_MCAR20":[row for row in summary
            if row["case"]=="healthy_energy" and row["stress"]=="MCAR20"],
        "U8_MCAR20":[row for row in summary
            if row["case"]=="common_U8_matched_ON_OFF"
            and row["stress"]=="MCAR20"],
        "OFF_drift_MCAR20":[row for row in summary
            if row["case"]=="OFF_only_U4_drift" and row["stress"]=="MCAR20"],
        "OFF_leak_MCAR20":[row for row in summary
            if row["case"]=="OFF_source_leak_50pct"
            and row["stress"]=="MCAR20"],
        "sources":source_state,
        "paired_AP_CIs":comparisons,
    }),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    run(a.out_dir,smoke=a.smoke)
