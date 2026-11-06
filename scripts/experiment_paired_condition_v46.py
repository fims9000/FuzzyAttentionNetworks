"""V44 preregistered paired TEST experiment; never tune by TEST outcomes."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from conceptfan_discovery.paired_condition_world_v46 import DEVICES_V46, fresh_device_v46
from conceptfan_discovery.paired_condition_monitor_v46 import (
    fit_paired_condition_v46,apply_paired_condition_v46,
)
from conceptfan_discovery.off_reference_reliability_v45 import (
    fit_off_guard_v45, apply_off_guard_v45, gate_partial_view_source_v45,
)
from conceptfan_discovery.partial_view_concept_v44 import fit_partial_view_concept_v44, partial_view_concept_v44
from conceptfan_discovery.adaptive_pair_bench_world_v41 import render_observed_v41, masks_v41, measured_difference_v41
from conceptfan_discovery.adaptive_pair_consensus_v41 import fit_adaptive_source_v41, adaptive_pair_feature_v41
from conceptfan_discovery.threeview_centered_v37 import fit_centered_threeview_v37
from conceptfan_discovery.matched_raw_baselines_v40 import measured_temporal_quadratic_features_v40, fit_registered_raw_v40
from conceptfan_discovery.native_adaptive_type3_transformer_v34 import NativeAuditedFuzzyConceptFANV34
from conceptfan_discovery.adaptive_higher_order_concepts_v34 import FuzzySourceDefinition
from conceptfan_discovery.auto_measured_concept_admission import admitted_test_logit
from conceptfan_discovery.cross_environment_transport import paired_test_ap_difference
from scripts.experiment_adaptive_pair_v41 import _source_head_admission, _missing_source
from scripts.experiment_nonlinear_anchor_repair_v30 import make_firewalled_k2
from scripts.experiment_automatic_measured_concepts import train_masked_measured_known_concepts, logit_from_head
from scripts.experiment_frozen_concept_head_refit_v26 import refit_linear_concept_head
from scripts.experiment_trained_conceptfan_omission_v25 import predict_native, sha_state, metrics

METHODS=("K2","V44_PARTIAL_VIEW","V45_OFF_GUARD","V46_PAIRED_CONCORDANCE",
         "RAW_DIFF_HGB_TRAIN1000","RAW_DIFF_HGB_ALL1800",
         "RAW_DIFF_HGB_ALL1800_PAIRED_GUARD")
CONCEPT_METHODS=("V44_PARTIAL_VIEW","V45_OFF_GUARD","V46_PAIRED_CONCORDANCE")
NATIVE_METHODS=("K2","V44_PARTIAL_VIEW","V45_OFF_GUARD",
                "V46_PAIRED_CONCORDANCE","RAW_DIFF_HGB_ALL1800_PAIRED_GUARD")
RAW_METHODS=("RAW_DIFF_HGB_TRAIN1000","RAW_DIFF_HGB_ALL1800")
STRESS=("MCAR20","MCAR40","MCAR50","missing_view_A","missing_reference")
SCENARIOS=(
    ("healthy_energy","energy","none","none"),
    ("common_U8_matched","energy","none","common8"),
    ("unknown_OFF_drift","energy","none","off_drift"),
    ("unknown_A_inversion","energy","none","invert_on_A"),
    ("OFF_source_leak","energy","none","leak"),
    ("complete_K2_dictionary","complete","none","none"),
    ("known_VALID2_OFF_drift","energy","off_drift","off_drift"),
)
COMPARE=(
    ("common_U8_matched","MCAR40","V46_PAIRED_CONCORDANCE","V45_OFF_GUARD"),
    ("unknown_OFF_drift","MCAR40","V46_PAIRED_CONCORDANCE","V45_OFF_GUARD"),
    ("unknown_OFF_drift","MCAR40","V46_PAIRED_CONCORDANCE","K2"),
    ("healthy_energy","MCAR40","V46_PAIRED_CONCORDANCE","V44_PARTIAL_VIEW"),
    ("healthy_energy","MCAR40","V46_PAIRED_CONCORDANCE","RAW_DIFF_HGB_ALL1800"),
    ("unknown_OFF_drift","MCAR40","V46_PAIRED_CONCORDANCE","RAW_DIFF_HGB_ALL1800_PAIRED_GUARD"),
)


def prepare_case(on,off,mask_on,mask_off,old_fit,new_fit,off_guard,paired_guard):
    diff,diff_mask=measured_difference_v41(on,off,mask_on,mask_off)
    n=len(on)
    source=(partial_view_concept_v44(new_fit,diff,diff_mask) if new_fit else
            _missing_source(n,"NO_VALID_V44_TRAIN_FIT"))
    if off_guard is None:
        off_monitor={
            "usable_for_additional_concept":np.zeros(n,dtype=bool),
            "off_background_alarm":np.zeros(n,dtype=bool),
            "decision_reason":np.repeat("NO_HEALTHY_OFF_TRAIN_VALID_CERTIFICATE",n),
            "score":np.zeros(n),"support":np.zeros(n,dtype=bool),
        }
    else:
        off_monitor=apply_off_guard_v45(off_guard,off,mask_off)
    if paired_guard is None:
        paired_monitor={
            "usable_for_additional_concept":np.zeros(n,dtype=bool),
            "joint_condition_alarm":np.zeros(n,dtype=bool),
            "condition_support":np.zeros(n,dtype=bool),
            "decision_reason":np.repeat("NO_HEALTHY_PAIRED_VALID_CERTIFICATE",n),
            "score":np.zeros(n),"difference_score":np.zeros(n),
        }
    else:
        paired_monitor=apply_paired_condition_v46(
            paired_guard,on,off,mask_on,mask_off)
    return {
        "on":on,"off":off,"on_mask":mask_on,"off_mask":mask_off,
        "diff":diff,"diff_mask":diff_mask,
        "source":{
            "V44_PARTIAL_VIEW":source,
            "V45_OFF_GUARD":gate_partial_view_source_v45(source,off_monitor),
            "V46_PAIRED_CONCORDANCE":gate_partial_view_source_v45(source,paired_monitor),
        },
        "off_monitor":off_monitor,"paired_monitor":paired_monitor,
        "raw":measured_temporal_quadratic_features_v40(
            on,mask_on,diff,diff_mask,use_reference=True),
    }


def check_native_source_graph(native,head,gamma,approved,case,expected_logits):
    """The added concept must be computed inside the actual native torch graph."""
    n=min(12,len(expected_logits))
    definition=FuzzySourceDefinition(
        source_id="TRAIN_measured_partial_pair_energy",
        sensor_indices=(4,5,6,7,8,9),threshold_train=0.,width_train=1.,
        train_missingness_spread=0.,boundary_spread_anchor=0.,
        certified_calibration_environments=0,external_anchor_verified=False)
    wrapper=NativeAuditedFuzzyConceptFANV34(
        native,np.asarray(head.weights),head.bias,definition,fuzzy_order="type1")
    with torch.no_grad():
        wrapper.source_slot.gamma.fill_(gamma)
        source=case["source"]["V46_PAIRED_CONCORDANCE"]
        q=np.clip(source["q"][:n],1e-6,1-1e-6)
        result=wrapper(
            torch.from_numpy(case["on"][:n,:,:4]),
            torch.from_numpy(case["on_mask"][:n,:,:4]),
            extra_source_amplitude=torch.from_numpy(
                np.log(q/(1-q)).astype(np.float32)),
            extra_source_uncertainty=torch.zeros(n),
            extra_source_observed_information=torch.from_numpy(
                source["support"][:n].astype(np.float32)),
            extra_source_gate_approved=approved,
        )
        actual=result["logit"].cpu().numpy()
    if np.max(np.abs(actual-expected_logits[:n]))>3e-5:
        raise AssertionError("V46 paired-condition concept does not execute in actual torch.forward")
    return {"native_concept_in_graph":True,"physical_identity_certified":False}


def run(out:Path,*,smoke=False):
    if out.exists(): raise FileExistsError("Never overwrite TEST evidence")
    torch.set_num_threads(2)
    devices=(20262107,) if smoke else tuple(DEVICES_V46)
    nn_seeds=(11,) if smoke else (11,23,37)
    ntr,nva,nte=(360,300,240) if smoke else (1000,800,600)
    bootstrap=199 if smoke else 399
    summary=[];rows=[];gates=[];audits=[];paired=[];predictions={}
    test_labels={};trained_models=[];native_checkpoints=[];off_guard_receipts=[];paired_guard_receipts=[]
    monitor_evidence={};risk_use_evidence={}
    for device in devices:
        world=fresh_device_v46(device_seed=device,n_train=ntr,n_valid=nva,n_test=nte)
        tr,v1,v2,te=(world[k] for k in ("train","valid1","valid2","test"))
        labels=world["outcomes"]
        ontr,offtr=render_observed_v41(world,tr,fault="none",seed=20264401+device%101)
        mo,mr=masks_v41(ontr,offtr,"MCAR20",device_seed=device,mask_seed=20264411)
        dtr,dmtr=measured_difference_v41(ontr,offtr,mo,mr)
        try:
            v37=fit_centered_threeview_v37(
                dtr,dmtr,independent_view_noise_documented=True,
                seed=20264421+device%101,bootstrap=24)
            old_fit=fit_adaptive_source_v41(v37,dtr,dmtr)
            new_fit=fit_partial_view_concept_v44(v37,dtr,dmtr)
            fit_error=None
        except (ValueError,AssertionError) as exc:
            v37=None;old_fit=None;new_fit=None;fit_error=str(exc)
        ctr=prepare_case(ontr,offtr,mo,mr,old_fit,new_fit,None,None)
        on1,off1=render_observed_v41(world,v1,fault="none",seed=20264402+device%101)
        mo1,mr1=masks_v41(on1,off1,"MCAR20",device_seed=device,mask_seed=20264412)
        c1=prepare_case(on1,off1,mo1,mr1,old_fit,new_fit,None,None)
        val2={}
        for fault in ("none","off_drift"):
            on2,off2=render_observed_v41(world,v2,fault=fault,seed=20264403+device%101)
            mo2,mr2=masks_v41(on2,off2,"MCAR20",device_seed=device,mask_seed=20264413)
            val2[fault]=prepare_case(on2,off2,mo2,mr2,old_fit,new_fit,None,None)
        # Train/current OFF sentinel uses no TEST, outcome Y, latent H/S/E.
        try:
            healthy2=val2["none"]
            off_guard=fit_off_guard_v45(
                offtr,mr,off1,mr1,healthy2["off"],healthy2["off_mask"])
            off_guard_error=None
        except (ValueError,AssertionError) as exc:
            off_guard=None
            off_guard_error=str(exc)
        off_guard_receipts.append({
            "device":device,"fit_error":off_guard_error,
            "train_valid_only":off_guard.audit() if off_guard else None,
            "physical_identity_certified":False,
        })
        try:
            paired_guard=fit_paired_condition_v46(
                ontr,offtr,mo,mr,
                on1,off1,mo1,mr1,
                healthy2["on"],healthy2["off"],
                healthy2["on_mask"],healthy2["off_mask"])
            paired_guard_error=None
        except (ValueError,AssertionError) as exc:
            paired_guard=None
            paired_guard_error=str(exc)
        paired_guard_receipts.append({
            "device":device,"fit_error":paired_guard_error,
            "train_valid_only":paired_guard.audit() if paired_guard else None,
            "physical_identity_certified":False,
        })
        ctr=prepare_case(ontr,offtr,mo,mr,old_fit,new_fit,off_guard,paired_guard)
        c1=prepare_case(on1,off1,mo1,mr1,old_fit,new_fit,off_guard,paired_guard)
        old_valid2=val2
        val2={
            fault:prepare_case(c["on"],c["off"],c["on_mask"],c["off_mask"],
                               old_fit,new_fit,off_guard,paired_guard)
            for fault,c in old_valid2.items()
        }
        known_test_mask=(np.random.default_rng(20264441+device%101)
                         .random((len(te),16,4))>=.2)
        test_cases={}
        for ci,(case,law,valid_fault,test_fault) in enumerate(SCENARIOS):
            on,off=render_observed_v41(world,te,fault=test_fault,
                                      seed=20264451+ci+device%101)
            test_labels[device,case]=labels[law][te]
            for mi,stress in enumerate(STRESS):
                ma,mb=masks_v41(on,off,stress,device_seed=device,
                               mask_seed=20264461+mi,known_TEST_first4=known_test_mask)
                if stress=="missing_reference": mb[:,:,4:10]=False
                test_cases[case,stress]=prepare_case(on,off,ma,mb,old_fit,new_fit,off_guard,paired_guard)
                # Post-TEST read-only provenance: actual OFF sensor condition and
                # measured source support, no Y or hidden simulator S/H/E/U.
                observed_test=test_cases[case,stress]
                stamp=f"device{device}_{case}_{stress}"
                current=observed_test["off_monitor"]
                monitor_evidence[stamp+"_off_score"]=np.asarray(
                    current["score"],dtype=np.float64).copy()
                monitor_evidence[stamp+"_off_usable"]=np.asarray(
                    current["usable_for_additional_concept"],dtype=bool).copy()
                monitor_evidence[stamp+"_off_alarm"]=np.asarray(
                    current["off_background_alarm"],dtype=bool).copy()
                monitor_evidence[stamp+"_off_support"]=np.asarray(
                    current["support"],dtype=bool).copy()
                paired=observed_test["paired_monitor"]
                monitor_evidence[stamp+"_paired_off_score"]=np.asarray(
                    paired["score"],dtype=np.float64).copy()
                monitor_evidence[stamp+"_paired_diff_score"]=np.asarray(
                    paired["difference_score"],dtype=np.float64).copy()
                monitor_evidence[stamp+"_paired_usable"]=np.asarray(
                    paired["usable_for_additional_concept"],dtype=bool).copy()
                monitor_evidence[stamp+"_paired_alarm"]=np.asarray(
                    paired["joint_condition_alarm"],dtype=bool).copy()
                monitor_evidence[stamp+"_paired_support"]=np.asarray(
                    paired["condition_support"],dtype=bool).copy()
                for method in CONCEPT_METHODS:
                    source=observed_test["source"][method]
                    monitor_evidence[stamp+"_"+method+"_measured_support"]=np.asarray(
                        source["support"],dtype=bool).copy()

        audits.append({
            "device":device,"family":world["family"],"fit_failure":fit_error,
            "v37_train_only":v37.audit() if v37 else None,
            "v44_train_only":new_fit.audit() if new_fit else None,
            "independent_ON_OFF_source_isolation_assumed":True,
        })
        raw_models={}
        for law in ("energy","complete"):
            for method in RAW_METHODS:
                feat=ctr["raw"]
                yy=labels[law][tr]
                if method.endswith("ALL1800"):
                    feat=np.vstack((feat,c1["raw"],val2["none"]["raw"]))
                    yy=np.concatenate((yy,labels[law][v1],labels[law][v2]))
                raw_models[law,method]=fit_registered_raw_v40(method,feat,yy)
        for case,law,_,_ in SCENARIOS:
            for stress in STRESS:
                c=test_cases[case,stress]
                for method in RAW_METHODS:
                    p=raw_models[law,method].predict_proba(c["raw"])[:,1]
                    predictions[device,case,stress,method,None]=p
                    rows.append({"device":device,"case":case,"stress":stress,
                                 "method":method,"seed":None,
                                 "metrics":metrics(labels[law][te],p),"source_use":0.})
        for seed in nn_seeds:
            native=make_firewalled_k2(seed)
            record=train_masked_measured_known_concepts(
                native,ontr[:,:,:4],mo[:,:,:4],seed=seed,epochs=2 if smoke else 10)
            sha=sha_state(native)
            trained_models.append({"device":device,"seed":seed,"sha":sha,**record})
            native_checkpoints.append({
                "device":device,"seed":seed,"sha":sha,
                "state_dict":{k:v.detach().cpu().clone()
                              for k,v in native.state_dict().items()},
            })
            qtr=predict_native(native,ontr[:,:,:4],mo[:,:,:4])["concept_summary"]
            q1=predict_native(native,on1[:,:,:4],mo1[:,:,:4])["concept_summary"]
            ontest=test_cases["healthy_energy","MCAR20"]["on"][:,:,:4]
            qtest=predict_native(native,ontest,known_test_mask)["concept_summary"]
            for law in ("energy","complete"):
                head=refit_linear_concept_head(qtr,labels[law][tr])
                ztr=logit_from_head(head,qtr)
                zv1=logit_from_head(head,q1)
                ztest=logit_from_head(head,qtest)
                gate_cache={}
                for fault in ("none","off_drift"):
                    c2=val2[fault]
                    q2=predict_native(native,c2["on"][:,:,:4],
                                      c2["on_mask"][:,:,:4])["concept_summary"]
                    zv2=logit_from_head(head,q2)
                    for ki,method in enumerate(CONCEPT_METHODS):
                        gamma,gate=_source_head_admission(
                            method,ctr["source"],c1["source"],c2["source"],
                            y=labels[law],train=tr,valid1=v1,valid2=v2,
                            base_train=ztr,base_v1=zv1,base_v2=zv2,
                            bootstrap=bootstrap,seed=20264501+device%101*100+seed*7+ki)
                        gate_cache[fault,method]=(gamma,gate)
                for case,this_law,valid_fault,_ in SCENARIOS:
                    if this_law!=law: continue
                    for stress in STRESS:
                        c=test_cases[case,stress]
                        y=labels[law][te]
                        baseline=expit(ztest)
                        predictions[device,case,stress,"K2",seed]=baseline
                        rows.append({"device":device,"case":case,"stress":stress,
                                     "method":"K2","seed":seed,
                                     "metrics":metrics(y,baseline),"source_use":0.})
                        off_ok=np.asarray(
                            c["paired_monitor"]["usable_for_additional_concept"],dtype=bool)
                        raw_ungated=predictions[
                            device,case,stress,"RAW_DIFF_HGB_ALL1800",None]
                        raw_gated=np.where(off_ok,raw_ungated,baseline)
                        predictions[device,case,stress,
                                    "RAW_DIFF_HGB_ALL1800_PAIRED_GUARD",seed]=raw_gated
                        rows.append({
                            "device":device,"case":case,"stress":stress,
                            "method":"RAW_DIFF_HGB_ALL1800_PAIRED_GUARD",
                            "seed":seed,"metrics":metrics(y,raw_gated),
                            "source_use":0.,"raw_OFF_guard_kept_fraction":float(off_ok.mean()),
                        })
                        for method in CONCEPT_METHODS:
                            gamma,gate=gate_cache[valid_fault,method]
                            source=c["source"][method]
                            logits,used=admitted_test_logit(
                                ztest,source["q"],source["support"],gamma=gamma,
                                approved=gate["approved"],minimum_fraction=.50)
                            np.testing.assert_array_equal(logits[~used],ztest[~used])
                            p=expit(logits)
                            risk_use_evidence[
                                f"device{device}_{case}_{stress}_{method}_seed{seed}_used"
                            ]=np.asarray(used,dtype=bool).copy()

                            predictions[device,case,stress,method,seed]=p
                            rows.append({
                                "device":device,"case":case,"stress":stress,
                                "method":method,"seed":seed,"metrics":metrics(y,p),
                                "source_use":float(used.mean()),
                                "measured_support":float((source["support"]>0).mean()),
                                "gate_approved":bool(gate["approved"])})
                            if (case=="healthy_energy" and stress=="MCAR20"
                                    and method=="V46_PAIRED_CONCORDANCE"):
                                audits.append({"device":device,"seed":seed,
                                               "native_graph":check_native_source_graph(
                                                   native,head,gamma,gate["approved"],c,logits)})
                for fault in ("none","off_drift"):
                    for method in CONCEPT_METHODS:
                        gamma,gate=gate_cache[fault,method]
                        gates.append({"device":device,"seed":seed,"law":law,
                                      "VALID2_fault":fault,"method":method,
                                      "TRAIN_gamma":gamma,"dual_VALID_gate":gate})
            if sha_state(native)!=sha:
                raise AssertionError("Source modifier altered frozen original K2")
            print(json.dumps({"event":"V46_NATIVE_FIT","device":device,"seed":seed}),flush=True)
        print(json.dumps({"event":"V46_DEVICE_COMPLETE","device":device,
                          "rows":len(rows)}),flush=True)
    for device in devices:
        for idx,(case,stress,method_a,method_b) in enumerate(COMPARE):
            y=test_labels[device,case]
            def preds(method):
                seeds=nn_seeds if method in NATIVE_METHODS else (None,)
                matrix=np.stack([predictions[device,case,stress,method,s] for s in seeds])
                return (np.repeat(matrix,len(nn_seeds),axis=0)
                        if len(seeds)==1 and len(nn_seeds)>1 else matrix)
            paired.append({"device":device,"case":case,"stress":stress,
                           "a":method_a,"b":method_b,
                           **paired_test_ap_difference(y,preds(method_a),preds(method_b),
                               seed=20264601+device%101*100+idx,resamples=bootstrap)})
    for device in devices:
        for case,_,_,_ in SCENARIOS:
            for stress in STRESS:
                for method in METHODS:
                    subset=[r for r in rows if r["device"]==device and
                            r["case"]==case and r["stress"]==stress and r["method"]==method]
                    expected=len(nn_seeds) if method in NATIVE_METHODS else 1
                    if len(subset)!=expected:
                        raise AssertionError("Incomplete experiment table")
                    summary.append({
                        "device":device,"case":case,"stress":stress,"method":method,
                        "AUPRC":float(np.mean([r["metrics"]["auprc"] for r in subset])),
                        "NLL":float(np.mean([r["metrics"]["nll"] for r in subset])),
                        "source_use":float(np.mean([r["source_use"] for r in subset])),
                        "measured_support":float(np.mean([
                            r.get("measured_support",0.) for r in subset]))})
    result={
        "status":"V46_SMOKE" if smoke else "V46_EXECUTED_NO_POST_TEST_TUNING",
        "frozen_protocol":"docs/medical/CONCEPTFAN_V46_FROZEN_PAIRED_CONDITION_PROTOCOL_20261106.md",
        "device_seeds":list(devices),"n_devices":len(devices),
        "n_real_native_fits":len(trained_models),
        "n_train":ntr,"n_valid1":nva//2,"n_valid2":nva//2,"n_test":nte,
        "metrics":summary,"test_rows":rows,"gates":gates,
        "source_fits_and_graph_audits":audits,"off_guard_TRAIN_VALID_receipts":off_guard_receipts,"paired_condition_TRAIN_VALID_receipts":paired_guard_receipts,"paired_AP_bootstrap":paired,
        "native_training":trained_models,
        "n_OFF_monitor_condition_receipts":len(monitor_evidence),
        "n_actual_source_use_decision_arrays":len(risk_use_evidence),
        "measured_OFF_dynamic_energy_gate_is_not_physical_certification":True,
        "paired_condition_is_not_physical_certification":True,
        "physical_source_identity_certified":False,
        "general_predictor_superiority_claim":False,
    }
    out.mkdir(parents=True,exist_ok=False)
    for name,data in (("STATUS.json",result),("SUMMARY.json",summary),
                      ("TEST_ROWS.json",rows),("PAIRED_CIS.json",paired),
                      ("GATES.json",gates)):
        (out/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))
    archive={f"device{d}_{c}_{s}_{m}_seed{seed}":p
             for (d,c,s,m,seed),p in predictions.items()}
    for (device,case),y in test_labels.items():
        archive[f"device{device}_{case}_TEST_Y"]=y
    np.savez_compressed(out/"TEST_PREDICTIONS.npz",**archive)
    np.savez_compressed(
        out/"TEST_PAIRED_CONDITION_SOURCE_RECEIPTS.npz",
        **monitor_evidence,**risk_use_evidence,
    )

    checkpoint_dir=out/"NATIVE_CHECKPOINTS"
    checkpoint_dir.mkdir()
    for ckpt in native_checkpoints:
        torch.save(ckpt,checkpoint_dir/
                   f"native_K2_device{ckpt['device']}_seed{ckpt['seed']}.pt")
    if len(list(checkpoint_dir.glob("*.pt")))!=len(trained_models):
        raise AssertionError("Incomplete frozen native K2 evidence checkpoints")
    print(json.dumps({"event":"V46_COMPLETE","n_rows":len(rows),
                      "healthy_MCAR40":[r for r in summary
                         if r["case"]=="healthy_energy" and r["stress"]=="MCAR40"]}),
          flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args()
    run(args.out_dir,smoke=args.smoke)
