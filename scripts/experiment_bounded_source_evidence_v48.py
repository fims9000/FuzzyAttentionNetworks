"""V48 frozen experiment: bounded monotone source-evidence link with matched joint risk."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score, log_loss

from conceptfan_discovery.logodds_world_v48 import DEVICES_V48, fresh_device_v48
from conceptfan_discovery.bounded_source_evidence_v48 import (
    risk_concept_input_v48,describe_source_link_v48,
)
from conceptfan_discovery.anchored_joint_bottleneck_v47 import (
    AnchoredJointConceptBottleneckV47,
    run_joint_epochs_v47,predict_joint_v47,safe_joint_logits_v47,
    known_measured_proxy_v47,
)
from conceptfan_discovery.adaptive_pair_bench_world_v41 import (
    render_observed_v41,masks_v41,measured_difference_v41,
)
from conceptfan_discovery.threeview_centered_v37 import fit_centered_threeview_v37
from conceptfan_discovery.partial_view_concept_v44 import (
    fit_partial_view_concept_v44,partial_view_concept_v44,
)
from conceptfan_discovery.matched_raw_baselines_v40 import (
    measured_temporal_quadratic_features_v40,fit_registered_raw_v40,
)
from conceptfan_discovery.cross_environment_transport import paired_test_ap_difference
from conceptfan_discovery.isolated_source_admission_v28 import train_source_coefficient
from scripts.experiment_adaptive_pair_v41 import _source_head_admission,_missing_source
from scripts.experiment_nonlinear_anchor_repair_v30 import make_firewalled_k2
from scripts.experiment_automatic_measured_concepts import (
    train_masked_measured_known_concepts,logit_from_head,
)
from scripts.experiment_frozen_concept_head_refit_v26 import refit_linear_concept_head
from scripts.experiment_trained_conceptfan_omission_v25 import predict_native,sha_state,metrics

METHODS=("K2","V44_FROZEN","V47_JOINT_LINEAR","V48_JOINT_LOGODDS",
         "RAW_DIFF_HGB_TRAIN1000","RAW_DIFF_HGB_ALL1800")
NATIVE_METHODS=METHODS[:4]
RAW_METHODS=METHODS[4:]
JOINT_METHODS=("V47_JOINT_LINEAR","V48_JOINT_LOGODDS")
STRESS=("MCAR20","MCAR40","MCAR50","missing_reference")
CASES=(
    ("healthy_energy","none"),
    ("common_U8_matched","common8"),
    ("unknown_OFF_drift","off_drift"),
    ("OFF_source_leak","leak"),
)
COMPARE=(
    ("healthy_energy","MCAR40","V48_JOINT_LOGODDS","V47_JOINT_LINEAR"),
    ("healthy_energy","MCAR40","V48_JOINT_LOGODDS","V44_FROZEN"),
    ("healthy_energy","MCAR40","V48_JOINT_LOGODDS","RAW_DIFF_HGB_TRAIN1000"),
    ("healthy_energy","MCAR40","V48_JOINT_LOGODDS","RAW_DIFF_HGB_ALL1800"),
    ("unknown_OFF_drift","MCAR40","V48_JOINT_LOGODDS","K2"),
)

def source_case(on,off,m_on,m_off,v44_fit):
    diff,observed=measured_difference_v41(on,off,m_on,m_off)
    n=len(on)
    source=(partial_view_concept_v44(v44_fit,diff,observed) if v44_fit is not None
            else _missing_source(n,"NO_SOURCE_IDENTIFIED_ON_TRAIN"))
    return {
        "on":on,"mask_on":m_on,
        "q":np.asarray(source["q"],dtype=np.float64),
        "support":np.asarray(source["support"],dtype=bool),
        "raw":measured_temporal_quadratic_features_v40(
            on,m_on,diff,observed,use_reference=True),
    }


@torch.no_grad()
def proxy_mse_native_v47(native,x,m):
    native.eval()
    total=0.;count=0.
    for start in range(0,len(x),256):
        xx=torch.from_numpy(np.asarray(x[start:start+256,:,:4],dtype=np.float32))
        mm=torch.from_numpy(np.asarray(m[start:start+256,:,:4],dtype=bool))
        masked=torch.where(mm,xx,torch.zeros_like(xx))
        output=native(torch.cat((masked,mm.to(xx.dtype)),dim=-1))
        targets,support=known_measured_proxy_v47(xx,mm)
        total+=float((((output.concept_trajectories-targets)**2)*
                      support.to(xx.dtype)).sum())
        count+=int(support.sum())
    return total/max(1,count)


@torch.no_grad()
def proxy_mse_joint_v47(model,x,m,q,support):
    model.eval()
    total=0.;count=0.
    for start in range(0,len(x),256):
        xx=torch.from_numpy(np.asarray(x[start:start+256,:,:4],dtype=np.float32))
        mm=torch.from_numpy(np.asarray(m[start:start+256,:,:4],dtype=bool))
        qq=torch.from_numpy(np.asarray(q[start:start+256],dtype=np.float32))
        ss=torch.from_numpy(np.asarray(support[start:start+256],dtype=bool))
        output=model(xx,mm,qq,ss)
        targets,available=known_measured_proxy_v47(xx,mm)
        total+=float((((output.concept_trajectories-targets)**2)*
                      available.to(xx.dtype)).sum())
        count+=int(available.sum())
    return total/max(1,count)


def _validate_joint_candidate(base1,cand1,y1,support1,base_mse1,new_mse1,
                              base2,cand2,y2,support2,base_mse2,new_mse2):
    """Both disjoint VALID400/VALID150 must support same prespecified gate."""
    receptions=[]
    for name,base,cand,y,support,baseline_mse,candidate_mse in (
        ("VALID1",base1,cand1,y1,support1,base_mse1,new_mse1),
        ("VALID2",base2,cand2,y2,support2,base_mse2,new_mse2),
    ):
        attempted,_=safe_joint_logits_v47(base,cand,support,approved=True)
        y=np.asarray(y,dtype=np.int64)
        # Labels only in held-out VALID, never TEST.
        nll_gain=log_loss(y,expit(base),labels=[0,1])-log_loss(y,expit(attempted),labels=[0,1])
        ap_gain=average_precision_score(y,expit(attempted))-average_precision_score(y,expit(base))
        mse_ratio=float(candidate_mse/max(1e-8,baseline_mse))
        adequate=(float(np.mean(support))>=.25 and nll_gain>=-.002
                  and ap_gain>=-.02 and mse_ratio<=1.25)
        receptions.append({
            "cohort":name,"n_valid_labels":len(y),
            "source_measured_support":float(np.mean(support)),
            "nll_gain_vs_original_K2":float(nll_gain),
            "AP_gain_vs_original_K2":float(ap_gain),
            "measured_known_proxy_mse_base":float(baseline_mse),
            "measured_known_proxy_mse_joint":float(candidate_mse),
            "measured_proxy_mse_ratio":mse_ratio,
            "passed":bool(adequate),
        })
    return {"approved":all(x["passed"] for x in receptions),
            "cohorts":receptions,
            "no_TEST_labels_used_for_candidate":True,
            "concept_proxy_is_not_physical_truth":True}


def _float_metrics(y,p):
    return metrics(np.asarray(y,dtype=np.int64),np.asarray(p,dtype=np.float64))


def run(out:Path,*,smoke:bool=False):
    if out.exists(): raise FileExistsError("Frozen V47 evidence path exists; never overwrite")
    torch.set_num_threads(2)
    devices=(20262707,) if smoke else tuple(DEVICES_V48)
    nn_seeds=(11,) if smoke else (11,23,37)
    ntr,nva,nte=(360,300,240) if smoke else (1000,800,600)
    n_boot=199 if smoke else 399
    summary=[];test_rows=[];gates=[];source_audits=[];training=[]
    ckpts=[];predictions={};y_tests={};stability={};explanations=[];use_flags={};sensor_support={};known_concept_arrays={};risk_evidence={}
    for device in devices:
        world=fresh_device_v48(device_seed=device,n_train=ntr,n_valid=nva,n_test=nte)
        tr,v1,v2,te=(world[k] for k in ("train","valid1","valid2","test"))
        y=world["outcomes"]["energy"]
        ontr,offtr=render_observed_v41(world,tr,fault="none",seed=20264711+device%101)
        mtr_on,mtr_off=masks_v41(ontr,offtr,"MCAR20",device_seed=device,mask_seed=20264721)
        dtr,dmask=measured_difference_v41(ontr,offtr,mtr_on,mtr_off)
        try:
            v37=fit_centered_threeview_v37(
                dtr,dmask,independent_view_noise_documented=True,
                seed=20264731+device%101,bootstrap=24)
            src_fit=fit_partial_view_concept_v44(v37,dtr,dmask)
            error=None
        except (ValueError,AssertionError) as exc:
            v37=None;src_fit=None;error=str(exc)
        train=source_case(ontr,offtr,mtr_on,mtr_off,src_fit)
        on1,off1=render_observed_v41(world,v1,fault="none",seed=20264712+device%101)
        m1_on,m1_off=masks_v41(on1,off1,"MCAR20",device_seed=device,mask_seed=20264722)
        valid1=source_case(on1,off1,m1_on,m1_off,src_fit)
        on2,off2=render_observed_v41(world,v2,fault="none",seed=20264713+device%101)
        m2_on,m2_off=masks_v41(on2,off2,"MCAR20",device_seed=device,mask_seed=20264723)
        valid2=source_case(on2,off2,m2_on,m2_off,src_fit)
        shared_mask=(np.random.default_rng(20264744+device%101)
                     .random((len(te),16,4))>=.20)
        cases={}
        for i,(name,fault) in enumerate(CASES):
            on,off=render_observed_v41(world,te,fault=fault,seed=20264751+i+device%101)
            y_tests[device,name]=y[te]
            for si,stress in enumerate(STRESS):
                mask_on,mask_off=masks_v41(
                    on,off,stress,device_seed=device,mask_seed=20264761+si,
                    known_TEST_first4=shared_mask)
                if stress=="missing_reference":
                    mask_off[:,:,4:10]=False
                cases[name,stress]=source_case(on,off,mask_on,mask_off,src_fit)
                sensor_support[f"device{device}_{name}_{stress}_measured_source"]=( 
                    cases[name,stress]["support"].copy())
                # Store the full prospectively defined q->risk evidence for every
                # TEST subject; labels are never used for these transformations.
                risk_evidence[f"device{device}_{name}_{stress}_source_q"]=(
                    cases[name,stress]["q"].copy())
                for joint_method in JOINT_METHODS:
                    link=("linear_membership" if joint_method=="V47_JOINT_LINEAR"
                          else "bounded_logodds")
                    raw_q=cases[name,stress]["q"]
                    risk_evidence[
                        f"device{device}_{name}_{stress}_{joint_method}_risk_score"
                    ]=(risk_concept_input_v48(raw_q,link=link)-.5).copy()


        source_audits.append({
            "device":device,"family":world["family"],"TRAIN_source_fit_failure":error,
            "TRAIN_V37_audit":v37.audit() if v37 is not None else None,
            "TRAIN_V44_audit":src_fit.audit() if src_fit is not None else None,
            "physical_source_identified":False,
        })
        raw={}
        for method in RAW_METHODS:
            xx=train["raw"];yy=y[tr]
            if method.endswith("ALL1800"):
                xx=np.vstack((xx,valid1["raw"],valid2["raw"]))
                yy=np.concatenate((yy,y[v1],y[v2]))
            raw[method]=fit_registered_raw_v40(method,xx,yy)
        for name,_ in CASES:
            for stress in STRESS:
                condition=cases[name,stress]
                for method in RAW_METHODS:
                    p=raw[method].predict_proba(condition["raw"])[:,1]
                    predictions[device,name,stress,method,None]=p
                    test_rows.append({
                        "device":device,"case":name,"stress":stress,"method":method,
                        "seed":None,"metrics":_float_metrics(y[te],p),
                        "source_use":0.,"n_outcome_train":ntr+nva if method.endswith("ALL1800") else ntr,
                    })
        for seed in nn_seeds:
            native=make_firewalled_k2(seed)
            pretrain=train_masked_measured_known_concepts(
                native,ontr[:,:,:4],mtr_on[:,:,:4],seed=seed,
                epochs=2 if smoke else 10)
            native_sha=sha_state(native)
            qtr=predict_native(native,ontr[:,:,:4],mtr_on[:,:,:4])["concept_summary"]
            q1=predict_native(native,on1[:,:,:4],m1_on[:,:,:4])["concept_summary"]
            q2=predict_native(native,on2[:,:,:4],m2_on[:,:,:4])["concept_summary"]
            qtest=predict_native(
                native,cases["healthy_energy","MCAR20"]["on"][:,:,:4],shared_mask
            )["concept_summary"]
            head=refit_linear_concept_head(qtr,y[tr])
            ztr=logit_from_head(head,qtr)
            z1=logit_from_head(head,q1)
            z2=logit_from_head(head,q2)
            ztest=logit_from_head(head,qtest)
            source_dict_train={"V44_FROZEN":{"q":train["q"],"support":train["support"].astype(float)}}
            source_dict_v1={"V44_FROZEN":{"q":valid1["q"],"support":valid1["support"].astype(float)}}
            source_dict_v2={"V44_FROZEN":{"q":valid2["q"],"support":valid2["support"].astype(float)}}
            gamma44,gate44=_source_head_admission(
                "V44_FROZEN",source_dict_train,source_dict_v1,source_dict_v2,
                y=y,train=tr,valid1=v1,valid2=v2,
                base_train=ztr,base_v1=z1,base_v2=z2,
                bootstrap=n_boot,seed=20264781+device%101*100+seed)
            # TRAIN gamma was estimated before any held-out VALID outcome checks.
            if not np.any(train["support"]):
                gamma44=0.
            base_mse1=proxy_mse_native_v47(native,on1,m1_on)
            base_mse2=proxy_mse_native_v47(native,on2,m2_on)
            gates.append({"device":device,"seed":seed,"method":"V44_FROZEN",
                          "approved":bool(gate44["approved"]),
                          "receipt":gate44,"train_gamma":float(gamma44)})
            fitted={}
            for mi,method in enumerate(JOINT_METHODS):
                link=("linear_membership" if method=="V47_JOINT_LINEAR"
                      else "bounded_logodds")
                # Same native K2 pretrain, 1000 TRAIN Y and 8 joint epochs,
                # with only a different deterministic measured membership link.
                qtrain=risk_concept_input_v48(train["q"],link=link)
                qv1=risk_concept_input_v48(valid1["q"],link=link)
                qv2=risk_concept_input_v48(valid2["q"],link=link)
                model=AnchoredJointConceptBottleneckV47(
                    native,np.asarray(head.weights),head.bias,
                    initial_gamma=float(gamma44)*(1. if method=="V47_JOINT_LINEAR" else 2.))
                joint_receipt=run_joint_epochs_v47(
                    model,ontr[:,:,:4],mtr_on[:,:,:4],qtrain,train["support"],y[tr],
                    seed=seed+mi*101,consistency_weight=0.,
                    epochs=2 if smoke else 8,batch_size=128,proxy_weight=5.)
                joint_receipt["registered_source_risk_link"]=describe_source_link_v48(link)
                outv1=predict_joint_v47(
                    model,on1[:,:,:4],m1_on[:,:,:4],qv1,valid1["support"])
                outv2=predict_joint_v47(
                    model,on2[:,:,:4],m2_on[:,:,:4],qv2,valid2["support"])
                gate=_validate_joint_candidate(
                    z1,outv1["logits"],y[v1],valid1["support"],base_mse1,
                    proxy_mse_joint_v47(model,on1,m1_on,qv1,valid1["support"]),
                    z2,outv2["logits"],y[v2],valid2["support"],base_mse2,
                    proxy_mse_joint_v47(model,on2,m2_on,qv2,valid2["support"]))
                gates.append({"device":device,"seed":seed,"method":method,
                              "approved":bool(gate["approved"]),"receipt":gate})
                fitted[method]=(model,gate)
                ckpts.append({
                    "device":device,"seed":seed,"method":method,
                    "pretrain_K2_SHA":native_sha,
                    "joint_state_dict":{
                        k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                })
                training.append({"device":device,"seed":seed,"method":method,
                                 "pretrained_encoder_SHA":native_sha,
                                 "pretrain":pretrain,"joint":joint_receipt})
            ckpts.append({
                "device":device,"seed":seed,"method":"K2_PRETRAINED",
                "pretrain_K2_SHA":native_sha,
                "joint_state_dict":{
                    k:v.detach().cpu().clone() for k,v in native.state_dict().items()},
            })
            if sha_state(native)!=native_sha:
                raise AssertionError("Joint training modified original frozen K2 model")
            for name,_ in CASES:
                for stress in STRESS:
                    c=cases[name,stress]
                    basep=expit(ztest)
                    predictions[device,name,stress,"K2",seed]=basep
                    test_rows.append({"device":device,"case":name,"stress":stress,
                                      "method":"K2","seed":seed,
                                      "source_use":0.,"metrics":_float_metrics(y[te],basep),
                                      "n_outcome_train":ntr})
                    z44,used44=safe_joint_logits_v47(
                        ztest,ztest+gamma44*(c["q"]-.5)*c["support"],
                        c["support"],approved=gate44["approved"])
                    use_flags[f"device{device}_{name}_{stress}_V44_FROZEN_seed{seed}_used"]=used44.copy()
                    p44=expit(z44)
                    predictions[device,name,stress,"V44_FROZEN",seed]=p44
                    test_rows.append({"device":device,"case":name,"stress":stress,
                                      "method":"V44_FROZEN","seed":seed,
                                      "source_use":float(used44.mean()),
                                      "metrics":_float_metrics(y[te],p44),
                                      "n_outcome_train":ntr})
                    for method,(model,gate) in fitted.items():
                        link=("linear_membership" if method=="V47_JOINT_LINEAR"
                              else "bounded_logodds")
                        effective_q=risk_concept_input_v48(c["q"],link=link)
                        jr=predict_joint_v47(
                            model,c["on"][:,:,:4],c["mask_on"][:,:,:4],
                            effective_q,c["support"])
                        logit,used=safe_joint_logits_v47(
                            ztest,jr["logits"],c["support"],approved=gate["approved"])
                        if not np.array_equal(logit[~used],ztest[~used]):
                            raise AssertionError("Broken exact K2 fallback for unsupported source")
                        use_flags[f"device{device}_{name}_{stress}_{method}_seed{seed}_used"]=used.copy()
                        p=expit(logit)
                        predictions[device,name,stress,method,seed]=p
                        test_rows.append({
                            "device":device,"case":name,"stress":stress,
                            "method":method,"seed":seed,
                            "source_use":float(used.mean()),
                            "gate_approved":bool(gate["approved"]),
                            "metrics":_float_metrics(y[te],p),
                            "n_outcome_train":ntr,
                            "max_exact_additivity_error":jr["max_additivity_error"],
                        })
                        if name=="healthy_energy" and stress=="MCAR40":
                            stability[device,seed,method]=jr["known_concepts"]
                            known_concept_arrays[
                                f"device{device}_healthy_MCAR40_{method}_seed{seed}_known"
                            ]=jr["known_concepts"].copy()
                        if (name=="healthy_energy" and stress=="MCAR20"):
                            explanations.append({
                                "device":device,"seed":seed,"method":method,
                                "registered_source_risk_link":link,
                                "example0_original_measured_membership":float(c["q"][0]),
                                "example0_additive_risk_source_evidence":float(effective_q[0]-.5),
                                "joint_signed_concept_head":{
                                    "bias":float(model.bias.detach()),
                                    "w1":float(model.known_weight[0].detach()),
                                    "w2":float(model.known_weight[1].detach()),
                                    "gamma":float(model.gamma.detach()),
                                },
                                "gate_approved":bool(gate["approved"]),
                                "is_measured_proxy_not_physical_truth":True,
                                "input_firewall_only_4_known_sensors_to_transformer":True,
                                "new_source_from_observed_pair_only":True,
                            })
            print(json.dumps({
                "event":"V48_NATIVE_K2_AND_JOINT_TRAINED",
                "device":device,"seed":seed,
                "pretrained_K2_SHA":native_sha,
            }),flush=True)
        print(json.dumps({"event":"V48_DEVICE_COMPLETE","device":device,
                          "n_test_rows":len(test_rows)}),flush=True)

    paired=[]
    for device in devices:
        for i,(name,stress,a,b) in enumerate(COMPARE):
            yy=y_tests[device,name]
            def method_predictions(method):
                seeds=nn_seeds if method in NATIVE_METHODS else (None,)
                a=np.stack([predictions[device,name,stress,method,s] for s in seeds])
                return np.repeat(a,len(nn_seeds),axis=0) if len(seeds)==1 and len(nn_seeds)>1 else a
            paired.append({
                "device":device,"case":name,"stress":stress,"a":a,"b":b,
                **paired_test_ap_difference(
                    yy,method_predictions(a),method_predictions(b),
                    seed=20264821+device%101*100+i,resamples=n_boot),
            })
    stability_report=[]
    for device in devices:
        for method in JOINT_METHODS:
            vectors=[stability[device,seed,method] for seed in nn_seeds]
            correlations=[]
            for i in range(len(vectors)):
                for j in range(i+1,len(vectors)):
                    correlations.extend([
                        float(np.corrcoef(vectors[i][:,k],vectors[j][:,k])[0,1])
                        for k in range(2)
                    ])
            stability_report.append({
                "device":device,"method":method,
                "n_trained_initializations":len(nn_seeds),
                "known_concept_cross_seed_pearson_mean":(
                    float(np.mean(correlations)) if correlations else None),
                "known_concept_cross_seed_pearson_min":(
                    float(np.min(correlations)) if correlations else None),
                "not_independent_physical_semantics":True,
            })
    for device in devices:
        for name,_ in CASES:
            for stress in STRESS:
                for method in METHODS:
                    group=[r for r in test_rows if
                           (r["device"],r["case"],r["stress"],r["method"])==
                           (device,name,stress,method)]
                    expected=len(nn_seeds) if method in NATIVE_METHODS else 1
                    if len(group)!=expected:
                        raise AssertionError("Incomplete fixed TEST cohort")
                    summary.append({
                        "device":device,"case":name,"stress":stress,"method":method,
                        "AUPRC":float(np.mean([v["metrics"]["auprc"] for v in group])),
                        "NLL":float(np.mean([v["metrics"]["nll"] for v in group])),
                        "source_use":float(np.mean([v["source_use"] for v in group])),
                    })
    result={
        "status":"V48_SMOKE" if smoke else "V48_BOUND_LOGODDS_EVIDENCE_EXECUTED",
        "protocol_frozen_before_test":"docs/medical/CONCEPTFAN_V48_FROZEN_LOGODDS_EVIDENCE_PROTOCOL_20261106.md",
        "device_seeds":list(devices),
        "n_train":ntr,"n_valid1":nva//2,"n_valid2":nva//2,"n_test":nte,
        "native_seeds":list(nn_seeds),
        "methods":list(METHODS),"stress":list(STRESS),"cases":[x[0] for x in CASES],
        "n_real_pretrained_native_fits":len(devices)*len(nn_seeds),
        "n_real_joint_fits":len(training),
        "n_saved_source_support_arrays":len(sensor_support),
        "n_saved_decision_use_arrays":len(use_flags),
        "n_saved_known_concept_arrays":len(known_concept_arrays),
        "n_saved_bounded_source_evidence_arrays":len(risk_evidence),
        "risk_evidence_includes_untouched_outcome_blind_original_q":True,
        "training":training,"TEST_rows":test_rows,"TEST_summary":summary,
        "dual_VALID_gate_receipts":gates,"source_fits":source_audits,
        "stability_without_y_labels":stability_report,
        "explanation_samples":explanations,
        "prespecified_paired_AP_intervals":paired,
        "n_prespecified_paired_CIs":len(paired),
        "true_physical_identity_claimed":False,
        "train_known_proxy_only_not_latent_oracle":True,
        "no_raw_input_direct_risk_bypass":True,
        "no_TEST_label_used_for_training_or_admission":True,
    }
    out.mkdir(parents=True,exist_ok=False)
    for name,obj in (
        ("STATUS.json",result),("TEST_ROWS.json",test_rows),("SUMMARY.json",summary),
        ("DUAL_VALID_GATES.json",gates),("PAIRED_CIS.json",paired),
        ("SOURCE_FIT_RECEIPTS.json",source_audits),
        ("CONCEPT_STABILITY.json",stability_report),("EXPLANATIONS_RU.json",explanations)):
        (out/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))
    freezed={f"device{dev}_{name}_{stress}_{method}_seed{seed}":v
             for (dev,name,stress,method,seed),v in predictions.items()}
    for (dev,name),label in y_tests.items():
        freezed[f"device{dev}_{name}_TEST_Y"]=label
    np.savez_compressed(out/"FROZEN_TEST_PREDICTIONS.npz",**freezed)
    np.savez_compressed(
        out/"FROZEN_SOURCE_AND_USE_FLAGS.npz",**sensor_support,**use_flags,
    )
    np.savez_compressed(
        out/"FROZEN_RISK_EVIDENCE.npz",**risk_evidence,
    )
    if known_concept_arrays:
        np.savez_compressed(out/"FROZEN_KNOWN_CONCEPTS.npz",**known_concept_arrays)

    ckptdir=out/"CHECKPOINTS";ckptdir.mkdir()
    for item in ckpts:
        filename=f"native_device{item['device']}_seed{item['seed']}_{item['method']}.pt"
        torch.save(item,ckptdir/filename)
    if len(list(ckptdir.glob("*.pt")))!=len(ckpts):
        raise AssertionError("Missing true trained Transformer weights")
    print(json.dumps({
        "event":"V48_COMPLETED","status":result["status"],
        "n_native_models":len(training),
        "healthy_MCAR40":[v for v in summary
           if v["case"]=="healthy_energy" and v["stress"]=="MCAR40"],
        "unknown_OFF_drift_MCAR40":[v for v in summary
           if v["case"]=="unknown_OFF_drift" and v["stress"]=="MCAR40"],
    }),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    run(a.out_dir,smoke=a.smoke)
