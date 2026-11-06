"""V34: real Temporal ConceptFAN + unsupervised conditional flow + fuzzy orders.

Two DIFFERENT previously registered synthetic simulator families; one fresh
subject draw per family. Real K2 Temporal Transformer is trained on
measured-pair proxies, then frozen. TRAIN-only label-free source grouping,
masked WLS temporal template and conditional rectified flow; limited fuzzy
alpha-plane membership, TRAIN-only gamma, 2-disjoint-VALID source admission,
full unchanged TEST scoring (including harmful shifts and source outages).

The synthetic reference-device phantom anchors Type3 boundary uncertainty.
No known reference in actual deployment => cannot claim a Type3 verified
instrument calibration. Not a full diffusion model, full IT3 FLS, or
demonstration of clinical concept semantics; strong ordinary baselines included.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conceptfan_discovery.adaptive_higher_order_concepts_v34 import (
    fit_measured_definition,source_membership,AdditiveFuzzySourceSlot,
    select_order_from_train,
)
from conceptfan_discovery.native_adaptive_type3_transformer_v34 import (
    NativeAuditedFuzzyConceptFANV34,readable_source_explanation,
    render_concept_explanation_ru,
)
from conceptfan_discovery.conditional_source_flow_v34 import (
    train_conditional_measured_source_flow,flow_amplitude_samples,
)
from conceptfan_discovery.independent_device_anchor_v34 import (
    synthetic_known_reference_training_only,
)
from conceptfan_discovery.phase_pulse_benchmark_v31 import (
    phase_pulse_world,temporal_sensor_features,
)
from conceptfan_discovery.auto_measured_concept_benchmark import (
    extended_nonlinear_world,
)
from conceptfan_discovery.synchronized_source_outages_v33 import (
    synchronized_outage_mask,
)
from conceptfan_discovery.auto_measured_concept_proposals import (
    form_train_measured_groups,
)
from conceptfan_discovery.masked_temporal_template import (
    fit_masked_rank1_template,project_masked_template,
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

FAMILIES=("pulse_order","nonlinear_switching")
METHODS=("FROZEN_K2","WLS_TYPE1","FLOW_TYPE1","FLOW_IT2",
         "FLOW_RESTRICTED_IT3","FLOW_AUTO","CBM_ALL2000",
         "RAW_TRAIN1200","RAW_ALL2000")
NATIVE_METHODS=METHODS[:7]
TEST_STRESS=("MCAR20","MCAR40","independent_block8",
             "synchronized_block8","synchronized_full16")
SCENARIOS=(
    ("healthy_physical","physical","benign","none"),
    ("known_physical_flip","physical","flip_physical","flip_physical"),
    ("UNSEEN_physical_flip","physical","benign","flip_physical"),
    ("complete_dictionary","complete","benign","none"),
    ("predictive_recording_artifact","artifact","benign","none"),
)
TRAIN_N,VALID_N,TEST_N=1200,800,600
SEEDS=(11,23,37)
SOURCE_FUZZY_ORDERS={
    "WLS_TYPE1":"type1",
    "FLOW_TYPE1":"type1",
    "FLOW_IT2":"interval_type2",
    "FLOW_RESTRICTED_IT3":"restricted_interval_type3",
    "FLOW_AUTO":"auto",
}


def _data(family,smoke):
    ntr,nval,nte=(300,300,240) if smoke else (TRAIN_N,VALID_N,TEST_N)
    if family=="pulse_order":
        d=phase_pulse_world(seed=20261126,n_train=ntr,n_valid=nval,n_test=nte)
        split=(d["train"],d["valid1"],d["valid2"],d["test"])
        mapping={"physical":"physical","complete":"complete","artifact":"artifact"}
    elif family=="nonlinear_switching":
        d=extended_nonlinear_world(
            seed=20261127,train_n=ntr,valid_n=nval,test_n=nte)
        split=(d["train"],d["valid_env1"],d["valid_env2"],d["test"])
        mapping={"physical":"physical_extra",
                 "complete":"complete_nuisance",
                 "artifact":"artifact_predictive"}
    else:
        raise ValueError("Unknown predeclared generator family")
    return d,split,mapping


def _device_intervention(x,mode):
    xx=np.asarray(x,dtype=np.float32).copy()
    if mode=="none":
        pass
    elif mode=="benign":
        xx[:,:,4:]*=1.10
        xx[:,:,4:]+=.05
    elif mode=="flip_physical":
        xx[:,:,4:7]*=-1.
    else:
        raise ValueError("Unsupported registered device mapping")
    np.testing.assert_array_equal(xx[:,:,:4],x[:,:,:4])
    return xx


def _test_mask(x,stress,seed):
    return synchronized_outage_mask(x,stress,seed=seed)


def _raw_logistic(x,m,y):
    model=make_pipeline(StandardScaler(),LogisticRegression(
        C=1.0,max_iter=1000))
    model.fit(temporal_sensor_features(x,m),y)
    return model


def _source_proxy(group,template,x,m):
    p=project_masked_template(template,group,x,m)
    return p["amplitude_estimate"],p["observed_template_information_fraction"],p


def _fuzzy_q(amplitude,sd,info,definition,mode):
    bundle=source_membership(
        torch.as_tensor(amplitude),torch.as_tensor(sd),
        torch.as_tensor(info),definition,order=mode)
    return bundle.center.detach().cpu().numpy().astype(np.float64),bundle


def _train_risk_gamma(z,q,y):
    """Same single signed TRAIN-only risk parameter for all fuzzy orders."""
    return float(train_source_coefficient(z,q,y))


def _all_wls_groups(groups,templates,x,m):
    return np.column_stack([
        _source_proxy(g,templates[g.candidate_id],x,m)[0]
        for g in groups
    ])


def _candidate_for_mode(
    name,group,template,flow,definition,x,m,
    *,
    draw_seed,
    generated=None,
):
    wls,info,raw=_source_proxy(group,template,x,m)
    if name=="WLS_TYPE1":
        amplitude=wls
        uncertainty=np.zeros_like(wls)
        sample_source="OBSERVED_ONLY_WLS"
    else:
        if generated is None:
            generated=flow_amplitude_samples(
                flow,template,group,x,m,
                seed=draw_seed,n_samples=48)
        amplitude=generated["amplitude_mean"]
        uncertainty=generated["amplitude_sd"]
        sample_source="SELF_SUPERVISED_GENERATED_AMPLITUDE_MEAN"
    order=SOURCE_FUZZY_ORDERS[name]
    q,bundle=_fuzzy_q(amplitude,uncertainty,info,definition,order)
    return {
        "q":q,
        "amplitude":np.asarray(amplitude,dtype=np.float32),
        "uncertainty":np.asarray(uncertainty,dtype=np.float32),
        "info":np.asarray(info,dtype=np.float32),
        "bundle":bundle,
        "order":bundle.selected_order,
        "source_origin":sample_source,
    }


def _fit_one_selected_source(
    groups,templates,xtrain,mtrain,ztrain,ytrain,
):
    """Only TRAIN labels choose which of label-free candidates is relevant.

    The selected GROUP is shared by all five order variants. No true source
    identity, VALID or TEST is used in this step. This is deliberately a
    TRAIN-label source screen rather than pretending causal discovery.
    """
    winner=None
    for g in groups:
        wls,info,_=_source_proxy(g,templates[g.candidate_id],xtrain,mtrain)
        train_q=expit(wls)*info+.5*(1-info)
        gamma=_train_risk_gamma(ztrain,train_q,ytrain)
        new=ztrain+gamma*(train_q-.5)
        gain=float((np.logaddexp(0,ztrain)-ytrain*ztrain
                    -np.logaddexp(0,new)+ytrain*new).mean())
        item={"group_id":g.candidate_id,"gain_train":gain,
              "group_sensors":g.sensor_indices}
        if winner is None or gain>winner[0]:
            winner=(gain,g,item)
    if winner is None:
        raise RuntimeError("No unlabeled candidate source group")
    return winner[1],winner[2]


def _fit_group_flow_cached(cache,group,template,xtrain,mtrain,seed,smoke):
    if group.candidate_id not in cache:
        cache[group.candidate_id]=train_conditional_measured_source_flow(
            template,group,xtrain,mtrain,seed=seed,
            steps=20 if smoke else 220)
    return cache[group.candidate_id]


def _summarize(rows,family,case,stress,method,fit_count):
    records=[r for r in rows
             if (r["family"],r["case"],r["stress"],r["method"])==
                 (family,case,stress,method)]
    expected=fit_count if method in NATIVE_METHODS else 1
    if len(records)!=expected:
        raise AssertionError(f"Wrong repetition count for {family}/{case}/{stress}/{method}")
    return {
        "family":family,"case":case,"stress":stress,"method":method,
        "n_native_fits":expected if method in NATIVE_METHODS else 0,
        "AUPRC":float(np.mean([r["metric"]["auprc"] for r in records])),
        "AUROC":float(np.mean([r["metric"]["auroc"] for r in records])),
        "NLL":float(np.mean([r["metric"]["nll"] for r in records])),
        "Brier":float(np.mean([r["metric"]["brier"] for r in records])),
        "source_use":float(np.mean([r.get("source_use",0.) for r in records])),
    }


def run(output:Path,*,smoke:bool=False):
    if output.exists():
        raise FileExistsError("Do not overwrite prior registered research evidence")
    torch.set_num_threads(2)
    seeds=(11,) if smoke else SEEDS
    epochs=2 if smoke else 10
    bootstrap=199 if smoke else 399
    all_rows,all_gates,all_selections,all_training=[],[],[],[]
    explanations,flow_receipts,alpha_examples=[],[],[]
    source_proposals,anchor_records={},{}
    test_predictions,test_labels={},{}
    saved_native_states,saved_flow_states=[],[]
    CI_CASES=("healthy_physical","UNSEEN_physical_flip")
    CI_STRESSES=("MCAR20","synchronized_block8")
    CI_METHODS=(
        "FROZEN_K2","WLS_TYPE1","FLOW_TYPE1","FLOW_IT2",
        "FLOW_RESTRICTED_IT3","FLOW_AUTO",
    )
    for family_index,family in enumerate(
            FAMILIES[:1] if smoke else FAMILIES):
        data,(tr,v1,v2,te),label_map=_data(family,smoke)
        x=data["x"]
        mtr=_test_mask(x[tr],"MCAR20",20263401+family_index)
        mv1=_test_mask(x[v1],"MCAR20",20263402+family_index)
        mv2=_test_mask(x[v2],"MCAR20",20263403+family_index)
        proposed=form_train_measured_groups(
            x[tr],mtr,known_sensor_indices=(0,1,2,3),
            seed=20263404+family_index,min_pair_abs_corr=.60,
            min_joint_cells_per_fold=300,max_candidates=5)
        groups=proposed.candidates
        if len(groups)<2:
            raise AssertionError("TRAIN source detection failed, keep failure visible")
        source_proposals[family]=proposed.as_dict()
        templates={
            g.candidate_id:fit_masked_rank1_template(g,x[tr],mtr)
            for g in groups
        }
        flow_cache={}
        flow_draw_cache={}
        synthetic_reference=synthetic_known_reference_training_only(
            seed=20263420+family_index)
        # This traceable input is a SIMULATOR DEVICE PHANTOM, not medical truth
        # and not a source median shift or Y-labelled concept supervision.
        anchor_records[family]=asdict(synthetic_reference)
        stress_data={}
        for case_index,(case,law,val2_shift,test_shift) in enumerate(SCENARIOS):
            test_x=_device_intervention(x[te],test_shift)
            for stress_index,stress in enumerate(TEST_STRESS):
                mask=_test_mask(test_x,stress,20263900+stress_index)
                stress_data[(case,stress)]=(test_x,mask)
        for seed in seeds:
            core=make_firewalled_k2(seed)
            fitting=train_masked_measured_known_concepts(
                core,x[tr,:,:4],mtr[:,:,:4],seed=seed,epochs=epochs)
            core_sha=sha_state(core)
            saved_native_states.append({
                "family":family,"seed":seed,
                "state_dict":{k:v.detach().cpu().clone()
                    for k,v in core.state_dict().items()},
            })
            all_training.append({"family":family,"nn_seed":seed,
                                 "original_encoder_sha":core_sha,**fitting})
            qt=predict_native(core,x[tr,:,:4],mtr[:,:,:4])["concept_summary"]
            q1=predict_native(core,x[v1,:,:4],mv1[:,:,:4])["concept_summary"]
            q2=predict_native(core,x[v2,:,:4],mv2[:,:,:4])["concept_summary"]
            k2_test={}
            for stress_index,stress in enumerate(TEST_STRESS):
                tx,tm=stress_data[("healthy_physical",stress)]
                k2_test[stress]=predict_native(
                    core,tx[:,:,:4],tm[:,:,:4])["concept_summary"]
            for case_index,(case,law,val2_shift,test_shift) in enumerate(SCENARIOS):
                y=data["outcomes"][label_map[law]]
                if case in CI_CASES:
                    test_labels[(family,case)]=y[te].copy()
                head=refit_linear_concept_head(qt,y[tr])
                base_train=logit_from_head(head,qt)
                base_v1=logit_from_head(head,q1)
                base_v2=logit_from_head(head,q2)
                group,screen=_fit_one_selected_source(
                    groups,templates,x[tr],mtr,base_train,y[tr])
                template=templates[group.candidate_id]
                flow=_fit_group_flow_cached(
                    flow_cache,group,template,x[tr],mtr,
                    seed=20263440+family_index*17+len(flow_cache),
                    smoke=smoke)
                train_amp,train_info,_=_source_proxy(
                    group,template,x[tr],mtr)
                measured_sd=np.sqrt(
                    template.train_residual_variance/
                    np.maximum(1.,16.*train_info))
                definition=fit_measured_definition(
                    group.candidate_id,group.sensor_indices,
                    train_amp,measured_sd,
                    independently_calibrated_boundary_spread=
                        synthetic_reference.half_difference_boundary_uncertainty,
                    certified_environments=2,
                    trusted_external_anchor=True,
                )
                all_selections.append({
                    "family":family,"case":case,"nn_seed":seed,
                    "candidate_group_train_only":screen,
                    "registered_auto_order":select_order_from_train(definition),
                    "source_physical_semantics_verified":False,
                })
                shifts={
                    "train":(x[tr],mtr),
                    "val1":(x[v1],mv1),
                    "val2":(_device_intervention(x[v2],val2_shift),mv2),
                }
                samples_by_method={}
                for method in SOURCE_FUZZY_ORDERS:
                    fitted={}
                    for split_index,split in enumerate(("train","val1","val2")):
                        bx,bm=shifts[split]
                        draw_id=(case,split,group.candidate_id)
                        draw_seed=(20263490+
                            10000*family_index+1000*case_index+
                            split_index)
                        if (method!="WLS_TYPE1" and
                            draw_id not in flow_draw_cache):
                            flow_draw_cache[draw_id]=flow_amplitude_samples(
                                flow,template,group,bx,bm,
                                seed=draw_seed,n_samples=48)
                        fitted[split]=_candidate_for_mode(
                            method,group,template,flow,definition,bx,bm,
                            draw_seed=draw_seed,
                            generated=flow_draw_cache.get(draw_id),
                        )
                    gamma=_train_risk_gamma(
                        base_train,fitted["train"]["q"],y[tr])
                    g=evaluate_two_independent_environments(
                        method,gamma,
                        ("VALID1",y[v1],base_v1,
                         fitted["val1"]["q"],fitted["val1"]["info"]),
                        ("VALID2_SHIFTED",y[v2],base_v2,
                         fitted["val2"]["q"],fitted["val2"]["info"]),
                        train_label_count=len(tr),
                        min_mean_gain=.003,min_support_fraction=.75,
                        min_subject_support=.50,bootstrap_reps=bootstrap,
                        seed=20263500+case_index*100+seed,
                    )
                    fitted["gate"]=g
                    fitted["gamma"]=gamma
                    samples_by_method[method]=fitted
                    all_gates.append({
                        "family":family,"case":case,"nn_seed":seed,
                        "method":method,"candidate_group":group.candidate_id,
                        "fuzzy_order":fitted["train"]["order"],
                        "gate":g.as_dict(),
                    })
                train_all_groups=_all_wls_groups(
                    groups,templates,x[tr],mtr)
                val1_all_groups=_all_wls_groups(
                    groups,templates,x[v1],mv1)
                val2_all_groups=_all_wls_groups(
                    groups,templates,shifts["val2"][0],mv2)
                cbm_features=np.concatenate([
                    np.column_stack([qt,train_all_groups]),
                    np.column_stack([q1,val1_all_groups]),
                    np.column_stack([q2,val2_all_groups]),
                ])
                cbm_labels=np.concatenate([y[tr],y[v1],y[v2]])
                cbm=refit_linear_concept_head(cbm_features,cbm_labels)
                raw_fit=_raw_logistic(x[tr],mtr,y[tr])
                all_x=np.concatenate([x[tr],x[v1],shifts["val2"][0]])
                all_m=np.concatenate([mtr,mv1,mv2])
                raw_all=_raw_logistic(all_x,all_m,cbm_labels)
                if seed==seeds[0]:
                    for stress in TEST_STRESS:
                        tx,tm=stress_data[(case,stress)]
                        raw_feats=temporal_sensor_features(tx,tm)
                        for method,model in (
                            ("RAW_TRAIN1200",raw_fit),
                            ("RAW_ALL2000",raw_all),
                        ):
                            all_rows.append({
                                "family":family,"case":case,
                                "stress":stress,"method":method,
                                "nn_seed":None,
                                "metric":metrics(
                                    y[te],model.predict_proba(raw_feats)[:,1]),
                            })
                for stress_index,stress in enumerate(TEST_STRESS):
                    tx,tm=stress_data[(case,stress)]
                    qknown=k2_test[stress]
                    base=logit_from_head(head,qknown)
                    raw_group_features=_all_wls_groups(
                        groups,templates,tx,tm)
                    pred_cbm=expit(logit_from_head(
                        cbm,np.column_stack([qknown,raw_group_features])))
                    base_probability=expit(base)
                    if case in CI_CASES and stress in CI_STRESSES:
                        test_predictions[(family,case,stress,
                                          "FROZEN_K2",seed)]=base_probability.copy()
                    all_rows.append({
                        "family":family,"case":case,"stress":stress,
                        "method":"FROZEN_K2","nn_seed":seed,
                        "metric":metrics(y[te],base_probability),
                        "source_use":0.,
                    })
                    all_rows.append({
                        "family":family,"case":case,"stress":stress,
                        "method":"CBM_ALL2000","nn_seed":seed,
                        "metric":metrics(y[te],pred_cbm),
                        "source_use":1.,
                    })
                    draw_id=(case,stress,group.candidate_id)
                    draw_seed=(20263950+10000*family_index+
                        1000*case_index+stress_index)
                    for method in SOURCE_FUZZY_ORDERS:
                        rec=samples_by_method[method]
                        if (method!="WLS_TYPE1" and
                            draw_id not in flow_draw_cache):
                            flow_draw_cache[draw_id]=flow_amplitude_samples(
                                flow,template,group,tx,tm,
                                seed=draw_seed,n_samples=48)
                        source=_candidate_for_mode(
                            method,group,template,flow,definition,tx,tm,
                            draw_seed=draw_seed,
                            generated=flow_draw_cache.get(draw_id),
                        )
                        actual_slot=AdditiveFuzzySourceSlot(
                            definition,order=SOURCE_FUZZY_ORDERS[method])
                        with torch.no_grad():
                            actual_slot.gamma.fill_(rec["gamma"])
                        prediction=actual_slot(
                            torch.from_numpy(base.astype(np.float32)),
                            torch.from_numpy(source["amplitude"]),
                            torch.from_numpy(source["uncertainty"]),
                            torch.from_numpy(source["info"]),
                            approved=rec["gate"].approved)
                        z=prediction["logit"].detach().numpy()
                        used=prediction["source_used"].detach().numpy()
                        np.testing.assert_array_equal(
                            z[~used],base.astype(np.float32)[~used])
                        if not np.isfinite(z).all():
                            raise AssertionError("Source induced nonfinite TEST logits")
                        # REAL native Transformer + source slot in same
                        # torch forward for a distinct batch, compare logits.
                        if (case=="healthy_physical" and stress=="MCAR20"
                            and method=="FLOW_AUTO"):
                            wrapper=NativeAuditedFuzzyConceptFANV34(
                                core,np.asarray(head.weights),head.bias,
                                definition,fuzzy_order="auto")
                            with torch.no_grad():
                                wrapper.source_slot.gamma.fill_(rec["gamma"])
                            small=32
                            full_native=wrapper(
                                torch.from_numpy(tx[:small,:,:4]),
                                torch.from_numpy(tm[:small,:,:4]),
                                extra_source_amplitude=torch.from_numpy(
                                    source["amplitude"][:small]),
                                extra_source_uncertainty=torch.from_numpy(
                                    source["uncertainty"][:small]),
                                extra_source_observed_information=torch.from_numpy(
                                    source["info"][:small]),
                                extra_source_gate_approved=rec["gate"].approved,
                            )
                            if float(np.max(np.abs(
                                full_native["logit"].detach().numpy()-
                                z[:small])))>3e-5:
                                raise AssertionError("In-graph source differs from actual K2 baseline")
                            explanations.append({
                                "family":family,"case":case,"nn_seed":seed,
                                "source_group_sensors":list(group.sensor_indices),
                                "native_encoder_hash_unchanged":
                                    wrapper.known_model_sha256(),
                                "report":readable_source_explanation(
                                    full_native,0,
                                    has_independent_external_device_calibration=True),
                                "human_facing":render_concept_explanation_ru(
                                    full_native,0,
                                    observed_source_template=template.phi),
                            })
                            alpha_examples.append({
                                "family":family,"case":case,"nn_seed":seed,
                                "source_alpha_planes":
                                    source["bundle"].alpha_intervals[:4]
                                    .detach().numpy().tolist(),
                                "meaning":"alpha planes are nested membership-grade uncertainty, NOT event-probability intervals",
                            })
                        predicted_probability=expit(z)
                        if case in CI_CASES and stress in CI_STRESSES:
                            test_predictions[(family,case,stress,
                                              method,seed)]=predicted_probability.copy()
                        all_rows.append({
                            "family":family,"case":case,"stress":stress,
                            "method":method,"nn_seed":seed,
                            "metric":metrics(y[te],predicted_probability),
                            "source_use":float(used.mean()),
                            "selected_source":group.candidate_id,
                            "gate_approved":rec["gate"].approved,
                            "fuzzy_order":source["order"],
                            "generated_samples_not_observed":
                                method!="WLS_TYPE1",
                        })
                if core_sha!=sha_state(core):
                    raise AssertionError("Original K2 Transformer weights changed")
            print(json.dumps({
                "event":"V34_NATIVE_FIT","family":family,
                "nn_seed":seed,"k2_original_sha":core_sha,
                "known_K2_proxy_only":True,
            }),flush=True)
        for flow_id,flow in flow_cache.items():
            saved_flow_states.append({
                "family":family,"source_group":flow_id,
                "state_dict":{k:v.detach().cpu().clone()
                    for k,v in flow.model.state_dict().items()},
                "context_mean":flow.context_mean,
                "context_std":flow.context_std,
                "target_mean":flow.target_mean,
                "target_std":flow.target_std,
                "pseudo_train_interval_width_multiplier":
                    flow.heldout_train_width_multiplier,
            })
            flow_receipts.append({
                "family":family,"source_group":flow_id,
                "n_fit":flow.n_fit_samples,
                "n_heldout_train_calibration":flow.n_calibration_samples,
                "heldout_train_proxy_80pct_interval_coverage":
                    flow.heldout_train_nominal_80pct_coverage,
                "heldout_train_width_multiplier":
                    flow.heldout_train_width_multiplier,
                "training_uses_outcomes":flow.training_used_outcomes,
                "training_uses_oracle_latent":flow.training_used_latent_source_truth,
                "nominal_interval_is_clinical":False,
            })
    chosen_families=FAMILIES[:1] if smoke else FAMILIES
    summary=[
        _summarize(all_rows,family,case,stress,method,len(seeds))
        for family in chosen_families
        for case,_,_,_ in SCENARIOS
        for stress in TEST_STRESS
        for method in METHODS
    ]
    paired_uncertainty=[]
    for family_index,family in enumerate(chosen_families):
        for case_index,case in enumerate(CI_CASES):
            yy=test_labels[(family,case)]
            for stress_index,stress in enumerate(CI_STRESSES):
                def same_subject_predictions(method):
                    return np.stack([
                        test_predictions[(family,case,stress,method,seed)]
                        for seed in seeds
                    ])
                for k,(a,b) in enumerate((
                    ("FLOW_AUTO","FLOW_TYPE1"),
                    ("FLOW_AUTO","WLS_TYPE1"),
                    ("FLOW_RESTRICTED_IT3","FLOW_IT2"),
                    ("FLOW_AUTO","FROZEN_K2"),
                )):
                    delta=paired_test_ap_difference(
                        yy,same_subject_predictions(a),
                        same_subject_predictions(b),
                        seed=20263980+1000*family_index+
                            100*case_index+10*stress_index+k,
                        resamples=bootstrap,
                    )
                    paired_uncertainty.append({
                        "family":family,"case":case,"stress":stress,
                        "method_a":a,"method_b":b,**delta,
                    })
    result={
        "status":"SMOKE_ONLY" if smoke else "V34_ADAPTIVE_IT3_FLOW_EXECUTED",
        "frozen_protocol":"docs/medical/CONCEPTFAN_V34_FROZEN_ADAPTIVE_IT3_FLOW_PROTOCOL_20261106.md",
        "families":list(FAMILIES[:1] if smoke else FAMILIES),
        "NN_seeds":list(seeds),"NN_epochs":epochs,
        "n_unique_native_transformer_fits":len(all_training),
        "n_rows":len(all_rows),"n_summary":len(summary),
        "n_gate_receipts":len(all_gates),
        "train_valid_test_disjoint":True,
        "observational_source_grouping_without_oracle":True,
        "no_outcome_gradients_into_known_concept_encoder":True,
        "conditional_flow_trained_on_train_measured_proxy_not_latent":True,
        "fuzzy_order_chosen_only_from_train_and_independent_device_reference":True,
        "type3_discrete_alpha_slice_restricted_not_full_IT3_system":True,
        "flow_is_not_a_full_multivariate_diffusion_model":True,
        "output_fuzzy_membership_not_event_probability":True,
        "test_used_to_tune_source_thresholds":False,
        "source_semantics_physical_verified":False,
        "true_inference_backpropagates_through_native_transformer_and_source":False,
        "source_slot_added_to_real_native_torch_forward":True,
        "source_proposals":source_proposals,
        "calibration_anchors_simulator_known_reference":anchor_records,
        "training":all_training,"flow":flow_receipts,
        "gates":all_gates,"selections":all_selections,
        "rows":all_rows,"summary":summary,
        "explanation_passports":explanations,
        "type3_alpha_plane_examples":alpha_examples,
        "TEST_patient_paired_AP_bootstrap":paired_uncertainty,
        "CI95_conditional_on_three_fixed_models_and_shared_TEST":True,
        "saved_source_predictions_for_reanalysis":True,
        "saved_native_transformer_and_flow_checkpoints":True,
    }
    output.mkdir(parents=True,exist_ok=False)
    for path,data in (
        ("STATUS.json",result),("SUMMARY.json",summary),
        ("GATES.json",all_gates),("SOURCE_SELECTION.json",all_selections),
        ("FLOW_TRAIN_RECEIPTS.json",flow_receipts),
        ("SOURCE_EXPLANATIONS.json",explanations),
        ("PAIRED_TEST_BOOTSTRAP.json",paired_uncertainty),
    ):
        (output/path).write_text(
            json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))
    saved={"source_probability_recorded_not_physical_truth":np.asarray([1])}
    for (family,case),yy in test_labels.items():
        saved[f"{family}_{case}_TEST_Y"]=yy
    for (family,case,stress,method,seed),p in test_predictions.items():
        saved[f"{family}_{case}_{stress}_{method}_seed{seed}"]=p
    np.savez_compressed(output/"FROZEN_TEST_PREDICTIONS.npz",**saved)
    checkpoints=output/"CHECKPOINTS"
    checkpoints.mkdir(exist_ok=False)
    for model in saved_native_states:
        torch.save(
            {"family":model["family"],"seed":model["seed"],
             "architecture":"real native K2 Transformer ConceptFAN",
             "model_state_dict":model["state_dict"]},
            checkpoints/f"native_K2_{model['family']}_seed{model['seed']}.pt",
        )
    for model in saved_flow_states:
        torch.save(model,checkpoints/(
            f"conditional_flow_{model['family']}_{model['source_group']}.pt"))
    print(json.dumps({
        "event":"V34_COMPLETE","status":result["status"],
        "trained_native_transformers":len(all_training),
        "rows":len(all_rows),"summary":len(summary),
        "gate_receipts":len(all_gates),
        "flow_models":len(flow_receipts),
        "n_paired_CI_rows":len(paired_uncertainty),
        "paired_AP_comparisons":paired_uncertainty,
        "healthy_MCAR20":[x for x in summary if
            x["case"]=="healthy_physical" and x["stress"]=="MCAR20"],
        "synchronized_missing":[x for x in summary if
            x["case"]=="healthy_physical" and x["stress"]=="synchronized_block8"],
        "unknown_flip":[x for x in summary if
            x["case"]=="UNSEEN_physical_flip" and x["stress"]=="MCAR20"],
    }),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--out-dir",type=Path,required=True)
    options=parser.parse_args()
    run(options.out_dir,smoke=options.smoke)
