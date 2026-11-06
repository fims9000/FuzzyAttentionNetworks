"""V37: registered experiment; six REAL native K2 Transformer fits, two mixes.

The source estimator is a pure function of TRAIN sensor histories and masks:
no source H, S, outcome Y, VALID or TEST information enters its covariance,
spectral selection, pair alignment or discrepancy threshold. New forecast gamma
gets TRAIN labels, independently validated on VALID1+VALID2. Unseen TEST
faults and missingness are measured, not used for selection.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit, logit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from conceptfan_discovery.threeview_world_v37 import (
    MIX_SEEDS,CASES,STRESSES,generate_threeview_world_v37,
    apply_device_world_v37,view_masks_v37,
)
from conceptfan_discovery.threeview_centered_v37 import (
    fit_centered_threeview_v37,threeview_feature_v37,pair_feature_v37,
    raw_ab_v36_feature,
)
from conceptfan_discovery.gauge_source_v36 import fit_blind_gauge_source_v36
from conceptfan_discovery.auto_measured_concept_admission import (
    evaluate_two_independent_environments,admitted_test_logit,
)
from conceptfan_discovery.isolated_source_admission_v28 import train_source_coefficient
from conceptfan_discovery.cross_environment_transport import paired_test_ap_difference
from conceptfan_discovery.phase_pulse_benchmark_v31 import temporal_sensor_features
from conceptfan_discovery.adaptive_higher_order_concepts_v34 import FuzzySourceDefinition
from conceptfan_discovery.native_adaptive_type3_transformer_v34 import NativeAuditedFuzzyConceptFANV34
from scripts.experiment_automatic_measured_concepts import (
    train_masked_measured_known_concepts,logit_from_head,
)
from scripts.experiment_nonlinear_anchor_repair_v30 import make_firewalled_k2
from scripts.experiment_frozen_concept_head_refit_v26 import refit_linear_concept_head
from scripts.experiment_trained_conceptfan_omission_v25 import (
    predict_native,sha_state,metrics,
)

SEEDS=(11,23,37)
NATIVE_METHODS=("frozen_K2","AB_centered","ABC_threeview","AB_V36_uncentered")
METHODS=NATIVE_METHODS+("raw_temporal_ALL1800","raw_quadratic_ALL1800")
SOURCES=("AB_centered","ABC_threeview","AB_V36_uncentered")


def _raw_feature(x,m,*,quadratic):
    linear=temporal_sensor_features(x,m)
    if not quadratic:
        return linear
    extras=[]
    for ia,ib in (((4,5),(6,7)),((4,5),(8,9)),((6,7),(8,9))):
        complete=m[:,:,list(ia+ib)].all(axis=-1)
        a=np.where(m[:,:,list(ia)],x[:,:,list(ia)],0.)
        b=np.where(m[:,:,list(ib)],x[:,:,list(ib)],0.)
        cross=np.einsum("nti,ntj->ntij",a,b).reshape(len(x),16,4)
        mean=np.where(complete[:,:,None],cross,0.).sum(axis=1)/(
            np.maximum(1,complete.sum(axis=1))[:,None])
        extras.extend((mean,complete.mean(axis=1)[:,None]))
    return np.column_stack((linear,*extras))


def _raw_fit(x,m,y,quadratic):
    model=make_pipeline(StandardScaler(),
                        LogisticRegression(C=1,max_iter=1000))
    model.fit(_raw_feature(x,m,quadratic=quadratic),y)
    return model


def _candidate(method,fit,x,m):
    if method=="AB_centered":
        return pair_feature_v37(fit.pairs["AB"],x,m)
    if method=="ABC_threeview":
        return threeview_feature_v37(fit,x,m)
    if method=="AB_V36_uncentered":
        return raw_ab_v36_feature(
            getattr(fit,"_old_ab",None),x,m)
    raise ValueError("Unregistered V37 method")


def _aggregate(rows,mix,case,stress,method,seed_count):
    selected=[r for r in rows if
              (r["mix_seed"],r["case"],r["stress"],r["method"])==
              (mix,case,stress,method)]
    expected=seed_count if method in NATIVE_METHODS else 1
    if len(selected)!=expected:
        raise AssertionError("TEST result count mismatch")
    return {
        "mix_seed":mix,"case":case,"stress":stress,"method":method,
        "AUPRC":float(np.mean([r["metrics"]["auprc"] for r in selected])),
        "AUROC":float(np.mean([r["metrics"]["auroc"] for r in selected])),
        "NLL":float(np.mean([r["metrics"]["nll"] for r in selected])),
        "Brier":float(np.mean([r["metrics"]["brier"] for r in selected])),
        "source_use":float(np.mean([r["source_use"] for r in selected])),
    }


def _safe_old_fit(x,m,seed):
    try:
        old=fit_blind_gauge_source_v36(
            x,m,independent_view_errors_documented=True,
            seed=seed,bootstrap=24)
        return old,None
    except (ValueError,AssertionError) as exc:
        return None,str(exc)


def _masks(x,stress,*,mix,seed):
    return view_masks_v37(x,stress,
                          seed=seed+mix%97,
                          known_seed=20263788+mix%97)


def run(output:Path,*,smoke=False):
    if output.exists():
        raise FileExistsError("V37 evidence output exists; do not overwrite")
    torch.set_num_threads(2)
    n_train,n_valid,n_test=(360,300,240) if smoke else (1000,800,600)
    seeds=(11,) if smoke else SEEDS
    mix_seeds=MIX_SEEDS[:1] if smoke else MIX_SEEDS
    boost=199 if smoke else 399
    results=[];gates=[];training=[];sources=[];passports=[]
    all_predictions={};all_y={};all_k2_states=[]
    def receipt(mix,case,stress,method,seed,pred):
        all_predictions[(mix,case,stress,method,seed)]=np.asarray(pred,dtype=np.float64)

    for mix in mix_seeds:
        world=generate_threeview_world_v37(
            mix_seed=mix,n_train=n_train,n_valid=n_valid,n_test=n_test)
        x=world["x"]
        tr,v1,v2,te=(world[k] for k in ("train","valid1","valid2","test"))
        mtrain=_masks(x[tr],"MCAR20",mix=mix,seed=20263701)
        mvalid1=_masks(x[v1],"MCAR20",mix=mix,seed=20263702)
        mvalid2=_masks(x[v2],"MCAR20",mix=mix,seed=20263703)
        fit=fit_centered_threeview_v37(
            x[tr],mtrain,independent_view_noise_documented=True,
            seed=20263741+mix%97,bootstrap=24)
        old,old_fit_failure=_safe_old_fit(x[tr],mtrain,20263801+mix%97)
        # Old uncentered method is FROZEN. Keep its failure as exact abstention.
        src={
            "AB_centered":lambda xx,mm:pair_feature_v37(fit.pairs["AB"],xx,mm),
            "ABC_threeview":lambda xx,mm:threeview_feature_v37(fit,xx,mm),
            "AB_V36_uncentered":lambda xx,mm:raw_ab_v36_feature(old,xx,mm),
        }
        source_training={name:f(x[tr],mtrain) for name,f in src.items()}
        source_valid1={name:f(x[v1],mvalid1) for name,f in src.items()}
        sources.append({
            "mix_seed":mix,"threeview_measured_only":fit.audit(),
            "historic_old_uncentered_fit_error":old_fit_failure,
            "not_a_named_clinical_source":True,
            "synthetic_mixing_matrices_used_for_fit":False,
        })
        case_values={}
        for ci,(case,law,val2fault,testfault) in enumerate(CASES):
            xvalid2=apply_device_world_v37(x[v2],val2fault,
                              seed=20263850+ci+mix%97)
            xte=apply_device_world_v37(x[te],testfault,
                            seed=20263875+ci+mix%97)
            by_stress={}
            for si,stress in enumerate(STRESSES):
                mask=_masks(xte,stress,mix=mix,seed=20263905+si)
                measured={name:fn(xte,mask) for name,fn in src.items()}
                by_stress[stress]={"x":xte,"mask":mask,"measured":measured}
            case_values[case]={
                "law":law,"valid2_x":xvalid2,
                "valid2_measured":{
                    name:fn(xvalid2,mvalid2) for name,fn in src.items()
                },
                "test":by_stress,
            }
            all_y[(mix,case)]=world["outcomes"][law][te]

        # TRUE off-TASK offset invariance: standalone feature audit performed
        # BEFORE looking at TEST outcome scores (not a retroactive risk fix).
        clean=case_values["healthy_energy"]["test"]["MCAR20"]["measured"]["ABC_threeview"]
        for case in ("TEST_offset_A_plus2","TEST_offset_all_plus2"):
            test=case_values[case]["test"]["MCAR20"]["measured"]["ABC_threeview"]
            if (not np.array_equal(clean["support"],test["support"]) or
                not np.allclose(clean["q"],test["q"],atol=4e-6,rtol=0)):
                raise AssertionError("Episode-centered TEST source lost exact offset invariance")

        # Feature/scaling of the learned source is outcome blind; use
        # synthetic truth ONLY to evaluate recovery after training.
        truth=world["synthetic_latent_fluctuation_power_AUDIT_ONLY"][te]
        healthy=clean
        allow=healthy["support"]>0
        source_correlation=(float(np.corrcoef(truth[allow],
                           healthy["q"][allow])[0,1]) if allow.sum()>20 else None)
        sources[-1]["TEST_lag_energy_correlation_AUDIT_ONLY"]=source_correlation

        for seed in seeds:
            native=make_firewalled_k2(seed)
            trained=train_masked_measured_known_concepts(
                native,x[tr,:,:4],mtrain[:,:,:4],seed=seed,
                epochs=2 if smoke else 10)
            native_sha=sha_state(native)
            all_k2_states.append({
                "mix_seed":mix,"NN_seed":seed,
                "native_sha":native_sha,
                "state_dict":{k:v.detach().cpu().clone()
                              for k,v in native.state_dict().items()},
            })
            training.append({"mix_seed":mix,"NN_seed":seed,
                             "native_SHA":native_sha,**trained})
            qt=predict_native(native,x[tr,:,:4],mtrain[:,:,:4])["concept_summary"]
            q1=predict_native(native,x[v1,:,:4],mvalid1[:,:,:4])["concept_summary"]
            q2=predict_native(native,x[v2,:,:4],mvalid2[:,:,:4])["concept_summary"]
            known_test_mask=case_values["healthy_energy"]["test"]["MCAR20"]["mask"][:,:,:4]
            qtest=predict_native(native,x[te,:,:4],known_test_mask)["concept_summary"]

            for ci,(case,law,val2fault,testfault) in enumerate(CASES):
                y=world["outcomes"][law]
                head=refit_linear_concept_head(qt,y[tr])
                ztr=logit_from_head(head,qt)
                zv1=logit_from_head(head,q1)
                zv2=logit_from_head(head,q2)
                zte=logit_from_head(head,qtest)
                source_fits={}
                for method in SOURCES:
                    measured=source_training[method]
                    gamma=train_source_coefficient(
                        ztr,measured["q"],y[tr])
                    first=source_valid1[method]
                    second=case_values[case]["valid2_measured"][method]
                    gate=evaluate_two_independent_environments(
                        method,gamma,
                        ("VALID1",y[v1],zv1,first["q"],first["support"]),
                        ("VALID2",y[v2],zv2,second["q"],second["support"]),
                        train_label_count=len(tr),min_mean_gain=.003,
                        min_support_fraction=.25,min_subject_support=.50,
                        bootstrap_reps=boost,
                        seed=20263960+int(mix%97)*100+ci*5+seed,
                    )
                    source_fits[method]=(gamma,gate)
                    gates.append({
                        "mix_seed":mix,"case":case,"NN_seed":seed,
                        "method":method,"gamma":gamma,
                        "gate":gate.as_dict(),
                    })
                for stress in STRESSES:
                    rec=case_values[case]["test"][stress]
                    baseline=expit(zte)
                    receipt(mix,case,stress,"frozen_K2",seed,baseline)
                    results.append({
                        "mix_seed":mix,"case":case,"stress":stress,
                        "method":"frozen_K2","NN_seed":seed,
                        "metrics":metrics(y[te],baseline),"source_use":0.,
                    })
                    for method in SOURCES:
                        gamma,gate=source_fits[method]
                        f=rec["measured"][method]
                        z,used=admitted_test_logit(
                            zte,f["q"],f["support"],gamma=gamma,
                            approved=gate.approved,minimum_fraction=.50)
                        np.testing.assert_array_equal(z[~used],zte[~used])
                        probs=expit(z)
                        receipt(mix,case,stress,method,seed,probs)
                        results.append({
                            "mix_seed":mix,"case":case,"stress":stress,
                            "method":method,"NN_seed":seed,
                            "metrics":metrics(y[te],probs),
                            "source_use":float(used.mean()),
                            "source_approved_by_both_VALID":gate.approved,
                        })
                        if (case=="healthy_energy" and stress=="MCAR20"
                            and method=="ABC_threeview"):
                            definition=FuzzySourceDefinition(
                                source_id="mean_centered_three_independent_views_fluctuation_power",
                                sensor_indices=(4,5,6,7,8,9),
                                threshold_train=0.,width_train=1.,
                                train_missingness_spread=0.,
                                boundary_spread_anchor=0.,
                                certified_calibration_environments=0,
                                external_anchor_verified=False,
                            )
                            wrapper=NativeAuditedFuzzyConceptFANV34(
                                native,np.array(head.weights),head.bias,
                                definition,fuzzy_order="type1")
                            with torch.no_grad():
                                wrapper.source_slot.gamma.fill_(gamma)
                            samples=min(16,len(te))
                            # q is a measured source definition; logit(q) is
                            # an INVERTIBLE coordinate converter into the
                            # existing torch sigmoid concept membership.
                            amplitude=logit(np.clip(
                                f["q"][:samples],1e-6,1-1e-6))
                            original_hash=sha_state(native)
                            native_result=wrapper(
                                torch.from_numpy(
                                    rec["x"][:samples,:,:4]),
                                torch.from_numpy(
                                    rec["mask"][:samples,:,:4]),
                                extra_source_amplitude=torch.from_numpy(
                                    amplitude.astype(np.float32)),
                                extra_source_uncertainty=torch.zeros(samples),
                                extra_source_observed_information=torch.from_numpy(
                                    f["support"][:samples].astype(np.float32)),
                                extra_source_gate_approved=gate.approved,
                            )
                            if np.max(abs(
                                native_result["probability"].detach().numpy()-
                                probs[:samples]))>3e-5:
                                raise AssertionError("Native in-graph concept differs from reported risk")
                            if original_hash!=sha_state(native):
                                raise AssertionError("Adding a source modified frozen K2")
                            passports.append({
                                "mix_seed":mix,"NN_seed":seed,
                                "name":"Positive lag-order temporal fluctuation energy",
                                "sensor_groups":{"A":[4,5],"B":[6,7],"C":[8,9]},
                                "pair_measured_times_first_episode":
                                   f["pair_timepoints"][0].tolist(),
                                "all_three_pairs_accepted":bool(
                                   f["all_three_pairs_measured"][0]),
                                "source_disagreement_first_episode":float(
                                   f["pairwise_max_membership_discrepancy"][0]),
                                "first_episode_source_used":bool(used[0]),
                                "model_prediction_change_percentage_points":float(
                                   100*(probs[0]-baseline[0])),
                                "physical_semantics_certified":False,
                                "view_noise_independence_is_external_assumption":True,
                                "native_K2_SHA_unchanged":original_hash,
                            })
                if sha_state(native)!=native_sha:
                    raise AssertionError("Original native K2 modified by additive source")
            print(json.dumps({
                "event":"V37_TRAINED_REAL_NATIVE_TRANSFORMER",
                "mix_seed":mix,"NN_seed":seed,"K2_SHA":native_sha,
            }),flush=True)

        # Baselines ALL labels and same TEST subjects, but unequal FLOPs
        for ci,(case,law,val2fault,testfault) in enumerate(CASES):
            labels=world["outcomes"][law]
            xall=np.concatenate((x[tr],x[v1],case_values[case]["valid2_x"]))
            mall=np.concatenate((mtrain,mvalid1,mvalid2))
            yall=np.concatenate((labels[tr],labels[v1],labels[v2]))
            for method,quadratic in (("raw_temporal_ALL1800",False),
                                      ("raw_quadratic_ALL1800",True)):
                model=_raw_fit(xall,mall,yall,quadratic)
                for stress in STRESSES:
                    r=case_values[case]["test"][stress]
                    p=model.predict_proba(
                        _raw_feature(r["x"],r["mask"],quadratic=quadratic))[:,1]
                    receipt(mix,case,stress,method,None,p)
                    results.append({
                        "mix_seed":mix,"case":case,"stress":stress,
                        "method":method,"NN_seed":None,
                        "metrics":metrics(labels[te],p),
                        "source_use":0.,
                    })

    summaries=[
        _aggregate(results,mix,case,stress,method,len(seeds))
        for mix in mix_seeds for case,*_ in CASES
        for stress in STRESSES for method in METHODS
    ]
    CI_TESTS=(
       ("healthy_energy","MCAR20","ABC_threeview","frozen_K2"),
       ("healthy_energy","MCAR20","ABC_threeview","AB_centered"),
       ("healthy_energy","MCAR20","ABC_threeview","raw_quadratic_ALL1800"),
       ("TEST_common_AB_colored","MCAR20","ABC_threeview","AB_centered"),
       ("TEST_common_AB_colored","MCAR20","ABC_threeview","frozen_K2"),
       ("TEST_common_ABC_colored","MCAR20","ABC_threeview","frozen_K2"),
       ("TEST_offset_A_plus2","MCAR20","ABC_threeview","frozen_K2"),
       ("TEST_one_view_A_flip","MCAR20","ABC_threeview","frozen_K2"),
       ("healthy_energy","synchronized_block8","ABC_threeview","frozen_K2"),
    )
    ci=[]
    for mix in mix_seeds:
        for k,(case,stress,a,b) in enumerate(CI_TESTS):
            y=all_y[(mix,case)]
            def pred(method):
                s=(seeds if method in NATIVE_METHODS else (None,))
                arr=np.stack([all_predictions[(mix,case,stress,method,z)]
                              for z in s])
                return arr if len(arr)==len(seeds) else np.repeat(arr,len(seeds),axis=0)
            v=paired_test_ap_difference(
                y,pred(a),pred(b),seed=20264000+mix%97*100+k,
                resamples=boost)
            ci.append({
                "mix_seed":mix,"case":case,"stress":stress,
                "method_a":a,"method_b":b,
                **v,
            })
    response={
        "status":"V37_SMOKE_ONLY" if smoke else "V37_CENTERED_THREEVIEW_EXECUTED",
        "frozen_protocol":"docs/medical/CONCEPTFAN_V37_FROZEN_THREEVIEW_CENTERED_PROTOCOL_20261106.md",
        "mix_seeds":list(mix_seeds),
        "n_train":n_train,"n_valid1":n_valid//2,
        "n_valid2":n_valid//2,"n_test":n_test,
        "native_model_seeds":list(seeds),
        "n_native_fits":len(training),"n_rows":len(results),
        "n_summary":len(summaries),"n_valid_gates":len(gates),
        "n_paired_test_CI":len(ci),
        "source_mixing_H_not_supplied_to_learner":True,
        "third_view_assumed_independent_not_verified_from_data":True,
        "view_partition_fixed_externally":True,
        "train_only_threshold_and_centers":True,
        "test_outcomes_not_used_for_source_selection":True,
        "type2_3_or_flow_not_used_to_claim_physical_semantics":True,
        "no_real_clinical_subjects":True,
        "risk_gate_unchanged_across_test_faults_and_masks":True,
        "frozen_K2_preserved_when_source_refused":True,
        "source_fits":sources,"training":training,
        "gates":gates,"rows":results,"summary":summaries,
        "paired_CI":ci,"human_facing_source_passports":passports,
    }
    output.mkdir(parents=True,exist_ok=False)
    for filename,payload in (
        ("STATUS.json",response),("SUMMARY.json",summaries),
        ("GATES.json",gates),("PAIR_TEST_BOOTSTRAP.json",ci),
        ("SOURCE_FITS.json",sources),
        ("SOURCE_EXPLANATIONS.json",passports),
    ):
        (output/filename).write_text(
            json.dumps(payload,indent=2,allow_nan=False,ensure_ascii=False))
    for mix,case in all_y:
        all_predictions[(mix,case,"TEST_LABELS","labels",None)]=all_y[(mix,case)]
    save={
        f"device{mix}_{case}_{stress}_{method}_seed{seed}":np.asarray(p)
        for (mix,case,stress,method,seed),p in all_predictions.items()
    }
    np.savez_compressed(output/"FROZEN_TEST_PREDICTIONS.npz",**save)
    cp=output/"CHECKPOINTS"
    cp.mkdir(exist_ok=False)
    for model in all_k2_states:
        torch.save(model,cp/f"native_K2_device{model['mix_seed']}_seed{model['NN_seed']}.pt")
    print(json.dumps({
        "event":"V37_COMPLETE","status":response["status"],
        "native_fits":len(training),"n_rows":len(results),
        "n_summary":len(summaries),"n_gates":len(gates),
        "healthy_MCAR20":[z for z in summaries if
            z["case"]=="healthy_energy" and z["stress"]=="MCAR20"],
        "common_AB_noise_MCAR20":[z for z in summaries if
            z["case"]=="TEST_common_AB_colored" and z["stress"]=="MCAR20"],
        "common_ABC_noise_MCAR20":[z for z in summaries if
            z["case"]=="TEST_common_ABC_colored" and z["stress"]=="MCAR20"],
    }),flush=True)
    return response


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    run(args.out_dir,smoke=args.smoke)
