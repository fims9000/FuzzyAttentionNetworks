"""V36: actual Temporal ConceptFAN plus automatically recovered two-view energy.

Synthetic independent AR generator, three frozen native K2 fits, two VALID gates.
Source mixing and simulator latent values NEVER enter fitting; ordinal temporal
axis is NOT a signed/physically verified source. All TEST predictions retained.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conceptfan_discovery.gauge_source_v36 import (
    fit_blind_gauge_source_v36,source_energy_v36)
from conceptfan_discovery.crossview_world_v36 import (
    generate_world_v36,corrupt_views_v36,mask_views_v36,CASES,STRESS)
from conceptfan_discovery.auto_measured_concept_admission import (
    evaluate_two_independent_environments,admitted_test_logit)
from conceptfan_discovery.isolated_source_admission_v28 import train_source_coefficient
from conceptfan_discovery.cross_environment_transport import paired_test_ap_difference
from conceptfan_discovery.phase_pulse_benchmark_v31 import temporal_sensor_features
from conceptfan_discovery.adaptive_higher_order_concepts_v34 import FuzzySourceDefinition
from conceptfan_discovery.native_adaptive_type3_transformer_v34 import NativeAuditedFuzzyConceptFANV34
from scripts.experiment_automatic_measured_concepts import (
    train_masked_measured_known_concepts,logit_from_head)
from scripts.experiment_nonlinear_anchor_repair_v30 import make_firewalled_k2
from scripts.experiment_frozen_concept_head_refit_v26 import refit_linear_concept_head
from scripts.experiment_trained_conceptfan_omission_v25 import predict_native,sha_state,metrics

SEEDS=(11,23,37)
METHODS=("frozen_K2","blind_crossview_energy","unsafe_signed_viewA",
         "raw_temporal_ALL2000","raw_quadratic_ALL2000")


def _mask(x,mode,*,source_seed,known_seed):
    m=mask_views_v36(x,mode,seed=source_seed)
    m[:,:,:4]=np.random.default_rng(known_seed).random((len(x),16,4))>=.20
    return m


def _unsafe_signed(fit,x,m):
    observed=m[:,:,4:6].all(axis=2)
    n=observed.sum(axis=1)
    u=x[:,:,4:6]@fit.inv_a.T
    amp=np.where(observed,u[:,:,0],0.).sum(axis=1)/np.maximum(n,1)
    return expit(amp),(n>=4).astype(float)


def _raw_features(x,m,*,quadratic:bool):
    features=temporal_sensor_features(x,m)
    if not quadratic:
        return features
    paired=m[:,:,4:8].all(axis=-1)
    xa=np.where(m[:,:,4:6],x[:,:,4:6],0.)
    xb=np.where(m[:,:,6:8],x[:,:,6:8],0.)
    cross=np.einsum("nti,ntj->ntij",xa,xb).reshape(len(x),16,4)
    avg=np.where(paired[:,:,None],cross,0.).sum(axis=1)/np.maximum(1,paired.sum(axis=1))[:,None]
    return np.column_stack((features,avg,paired.mean(axis=1)))


def _raw_fit(x,m,y,quadratic):
    reg=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000))
    reg.fit(_raw_features(x,m,quadratic=quadratic),y)
    return reg


def _aggregate(rows,case,stress,method,nseeds):
    r=[v for v in rows if (v["case"],v["stress"],v["method"])==(case,stress,method)]
    expected=nseeds if method in METHODS[:3] else 1
    if len(r)!=expected:
        raise AssertionError("Wrong TEST repetition count")
    return {
        "case":case,"stress":stress,"method":method,
        "AUPRC":float(np.mean([v["metric"]["auprc"] for v in r])),
        "AUROC":float(np.mean([v["metric"]["auroc"] for v in r])),
        "NLL":float(np.mean([v["metric"]["nll"] for v in r])),
        "Brier":float(np.mean([v["metric"]["brier"] for v in r])),
        "source_use":float(np.mean([v["source_use"] for v in r])),
    }


def run(output:Path,*,smoke:bool=False):
    if output.exists():
        raise FileExistsError("Original research evidence must not be overwritten")
    torch.set_num_threads(2)
    ntr,nval,nte=(300,300,240) if smoke else (1200,800,600)
    seeds=(11,) if smoke else SEEDS
    world=generate_world_v36(seed=20261206,n_train=ntr,n_valid=nval,n_test=nte)
    x=world["x"]
    tr,v1,v2,te=(world[k] for k in ("train","valid1","valid2","test"))
    mtr=_mask(x[tr],"MCAR20",source_seed=20263601,known_seed=20263651)
    mv1=_mask(x[v1],"MCAR20",source_seed=20263602,known_seed=20263652)
    mv2=_mask(x[v2],"MCAR20",source_seed=20263603,known_seed=20263653)
    # The ONLY source-estimator inputs are measured TRAIN x and masks.
    fit=fit_blind_gauge_source_v36(
        x[tr],mtr,independent_view_errors_documented=True,
        bootstrap=24 if smoke else 32)
    qtr=source_energy_v36(fit,x[tr],mtr)
    qv1=source_energy_v36(fit,x[v1],mv1)
    cases={}
    for ci,(case,law,vshift,tshift) in enumerate(CASES):
        xv2=corrupt_views_v36(x[v2],vshift,seed=20263630+ci)
        xt=corrupt_views_v36(x[te],tshift,seed=20263640+ci)
        test={}
        for si,stress in enumerate(STRESS):
            mt=_mask(xt,stress,source_seed=20263680+si,known_seed=20263700)
            test[stress]={"x":xt,"m":mt,
                          "blind":source_energy_v36(fit,xt,mt),
                          "unsafe":_unsafe_signed(fit,xt,mt)}
        cases[case]={"labels":world["outcomes"][law],"xv2":xv2,"test":test,
                     "v2source":source_energy_v36(fit,xv2,mv2)}
    rows=[];gates=[];checkpoints=[];passports=[];all_predictions={}
    def save_pred(case,stress,method,seed,p):
        all_predictions[(case,stress,method,seed)]=np.asarray(p)
    for seed in seeds:
        model=make_firewalled_k2(seed)
        training=train_masked_measured_known_concepts(
            model,x[tr,:,:4],mtr[:,:,:4],seed=seed,
            epochs=2 if smoke else 10)
        before=sha_state(model)
        checkpoints.append({"seed":seed,"original_SHA":before,"training":training,
                            "state_dict":{k:v.detach().cpu().clone()
                                for k,v in model.state_dict().items()}})
        qt=predict_native(model,x[tr,:,:4],mtr[:,:,:4])["concept_summary"]
        q1=predict_native(model,x[v1,:,:4],mv1[:,:,:4])["concept_summary"]
        q2=predict_native(model,x[v2,:,:4],mv2[:,:,:4])["concept_summary"]
        mask_test=cases["healthy_energy"]["test"]["MCAR20"]["m"]
        qte=predict_native(model,x[te,:,:4],mask_test[:,:,:4])["concept_summary"]
        for ci,(case,law,vshift,tshift) in enumerate(CASES):
            y=cases[case]["labels"]
            head=refit_linear_concept_head(qt,y[tr])
            ztr=logit_from_head(head,qt)
            z1=logit_from_head(head,q1)
            z2=logit_from_head(head,q2)
            zte=logit_from_head(head,qte)
            fits={}
            for method in ("blind_crossview_energy","unsafe_signed_viewA"):
                if method=="blind_crossview_energy":
                    tqq,ts=qtr["q"],qtr["support"]
                    vqq,vs=qv1["q"],qv1["support"]
                    bqq,bs=cases[case]["v2source"]["q"],cases[case]["v2source"]["support"]
                else:
                    tqq,ts=_unsafe_signed(fit,x[tr],mtr)
                    vqq,vs=_unsafe_signed(fit,x[v1],mv1)
                    bqq,bs=_unsafe_signed(fit,cases[case]["xv2"],mv2)
                gamma=train_source_coefficient(ztr,tqq,y[tr])
                gate=evaluate_two_independent_environments(
                    method,gamma,("VALID1",y[v1],z1,vqq,vs),
                    ("VALID2",y[v2],z2,bqq,bs),
                    train_label_count=len(tr),min_mean_gain=.003,
                    min_support_fraction=.70,min_subject_support=.50,
                    bootstrap_reps=199 if smoke else 399,
                    seed=20263770+37*ci+seed)
                fits[method]=(gamma,gate)
                gates.append({"case":case,"nn_seed":seed,"method":method,
                              "signed_gamma":gamma,"gate":gate.as_dict()})
            for stress in STRESS:
                rec=cases[case]["test"][stress]
                baseline=expit(zte)
                save_pred(case,stress,"frozen_K2",seed,baseline)
                rows.append({"case":case,"stress":stress,"method":"frozen_K2",
                             "nn_seed":seed,"metric":metrics(y[te],baseline),
                             "source_use":0.})
                for method in ("blind_crossview_energy","unsafe_signed_viewA"):
                    gamma,gate=fits[method]
                    if method=="blind_crossview_energy":
                        q,s=rec["blind"]["q"],rec["blind"]["support"]
                    else:
                        q,s=rec["unsafe"]
                    z,used=admitted_test_logit(
                        zte,q,s,gamma=gamma,approved=gate.approved,
                        minimum_fraction=.50)
                    np.testing.assert_array_equal(z[~used],zte[~used])
                    probs=expit(z)
                    save_pred(case,stress,method,seed,probs)
                    rows.append({"case":case,"stress":stress,"method":method,
                                "nn_seed":seed,"metric":metrics(y[te],probs),
                                "source_use":float(used.mean()),
                                "approved_on_both_VALID":bool(gate.approved)})
                    if case=="healthy_energy" and stress=="MCAR20" and method=="blind_crossview_energy":
                        definition=FuzzySourceDefinition(
                            source_id="blind_slow_lag_two_view_energy",
                            sensor_indices=(4,5,6,7),
                            threshold_train=fit.train_median_power,
                            width_train=fit.train_power_scale,
                            train_missingness_spread=0.,
                            boundary_spread_anchor=0.,
                            certified_calibration_environments=0,
                            external_anchor_verified=False)
                        native=NativeAuditedFuzzyConceptFANV34(
                            model,np.array(head.weights),head.bias,
                            definition,fuzzy_order="type1")
                        with torch.no_grad():
                            native.source_slot.gamma.fill_(gamma)
                        n=16
                        native_out=native(
                            torch.from_numpy(x[te[:n],:,:4]),
                            torch.from_numpy(rec["m"][:n,:,:4]),
                            extra_source_amplitude=torch.from_numpy(
                                rec["blind"]["estimated_crossview_energy"][:n].astype("float32")),
                            extra_source_uncertainty=torch.zeros(n),
                            extra_source_observed_information=torch.from_numpy(
                                s[:n].astype("float32")),
                            extra_source_gate_approved=gate.approved)
                        if np.max(abs(native_out["probability"].detach().numpy()-probs[:n]))>3e-5:
                            raise AssertionError("Native torch forward is not additive and faithful")
                        passports.append({
                            "seed":seed,"sensor_A":[4,5],"sensor_B":[6,7],
                            "source":"unsigned energy of slow lag-one mode",
                            "measured_complete_paired_times":int(rec["blind"]["paired_sensor_times"][0]),
                            "measured_pair_fraction":float(rec["blind"]["actually_observed_pair_fraction"][0]),
                            "old_new_risk_difference_percentage_points":float(100*(probs[0]-baseline[0])),
                            "physical_semantics_verified":False,
                            "independent_view_error_provenance_only_simulated":True,
                            "source_used":bool(used[0]),
                            "signed_direction_unavailable":True})
            if sha_state(model)!=before:
                raise AssertionError("Original native K2 weights changed")
        print(json.dumps({"event":"V36_NATIVE_K2_TRAINED","seed":seed,
                          "source_lag_ratios":fit.lag_ratios.tolist()}),flush=True)
    # Ordinary classifiers get the full 2000-label budget, not equal compute.
    for ci,(case,law,vshift,tshift) in enumerate(CASES):
        y=cases[case]["labels"]
        allx=np.concatenate((x[tr],x[v1],cases[case]["xv2"]))
        allm=np.concatenate((mtr,mv1,mv2))
        all_y=np.concatenate((y[tr],y[v1],y[v2]))
        for method,quadratic in (("raw_temporal_ALL2000",False),("raw_quadratic_ALL2000",True)):
            clf=_raw_fit(allx,allm,all_y,quadratic)
            for stress in STRESS:
                rec=cases[case]["test"][stress]
                probs=clf.predict_proba(_raw_features(rec["x"],rec["m"],quadratic=quadratic))[:,1]
                save_pred(case,stress,method,None,probs)
                rows.append({"case":case,"stress":stress,"method":method,
                             "nn_seed":None,"metric":metrics(y[te],probs),
                             "source_use":0.})
    summary=[_aggregate(rows,c,stress,method,len(seeds))
             for c,*_ in CASES for stress in STRESS for method in METHODS]
    paired=[]
    for ci,(case,stress,a,b) in enumerate((
        ("healthy_energy","MCAR20","blind_crossview_energy","frozen_K2"),
        ("healthy_energy","MCAR20","blind_crossview_energy","raw_temporal_ALL2000"),
        ("healthy_energy","synced_block8","blind_crossview_energy","frozen_K2"),
        ("unseen_both_views_flip","MCAR20","blind_crossview_energy","frozen_K2"),
        ("unseen_one_view_flip","MCAR20","blind_crossview_energy","frozen_K2"),
        ("unseen_shared_colored_noise","MCAR20","blind_crossview_energy","frozen_K2"),
    )):
        def predictions(method):
            arr=np.stack([all_predictions[(case,stress,method,z)]
                          for z in (seeds if method in METHODS[:3] else (None,))])
            return np.repeat(arr,len(seeds),axis=0) if len(arr)==1 else arr
        conf=paired_test_ap_difference(
            cases[case]["labels"][te],predictions(a),predictions(b),
            seed=20263888+ci,resamples=199 if smoke else 399)
        paired.append({"case":case,"stress":stress,"method_a":a,"method_b":b,**conf})
    if not smoke:
        rec=cases["healthy_energy"]["test"]["MCAR20"]["blind"]
        ok=rec["support"]>0
        correlation=float(np.corrcoef(
            rec["estimated_crossview_energy"][ok],
            world["true_slow_energy_AUDIT_ONLY"][te][ok])[0,1])
    else:
        correlation=None
    result={
        "status":"V36_SMOKE_ONLY" if smoke else "V36_BLIND_MULTIVIEW_ENERGY_EXECUTED",
        "generator_seed":20261206,"n_train":ntr,"n_valid1":nval//2,
        "n_valid2":nval//2,"n_test":nte,
        "n_native_K2_fits":len(seeds),
        "n_results":len(rows),"n_summary":len(summary),
        "n_valid_candidate_gates":len(gates),
        "source_directions_trained_without_latent_or_mixing_or_Y":True,
        "noise_independence_documented_by_synthetic_design_not_identified_from_X":True,
        "physical_signed_source_identity_certified":False,
        "old_K2_frozen_under_native_concept_extension":True,
        "no_missing_measurements_are_generated_as_real":True,
        "source_fit":fit.receipt(),"rows":rows,"summary":summary,
        "gates":gates,"human_concept_passports":passports,
        "paired_test_bootstrap_CI":paired,
        "AUDIT_ONLY_true_source_energy_corr":correlation,
    }
    output.mkdir(parents=True,exist_ok=False)
    for name,content in (("STATUS.json",result),
                         ("SUMMARY.json",summary),("PAIRED_TEST_CI.json",paired)):
        (output/name).write_text(
            json.dumps(content,indent=2,allow_nan=False,ensure_ascii=False))
    named={f"{case}_{stress}_{method}_seed{seed}":p
           for (case,stress,method,seed),p in all_predictions.items()}
    for case,law,*_ in CASES:
        named[f"{case}_TEST_LABELS"]=cases[case]["labels"][te]
    np.savez_compressed(output/"FROZEN_TEST_PREDICTIONS.npz",**named)
    ckpt=output/"CHECKPOINTS"
    ckpt.mkdir()
    for z in checkpoints:
        torch.save(z,ckpt/f"native_K2_seed{z['seed']}.pt")
    print(json.dumps({"event":"V36_COMPLETE","status":result["status"],
        "n_models":len(seeds),"results":len(rows),"aggregates":len(summary),
        "gates":len(gates),"TEST_oracle_correlation_AUDIT_ONLY":correlation,
        "healthy_MCAR20":[v for v in summary if v["case"]=="healthy_energy"
                          and v["stress"]=="MCAR20"],
        "one_view_fault_MCAR20":[v for v in summary if v["case"]=="unseen_one_view_flip"
                                 and v["stress"]=="MCAR20"]}),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args()
    run(args.out_dir,smoke=args.smoke)
