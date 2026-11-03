"""Exploratory/new-seed synthetic V14: multi-environment concept discovery.

No randomized interventions, known target-to-source map, latent labels, or
preselected concept count enter training; only environment IDs are available.
This is a conditional three-environment second-order BSS benchmark, NOT ICU.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss

from .environment_contrast_v14 import fit_environment_contrast,project_environment_concepts
from .environment_synthetic_v14 import make_environment_scm
from .instrumented_v13 import InstrumentedConceptBottleneck

K_VALUES=(3,5,7)
FIT_SEEDS=(11,23,37)
GENERATOR_SEEDS=(20261104,20270104)


def _fit_native(signature,train_values,train_labels,seed:int,epochs:int=85):
    x=np.asarray(train_values,dtype=np.float32)
    y=np.asarray(train_labels,dtype=np.float32)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    with torch.no_grad():
        train_projections=x@np.linalg.pinv(signature).T.astype(np.float32)
    center=np.median(train_projections,axis=(0,1)).astype(np.float32)
    model=InstrumentedConceptBottleneck(signature,source_quantiles=center)
    with torch.no_grad():
        model.raw_beta.normal_(mean=0,std=.18)
        model.threshold.add_(torch.randn_like(model.threshold)*.06)
        model.raw_width.add_(torch.randn_like(model.raw_width)*.05)
    optim=torch.optim.AdamW(model.parameters(),lr=.012,weight_decay=.0002)
    xx=torch.from_numpy(x)
    yy=torch.from_numpy(y)
    for _ in range(epochs):
        optim.zero_grad(set_to_none=True)
        out=model(xx)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(out["logits"],yy)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite native concept-only loss")
        loss.backward()
        optim.step()
    model.eval()
    return model


def _rho(a,b):
    if np.std(a)<1e-10 or np.std(b)<1e-10:
        return float("nan")
    return float(spearmanr(a,b).statistic)


def optimistic_oracle_concept_score(estimated,truth):
    """Evaluation-only *optimistic* Hungarian abs-Spearman, not identity."""
    x=np.asarray(estimated,dtype=float)
    z=np.asarray(truth,dtype=float)
    if x.shape[0]!=z.shape[0] or x.ndim!=2 or z.ndim!=2:
        raise ValueError("Oracle must match heldout episode IDs")
    m=np.array([[_rho(x[:,i],z[:,j]) for j in range(z.shape[1])]
                for i in range(x.shape[1])])
    if not np.isfinite(m).all():
        raise ValueError("Constant or degenerate oracle comparison")
    rows,cols=linear_sum_assignment(-np.abs(m))
    return float(np.abs(m[rows,cols]).mean()), list(map(int,cols)),float(np.min(np.abs(m[rows,cols])))


def _metrics(y,p):
    if not np.isfinite(p).all() or not np.isin(y,[0,1]).all():
        raise ValueError("Nonfinite scores or invalid labels")
    return {"auprc":float(average_precision_score(y,p)),
            "auroc":float(roc_auc_score(y,p)),
            "brier":float(brier_score_loss(y,p))}


def run(out_dir:Path,*,generator_base:int=20261104,
        per_environment:int=4000,train_per_environment:int=3200,
        sample_per_environment:int=2800,epochs:int=85):
    if (generator_base not in GENERATOR_SEEDS or
        per_environment!=4000 or train_per_environment!=3200 or
        sample_per_environment!=2800 or epochs!=85):
        raise ValueError("Frozen V14 synthetic source and train/evaluation settings changed")
    if out_dir.exists():
        raise FileExistsError("Never overwrite frozen scientific artifacts")
    all_rows,all_pairs,cert_rows,adversarial=[],[],[],[]
    for true_k in K_VALUES:
        source=make_environment_scm(k=true_k,seed=generator_base+true_k,
                                    per_environment=per_environment)
        x=source.observations.astype(np.float32)
        y=source.labels
        env=source.environment
        # E-specific ordered split is deterministic and independent of labels.
        train=np.concatenate([np.flatnonzero(env==e)[:train_per_environment]
                              for e in range(3)])
        val=np.concatenate([np.flatnonzero(env==e)[train_per_environment:]
                            for e in range(3)])
        assert len(val)==2400 and len(train)==9600
        fits={}
        for seed in FIT_SEEDS:
            rng=np.random.default_rng(seed+true_k*1000)
            sub=np.concatenate([rng.choice(np.flatnonzero(env==e)[:train_per_environment],
                      sample_per_environment,replace=False) for e in range(3)])
            assert np.intersect1d(sub,val).size==0
            try:
                cert=fit_environment_contrast(x[sub],env[sub],seed=seed+true_k,
                            permutations=99,bootstraps=32)
            except ValueError as exc:
                cert_rows.append({"true_k_evaluation_only":true_k,"fit_seed":seed,
                     "status":"ABSTAIN","reason":str(exc),"selected_train_rank":None})
                print("V14 ABSTAIN true_k",true_k,"seed",seed,str(exc),flush=True)
                continue
            k=cert.selected_rank
            train_features=project_environment_concepts(x[sub],cert)
            val_features=project_environment_concepts(x[val],cert)
            # Both classifiers get exact same K train-estimated unmixing.
            linear=LogisticRegression(C=1.,max_iter=1000,solver="lbfgs",random_state=seed)
            linear.fit(train_features,y[sub])
            p_linear=linear.predict_proba(val_features)[:,1]
            phi_linear=val_features*linear.coef_[0]
            max_error=np.abs(linear.decision_function(val_features)
                -phi_linear.sum(1)-linear.intercept_[0]).max()
            assert max_error<1e-9
            model=_fit_native(cert.signature,x[sub],y[sub],seed,epochs=epochs)
            with torch.no_grad():
                output=model(torch.from_numpy(x[val]))
                p_native=output["probabilities"].numpy().astype(float)
                phi_native=output["contributions"].numpy().astype(float)
                native_error=(output["logits"]-output["contributions"].sum(-1)-model.bias).abs().max()
                assert float(native_error)<1e-5
            # Comparison without environment-derived source axes.
            full_train=x[sub,-4:,:].mean(axis=1)
            full_val=x[val,-4:,:].mean(axis=1)
            raw_logistic=LogisticRegression(C=1.,max_iter=1000,solver="lbfgs",random_state=seed)
            raw_logistic.fit(full_train,y[sub])
            p_raw=raw_logistic.predict_proba(full_val)[:,1]
            pca=PCA(n_components=k,svd_solver="full")
            pca.fit(full_train)
            pca_train=pca.transform(full_train)
            pca_val=pca.transform(full_val)
            pca_logistic=LogisticRegression(C=1.,max_iter=1000,solver="lbfgs",random_state=seed)
            pca_logistic.fit(pca_train,y[sub])
            p_pca=pca_logistic.predict_proba(pca_val)[:,1]
            # Oracle is touched only AFTER training/final predictions above.
            truth=source.true_sources[val,-4:,:].mean(axis=1)
            source_oracle,mapping,min_oracle=optimistic_oracle_concept_score(val_features,truth)
            pca_oracle,_,_=optimistic_oracle_concept_score(pca_val,truth)
            for method,p in [("v14_native_concept_head",p_native),
                  ("logistic_same_v14_concepts",p_linear),
                  ("pooled_pca_logistic",p_pca),
                  ("raw_sensor_logistic",p_raw)]:
                all_rows.append({"true_k_evaluation_only":true_k,"fit_seed":seed,
                     "method":method,"selected_K_train_only":k,
                     "n_train":len(sub),"n_eval":len(val),
                     "measured_environment_labels_known":True,
                     "intervention_target_map_available":False,
                     "oracle_used_for_fit":False,
                     "optimistic_source_abs_oracle_spearman":source_oracle if method.startswith("v14_") or method.startswith("logistic_same") else (pca_oracle if method.startswith("pooled_pca") else None),
                     **_metrics(y[val],p)})
            fits[seed]={"cert":cert,"native_phi":phi_native,"linear_phi":phi_linear,
                        "native_weights":(3*torch.tanh(model.raw_beta)).detach().numpy(),
                        "linear_weights":linear.coef_[0].copy()}
            cert_rows.append({"true_k_evaluation_only":true_k,"fit_seed":seed,
                  "status":"CERTIFIED","reason":"", "selected_train_rank":k,
                  "min_signature_bootstrap_q10":float(min(cert.response_overlap_q10)),
                  "min_ratio_gap_bootstrap_q10":float(min(cert.ratio_gap_q10)),
                  "permutation_null_p":cert.permutation_null_p,
                  "oracle_alignment_only_after_fit":mapping,
                  "oracle_source_min_abs_spearman":min_oracle})
            print("V14 K",true_k,"seed",seed,"selected",k,"native AUPRC",round(all_rows[-4]["auprc"],5),
                  "oracle",round(source_oracle,4),flush=True)
        if len(fits)==3 and len({f["cert"].selected_rank for f in fits.values()})==1:
            k=next(iter(fits.values()))["cert"].selected_rank
            for method in ("native","linear"):
                for sa,sb in itertools.combinations(FIT_SEEDS,2):
                    left,right=fits[sa],fits[sb]
                    for j in range(k):
                        beta_a=left[method+"_weights"][j]
                        beta_b=right[method+"_weights"][j]
                        all_pairs.append({"true_k_evaluation_only":true_k,
                             "method":method,"slot":j,"seed_a":sa,"seed_b":sb,
                             "same_slot_sensor_cosine":float(left["cert"].signature[:,j]@right["cert"].signature[:,j]),
                             "same_case_signed_effect_spearman":_rho(left[method+"_phi"][:,j],right[method+"_phi"][:,j]),
                             "both_nonzero_effect_coefficients":bool(min(abs(beta_a),abs(beta_b))>.08),
                             "same_effect_coefficient_sign":bool(np.sign(beta_a)==np.sign(beta_b))})
        # Negative and adversarial regimes: do not expect noise-shift refusal,
        # since source and source-aligned sensor noise are indistinguishable.
        for mode in ("no_shift","tied_profiles","mean_only_shift","sensor_noise_shift","noise_shift_aligned"):
            bad=make_environment_scm(k=true_k,seed=generator_base+700+true_k,
                per_environment=650,mode=mode)
            try:
                cb=fit_environment_contrast(bad.observations,bad.environment,
                    seed=537+true_k,permutations=99,bootstraps=24)
                found=True;reason=f"CERTIFIED {cb.selected_rank} contrast modes"
            except ValueError as exc:
                found=False;reason=str(exc)
            adversarial.append({"true_k_evaluation_only":true_k,"mode":mode,
                  "certified":found,"reason":reason,
                  "assumptions_violated":mode in ("sensor_noise_shift","noise_shift_aligned"),
                  "expected_must_refuse":mode in ("no_shift","mean_only_shift"),
                  "true_K_used_only_for_adversarial_audit":true_k})
    rows=pd.DataFrame(all_rows)
    certs=pd.DataFrame(cert_rows)
    pairs=pd.DataFrame(all_pairs)
    controls=pd.DataFrame(adversarial)
    if not controls.loc[controls.expected_must_refuse,"certified"].eq(False).all():
        raise AssertionError("Invalid negative control was incorrectly certified")
    if len(rows):
        summary=rows.groupby(["true_k_evaluation_only","method"]).agg(
           n_successful_refits=("fit_seed","nunique"),
           mean_auprc=("auprc","mean"),mean_auroc=("auroc","mean"),
           mean_brier=("brier","mean"),
           mean_optimistic_oracle_abs_spearman=("optimistic_source_abs_oracle_spearman","mean")
        ).reset_index()
    else:
        summary=pd.DataFrame()
    if len(pairs):
        stable=pairs.groupby(["true_k_evaluation_only","method","slot"]).agg(
            min_sensor_cosine=("same_slot_sensor_cosine","min"),
            min_signed_spearman=("same_case_signed_effect_spearman","min"),
            all_nonzero=("both_nonzero_effect_coefficients","all"),
            all_signs_agree=("same_effect_coefficient_sign","all")
        ).reset_index()
        stable["strict_cross_fit_signed_identification"]=(stable.min_sensor_cosine>.9)&(
            stable.min_signed_spearman>.5)&stable.all_nonzero&stable.all_signs_agree
    else:
        stable=pd.DataFrame()
    receipt={
       "status":"V14_CONDITIONAL_MULTI_ENVIRONMENT_COMPARISON_COMPLETE",
       "generator_base":generator_base,
       "stage":"exploratory" if generator_base==GENERATOR_SEEDS[0] else "independent_new_seed",
       "n_observations_per_environment":per_environment,
       "n_train_per_environment":train_per_environment,
       "n_evaluation_per_environment":per_environment-train_per_environment,
       "training_patient_subset_per_environment":sample_per_environment,
       "number_of_environment_groups":3,
       "latent_count_was_never_passed_to_fitting_algorithm":True,
       "no_known_action_labels_or_target_map":True,
       "holdout_oracle_only_used_after_fitting":True,
       "no_real_patient_or_previous_set_b_read":True,
       "comparators":["same_source_logistic","pooled_PCA_same_inferred_K_logistic","raw_sensor_logistic"],
       "abstention_count_by_true_k":{str(k):int((certs[certs.true_k_evaluation_only==k].status=="ABSTAIN").sum()) for k in K_VALUES},
       "limits":["Second-order blind source separation theory predates V14",
                 "Assumes independent factors and invariant sensor noise; these cannot be verified from covariance",
                 "K denotes detectable covariance-shift rank, not all possible biological factors",
                 "Data-specific sign/pivot and variance-ratio ordering are not semantic concept names",
                 "Environment-dependent sensor noise can fool the method",
                 "Purely synthetic data; one generator family and three refits"],
    }
    out_dir.mkdir(parents=True,exist_ok=False)
    for name,data in [("PER_FIT.csv",rows),("SOURCE_CERTIFICATES.csv",certs),
       ("CROSS_FIT_SIGNED_CONTRIBUTIONS.csv",pairs),("STRICT_SIGNED_SCREEN.csv",stable),
       ("NEGATIVE_CONTROLS.csv",controls),("SUMMARY.csv",summary)]:
        data.to_csv(out_dir/name,index=False)
    (out_dir/"STATUS.json").write_text(json.dumps(receipt,indent=2,allow_nan=False),encoding="utf8")
    print(summary.to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",required=True,type=Path)
    p.add_argument("--generator-base",type=int,default=GENERATOR_SEEDS[0])
    a=p.parse_args()
    run(a.out_dir,generator_base=a.generator_base)

if __name__=="__main__": main()
