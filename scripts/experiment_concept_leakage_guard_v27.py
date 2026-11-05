"""V27: fresh-draw, trained canonical ConceptFAN concept leakage control.

Concept-only pretraining vs outcome-joint concept training, same initialization,
gradient step count and concept supervision. Both use identical post-fit additive
linear risk heads, to isolate outcome-gradients entering the native concept path.
TEST is a NEW draw under an ALREADY inspected generator FAMILY, not external
medical confirmation or independently novel identifiability theorem.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fan.concept.completeness_audit import _paired_bootstrap_ci, _per_sample_logloss
from scripts.experiment_frozen_concept_head_refit_v26 import (
    refit_linear_concept_head
)
from scripts.experiment_trained_conceptfan_omission_v25 import (
    BATCH, CHANNELS, EPOCHS, FULL_CAL, FULL_TEST, FULL_TRAIN, LR, LAMBDA_C,
    NN_SEEDS, STRESS, _call, external_logistic, fit_native,
    flatten_measured_inputs, frozen_world, make_model,
    mask_mechanism, metrics, predict_native, sha_state,
)

NEW_SEED=20261107
TRAIN_MASK_SEED=20262101
TEST_MASK_BASE=20263300
MODES=("MCAR20","MCAR40","MNAR")


def fit_concepts_without_outcome_gradients(model, values, masks, oracle_concepts,
                                           *, seed:int, epochs:int):
    """Fixed native gradient-step schedule; no task label ever enters this phase."""
    model.train()
    xx=torch.from_numpy(values.astype(np.float32))
    mm=torch.from_numpy(masks.astype(np.float32))
    cc=torch.from_numpy(oracle_concepts[:,:,:2].astype(np.float32))
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=.001)
    rng=np.random.default_rng(82000+seed)
    first=last=None
    for epoch in range(epochs):
        perm=rng.permutation(len(values))
        losses=[]
        for start in range(0,len(perm),BATCH):
            idx=perm[start:start+BATCH]
            native,_=_call(model,xx[idx],mm[idx])
            if native.concept_trajectories.shape!=cc[idx].shape:
                raise AssertionError("Concept-only stage uses exactly two synthetic concepts")
            objective=LAMBDA_C*F.mse_loss(native.concept_trajectories,cc[idx])
            if not torch.isfinite(objective):
                raise FloatingPointError("Nonfinite pure concept objective")
            opt.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            losses.append(float(objective.detach()))
        if epoch==0:first=float(np.mean(losses))
        last=float(np.mean(losses))
    return {"epochs":epochs,"first_epoch_concept_loss":first,"final_epoch_concept_loss":last,
            "outcome_gradient_in_concept_stage":False}


def fit_logistic_probe(x_train,y_train,x_test):
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=1000))
    clf.fit(np.asarray(x_train,dtype=np.float64),y_train)
    return np.clip(clf.predict_proba(np.asarray(x_test,dtype=np.float64))[:,1],
                   1e-7,1-1e-7)


def conditional_leakage_audit(y_train,y_test,c_gold_train,c_gold_test,
                              concept_pred_train,concept_pred_test,*,seed:int,
                              bootstraps:int):
    """Whether learned K2 activations carry extra TEST outcome signal beyond C*2."""
    g0=fit_logistic_probe(c_gold_train,y_train,c_gold_test)
    g1=fit_logistic_probe(
        np.column_stack([c_gold_train,concept_pred_train]),y_train,
        np.column_stack([c_gold_test,concept_pred_test]))
    differences=_per_sample_logloss(y_test,g0)-_per_sample_logloss(y_test,g1)
    interval=_paired_bootstrap_ci(differences,seed=seed,replicates=bootstraps)
    return {
        "n_test":len(y_test),
        "oracle_only_test_nll":float(_per_sample_logloss(y_test,g0).mean()),
        "oracle_plus_learned_concepts_test_nll":float(_per_sample_logloss(y_test,g1).mean()),
        "incremental_learned_concept_nll_gain":float(differences.mean()),
        "paired_test_subject_bootstrap_ci95":list(interval),
        "positive_ci":bool(interval[0]>0),
        "observational_extra_information_not_semantic_identity":True,
    }


def conditional_third_factor_alignment(c_true_train2,c_true_test2,
                                       q_train,q_test,c_true_test3):
    """Eval-only association of model concept error with omitted oracle factor."""
    ridge=Ridge(alpha=1.0)
    ridge.fit(c_true_train2,q_train)
    residual=q_test-ridge.predict(c_true_test2)
    scores=[]
    for k in range(residual.shape[1]):
        x=residual[:,k]
        corr=float(np.corrcoef(x,c_true_test3)[0,1]) if np.std(x)>1e-10 else 0.
        scores.append(corr)
    return {"residual_concept_to_true_hidden_c3_pearson":scores,
            "max_abs_hidden_factor_correlation":float(max(map(abs,scores))),
            "oracle_c3_used_only_for_evaluation":True}


def run(*,out_dir:Path,smoke:bool=False):
    if out_dir.exists():
        raise FileExistsError("Existing V27 evidence is immutable")
    torch.set_num_threads(2)
    ntrain,ncal,ntest=((160,40,160) if smoke else
                        (FULL_TRAIN,FULL_CAL,FULL_TEST))
    epochs=2 if smoke else EPOCHS
    seeds=(11,) if smoke else NN_SEEDS
    bootstraps=199 if smoke else 499
    world=frozen_world(seed=NEW_SEED,n_train=ntrain,n_cal=ncal,n_test=ntest)
    tr,te=world["index_train"],world["index_test"]
    x,c=world["x"],world["concept_trajectories"]
    mtrain=mask_mechanism(x[tr],"MCAR20",seed=TRAIN_MASK_SEED)
    masks={mode:mask_mechanism(x[te],mode,seed=TEST_MASK_BASE+STRESS.index(mode))
           for mode in MODES}
    rows,leakage,training=[],[],[]
    stage_state_by_seed={}
    true_train2=c[tr,:,:2].mean(axis=1)
    true_test2=c[te,:,:2].mean(axis=1)
    true_test3=c[te,:,2].mean(axis=1)
    for family in ("complete","omitted"):
        y=world["y_complete"] if family=="complete" else world["y_omitted"]
        raw=external_logistic(x[tr],mtrain,y[tr])
        oracle_2=fit_logistic_probe(true_train2,y[tr],true_test2)
        oracle_3=fit_logistic_probe(c[tr].mean(axis=1),y[tr],c[te].mean(axis=1))
        for mode,mask in masks.items():
            raw_p=raw.predict_proba(flatten_measured_inputs(x[te],mask))[:,1]
            for method,p in (("raw_logistic",raw_p),
                             ("oracle_true_k2",oracle_2),("oracle_true_k3",oracle_3)):
                rows.append({"family":family,"seed":None,"mode":mode,
                             "method":method,"test":metrics(y[te],p),
                             "concept_mse":None,"additive_error":None})
        for seed in seeds:
            for mode in ("joint","staged"):
                model=make_model("native",2,seed)
                if mode=="joint":
                    train_stats=fit_native(model,x[tr],mtrain,c[tr],y[tr],
                                           seed=seed,epochs=epochs)
                else:
                    train_stats=fit_concepts_without_outcome_gradients(
                        model,x[tr],mtrain,c[tr],seed=seed,epochs=epochs)
                frozen_sha=sha_state(model)
                if mode=="staged":
                    if seed in stage_state_by_seed and stage_state_by_seed[seed]!=frozen_sha:
                        raise AssertionError("Outcome labels changed label-free concept formation")
                    stage_state_by_seed[seed]=frozen_sha
                training.append({"family":family,"seed":seed,"mode":mode,
                                 "core_sha256":frozen_sha,**train_stats})
                train_out=predict_native(model,x[tr],mtrain,c[tr])
                first_test=predict_native(model,x[te],masks["MCAR20"],c[te])
                linear=refit_linear_concept_head(train_out["concept_summary"],y[tr])
                audit=conditional_leakage_audit(
                    y[tr],y[te],true_train2,true_test2,
                    train_out["concept_summary"],first_test["concept_summary"],
                    seed=901210+seed,bootstraps=bootstraps)
                association=conditional_third_factor_alignment(
                    true_train2,true_test2,
                    train_out["concept_summary"],first_test["concept_summary"],
                    true_test3)
                leakage.append({"family":family,"seed":seed,"method":mode,
                                **audit,**association})
                for stress,mask in masks.items():
                    predicted=first_test if stress=="MCAR20" else predict_native(
                        model,x[te],mask,c[te])
                    logit,contrib=linear(predicted["concept_summary"])
                    err=float(np.max(np.abs(logit-(linear.bias+contrib.sum(axis=1)))))
                    if err>1e-10 or predicted["max_additive_error"]>2e-5:
                        raise AssertionError("Two-stage model lost exact concept-only additive decision")
                    rows.append({"family":family,"seed":seed,"mode":stress,
                                 "method":mode,
                                 "test":metrics(y[te],expit(logit)),
                                 "concept_mse":predicted["concept_mse"],
                                 "additive_error":err,
                                 "max_original_native_fuzzy_error":predicted["max_additive_error"],
                                 "linear_head_bias":linear.bias,
                                 "linear_head_weights":list(linear.weights)})
                if sha_state(model)!=frozen_sha:
                    raise AssertionError("TEST models were modified after training")
                score=next(z["test"] for z in rows if z["family"]==family
                           and z["seed"]==seed and z["method"]==mode
                           and z["mode"]=="MCAR20")
                print(json.dumps({"event":"V27_FIT","family":family,"seed":seed,
                    "training":mode,"test_auprc":score["auprc"],
                    "test_nll":score["nll"],
                    "leakage_nll_gain":audit["incremental_learned_concept_nll_gain"],
                    "leakage_ci":audit["paired_test_subject_bootstrap_ci95"],
                    "alignment":association["max_abs_hidden_factor_correlation"]}),flush=True)
    for collection in (rows,leakage,training):
        for row in list(collection):
            if row["family"]=="omitted":
                collection.append({**row,"family":"artifact",
                                   "observational_world_is_exact_omitted_duplicate":True})
    summary=[]
    for family in ("complete","omitted","artifact"):
        for stress in MODES:
            for method in ("raw_logistic","oracle_true_k2","oracle_true_k3","joint","staged"):
                group=[row for row in rows if row["family"]==family
                       and row["method"]==method and row["mode"]==stress]
                expected=1 if method in ("raw_logistic","oracle_true_k2","oracle_true_k3") else len(seeds)
                assert len(group)==expected
                summary.append({"family":family,"mode":stress,"method":method,
                    "fit_count":len(group),
                    "mean_auprc":float(np.mean([z["test"]["auprc"] for z in group])),
                    "mean_nll":float(np.mean([z["test"]["nll"] for z in group])),
                    "mean_brier":float(np.mean([z["test"]["brier"] for z in group])),
                    "mean_concept_mse":(
                        float(np.mean([z["concept_mse"] for z in group]))
                        if method in ("joint","staged") else None)})
    report={
        "status":"SMOKE_ONLY" if smoke else "V27_NEW_DRAW_CONCEPT_TASK_LEAKAGE_EXPERIMENT",
        "protocol":"docs/medical/IOI_V27_FROZEN_CONCEPT_LEAKAGE_PROTOCOL_20261106.md",
        "new_draw_from_previous_simulator_family":True,
        "generator_seed":NEW_SEED,
        "train_mask_seed":TRAIN_MASK_SEED,
        "original_piERE_camera_ready_unchanged":True,
        "no_semantic_causal_or_clinical_identity_proven":True,
        "artifact_observationally_same_as_omitted":True,
        "stage_pure_concept_path_independent_of_outcome_law":True,
        "train_concept_targets_are_synthetic_oracle_labels":True,
        "sample_sizes":{"train":ntrain,"unused_calibration":ncal,"test":ntest},
        "nn_seeds":list(seeds),"model_epochs":epochs,"model_batch":BATCH,
        "n_training_runs":int(2*len(seeds)*2),
        "frozen_true_data_sha256":sha256(
            np.ascontiguousarray(x).tobytes()
            +np.ascontiguousarray(c).tobytes()
            +np.ascontiguousarray(world["y_complete"]).tobytes()
            +np.ascontiguousarray(world["y_omitted"]).tobytes()).hexdigest(),
        "rows":rows,"leakage_audit":leakage,
        "training":training,"summary":summary,
    }
    out_dir.mkdir(parents=True,exist_ok=False)
    (out_dir/"STATUS.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    (out_dir/"SUMMARY.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    (out_dir/"LEAKAGE.json").write_text(json.dumps(leakage,indent=2,allow_nan=False))
    print(json.dumps({"event":"V27_DONE","status":report["status"],
          "n_training_runs":report["n_training_runs"],"n_rows":len(rows),
          "n_diagnostic_rows":len(leakage),
          "primary_summary":[z for z in summary
                             if z["mode"]=="MCAR20" and z["family"]!="artifact"],
          "primary_leakage":[{"family":z["family"],"method":z["method"],"seed":z["seed"],
                              "gain":z["incremental_learned_concept_nll_gain"],
                              "ci":z["paired_test_subject_bootstrap_ci95"]}
                              for z in leakage if z["family"]!="artifact"]}),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    run(out_dir=args.out_dir,smoke=args.smoke)


if __name__=="__main__":
    main()
