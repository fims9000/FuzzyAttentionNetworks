"""V26: post-V25 exploratory head refit, not new independent TEST evidence.

Train ACTUAL native ConceptFAN encoder/projector on same synthetic law, freeze,
then compare original fuzzy head, positive Platt, concept-only logistic head,
oracle concept logistic and the raw-input logistic baseline. The new linear head
is mathematically additive in native concept coordinates; it is a CBM baseline,
not a newly discovered physiological concept or a causal explanation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.experiment_trained_conceptfan_omission_v25 import (
    BATCH, CHANNELS, EPOCHS, FULL_CAL, FULL_TEST, FULL_TRAIN, NN_SEEDS,
    STRESS, external_logistic, fit_native, flatten_measured_inputs, frozen_world,
    make_model, mask_mechanism, metrics, predict_native, sha_state,
)


@dataclass(frozen=True)
class LinearNativeHead:
    """Exactly additive *original-coordinate* decision over native concept summaries."""
    bias: float
    weights: tuple[float, ...]

    def __call__(self, concept_summaries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q=np.asarray(concept_summaries,dtype=np.float64)
        w=np.asarray(self.weights,dtype=np.float64)
        if q.ndim!=2 or q.shape[1]!=len(w) or not np.isfinite(q).all():
            raise ValueError("Invalid concept-only readout, no raw feature bypass")
        contributions=q*w[None,:]
        logits=self.bias+contributions.sum(axis=1)
        return logits,contributions


def refit_linear_concept_head(train_concepts: np.ndarray,
                              train_labels: np.ndarray) -> LinearNativeHead:
    pipe=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=1000))
    q=np.asarray(train_concepts,dtype=np.float64)
    pipe.fit(q,train_labels)
    scale=pipe.named_steps["standardscaler"]
    clf=pipe.named_steps["logisticregression"]
    weights=clf.coef_[0]/scale.scale_
    bias=float(clf.intercept_[0]-np.dot(weights,scale.mean_))
    result=LinearNativeHead(bias,tuple(map(float,weights)))
    our_logits,_=result(q)
    np.testing.assert_allclose(our_logits,pipe.decision_function(q),
                               rtol=1e-8,atol=1e-8)
    return result


@dataclass(frozen=True)
class PositiveAffineCalibration:
    shift: float
    slope: float

    def __call__(self, logits: np.ndarray) -> np.ndarray:
        if self.slope <= 0.:
            raise ValueError("Native calibration slope must remain positive")
        return self.shift+self.slope*np.asarray(logits,dtype=np.float64)


def refit_positive_platt(train_logits: np.ndarray,
                         train_y: np.ndarray) -> PositiveAffineCalibration:
    z=np.asarray(train_logits,dtype=np.float64)
    y=np.asarray(train_y,dtype=np.float64)
    if z.ndim != 1 or z.shape!=y.shape or len(np.unique(y))!=2:
        raise ValueError("TRAIN-only logit regression requires two classes")
    def objective(v):
        logscale,shift=map(float,v)
        calibrated=np.exp(logscale)*z+shift
        return float((np.logaddexp(0,calibrated)-y*calibrated).mean()
                     +.002*(logscale**2+shift**2))
    fitted=minimize(objective,[0.0,0.0],method="L-BFGS-B",
                    bounds=[(-3.,3.),(-7.,7.)],
                    options={"maxiter":250})
    if not fitted.success or not np.isfinite(fitted.fun):
        raise RuntimeError("Train-only native Platt calibration did not converge")
    return PositiveAffineCalibration(shift=float(fitted.x[1]),
                                     slope=float(np.exp(fitted.x[0])))


def run(out_dir: Path, *, smoke: bool=False) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError("V26 diagnostic evidence is immutable")
    torch.set_num_threads(2)
    ntrain,ncal,ntest=((160,40,160) if smoke else
                        (FULL_TRAIN,FULL_CAL,FULL_TEST))
    seeds=(11,) if smoke else NN_SEEDS
    epochs=2 if smoke else EPOCHS
    world=frozen_world(n_train=ntrain,n_cal=ncal,n_test=ntest)
    tr,te=world["index_train"],world["index_test"]
    x,c=world["x"],world["concept_trajectories"]
    mtrain=mask_mechanism(x[tr],"MCAR20",seed=20262001)
    modes=("MCAR20","MCAR40","MNAR")
    test_masks={mode:mask_mechanism(x[te],mode,seed=20262200+STRESS.index(mode))
                for mode in modes}
    rows,heads,training=[],[],[]
    for family in ("complete","omitted"):
        y=world["y_complete"] if family=="complete" else world["y_omitted"]
        raw=external_logistic(x[tr],mtrain,y[tr])
        for mode,mask in test_masks.items():
            predictions=raw.predict_proba(flatten_measured_inputs(x[te],mask))[:,1]
            rows.append({"family":family,"seed":None,"concepts":None,
                         "method":"raw_logistic","mode":mode,
                         "test":metrics(y[te],predictions),
                         "additive_error":None})
        for k in (2,3):
            oracle_train=c[tr,:,:k].mean(axis=1).astype(np.float64)
            oracle_test=c[te,:,:k].mean(axis=1).astype(np.float64)
            ceiling=refit_linear_concept_head(oracle_train,y[tr])
            logits,_=ceiling(oracle_test)
            for mode in modes:
                rows.append({"family":family,"seed":None,"concepts":k,
                             "method":"oracle_true_concept_linear_ceiling",
                             "mode":mode,"test":metrics(y[te],expit(logits)),
                             "additive_error":0.0})
        for seed in seeds:
            for k in (2,3):
                core=make_model("native" if k==2 else "oracle_k3",k,seed)
                receipt=fit_native(core,x[tr],mtrain,c[tr],y[tr],
                                   seed=seed,epochs=epochs)
                trained_state=sha_state(core)
                training.append({"family":family,"seed":seed,"k":k,
                                 "native_state_sha256":trained_state,**receipt})
                predicted_train=predict_native(core,x[tr],mtrain,c[tr])
                linear=refit_linear_concept_head(predicted_train["concept_summary"],y[tr])
                platt=refit_positive_platt(predicted_train["logits"],y[tr])
                heads.append({"family":family,"seed":seed,"k":k,
                              "linear_native_bias":linear.bias,
                              "linear_native_weights":list(linear.weights),
                              "platt_shift":platt.shift,"platt_positive_slope":platt.slope})
                for mode,mask in test_masks.items():
                    tested=predict_native(core,x[te],mask,c[te])
                    orig_logit=tested["logits"]
                    cal_logit=platt(orig_logit)
                    linear_logit,contributions=linear(tested["concept_summary"])
                    if not np.isfinite(linear_logit).all() or not np.isfinite(cal_logit).all():
                        raise FloatingPointError("Nonfinite frozen concept-head refit output")
                    exact=float(np.max(np.abs(
                        linear_logit-(linear.bias+contributions.sum(axis=1)))))
                    if exact>1e-10 or tested["max_additive_error"]>2e-5:
                        raise AssertionError("Refitted model violates strict additive concept pathway")
                    if not np.array_equal(np.argsort(orig_logit),np.argsort(cal_logit)):
                        raise AssertionError("Positive Platt calibration changed outcome ranking")
                    for method,logits in (("native_fuzzy",orig_logit),
                                          ("positive_platt",cal_logit),
                                          ("frozen_concept_linear_refit",linear_logit)):
                        rows.append({"family":family,"seed":seed,"concepts":k,
                                     "method":method,"mode":mode,
                                     "test":metrics(y[te],expit(logits)),
                                     "concept_mse":tested["concept_mse"],
                                     "additive_error":(exact if method=="frozen_concept_linear_refit"
                                                       else tested["max_additive_error"])})
                if sha_state(core)!=trained_state:
                    raise AssertionError("Exploratory TEST pass changed frozen native model weights")
                native_mc=next(r["test"] for r in rows if r["family"]==family
                               and r["seed"]==seed and r["concepts"]==k
                               and r["method"]=="native_fuzzy" and r["mode"]=="MCAR20")
                refit_mc=next(r["test"] for r in rows if r["family"]==family
                               and r["seed"]==seed and r["concepts"]==k
                               and r["method"]=="frozen_concept_linear_refit" and r["mode"]=="MCAR20")
                print(json.dumps({"event":"V26_TRAINED","family":family,"seed":seed,
                    "k":k,"native_ap":native_mc["auprc"],"refit_ap":refit_mc["auprc"],
                    "native_nll":native_mc["nll"],"refit_nll":refit_mc["nll"]}),flush=True)
    # ARTIFACT is the exact same observed world as OMITTED; no fake repeats.
    for collection in (rows,heads,training):
        for row in list(collection):
            if row["family"]=="omitted":
                collection.append({**row,"family":"artifact",
                                   "same_observed_dataset_and_fitted_model":True})
    summary=[]
    for family in ("complete","omitted","artifact"):
        for mode in modes:
            for k in (2,3):
                for method in ("native_fuzzy","positive_platt",
                               "frozen_concept_linear_refit",
                               "oracle_true_concept_linear_ceiling"):
                    selected=[r for r in rows if r["family"]==family
                              and r["mode"]==mode and r["concepts"]==k
                              and r["method"]==method]
                    if len(selected)!=(1 if method=="oracle_true_concept_linear_ceiling"
                                       else len(seeds)):
                        raise AssertionError("Missing or duplicated concept-head comparison")
                    summary.append({"family":family,"mode":mode,"k":k,"method":method,
                        "mean_auprc":float(np.mean([r["test"]["auprc"] for r in selected])),
                        "mean_nll":float(np.mean([r["test"]["nll"] for r in selected])),
                        "mean_brier":float(np.mean([r["test"]["brier"] for r in selected])),
                        "n_fits":len(selected)})
            raw=[r for r in rows if r["family"]==family and r["mode"]==mode
                 and r["method"]=="raw_logistic"]
            assert len(raw)==1
            summary.append({"family":family,"mode":mode,"k":None,"method":"raw_logistic",
                            "mean_auprc":float(raw[0]["test"]["auprc"]),
                            "mean_nll":float(raw[0]["test"]["nll"]),
                            "mean_brier":float(raw[0]["test"]["brier"]), "n_fits":1})
    status={"status":"V26_POST_V25_EXPLORATORY_HEAD_DIAGNOSTIC",
            "not_independent_confirmation":True,
            "test_generator_already_inspected_in_V25":True,
            "extra_head_optimizer_compute_not_matched":True,
            "same_observed_artifact_and_omitted_world":True,
            "third_concept_has_synthetic_privileged_supervision":True,
            "no_raw_or_latent_to_risk_bypass":True,
            "samples":{"train":ntrain,"untouched_cal":ncal,"test":ntest},
            "NN_seeds":list(seeds),"modes":list(modes),
            "n_distinct_trained_native_models":int(2*len(seeds)*2),
            "rows":rows,"head_coefficients":heads,"training":training,
            "summary":summary}
    out_dir.mkdir(parents=True,exist_ok=False)
    (out_dir/"STATUS.json").write_text(json.dumps(status,indent=2,allow_nan=False))
    (out_dir/"SUMMARY.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps({"event":"V26_COMPLETE","status":status["status"],
          "n_distinct_trained_native_models":status["n_distinct_trained_native_models"],
          "n_rows":len(rows),"n_summaries":len(summary),
          "key_results":[r for r in summary
                         if r["family"]!="artifact" and r["mode"]=="MCAR20"]}),flush=True)
    return status


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    run(args.out_dir,smoke=args.smoke)


if __name__=="__main__":
    main()
