"""V39 independent recomputation of frozen tested ON/OFF source ablation.

Read ONLY model-emitted raw synthetic TEST patient probabilities and labels
after the full experiment. Recompute every per-seed and averaged metric.
Calculate exploratory paired TEST-subject NLL confidence intervals on the
PRE-REGISTERED 20 method/world contrasts, while never refitting a model,
tuning source threshold, choosing failure scenario or touching VALID gates.

The intervals are CONDITIONAL on 3 neural K2 models and ONE synthetic
600-episode TEST cohort per each of two devices, never population-wide.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    average_precision_score,roc_auc_score,log_loss,brier_score_loss,
)

NATIVE={"K2","ON_AB","ON_ABC","DIFFERENCE_AB","DIFFERENCE_ABC"}
SEEDS=(11,23,37)


def audit_v39(evidence:Path,output:Path,*,bootstrap_reps:int=399):
    if output.exists():
        raise FileExistsError("V39 audit must not overwrite old scientific evidence")
    d=json.loads((evidence/"STATUS.json").read_text())
    if (d["status"]!="V39_PAIRED_ACTIVE_SOURCE_ISOLATION_EXECUTED"
        or bootstrap_reps!=399):
        raise ValueError("Only full frozen V39 executed results and 399 bootstrap draws")
    archive=np.load(evidence/"FROZEN_TEST_PREDICTIONS.npz",allow_pickle=False)
    err=0.
    for row in d["test_rows"]:
        device,case,stress,method,nn_seed=(
            row["device_seed"],row["case"],row["stress"],
            row["method"],row["NN_seed"])
        yy=archive[f"device{device}_{case}_TEST_Y"]
        pp=archive[f"device{device}_{case}_{stress}_{method}_seed{nn_seed}"]
        if yy.shape!=(600,) or pp.shape!=(600,) or not np.isin(yy,[0,1]).all():
            raise AssertionError("TEST subject IDs/membership not preserved")
        p=np.clip(pp,1e-7,1-1e-7)
        metrics={
            "auprc":average_precision_score(yy,p),
            "auroc":roc_auc_score(yy,p),
            "nll":log_loss(yy,p,labels=[0,1]),
            "brier":brier_score_loss(yy,p),
        }
        for k,v in metrics.items():
            err=max(err,abs(float(v)-row["metrics"][k]))
    if err>1e-12:
        raise AssertionError(f"Frozen TEST metrics differ: {err}")
    for group in d["summary"]:
        rows=[v for v in d["test_rows"] if all(v[k]==group[k]
             for k in ("device_seed","case","stress","method"))]
        if not rows:
            raise AssertionError("Aggregate without source observations")
        for name,key in (("AUPRC","auprc"),("AUROC","auroc"),
                         ("NLL","nll"),("Brier","brier")):
            if abs(np.mean([x["metrics"][key] for x in rows])-group[name])>1e-12:
                raise AssertionError("Aggregated score does not reproduce real TEST")
    repeated={}
    for device in d["devices"]:
        for case in ("TEST_aligned_artifact_4x","TEST_aligned_artifact_8x"):
            differences=[]
            for stress in ("MCAR20","MCAR40",
                           "synchronized_block8","complete"):
                for seed in SEEDS:
                    healthy=archive[
                        f"device{device}_healthy_{stress}_DIFFERENCE_ABC_seed{seed}"]
                    altered=archive[
                        f"device{device}_{case}_{stress}_DIFFERENCE_ABC_seed{seed}"]
                    differences.append(float(np.max(abs(healthy-altered))))
            repeated[(device,case)]=max(differences)
    missing_exact=0
    for dev in d["devices"]:
        for case in sorted({c["case"] for c in d["summary"]
                            if c["stress"]=="MCAR20" and c["method"]=="K2"}):
            for stress in ("missing_reference",):
                for seed in SEEDS:
                    k2=archive[f"device{dev}_{case}_{stress}_K2_seed{seed}"]
                    for m in ("DIFFERENCE_AB","DIFFERENCE_ABC"):
                        p=archive[f"device{dev}_{case}_{stress}_{m}_seed{seed}"]
                        np.testing.assert_array_equal(p,k2)
                        missing_exact+=1
    nll=[]
    for device in d["devices"]:
        for j,c in enumerate([z for z in d["paired_TEST_AP_CI"]
                               if z["device_seed"]==device]):
            case,stress,a,b=(c[k] for k in
                ("case","stress","method_a","method_b"))
            y=archive[f"device{device}_{case}_TEST_Y"].astype(np.float64)
            def groups(method):
                seeds=SEEDS if method in NATIVE else (None,)
                p=np.stack([
                    archive[f"device{device}_{case}_{stress}_{method}_seed{seed}"]
                    for seed in seeds
                ])
                if len(p)==1:
                    p=np.repeat(p,len(SEEDS),axis=0)
                return np.clip(p,1e-7,1-1e-7)
            pa,pb=groups(a),groups(b)
            loss_a=-(y[None,:]*np.log(pa)+(1-y)[None,:]*np.log1p(-pa))
            loss_b=-(y[None,:]*np.log(pb)+(1-y)[None,:]*np.log1p(-pb))
            delta=loss_a-loss_b
            rng=np.random.default_rng(20264219+device%101*100+j)
            picks=rng.integers(0,len(y),size=(bootstrap_reps,len(y)))
            draws=delta[:,picks].mean(axis=(0,2))
            ci=np.quantile(draws,[.025,.975])
            nll.append({
                "device_seed":device,"case":case,"stress":stress,
                "method_a":a,"method_b":b,
                "delta_NLL_a_minus_b":float(delta.mean()),
                "paired_TEST_subject_bootstrap_95pct_ci_NLL":ci.tolist(),
                "n_shared_test_subjects":len(y),
                "fixed_nn_fit_seeds":list(SEEDS),
                "bootstraps":bootstrap_reps,
                "exploratory_complement_to_pre_registered_AP_contrast":True,
            })
    d2={
        "status":"V39_INDEPENDENT_METRIC_AUDIT",
        "verified_test_rows":len(d["test_rows"]),
        "verified_summary_rows":len(d["summary"]),
        "max_metric_error":err,
        "n_exact_missing_reference_preserved_K2_forecasts":missing_exact,
        "max_absolute_paired_aligned_artifact_prediction_difference":{
            f"device{dev}_{case}":val
            for (dev,case),val in repeated.items()},
        "n_paired_NLL_bootstrap_exploratory":len(nll),
        "scientific_limitations":{
            "new_EXTERNAL_physical_source_isolation_required":True,
            "ON_OFF_nuisance_match_NOT_empirically_proven":True,
            "source_causal_semantics_not_established":True,
            "no_TEST_model_selection_or_threshold_change":True,
        },
        "paired_NLL_bootstrap":nll,
    }
    if (d2["verified_test_rows"]!=2380 or
        d2["verified_summary_rows"]!=980 or
        d2["n_paired_NLL_bootstrap_exploratory"]!=20 or
        missing_exact!=2*10*3*2):
        raise AssertionError("Missing frozen test receipts or reference abstentions")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(
        d2,indent=2,ensure_ascii=False,allow_nan=False))
    print(json.dumps({
        "event":"V39_INDEPENDENT_METRIC_AUDIT",
        "verified_test_rows":len(d["test_rows"]),
        "verified_summary_rows":len(d["summary"]),
        "max_metric_error":err,
        "n_reference_abstentions":missing_exact,
        "paired_test_NLL_CI":len(nll),
        "max_matched_artifact_difference":max(repeated.values()),
    }),flush=True)
    return d2


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--evidence-dir",type=Path,required=True)
    parser.add_argument("--out-file",type=Path,required=True)
    args=parser.parse_args()
    audit_v39(args.evidence_dir,args.out_file)
