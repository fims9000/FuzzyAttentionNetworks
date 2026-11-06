"""V38 independent numerical audit of frozen synthetic TEST receipts.

Recomputes every AP/AUROC/NLL/Brier cell from saved predictions and labels.
Then conducts only the 18 contrasts PRE-REGISTERED in the frozen V38
protocol, using a paired 399-resample test-subject bootstrap of NLL,
conditional on the same 3 fitted models. NO new model, threshold, source
admission or test-driven rejection is performed by this script.

Usage:
PYTHONPATH=src:. python scripts/audit_exogenous_probe_v38.py
  --evidence-dir V38_EVIDENCE --out-file V38_EVIDENCE/INDEPENDENT_NLL_AUDIT.json
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    average_precision_score,roc_auc_score,log_loss,brier_score_loss,
)

NN_SEEDS=(11,23,37)
NN_METHODS={"K2","AB_centered","ABC_threeview",
            "ABC_monitor","ABC_probe_monitor"}


def audit(evidence_dir:Path,out_file:Path,*,resamples:int=399):
    if out_file.exists():
        raise FileExistsError("Do not overwrite original frozen audit")
    if resamples!=399:
        raise ValueError("Registered bootstrap count is 399")
    d=json.loads((evidence_dir/"STATUS.json").read_text())
    if d["status"]!="V38_INSTRUMENTED_CONCEPT_EXECUTED":
        raise ValueError("Only executed full frozen V38 TEST receipts")
    data=np.load(evidence_dir/"FROZEN_TEST_PREDICTIONS.npz",
                 allow_pickle=False)
    max_err=0.
    for r in d["test_rows"]:
        dev,case,stress,method,seed=(
            r["device_seed"],r["case"],r["stress"],
            r["method"],r["NN_seed"])
        y=data[f"device{dev}_{case}_TEST_LABELS"]
        raw=data[f"device{dev}_{case}_{stress}_{method}_seed{seed}"]
        if (y.shape!=(600,) or raw.shape!=(600,)
            or not np.isin(y,[0,1]).all()
            or not np.isfinite(raw).all()):
            raise AssertionError("Saved per-subject probability mismatch")
        p=np.clip(raw,1e-7,1-1e-7)
        scores={
            "auprc":average_precision_score(y,p),
            "auroc":roc_auc_score(y,p),
            "nll":log_loss(y,p,labels=[0,1]),
            "brier":brier_score_loss(y,p),
        }
        for key,v in scores.items():
            max_err=max(max_err,abs(float(v)-r["metrics"][key]))
    if max_err>1e-12:
        raise AssertionError(f"TEST scores disagree with raw evidence: {max_err}")
    for row in d["summary"]:
        selected=[r for r in d["test_rows"] if all(r[k]==row[k]
            for k in ("device_seed","case","stress","method"))]
        if not selected:
            raise AssertionError("Aggregate without source TEST subjects")
        for k,v in (
            ("AUPRC","auprc"),("AUROC","auroc"),
            ("NLL","nll"),("Brier","brier"),
        ):
            e=abs(float(np.mean([s["metrics"][v] for s in selected]))-row[k])
            if e>1e-12:
                raise AssertionError("Aggregation fails: "+str(row))
    ablation_equal=[]
    bootstrap=[]
    for dev in d["devices"]:
        for case in {r["case"] for r in d["summary"]}:
            for stress in {r["stress"] for r in d["summary"]}:
                for seed in NN_SEEDS:
                    # A separate external ACTUATOR validation certificate
                    # should NOT secretly change predictions when it passes:
                    # sentinel+probe vs sentinel alone must be identical.
                    a=data[f"device{dev}_{case}_{stress}_ABC_monitor_seed{seed}"]
                    b=data[f"device{dev}_{case}_{stress}_ABC_probe_monitor_seed{seed}"]
                    if not np.array_equal(a,b):
                        raise AssertionError(
                            "Probe validation passed but changed operational predictions"
                        )
                    ablation_equal.append(True)
        comparisons=[r for r in d["paired_AP_bootstrap"]
                     if r["device_seed"]==dev]
        for i,c in enumerate(comparisons):
            case,stress,a,b=(c[k] for k in
                ("case","stress","method_a","method_b"))
            y=data[f"device{dev}_{case}_TEST_LABELS"].astype(np.float64)
            def group(method):
                seeds=NN_SEEDS if method in NN_METHODS else (None,)
                v=np.stack([
                    data[f"device{dev}_{case}_{stress}_{method}_seed{s}"]
                    for s in seeds
                ])
                if len(seeds)==1:
                    v=np.repeat(v,len(NN_SEEDS),axis=0)
                return np.clip(v,1e-7,1-1e-7)
            pa,pb=group(a),group(b)
            losses_a=-(y[None,:]*np.log(pa)+
                       (1-y)[None,:]*np.log1p(-pa))
            losses_b=-(y[None,:]*np.log(pb)+
                       (1-y)[None,:]*np.log1p(-pb))
            difference=losses_a-losses_b
            observed=float(difference.mean())
            rng=np.random.default_rng(20264099+dev%101*100+i)
            picks=rng.integers(0,len(y),size=(resamples,len(y)))
            draws=difference[:,picks].mean(axis=(0,2))
            low,high=[float(z) for z in np.quantile(draws,[.025,.975])]
            bootstrap.append({
                "device_seed":dev,"case":case,"stress":stress,
                "method_a":a,"method_b":b,
                "paired_delta_NLL_a_minus_b":observed,
                "paired_TEST_subject_95pct_bootstrap_CI_NLL":[low,high],
                "n_test_shared_subjects":len(y),
                "n_fixed_native_model_seeds":len(NN_SEEDS),
                "resamples":resamples,
                "conditioned_on_single_synthetic_cohort_per_device":True,
                "postrun_numerical_audit_not_test_based_parameter_selection":True,
            })
    if (len(d["test_rows"])!=2380 or len(d["summary"])!=980
        or len(d["paired_AP_bootstrap"])!=18
        or len(bootstrap)!=18 or len(ablation_equal)!=420):
        raise AssertionError("Insufficient V38 frozen experimental evidence")
    result={
        "status":"V38_INDEPENDENT_METRIC_AND_PAIRED_NLL_AUDIT",
        "n_exactly_reproduced_TEST_rows":len(d["test_rows"]),
        "n_exactly_reproduced_summary_rows":len(d["summary"]),
        "max_abs_error_all_4_metrics":max_err,
        "n_exact_probe_pass_ablation_equivalences":len(ablation_equal),
        "n_pairwise_subject_bootstrap_NLL_receipts":len(bootstrap),
        "external_probe_certificate_changes_prediction_if_passes":False,
        "frozen_TEST_thresholds_and_VALID_decisions_unchanged":True,
        "negative_controls_not_causal_effect_identification":True,
        "paired_TEST_NLL_receipts":bootstrap,
    }
    out_file.parent.mkdir(parents=True,exist_ok=True)
    out_file.write_text(
        json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    print(json.dumps({
        "event":"V38_INDEPENDENT_TEST_AUDIT",
        "rows":len(d["test_rows"]),"summary":len(d["summary"]),
        "max_metric_error":max_err,
        "paired_NLL_CIs":len(bootstrap),
        "probe_pass_equals_sentinel_only_samples":len(ablation_equal),
    }),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--evidence-dir",type=Path,required=True)
    parser.add_argument("--out-file",type=Path,required=True)
    args=parser.parse_args()
    audit(args.evidence_dir,args.out_file)
