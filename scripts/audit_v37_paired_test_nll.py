"""Independent V37 scientific audit from IMMUTABLE per-patient TEST probabilities.

Read original V37 run receipts and recompute EVERY AUPRC/AUROC/NLL/Brier
and summary, then estimate pre-registered paired TEST-subject bootstrap NLL.
Never fit, calibrate, select concepts, adjust VALID gates or tune on TEST.
The result is conditional on the existing 3 NN seeds and a single cohort per
device, and cannot be interpreted as a CI over independent instruments.

Usage:
PYTHONPATH=src:. python scripts/audit_v37_paired_test_nll.py
  --evidence-dir V37_EVIDENCE --output V37_EVIDENCE/PAIRED_NLL_AUDIT.json
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    average_precision_score,roc_auc_score,log_loss,brier_score_loss,
)

METHODS_WITH_NATIVE_SEEDS={
    "frozen_K2","AB_centered","ABC_threeview","AB_V36_uncentered",
}
SEEDS=(11,23,37)


def audit_v37(evidence:Path,output:Path,*,bootstrap=399):
    if output.exists():
        raise FileExistsError("Do not overwrite independent TEST audit receipt")
    if bootstrap!=399:
        raise ValueError("V37 scientific protocol fixes 399 paired bootstrap replications")
    state=json.loads((evidence/"STATUS.json").read_text())
    if state["status"]!="V37_CENTERED_THREEVIEW_EXECUTED":
        raise ValueError("Only actual completed V37 TEST evidence may be independently audited")
    archive=np.load(evidence/"FROZEN_TEST_PREDICTIONS.npz",allow_pickle=False)
    max_abs=0.
    for r in state["rows"]:
        mix,case,stress,method,seed=(r[k] for k in
            ("mix_seed","case","stress","method","NN_seed"))
        y=archive[f"device{mix}_{case}_TEST_LABELS_labels_seedNone"]
        raw=archive[f"device{mix}_{case}_{stress}_{method}_seed{seed}"]
        p=np.clip(np.asarray(raw,dtype=np.float64),1e-7,1-1e-7)
        if p.shape!=y.shape or len(p)!=600:
            raise AssertionError("TEST patient prediction size/ID mismatch")
        scores={
            "auprc":average_precision_score(y,p),
            "auroc":roc_auc_score(y,p),
            "nll":log_loss(y,p,labels=[0,1]),
            "brier":brier_score_loss(y,p),
        }
        for key,actual in scores.items():
            max_abs=max(max_abs,abs(float(actual)-r["metrics"][key]))
    if max_abs>=1e-12:
        raise AssertionError(f"Saved TEST metrics not reproducible: {max_abs}")
    for group in state["summary"]:
        r=[x for x in state["rows"] if all(x[k]==group[k]
           for k in ("mix_seed","case","stress","method"))]
        if not r:
            raise AssertionError("Summary entry without underlying TEST result")
        for key,other in (
            ("AUPRC","auprc"),("AUROC","auroc"),
            ("NLL","nll"),("Brier","brier"),
        ):
            if abs(np.mean([x["metrics"][other] for x in r])-group[key])>=1e-12:
                raise AssertionError("Saved summary does not reproduce source rows")
    receipts=[]
    for j,s in enumerate(state["paired_CI"]):
        mix,case,stress,a,b=(s[k] for k in
            ("mix_seed","case","stress","method_a","method_b"))
        y=archive[f"device{mix}_{case}_TEST_LABELS_labels_seedNone"]
        def paired(method):
            seeds=SEEDS if method in METHODS_WITH_NATIVE_SEEDS else (None,)
            values=np.stack([
                archive[f"device{mix}_{case}_{stress}_{method}_seed{z}"]
                for z in seeds
            ])
            if len(seeds)==1:
                values=np.repeat(values,len(SEEDS),axis=0)
            return np.clip(values,1e-7,1-1e-7)
        pa,pb=paired(a),paired(b)
        yy=y[None,:].astype(np.float64)
        diff=(
            -yy*np.log(pa)-(1-yy)*np.log(1-pa)
            +yy*np.log(pb)+(1-yy)*np.log(1-pb)
        )
        nll_diff=float(diff.mean())
        rng=np.random.default_rng(20264133+mix%97*100+j)
        sample_ids=rng.integers(0,len(y),size=(bootstrap,len(y)))
        sample_deltas=diff[:,sample_ids].mean(axis=(0,2))
        low,high=np.quantile(sample_deltas,[.025,.975])
        receipts.append({
            "mix_seed":mix,"case":case,"stress":stress,
            "method_a":a,"method_b":b,
            "paired_delta_NLL_a_minus_b":nll_diff,
            "paired_TEST_subject_bootstrap_95pct_ci_NLL":[float(low),float(high)],
            "resamples":bootstrap,
            "n_SHARED_TEST_episodes":len(y),
            "n_frozen_K2_network_fits":len(SEEDS),
            "confidence_interval_is_conditioned_on_this_device_and_fixed_models":True,
            "no_new_TEST_threshold_or_model_selection":True,
        })
    result={
        "verified_rows":len(state["rows"]),
        "verified_aggregate_rows":len(state["summary"]),
        "max_absolute_row_metric_recomputation_error":max_abs,
        "n_pre_registered_NLL_contrasts":len(receipts),
        "TEST_labels_never_used_for_fitting_or_admission":True,
        "independent_postrun_metric_audit_not_a_new_prediction_model":True,
        "paired_NLL":receipts,
    }
    if result["verified_rows"]!=2016 or result["verified_aggregate_rows"]!=864 or len(receipts)!=18:
        raise AssertionError("Incomplete frozen V37 evidence")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    print(json.dumps({
        "event":"V37_INDEPENDENT_METRIC_AUDIT",
        "verified_rows":result["verified_rows"],
        "verified_summaries":result["verified_aggregate_rows"],
        "maximum_metric_error":max_abs,
        "n_paired_NLL_bootstraps":len(receipts),
    }),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--evidence-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    audit_v37(args.evidence_dir,args.output)
