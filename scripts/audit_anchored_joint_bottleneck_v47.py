"""Independent numeric audit of V47 frozen TEST, no model or simulator import."""
from __future__ import annotations

import argparse,json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score,log_loss,brier_score_loss

PAIRS=(
    ("healthy_energy","MCAR40","V47_JOINT_CONSISTENCY","V44_FROZEN"),
    ("healthy_energy","MCAR40","V47_JOINT_CONSISTENCY","RAW_DIFF_HGB_TRAIN1000"),
    ("healthy_energy","MCAR40","V47_JOINT_CONSISTENCY","RAW_DIFF_HGB_ALL1800"),
    ("healthy_energy","MCAR40","V47_JOINT_CONSISTENCY","V47_JOINT_PROXY"),
    ("unknown_OFF_drift","MCAR40","V47_JOINT_CONSISTENCY","K2"),
)
NATIVE={"K2","V44_FROZEN","V47_JOINT_PROXY","V47_JOINT_CONSISTENCY"}
SOURCE_METHODS={"V44_FROZEN","V47_JOINT_PROXY","V47_JOINT_CONSISTENCY"}


def _scored_ap(y,a,b,seed,resamples):
    rng=np.random.default_rng(seed)
    def difference(ix):
        return float(np.mean([
            average_precision_score(y[ix],a[j,ix])-
            average_precision_score(y[ix],b[j,ix])
            for j in range(a.shape[0])
        ]))
    delta=difference(np.arange(len(y)))
    draws=[difference(rng.integers(0,len(y),len(y))) for _ in range(resamples)]
    return delta,np.quantile(draws,[.025,.975])


def _arr(series,device,case,stress,method,seeds):
    ms=seeds if method in NATIVE else (None,)
    v=np.stack([
        series[f"device{device}_{case}_{stress}_{method}_seed{seed}"]
        for seed in ms
    ])
    return np.repeat(v,len(seeds),axis=0) if len(ms)==1 and len(seeds)>1 else v


def audit(path:Path,out:Path):
    status=json.loads((path/"STATUS.json").read_text())
    rows=json.loads((path/"TEST_ROWS.json").read_text())
    summary=json.loads((path/"SUMMARY.json").read_text())
    cis=json.loads((path/"PAIRED_CIS.json").read_text())
    gates=json.loads((path/"DUAL_VALID_GATES.json").read_text())
    stability=json.loads((path/"CONCEPT_STABILITY.json").read_text())
    probs=np.load(path/"FROZEN_TEST_PREDICTIONS.npz",allow_pickle=False)
    masks=np.load(path/"FROZEN_SOURCE_AND_USE_FLAGS.npz",allow_pickle=False)
    known=np.load(path/"FROZEN_KNOWN_CONCEPTS.npz",allow_pickle=False)
    seeds=sorted({r["seed"] for r in rows if r["method"]=="K2"})
    n_devices=len(status["device_seeds"])
    expected_support=n_devices*4*4
    expected_flags=expected_support*3*len(seeds)
    if len(masks.files)!=expected_support+expected_flags:
        raise AssertionError("Missing source-support or concept-use receipt arrays")
    if len(known.files)!=n_devices*len(seeds)*2:
        raise AssertionError("Incomplete frozen known concept values")
    gate_map={(r["device"],r["seed"],r["method"]):r for r in gates}
    max_diff=0.
    fallback_checks=0;full_missing_checks=0
    for row in rows:
        d,c,stress,method,seed=(row[x] for x in ("device","case","stress","method","seed"))
        yy=probs[f"device{d}_{c}_TEST_Y"]
        p=probs[f"device{d}_{c}_{stress}_{method}_seed{seed}"]
        if len(p)!=status["n_test"] or p.shape!=yy.shape or not np.isin(yy,[0,1]).all():
            raise AssertionError("Wrong TEST subjects or labels in stored predictions")
        if not np.isfinite(p).all() or (p<0).any() or (p>1).any():
            raise AssertionError("Invalid frozen probability")
        pp=np.clip(p.astype(np.float64),1e-7,1-1e-7)
        expected={
            "n":len(yy),
            "positives":int(yy.sum()),
            "auprc":average_precision_score(yy,pp),
            "auroc":roc_auc_score(yy,pp),
            "nll":log_loss(yy,pp,labels=[0,1]),
            "brier":brier_score_loss(yy,pp),
        }
        for key,value in expected.items():
            diff=abs(float(value)-float(row["metrics"][key]))
            max_diff=max(max_diff,diff)
            if diff>1e-12:
                raise AssertionError(f"Original TEST metric mismatch {d}/{c}/{stress}/{method}/{key}")
        if method in SOURCE_METHODS:
            prefix=f"device{d}_{c}_{stress}"
            observed=masks[prefix+"_measured_source"]
            used=masks[f"{prefix}_{method}_seed{seed}_used"]
            if used.shape!=observed.shape or used.shape!=p.shape:
                raise AssertionError("Invalid source admission arrays")
            gate=gate_map[d,seed,method]["approved"]
            if not np.array_equal(used,observed & bool(gate)):
                raise AssertionError("TEST source decision differs from TRAIN/VALID gate")
            if abs(float(used.mean())-row["source_use"])>1e-12:
                raise AssertionError("TEST concept-use percentage invented")
            baseline=probs[f"device{d}_{c}_{stress}_K2_seed{seed}"]
            if not np.array_equal(p[~used],baseline[~used]):
                raise AssertionError("Unsupported or rejected joint risk didn't return original frozen K2")
            fallback_checks+=1
            if stress=="missing_reference":
                if not np.array_equal(p,baseline):
                    raise AssertionError("No OFF measurement: concept bypassed exact fallback")
                full_missing_checks+=1
    for aggregate in summary:
        rr=[r for r in rows if all(r[k]==aggregate[k] for k in
            ("device","case","stress","method"))]
        nfits=len(seeds) if aggregate["method"] in NATIVE else 1
        if len(rr)!=nfits:
            raise AssertionError("Missing method/fitted init in TEST summary")
        for key,metric in (("AUPRC","auprc"),("NLL","nll")):
            diff=abs(float(np.mean([x["metrics"][metric] for x in rr]))-
                     float(aggregate[key]))
            max_diff=max(max_diff,diff)
            if diff>1e-12: raise AssertionError("Aggregate score mismatch")
        if abs(float(np.mean([x["source_use"] for x in rr]))-
               float(aggregate["source_use"]))>1e-12:
            raise AssertionError("Aggregation of source uses differs from independent rows")
    for row in cis:
        d,c,stress,a,b=(row[k] for k in ("device","case","stress","a","b"))
        contrast=(c,stress,a,b)
        if contrast not in PAIRS:
            raise AssertionError("Post-TEST invented comparison")
        yi=probs[f"device{d}_{c}_TEST_Y"]
        av=_arr(probs,d,c,stress,a,seeds)
        bv=_arr(probs,d,c,stress,b,seeds)
        obs,interval=_scored_ap(
            yi,av,bv,20264821+d%101*100+PAIRS.index(contrast),row["resamples"])
        diff=max(abs(obs-row["difference_AUPRC_a_minus_b"]),
                 float(np.max(np.abs(interval-row["paired_TEST_subject_bootstrap_95pct_ci"]))))
        max_diff=max(max_diff,diff)
        if diff>1e-12: raise AssertionError("Saved paired AP interval mismatch")
    for rec in stability:
        dev,method=rec["device"],rec["method"]
        arr=[known[f"device{dev}_healthy_MCAR40_{method}_seed{seed}_known"] for seed in seeds]
        corrs=[]
        for i in range(len(arr)):
            for j in range(i+1,len(arr)):
                corrs.extend([np.corrcoef(arr[i][:,k],arr[j][:,k])[0,1] for k in range(2)])
        if corrs:
            for field,value in (("known_concept_cross_seed_pearson_mean",np.mean(corrs)),
                                ("known_concept_cross_seed_pearson_min",np.min(corrs))):
                diff=abs(rec[field]-value)
                max_diff=max(max_diff,float(diff))
                if diff>1e-12: raise AssertionError("Known-concept stability score mismatch")
        elif rec["known_concept_cross_seed_pearson_mean"] is not None:
            raise AssertionError("One seed cannot give independent-seed stability")
    answer={
        "status":"V47_ALL_TEST_FITS_METRICS_CIS_AND_CONCEPTS_INDEPENDENTLY_VERIFIED",
        "n_TEST_rows":len(rows),"n_verified_TEST_metrics":len(rows)*6,
        "n_summary":len(summary),"n_preregistered_paired_CIs":len(cis),
        "n_direct_source_fallback_comparisons":fallback_checks,
        "n_exact_no_OFF_K2_fallbacks":full_missing_checks,
        "n_stability_receipts":len(stability),
        "n_saved_source_support_and_use_arrays":len(masks.files),
        "max_numerical_error":max_diff,
        "auditor_used_training_or_simulator":False,
        "is_real_clinical_validation":False,
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(answer,indent=2,ensure_ascii=False,allow_nan=False))
    return answer


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--evidence-dir",type=Path,required=True)
    p.add_argument("--out-file",type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(audit(args.evidence_dir,args.out_file),ensure_ascii=False))
