#!/usr/bin/env python3
"""Independent V53 saved-TEST-logit verification; post-test CI is descriptive."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score,log_loss

FIELDS={"K2":"k2_logits","K3_auto":"k3_logits","raw_logistic":"raw_logits"}

def measure(y,z):
    return {"auprc":float(average_precision_score(y,expit(z))),
            "nll":float(log_loss(y,expit(z),labels=[0,1])),
            "positive_rate":float(np.mean(y))}

def run(root,bootstrap=399):
    root=Path(root)
    report=json.loads((root/"report_v53.json").read_text())
    manifest=report["per_episode_prediction_files"]
    if len(manifest)!=24 or len(report["results"])!=24:raise AssertionError("Incomplete TEST records")
    for a in manifest:
        p=root/a["path"]
        if hashlib.sha256(p.read_bytes()).hexdigest()!=a["sha256"]:
            raise AssertionError("Mismatch in per-patient TEST array: "+str(p))
    receipts={(r["mode"],r["seed"]):r for r in report["receipts"]}
    arrays={}
    for row in report["results"]:
        k=(row["mode"],int(row["seed"]),row["stress"])
        with np.load(root/"predictions"/f"{k[0]}_seed{k[1]}_{k[2]}.npz",allow_pickle=False) as z:
            r={q:z[q].copy() for q in ("y","k2_logits","k3_logits","raw_logits","used","q","measured_support")}
        n=report["n_test"]
        if not all(v.shape==(n,) for v in r.values()):raise AssertionError("TEST cohort shape mismatch")
        if not np.isin(r["y"],[0,1]).all() or not np.isin(r["used"],[0,1]).all():
            raise AssertionError("Invalid labels or source flags")
        if not all(np.isfinite(v).all() for v in r.values()):
            raise AssertionError("Nonfinite TEST evidence")
        used=r["used"].astype(bool)
        if not np.array_equal(r["k2_logits"][~used],r["k3_logits"][~used]):
            raise AssertionError("Native K2 exact missing/rejected fallback changed")
        gamma=receipts[k[:2]]["source_gamma_train_only"]
        expected=r["k2_logits"]+float(gamma)*np.where(used,r["q"]-.5,0.)
        np.testing.assert_allclose(r["k3_logits"],expected,atol=2e-6,rtol=0)
        for label,field in FIELDS.items():
            got=measure(r["y"],r[field])
            for name,value in got.items():
                if abs(value-row[label][name])>1e-12:
                    raise AssertionError(f"V53 metric drift: {k} {label} {name}")
        arrays[k]=r
    if len(arrays)!=24:raise AssertionError("Duplicated source-scenario keys")
    summaries=[]
    for mode in ("physical_extra","irrelevant_source","predictive_artifact"):
        for stress in ("complete","MCAR20","MCAR40","source_sign_flip"):
            seeds=tuple(report["seeds"])
            same=[arrays[mode,s,stress] for s in seeds]
            for case in same[1:]:np.testing.assert_array_equal(case["y"],same[0]["y"])
            y=same[0]["y"]
            rng=np.random.default_rng(20261115+sum(map(ord,mode+stress)))
            k2_boot=[];raw_boot=[];nll_boot=[]
            for _ in range(bootstrap):
                ix=rng.integers(0,len(y),len(y))
                yy=y[ix]
                if np.unique(yy).size<2:continue
                vs=[];wr=[];nl=[]
                for rr in same:
                    k2=expit(rr["k2_logits"][ix])
                    k3=expit(rr["k3_logits"][ix])
                    raw=expit(rr["raw_logits"][ix])
                    vs.append(average_precision_score(yy,k3)-average_precision_score(yy,k2))
                    wr.append(average_precision_score(yy,k3)-average_precision_score(yy,raw))
                    nl.append(log_loss(yy,k3,labels=[0,1])-log_loss(yy,k2,labels=[0,1]))
                k2_boot.append(float(np.mean(vs)))
                raw_boot.append(float(np.mean(wr)))
                nll_boot.append(float(np.mean(nl)))
            original=[r for r in report["results"] if r["mode"]==mode and r["stress"]==stress]
            summaries.append({
                "mode":mode,"stress":stress,
                "N_shared_TEST":len(y),"N_model_initializations":len(same),
                "delta_AUPRC_K3_minus_K2":float(np.mean([r["K3_auto"]["auprc"]-r["K2"]["auprc"] for r in original])),
                "delta_AUPRC_K3_minus_raw":float(np.mean([r["K3_auto"]["auprc"]-r["raw_logistic"]["auprc"] for r in original])),
                "posthoc_patient_bootstrap_CI_AUPRC_K3_minus_K2":list(map(float,np.quantile(k2_boot,[.025,.975]))),
                "posthoc_patient_bootstrap_CI_AUPRC_K3_minus_raw":list(map(float,np.quantile(raw_boot,[.025,.975]))),
                "posthoc_patient_bootstrap_CI_NLL_K3_minus_K2":list(map(float,np.quantile(nll_boot,[.025,.975]))),
            })
    result={
        "status":"CASE_LEVEL_INDEPENDENT_RECOMPUTATION_PASS",
        "n_verified_test_prediction_arrays":len(manifest),
        "n_shared_test_episodes_per_mode":report["n_test"],
        "test_resamples_are_paired_over_patient_not_model":True,
        "posthoc_uncertainty_only_no_multiple_comparison_control":True,
        "no_independent_clinical_or_physical_identity_validation":True,
        "paired_bootstrap_replicates":bootstrap,
        "scenarios":summaries,
    }
    (root/"verify_v53_audit.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"status":result["status"],"N_scenarios":len(summaries),
                      "n_arrays":len(manifest)}))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dir",required=True,type=Path)
    a=p.parse_args();run(a.dir)
