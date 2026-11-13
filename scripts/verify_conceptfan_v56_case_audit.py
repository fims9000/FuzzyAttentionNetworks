#!/usr/bin/env python3
"""Independent V56 saved-TEST-case audit of measured-source selective risk.

No simulator latent sources, independent assay TEST values, or TEST outcomes
are used for model fitting. Recomputes all reported TEST metrics only after
predictions, and verifies signed native contribution and exact K2 refusal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score,log_loss

PREDICTIONS={
    "K2":"k2_logits",
    "V54_frozen":"v54_logits",
    "V55_unrestricted":"v55_logits",
    "V56_corroborated":"v56_logits",
    "raw_logistic":"raw_logits",
    "energy_logistic":"energy_logits",
}


def metric(y,z):
    p=expit(z)
    return {"auprc":float(average_precision_score(y,p)),
            "nll":float(log_loss(y,p,labels=[0,1])),
            "positive_rate":float(y.mean())}


def run(root,bootstrap=399):
    root=Path(root)
    report=json.loads((root/"report_v56.json").read_text())
    if not(report["test_assay_purchased"]==0 and
           report["test_assay_accessed_for_inference"] is False and
           report["hidden_generator_latents_accessed_during_training"] is False):
        raise AssertionError("External TEST evidence was accessed")
    if len(report["rows"])!=48 or len(report["prediction_files"])!=48:
        raise AssertionError("Expected 48 preregistered TEST arrays")
    for item in report["prediction_files"]:
        p=root/item["path"]
        if hashlib.sha256(p.read_bytes()).hexdigest()!=item["sha256"]:
            raise AssertionError("TEST prediction hash does not match manifest")
    receipts={(r["mode"],r["seed"]):r for r in report["receipts"]}
    arrays={}
    for row in report["rows"]:
        mode,seed,stress=row["mode"],int(row["seed"]),row["stress"]
        k=(mode,seed,stress)
        with np.load(root/"predictions"/f"{mode}_seed{seed}_{stress}.npz",
                     allow_pickle=False) as z:
            case={key:z[key].copy() for key in (
                "y","k2_logits","v54_logits","v55_logits","v56_logits",
                "raw_logits","energy_logits","used_v55","used_v56",
                "q_v55","q_v56","support_v55","support_v56")}
        if not all(v.shape==(report["n_test"],) for v in case.values()):
            raise AssertionError("Corrupted per-subject array dimensions")
        if not np.isin(case["y"],[0,1]).all() or not all(
            np.isfinite(v).all() for v in case.values()):
            raise AssertionError("Invalid outcome, source or risk")
        rec=receipts[k[:2]]
        for name,field,receipt_name in (
            ("v55","v55_logits","unrestricted_v55"),
            ("v56","v56_logits","selected_v56"),
        ):
            used=case["used_"+name].astype(bool)
            if np.any(used&(case["support_"+name]<.5)):
                raise AssertionError("Used source without measured support")
            if not np.array_equal(case[field][~used],
                                  case["k2_logits"][~used]):
                raise AssertionError("Missing/rejected source changed frozen K2")
            selected=rec[receipt_name]
            gamma=float(selected["gamma"]) if selected else 0.
            expected=case["k2_logits"]+gamma*np.where(
                used,case["q_"+name]-.5,0.)
            np.testing.assert_allclose(
                case[field],expected,atol=2e-6,rtol=0)
            if selected is None and np.any(used):
                raise AssertionError("Refused concept contributed to risk")
        if row["v56_status"]!="ASSAY_CORROBORATED_MEASURED_PROXY":
            assert not case["used_v56"].any()
            np.testing.assert_array_equal(
                case["k2_logits"],case["v56_logits"])
        if stress=="source_missing":
            assert not case["used_v56"].any()
        for display,field in PREDICTIONS.items():
            got=metric(case["y"],case[field])
            for measure,value in got.items():
                if abs(row[display][measure]-value)>1e-12:
                    raise AssertionError(f"Metric drift {k} {display} {measure}")
        arrays[k]=case
    modes=sorted({r["mode"] for r in report["rows"]})
    stresses=sorted({r["stress"] for r in report["rows"]})
    summaries=[]
    for mode in modes:
        for stress in stresses:
            seed_arrays=[arrays[(mode,seed,stress)] for seed in report["native_seeds"]]
            labels=seed_arrays[0]["y"]
            for case in seed_arrays[1:]:
                np.testing.assert_array_equal(labels,case["y"])
            rng=np.random.default_rng(20265681+sum(map(ord,mode+stress)))
            contrast={"V56_minus_K2":[],"V56_minus_V55":[],
                      "V56_minus_energy_logistic":[]}
            for _ in range(bootstrap):
                ids=rng.integers(0,len(labels),len(labels))
                yy=labels[ids]
                if np.unique(yy).size<2:
                    continue
                per={name:[] for name in contrast}
                for arr in seed_arrays:
                    v56=average_precision_score(
                        yy,expit(arr["v56_logits"][ids]))
                    per["V56_minus_K2"].append(
                        v56-average_precision_score(yy,expit(arr["k2_logits"][ids])))
                    per["V56_minus_V55"].append(
                        v56-average_precision_score(yy,expit(arr["v55_logits"][ids])))
                    per["V56_minus_energy_logistic"].append(
                        v56-average_precision_score(yy,expit(arr["energy_logits"][ids])))
                for name,v in per.items():
                    contrast[name].append(float(np.mean(v)))
            rows=[r for r in report["rows"] if
                  r["mode"]==mode and r["stress"]==stress]
            summaries.append({
                "mode":mode,"stress":stress,
                "shared_test_subjects":len(labels),
                "n_native_initializations":len(rows),
                "v56_reasons":sorted({r["v56_status"] for r in rows}),
                "v56_mean_use":float(np.mean([r["v56_usage"] for r in rows])),
                "mean_test_metrics":{
                    name:{k:float(np.mean([r[name][k] for r in rows]))
                          for k in ("auprc","nll")}
                    for name in PREDICTIONS},
                "posthoc_paired_95pct_AUPRC":{
                    name:list(map(float,np.quantile(v,[.025,.975])))
                    for name,v in contrast.items()},
            })
    # Permuting the reference must not alter the original observed subjects,
    # labels, native known K2, or unrestricted predictive source behavior.
    for seed in report["native_seeds"]:
        for stress in stresses:
            positive=arrays[("nonlinear_energy",seed,stress)]
            permuted=arrays[("reference_permuted",seed,stress)]
            np.testing.assert_array_equal(positive["y"],permuted["y"])
            np.testing.assert_array_equal(
                positive["k2_logits"],permuted["k2_logits"])
            np.testing.assert_array_equal(
                positive["v55_logits"],permuted["v55_logits"])
    result={
        "status":"V56_INDEPENDENT_CASE_AUDIT_PASS",
        "verified_arrays":len(report["prediction_files"]),
        "n_scenario_groups":len(summaries),
        "n_test_people_per_label_mode":report["n_test"],
        "test_subjects_shared_between_model_seeds":True,
        "test_assay_absent":True,
        "posthoc_not_multiple_test_corrected":True,
        "no_physical_causal_semantic_claim":True,
        "paired_bootstrap_replicates":bootstrap,
        "scenarios":summaries,
    }
    (root/"verify_v56_audit.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"status":result["status"],
                      "arrays":result["verified_arrays"],
                      "scenarios":len(summaries)}))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dir",required=True,type=Path)
    a=p.parse_args()
    run(a.dir)
