#!/usr/bin/env python3
"""Independent V57 TEST audit: no TEST assay, only saved scores and frozen passports.

This verification happens after all selected sources and source-risk scales
are fixed. It does not authorize physical or medical causal naming.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score,log_loss

PRED={
    "K2":"k2_logits",
    "V55_unrestricted":"v55_logits",
    "V56_assay_gated":"v56_logits",
    "V57_retrained_calibrated":"v57_logits",
    "raw_logistic":"raw_logits",
    "energy_logistic":"energy_logits",
    "assay_energy_logistic":"assay_energy_logistic_logits",
}


def metrics(y,logits):
    prob=expit(logits)
    return {"auprc":float(average_precision_score(y,prob)),
            "nll":float(log_loss(y,prob,labels=[0,1])),
            "positive_rate":float(y.mean())}


def run(root,bootstrap=399):
    root=Path(root)
    d=json.loads((root/"report_v57.json").read_text())
    if (len(d["rows"])!=48 or len(d["prediction_files"])!=48
        or d["assay_budget"]["TEST"]!=0
        or d["reuses_statistical_v55_gamma_on_assay_q"]):
        raise AssertionError("TEST evidence leaked or incomplete")
    for item in d["prediction_files"]:
        p=root/item["path"]
        if hashlib.sha256(p.read_bytes()).hexdigest()!=item["sha256"]:
            raise AssertionError("Saved prediction hash mismatch")
    reports={(r["mode"],r["seed"]):r for r in d["receipts"]}
    saved={}
    for row in d["rows"]:
        k=(row["mode"],int(row["seed"]),row["stress"])
        p=root/"predictions"/f"{k[0]}_seed{k[1]}_{k[2]}.npz"
        with np.load(p,allow_pickle=False) as z:
            rr={name:z[name].copy() for name in (
                "y","k2_logits","v55_logits","v56_logits","v57_logits",
                "raw_logits","energy_logits","assay_energy_logistic_logits",
                "used_v55","used_v56","used_v57",
                "q_v55","q_v56","q_v57",
                "support_v55","support_v56","support_v57")}
        if not all(v.shape==(d["n_test"],) for v in rr.values()):
            raise AssertionError("TEST episode dimension mismatch")
        if not np.isin(rr["y"],[0,1]).all() or not all(
            np.isfinite(x).all() for x in rr.values()):
            raise AssertionError("Nonfinite or invalid case-level evidence")
        rec=reports[k[:2]]
        for prefix,field,contract_name in (
            ("v55","v55_logits","v55_unrestricted"),
            ("v56","v56_logits","v56_assay_selected"),
            ("v57","v57_logits","v57_calibration_contract"),
        ):
            used=rr["used_"+prefix].astype(bool)
            if np.any(used & (rr["support_"+prefix]<.5)):
                raise AssertionError("Unobserved source passed to signed risk")
            if not np.array_equal(rr[field][~used],rr["k2_logits"][~used]):
                raise AssertionError("Rejected source broke exact native K2")
            contract=rec[contract_name]
            if prefix=="v57":
                gamma=0. if contract is None else float(contract["risk_gamma_train_only"])
                if contract is not None and not contract["source_gamma_does_not_reuse_v55"]:
                    raise AssertionError("Reused incompatible V55 gamma")
            else:
                gamma=0. if contract is None else float(contract["gamma"])
            if contract is None and used.any():
                raise AssertionError("Nonadmitted source contributed to risk")
            expectation=rr["k2_logits"]+gamma*np.where(
                used,rr["q_"+prefix]-.5,0.)
            np.testing.assert_allclose(
                rr[field],expectation,atol=2e-6,rtol=0)
        if row["status_v57"]!="ASSAY_SCALED_MEASURED_PROXY":
            if rr["used_v57"].any():
                raise AssertionError("Source used while out of scope")
            np.testing.assert_array_equal(rr["v57_logits"],rr["k2_logits"])
        if k[2]=="source_missing":
            assert not rr["used_v57"].any()
        for label,field in PRED.items():
            got=metrics(rr["y"],rr[field])
            for metric,val in got.items():
                if abs(val-row[label][metric])>1e-12:
                    raise AssertionError(f"V57 metric drift {k} {label} {metric}")
        saved[k]=rr
    scenarios=[]
    for mode in sorted({r["mode"] for r in d["rows"]}):
        for stress in sorted({r["stress"] for r in d["rows"]}):
            rows=[r for r in d["rows"] if
                  r["mode"]==mode and r["stress"]==stress]
            population=[saved[mode,seed,stress] for seed in d["native_seeds"]]
            first=population[0]["y"]
            for other in population[1:]:
                np.testing.assert_array_equal(first,other["y"])
            rng=np.random.default_rng(20265783+sum(map(ord,mode+stress)))
            bands={"v57_minus_K2":[],"v57_minus_V56":[],
                   "v57_minus_engineered":[],"v57_minus_assay_engineered":[]}
            for _ in range(bootstrap):
                ix=rng.integers(0,len(first),len(first))
                yy=first[ix]
                if np.unique(yy).size<2:continue
                temp={name:[] for name in bands}
                for p in population:
                    v57=average_precision_score(yy,expit(p["v57_logits"][ix]))
                    for name,col in (
                        ("v57_minus_K2","k2_logits"),
                        ("v57_minus_V56","v56_logits"),
                        ("v57_minus_engineered","energy_logits"),
                        ("v57_minus_assay_engineered","assay_energy_logistic_logits"),
                    ):
                        temp[name].append(
                            v57-average_precision_score(yy,expit(p[col][ix])))
                for name,vals in temp.items():
                    bands[name].append(float(np.mean(vals)))
            scenarios.append({
                "mode":mode,"stress":stress,
                "n_shared_test_episodes":len(first),
                "native_initializations":len(rows),
                "v57_statuses":sorted({r["status_v57"] for r in rows}),
                "v57_mean_usage":float(np.mean([r["v57_usage"] for r in rows])),
                "mean_metrics":{
                    m:{key:float(np.mean([r[m][key] for r in rows]))
                       for key in ("auprc","nll")} for m in PRED
                },
                "descriptive_paired_95pct_AUPRC":{
                    name:list(map(float,np.quantile(vals,[.025,.975])))
                    for name,vals in bands.items()},
            })
    # The corrupted-reference control keeps the same observations/labels:
    # only measurement linkage is broken. Differences in V57 must be due
    # to source certification, not a new y or altered K2 baseline.
    for seed in d["native_seeds"]:
        for stress in sorted({r["stress"] for r in d["rows"]}):
            good=saved["nonlinear_energy",seed,stress]
            swapped=saved["reference_permuted",seed,stress]
            for field in ("y","k2_logits","v55_logits"):
                np.testing.assert_array_equal(good[field],swapped[field])
    audit={
        "status":"V57_INDEPENDENT_CASE_AUDIT_PASS",
        "verified_individual_test_arrays":len(d["prediction_files"]),
        "n_scenarios":len(scenarios),
        "test_external_assay_not_accessed":True,
        "new_source_gamma_retrained_for_assay_q":True,
        "shared_subjects_not_independent_seeds":True,
        "descriptive_bootstrap_only_unadjusted":True,
        "not_a_physical_causal_certificate":True,
        "paired_subject_bootstrap_replicates":bootstrap,
        "scenarios":scenarios,
    }
    (root/"verify_v57_audit.json").write_text(
        json.dumps(audit,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"status":audit["status"],
                      "verified_arrays":audit["verified_individual_test_arrays"],
                      "scenarios":len(scenarios)}))
    return audit


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dir",required=True,type=Path)
    args=p.parse_args()
    run(args.dir)
