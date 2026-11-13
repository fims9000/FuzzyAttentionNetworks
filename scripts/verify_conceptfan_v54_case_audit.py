#!/usr/bin/env python3
"""Independent V54 saved individual TEST prediction audit.

Recomputes all reported AUPRC/NLL; checks native K2 mathematical additivity,
fallback and measured support. Paired subject bootstrap is descriptive only,
not a prospective error guarantee or a physical concept certification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score,log_loss

FIELDS={"K2":"k2_logits","V53_signed":"v53_logits",
        "V54_library":"v54_logits","raw_logistic":"raw_logits",
        "energy_logistic":"energy_logits"}


def metrics(y,z):
    return {"auprc":float(average_precision_score(y,expit(z))),
            "nll":float(log_loss(y,expit(z),labels=[0,1])),
            "positive_rate":float(np.mean(y))}


def run(root,bootstrap=399):
    root=Path(root)
    report=json.loads((root/"report_v54.json").read_text())
    files=report["prediction_files"]
    rows=report["rows"]
    if len(files)!=30 or len(rows)!=30:
        raise AssertionError("Expected all 30 registered TEST arrays and metrics")
    for item in files:
        p=root/item["path"]
        if hashlib.sha256(p.read_bytes()).hexdigest()!=item["sha256"]:
            raise AssertionError("Prediction artifact checksum mismatch")
    receipts={(r["mode"],r["seed"]):r for r in report["receipts"]}
    arrays={}
    for row in rows:
        mode,seed,stress=row["mode"],int(row["seed"]),row["stress"]
        path=root/"predictions"/f"{mode}_seed{seed}_{stress}.npz"
        with np.load(path,allow_pickle=False) as data:
            item={key:data[key].copy() for key in (
                "y","k2_logits","v53_logits","v54_logits","raw_logits",
                "energy_logits","used_v53","used_v54",
                "q_v53","q_v54","support_v53","support_v54")}
        if not all(v.shape==(report["test_subjects"],) for v in item.values()):
            raise AssertionError("Wrong per-subject dimension")
        if not np.isin(item["y"],[0,1]).all():
            raise AssertionError("Invalid binary outcomes")
        if not all(np.isfinite(z).all() for z in item.values()):
            raise AssertionError("Nonfinite individual scores")
        for tag,prefix in (("v53","v53"),("v54","v54")):
            used=item[f"used_{tag}"].astype(bool)
            base=item["k2_logits"]
            proposal=item[f"{tag}_logits"]
            assert np.array_equal(proposal[~used],base[~used]),"Exact neutral K2 violated"
            gamma=float(receipts[mode,seed]["gamma_"+prefix+"_train_only"])
            expected=base+np.where(used,gamma*(item[f"q_{tag}"]-.5),0.)
            np.testing.assert_allclose(proposal,expected,atol=2e-6,rtol=0)
            assert not np.any(used&(item[f"support_{tag}"]<.5))
            if receipts[mode,seed][f"{prefix}_selected"] is None:
                assert not used.any()
        for display,field in FIELDS.items():
            val=metrics(item["y"],item[field])
            for key,v in val.items():
                if abs(row[display][key]-v)>1e-12:
                    raise AssertionError(f"V54 TEST metrics drift {mode} {seed} {stress} {display}")
        arrays[mode,seed,stress]=item
    modes=sorted(set(r["mode"] for r in rows))
    stresses=sorted(set(r["stress"] for r in rows))
    seeds=tuple(report["seeds"])
    summary=[]
    for mode in modes:
        for stress in stresses:
            cohorts=[arrays[mode,seed,stress] for seed in seeds]
            first=cohorts[0]["y"]
            for other in cohorts[1:]:
                np.testing.assert_array_equal(first,other["y"])
            # One subject resample is applied to BOTH initializations; seed
            # bootstrap replicates are not falsely interpreted as independent.
            seed_value=20265488+sum(map(ord,mode+stress))
            rng=np.random.default_rng(seed_value)
            bands={"v54_minus_K2":[],"v54_minus_signed":[],
                   "v54_minus_engineered":[]}
            for _ in range(bootstrap):
                ix=rng.integers(0,len(first),len(first))
                yy=first[ix]
                if len(np.unique(yy))<2:continue
                per={"v54_minus_K2":[],"v54_minus_signed":[],
                     "v54_minus_engineered":[]}
                for z in cohorts:
                    v54=average_precision_score(yy,expit(z["v54_logits"][ix]))
                    per["v54_minus_K2"].append(
                        v54-average_precision_score(yy,expit(z["k2_logits"][ix])))
                    per["v54_minus_signed"].append(
                        v54-average_precision_score(yy,expit(z["v53_logits"][ix])))
                    per["v54_minus_engineered"].append(
                        v54-average_precision_score(yy,expit(z["energy_logits"][ix])))
                for key,val in per.items():
                    bands[key].append(float(np.mean(val)))
            summary.append({"mode":mode,"stress":stress,"shared_test_subjects":len(first),
                "n_native_seeds":len(seeds),
                "descriptive_95pct_bands_AUPRC":{
                    key:list(map(float,np.quantile(val,[.025,.975])))
                    for key,val in bands.items()},
                "mean_metrics":{
                    method:{metric:float(np.mean([next(x for x in rows if
                        x["mode"]==mode and x["seed"]==seed and x["stress"]==stress)[method][metric]
                        for seed in seeds])) for metric in ("auprc","nll")}
                    for method in FIELDS},
            })
    receipt={"status":"V54_CASE_LEVEL_INDEPENDENT_RECOMPUTATION_PASS",
        "verified_saved_arrays":len(files),
        "shared_test_subjects_per_scenario":report["test_subjects"],
        "paired_patient_bootstrap_reps":bootstrap,
        "posthoc_descriptive_only":True,
        "no_physical_identity_validation":True,
        "scenarios":summary}
    (root/"verify_v54_audit.json").write_text(
        json.dumps(receipt,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"status":receipt["status"],
        "scenarios":len(summary),"verified_arrays":len(files)}))
    return receipt

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dir",required=True,type=Path)
    a=p.parse_args()
    run(a.dir)

if __name__=="__main__":main()
