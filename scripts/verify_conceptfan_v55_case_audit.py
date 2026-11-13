#!/usr/bin/env python3
"""Independent V55 TEST-case audit and descriptive paired-subject intervals."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score,log_loss

FIELDS={"K2":"k2_logits","V53_signed":"v53_logits",
        "V54_frozen":"v54_logits","V55_joint":"v55_logits",
        "raw_logistic":"raw_logits","energy_logistic":"energy_logits"}


def measure(y,z):
    return {"auprc":float(average_precision_score(y,expit(z))),
            "nll":float(log_loss(y,expit(z),labels=[0,1])),
            "positive_rate":float(np.mean(y))}


def run(root,bootstrap=399):
    root=Path(root)
    report=json.loads((root/"report_v55.json").read_text())
    if len(report["prediction_files"])!=30 or len(report["rows"])!=30:
        raise AssertionError("Missing preregistered V55 TEST arrays")
    for f in report["prediction_files"]:
        path=root/f["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=f["sha256"]:
            raise AssertionError("TEST array hash mismatch")
    receipts={(r["mode"],r["seed"]):r for r in report["receipts"]}
    arrays={}
    for row in report["rows"]:
        key=(row["mode"],int(row["seed"]),row["stress"])
        with np.load(root/"predictions"/f"{key[0]}_seed{key[1]}_{key[2]}.npz",
                     allow_pickle=False) as zz:
            a={x:zz[x].copy() for x in
                ("y","k2_logits","v53_logits","v54_logits","v55_logits",
                 "raw_logits","energy_logits","used_v54","used_v55",
                 "q_v54","q_v55","support_v54","support_v55")}
        if not all(x.shape==(report["n_test"],) for x in a.values()):
            raise AssertionError("Broken case alignment")
        if not np.isin(a["y"],[0,1]).all():
            raise AssertionError("Invalid outcomes")
        if not all(np.isfinite(x).all() for x in a.values()):
            raise AssertionError("Invalid TEST scores")
        rec=receipts[key[:2]]
        for prefix,field,gamma_field,selection_field in (
            ("v54","v54_logits","v54_train_gamma","v54_accepted"),
            ("v55","v55_logits","v55_accepted","v55_accepted")
        ):
            support=a[f"support_{prefix}"]
            used=a[f"used_{prefix}"].astype(bool)
            if not np.array_equal(a[field][~used],a["k2_logits"][~used]):
                raise AssertionError("Exact native K2 fallback violated")
            if np.any(used & (support<.5)):
                raise AssertionError("Source without minimum actually measured support")
            selected=rec[selection_field]
            gamma=float(rec[gamma_field] if prefix=="v54" else
                        (selected["gamma"] if selected is not None else 0.))
            expected=a["k2_logits"]+gamma*np.where(used,a[f"q_{prefix}"]-.5,0.)
            np.testing.assert_allclose(expected,a[field],atol=2e-6,rtol=0)
            if selected is None and used.any():
                raise AssertionError("Rejected source produced a risk contribution")
        for method,field in FIELDS.items():
            computed=measure(a["y"],a[field])
            for metric,val in computed.items():
                if abs(val-row[method][metric])>1e-12:
                    raise AssertionError(f"V55 drift {key} {method} {metric}")
        arrays[key]=a
    summary=[]
    for mode in sorted({r["mode"] for r in report["rows"]}):
        for stress in sorted({r["stress"] for r in report["rows"]}):
            seeds=tuple(report["seeds"])
            batch=[arrays[mode,seed,stress] for seed in seeds]
            for b in batch[1:]:
                np.testing.assert_array_equal(batch[0]["y"],b["y"])
            y=batch[0]["y"]
            rng=np.random.default_rng(20265587+sum(map(ord,mode+stress)))
            bands={"V55_minus_K2":[],"V55_minus_V54":[],
                   "V55_minus_energy_logistic":[]}
            for _ in range(bootstrap):
                ix=rng.integers(0,len(y),len(y))
                yy=y[ix]
                if np.unique(yy).size<2:continue
                diffs={name:[] for name in bands}
                for b in batch:
                    p=average_precision_score(yy,expit(b["v55_logits"][ix]))
                    diffs["V55_minus_K2"].append(
                        p-average_precision_score(yy,expit(b["k2_logits"][ix])))
                    diffs["V55_minus_V54"].append(
                        p-average_precision_score(yy,expit(b["v54_logits"][ix])))
                    diffs["V55_minus_energy_logistic"].append(
                        p-average_precision_score(yy,expit(b["energy_logits"][ix])))
                for name,values in diffs.items():
                    bands[name].append(float(np.mean(values)))
            group=[r for r in report["rows"] if
                   r["mode"]==mode and r["stress"]==stress]
            summary.append({"mode":mode,"stress":stress,
                "n_shared_test":len(y),"n_model_initializations":len(group),
                "means":{method:{k:float(np.mean([r[method][k] for r in group]))
                       for k in ("auprc","nll")} for method in FIELDS},
                "posthoc_paired_95pct_AUPRC":{
                   k:list(map(float,np.quantile(v,[.025,.975])))
                   for k,v in bands.items()}})
    receipt={"status":"V55_CASE_LEVEL_INDEPENDENT_RECOMPUTATION_PASS",
             "verified_arrays":len(report["prediction_files"]),
             "n_scenarios":len(summary),
             "posthoc_not_multiplicity_adjusted":True,
             "source_semantics_not_physically_verified":True,
             "bootstrap_subject_replicates":bootstrap,"scenarios":summary}
    (root/"verify_v55_audit.json").write_text(
        json.dumps(receipt,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"status":receipt["status"],"arrays":receipt["verified_arrays"],
                      "scenarios":receipt["n_scenarios"]}))
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dir",required=True,type=Path)
    args=parser.parse_args()
    run(args.dir)
