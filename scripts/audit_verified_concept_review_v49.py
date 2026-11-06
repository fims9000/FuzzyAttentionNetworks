"""V49 frozen independent audit (no ConceptFAN models, generator, or review API)."""
from __future__ import annotations

import argparse,json
from pathlib import Path
import numpy as np
from scipy.special import expit,logit
from sklearn.metrics import average_precision_score,log_loss,brier_score_loss

DEVICES=(20262907,20262909,20262921,20262923)
SEEDS=(11,23,37)
CASES=("healthy_energy","common_U8_matched","unknown_OFF_drift","OFF_source_leak")
STRESS=("MCAR20","MCAR40","MCAR50","missing_reference")
COMPARE=(
    ("unknown_OFF_drift","MCAR40","ACTIVE20","V48_UNREVIEWED"),
    ("unknown_OFF_drift","MCAR40","ACTIVE20","RANDOM20"),
    ("healthy_energy","MCAR40","ACTIVE20","V48_UNREVIEWED"),
    ("common_U8_matched","MCAR40","ACTIVE20","V48_UNREVIEWED"),
    ("unknown_OFF_drift","MCAR40","HGB_ACTIVE20","HGB_UNREVIEWED"),
)


def independent_calibration(q,m,e):
    eligible=np.asarray(m,dtype=bool)
    count=int(eligible.sum())
    if count<200: return None
    q0=q[eligible]
    energy=e[eligible]
    center=float(np.median(energy))
    scale=float(max(.2,(np.quantile(energy,.75)-np.quantile(energy,.25))/1.349))
    x=(energy-center)/scale
    l=logit(np.clip(q0,.001,.999))
    xv=x-x.mean()
    lv=l-l.mean()
    slope=float(np.mean(xv*lv)/(np.mean(xv*xv)+1e-8))
    intercept=float(l.mean()-slope*x.mean())
    corr=float(np.corrcoef(x,q0)[0,1]) if (
        np.std(x)>1e-9 and np.std(q0)>1e-9) else 0.
    threshold=float(max(.05,np.quantile(np.abs(q0-expit(slope*x+intercept)),.95)))
    return {
        "center_energy":center,"scale_energy":scale,"slope":slope,
        "intercept":intercept,"reject_threshold":threshold,
        "supported_train_count":count,"source_energy_train_correlation":corr,
        "certified_available":bool(corr>=.35 and .15<=slope<=10.
                                   and np.isfinite(slope) and np.isfinite(intercept)),
    }


def deterministic_selection(p,k,used,budget,selection,seed):
    mask=np.asarray(used,dtype=bool)
    candidates=np.flatnonzero(mask)
    n=min(int(budget),len(candidates))
    result=np.zeros(len(p),dtype=bool)
    if not n:return result
    if selection=="active":
        score=np.abs(logit(p)-logit(k))
        ranking=np.lexsort((candidates,-score[candidates]))
        selected=candidates[ranking[:n]]
    elif selection=="random":
        selected=np.random.default_rng(seed).choice(candidates,size=n,replace=False)
    else:
        raise ValueError("Unknown frozen reviewer")
    result[selected]=True
    return result


def fixed_random_seed(device,seed,case,stress,budget,repeat,is_hgb):
    return (20264901+(device%100)*100000+seed*1000+
            CASES.index(case)*100+STRESS.index(stress)*20+
            int(budget*100)*5+repeat+int(is_hgb)*10000000)


def _method_matrix(data,device,case,stress,method,seeds):
    prefix=f"device{device}_{case}_{stress}"
    entries=[]
    for seed in seeds:
        rep_count=3 if method=="RANDOM20" else 1
        for rep in range(rep_count):
            entries.append(data[f"{prefix}_{method}_seed{seed}_rep{rep}"])
    matrix=np.stack(entries)
    return np.repeat(matrix,3,axis=0) if len(matrix)==len(seeds) else matrix


def _paired_ap(y,a,b,seed,resamples):
    rng=np.random.default_rng(seed)
    def paired(idx):
        return float(np.mean([
            average_precision_score(y[idx],a[j,idx])-
            average_precision_score(y[idx],b[j,idx])
            for j in range(len(a))]))
    point=paired(np.arange(len(y)))
    draws=[paired(rng.integers(0,len(y),len(y))) for _ in range(resamples)]
    return point,np.quantile(draws,[.025,.975])


def audit(base:Path,review:Path,out:Path):
    base_status=json.loads((base/"STATUS.json").read_text())
    status=json.loads((review/"STATUS.json").read_text())
    model=np.load(base/"FROZEN_TEST_PREDICTIONS.npz",allow_pickle=False)
    mask=np.load(base/"FROZEN_SOURCE_AND_USE_FLAGS.npz",allow_pickle=False)
    source=np.load(base/"FROZEN_RISK_EVIDENCE.npz",allow_pickle=False)
    assay=np.load(base/"SEALED_IDEALIZED_EXPERT_ASSAY_AUDIT_ONLY.npz",allow_pickle=False)
    edited=np.load(review/"FROZEN_REVIEW_PREDICTIONS.npz",allow_pickle=False)
    selections=np.load(review/"FROZEN_REVIEW_SELECTION_AND_ACTIONS.npz",allow_pickle=False)
    rows=json.loads((review/"REVIEW_TEST_ROWS.json").read_text())
    summaries=json.loads((review/"REVIEW_SUMMARY.json").read_text())
    cis=json.loads((review/"PAIRED_CIS.json").read_text())
    fits=json.loads((review/"TRAIN_ASSAY_CALIBRATION.json").read_text())
    seeds=tuple(status["native_seeds"])
    devices=tuple(status["device_seeds"])
    if (status["n_test_method_rows"]!=len(rows)
            or status["n_summary_rows"]!=len(summaries)
            or status["n_expert_action_arrays"]!=len(selections.files)
            or len(cis)!=len(devices)*len(COMPARE)):
        raise AssertionError("Frozen review receipts incomplete")
    if base_status["device_seeds"]!=list(devices):
        raise AssertionError("Different TEST devices in base and intervention")
    bydev={r["device"]:r for r in fits}
    maxerr=0.;verified_selected=0;exact_fallback=0
    for device in devices:
        calibr=independent_calibration(
            assay[f"device{device}_TRAIN_measured_q"],
            assay[f"device{device}_TRAIN_measured_support"],
            assay[f"device{device}_TRAIN_energy"])
        stated=bydev[device]["TRAIN_only_calibration"]
        if calibr is None:
            if stated["certified_available"]:raise AssertionError("Invalid assay certificate")
        else:
            for key,value in calibr.items():
                if isinstance(value,bool):
                    if bool(stated[key])!=value:raise AssertionError("Wrong TRAIN physical fit")
                else:
                    if abs(float(stated[key])-float(value))>1e-12:
                        raise AssertionError("Wrong TRAIN assay threshold")
    for row in rows:
        d,seed,case,stress,method,repeat=(row[k] for k in
            ("device","seed","case","stress","method","repeat"))
        prefix=f"device{d}_{case}_{stress}"
        key=f"{prefix}_{method}_seed{seed}_rep{repeat}"
        p=edited[key]
        y=model[f"device{d}_{case}_TEST_Y"]
        basep=model[f"{prefix}_K2_seed{seed}"]
        v48=model[f"{prefix}_V48_JOINT_LOGODDS_seed{seed}"]
        hgb=model[f"{prefix}_RAW_DIFF_HGB_ALL1800_seedNone"]
        used=mask[f"{prefix}_V48_JOINT_LOGODDS_seed{seed}_used"]
        q=source[prefix+"_source_q"]
        if len(p)!=len(y) or not np.isfinite(p).all():
            raise AssertionError("Incorrect saved TEST risk array")
        if method.endswith("UNREVIEWED"):
            expected={"K2_UNREVIEWED":basep,
                      "V48_UNREVIEWED":v48,
                      "HGB_UNREVIEWED":hgb}[method]
            if not np.array_equal(p,expected):
                raise AssertionError("Frozen unreviewed predictions mutated")
        else:
            chosen=selections[key+"_selected"]
            actions=selections[key+"_actions"]
            if method.startswith("ACTIVE"):
                budget=int(method[-2:])*.01
                expected=deterministic_selection(
                    v48,basep,used,int(np.floor(len(y)*budget)),
                    "active",fixed_random_seed(d,seed,case,stress,budget,repeat,False))
                candidate=v48
            elif method.startswith("RANDOM"):
                budget=int(method[-2:])*.01
                expected=deterministic_selection(
                    v48,basep,used,int(np.floor(len(y)*budget)),
                    "random",fixed_random_seed(d,seed,case,stress,budget,repeat,False))
                candidate=v48
            elif method=="HGB_ACTIVE20":
                budget=.20
                expected=deterministic_selection(
                    hgb,basep,used,int(np.floor(len(y)*budget)),
                    "active",fixed_random_seed(d,seed,case,stress,budget,repeat,True))
                candidate=hgb
            else:
                raise AssertionError("Unregistered review intervention method")
            if not np.array_equal(chosen,expected):
                raise AssertionError("Review selection used TEST oracle or Y / wrong budget")
            calibration=bydev[d]["TRAIN_only_calibration"]
            actions_expected=np.full(len(y),"keep",dtype="<U8")
            if calibration["certified_available"]:
                true_e=assay[f"device{d}_TEST_energy"]
                reference_q=expit(calibration["slope"]*
                    ((true_e[chosen]-calibration["center_energy"])/calibration["scale_energy"])
                    +calibration["intercept"])
                rejected=np.abs(q[chosen]-reference_q)>calibration["reject_threshold"]
                actions_expected[chosen]="confirm"
                actions_expected[np.flatnonzero(chosen)[rejected]]="reject"
            if not np.array_equal(actions,actions_expected):
                raise AssertionError("Synthetic independent expert verdict mismatch")
            expected_p=np.where(actions=="reject",basep,candidate)
            if not np.array_equal(p,expected_p):
                raise AssertionError("Concept correction is not exact signed K2 rejection")
            verified_selected+=int((actions!="keep").sum())
            exact_fallback+=int((actions=="reject").sum())
            if int((actions!="keep").sum())!=row["selected"]:
                raise AssertionError("Missing intervention receipt")
            if int((actions=="reject").sum())!=row["expert_rejected"]:
                raise AssertionError("Inconsistent number of failed concepts")
        pp=np.clip(p,1e-7,1-1e-7)
        scored={
            "n":len(y),"positives":int(np.sum(y)),
            "auprc":average_precision_score(y,pp),
            "nll":log_loss(y,pp,labels=[0,1]),
            "brier":brier_score_loss(y,pp),
        }
        for name,value in scored.items():
            diff=abs(float(value)-float(row["metrics"][name]))
            maxerr=max(maxerr,diff)
            if diff>1e-12:raise AssertionError("Independent TEST risk metric mismatch")
    for agg in summaries:
        subset=[r for r in rows if all(r[k]==agg[k] for k in
            ("device","case","stress","method"))]
        expected_count=len(seeds)*(3 if agg["method"].startswith("RANDOM") else 1)
        if len(subset)!=expected_count:
            raise AssertionError("Missing random reviewer or native seed")
        for key,metric in (("AUPRC","auprc"),("NLL","nll"),("Brier","brier"),
                           ("reviewed","selected"),("rejected","expert_rejected"),
                           ("eligible","eligible_source_count")):
            error=abs(float(np.mean([r[metric] if metric in r else r["metrics"][metric]
                  for r in subset]))-agg[key])
            maxerr=max(maxerr,error)
            if error>1e-12: raise AssertionError("Mismatched aggregate")
    for row in cis:
        d=row["device"]
        compare=(row["case"],row["stress"],row["a"],row["b"])
        if compare not in COMPARE:raise AssertionError("TEST contrast was not preregistered")
        i=COMPARE.index(compare)
        y=model[f"device{d}_{row['case']}_TEST_Y"]
        a=_method_matrix(edited,d,row["case"],row["stress"],row["a"],seeds)
        b=_method_matrix(edited,d,row["case"],row["stress"],row["b"],seeds)
        point,interval=_paired_ap(y,a,b,20264981+(d%100)*100+i,row["resamples"])
        e=max(abs(point-row["difference_AUPRC_a_minus_b"]),
              float(np.max(np.abs(interval-row["paired_TEST_subject_bootstrap_95pct_ci"]))))
        maxerr=max(maxerr,e)
        if e>1e-12: raise AssertionError("Paired AP interval differs from TEST artifact")
    answer={
        "status":"V49_ASSAY_AND_REVIEW_TEST_INDEPENDENTLY_RECALCULATED",
        "verified_test_method_rows":len(rows),
        "verified_test_metrics":len(rows)*5,
        "verified_review_summary_rows":len(summaries),
        "verified_preregistered_AP_intervals":len(cis),
        "verified_actual_external_expert_review_actions":verified_selected,
        "verified_exact_source_rejections":exact_fallback,
        "verified_selection_and_action_arrays":len(selections.files),
        "max_absolute_numeric_error":maxerr,
        "selection_reconstructed_from_labels":False,
        "oracle_used_only_for_selected_case_expert_adjudication":True,
        "not_real_physical_or_clinical_validation":True,
    }
    out.write_text(json.dumps(answer,ensure_ascii=False,indent=2))
    return answer


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--base-dir",type=Path,required=True)
    p.add_argument("--review-dir",type=Path,required=True)
    p.add_argument("--out-file",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(audit(a.base_dir,a.review_dir,a.out_file),ensure_ascii=False))
