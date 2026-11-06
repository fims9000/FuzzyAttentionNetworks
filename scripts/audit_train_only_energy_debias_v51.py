"""V51 postfit independent verifier: no world or estimator module imports.

Recomputes every TRAIN-only source Ridge fit, assay-energy scale/affine,
two disjoint measured VALID MAE gates, all 216 TEST rows and 72 paired
TEST-patient bootstrap intervals from archived primary arrays.
"""
from __future__ import annotations

import argparse,json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

BUDGETS=(.05,.20,1.)
REGIMES=("healthy_MCAR20","healthy_MCAR40","amplifier_shift_MCAR20")
METHODS=("V50_RAW","V51_SCALE","V51_AFFINE")
CONTRASTS=(("V51_SCALE","V50_RAW"),
           ("V51_AFFINE","V50_RAW"),
           ("V51_AFFINE","V51_SCALE"))


def masked_features(x,m):
    a=np.asarray(x,dtype=np.float64)
    mask=np.asarray(m,dtype=bool)
    if a.ndim!=3 or a.shape[1:]!=(16,4) or a.shape!=mask.shape:
        raise AssertionError("Wrong measured input shape")
    return np.concatenate((np.where(mask,a,0.),mask.astype(np.float64)),axis=2)


def source_energy(coef,intercept,x,m):
    prediction=np.einsum("ntk,k->nt",masked_features(x,m),
                         np.asarray(coef,dtype=np.float64))+float(intercept)
    return np.var(prediction,axis=1,ddof=1)


def noisy_reference_energy(a):
    return np.maximum(0.,np.var(np.asarray(a,dtype=np.float64),axis=1,ddof=1)-.16**2)


def paired_bootstrap_mae(a,b,y,seed,n_boot):
    err=np.abs(a-y)-np.abs(b-y)
    rng=np.random.default_rng(seed)
    estimates=np.array([np.mean(err[rng.integers(0,len(y),len(y))])
                        for _ in range(n_boot)])
    return float(np.mean(err)),np.quantile(estimates,[.025,.975])


def original_source_cov(x,m):
    a=np.asarray(x,dtype=np.float64).reshape(-1,4)
    mask=np.asarray(m,dtype=bool).reshape(-1,4)
    result=np.empty((4,4))
    for i in range(4):
        for j in range(4):
            together=mask[:,i]&mask[:,j]
            xx=a[together,i];yy=a[together,j]
            if len(xx)<50:raise AssertionError("Insufficient pairwise observations")
            result[i,j]=np.sum((xx-xx.mean())*(yy-yy.mean()))/(len(xx)-1)
    return result


def relative_cov_shift(a,b):
    return float(np.linalg.norm(a-b)/max(1e-8,np.linalg.norm(b)))


def audit(p:Path,out:Path):
    status=json.loads((p/"STATUS.json").read_text())
    receipts=json.loads((p/"TRAIN_VALID_CALIBRATION_RECEIPTS.json").read_text())
    rows=json.loads((p/"TEST_ENERGY_ROWS.json").read_text())
    cis=json.loads((p/"PAIRED_TEST_MAE_CIS.json").read_text())
    twin=json.loads((p/"OBSERVATIONAL_TWIN_PROOFS.json").read_text())
    arr=np.load(p/"FROZEN_TEST_SOURCE_ENERGY_AND_ASSAY_RECEIPTS.npz",allow_pickle=False)
    devs=status["new_device_seeds"]
    if (len(receipts)!=len(devs)*2*3 or len(rows)!=len(devs)*2*3*3*3
            or len(cis)!=len(devs)*2*3*3 or len(twin)!=len(devs)):
        raise AssertionError("Missing independent devices, physical gauges or frozen TEST rows")
    receipt_map={(r["device"],r["gauge"],r["budget"]):r for r in receipts}
    rows_map={(r["device"],r["gauge"],r["budget"],r["regime"],r["method"]):r
              for r in rows}
    cis_map={(r["device"],r["gauge"],r["budget"],r["a"],r["b"]):r
             for r in cis}
    max_error=0.
    exact_fallback=0
    val_rejections=0
    for dev in devs:
        prefix=f"device{dev}"
        tr_x=arr[prefix+"_TRAIN_X"];tr_m=arr[prefix+"_TRAIN_mask"]
        v1x=arr[prefix+"_VALID1_X"];v1m=arr[prefix+"_VALID1_mask"]
        v2x=arr[prefix+"_VALID2_X"];v2m=arr[prefix+"_VALID2_mask"]
        obs=arr[prefix+"_TEST_X"]
        xa=arr[prefix+"_TEST_shifted_X"]
        m20=arr[prefix+"_TEST_mask20"]
        m40=arr[prefix+"_TEST_mask40"]
        energyA=arr[prefix+"_gaugeA_TEST_E_AUDIT_ONLY"]
        energyB=arr[prefix+"_gaugeB_TEST_E_AUDIT_ONLY"]
        gap=float(np.mean(np.abs(energyA-energyB)))
        pproof=next(r for r in twin if r["device"]==dev)
        if gap<.05 or abs(gap-pproof["average_physical_source1_energy_disagreement"])>1e-12:
            raise AssertionError("Non-identifiable source gauge was secretly chosen by label")
        train_cov=original_source_cov(tr_x,tr_m)
        for gauge in ("A","B"):
            truth=energyA if gauge=="A" else energyB
            anchor_tr=arr[f"{prefix}_gauge{gauge}_TRAIN_assay"]
            anchor_v1=arr[f"{prefix}_gauge{gauge}_VALID1_assay"]
            anchor_v2=arr[f"{prefix}_gauge{gauge}_VALID2_assay"]
            for budget in BUDGETS:
                cr=receipt_map[dev,gauge,budget]
                orig=cr["TRAIN_only_source_calibration"]
                fitted=cr["TRAIN_and_two_VALID_energy_correction"]
                n=int(np.floor(len(tr_x)*budget))
                seed=20265041+dev%101
                idx=np.sort(np.random.default_rng(seed+1).choice(len(tr_x),size=n,replace=False))
                source_model=Ridge(alpha=10.,fit_intercept=True)
                source_model.fit(masked_features(tr_x[idx],tr_m[idx]).reshape(-1,8),
                                 anchor_tr[idx].reshape(-1))
                alpha_error=float(np.max(np.abs(source_model.coef_-
                                     np.asarray(orig["ridge_coefficients"]))))
                inter_error=abs(float(source_model.intercept_)-orig["ridge_intercept"])
                max_error=max(max_error,alpha_error,inter_error)
                if max(alpha_error,inter_error)>1e-12:
                    raise AssertionError("Source encoder used non-TRAIN or false assayed labels")
                source_train=source_energy(source_model.coef_,source_model.intercept_,
                                           tr_x[idx],tr_m[idx])
                assay_train=noisy_reference_energy(anchor_tr[idx])
                scale=float(np.clip(np.sum(source_train*assay_train)/
                                    (np.sum(source_train**2)+1e-8),.5,2.))
                energy_ridge=Ridge(alpha=5.,fit_intercept=True).fit(
                    source_train.reshape(-1,1),assay_train)
                slope=float(np.clip(energy_ridge.coef_[0],.5,2.))
                intercept=float(np.clip(energy_ridge.intercept_,-.30,.30))
                for field,value in (
                    ("scale_alpha",scale),("affine_alpha",slope),
                    ("affine_intercept",intercept),
                    ("raw_energy_train_mean",float(np.mean(source_train))),
                    ("reference_energy_train_mean",float(np.mean(assay_train))),
                ):
                    delta=abs(fitted[field]-value)
                    max_error=max(max_error,delta)
                    if delta>1e-12:
                        raise AssertionError("TRAIN energy regression was not independently reproducible")
                valstats=[]
                for fold,(xx,mm,assay) in enumerate(
                    ((v1x,v1m,anchor_v1),(v2x,v2m,anchor_v2)),start=1):
                    ids=np.sort(np.random.default_rng(seed+fold*41).choice(
                        len(xx),size=80,replace=False))
                    raw=source_energy(source_model.coef_,source_model.intercept_,
                                      xx[ids],mm[ids])
                    ref=noisy_reference_energy(assay[ids])
                    d={
                        "raw":float(np.mean(np.abs(raw-ref))),
                        "scale":float(np.mean(np.abs(scale*raw-ref))),
                        "affine":float(np.mean(np.abs(np.maximum(0.,slope*raw+intercept)-ref))),
                    }
                    valstats.append(d)
                    for kind,value in d.items():
                        field=f"valid{fold}_{kind}_assay_MAE"
                        error=abs(fitted[field]-value)
                        max_error=max(max_error,error)
                        if error>1e-12:raise AssertionError("VALID assay was not independent")
                approvals={
                    "V50_RAW":True,
                    "V51_SCALE":all(z["scale"]<=z["raw"]+.015 for z in valstats),
                    "V51_AFFINE":all(z["affine"]<=z["raw"]+.015 for z in valstats),
                }
                if fitted["scale_approved_two_VALID"]!=approvals["V51_SCALE"]:
                    raise AssertionError("Physical scale gate changed after TEST")
                if fitted["affine_approved_two_VALID"]!=approvals["V51_AFFINE"]:
                    raise AssertionError("Physical affine gate changed after TEST")
                for regime in REGIMES:
                    x=xa if regime.startswith("amplifier") else obs
                    m=m40 if regime=="healthy_MCAR40" else m20
                    raw=source_energy(source_model.coef_,source_model.intercept_,x,m)
                    predicted={
                        "V50_RAW":raw,
                        "V51_SCALE":scale*raw if approvals["V51_SCALE"] else raw,
                        "V51_AFFINE":np.maximum(0.,slope*raw+intercept)
                            if approvals["V51_AFFINE"] else raw,
                    }
                    deviation=relative_cov_shift(original_source_cov(x,m),train_cov)
                    in_scope=bool(orig["dual_VALID_admitted"] and
                                  deviation<=orig["train_only_covariance_shift_limit"])
                    for method,guess in predicted.items():
                        key=f"{prefix}_gauge{gauge}_budget{int(budget*100)}_{regime}_{method}_source_energy"
                        saved=arr[key]
                        difference=float(np.max(np.abs(saved-guess)))
                        max_error=max(max_error,difference)
                        if difference>1e-12:
                            raise AssertionError("TEST energy estimate leaked latent true source")
                        row=rows_map[dev,gauge,budget,regime,method]
                        mae=float(np.mean(np.abs(saved-truth)))
                        rmse=float(np.sqrt(np.mean((saved-truth)**2)))
                        pearson=float(np.corrcoef(saved,truth)[0,1])
                        for field,value in (("mae",mae),("rmse",rmse),("pearson",pearson)):
                            err=abs(row["TEST_postfit_hidden_true_energy_metrics"][field]-value)
                            max_error=max(max_error,err)
                            if err>1e-12:raise AssertionError("TEST physical E metric differs")
                        if bool(row["physical_source_name_allowed"])!=in_scope:
                            raise AssertionError("Semantic label admitted outside TRAIN sensor scope")
                        if method!="V50_RAW" and not approvals[method]:
                            if not np.array_equal(saved,raw):
                                raise AssertionError("Failed VALID gate did not return raw V50 exactly")
                            exact_fallback+=1
                            val_rejections+=1
                    if regime=="healthy_MCAR40":
                        for comparison,(aa,bb) in enumerate(CONTRASTS):
                            rec=cis_map[dev,gauge,budget,aa,bb]
                            seedci=(20265181+(dev%101)*100+(0 if gauge=="A" else 25)+
                                    int(budget*100)*3+comparison)
                            estimate,interval=paired_bootstrap_mae(
                                predicted[aa],predicted[bb],truth,
                                seedci,rec["resamples"])
                            error=max(abs(rec["difference_MAE_a_minus_b"]-estimate),
                                      float(np.max(np.abs(np.asarray(
                                          rec["paired_TEST_subject_bootstrap_95pct_ci"])-interval))))
                            max_error=max(max_error,error)
                            if error>1e-12:
                                raise AssertionError("Unregistered/altered paired TEST MAE interval")
    result={
        "status":"V51_ALL_SOURCE_FITS_MAEs_AND_72_PAIRED_CIs_INDEPENDENTLY_VERIFIED",
        "n_independent_new_devices":len(devs),
        "n_refitted_source_estimates_and_energy_calibrations":len(receipts),
        "n_recalculated_TEST_rows":len(rows),
        "n_recalculated_TEST_energy_metric_values":len(rows)*3,
        "n_recalculated_paired_TEST_MAE_CIs":len(cis),
        "n_exact_rejected_correction_raw_fallbacks":exact_fallback,
        "n_no_validated_correction_TEST_rows":val_rejections,
        "max_numerical_error":max_error,
        "audit_did_not_import_simulator_or_train_predictor":True,
        "hidden_oracle_source_used_only_for_postfit_scores":True,
        "not_a_new_causal_semantic_identification_theorem":True,
    }
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--evidence-dir",type=Path,required=True)
    parser.add_argument("--out-file",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.evidence_dir,args.out_file),ensure_ascii=False))
