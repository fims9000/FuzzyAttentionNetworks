"""Independent V50 numerical audit; does NOT import world/generator/calibrator.

Recomputes externally anchored Ridge, both 80-episode VALID checks, TRAIN
covariance limits, TEST predicted source energies and no-claim decisions.
Only saved primary X/masks, attested assays, TEST energies, predictions.
"""
from __future__ import annotations

import argparse,json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score,log_loss

METHODS=("healthy_MCAR20","healthy_MCAR40","amplifier_shift_MCAR20")
BUDGETS=(.05,.20,1.)


def _features(x,m):
    arr=np.asarray(x,dtype=np.float64)
    obs=np.asarray(m,dtype=bool)
    if arr.ndim!=3 or arr.shape[1:]!=(16,4) or arr.shape!=obs.shape:
        raise AssertionError("Malformed sensor observations")
    return np.concatenate((np.where(obs,arr,0.),obs.astype(np.float64)),axis=2)


def _cov(x,m):
    data=np.asarray(x,dtype=np.float64).reshape(-1,4)
    mask=np.asarray(m,dtype=bool).reshape(-1,4)
    out=np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            eligible=mask[:,i]&mask[:,j]
            a=data[eligible,i]
            b=data[eligible,j]
            if len(a)<50:raise AssertionError("Not enough actually measured pairs")
            out[i,j]=np.mean((a-a.mean())*(b-b.mean()))*len(a)/(len(a)-1)
    return out


def _cov_distance(c1,c2):
    return float(np.linalg.norm(c1-c2)/
                 max(1e-8,np.linalg.norm(c2)))


def _pearson(a,b):
    return float(np.corrcoef(a,b)[0,1])


def audit(path:Path,out:Path):
    status=json.loads((path/"STATUS.json").read_text())
    proofs=json.loads((path/"OBSERVATIONAL_EQUIVALENCE_PROOFS.json").read_text())
    calibrations=json.loads((path/"EXTERNAL_CALIBRATIONS.json").read_text())
    rows=json.loads((path/"TEST_ASSAY_CALIBRATED_CONCEPT_RESULTS.json").read_text())
    risks=json.loads((path/"RAW_HGB_PREDICTION_CONTROL.json").read_text())
    d=np.load(path/"TEST_OBSERVATIONS_AND_LATENT_AUDIT.npz",allow_pickle=False)
    devices=status["device_seeds"]
    if len(proofs)!=len(devices) or len(calibrations)!=len(devices)*6:
        raise AssertionError("Incomplete physical worlds and TRAIN-only calibrations")
    if len(rows)!=len(devices)*2*3*3 or len(risks)!=len(devices)*3:
        raise AssertionError("Incomplete, cherry-picked V50 TEST rows")
    certificate_map={(r["device"],r["gauge"],
                      r["TRAIN_assay_budget_fraction"]):r for r in calibrations}
    row_map={(r["device"],r["gauge"],r["TRAIN_assay_budget_fraction"],
             r["regime"]):r for r in rows}
    risk_map={(r["device"],r["regime"]):r for r in risks}
    max_err=0.
    lower_bounds=[]
    refitted=0
    semantic_labels=0
    total_healthy_false_refusals=0
    total_shift_detected=0
    for device in devices:
        key=f"device{device}"
        x=d[key+"_shared_TEST_X"]
        x_twin=d[key+"_explicitly_identical_twin_TEST_X"]
        y=d[key+"_shared_TEST_Y"]
        ea=d[key+"_gaugeA_AUDIT_ONLY_source1_E"]
        eb=d[key+"_gaugeB_AUDIT_ONLY_source1_E"]
        blind=d[key+"_blind_first_sensor_energy"]
        if not np.array_equal(x,x_twin):
            raise AssertionError("Not observationally identical TEST signals")
        if not np.array_equal(blind,np.var(x[:,:,0],axis=1,ddof=1)):
            raise AssertionError("Blind estimator was given oracle calibration")
        bound=float(np.mean(np.abs(ea-eb))/2.)
        blind_error=.5*(float(np.mean(np.abs(blind-ea)))+
                        float(np.mean(np.abs(blind-eb))))
        if blind_error+1e-12<bound:raise AssertionError("Violation of exact gauge lower bound")
        proof=next(r for r in proofs if r["device"]==device)
        for name,value in (
            ("blind_estimator_minimax_absolute_error_lower_bound",bound),
            ("blind_first_sensor_energy_two_world_mean_MAE",blind_error),
            ("mean_absolute_difference_named_source1_energy",2*bound),
        ):
            max_err=max(max_err,abs(value-proof[name]))
            if abs(value-proof[name])>1e-12:raise AssertionError("Twin theorem receipt incorrect")
        lower_bounds.append(bound)
        train_x=d[key+"_TRAIN_X"]
        train_m=d[key+"_TRAIN_mask"]
        v1x=d[key+"_VALID1_X"]
        v1m=d[key+"_VALID1_mask"]
        v2x=d[key+"_VALID2_X"]
        v2m=d[key+"_VALID2_mask"]
        train_cov=_cov(train_x,train_m)
        for regime in METHODS:
            mx=d[key+("_known_MCAR40_mask" if regime=="healthy_MCAR40"
                      else "_known_MCAR20_mask")]
            observed=d[key+("_TEST_shifted_X" if regime.startswith("amplifier")
                       else "_shared_TEST_X")]
            p=d[f"{key}_{regime}_raw_HGB_probability"]
            r=risk_map[device,regime]
            if len(p)!=len(y) or np.any(~np.isfinite(p)):
                raise AssertionError("Missing ordinary raw task predictions")
            ap=float(average_precision_score(y,p))
            nll=float(log_loss(y,p,labels=[0,1]))
            for k,val in (("TEST_AUPRC",ap),("TEST_NLL",nll)):
                max_err=max(max_err,abs(r[k]-val))
                if abs(r[k]-val)>1e-12:raise AssertionError("Strong raw task score incorrect")
        for gauge in ("A","B"):
            truth=ea if gauge=="A" else eb
            ref_train=d[f"{key}_gauge{gauge}_TRAIN_assay_ONLY"]
            ref_v1=d[f"{key}_gauge{gauge}_VALID1_assay_ONLY"]
            ref_v2=d[f"{key}_gauge{gauge}_VALID2_assay_ONLY"]
            for budget in BUDGETS:
                report=certificate_map[device,gauge,budget]
                cert=report["calibration"]
                count=int(np.floor(len(train_x)*budget))
                seed=20265041+device%101
                indices=np.sort(np.random.default_rng(seed+1).choice(
                    len(train_x),size=count,replace=False))
                model=Ridge(alpha=10.,fit_intercept=True)
                model.fit(_features(train_x[indices],train_m[indices]).reshape(-1,8),
                          ref_train[indices].reshape(-1))
                refitted+=1
                diffcoef=float(np.max(np.abs(np.asarray(cert["ridge_coefficients"])-
                                             model.coef_)))
                diffintercept=abs(cert["ridge_intercept"]-float(model.intercept_))
                max_err=max(max_err,diffcoef,diffintercept)
                if max(diffcoef,diffintercept)>1e-12:
                    raise AssertionError("TRAIN-only physical readout coefficients mismatch")
                valcorr=[]
                for i,(vxx,vmm,vassay) in enumerate((
                    (v1x,v1m,ref_v1),(v2x,v2m,ref_v2)),start=1):
                    vi=np.sort(np.random.default_rng(seed+i*41).choice(
                        len(vxx),size=80,replace=False))
                    pred=model.predict(_features(vxx[vi],vmm[vi]).reshape(-1,8))
                    corr=_pearson(pred,vassay[vi].reshape(-1))
                    valcorr.append(corr)
                    keyfield=f"VALID{i}_assay_pearson"
                    max_err=max(max_err,abs(cert[keyfield]-corr))
                    if abs(cert[keyfield]-corr)>1e-12:
                        raise AssertionError("VALID anchored correlation differs")
                if cert["dual_VALID_admitted"]!=all(c>=.75 for c in valcorr):
                    raise AssertionError("Certificate ignored disjoint VALID rejection")
                if cert["train_assay_episodes"]!=count:
                    raise AssertionError("Wrong budget of costly reference measurements")
                train_fingerprint=np.asarray(cert["train_sensor_covariance"])
                if not np.allclose(train_fingerprint,train_cov,atol=1e-12,rtol=0):
                    raise AssertionError("Source name certificate uses wrong measured device")
                rng=np.random.default_rng(seed+1510)
                boot=[]
                for i in range(48):
                    ids=rng.choice(len(train_x),size=250,replace=True)
                    boot.append(_cov_distance(_cov(train_x[ids],train_m[ids]),train_cov))
                threshold=float(max(.04,np.quantile(boot,.99)*1.15))
                max_err=max(max_err,abs(cert["train_only_covariance_shift_limit"]-threshold))
                if abs(cert["train_only_covariance_shift_limit"]-threshold)>1e-12:
                    raise AssertionError("TEST-based threshold tuning detected")
                for regime in METHODS:
                    condition=(d[key+"_TEST_shifted_X"] if regime.startswith("amplifier")
                               else x)
                    mm=d[key+("_known_MCAR40_mask" if regime=="healthy_MCAR40"
                              else "_known_MCAR20_mask")]
                    result_key=f"{key}_gauge{gauge}_budget{int(budget*100)}_{regime}"
                    predicted=d[result_key+"_inferred_source_energy"]
                    timesteps=np.einsum("ntk,k->nt",_features(condition,mm),
                                        np.asarray(cert["ridge_coefficients"]))+cert["ridge_intercept"]
                    independently_computed=np.var(timesteps,axis=1,ddof=1)
                    numerical=float(np.max(np.abs(predicted-independently_computed)))
                    max_err=max(max_err,numerical)
                    if numerical>1e-12:raise AssertionError("Actual inferred source energy changed")
                    row=row_map[device,gauge,budget,regime]
                    corr=_pearson(predicted,truth)
                    mae=float(np.mean(np.abs(predicted-truth)))
                    for k,val in (("pearson_with_postfit_hidden_source",corr),
                                  ("energy_MAE_vs_postfit_hidden_source",mae)):
                        max_err=max(max_err,abs(row[k]-val))
                        if abs(row[k]-val)>1e-12:
                            raise AssertionError("Source reconstruction audit incorrect")
                    shift=_cov_distance(_cov(condition,mm),train_cov)
                    allowed=bool(cert["dual_VALID_admitted"] and shift<=threshold)
                    status_name=("DUAL_VALID_ASSAY_REJECTED" if not cert["dual_VALID_admitted"]
                                 else ("ASSAY_CALIBRATED_SAME_ENVIRONMENT" if allowed else
                                       "READOUT_DISTRIBUTION_SHIFT_REVOKED"))
                    if row["semantic_scope_status"]!=status_name or (
                            bool(row["physical_label_allowed"])!=allowed):
                        raise AssertionError("Unattested or shifted physical source was named")
                    max_err=max(max_err,abs(row["TEST_sensor_covariance_shift"]-shift))
                    if abs(row["TEST_sensor_covariance_shift"]-shift)>1e-12:
                        raise AssertionError("TEST covariance shift incorrectly reported")
                    semantic_labels+=int(allowed)
                    if regime.startswith("amplifier"):
                        total_shift_detected+=int(not allowed)
                    elif allowed is False:
                        total_healthy_false_refusals+=1
    ans={
        "status":"V50_BLIND_NONIDENTIFIABILITY_AND_ASSAY_REFITS_INDEPENDENTLY_VERIFIED",
        "n_devices":len(devices),
        "n_exact_observational_twin_pairs":len(devices),
        "n_independently_refitted_external_source_calibrations":refitted,
        "n_verified_external_TEST_source_rows":len(rows),
        "n_verified_raw_HGB_risk_rows":len(risks),
        "mean_minimax_absolute_error_lower_bound":float(np.mean(lower_bounds)),
        "n_semantic_scope_approvals":semantic_labels,
        "n_shift_condition_refusals_across_all_calibrations":total_shift_detected,
        "n_healthy_condition_refusals_across_all_calibrations":total_healthy_false_refusals,
        "maximum_absolute_numerical_error":max_err,
        "did_not_import_generator_or_model_or_calibration_module":True,
        "latent_source_truth_used_for_postfit_metric_audit_only":True,
        "real_physical_identity_not_proven":True,
    }
    out.write_text(json.dumps(ans,indent=2,ensure_ascii=False,allow_nan=False))
    return ans


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--evidence-dir",type=Path,required=True)
    parser.add_argument("--out-file",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.evidence_dir,args.out_file),ensure_ascii=False))
