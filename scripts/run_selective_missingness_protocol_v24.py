"""Executable mask-stress protocol sanity on a *linear concept surrogate*.

NOT a ConceptFAN/V10 benchmark: does not instantiate or retrain its neural
model. It checks whether an observational support gate can improve the
quality of *selected* concept estimates without hiding selective coverage.
The actual V10 wrapper preserves logits and cannot claim a predictive gain.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

N_TRAIN, N_TEST, T, K = 2400, 1200, 24, 3
SEEDS = (116, 227, 338)
MODES = ("complete", "MCAR10", "MCAR20", "MCAR40", "MAR", "MNAR", "block12")


def draw_episodes(rng: np.random.Generator, n: int):
    noise=rng.standard_normal((n,T,K))
    latent=np.empty_like(noise)
    latent[:,0,:]=noise[:,0,:]
    for t in range(1,T):
        latent[:,t,:] = .72*latent[:,t-1,:] + np.sqrt(1-.72**2)*noise[:,t,:]
    values=latent + .28*rng.standard_normal(latent.shape)
    gold=latent[:,-6:,:].mean(axis=1)
    z=(1.2*gold[:,0]-1.0*gold[:,1]+.9*gold[:,2])
    outcomes=rng.binomial(1,expit(z))
    return values,gold,outcomes


def masked_view(rng, values, mode):
    n,t,k=values.shape
    mask=np.ones_like(values,dtype=bool)
    if mode.startswith("MCAR"):
        probability=int(mode[4:])/100
        mask=rng.random((n,t,k))>=probability
    elif mode=="MAR":
        # Channel 0 always observed. Channel 1 sampling depends on *observed*
        # channel 0 and time, so dropout is conditionally at random (MAR).
        logits=1.6*values[:,:,0]+1.3*(np.arange(t)[None,:]>=t//2)-.5
        p=.05+.80*expit(logits)
        mask[:,:,1]=rng.random((n,t))>=p
        mask[:,:,2]=rng.random((n,t))>=.15
    elif mode=="MNAR":
        # Each target's own (possibly hidden) value determines its missingness.
        p=.05+.85*expit(2.3*values)
        mask=rng.random(values.shape)>=p
    elif mode=="block12":
        mask=rng.random(values.shape)>=.12
        starts=rng.integers(0,t-12+1,size=(n,k))
        for i in range(n):
            for ch in range(k):
                start=starts[i,ch]
                mask[i,start:start+12,ch]=False
    elif mode!="complete":
        raise ValueError(mode)
    return mask


def feature_and_support(values,mask):
    # No imputed unobserved payload is permitted to enter the concept mean.
    observed=mask[:,-6:,:]
    x=values[:,-6:,:]
    counts=observed.sum(axis=1)
    features=(x*observed).sum(axis=1)/np.maximum(1,counts)
    fraction=counts/6
    return features,fraction


def metric(y,p):
    p=np.clip(np.asarray(p),1e-6,1-1e-6)
    return dict(n=len(y),positives=int(np.asarray(y).sum()),
                auprc=float(average_precision_score(y,p)) if len(np.unique(y))>1 else None,
                brier=float(brier_score_loss(y,p)),
                logloss=float(log_loss(y,p,labels=[0,1])))


def run(seed):
    rng=np.random.default_rng(seed)
    train,goldtrain,ytrain=draw_episodes(rng,N_TRAIN)
    test,goldtest,ytest=draw_episodes(rng,N_TEST)
    m_train=masked_view(rng,train,"MCAR20")
    x_train,f_train=feature_and_support(train,m_train)
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000))
    model.fit(x_train,ytrain)
    # Support threshold is selected on TRAIN coverage ONLY, no TEST labels.
    threshold=float(np.quantile(f_train.min(axis=1),.20))
    rows=[]
    for mode in MODES:
        m=masked_view(rng,test,mode)
        x,f=feature_and_support(test,m)
        probs=model.predict_proba(x)[:,1]
        covered=(f.min(axis=1)>=threshold)
        assert len(covered)==len(test) and 0<=covered.mean()<=1
        assert np.isfinite(probs).all()
        row=dict(seed=seed,mode=mode,threshold_from_train_only=threshold,
                 coverage=float(covered.mean()),abstentions=int((~covered).sum()),
                 raw_test=metric(ytest,probs),
                 selected_test=metric(ytest[covered],probs[covered]) if covered.any() else None,
                 select_model_changes_predictions=False,
                 mean_abs_concept_error_covered=float(np.abs(x[covered]-goldtest[covered]).mean()) if covered.any() else None,
                 mean_abs_concept_error_refused=float(np.abs(x[~covered]-goldtest[~covered]).mean()) if (~covered).any() else None,
                 recorded_observation_schedule_unchanged=True)
        rows.append(row)
    return rows


def main():
    rows=[row for seed in SEEDS for row in run(seed)]
    assert len(rows)==len(SEEDS)*len(MODES)
    result={"status":"SURROGATE_MASK_STRESS_ONLY", "pretrained_neural_conceptfan":False,
            "uses_true_generator_concepts_only_for_evaluation":True,
            "threshold_selected_without_test_labels":True,
            "synthetic_subjects_per_seed":N_TEST,"cases":rows,
            "conclusions":{
                "abstention_cannot_improve_predictions_on_full_cohort":True,
                "selective_metric_changes_must_report_coverage":True,
                "surrogate_results_are_not_v10_effectiveness_evidence":True}}
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":main()
