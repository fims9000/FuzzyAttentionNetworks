"""V33 heldout concept interventions in EXACT additive native ConceptFAN.

This uses privileged SIMULATOR concept trajectories only at evaluation.
No oracle states become training targets, labels used to select candidates,
or medical "corrected" concepts. The intervention is an arithmetic change
of one known coordinate with all other paths and weights FROZEN; NOT do().
Expected log-loss evaluates against the known synthetic generator law to
distinguish genuine expected harm from one sampled Bernoulli outcome.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass

import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class KnownConceptIntervention:
    intervention: str
    baseline_mean_observed_nll: float
    corrected_mean_observed_nll: float
    delta_observed_nll: float
    baseline_mean_expected_nll: float
    corrected_mean_expected_nll: float
    delta_expected_nll: float
    per_episode_observed_harm_rate: float
    per_episode_expected_harm_rate: float
    mean_corrected_concept_absolute_error_before: float
    mean_logit_change: float
    max_exact_intervention_reconstruction_error: float
    uses_simulator_truth_ONLY_for_heldout_evaluation: bool = True
    causal_intervention_certified: bool = False
    clinical_concept_semantics_certified: bool = False

    def as_dict(self)->dict:
        return asdict(self)


def generator_conditional_probability(
    oracle_known_trajectories:np.ndarray,
    *,
    outcome_law:str,
    physical_amplitude:np.ndarray,
    artifact_amplitude:np.ndarray,
)->np.ndarray:
    known=np.asarray(oracle_known_trajectories,dtype=np.float64)
    phys=np.asarray(physical_amplitude,dtype=np.float64).reshape(-1)
    art=np.asarray(artifact_amplitude,dtype=np.float64).reshape(-1)
    if (known.ndim!=3 or known.shape[1:]!=(16,2)
        or len(known)!=len(phys) or len(known)!=len(art)
        or not np.isfinite(known).all() or
        not np.isfinite(phys).all() or not np.isfinite(art).all()):
        raise ValueError("Oracle generator truth is evaluation-only and must align")
    q=known.mean(axis=1)
    z=3.7*(q[:,0]-.5)-3.6*(q[:,1]-.5)-.10
    if outcome_law=="physical":
        z+=2.5*phys
    elif outcome_law=="artifact":
        z+=2.5*art
    elif outcome_law!="complete":
        raise ValueError("Unknown registered outcome law")
    return expit(z)


def bernoulli_loss_from_logit(
    risk:np.ndarray,
    logit:np.ndarray,
)->np.ndarray:
    """Expected Bernoulli NLL when risk is p_true; also accepts 0/1 Y."""
    p=np.asarray(risk,dtype=np.float64).reshape(-1)
    z=np.asarray(logit,dtype=np.float64).reshape(-1)
    if (len(p)!=len(z) or not np.isfinite(p).all()
            or not np.isfinite(z).all()
            or ((p<0)|(p>1)).any()):
        raise ValueError("Invalid truth risks/logits")
    return np.logaddexp(0.,z)-p*z


def known_concept_intervention_audit(
    predicted_known_q:np.ndarray,
    oracle_known_q:np.ndarray,
    *,
    original_head_bias:float,
    original_head_weights:np.ndarray,
    additional_source_contribution:np.ndarray,
    observed_test_outcomes:np.ndarray,
    true_conditional_event_probability:np.ndarray,
)->tuple[list[KnownConceptIntervention],dict[str,np.ndarray]]:
    known=np.asarray(predicted_known_q,dtype=np.float64)
    true=np.asarray(oracle_known_q,dtype=np.float64)
    w=np.asarray(original_head_weights,dtype=np.float64).reshape(-1)
    additional=np.asarray(additional_source_contribution,dtype=np.float64).reshape(-1)
    yy=np.asarray(observed_test_outcomes,dtype=np.float64).reshape(-1)
    pt=np.asarray(true_conditional_event_probability,dtype=np.float64).reshape(-1)
    if (known.ndim!=2 or known.shape[1]!=2 or true.shape!=known.shape or w.shape!=(2,)
        or len(additional)!=len(known) or len(yy)!=len(known)
        or len(pt)!=len(known) or not np.isin(yy,[0,1]).all()
        or not np.isfinite(known).all() or not np.isfinite(true).all()
        or not np.isfinite(w).all() or not np.isfinite(additional).all()
        or not np.isfinite(pt).all() or ((pt<0)|(pt>1)).any()):
        raise ValueError("Intervention audit must align real TEST subjects")
    if np.max(np.abs(known))>1.000001 or np.min(known)<-1e-6:
        raise ValueError("Measured concept predictions outside [0,1]")
    if np.max(true)>1.000001 or np.min(true)<-1e-6:
        raise ValueError("Synthetic oracle concepts outside [0,1]")
    z0=float(original_head_bias)+known@w+additional
    baseline_sample=bernoulli_loss_from_logit(yy,z0)
    baseline_expected=bernoulli_loss_from_logit(pt,z0)
    outputs={"none":z0.copy()}
    rows=[]
    for which,indices in (
        ("concept1_ONLY",(0,)),
        ("concept2_ONLY",(1,)),
        ("both_known_concepts",(0,1)),
    ):
        modified=known.copy()
        modified[:,indices]=true[:,indices]
        # Exact one-coordinate additive intervention: original source and
        # all unrelated concept terms are NEVER recomputed.
        delta=((true-known)[:,indices]@w[list(indices)]).reshape(-1)
        new=z0+delta
        exact=float(np.max(np.abs(
            new-(float(original_head_bias)+modified@w+additional))))
        if exact>2e-12:
            raise AssertionError("Intervention changed a frozen unrelated contribution")
        obs_new=bernoulli_loss_from_logit(yy,new)
        exp_new=bernoulli_loss_from_logit(pt,new)
        harm_obs=(obs_new-baseline_sample)>1e-12
        harm_exp=(exp_new-baseline_expected)>1e-12
        outputs[which]=new.copy()
        rows.append(KnownConceptIntervention(
            intervention=which,
            baseline_mean_observed_nll=float(baseline_sample.mean()),
            corrected_mean_observed_nll=float(obs_new.mean()),
            delta_observed_nll=float((obs_new-baseline_sample).mean()),
            baseline_mean_expected_nll=float(baseline_expected.mean()),
            corrected_mean_expected_nll=float(exp_new.mean()),
            delta_expected_nll=float((exp_new-baseline_expected).mean()),
            per_episode_observed_harm_rate=float(harm_obs.mean()),
            per_episode_expected_harm_rate=float(harm_exp.mean()),
            mean_corrected_concept_absolute_error_before=float(
                np.mean(np.abs(known[:,indices]-true[:,indices]))),
            mean_logit_change=float(delta.mean()),
            max_exact_intervention_reconstruction_error=exact,
        ))
    return rows,outputs


def observational_indistinguishability_falsifier(
    *,
    n_subjects:int=120,
    amplitude:float=1.30,
)->dict:
    """Construct two worlds with SAME observed readings, DIFFERENT hidden law.

    The differing early source amplitude is hidden behind a *synchronous*
    source group outage. This pair intentionally permits changing the
    SOURCE LAW on the unobserved portion; its point is that no deterministic
    observation-only algorithm can infer counterfactual hidden semantics
    without the trained stationarity assumption.
    """
    if n_subjects<100 or amplitude<=0 or amplitude>5:
        raise ValueError("Invalid synthetic indistinguishability control")
    original=np.zeros((n_subjects,16,12),dtype=np.float32)
    alternate=original.copy()
    original[:,:8,4:7]=+float(amplitude)
    alternate[:,:8,4:7]=-float(amplitude)
    # Force identical actually observed second-half readings.
    signal=np.tile(np.linspace(-.15,.15,8,dtype=np.float32),(n_subjects,1))
    original[:,8:,4:7]=signal[:,:,None]
    alternate[:,8:,4:7]=signal[:,:,None]
    mask=np.ones_like(original,dtype=bool)
    mask[:,:8,4:7]=False
    assert np.array_equal(original[mask],alternate[mask])
    return {
        "world_a_measured_history":original,
        "world_b_measured_history":alternate,
        "observation_mask":mask,
        "world_a_true_event_probability":float(expit(2.5*amplitude)),
        "world_b_true_event_probability":float(expit(-2.5*amplitude)),
        "known_K2_same_in_both_worlds":True,
        "different_true_outcome_laws_allowed_outside_model_stationarity":True,
    }
