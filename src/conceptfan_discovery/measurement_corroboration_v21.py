"""V21: independent measurement corroboration is NOT physical certification.

A model-free TRAIN-only test of whether a reproducible V17 observed sensor
direction agrees with an independently registered reference device on
disjoint TRAIN episodes. A common-mode artifact can pass this test; such
a falsification is an explicit supported outcome, never hidden.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
from typing import Iterable
import numpy as np

from .candidate_admission_v17 import ProposedObservableConcept


@dataclass(frozen=True)
class MeasurementCorroboration:
    candidate_id: str
    status: str
    n_train_provenance_episodes: int
    observed_signed_pearson_r: float
    absolute_pearson_r: float
    permutation_p_two_sided: float
    signed_bootstrap_ci95: tuple[float,float]
    two_halves_signed_correlations: tuple[float,float]
    reference_orientation_from_train_only: int
    reference_device_id: str
    reference_not_in_model_inputs: bool = True
    clinical_or_physical_causal_identity_established: bool = False
    cross_device_common_mode_artifact_ruled_out: bool = False


def _pearson(x:np.ndarray,y:np.ndarray)->float:
    if len(x)<2 or float(np.std(x))<1e-10 or float(np.std(y))<1e-10:
        return 0.
    return float(np.mean((x-x.mean())*(y-y.mean()))/(x.std()*y.std()))


def corroborate_candidate_on_disjoint_measured_reference(
    candidate:ProposedObservableConcept,
    provenance_histories:np.ndarray,
    independent_measured_reference:np.ndarray,
    *,
    provenance_episode_ids:Iterable[int],
    discovery_episode_ids:Iterable[int],
    risk_screen_episode_ids:Iterable[int],
    model_channel_devices:tuple[str,...],
    reference_device_id:str,
    seed:int=211104,
    permutations:int=199,
    bootstraps:int=199,
    window:int=6,
    minimum_abs_r:float=.35,
    minimum_bootstrap_abs_bound:float=.20,
)->MeasurementCorroboration:
    x=np.asarray(provenance_histories,dtype=np.float64)
    reference=np.asarray(independent_measured_reference,dtype=np.float64)
    ids=[int(i) for i in provenance_episode_ids]
    discovery={int(i) for i in discovery_episode_ids}
    risk={int(i) for i in risk_screen_episode_ids}
    if (not isinstance(candidate,ProposedObservableConcept)
            or x.ndim!=3 or x.shape[0]<180 or x.shape[-1]!=len(model_channel_devices)
            or reference.shape!=x.shape[:2] or len(ids)!=x.shape[0]
            or candidate.direction.shape!=(x.shape[-1],)
            or not np.isfinite(x).all() or not np.isfinite(reference).all()
            or not isinstance(reference_device_id,str) or not reference_device_id
            or reference_device_id in set(model_channel_devices)
            or not 2<=window<=x.shape[1]):
        raise ValueError("ABSTAIN: invalid independently registered measurement provenance")
    if (len(set(ids))!=len(ids) or not discovery or not risk
            or set(ids)&discovery or set(ids)&risk or discovery&risk):
        raise ValueError("ABSTAIN: discovery/risk/provenance TRAIN cohorts must be disjoint")
    if permutations<99 or bootstraps<99 or not .1<=minimum_abs_r<=.95:
        raise ValueError("Invalid frozen provenance screening thresholds")
    if not 0<=minimum_bootstrap_abs_bound<minimum_abs_r:
        raise ValueError("Invalid bootstrap lower bound")
    score=(x[:,-window:,:]@candidate.direction).mean(axis=1)
    independent=(reference[:,-window:]).mean(axis=1)
    if score.std()<1e-8 or independent.std()<1e-8:
        return MeasurementCorroboration(
            candidate.candidate_id,"ABSTAIN_CONSTANT_OR_UNOBSERVED_REFERENCE",len(ids),
            0.,0.,1.,(0.,0.),(0.,0.),0,reference_device_id,
        )
    measured_r=_pearson(score,independent)
    rng=np.random.default_rng(seed)
    shuffled=rng.permutation(len(ids))
    half1,half2=np.array_split(shuffled,2)
    halves=(_pearson(score[half1],independent[half1]),
            _pearson(score[half2],independent[half2]))
    perm=np.asarray([
        abs(_pearson(score,rng.permutation(independent)))
        for _ in range(permutations)
    ])
    p=float((1+int(np.count_nonzero(perm>=abs(measured_r))))/(1+permutations))
    bs=np.asarray([
        _pearson(score[idx],independent[idx])
        for idx in (
            rng.integers(0,len(ids),size=len(ids))
            for _ in range(bootstraps)
        )
    ])
    lower,upper=map(float,np.quantile(bs,[.025,.975]))
    direction=1 if measured_r>0 else -1 if measured_r<0 else 0
    bound=lower if direction==1 else -upper if direction==-1 else 0.
    consistent=(direction!=0 and
                all(np.sign(h)==direction and abs(h)>=.15 for h in halves))
    accepted=(abs(measured_r)>=minimum_abs_r and p<=.01
              and bound>minimum_bootstrap_abs_bound and consistent)
    return MeasurementCorroboration(
        candidate_id=candidate.candidate_id,
        status=("ACCEPT_MEASUREMENT_CORROBORATION" if accepted
                else "ABSTAIN_UNCORROBORATED_SENSOR_DIRECTION"),
        n_train_provenance_episodes=len(ids),
        observed_signed_pearson_r=measured_r,
        absolute_pearson_r=abs(measured_r),
        permutation_p_two_sided=p,
        signed_bootstrap_ci95=(lower,upper),
        two_halves_signed_correlations=tuple(map(float,halves)),
        reference_orientation_from_train_only=(direction if accepted else 0),
        reference_device_id=reference_device_id,
    )


def provenance_passport(value:MeasurementCorroboration)->dict:
    return {
        **asdict(value),
        "provenance_evidence_measured_not_latent_oracle":True,
        "outcome_labels_used_to_correlate_reference":False,
        "causal_physiological_mechanism_identified":False,
        "risk_association_proven":False,
        "common_mode_artifact_still_possible":True,
        "individual_patient_reference_records_exported":False,
    }
