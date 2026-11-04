"""V22: off-axis multisensor mixtures with separate observable environments.

Physical-generated and instrument-generated colored sources each affect all
five recorded sensors. Their mixing signatures overlap substantially. The
model receives ONLY observed measurements, observed environment IDs in the
unsupervised source-discovery stage, and a binary outcome for TRAIN task
learning. No true latent source trajectory is passed to the model or risk
screen. Truth can be re-generated only by an explicitly named audit function.

The target can depend on the physical-generated source; this does NOT mean
the algorithm can certify whether a recovered component is physiological.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import DiscoveryData, DiscoverySplit


FAMILIES=(
    "mixed_distinct_variance",
    "mixed_proportional_variance",
    "mixed_sensor_gain_drift",
    "mixed_artifact_only_variance",
)
SEEDS={"train":522201,"validation":522202,"test":522203}
SIZES={"train":800,"validation":300,"test":300}
HISTORY=24
HORIZON=6
SOURCE_COUNT=2
N_SENSORS=5
MODEL_CHANNELS=(
    "train_whitened_component_0",
    "train_whitened_component_1",
    "zero_padded_2",
    "zero_padded_3",
    "zero_padded_4",
)
RAW_CHANNELS=tuple(f"observable_sensor_{i}" for i in range(N_SENSORS))
# Dense, fixed, and deliberately NON-orthogonal source mixing.
MIXING=np.asarray([
    [.62,.53],
    [.49,.72],
    [.31,-.41],
    [.48,.22],
    [.39,-.18],
],dtype=np.float64)
MIXING=MIXING/np.linalg.norm(MIXING,axis=0,keepdims=True)


@dataclass(frozen=True)
class ObservableEnvironments:
    family: str
    train_x_full: np.ndarray
    validation_x_full: np.ndarray
    train_environment: np.ndarray
    validation_environment: np.ndarray
    train_labels: np.ndarray
    validation_labels: np.ndarray
    train_episode_ids: np.ndarray
    validation_episode_ids: np.ndarray
    original_sensor_names: tuple[str,...]
    no_oracle_latent_trajectories_exposed: bool = True


@dataclass(frozen=True)
class TrainWhitening:
    center: np.ndarray
    linear_map: np.ndarray
    pooled_eigenvalues: np.ndarray
    original_sensor_names: tuple[str,...]
    trained_episodes: int

    def transform(self,x:np.ndarray)->np.ndarray:
        source=np.asarray(x,dtype=np.float64)
        if source.ndim!=3 or source.shape[-1]!=len(self.center):
            raise ValueError("Expected observed [episodes,steps,5] sensor measurements")
        if not np.isfinite(source).all():
            raise ValueError("Cannot whiten nonfinite observed sensor histories")
        low=(source-self.center)@self.linear_map
        # Both models see the SAME two whitened measurement channels.
        # Padding is zero: it cannot convey outcomes or hidden source truth.
        zeros=np.zeros((*low.shape[:2],len(self.center)-SOURCE_COUNT),dtype=np.float64)
        return np.concatenate([low,zeros],axis=2).astype(np.float32)


def _simulate(family:str,seed:int,n:int):
    if family not in FAMILIES or n<10:
        raise ValueError("Unregistered V22 synthetic source family")
    rng=np.random.default_rng(seed)
    environment=np.arange(n,dtype=np.int8)%2
    rng.shuffle(environment)
    t=HISTORY+HORIZON
    baseline=np.zeros((n,t,SOURCE_COUNT),dtype=np.float64)
    baseline[:,0]=rng.normal(size=(n,SOURCE_COUNT))
    phi=.72
    innovation_scale=float(np.sqrt(1-phi**2))
    for time in range(1,t):
        baseline[:,time]=(
            phi*baseline[:,time-1]
            +innovation_scale*rng.normal(size=(n,SOURCE_COUNT))
        )
    if family=="mixed_distinct_variance":
        scaling=np.where(
            environment[:,None]==0,
            np.array([[.55,1.45]]),
            np.array([[1.45,.55]]),
        )
    elif family=="mixed_proportional_variance":
        scaling=np.where(
            environment[:,None]==0,
            np.array([[.8,.8]]),
            np.array([[1.35,1.35]]),
        )
    elif family=="mixed_artifact_only_variance":
        scaling=np.where(
            environment[:,None]==0,
            np.array([[1.,.5]]),
            np.array([[1.,1.5]]),
        )
    else:
        scaling=np.ones((n,SOURCE_COUNT))
    latent=baseline*scaling[:,None,:]
    raw=latent@MIXING.T+.045*rng.normal(size=(n,t,N_SENSORS))
    if family=="mixed_sensor_gain_drift":
        # A deliberately broken condition: changing acquisition gain on a
        # sensor changes mixing geometry, not the latent variance profiles.
        raw[environment==1,:,1]*=1.65
    # Outcome depends on physical-generated factor only; the independent
    # colored instrument source can still look like a coherent concept.
    signal=latent[:,HISTORY-6:HISTORY,0].mean(1)
    probability=1/(1+np.exp(-np.clip(2.1*signal,-15.,15.)))
    outcomes=(rng.random(n)<probability).astype(np.float32)
    return raw,environment,outcomes,latent


def make_observable_v22(family:str)->ObservableEnvironments:
    # Do not even instantiate the synthetic TEST generator here.
    train,train_env,train_y,_=_simulate(family,SEEDS["train"],SIZES["train"])
    val,val_env,val_y,_=_simulate(
        family,SEEDS["validation"],SIZES["validation"],
    )
    if (len(np.unique(train_y))!=2 or len(np.unique(val_y))!=2
            or len(np.unique(train_env))!=2
            or len(np.unique(val_env))!=2):
        raise ValueError("V22 requires both TRAIN/VALIDATION outcomes and environments")
    return ObservableEnvironments(
        family=family,
        train_x_full=train.astype(np.float32),
        validation_x_full=val.astype(np.float32),
        train_environment=train_env,
        validation_environment=val_env,
        train_labels=train_y,
        validation_labels=val_y,
        train_episode_ids=np.arange(0,len(train_y),dtype=np.int64),
        validation_episode_ids=np.arange(10000,10000+len(val_y),dtype=np.int64),
        original_sensor_names=RAW_CHANNELS,
    )


def train_only_whitening(measured:ObservableEnvironments)->TrainWhitening:
    if not isinstance(measured,ObservableEnvironments):
        raise ValueError("Only observable V22 TRAIN measurements can fit whitening")
    values=np.asarray(measured.train_x_full[:,:HISTORY],dtype=np.float64)
    if values.shape[0]<300 or values.shape[-1]!=N_SENSORS or not np.isfinite(values).all():
        raise ValueError("Invalid TRAIN-only sensor histories")
    flattened=values.reshape(-1,N_SENSORS)
    center=flattened.mean(0)
    cov=np.cov(flattened.T)
    eigenvalues,vectors=np.linalg.eigh(cov)
    indices=np.argsort(eigenvalues)[::-1][:SOURCE_COUNT]
    top=eigenvalues[indices]
    if (top[-1]<.025 or top[0]/top[-1]>200):
        raise ValueError("ABSTAIN: rank-2 measurement subspace is not supported")
    transform=vectors[:,indices]/np.sqrt(top)[None,:]
    if not np.allclose(
        np.cov(((flattened-center)@transform).T),np.eye(SOURCE_COUNT),atol=.015,
    ):
        raise AssertionError("TRAIN whitening contract failed")
    return TrainWhitening(
        center=center,
        linear_map=transform,
        pooled_eigenvalues=top,
        original_sensor_names=RAW_CHANNELS,
        trained_episodes=len(values),
    )


def build_native_v22_data(
    observed:ObservableEnvironments,
    whitener:TrainWhitening,
)->DiscoveryData:
    def split(full,labels,ids):
        transformed=whitener.transform(full)
        mask=np.ones_like(transformed,dtype=np.float32)
        return DiscoverySplit(
            x=transformed[:,:HISTORY],
            mask=mask[:,:HISTORY],
            future=transformed[:,HISTORY:HISTORY+HORIZON],
            future_mask=mask[:,HISTORY:HISTORY+HORIZON],
            y=np.asarray(labels,dtype=np.float32),
            oracle_states=None,
            episode_ids=np.asarray(ids,dtype=np.int64),
        )
    data=DiscoveryData(
        train=split(
            observed.train_x_full,observed.train_labels,observed.train_episode_ids,
        ),
        validation=split(
            observed.validation_x_full,
            observed.validation_labels,observed.validation_episode_ids,
        ),
        receipt={
            "channel_names":list(MODEL_CHANNELS),
            "unmixed_concepts_are_native_model_inputs_not_posthoc_probes":True,
            "both_arms_receive_identical_whitened_observable_input":True,
            "source_whitening_fitted_only_on_train":True,
            "observation_environments_not_given_to_native_task_model":True,
            "true_physical_or_instrument_latent_states_never_exported":True,
            "original_sensor_names":list(RAW_CHANNELS),
            "generator_family":observed.family,
        },
        medians=whitener.center,
        scales=whitener.pooled_eigenvalues,
    )
    return data


def load_oracle_v22_validation_for_postfit_audit(family:str)->dict[str,np.ndarray]:
    """ONLY postfit source diagnostic, never source creation/training/model selection."""
    _,_,_,latent=_simulate(family,SEEDS["validation"],SIZES["validation"])
    return {
        "generated_physical_source":latent[:,:HISTORY,0].astype(np.float32),
        "generated_instrument_source":latent[:,:HISTORY,1].astype(np.float32),
    }
