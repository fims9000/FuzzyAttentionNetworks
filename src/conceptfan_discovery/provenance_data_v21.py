"""V21: distinct measurement-vs-instrument synthetic source audit.

Only model-visible [B,24,4] measurements enter ConceptFAN. A second,
independently registered device reading is for TRAIN-only provenance and
simulated reviewer labels, never a direct neural input. Hidden physical
source and instrument truth are returned only by an explicit post-training
audit function, never in DiscoverySplit.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .data import DiscoveryData,DiscoverySplit


FAMILIES=(
    "physical_predictive",
    "physical_nontask",
    "artifact_predictive",
    "artifact_nontask",
    "artifact_common_mode_reference",
)
SEEDS={"train":405011,"validation":405012,"test":405013}
SIZES={"train":960,"validation":240,"test":240}
HISTORY=24
HORIZON=6
CHANNELS=("prior_device_a","prior_device_b","new_measurement_device_b","suspect_device_a")
DEVICE_IDS=("device_a","device_b","device_b","device_a")
REFERENCE_DEVICE_ID="device_c"
DISCOVERY_N=350
SCREEN_N=300
CORROBORATION_N=250


@dataclass(frozen=True)
class ProvenanceCohort:
    discovery: DiscoveryData
    independent_train_reference: np.ndarray
    independent_validation_reference: np.ndarray
    model_sensor_names: tuple[str,...]
    model_sensor_devices: tuple[str,...]
    reference_device_id: str


def _simulate(family:str, seed:int, n:int):
    if family not in FAMILIES or n<1:
        raise ValueError("Unregistered V21 generating scenario")
    rng=np.random.default_rng(int(seed))
    t=HISTORY+HORIZON
    physical_old=np.zeros((n,t),dtype=np.float64)
    physical_new=np.zeros((n,t),dtype=np.float64)
    instrument=np.zeros((n,t),dtype=np.float64)
    for j in range(t):
        u=rng.normal(size=(n,3))
        if j==0:
            physical_old[:,j]=.65*u[:,0]
            physical_new[:,j]=.60*u[:,1]
            instrument[:,j]=.42*u[:,2]
        else:
            physical_old[:,j]=.66*physical_old[:,j-1]+.58*u[:,0]
            physical_new[:,j]=.84*physical_new[:,j-1]+.62*u[:,1]
            instrument[:,j]=.92*instrument[:,j-1]+.43*u[:,2]
    noise=rng.normal(scale=.11,size=(n,t,4))
    raw=np.empty_like(noise)
    raw[:,:,0]=.85*physical_old+noise[:,:,0]
    raw[:,:,1]=.75*physical_old+noise[:,:,1]
    physical_mode=family.startswith("physical_")
    if physical_mode:
        raw[:,:,2]=1.02*physical_new+noise[:,:,2]
        raw[:,:,3]=rng.normal(scale=.20,size=(n,t))
    else:
        raw[:,:,2]=rng.normal(scale=.20,size=(n,t))
        raw[:,:,3]=1.08*instrument+noise[:,:,3]

    # This separate registered device is never an input to model or proposal.
    reference=.92*physical_new+rng.normal(scale=.13,size=(n,t))
    if family=="artifact_common_mode_reference":
        # A negative control: independent device ID alone cannot rule out
        # a shared artifact contaminating both instrument acquisition paths.
        reference=.95*instrument+.06*physical_new+rng.normal(scale=.13,size=(n,t))
    if family in ("physical_predictive",):
        latent_future=physical_new[:,HISTORY:HISTORY+HORIZON].mean(1)
    elif family in ("artifact_predictive","artifact_common_mode_reference"):
        latent_future=instrument[:,HISTORY:HISTORY+HORIZON].mean(1)
    else:
        latent_future=np.zeros(n)
    probabilities=1/(1+np.exp(-np.clip(2.8*latent_future,-20,20)))
    labels=(rng.random(n)<probabilities).astype(np.float32)
    if len(np.unique(labels))<2:
        raise ValueError("Degenerate V21 target labels")
    return raw,reference,labels,physical_new,instrument


def make_v21_provenance_cohort(family:str)->ProvenanceCohort:
    generated={
        key:_simulate(family,SEEDS[key],SIZES[key])
        for key in ("train","validation")
    }
    raw_train,ref_train,_,_,_=generated["train"]
    median=np.median(raw_train[:,:HISTORY,:],axis=(0,1))
    iqr=np.maximum(
        np.quantile(raw_train[:,:HISTORY,:],.75,axis=(0,1))
        -np.quantile(raw_train[:,:HISTORY,:],.25,axis=(0,1)),.05,
    )
    reference_median=float(np.median(ref_train[:,:HISTORY]))
    reference_iqr=float(max(
        np.quantile(ref_train[:,:HISTORY],.75)
        -np.quantile(ref_train[:,:HISTORY],.25),.05
    ))
    def split(part,first_id):
        raw,_,labels,_,_=generated[part]
        normalized=np.clip((raw-median)/iqr,-6,6).astype(np.float32)
        return DiscoverySplit(
            x=normalized[:,:HISTORY],
            mask=np.ones_like(normalized[:,:HISTORY],dtype=np.float32),
            future=normalized[:,HISTORY:HISTORY+HORIZON],
            future_mask=np.ones_like(normalized[:,HISTORY:HISTORY+HORIZON],dtype=np.float32),
            y=labels,
            oracle_states=None,
            episode_ids=np.arange(first_id,first_id+len(labels),dtype=np.int64),
        )
    train=split("train",0)
    val=split("validation",10000)
    data=DiscoveryData(
        train=train,validation=val,
        receipt={
            "channel_names":list(CHANNELS),
            "model_sensor_device_ids":list(DEVICE_IDS),
            "reference_device_id":REFERENCE_DEVICE_ID,
            "source_audit_truth_excluded_from_all_splits":True,
            "reference_measured_but_not_in_model_input":True,
            "generator_family_id":family,
            "train_only_median_and_iqr":True,
            "synthetic_test_episodes_sealed":SIZES["test"],
        },medians=median,scales=iqr,
    )
    def norm_reference(part):
        # Measured external reference; independently train-scaled.
        x=generated[part][1][:,:HISTORY]
        return np.clip((x-reference_median)/reference_iqr,-6,6).astype(np.float32)
    return ProvenanceCohort(
        discovery=data,
        independent_train_reference=norm_reference("train"),
        independent_validation_reference=norm_reference("validation"),
        model_sensor_names=CHANNELS,
        model_sensor_devices=DEVICE_IDS,
        reference_device_id=REFERENCE_DEVICE_ID,
    )


def load_oracle_source_truth_for_postfit_audit(family:str, *, split:str="validation")->dict[str,np.ndarray]:
    """Auditing ONLY after all native fits. Never called by any trainer/gate."""
    if split not in ("validation",):
        raise ValueError("V21 only permits post-training validation truth")
    _,_,_,physical,instrument=_simulate(family,SEEDS[split],SIZES[split])
    return {
        "synthetic_physical_new":physical[:,:HISTORY].astype(np.float32),
        "synthetic_instrument_artifact":instrument[:,:HISTORY].astype(np.float32),
    }
