"""V7: explicitly separate physiological evidence from observation-policy evidence.

When sampling frequency predicts mortality in the ICU, removing mask
information from the *physiological* concept can degrade AUPRC. Putting
missingness back in the same concept would reintroduce semantic confusion.

Instead use TWO distinguishable, inspectable, jointly trained bottlenecks:

  physical C_k = native signed measured operators, UNKNOWN if unseen;
  process O_j  = signed sampling-rate/recency/staleness/change operators,
                 derived solely from M (not measurement values);
  logit = bias + SUM_k beta_k*(C_k - 0.5)
               + SUM_j gamma_j*(O_j - 0.5).

No direct raw-X or raw-mask bypass to a hidden classifier. Both sums
directly influence the fitted task logit during learning. Forecasting
and denoising clinical measurements are routed only through physical C.

Sampling-process concepts are NEVER named physiological diseases and
their direct mathematical intervention is not a treatment effect.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import DiscoveryOutput,sparsemax
from .unknown_neutral_v6 import UnknownNeutralSpectralConceptFAN


OBSERVATION_OPERATORS=(
    "observation_fraction","recent_observation_fraction",
    "normalized_time_since_last_observation","sampling_transition_fraction",
)


def observation_process_operator_bank(mask:torch.Tensor)->torch.Tensor:
    """Deterministic time-aware operator bank of OBSERVATION events ONLY."""
    if mask.ndim!=3 or mask.shape[1]<3 or mask.shape[2]<2:
        raise ValueError("Expected observed mask [patients,time,>=2_channels]")
    if not torch.isfinite(mask).all() or torch.any((mask<0)|(mask>1)):
        raise ValueError("Observation mask must be finite and in [0,1]")
    if not torch.logical_or(mask==0,mask==1).all():
        raise ValueError("Expected binary observation events, not propensity scores")
    b,t,d=mask.shape
    rate=mask.mean(dim=1)
    recent=mask[:,-max(2,t//6):,:].mean(dim=1)
    observed_times=torch.arange(t,device=mask.device).view(1,t,1).expand(b,t,d)
    last=torch.where(mask>0,observed_times,-1).amax(dim=1)
    staleness=((t-1)-last).to(dtype=mask.dtype)/float(t)
    changes=(mask[:,1:,:]-mask[:,:-1,:]).abs().mean(dim=1)
    # Flatten by operator-major convention for explicit passports.
    return torch.cat((rate,recent,staleness,changes),dim=-1)


class ObservationProcessConceptHead(nn.Module):
    def __init__(self,n_channels:int,n_concepts:int=3):
        super().__init__()
        if n_channels<2 or not 1<=n_concepts<=8:
            raise ValueError("Invalid observable sensor or mask concept capacity")
        self.n_channels=int(n_channels)
        self.n_concepts=int(n_concepts)
        d=n_channels*len(OBSERVATION_OPERATORS)
        self.feature_logits=nn.Parameter(torch.randn(n_concepts,2*d)*.1)
        self.threshold_logits=nn.Parameter(torch.zeros(n_concepts))
        self.width_logits=nn.Parameter(torch.full((n_concepts,),-1.75))
        self.signed_beta_raw=nn.Parameter(torch.randn(n_concepts)*.08)

    def feature_weights(self)->torch.Tensor:
        return sparsemax(self.feature_logits,dim=-1)

    def direct_weights(self)->torch.Tensor:
        return 3.0*torch.tanh(self.signed_beta_raw)

    def forward(
        self,mask:torch.Tensor,
    )->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        base=observation_process_operator_bank(mask)
        signed=torch.cat((base,-base),dim=-1)
        weights=self.feature_weights()
        score=signed@weights.T
        threshold=torch.tanh(self.threshold_logits)
        width=.08+F.softplus(self.width_logits)
        concept=torch.sigmoid((score-threshold)/width)
        contribution=(concept-.5)*self.direct_weights()
        return concept,contribution,score,weights

    def passports(self,channels:tuple[str,...])->list[dict]:
        if len(channels)!=self.n_channels:
            raise ValueError("Channel names must exactly match observation operators")
        names=[
            f"{direction}|{operator}|{channel}"
            for direction in ("positive","negative")
            for operator in OBSERVATION_OPERATORS
            for channel in channels
        ]
        weights=self.feature_weights().detach().cpu().numpy()
        beta=self.direct_weights().detach().cpu().numpy()
        threshold=torch.tanh(self.threshold_logits).detach().cpu().numpy()
        width=(.08+F.softplus(self.width_logits)).detach().cpu().numpy()
        items=[]
        for slot in range(self.n_concepts):
            indices=weights[slot].argsort()[::-1][:6]
            items.append({
                "observation_concept":slot,
                "evidence_family":"care_observation_policy_not_patient_physiology",
                "math":"sigmoid((signed mask process operators dot simplex weights - learned threshold) / learned width)",
                "top_signed_sampling_operators":[
                    {"operator":names[i],"simplex_weight":float(weights[slot,i])}
                    for i in indices if weights[slot,i]>1e-7
                ],
                "learned_threshold":float(threshold[slot]),
                "learned_width":float(width[slot]),
                "signed_direct_logit_coefficient":float(beta[slot]),
                "causal_intervention_claim":False,
                "clinical_diagnosis_identity_verified":False,
                "directly_used_in_training_task_logit":True,
            })
        return items


@dataclass(frozen=True)
class DualProcessOutput:
    physical:DiscoveryOutput
    observation_concepts:torch.Tensor
    observation_contributions:torch.Tensor
    logits:torch.Tensor
    probabilities:torch.Tensor

    def decomposition_error(self)->torch.Tensor:
        return (
            self.logits
            -self.physical.decision_bias
            -self.physical.contributions.sum(dim=-1)
            -self.observation_contributions.sum(dim=-1)
        ).abs().max()

    def contributions(self)->torch.Tensor:
        return torch.cat(
            (self.physical.contributions,self.observation_contributions),
            dim=-1,
        )


class DualProcessConceptFAN(nn.Module):
    def __init__(
        self,*,observable_loadings,channels:tuple[str,...],
        forecast_horizon:int,n_observation_concepts:int=3,
        freeze_physical_definitions:bool=False,
    ):
        super().__init__()
        self.physical=UnknownNeutralSpectralConceptFAN(
            observable_loadings=observable_loadings,
            channels=channels,
            forecast_horizon=forecast_horizon,
            freeze_definitions=freeze_physical_definitions,
        )
        self.process=ObservationProcessConceptHead(
            n_channels=len(channels),
            n_concepts=n_observation_concepts,
        )

    def forward(
        self,values:torch.Tensor,observed_mask:torch.Tensor,
    )->DualProcessOutput:
        physical=self.physical(values,observed_mask)
        observations,contributions,_,_=self.process(observed_mask)
        logit=physical.logits+contributions.sum(dim=-1)
        return DualProcessOutput(
            physical=physical,
            observation_concepts=observations,
            observation_contributions=contributions,
            logits=logit,
            probabilities=torch.sigmoid(logit),
        )

    def intervene_on_observation_concept(
        self,out:DualProcessOutput,concept_id:int,override:torch.Tensor,
    )->torch.Tensor:
        if not 0<=concept_id<self.process.n_concepts:
            raise ValueError("Unknown observation-process concept index")
        if override.shape!=out.observation_concepts[:,concept_id].shape:
            raise ValueError("Override must give one interpretable score per patient")
        if torch.any((override<0)|(override>1)):
            raise ValueError("Concept score must remain in [0,1]")
        old=out.observation_contributions[:,concept_id]
        new=(override-.5)*self.process.direct_weights()[concept_id]
        return out.logits-old+new

    def intervene_on_physical_concept(
        self,out:DualProcessOutput,concept_id:int,override:torch.Tensor,
    )->torch.Tensor:
        physical_logit,_=self.physical.intervene(
            out.physical,concept_id,override
        )
        return physical_logit+out.observation_contributions.sum(dim=-1)


def dual_training_objective(
    model:DualProcessConceptFAN,
    out:DualProcessOutput,
    target:torch.Tensor,
    physical_loss:torch.Tensor,
    physical_bce:torch.Tensor,
    *,
    pos_weight:torch.Tensor,
    mask_coeff_l1:float=.03,
    mask_redundancy_weight:float=.01,
)->tuple[torch.Tensor,dict]:
    if mask_coeff_l1<0 or mask_redundancy_weight<0:
        raise ValueError("No negative concept sparsity regularization")
    # Replace, rather than duplicate, physical-only task BCE.
    total_task=F.binary_cross_entropy_with_logits(
        out.logits,target,pos_weight=pos_weight,
    )
    weight=model.process.feature_weights()
    support=(torch.sqrt(weight+1e-8).sum(dim=-1)-1.).clamp_min(0.).mean()
    gram=weight@weight.T
    off=gram-torch.diag_embed(gram.diagonal())
    redundant=off.square().mean()
    observational_magnitude=model.process.direct_weights().abs().mean()
    total=(
        physical_loss-physical_bce+total_task
        +float(mask_coeff_l1)*(observational_magnitude+support*.05)
        +float(mask_redundancy_weight)*redundant
    )
    return total,{
        "total_task_bce":total_task,
        "physical_task_replaced_not_doubled":True,
        "mask_coefficient_l1":observational_magnitude,
        "mask_dictionary_redundancy":redundant,
        "total":total,
    }
