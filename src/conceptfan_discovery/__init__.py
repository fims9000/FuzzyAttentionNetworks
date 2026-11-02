"""Auditable, fully train-time discovery of dynamic predictive concepts.

Experimental research module: does not overwrite the frozen ConceptFAN results.
"""
from .model import PredictiveDynamicsConceptFAN, discovery_loss
from .operators import CHANNELS, OPERATORS, causal_operator_bank

__all__ = ["PredictiveDynamicsConceptFAN", "discovery_loss", "CHANNELS",
           "OPERATORS", "causal_operator_bank"]
