"""Compatibility declarations for optional upstream integrations."""

from .deepmd import (
    CapabilityEvidence,
    CapabilityKey,
    CapabilityMatrix,
    CapabilityRecord,
    CapabilityUnavailable,
    validate_promotion_evidence,
)
from .attestation import CapabilityAttestation
from .adapter import DeepMDAdapter

__all__ = [
    "CapabilityEvidence",
    "CapabilityKey",
    "CapabilityMatrix",
    "CapabilityRecord",
    "CapabilityUnavailable",
    "validate_promotion_evidence",
    "DeepMDAdapter",
    "CapabilityAttestation",
]
