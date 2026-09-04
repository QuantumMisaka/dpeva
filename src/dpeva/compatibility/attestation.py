"""Machine-readable, per-capability qualification attestations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from .deepmd import CapabilityKey


class CapabilityAttestation(BaseModel):
    """The smallest evidence object a producer may issue for promotion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["finished"]
    returncode: StrictInt
    capability_key: CapabilityKey
    verification_command: StrictStr = Field(min_length=1)
    deepmd_version: Literal["DeePMD-kit v3.2.0"]
    source: Literal["cpu-contract", "sai-v100-qualification"]
    case: StrictStr | None = Field(default=None, min_length=1)
    job_id: StrictInt | StrictStr | None = None
    gpu: StrictStr | None = None

    @model_validator(mode="after")
    def validate_success(self) -> "CapabilityAttestation":
        if self.returncode != 0:
            raise ValueError("finished attestation requires returncode 0")
        if self.source == "sai-v100-qualification":
            if self.job_id is None or not str(self.job_id).isdigit():
                raise ValueError("SAI attestation requires numeric job_id")
            if not self.gpu or "v100" not in self.gpu.lower():
                raise ValueError("SAI attestation requires V100 GPU evidence")
        return self

    def model_dump_json_compact(self) -> str:
        return self.model_dump_json(indent=2) + "\n"


__all__ = ["CapabilityAttestation"]
