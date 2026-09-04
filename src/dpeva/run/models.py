"""Versioned, closed models for persisted scientific run evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dpeva.run.status import RunState


class RunModel(BaseModel):
    """Base model for run evidence; unknown fields are never silently ignored."""

    model_config = ConfigDict(extra="forbid")


class RunEvent(RunModel):
    state: RunState
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    kind: Literal["transition", "resume", "recovery", "force"] = "transition"
    attempt_id: int = Field(default=1, ge=1)


class FailureRecord(RunModel):
    category: Literal[
        "CONFIG",
        "CAPABILITY",
        "ENVIRONMENT",
        "EXECUTION",
        "ARTIFACT",
        "DATA_INTEGRITY",
        "UPSTREAM",
    ]
    message: str


class ArtifactRecord(RunModel):
    kind: str
    path: str
    producer_run: str
    status: Literal["declared", "verified", "missing"]
    checksum: str | None = None


class JobRecord(RunModel):
    name: str
    backend: Literal["local", "slurm"]
    job_id: str | None = None
    status: RunState
    failure: str | None = None


class RunManifest(RunModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    workflow: str
    status: RunState = RunState.CREATED
    source: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, str] = Field(default_factory=dict)
    config: dict[str, str] = Field(default_factory=dict)
    inputs: list[dict[str, str]] = Field(default_factory=list)
    jobs: list[JobRecord] = Field(default_factory=list)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    events: list[RunEvent] = Field(default_factory=list)
    failure: FailureRecord | None = None
