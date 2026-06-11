"""Stable exploration backend contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ase import Atoms


WorkflowType = Literal["md", "relax"]
ResultStatus = Literal["success", "failed"]


@dataclass(slots=True)
class ExplorationArtifact:
    kind: str
    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExplorationRequest:
    backend: str
    workflow_type: str
    work_dir: Path
    config_path: Path
    input_structures: list[Atoms] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExplorationResult:
    backend: str
    workflow_type: str
    work_dir: Path
    status: ResultStatus
    structures: list[Atoms] = field(default_factory=list)
    artifacts: list[ExplorationArtifact] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None


class ExplorationBackend(ABC):
    name: str
    supported_workflow_types: set[str]

    @abstractmethod
    def run(self, request: ExplorationRequest) -> ExplorationResult:
        """Run the exploration backend and return a normalized result."""
