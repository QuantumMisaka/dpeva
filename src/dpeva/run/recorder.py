"""Atomic persistence for the run state and its evidence."""

from __future__ import annotations

import os
from pathlib import Path

from dpeva.run.models import (
    ArtifactRecord,
    FailureRecord,
    JobRecord,
    RunEvent,
    RunManifest,
)
from dpeva.run.status import RunEventKind, RunState, transition


class StatusRecorder:
    """Persist a run manifest after every valid state or evidence update."""

    def __init__(self, path: Path, manifest: RunManifest, *, attempt_id: int = 1) -> None:
        self.path = path
        self.manifest = manifest
        self.attempt_id = attempt_id

    @classmethod
    def create(
        cls,
        path: str | Path,
        run_id: str,
        workflow: str,
        *,
        source: dict[str, object] | None = None,
        environment: dict[str, str] | None = None,
        config: dict[str, str] | None = None,
        inputs: list[dict[str, str]] | None = None,
        attempt_id: int = 1,
    ) -> "StatusRecorder":
        recorder = cls(
            Path(path),
            RunManifest(
                run_id=run_id,
                workflow=workflow,
                source=source or {},
                environment=environment or {},
                config=config or {},
                inputs=inputs or [],
            ),
            attempt_id=attempt_id,
        )
        recorder._write()
        return recorder

    @classmethod
    def load(cls, path: str | Path, *, attempt_id: int = 1) -> "StatusRecorder":
        manifest_path = Path(path)
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        return cls(manifest_path, manifest, attempt_id=attempt_id)

    def transition(self, target: RunState, event: RunEventKind | None = None) -> RunState:
        """Validate and persist one state transition.

        Validation is performed before mutating the in-memory manifest, so an
        illegal transition leaves both memory and the persisted evidence intact.
        """

        next_state = transition(self.manifest.status, target, event)
        kind = event.value if event is not None else "transition"
        self.manifest.status = next_state
        self.manifest.events.append(
            RunEvent(state=next_state, kind=kind, attempt_id=self.attempt_id)
        )
        self._write()
        return next_state

    def fail(self, *, category: str, message: str) -> RunState:
        self._record_terminal_failure(RunState.FAILED, category=category, message=message)
        return self.manifest.status

    def partial(self, *, category: str, message: str) -> RunState:
        self._record_terminal_failure(RunState.PARTIAL, category=category, message=message)
        return self.manifest.status

    def add_job(self, job: JobRecord) -> None:
        self.manifest.jobs.append(job)
        self._write()

    def add_artifact(self, artifact: ArtifactRecord) -> None:
        self.manifest.artifacts.append(artifact)
        self._write()

    def save(self) -> None:
        """Persist caller-managed manifest evidence atomically."""

        self._write()

    def _record_terminal_failure(self, target: RunState, *, category: str, message: str) -> None:
        next_state = transition(self.manifest.status, target)
        failure = FailureRecord(category=category, message=message)
        self.manifest.status = next_state
        self.manifest.failure = failure
        self.manifest.events.append(
            RunEvent(state=next_state, attempt_id=self.attempt_id)
        )
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(self.manifest.model_dump_json(indent=2))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

