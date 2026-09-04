"""Atomic persistence for the run state and its evidence."""

from __future__ import annotations

import copy
import errno
import os
from pathlib import Path
from typing import Any, Literal

from dpeva.run.models import (
    ArtifactRecord,
    FailureRecord,
    JobRecord,
    RunEvent,
    RunManifest,
)
from dpeva.run.status import RunEventKind, RunState, transition


class StatusRecorder:
    """Persist a run manifest after every valid state or evidence update.

    The published manifest is private. ``manifest`` returns a deep copy so a
    caller cannot bypass transition and schema validation by mutating a nested
    list or record in place.
    """

    def __init__(self, path: Path, manifest: RunManifest, *, attempt_id: int = 1) -> None:
        if attempt_id < 1:
            raise ValueError("attempt_id must be positive")
        self.path = path
        self._manifest = manifest
        self.attempt_id = attempt_id

    @property
    def manifest(self) -> RunManifest:
        """Return an isolated snapshot of the last durably published manifest."""

        return self._manifest.model_copy(deep=True)

    @classmethod
    def create(
        cls,
        path: str | Path,
        run_id: str,
        workflow: str,
        *,
        source: dict[str, Any] | None = None,
        environment: dict[str, str] | None = None,
        config: dict[str, str] | None = None,
        inputs: list[dict[str, str]] | None = None,
        attempt_id: int = 1,
    ) -> "StatusRecorder":
        manifest = RunManifest(
            run_id=run_id,
            workflow=workflow,
            source=copy.deepcopy(source) if source is not None else {},
            environment=copy.deepcopy(environment) if environment is not None else {},
            config=copy.deepcopy(config) if config is not None else {},
            inputs=copy.deepcopy(inputs) if inputs is not None else [],
        )
        recorder = cls(Path(path), manifest, attempt_id=attempt_id)
        recorder._persist(manifest)
        recorder._manifest = manifest
        return recorder

    @classmethod
    def load(cls, path: str | Path, *, attempt_id: int = 1) -> "StatusRecorder":
        manifest_path = Path(path)
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        return cls(manifest_path, manifest, attempt_id=attempt_id)

    def transition(self, target: RunState, event: RunEventKind | None = None) -> RunState:
        """Validate and persist one state transition.

        A candidate is constructed and durably written before it becomes the
        recorder's published in-memory state. Illegal transitions and I/O
        failures therefore leave the recorder unchanged.
        """

        current_state = self._manifest.status
        next_state = transition(current_state, target, event)
        candidate = self._manifest.model_copy(deep=True)
        prior_failure = candidate.failure
        if (
            current_state in {RunState.PARTIAL, RunState.FAILED}
            and next_state not in {RunState.PARTIAL, RunState.FAILED}
            and prior_failure is not None
        ):
            candidate = self._enrich_terminal_failure(
                candidate, current_state, prior_failure, self.attempt_id
            )
        candidate.status = next_state
        if next_state not in {RunState.PARTIAL, RunState.FAILED}:
            # Recovery starts a new current attempt. The prior failed state
            # remains in events; current failure evidence must not leak into a
            # running/validated/submitted/finished manifest.
            candidate.failure = None
        kind = event.value if event is not None else "transition"
        candidate.events.append(
            RunEvent(
                state=next_state,
                kind=kind,
                attempt_id=self.attempt_id,
                failure=candidate.failure
                if next_state in {RunState.PARTIAL, RunState.FAILED}
                else None,
            )
        )
        candidate = self._validate(candidate)
        self._persist(candidate)
        return next_state

    def fail(self, *, category: str, message: str) -> RunState:
        return self._record_terminal_failure(RunState.FAILED, category=category, message=message)

    def partial(self, *, category: str, message: str) -> RunState:
        return self._record_terminal_failure(RunState.PARTIAL, category=category, message=message)

    def add_job(self, job: JobRecord) -> None:
        candidate = self._manifest.model_copy(deep=True)
        candidate.jobs.append(job)
        candidate = self._validate(candidate)
        self._persist(candidate)

    def add_artifact(self, artifact: ArtifactRecord) -> None:
        candidate = self._manifest.model_copy(deep=True)
        candidate.artifacts.append(artifact)
        candidate = self._validate(candidate)
        self._persist(candidate)

    def update_metadata(
        self,
        *,
        source: dict[str, Any] | None = None,
        environment: dict[str, str] | None = None,
        config: dict[str, str] | None = None,
        inputs: list[dict[str, str]] | None = None,
    ) -> None:
        """Update manifest metadata through the same atomic publication path."""

        candidate = self._manifest.model_copy(deep=True)
        if source is not None:
            candidate.source = copy.deepcopy(source)
        if environment is not None:
            candidate.environment = copy.deepcopy(environment)
        if config is not None:
            candidate.config = copy.deepcopy(config)
        if inputs is not None:
            candidate.inputs = copy.deepcopy(inputs)
        candidate = self._validate(candidate)
        self._persist(candidate)

    def record_event(
        self,
        *,
        kind: Literal["transition", "resume", "recovery", "force"],
        state: RunState | None = None,
        attempt_id: int | None = None,
    ) -> None:
        """Persist an explicit non-transition event, such as a force action."""

        candidate = self._manifest.model_copy(deep=True)
        candidate.events.append(
            RunEvent(
                state=state or candidate.status,
                kind=kind,
                attempt_id=attempt_id if attempt_id is not None else self.attempt_id,
            )
        )
        candidate = self._validate(candidate)
        self._persist(candidate)
        self._manifest = candidate

    def save(self) -> None:
        """Re-persist the last published manifest without accepting external mutation."""

        self._persist(self._manifest)

    def _record_terminal_failure(self, target: RunState, *, category: str, message: str) -> RunState:
        next_state = transition(self._manifest.status, target)
        candidate = self._manifest.model_copy(deep=True)
        candidate.status = next_state
        candidate.failure = FailureRecord(category=category, message=message)
        candidate.events.append(
            RunEvent(state=next_state, attempt_id=self.attempt_id, failure=candidate.failure)
        )
        candidate = self._validate(candidate)
        self._persist(candidate)
        return next_state

    @staticmethod
    def _validate(candidate: RunManifest) -> RunManifest:
        # ``model_copy(update=...)`` does not validate updates in Pydantic v2.
        # Round-trip through the model to enforce all field and root rules.
        return RunManifest.model_validate(candidate.model_dump())

    @staticmethod
    def _enrich_terminal_failure(
        candidate: RunManifest, state: RunState, failure: FailureRecord, attempt_id: int
    ) -> RunManifest:
        for index in range(len(candidate.events) - 1, -1, -1):
            event = candidate.events[index]
            if event.state is state:
                if event.failure is None:
                    candidate.events[index] = event.model_copy(update={"failure": failure})
                return candidate
        # A malformed-but-schema-valid legacy manifest may omit event history;
        # retain its manifest-level evidence in a synthetic terminal event.
        candidate.events.append(
            RunEvent(state=state, attempt_id=attempt_id, failure=failure)
        )
        return candidate

    def _persist(self, candidate: RunManifest) -> None:
        candidate = self._validate(candidate)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(candidate.model_dump_json(indent=2))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            # A successful replace is the publication point. If the directory
            # fsync reports a durability error, keep memory aligned with the
            # on-disk candidate while surfacing that error to the caller.
            self._manifest = candidate
            self._fsync_parent_directory()
        except BaseException:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def _fsync_parent_directory(self) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            directory_fd = os.open(str(self.path.parent), flags)
        except OSError as error:
            if not self._is_unsupported_directory_error(error):
                raise
            # Directory handles are unavailable on a few platforms/filesystems;
            # the file has still been flushed and the rename remains atomic.
            return
        try:
            try:
                os.fsync(directory_fd)
            except OSError as error:
                # A few platforms/filesystems do not support directory fsync;
                # propagate other errors so callers can react to the reduced
                # durability guarantee after publication.
                if self._is_unsupported_directory_error(error):
                    return
                raise
        finally:
            os.close(directory_fd)

    @staticmethod
    def _is_unsupported_directory_error(error: OSError) -> bool:
        return error.errno in {
            errno.EBADF,
            errno.EINVAL,
            errno.ENOSYS,
            errno.ENOTSUP,
            errno.EOPNOTSUPP,
        }
