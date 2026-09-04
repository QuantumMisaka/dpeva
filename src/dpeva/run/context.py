"""Immutable allocation and evidence context for one scientific run."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import fcntl

from dpeva.run.models import ArtifactRecord
from dpeva.run.recorder import StatusRecorder
from dpeva.run.status import RunState


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TERMINAL_STATES = frozenset({RunState.FAILED, RunState.FINISHED})
_CHECKSUM_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class RunOptions:
    """Explicit allocation policy for a run identity."""

    run_id: str | None = None
    resume: bool = False
    force: bool = False
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.resume and self.force:
            raise ValueError("--resume and --force are mutually exclusive")
        if (self.resume or self.force) and not self.run_id:
            raise ValueError("--resume/--force requires --run-id")
        if self.force and not self.reason:
            raise ValueError("--force requires --reason")
        if self.run_id is not None:
            _validate_component(self.run_id, "run_id")


@dataclass(frozen=True)
class RunContext:
    """The immutable identity and persistence handles for one run attempt."""

    work_dir: Path
    run_dir: Path
    run_id: str
    workflow: str
    attempt_id: int
    recorder: StatusRecorder

    @classmethod
    def create(
        cls,
        work_dir: str | Path,
        workflow: str,
        options: RunOptions,
        original_config: dict[str, Any],
        normalized_config: dict[str, Any],
    ) -> "RunContext":
        root = Path(work_dir).expanduser().resolve()
        _validate_component(workflow, "workflow")
        if not isinstance(options, RunOptions):
            raise TypeError("options must be a RunOptions instance")

        runs_root = root / ".dpeva" / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        if runs_root.resolve() != runs_root:
            raise ValueError("run root must not resolve outside work_dir")

        if options.run_id is None:
            return cls._create_new_generated(
                root,
                runs_root,
                workflow,
                original_config,
                normalized_config,
            )

        run_dir = runs_root / options.run_id
        if options.resume:
            return cls._resume_existing(root, run_dir, workflow)
        if options.force:
            return cls._force_existing(
                root,
                run_dir,
                workflow,
                options,
                original_config,
                normalized_config,
            )

        try:
            run_dir.mkdir()
        except FileExistsError:
            raise FileExistsError(f"run already exists: {options.run_id}") from None
        return cls._initialize_new(
            root,
            run_dir,
            options.run_id,
            workflow,
            original_config,
            normalized_config,
        )

    @classmethod
    def _create_new_generated(
        cls,
        root: Path,
        runs_root: Path,
        workflow: str,
        original_config: dict[str, Any],
        normalized_config: dict[str, Any],
    ) -> "RunContext":
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for _ in range(100):
            run_id = f"{workflow}-{timestamp}-{secrets.token_hex(3)}"
            run_dir = runs_root / run_id
            try:
                run_dir.mkdir()
            except FileExistsError:
                continue
            return cls._initialize_new(
                root,
                run_dir,
                run_id,
                workflow,
                original_config,
                normalized_config,
            )
        raise FileExistsError("could not allocate a unique generated run id")

    @classmethod
    def _initialize_new(
        cls,
        root: Path,
        run_dir: Path,
        run_id: str,
        workflow: str,
        original_config: dict[str, Any],
        normalized_config: dict[str, Any],
    ) -> "RunContext":
        try:
            _atomic_json_write(run_dir / "config.original.json", original_config)
            _atomic_json_write(run_dir / "config.resolved.json", normalized_config)
            recorder = StatusRecorder.create(
                run_dir / "run.json",
                run_id,
                workflow,
                config={
                    "original": "config.original.json",
                    "resolved": "config.resolved.json",
                },
            )
        except BaseException as error:
            _preserve_initialization_failure(run_dir, run_id, workflow, error)
            raise
        return cls(root, run_dir, run_id, workflow, 1, recorder)

    @classmethod
    def _resume_existing(cls, root: Path, run_dir: Path, workflow: str) -> "RunContext":
        with _run_lock(run_dir):
            recorder = _load_existing(run_dir, workflow)
            current = recorder.manifest.status
            if current in _TERMINAL_STATES:
                raise ValueError(
                    f"cannot resume terminal run {recorder.manifest.run_id!r} "
                    f"in state {current.value}"
                )
            attempt_id = _next_attempt_id(recorder)
            recorder.attempt_id = attempt_id
            recorder.record_event(kind="resume", attempt_id=attempt_id)
            return cls(root, run_dir, recorder.manifest.run_id, workflow, attempt_id, recorder)

    @classmethod
    def _force_existing(
        cls,
        root: Path,
        run_dir: Path,
        workflow: str,
        options: RunOptions,
        original_config: dict[str, Any],
        normalized_config: dict[str, Any],
    ) -> "RunContext":
        with _run_lock(run_dir):
            recorder = _load_existing(run_dir, workflow)
            previous_manifest = (run_dir / "run.json").read_bytes()
            previous_attempt = _next_attempt_id(recorder)
            _archive_manifest(run_dir, previous_manifest, previous_attempt - 1)

            version = previous_attempt
            original_path = run_dir / f"config.original.attempt-{version:04d}.json"
            resolved_path = run_dir / f"config.resolved.attempt-{version:04d}.json"
            written_snapshots: list[Path] = []
            try:
                _atomic_json_write(original_path, original_config, overwrite=False)
                written_snapshots.append(original_path)
                _atomic_json_write(resolved_path, normalized_config, overwrite=False)
                written_snapshots.append(resolved_path)
                fresh = StatusRecorder.create(
                    run_dir / "run.json",
                    recorder.manifest.run_id,
                    workflow,
                    config={
                        "original": original_path.name,
                        "resolved": resolved_path.name,
                    },
                    attempt_id=previous_attempt,
                )
                fresh.record_event(
                    kind="force",
                    attempt_id=previous_attempt,
                    reason=options.reason,
                )
            except BaseException:
                _clean_unpublished_config_snapshots(
                    run_dir,
                    previous_manifest,
                    written_snapshots,
                )
                raise
            return cls(root, run_dir, recorder.manifest.run_id, workflow, previous_attempt, fresh)

    def register_verified_artifacts(self, kind: str, paths: Sequence[Path]) -> None:
        """Register existing, non-empty files with streaming SHA-256 identity."""

        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("artifact kind must be a non-empty string")

        records: list[ArtifactRecord] = []
        for supplied in paths:
            candidate = Path(supplied)
            if not candidate.is_absolute():
                candidate = self.work_dir / candidate
            resolved = candidate.resolve(strict=False)
            try:
                relative = resolved.relative_to(self.work_dir)
            except ValueError:
                raise ValueError(f"artifact path is outside work_dir: {supplied}") from None
            if not resolved.is_file():
                raise ValueError(f"artifact path is not a regular file: {supplied}")
            if resolved.stat().st_size == 0:
                raise ValueError(f"artifact path is empty: {supplied}")
            records.append(
                ArtifactRecord(
                    kind=kind,
                    path=relative.as_posix(),
                    producer_run=self.run_id,
                    status="verified",
                    checksum=_sha256(resolved),
                )
            )

        self.recorder.add_artifacts(records)


def _validate_component(value: str, label: str) -> None:
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f"{label} must be a safe single path component")


def _load_existing(run_dir: Path, workflow: str) -> StatusRecorder:
    if run_dir.is_symlink():
        raise ValueError("run directory must not be a symlink")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run does not exist: {run_dir.name}")
    recorder = StatusRecorder.load(run_dir / "run.json")
    if recorder.manifest.workflow != workflow:
        raise ValueError(
            f"run workflow mismatch: expected {workflow!r}, "
            f"found {recorder.manifest.workflow!r}"
        )
    if recorder.manifest.run_id != run_dir.name:
        raise ValueError("run manifest identity does not match its directory")
    return recorder


@contextmanager
def _run_lock(run_dir: Path):
    """Serialize a run's resume/force read-modify-write sequence."""

    if run_dir.is_symlink() or not run_dir.is_dir():
        if run_dir.is_symlink():
            raise ValueError("run directory must not be a symlink")
        raise FileNotFoundError(f"run does not exist: {run_dir.name}")
    lock_path = run_dir / ".context.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise OSError(f"cannot open run lock {lock_path}: {error}") from error
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _next_attempt_id(recorder: StatusRecorder) -> int:
    attempts = [event.attempt_id for event in recorder.manifest.events]
    return max([recorder.attempt_id, *attempts], default=1) + 1


def _archive_manifest(run_dir: Path, payload: bytes, attempt_id: int) -> None:
    attempts_dir = run_dir / "attempts"
    if attempts_dir.is_symlink():
        raise ValueError("attempt archive directory must not be a symlink")
    attempts_dir.mkdir(exist_ok=True)
    candidate_id = max(1, attempt_id)
    while True:
        archive = attempts_dir / f"attempt-{candidate_id:04d}.json"
        try:
            with archive.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            return
        except FileExistsError:
            candidate_id += 1
        except BaseException:
            archive.unlink(missing_ok=True)
            raise


def _atomic_json_write(path: Path, value: dict[str, Any], *, overwrite: bool = True) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
        if not overwrite and path.exists():
            raise FileExistsError(f"refusing to overwrite config snapshot: {path}")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _preserve_initialization_failure(
    run_dir: Path, run_id: str, workflow: str, error: BaseException
) -> None:
    """Leave a failed manifest behind when a newly allocated run cannot initialize."""

    manifest_path = run_dir / "run.json"
    try:
        if manifest_path.exists():
            recorder = StatusRecorder.load(manifest_path)
        else:
            recorder = StatusRecorder.create(manifest_path, run_id, workflow)
        if recorder.manifest.status not in _TERMINAL_STATES:
            recorder.fail(
                category="CONFIG",
                message=f"run initialization failed: {type(error).__name__}: {error}",
            )
    except BaseException:
        # Preserve the original initialization error. A best-effort failed
        # manifest is preferable, but must not mask the actionable root cause.
        return


def _clean_unpublished_config_snapshots(
    run_dir: Path, previous_manifest: bytes, paths: Sequence[Path]
) -> None:
    """Remove only versioned snapshots if the old manifest is still published."""

    manifest_path = run_dir / "run.json"
    try:
        if manifest_path.read_bytes() != previous_manifest:
            return
    except OSError:
        return
    for path in paths:
        path.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHECKSUM_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()
