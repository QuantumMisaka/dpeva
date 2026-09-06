"""Immutable allocation and evidence context for one scientific run."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import subprocess
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
from typing import Any, Sequence

import fcntl

from dpeva.run.models import ArtifactRecord, RunEvent
from dpeva.run.recorder import StatusRecorder
from dpeva.run.status import RunState
from dpeva.run.artifacts import AttemptOutputBaseline


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TERMINAL_STATES = frozenset({RunState.FAILED, RunState.FINISHED})
_CHECKSUM_CHUNK_SIZE = 1024 * 1024
_STRUCTURAL_ENTRY_LIMIT = 256
_STRUCTURAL_NODE_LIMIT = 4096
_RUNTIME_FINGERPRINT_VERSION = "1"
_RUNTIME_FINGERPRINT_SCOPE = ["src/dpeva", "pyproject.toml"]
_RUNTIME_GIT_PATHSPEC = [
    *_RUNTIME_FINGERPRINT_SCOPE,
    ":(exclude,glob)**/.dpeva",
    ":(exclude,glob)**/.dpeva/**",
]
_DEFAULT_CONFIG_METADATA: dict[str, Any] = {
    "schema_version": "1.0",
    "input_schema_version": "1.0",
    "migration_warnings": [],
}


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
        if self.force and (not isinstance(self.reason, str) or not self.reason.strip()):
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
        source: dict[str, Any] | None = None,
        inputs: list[dict[str, str]] | None = None,
        config_metadata: dict[str, Any] | None = None,
        source_factory: Callable[[], dict[str, Any]] | None = None,
        input_factories: Sequence[Callable[[], dict[str, str]]] | None = None,
    ) -> "RunContext":
        root = Path(work_dir).expanduser().resolve()
        _validate_component(workflow, "workflow")
        if not isinstance(options, RunOptions):
            raise TypeError("options must be a RunOptions instance")
        # Reject contradictory explicit evidence before allocating a run
        # directory.  Factory failures happen after allocation and are
        # governed; malformed explicit evidence is a caller error with no run.
        _merge_inputs([], inputs or [])

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
                source,
                inputs,
                config_metadata,
                source_factory,
                input_factories,
            )

        run_dir = runs_root / options.run_id
        if options.resume:
            return cls._resume_existing(
                root, run_dir, workflow, original_config, normalized_config,
                source, inputs, config_metadata, source_factory, input_factories,
            )
        if options.force:
            return cls._force_existing(
                root,
                run_dir,
                workflow,
                options,
                original_config,
                normalized_config,
                source,
                inputs,
                config_metadata,
                source_factory,
                input_factories,
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
            source,
            inputs,
            config_metadata,
            source_factory,
            input_factories,
        )

    @classmethod
    def _create_new_generated(
        cls,
        root: Path,
        runs_root: Path,
        workflow: str,
        original_config: dict[str, Any],
        normalized_config: dict[str, Any],
        source: dict[str, Any] | None,
        inputs: list[dict[str, str]] | None,
        config_metadata: dict[str, Any] | None,
        source_factory: Callable[[], dict[str, Any]] | None,
        input_factories: Sequence[Callable[[], dict[str, str]]] | None,
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
                source,
                inputs,
                config_metadata,
                source_factory,
                input_factories,
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
        source: dict[str, Any] | None,
        inputs: list[dict[str, str]] | None,
        config_metadata: dict[str, Any] | None,
        source_factory: Callable[[], dict[str, Any]] | None,
        input_factories: Sequence[Callable[[], dict[str, str]]] | None,
    ) -> "RunContext":
        published_snapshots: list[Path] = []
        try:
            _atomic_json_write(run_dir / "config.original.json", original_config)
            published_snapshots.append(run_dir / "config.original.json")
            _atomic_json_write(run_dir / "config.resolved.json", normalized_config)
            published_snapshots.append(run_dir / "config.resolved.json")
            config_references = {
                "original": "config.original.json",
                "resolved": "config.resolved.json",
            }
            metadata = _metadata_or_default(config_metadata)
            _atomic_json_write(run_dir / "config.metadata.json", metadata)
            published_snapshots.append(run_dir / "config.metadata.json")
            config_references["metadata"] = "config.metadata.json"
            initial_inputs = _merge_inputs([], inputs or [])
            recorder = StatusRecorder.create(
                run_dir / "run.json",
                run_id,
                workflow,
                config=config_references,
                source=source if source_factory is None else {},
                inputs=initial_inputs,
            )
        except BaseException as error:
            _preserve_initialization_failure(
                run_dir,
                run_id,
                workflow,
                error,
                published_snapshots,
                source,
                inputs,
            )
            raise
        context = cls(root, run_dir, run_id, workflow, 1, recorder)
        if source_factory is not None or input_factories is not None:
            try:
                context._populate_evidence(source, inputs, source_factory, input_factories)
            except BaseException as error:
                context.recorder.fail(category="ARTIFACT", message=str(error))
                raise
        return context

    @classmethod
    def _resume_existing(
        cls, root: Path, run_dir: Path, workflow: str,
        original_config: dict[str, Any], normalized_config: dict[str, Any],
        source: dict[str, Any] | None, inputs: list[dict[str, str]] | None,
        config_metadata: dict[str, Any] | None,
        source_factory: Callable[[], dict[str, Any]] | None,
        input_factories: Sequence[Callable[[], dict[str, str]]] | None,
    ) -> "RunContext":
        with _run_lock(run_dir):
            recorder = _load_existing(run_dir, workflow)
            current = recorder.manifest.status
            # Submitted runs never evaluate user-provided evidence factories:
            # scheduler recovery is deliberately outside this pilot.
            if current is RunState.SUBMITTED:
                raise ValueError(
                    "cannot resume submitted run: scheduler recovery/polling is out of scope"
                )
            if current in _TERMINAL_STATES:
                raise ValueError(
                    f"cannot resume terminal run {recorder.manifest.run_id!r} "
                    f"in state {current.value}"
                )
            supplied_source, supplied_inputs = _materialize_evidence(
                source, inputs, source_factory, input_factories
            )
            _compare_resume_evidence(
                recorder, run_dir, original_config, normalized_config,
                _metadata_or_default(config_metadata), supplied_source, supplied_inputs,
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
        source: dict[str, Any] | None,
        inputs: list[dict[str, str]] | None,
        config_metadata: dict[str, Any] | None,
        source_factory: Callable[[], dict[str, Any]] | None,
        input_factories: Sequence[Callable[[], dict[str, str]]] | None,
    ) -> "RunContext":
        with _run_lock(run_dir):
            recorder = _load_existing(run_dir, workflow)
            previous_manifest = (run_dir / "run.json").read_bytes()
            previous_attempt = _next_attempt_id(recorder)

            version = previous_attempt
            original_path = run_dir / f"config.original.attempt-{version:04d}.json"
            resolved_path = run_dir / f"config.resolved.attempt-{version:04d}.json"
            original_stage = run_dir / f".config.original.attempt-{version:04d}.json.stage"
            resolved_stage = run_dir / f".config.resolved.attempt-{version:04d}.json.stage"
            metadata_path = run_dir / f"config.metadata.attempt-{version:04d}.json"
            metadata_stage = run_dir / f".config.metadata.attempt-{version:04d}.json.stage"
            staged_snapshots: list[Path] = []
            published_snapshots: list[Path] = []
            try:
                _atomic_json_write(original_stage, original_config, overwrite=False)
                staged_snapshots.append(original_stage)
                _atomic_json_write(resolved_stage, normalized_config, overwrite=False)
                staged_snapshots.append(resolved_stage)
                _atomic_json_write(
                    metadata_stage, _metadata_or_default(config_metadata), overwrite=False
                )
                staged_snapshots.append(metadata_stage)
                _archive_manifest(run_dir, previous_manifest, previous_attempt - 1)
                _publish_snapshot(original_stage, original_path)
                published_snapshots.append(original_path)
                staged_snapshots.remove(original_stage)
                _publish_snapshot(resolved_stage, resolved_path)
                published_snapshots.append(resolved_path)
                staged_snapshots.remove(resolved_stage)
                config_references = {
                    "original": original_path.name,
                    "resolved": resolved_path.name,
                }
                _publish_snapshot(metadata_stage, metadata_path)
                published_snapshots.append(metadata_path)
                staged_snapshots.remove(metadata_stage)
                config_references["metadata"] = metadata_path.name
                fresh = StatusRecorder.create(
                    run_dir / "run.json",
                    recorder.manifest.run_id,
                    workflow,
                    config=config_references,
                    source=source if source_factory is None else {},
                    inputs=_merge_inputs([], inputs or []),
                    attempt_id=previous_attempt,
                    events=[
                        RunEvent(
                            state=RunState.CREATED,
                            kind="force",
                            attempt_id=previous_attempt,
                            reason=options.reason,
                        )
                    ],
                )
            except BaseException:
                _clean_unpublished_config_snapshots(
                    run_dir,
                    previous_manifest,
                    [*staged_snapshots, *published_snapshots],
                )
                raise
            context = cls(root, run_dir, recorder.manifest.run_id, workflow, previous_attempt, fresh)
            if source_factory is not None or input_factories is not None:
                try:
                    context._populate_evidence(source, inputs, source_factory, input_factories)
                except BaseException as error:
                    context.recorder.fail(category="ARTIFACT", message=str(error))
                    raise
            return context

    def _populate_evidence(
        self,
        explicit_source: dict[str, Any] | None,
        explicit_inputs: list[dict[str, str]] | None,
        source_factory: Callable[[], dict[str, Any]] | None,
        input_factories: Sequence[Callable[[], dict[str, str]]] | None,
    ) -> None:
        if source_factory is not None:
            observed_source = source_factory()
            if explicit_source is not None and explicit_source != observed_source:
                raise ValueError("source identity conflict between explicit and factory evidence")
            self.recorder.update_metadata(source=observed_source)
        elif explicit_source is not None:
            self.recorder.update_metadata(source=explicit_source)

        observed_inputs = _merge_inputs([], self.recorder.manifest.inputs)
        for factory in input_factories or ():
            observed_inputs = _merge_inputs(observed_inputs, [factory()])
            # Persist each successful input independently, so a later
            # failure leaves all earlier evidence in the failed manifest.
            self.recorder.update_metadata(inputs=observed_inputs)

    def register_verified_artifacts(
        self,
        kind: str,
        paths: Sequence[Path],
        *,
        baseline: AttemptOutputBaseline | None = None,
    ) -> None:
        """Register current-attempt files with streaming SHA-256 identity."""

        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("artifact kind must be a non-empty string")

        verified_paths = list(paths)
        if baseline is not None:
            verified_paths = baseline.fresh(verified_paths)
            if not verified_paths:
                raise ValueError("no artifact was created or rewritten by the current attempt")

        records: list[ArtifactRecord] = []
        for supplied in verified_paths:
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


def _metadata_or_default(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return deepcopy(_DEFAULT_CONFIG_METADATA if metadata is None else metadata)


def _materialize_evidence(
    source: dict[str, Any] | None,
    inputs: list[dict[str, str]] | None,
    source_factory: Callable[[], dict[str, Any]] | None,
    input_factories: Sequence[Callable[[], dict[str, str]]] | None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    observed_source = source_factory() if source_factory is not None else (source or {})
    if source_factory is not None and source is not None and source != observed_source:
        raise ValueError("source identity conflict between explicit and factory evidence")
    observed_inputs = _merge_inputs([], inputs or [])
    for factory in input_factories or ():
        observed_inputs = _merge_inputs(observed_inputs, [factory()])
    return observed_source, observed_inputs


def _merge_inputs(
    existing: Sequence[dict[str, str]], additions: Sequence[dict[str, str]]
) -> list[dict[str, str]]:
    """Merge input evidence by logical identity without hiding conflicts."""
    merged = [dict(item) for item in existing]
    positions = {(item.get("kind"), item.get("ref")): index for index, item in enumerate(merged)}
    for item in additions:
        key = (item.get("kind"), item.get("ref"))
        if key in positions:
            if merged[positions[key]] != item:
                raise ValueError(f"conflicting input identity for {key[0]}:{key[1]}")
            continue
        positions[key] = len(merged)
        merged.append(dict(item))
    return merged


def _compare_source_identity(existing: dict[str, Any], current: dict[str, Any]) -> None:
    """Compare provenance, preferring the scoped runtime fingerprint."""
    if "runtime_fingerprint" in existing:
        if "runtime_fingerprint" not in current:
            raise ValueError("source identity mismatch: scoped runtime fingerprint missing")
        for key in (
            "dpeva_version",
            "package_version",
            "runtime_fingerprint_version",
            "runtime_fingerprint_scope",
            "runtime_fingerprint",
        ):
            if existing.get(key) != current.get(key):
                raise ValueError("source identity mismatch")
        return
    if "runtime_fingerprint" in current:
        raise ValueError("source identity mismatch: legacy unscoped provenance cannot be compared")
    if "dirty_fingerprint" not in existing and existing.get("git_commit"):
        if (
            existing.get("git_commit") != current.get("git_commit")
            or existing.get("dirty") is not False
            or current.get("dirty") is not False
        ):
            raise ValueError(
                "source identity mismatch: legacy dirty provenance cannot be proven equal"
            )
        for key, value in existing.items():
            if current.get(key) != value:
                raise ValueError("source identity mismatch")
        return
    if existing != current:
        raise ValueError("source identity mismatch")


def _read_config_snapshot(recorder: StatusRecorder, run_dir: Path, key: str) -> Any:
    reference = recorder.manifest.config.get(key)
    if not isinstance(reference, str) or Path(reference).is_absolute():
        raise ValueError(f"configuration evidence missing or unsafe: {key}")
    snapshot = (run_dir / reference).resolve(strict=False)
    try:
        snapshot.relative_to(run_dir.resolve())
    except ValueError:
        raise ValueError(f"configuration evidence escapes run directory: {key}") from None
    try:
        return json.loads(snapshot.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"configuration evidence unreadable: {key}: {error}") from error


def _compare_resume_evidence(
    recorder: StatusRecorder,
    run_dir: Path,
    original_config: dict[str, Any],
    normalized_config: dict[str, Any],
    config_metadata: dict[str, Any],
    source: dict[str, Any],
    inputs: list[dict[str, str]],
) -> None:
    # Legacy manifests predating immutable config snapshots remain resumable;
    # there is no referenced evidence to compare.  New manifests always carry
    # all three references and therefore take the strict path below.
    if recorder.manifest.config:
        if _read_config_snapshot(recorder, run_dir, "original") != original_config:
            raise ValueError("configuration evidence mismatch: original configuration")
        if _read_config_snapshot(recorder, run_dir, "resolved") != normalized_config:
            raise ValueError("configuration evidence mismatch: resolved configuration")
        if "metadata" in recorder.manifest.config:
            if _read_config_snapshot(recorder, run_dir, "metadata") != config_metadata:
                raise ValueError("configuration metadata mismatch")
        elif config_metadata != _metadata_or_default(None):
            raise ValueError("configuration metadata mismatch: legacy implicit default")
    _compare_source_identity(recorder.manifest.source, source)
    if recorder.manifest.inputs != inputs:
        raise ValueError("input identity mismatch")


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
    archive = attempts_dir / f"attempt-{max(1, attempt_id):04d}.json"
    if archive.exists():
        if archive.read_bytes() == payload:
            return
        raise FileExistsError(f"attempt archive identity collision: {archive}")
    try:
        with archive.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if archive.read_bytes() != payload:
            raise FileExistsError(f"attempt archive identity collision: {archive}") from None
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
    run_dir: Path,
    run_id: str,
    workflow: str,
    error: BaseException,
    published_snapshots: Sequence[Path],
    source: dict[str, Any] | None = None,
    inputs: list[dict[str, str]] | None = None,
) -> None:
    """Leave a failed manifest behind when a newly allocated run cannot initialize."""

    manifest_path = run_dir / "run.json"
    try:
        if manifest_path.exists():
            recorder = StatusRecorder.load(manifest_path)
        else:
            config = {
                "original": path.name
                for path in published_snapshots
                if path.name == "config.original.json"
            }
            config.update(
                {
                    "resolved": path.name
                    for path in published_snapshots
                    if path.name == "config.resolved.json"
                }
            )
            config.update(
                {
                    "metadata": path.name
                    for path in published_snapshots
                    if path.name == "config.metadata.json"
                }
            )
            recorder = StatusRecorder.create(
                manifest_path,
                run_id,
                workflow,
                config=config,
                source=source,
                inputs=inputs,
            )
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
        if path.name.encode() not in previous_manifest:
            path.unlink(missing_ok=True)


def _publish_snapshot(stage: Path, target: Path) -> None:
    """Publish one staged snapshot without replacing an existing evidence file."""

    if target.is_symlink():
        raise ValueError(f"config snapshot target must not be a symlink: {target}")
    if target.exists():
        if target.read_bytes() != stage.read_bytes():
            raise FileExistsError(f"config snapshot identity collision: {target}")
        stage.unlink()
        return
    os.replace(stage, target)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHECKSUM_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(
    source_file: str | Path | None = None,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Return publishable package/source identity without machine paths."""

    import dpeva

    identity: dict[str, Any] = {
        "dpeva_version": dpeva.__version__,
        "package_version": dpeva.__version__,
    }
    source = Path(source_file or __file__).expanduser().resolve(strict=False)
    repository = _discover_tracked_repository(source, run=run)
    if repository is None:
        return identity
    try:
        commit_result = run(
            ["git", "rev-parse", "HEAD"], cwd=repository, check=False,
            text=True, capture_output=True,
        )
        status_result = run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all",
             "--", *_RUNTIME_GIT_PATHSPEC],
            cwd=repository, check=False, text=False, capture_output=True,
        )
    except (OSError, subprocess.SubprocessError):
        return identity
    if commit_result.returncode == 0 and commit_result.stdout.strip():
        identity["git_commit"] = commit_result.stdout.strip()
    identity["runtime_fingerprint_version"] = _RUNTIME_FINGERPRINT_VERSION
    identity["runtime_fingerprint_scope"] = list(_RUNTIME_FINGERPRINT_SCOPE)
    identity["runtime_fingerprint"] = _runtime_fingerprint(
        repository,
        source,
        status_result.stdout if status_result.returncode == 0 else b"",
        run=run,
    )
    if status_result.returncode == 0:
        entries = _publishable_git_status(status_result.stdout)
        identity["dirty"] = bool(entries)
        # Reuse the scoped content digest: informational dirty metadata must
        # never trigger a second, repository-wide pass over dataset contents.
        identity["dirty_fingerprint"] = hashlib.sha256(
            b"\0".join([*entries, identity["runtime_fingerprint"].encode("ascii")])
        ).hexdigest()
    return identity


def _is_runtime_path(relative: str) -> bool:
    if ".dpeva" in Path(relative).parts:
        return False
    return relative == "pyproject.toml" or relative == "src/dpeva" or relative.startswith(
        "src/dpeva/"
    )


def _runtime_fingerprint(
    repository: Path,
    source: Path,
    status_output: bytes | str,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]],
) -> str:
    """Hash tracked runtime files and untracked Python package additions.

    The digest contains relative names and content identities only.  This
    keeps it stable when the same source tree is checked out at another path,
    while still recording deleted tracked files and symlink targets.
    """
    tracked_result = run(
        ["git", "ls-files", "--cached", "-z", "--", *_RUNTIME_GIT_PATHSPEC],
        cwd=repository,
        check=False,
        text=False,
        capture_output=True,
    )
    runtime_paths: set[str] = set()
    if tracked_result.returncode == 0:
        raw = tracked_result.stdout
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        for field in raw.split(b"\0"):
            if not field:
                continue
            relative = os.fsdecode(field)
            if _is_runtime_path(relative):
                runtime_paths.add(relative)

    source_relative = _relative_runtime_path(source, repository)
    if source_relative is not None:
        runtime_paths.add(source_relative)

    for paths in _status_paths(status_output):
        for path in paths:
            relative = os.fsdecode(path)
            if _is_runtime_path(relative) and (
                relative.endswith(".py") or relative == "pyproject.toml"
            ):
                runtime_paths.add(relative)

    records: list[bytes] = []
    for relative in sorted(runtime_paths):
        candidate = repository / relative
        if candidate.is_symlink():
            try:
                content = b"symlink:" + os.fsencode(os.readlink(candidate))
            except OSError:
                content = b"unreadable"
        elif candidate.is_file():
            try:
                content = b"sha256:" + _sha256(candidate).encode("ascii")
            except OSError:
                content = b"unreadable"
        elif not candidate.exists():
            content = b"deleted"
        else:
            content = b"non-file"
        records.append(os.fsencode(relative) + b"\0" + content)
    return hashlib.sha256(b"\0".join(records)).hexdigest()


def _relative_runtime_path(path: Path, repository: Path) -> str | None:
    try:
        relative = path.relative_to(repository).as_posix()
    except ValueError:
        return None
    return relative if _is_runtime_path(relative) else None


def _status_paths(output: bytes | str) -> list[list[bytes]]:
    """Decode porcelain-v1 NUL records into one or two path fields."""
    raw = output if isinstance(output, bytes) else output.encode("utf-8")
    fields = raw.split(b"\0")
    paths: list[list[bytes]] = []
    index = 0
    while index < len(fields) - 1:
        record = fields[index]
        index += 1
        if len(record) < 4:
            continue
        status, path = record[:2], record[3:]
        current = [path]
        if status[:1] in {b"R", b"C"} or status[1:2] in {b"R", b"C"}:
            if index >= len(fields) - 1:
                continue
            current.append(fields[index])
            index += 1
        paths.append(current)
    return paths


def _publishable_git_status(output: bytes | str) -> list[bytes]:
    """Collect scoped porcelain metadata without opening any files."""
    raw = output if isinstance(output, bytes) else output.encode("utf-8")
    fields = raw.split(b"\0")
    entries: list[bytes] = []
    index = 0
    while index < len(fields) - 1:
        record = fields[index]
        index += 1
        if len(record) < 4:
            continue
        status, path = record[:2], record[3:]
        paths = [path]
        if status[:1] in {b"R", b"C"} or status[1:2] in {b"R", b"C"}:
            if index >= len(fields) - 1:
                continue
            paths.append(fields[index])
            index += 1
        scoped = [
            item for item in paths
            if _is_runtime_path(os.fsdecode(item))
            and (status != b"??" or item.endswith(b".py") or item == b"pyproject.toml")
        ]
        if scoped:
            entries.append(b"\0".join([status, *scoped]))
    return sorted(entries)


def _discover_tracked_repository(
    source: Path,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]],
) -> Path | None:
    """Find the nearest git root which actually tracks the package source."""
    for candidate in (source.parent, *source.parents):
        if not (candidate / ".git").exists():
            continue
        try:
            root_result = run(
                ["git", "rev-parse", "--show-toplevel"], cwd=candidate,
                check=False, text=True, capture_output=True,
            )
            if root_result.returncode != 0 or not root_result.stdout.strip():
                continue
            repository = Path(root_result.stdout.strip()).resolve()
            relative = source.relative_to(repository)
            tracked = run(
                ["git", "ls-files", "--error-unmatch", "--", relative.as_posix()],
                cwd=repository, check=False, text=True, capture_output=True,
            )
            if tracked.returncode == 0 and tracked.stdout.strip():
                return repository
        except (OSError, subprocess.SubprocessError, ValueError):
            continue
    return None


def input_identity(
    path: str | Path,
    kind: str,
    work_dir: str | Path,
    *,
    require_exists: bool = False,
) -> dict[str, str]:
    """Describe an input with a relative reference and truthful identity scope."""

    candidate = Path(path).expanduser().resolve(strict=False)
    root = Path(work_dir).expanduser().resolve()
    try:
        reference = candidate.relative_to(root).as_posix()
    except ValueError:
        reference = None
    result: dict[str, str] = {"kind": kind, "ref": ""}
    if candidate.is_file():
        try:
            digest = _sha256(candidate)
        except OSError as error:
            label = reference or f"external/{candidate.name}"
            raise OSError(f"{kind} input unreadable: {label}") from error
        result["ref"] = reference or f"external/{candidate.name}-{digest}"
        result.update(identity=f"sha256:{digest}", identity_scope="full-content")
    elif candidate.is_dir():
        digest, count = _structural_identity(candidate)
        result["ref"] = reference or f"external/{candidate.name}-{digest}"
        result.update(
            identity=f"structural-sha256:{digest}",
            identity_scope="bounded-structural",
            identity_entries=str(count),
            identity_bound=f"first-{_STRUCTURAL_ENTRY_LIMIT}-files/{_STRUCTURAL_NODE_LIMIT}-nodes",
        )
    else:
        result["ref"] = reference or f"external/{candidate.name}"
        if require_exists:
            raise FileNotFoundError(f"{kind} input does not exist: {result['ref']}")
        result.update(identity="unavailable", identity_scope="unverified")
    return result


def _structural_identity(directory: Path) -> tuple[str, int]:
    entries: list[str] = []
    pending = [directory]
    visited = 0
    while pending and len(entries) < _STRUCTURAL_ENTRY_LIMIT and visited < _STRUCTURAL_NODE_LIMIT:
        current = pending.pop(0)
        try:
            children = sorted(os.scandir(current), key=lambda item: item.name)
        except OSError:
            continue
        for child in children:
            visited += 1
            if visited > _STRUCTURAL_NODE_LIMIT:
                break
            try:
                if child.is_dir(follow_symlinks=False):
                    pending.append(Path(child.path))
                elif child.is_file(follow_symlinks=False):
                    relative = Path(child.path).relative_to(directory).as_posix()
                    entries.append(f"{relative}\0{child.stat(follow_symlinks=False).st_size}\n")
            except OSError:
                continue
            if len(entries) >= _STRUCTURAL_ENTRY_LIMIT:
                break
    payload = "".join(entries).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(entries)
